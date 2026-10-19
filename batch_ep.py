"""Batch entropy-production estimation with parallel or warm-start lambda paths.

Independent candidates or whole warm-started paths run in a loky pool.
Data are loaded once and stored for memory-mapped access by workers.
"""
from __future__ import annotations

import csv
from contextlib import ExitStack
import importlib
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from phase_estimators import KTG, ARTG

STATES = ('baseline', 'mild', 'moderate', 'recovery')
CSV_FIELDS = ('group', 'patient_id', 'band', 'state', 'model', 'n_samples', 'n_variables', 'lags',
              'n_transitions', 'value_type', 'conjugate', 'group_lasso', 'lambda', 'aic_refit',
              'mean_log_likelihood', 'ep_nats_per_step', 'ep_normal', 'ep_conjugate', 'active_groups', 'converged', 'selection_status')


def resolve_loader(spec):
    if ':' not in spec:
        raise ValueError('--loader must be module:function')
    module, name = spec.split(':', 1)
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError(f'{spec} is not callable')
    return fn


def _lambda_grid(X, name, *, lags, n_lambdas, lambda_min_ratio, include_zero=False):
    """Same lambda grid, including zero, as the estimator's AIC search."""
    cls = KTG if name == 'ktg' else ARTG
    estimator = cls(lags=lags)
    a, b = estimator._features(X)
    obs_cos, obs_sin = np.cos(X[lags:]), np.sin(X[lags:])
    zero = np.zeros((lags, X.shape[1], X.shape[1], cls._n_features))
    _, grad = estimator._loss_grad(zero, a, b, obs_cos, obs_sin)
    lam_max = float(np.linalg.norm(grad, axis=-1).max())
    if lam_max <= 1e-14:
        return np.array([0.])
    return np.r_[np.geomspace(lam_max, lam_max * lambda_min_ratio, n_lambdas), ([0.] if include_zero else [])]


@lru_cache(maxsize=8)
def _cached_input(path, model_name, lags):
    """Cache trigonometric features once per (worker, dataset, model)."""
    X = np.load(path, mmap_mode='r', allow_pickle=False)
    cls = KTG if model_name == 'ktg' else ARTG
    est = cls(lags=lags)
    a, b = est._features(X)
    return X, (a, b, np.cos(X[lags:]), np.sin(X[lags:]))


def _candidate_job(cell_idx, lam_idx, lam, data_path, model_name, settings):
    """Stage 1: penalized support selection; NO expensive refit here."""
    X, prepared = _cached_input(data_path, model_name, settings['lags'])
    cls = KTG if model_name == 'ktg' else ARTG
    est = cls(lags=settings['lags'], group_lasso=settings['group_lasso'],
              lambda_=float(lam), max_iter=settings['max_iter'], tol=settings['tol'],
              initial_step=settings['initial_step'], lambda_jobs=1)
    est._prepared = prepared
    est.fit(X)
    support = np.linalg.norm(est.coef_, axis=-1) > 1e-7
    return (cell_idx, lam_idx, float(lam), est.coef_, support,
            bool(est.success_), int(est.n_iter_), float(est.fit_seconds_),
            int(est.backtracking_trials_), float(est.kkt_residual_))


def _path_job(cell_idx, grid, data_path, model_name, settings, progress_every=5):
    """Warm-start lambda path for one cell; return each candidate to shared refit stage."""
    X, prepared = _cached_input(data_path, model_name, settings['lags'])
    cls = KTG if model_name == 'ktg' else ARTG
    est = cls(lags=settings['lags'], group_lasso=settings['group_lasso'],
              max_iter=settings['max_iter'], tol=settings['tol'],
              initial_step=settings['initial_step'], lambda_jobs=1)
    est._prepared = prepared
    previous = None
    results = []
    path_started = time.monotonic()
    ordered = sorted(enumerate(grid), key=lambda item: -item[1])
    for step_no, (li, lam) in enumerate(ordered, start=1):
        est._fit_fixed(X, float(lam), initial_coef=previous)
        support = np.linalg.norm(est.coef_, axis=-1) > 1e-7
        results.append((cell_idx, li, float(lam), est.coef_.copy(),
                        support, bool(est.success_), int(est.n_iter_),
                        float(est.fit_seconds_), int(est.backtracking_trials_), float(est.kkt_residual_)))
        # Preserve the last computed solution even when it reached max_iter:
        # this is an initialization only; failed candidates remain ineligible.
        previous = est.coef_.copy()
        if progress_every and (step_no % progress_every == 0 or step_no == len(ordered)):
            print(f"[path cell={cell_idx+1} {step_no}/{len(ordered)}] "
                  f"elapsed={time.monotonic()-path_started:.1f}s "
                  f"lambda={lam:.5g} edges={int(support.sum())} "
                  f"converged={bool(est.success_)} iterations={int(est.n_iter_)} "
                  f"fit_seconds={est.fit_seconds_:.2f} backtracks={est.backtracking_trials_} "
                  f"kkt={est.kkt_residual_:.3g}", flush=True)
    return results


def _refit_job(cell_idx, support, init_coef, data_path, model_name, settings):
    """Stage 2: unpenalized restricted MLE, once per unique support."""
    X, prepared = _cached_input(data_path, model_name, settings['lags'])
    cls = KTG if model_name == 'ktg' else ARTG
    est = cls(lags=settings['lags'], max_iter=settings['max_iter'], tol=settings['tol'])
    est.coef_ = init_coef
    est.n_features_in_ = X.shape[1]
    est._prepared = prepared
    est.refit_active(X)
    theta = est.refit_coef_
    a, b, oc, os = prepared
    loss = est._loss_only(theta, a, b, oc, os)
    n_parameters = int(support.sum()) * cls._n_features
    aic = 2 * len(oc) * loss + 2 * n_parameters
    return cell_idx, support.tobytes(), float(aic), theta, bool(est.refit_success_), int(est.refit_n_iter_)


def run_batch_ep(*, loader, patient_ids=None, patient_groups=None, bands,
                 models=('ktg', 'artg'), states=STATES, lags=1, group_lasso=False,
                 lambda_=None, n_lambdas=60, lambda_min_ratio=0.01,
                 max_iter=10000, tol=1e-7, initial_step=1.0,
                 value_type='absolute', conjugate=False,
                 output_csv='results/ep_by_band_state.csv', progress=True,
                 lambda_progress_every=5, n_jobs=128, include_zero_lambda=False,
                 aic_path_csv=None, lambda_solver="parallel", **deprecated):
    """Compute one CSV row per patient/state/model.

    parallel: independent lambda jobs with zero initialization.
    path: sequential descending lambdas with warm starts within each cell;
    different cells remain parallel. Both modes share the same refit stage.
    """
    if lambda_solver not in ("parallel", "path"):
        raise ValueError("lambda_solver must be parallel or path")
    if deprecated:
        raise TypeError(f'Unsupported parameters: {list(deprecated)}; use n_jobs')
    if patient_groups is None:
        if not patient_ids:
            raise ValueError('Provide patient_groups or patient_ids')
        patient_groups = {'unspecified': list(patient_ids)}
    if not patient_groups or not all(patient_groups.values()):
        raise ValueError('Patient groups must be nonempty')
    ids = [int(pid) for values in patient_groups.values() for pid in values]
    if len(set(ids)) != len(ids):
        raise ValueError('Each patient must belong to exactly one group')
    if not bands or not models or any(m not in ('ktg', 'artg') for m in models):
        raise ValueError('Provide bands and valid model names')
    if any(s not in STATES for s in states):
        raise ValueError(f'Invalid state; valid states are {STATES}')
    if value_type not in ('absolute', 'relative'):
        raise ValueError('value_type must be absolute or relative')
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError('n_jobs must be positive')
    if lambda_ is not None and lambda_ < 0:
        raise ValueError('lambda_ must be nonnegative')

    cells = [(group, int(pid), band, state, model)
             for group, pids in patient_groups.items() for pid in pids
             for band in bands for state in states for model in models]
    started = time.monotonic()

    def report(message):
        if progress:
            print(message, flush=True)

    settings = dict(lags=lags, group_lasso=group_lasso,
                    max_iter=max_iter, tol=tol, initial_step=initial_step)
    rows = [None] * len(cells)
    diagnostic_fields = ['group','patient_id','band','state','model','lambda_', 'aic',
                         'active_groups','ista_converged','refit_converged','iterations',
                         'fit_seconds','backtracking_trials','kkt_residual']
    with ExitStack() as stack:
        diagnostic_writer = None
        diagnostic_file = None
        if aic_path_csv is not None:
            diagnostic_path = Path(aic_path_csv)
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_file = stack.enter_context(diagnostic_path.open('w', newline='', encoding='utf-8'))
            diagnostic_writer = csv.DictWriter(diagnostic_file, fieldnames=diagnostic_fields)
            diagnostic_writer.writeheader()
            diagnostic_file.flush()
            report(f'[batch-ep] AIC candidate results: {diagnostic_path.resolve()}')
        tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix='phase_ep_'))
        # Load each patient/state once; both models point to the same file.
        paths = {}
        meta = {}
        jobs = []
        counts = []
        path_jobs = []
        data_cache = {}
        report('[batch-ep] Loading patient/state data and building lambda grids ...')
        for idx, (group, pid, band, state, name) in enumerate(cells):
            key = (pid, band, state)
            if key not in paths:
                X = np.asarray(loader(pid, band, state, value_type), dtype=np.float64)
                if X.ndim != 2 or X.shape[0] <= lags or X.shape[1] < 1 or not np.isfinite(X).all():
                    raise ValueError(f'Invalid phase array for patient={pid}, band={band}, state={state}')
                path = str(Path(tmp) / f'data_{len(paths)}.npy')
                np.save(path, X, allow_pickle=False)
                paths[key] = path
                meta[key] = X.shape
                data_cache[key] = X
            else:
                X = data_cache[key]
            if not group_lasso:
                grid = np.array([0.])
            elif lambda_ is not None:
                grid = np.array([lambda_], dtype=float)
            else:
                grid = _lambda_grid(X, name, lags=lags, n_lambdas=n_lambdas,
                                    lambda_min_ratio=lambda_min_ratio, include_zero=include_zero_lambda)
            counts.append(len(grid))
            path_jobs.append((idx, tuple(float(v) for v in grid), paths[key], name))
            for li, lam in enumerate(grid):
                jobs.append((idx, li, float(lam), paths[key], name))
        del data_cache  # Workers use memmaps, not copies of all arrays.
        total_jobs = len(jobs)
        report(f'[batch-ep] {len(cells)} fits, {total_jobs} candidate tasks, '
               f'{min(n_jobs, len(jobs) if lambda_solver == "parallel" else len(cells))} workers '
               f'(support stage; solver={lambda_solver})')
        candidate_results = [[] for _ in cells]
        unique_supports = [{} for _ in cells]
        done = 0
        def collect(result):
            nonlocal done
            idx, li, lam, coef, support, success, nit, seconds, trials, kkt = result
            done += 1
            key = support.tobytes()
            candidate_results[idx].append((li, lam, key, success, nit, int(support.sum()), seconds, trials, kkt))
            if success and key not in unique_supports[idx]:
                unique_supports[idx][key] = (support, coef)
            if progress and (done % max(1, lambda_progress_every) == 0 or done == total_jobs):
                report(f'[supports {done}/{total_jobs}] elapsed={time.monotonic()-started:.1f}s '
                       f'cell={idx+1}/{len(cells)} lambda={lam:.5g} '
                       f'edges={int(support.sum())} converged={success} '
                       f'fit_seconds={seconds:.2f} backtracks={trials} kkt={kkt:.3g}')

        if lambda_solver == 'parallel':
            parallel = Parallel(n_jobs=min(n_jobs, len(jobs)), backend='loky', batch_size=1,
                                return_as='generator_unordered', pre_dispatch='2*n_jobs')
            stream = parallel(delayed(_candidate_job)(idx, li, lam, path, name, settings)
                              for idx, li, lam, path, name in jobs)
            try:
                for result in stream:
                    collect(result)
            finally:
                if hasattr(stream, 'close'):
                    stream.close()
        else:
            parallel = Parallel(n_jobs=min(n_jobs, len(path_jobs)), backend='loky', batch_size=1,
                                return_as='generator_unordered', pre_dispatch='2*n_jobs')
            stream = parallel(delayed(_path_job)(idx, grid, path, name, settings,
                                                   lambda_progress_every if progress else 0)
                              for idx, grid, path, name in path_jobs)
            try:
                for path_results in stream:
                    for result in path_results:
                        collect(result)
            finally:
                if hasattr(stream, 'close'):
                    stream.close()

        refit_jobs = [(idx, support, coef, paths[(cells[idx][1], cells[idx][2], cells[idx][3])], cells[idx][4])
                      for idx, structures in enumerate(unique_supports)
                      for support, coef in structures.values()]
        report(f'[batch-ep] {sum(map(len, unique_supports))} unique converged supports '
               f'from {total_jobs} lambda candidates; refitting each once')
        if not refit_jobs:
            raise RuntimeError('No converged Group Lasso supports; change solver settings.')
        refit_by_cell = [{} for _ in cells]
        parallel = Parallel(n_jobs=min(n_jobs, len(refit_jobs)), backend='loky', batch_size=1,
                            return_as='generator_unordered', pre_dispatch='2*n_jobs')
        stream = parallel(delayed(_refit_job)(idx, support, coef, path, name, settings)
                          for idx, support, coef, path, name in refit_jobs)
        try:
            for idx, key, aic, theta, ok, nit in stream:
                refit_by_cell[idx][key] = (aic, theta, ok, nit)
        finally:
            if hasattr(stream, 'close'):
                stream.close()

        for idx, candidates in enumerate(candidate_results):
            group, pid, band, state, name = cells[idx]
            best = None
            for li, lam, key, success, nit, edges, seconds, trials, kkt in sorted(candidates):
                data = refit_by_cell[idx].get(key)
                aic, refit_coef, refit_success = (data[0], data[1], data[2]) if data else (float('nan'), None, False)
                if diagnostic_writer is not None:
                    diagnostic_writer.writerow(dict(group=group, patient_id=pid, band=band,
                        state=state, model=name, lambda_=lam, aic=aic,
                        active_groups=edges, ista_converged=success,
                        refit_converged=refit_success, iterations=nit,
                        fit_seconds=seconds, backtracking_trials=trials, kkt_residual=kkt))
                    diagnostic_file.flush()
                if success and refit_success and np.isfinite(aic):
                    if best is None or (aic, li) < (best[0], best[1]):
                        best = (aic, li, lam, refit_coef, edges)
            if best is None:
                raise RuntimeError(f'No converged finite-AIC candidate for {cells[idx]}')
            selected_aic, _, selected_lam, refit_coef, edges = best
            X = np.load(paths[(pid, band, state)], mmap_mode='r', allow_pickle=False)
            cls = KTG if name == 'ktg' else ARTG
            fitted = cls(lags=lags)
            fitted.coef_ = refit_coef  # IMPORTANT: selected UNPENALIZED refit coefficients
            fitted.n_features_in_ = X.shape[1]
            ep_normal = float(fitted.entropy_production(X, conjugate=False))
            ep_conjugate = float(fitted.entropy_production(X, conjugate=True))
            row = dict(group=group, patient_id=pid, band=band, state=state, model=name,
                       n_samples=X.shape[0], n_variables=X.shape[1], lags=lags,
                       n_transitions=X.shape[0]-lags, value_type=value_type,
                       conjugate=bool(conjugate), group_lasso=bool(group_lasso),
                       **{'lambda':selected_lam}, aic_refit=selected_aic,
                       mean_log_likelihood=float(fitted.score(X)),
                       ep_nats_per_step=ep_conjugate if conjugate else ep_normal,
                       ep_normal=ep_normal, ep_conjugate=ep_conjugate,
                       active_groups=edges, converged=True, selection_status='converged')
            rows[idx] = row
            report(f'[fit {idx+1}/{len(cells)}] {name} patient={pid} {state} DONE '
                   f'lambda={selected_lam:.5g} edges={edges} '+
                   f'EP_normal={ep_normal:.6g} EP_conjugate={ep_conjugate:.6g}')
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report(f'[batch-ep] Saved {len(rows)} rows to {output_csv}; elapsed={time.monotonic()-started:.1f}s')
    return rows
