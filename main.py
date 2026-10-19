"""Command-line experiments for KTG and AR-TG.

Examples
--------
python main.py --experiment compare --data phases.npy
python main.py --experiment fit --model ktg --data phases.csv --lags 2
python main.py --experiment lambda-sweep --model artg --data phases.npy --lambdas 0.001 0.01 0.1
python main.py --experiment compare --n-samples 500 --n-variables 4 --seed 7

Without --data, synthetic data are generated from a specified KTG/AR-TG
transition model (with no intercept). Arrays have shape (N, d), in radians.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase_estimators import KTG, ARTG
from batch_ep import run_batch_ep, resolve_loader, STATES
from ep_plots import save_ep_plots
from aic_plots import save_aic_plots


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='KTG / AR-TG experiments')
    parser.add_argument('--experiment', choices=('fit', 'compare', 'lambda-sweep', 'batch-ep'),
                        default='compare', help='Experiment to run')
    parser.add_argument('--loader', type=str, default='data:get_phase_data',
                        help='Required for batch-ep: module:function taking (patient_id, band, state)')
    parser.add_argument('--plots-dir', type=Path, default=None,
                        help='PNG output directory for batch-ep (default: CSV parent / plots); use --no-plots to disable')
    parser.add_argument('--n-jobs', type=int, default=128, help='Single worker pool size for batch-ep')
    parser.add_argument('--quiet', action='store_true', help='Disable batch-ep progress messages')
    parser.add_argument('--lambda-solver', choices=('parallel', 'path'), default='parallel',
                        help='Independent parallel lambdas or warm-started sequential lambda path per cell')
    parser.add_argument('--lambda-progress-every', type=int, default=5, help='Report every Kth AIC lambda candidate')
    parser.add_argument('--no-plots', action='store_true', help='Disable batch-ep PNG output')
    parser.add_argument('--drowsy-ids', nargs='+', type=int, default=None, help='Drowsy group patient IDs')
    parser.add_argument('--responsive-ids', nargs='+', type=int, default=None, help='Responsive group patient IDs')
    parser.add_argument('--patient-ids', nargs='+', default=None,
                        help='Patient IDs for batch-ep')
    parser.add_argument('--value-type', choices=('absolute', 'relative'), default='absolute',
                        help='Phase representation passed to data loader')
    parser.add_argument('--conjugate', action='store_true',
                        help='Negate phase values of time-reversed data when computing EP')
    parser.add_argument('--band', type=str, default=None,
                        help='Single frequency band for batch-ep (one independent run per band)')
    parser.add_argument('--states', nargs='+', choices=STATES, default=list(STATES),
                        help='States for batch-ep (default: all four)')
    parser.add_argument('--models', nargs='+', choices=('ktg', 'artg'),
                        default=['ktg', 'artg'], help='Models for batch-ep')
    parser.add_argument('--output-csv', type=Path, default=None,
                        help='CSV output path for batch-ep')
    parser.add_argument('--model', choices=('ktg', 'artg'), default='ktg',
                        help='Model for fit and lambda-sweep')
    parser.add_argument('--data', type=Path, default=None,
                        help='Input .npy or .csv matrix (N,d) of radians; omit for synthetic data')
    parser.add_argument('--delimiter', default=',', help='CSV delimiter')
    parser.add_argument('--lags', type=int, default=1)
    parser.add_argument('--group-lasso', action='store_true', help='Use FISTA group lasso')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=None,
                        help='Group-lasso strength; if omitted with --group-lasso, select via AIC')
    parser.add_argument('--include-zero-lambda', action='store_true',
                        help='Include unregularized MLE as an AIC candidate (default: excluded)')
    parser.add_argument('--aic-path-csv', type=Path, default=None,
                        help='Save per-lambda fit diagnostics in CSV')
    parser.add_argument('--n-lambdas', type=int, default=60, help='Number of positive log-spaced lambdas in AIC search')
    parser.add_argument('--lambda-min-ratio', type=float, default=0.01,
                        help='Smallest positive grid lambda as fraction of lambda_max')
    parser.add_argument('--lambdas', type=float, nargs='+', default=None,
                        help='Values for lambda-sweep (requires positive values for ISTA)')
    parser.add_argument('--max-iter', type=int, default=10000)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--initial-step', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--n-variables', type=int, default=4)
    parser.add_argument('--generator', choices=('ktg', 'artg'), default='ktg',
                        help='Synthetic data generating model')
    parser.add_argument('--coupling', type=float, default=0.8)
    parser.add_argument('--output', type=Path, default=None, help='Optional JSON results path')
    parser.add_argument('--save-coef', type=Path, default=None,
                        help='Optional .npy file for coefficients; only for fit experiment')
    return parser.parse_args(argv)


def synthetic_data(n_samples, n_variables, lags, generator, coupling, seed):
    if n_samples <= lags or n_variables < 1 or lags < 1:
        raise ValueError('Require n_samples > lags >= 1 and n_variables >= 1')
    rng = np.random.default_rng(seed)
    d = n_variables
    nf = 2 if generator == 'ktg' else 4
    coef = np.zeros((lags, d, d, nf), dtype=float)
    # Directed ring coupling, plus persistence, scaled by lag.
    for lag in range(lags):
        for i in range(d):
            coef[lag, i, i, 0] = coupling / (lag + 1)
            if d > 1:
                coef[lag, i, (i-1) % d, 1] = 0.5 * coupling / (lag + 1)
            if nf == 4:
                coef[lag, i, i, 2] = 0.3 * coupling / (lag + 1)
    X = np.empty((n_samples, d))
    X[:lags] = rng.uniform(-np.pi, np.pi, size=(lags, d))
    for t in range(lags, n_samples):
        eta = np.zeros((d, 2))
        for lag in range(lags):
            xp = X[t - lag - 1]
            cosx, sinx = np.cos(xp), np.sin(xp)
            a, b = coef[lag, :, :, 0], coef[lag, :, :, 1]
            eta[:, 0] += (a @ cosx) - (b @ sinx)
            eta[:, 1] += (b @ cosx) + (a @ sinx)
            if nf == 4:
                c, s = coef[lag, :, :, 2], coef[lag, :, :, 3]
                eta[:, 0] += (c @ cosx) + (s @ sinx)
                eta[:, 1] += -(c @ sinx) + (s @ cosx)
        X[t] = rng.vonmises(np.arctan2(eta[:, 1], eta[:, 0]),
                           np.linalg.norm(eta, axis=1))
    return X


def load_data(args):
    if args.data is None:
        X = synthetic_data(args.n_samples, args.n_variables, args.lags,
                           args.generator, args.coupling, args.seed)
        source = f'synthetic:{args.generator}'
    elif args.data.suffix.lower() == '.npy':
        X = np.load(args.data, allow_pickle=False)
        source = str(args.data)
    elif args.data.suffix.lower() == '.csv':
        X = np.loadtxt(args.data, delimiter=args.delimiter, ndmin=2)
        source = str(args.data)
    else:
        raise ValueError('--data must be .npy or .csv')
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] <= args.lags or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError('Expected a finite (N,d) array, N > lags and d >= 1')
    return X, source


def fit_model(model_name, X, args, *, group_lasso=None, lambda_=None):
    cls = {'ktg': KTG, 'artg': ARTG}[model_name]
    model = cls(lags=args.lags,
                group_lasso=args.group_lasso if group_lasso is None else group_lasso,
                lambda_=args.lambda_ if lambda_ is None else lambda_,
                n_lambdas=args.n_lambdas, lambda_min_ratio=args.lambda_min_ratio,
                max_iter=args.max_iter, tol=args.tol,
                initial_step=args.initial_step)
    model.fit(X)
    return model


def summarize(model, model_name, X, conjugate=False):
    strengths = model.edge_strengths()
    return {
        'model': model_name,
        'group_lasso': bool(model.group_lasso),
        'lambda': float(model.best_lambda_),
        'aic': float(model.aic(X)),
        'effective_df': int(model.effective_df()),
        'aic_path': model.aic_path_,
        'coef_shape': list(model.coef_.shape),
        'n_iter': int(model.n_iter_),
        'converged': bool(model.success_),
        'mean_log_likelihood': float(model.score(X)),
        'entropy_production_nats_per_step': float(model.entropy_production(X, conjugate=conjugate)),
        'nonzero_groups': int(np.count_nonzero(strengths > 1e-9)),
        'total_groups': int(strengths.size),
    }


def main(argv=None):
    args = parse_args(argv)
    import os
    if args.n_jobs < 1:
        raise ValueError('--n-jobs must be positive')
    if args.experiment == 'batch-ep':
        if not args.band:
            raise ValueError('--experiment batch-ep requires --band (one band per execution)')
        if not args.drowsy_ids or not args.responsive_ids:
            raise ValueError('--experiment batch-ep requires --drowsy-ids and --responsive-ids')
        patient_groups = {'drowsy': args.drowsy_ids, 'responsive': args.responsive_ids}
        if args.lambda_ is not None and not args.group_lasso:
            raise ValueError('--lambda requires --group-lasso')
        output_csv = args.output_csv or Path('results') / args.band / 'ep.csv'
        aic_csv = args.aic_path_csv or output_csv.parent / 'aic_path.csv'
        rows = run_batch_ep(
            loader=resolve_loader(args.loader), patient_groups=patient_groups, bands=[args.band],
            models=args.models, states=args.states, lags=args.lags,
            group_lasso=args.group_lasso, lambda_=args.lambda_,
            n_lambdas=args.n_lambdas, lambda_min_ratio=args.lambda_min_ratio,
            max_iter=args.max_iter, tol=args.tol, initial_step=args.initial_step,
            value_type=args.value_type, conjugate=args.conjugate, output_csv=output_csv,
            progress=not args.quiet, lambda_progress_every=args.lambda_progress_every,
            n_jobs=args.n_jobs, include_zero_lambda=args.include_zero_lambda,
            aic_path_csv=aic_csv, lambda_solver=args.lambda_solver)
        plot_paths = [] if args.no_plots else save_ep_plots(
            output_csv, args.plots_dir or output_csv.parent / 'plots', models=args.models)
        if not args.no_plots and args.group_lasso:
            plot_paths += save_aic_plots(
                aic_csv, args.plots_dir or output_csv.parent / 'plots')
        payload = {'experiment': 'batch-ep', 'band': args.band, 'output_csv': str(output_csv),
                   'plots': [str(p) for p in plot_paths],
                   'value_type': args.value_type, 'conjugate': args.conjugate,
                   'n_rows': len(rows), 'results': rows}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload
    X, source = load_data(args)
    if args.experiment == 'compare':
        names = ('ktg', 'artg')
        results = [summarize(fit_model(name, X, args), name, X, args.conjugate)
                   for name in names]
    elif args.experiment == 'fit':
        model = fit_model(args.model, X, args)
        results = [summarize(model, args.model, X, args.conjugate)]
        if args.save_coef is not None:
            args.save_coef.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_coef, model.coef_)
    else:
        if args.save_coef is not None:
            raise ValueError('--save-coef is supported only for --experiment fit')
        if args.lambdas is None:
            raise ValueError('--experiment lambda-sweep requires --lambdas')
        if any(lam < 0 for lam in args.lambdas):
            raise ValueError('--lambdas must be nonnegative')
        results = [summarize(
            fit_model(args.model, X, args, group_lasso=True, lambda_=lam),
            args.model, X, args.conjugate) for lam in args.lambdas]
    payload = {'experiment': args.experiment, 'source': source,
               'data_shape': list(X.shape), 'lags': args.lags, 'results': results}
    result_text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(result_text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result_text + '\n', encoding='utf-8')
    return payload


if __name__ == '__main__':
    main()
