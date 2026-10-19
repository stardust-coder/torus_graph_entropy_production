"""Maximum-likelihood KTG / AR-TG estimators for circular time series.

Inputs: X.shape == (n_samples, n_variables), angles in radians.
No intercept. Coefficients have shape (lags, d, d, k), where k=2 (KTG)
or k=4 (ARTG); coef_[lag, target, source, feature].

Feature order: cos(y-x), sin(y-x), [cos(y+x), sin(y+x)].
With group_lasso=True, minimize mean negative conditional log likelihood +
lambda_ * sum_{lag,target,source} ||coef_[lag,target,source,:]||_2
by monotone restarted FISTA with backtracking.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from scipy.special import i0e, i1e
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

_TWO_PI = 2.0 * np.pi


class _PhaseEstimator(BaseEstimator):
    _n_features = None

    def __init__(self, lags=1, group_lasso=False, lambda_=None, n_lambdas=60, lambda_min_ratio=0.01,
                 max_iter=10000, tol=1e-7, initial_step=1.0, verbose=False, lambda_jobs=1):
        self.lags = lags
        self.group_lasso = group_lasso
        self.lambda_ = lambda_
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.initial_step = initial_step
        self.verbose = verbose
        self.lambda_jobs = lambda_jobs

    def _validate_X(self, X, *, fitting=False):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError('X must be a finite (N, d) array of angles in radians.')
        if X.shape[0] <= self.lags:
            raise ValueError('N must exceed lags.')
        if fitting:
            if not isinstance(self.lags, (int, np.integer)) or self.lags < 1:
                raise ValueError('lags must be a positive integer.')
            if not isinstance(self.group_lasso, (bool, np.bool_)):
                raise ValueError('group_lasso must be a bool.')
            if ((self.lambda_ is not None and self.lambda_ < 0) or self.max_iter < 1
                    or self.tol <= 0 or self.initial_step <= 0
                    or self.n_lambdas < 2 or not 0 < self.lambda_min_ratio < 1):
                raise ValueError('lambda_ >= 0 or None, n_lambdas >= 2, 0 < lambda_min_ratio < 1, max_iter >= 1, tol > 0 and initial_step > 0 required.')
        else:
            check_is_fitted(self, 'coef_')
            if X.shape[1] != self.n_features_in_:
                raise ValueError('Number of variables differs from fit data.')
        return X

    def _features(self, X):
        n, d = X.shape
        past = np.stack([X[self.lags - lag:n - lag] for lag in range(1, self.lags + 1)], axis=1)
        cp, sp = np.cos(past), np.sin(past)  # (n-lags, lag, source)
        # The cosine/sine coefficients of each h_k(past, current)
        a = np.stack((cp, -sp), axis=-1)
        b = np.stack((sp, cp), axis=-1)
        if self._n_features == 4:
            a = np.concatenate((a, np.stack((cp, sp), axis=-1)), axis=-1)
            b = np.concatenate((b, np.stack((-sp, cp), axis=-1)), axis=-1)
        return a, b

    @staticmethod
    def _log_i0(r):
        return np.log(i0e(r)) + r

    @staticmethod
    def _linear_predictors(theta, a, b):
        """BLAS matrix products, without expensive multi-index einsum contraction.

        Flatten (lag, source, feature) consistently in both feature arrays and
        per-target parameter rows. Works for KTG and AR-TG, any number of lags.
        """
        n = a.shape[0]
        coef = theta.reshape(theta.shape[0], theta.shape[1], -1)
        coef = np.ascontiguousarray(coef.transpose(1, 0, 2).reshape(theta.shape[1], -1))
        af = a.reshape(n, -1)
        bf = b.reshape(n, -1)
        return af @ coef.T, bf @ coef.T

    def _loss_grad(self, theta, a, b, obs_cos, obs_sin):
        ec, es = self._linear_predictors(theta, a, b)
        radius = np.hypot(ec, es)
        loss = np.mean(np.sum(self._log_i0(radius) + np.log(_TWO_PI)
                              - obs_cos * ec - obs_sin * es, axis=1))
        ratio = np.divide(i1e(radius), i0e(radius), out=np.zeros_like(radius), where=radius > 0)
        multiplier = np.divide(ratio, radius, out=np.full_like(radius, 0.5), where=radius > 1e-8)
        dc = multiplier * ec - obs_cos
        ds = multiplier * es - obs_sin
        n, lag, source, feat = a.shape
        g_flat = (dc.T @ a.reshape(n, -1) + ds.T @ b.reshape(n, -1)) / n
        grad = g_flat.reshape(theta.shape[1], lag, source, feat).transpose(1, 0, 2, 3).copy()
        return float(loss), grad

    def _loss_only(self, theta, a, b, obs_cos, obs_sin):
        ec, es = self._linear_predictors(theta, a, b)
        radius = np.hypot(ec, es)
        return float(np.mean(np.sum(self._log_i0(radius) + np.log(_TWO_PI)
                              - obs_cos * ec - obs_sin * es, axis=1)))

    def _kkt_residual(self, theta, lam, a, b, obs_cos, obs_sin, step):
        _, grad = self._loss_grad(theta, a, b, obs_cos, obs_sin)
        mapped = (theta - self._prox(theta - step * grad, step * lam)) / step
        return float(np.linalg.norm(mapped) / max(1.0, np.linalg.norm(theta)))

    def _get_prepared(self, X):
        prepared = getattr(self, '_prepared', None)
        if prepared is not None:
            return prepared
        a, b = self._features(X)
        return a, b, np.cos(X[self.lags:]), np.sin(X[self.lags:])

    @staticmethod
    def _penalty(theta):
        return float(np.linalg.norm(theta, axis=-1).sum())

    @staticmethod
    def _prox(theta, threshold):
        norms = np.linalg.norm(theta, axis=-1, keepdims=True)
        scale = np.maximum(0.0, 1.0 - threshold / np.maximum(norms, 1e-300))
        return theta * scale

    def fit(self, X, y=None, progress_callback=None):
        """Fit with MLE or group Lasso; optional callback receives AIC-path progress."""
        X = self._validate_X(X, fitting=True)
        if not self.group_lasso or self.lambda_ is not None:
            self.best_lambda_ = float(self.lambda_ if self.group_lasso and self.lambda_ is not None else 0.0)
            self.aic_path_ = []
            return self._fit_fixed(X, self.best_lambda_)

        # Lambda max annihilates all groups for average negative log likelihood.
        a, b = self._features(X)
        oc, os = np.cos(X[self.lags:]), np.sin(X[self.lags:])
        zero = np.zeros((self.lags, X.shape[1], X.shape[1], self._n_features))
        _, grad = self._loss_grad(zero, a, b, oc, os)
        lambda_max = float(np.linalg.norm(grad, axis=-1).max())
        if lambda_max <= 1e-14:
            grid = np.array([0.0])
        else:
            grid = np.r_[np.geomspace(lambda_max, lambda_max * self.lambda_min_ratio,
                                       self.n_lambdas), 0.0]
        # Each candidate starts independently from zero: no warm start.
        results = Parallel(n_jobs=self.lambda_jobs, backend='loky', batch_size=1)(
            delayed(_fit_aic_candidate)(self, X, float(lam)) for lam in grid
        )
        path = []
        best = None
        for idx, (lam, aic, candidate) in enumerate(results, start=1):
            if progress_callback is not None:
                progress_callback(idx, len(grid), float(lam), float(aic))
            path.append({'lambda': float(lam), 'aic': float(aic),
                         'df': int(candidate.effective_df()),
                         'converged': bool(candidate.success_),
                         'refit_converged': bool(candidate.refit_success_)})
            if np.isfinite(aic) and candidate.success_ and candidate.refit_success_ and (best is None or aic < best[0]):
                best = (aic, candidate)
        if best is None:
            raise RuntimeError('All lambda candidates failed to produce finite AIC')
        self.best_lambda_ = float(best[1].best_lambda_)
        self.aic_path_ = path
        for attr in ('coef_', 'n_features_in_', 'n_iter_', 'success_',
                     'objective_', 'objective_history_', 'refit_coef_',
                     'refit_success_', 'refit_n_iter_'):
            setattr(self, attr, getattr(best[1], attr))
        self.penalized_coef_ = self.coef_.copy()
        self.coef_ = self.refit_coef_.copy()
        return self

    def _fit_fixed(self, X, lam, initial_coef=None):
        """Fit for a fixed lambda (used by AIC path)."""
        X = self._validate_X(X, fitting=True)
        n, d = X.shape
        self.n_features_in_ = d
        a, b, obs_cos, obs_sin = self._get_prepared(X)
        shape = (self.lags, d, d, self._n_features)
        theta = np.zeros(shape, dtype=float) if initial_coef is None else np.asarray(initial_coef, dtype=float).copy()
        if theta.shape != shape or not np.isfinite(theta).all():
            raise ValueError('initial_coef must be finite with shape (lags,d,d,n_features)')
        history = []
        fit_started = time.perf_counter()
        self.backtracking_trials_ = 0
        self.kkt_residual_ = float('nan')
        if self.group_lasso and lam > 0:
            # Monotone FISTA with adaptive restart and backtracking. The line
            # search only evaluates the loss, never the gradient at trial points.
            step = float(self.initial_step)
            y = theta.copy()
            momentum = 1.0
            obj_previous = self._loss_only(theta, a, b, obs_cos, obs_sin)
            success = False
            for it in range(1, self.max_iter + 1):
                loss, grad = self._loss_grad(y, a, b, obs_cos, obs_sin)
                for _ in range(70):
                    candidate = self._prox(y - step * grad, step * lam)
                    diff = candidate - y
                    self.backtracking_trials_ += 1
                    candidate_loss = self._loss_only(candidate, a, b, obs_cos, obs_sin)
                    quadratic_bound = loss + np.sum(grad * diff) + np.sum(diff * diff) / (2 * step)
                    if candidate_loss <= quadratic_bound + 1e-11:
                        break
                    step *= 0.5
                else:
                    raise RuntimeError('FISTA backtracking failed.')
                objective = candidate_loss + lam * self._penalty(candidate)
                if objective > obj_previous + 1e-10:
                    # Non-monotone acceleration: restart from the last iterate.
                    y = theta.copy()
                    momentum = 1.0
                    loss, grad = self._loss_grad(y, a, b, obs_cos, obs_sin)
                    for _ in range(70):
                        candidate = self._prox(y - step * grad, step * lam)
                        diff = candidate - y
                        self.backtracking_trials_ += 1
                        candidate_loss = self._loss_only(candidate, a, b, obs_cos, obs_sin)
                        if candidate_loss <= loss + np.sum(grad * diff) + np.sum(diff * diff) / (2 * step) + 1e-11:
                            break
                        step *= 0.5
                    else:
                        raise RuntimeError('FISTA restart line search failed.')
                    objective = candidate_loss + lam * self._penalty(candidate)
                history.append(objective)
                # Stationarity proxy at the accepted iterate (not only change
                # at the extrapolated point). Strict KKT checking can be done
                # through proximal_gradient_residual() after fitting.
                change = np.linalg.norm(candidate-theta) / max(1.0, np.linalg.norm(theta))
                prev_theta = theta
                theta = candidate
                obj_previous = objective
                if self.verbose and it % 100 == 0:
                    print(f'FISTA iteration={it}, objective={objective:.8f}')
                # Gradient mapping is the composite optimality residual.
                # Parameter change alone can be tiny because line search shrank
                # the step, even when the KKT conditions are far from satisfied.
                if it == 1 or it % 25 == 0 or change < self.tol:
                    self.kkt_residual_ = self._kkt_residual(
                        theta, lam, a, b, obs_cos, obs_sin, step)
                    if self.kkt_residual_ <= self.tol:
                        success = True
                        break
                next_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
                extrapolated = theta + ((momentum-1)/next_momentum) * (theta-prev_theta)
                if np.sum((theta-prev_theta)*(extrapolated-theta)) < 0:
                    y = theta.copy()
                    momentum = 1.0
                else:
                    y = extrapolated
                    momentum = next_momentum
                step = min(step*1.05, self.initial_step)
            self.n_iter_ = it
            self.success_ = success
            if not success:
                self.kkt_residual_ = self._kkt_residual(
                    theta, lam, a, b, obs_cos, obs_sin, step)
        else:
            def fn(flat):
                loss, grad = self._loss_grad(flat.reshape(shape), a, b, obs_cos, obs_sin)
                return loss, grad.ravel()
            result = minimize(fn, theta.ravel(), jac=True, method='L-BFGS-B',
                              options={'maxiter': self.max_iter, 'ftol': self.tol})
            theta = result.x.reshape(shape)
            self.n_iter_ = int(result.nit)
            self.success_ = bool(result.success)
            self.kkt_residual_ = float(np.linalg.norm(result.jac) / max(1., np.linalg.norm(theta)))
            history = [float(result.fun)]
        self.fit_seconds_ = time.perf_counter() - fit_started
        self.coef_ = theta
        self.objective_history_ = np.asarray(history)
        self.objective_ = history[-1]
        return self

    def _conditional_params(self, X):
        X = self._validate_X(X)
        a, b, _, _ = self._get_prepared(X)
        ec = np.einsum('nljk,lijk->ni', a, self.coef_, optimize=True)
        es = np.einsum('nljk,lijk->ni', b, self.coef_, optimize=True)
        return ec, es

    def predict(self, X):
        """Conditional mean directions; output shape (N-lags, d), radians."""
        ec, es = self._conditional_params(X)
        return np.arctan2(es, ec)

    def predict_concentration(self, X):
        """von Mises concentration parameters, shape (N-lags, d)."""
        ec, es = self._conditional_params(X)
        return np.hypot(ec, es)

    def score_samples(self, X):
        """Conditional log density per transition, shape (N-lags,)."""
        X = self._validate_X(X)
        ec, es = self._conditional_params(X)
        Y = X[self.lags:]
        return np.sum(ec * np.cos(Y) + es * np.sin(Y)
                      - self._log_i0(np.hypot(ec, es)) - np.log(_TWO_PI), axis=1)

    def score(self, X, y=None):
        """Mean conditional log density per transition (larger is better)."""
        return float(np.mean(self.score_samples(X)))

    def entropy_production(self, X, conjugate=False):
        """Empirical forward/reversed conditional log-likelihood ratio.

        Returns the mean (nats per modeled time step) of
            log p_theta(X[t] | X[t-1],...,X[t-L])
        minus the corresponding log likelihood evaluated on the reversed
        series, using the SAME fitted parameters theta (not a refit).

        For a correctly specified stationary Markov process of order L,
        this pathwise log ratio approaches the entropy-production rate
        (path-space relative entropy rate) as the sample size grows.
        The stationary initial-block density is omitted; its contribution
        is a boundary term that vanishes per time step asymptotically.
        The finite-sample estimate can be negative.

        When conjugate=True, reversed observations are negated, representing
        phase conjugation (complex conjugation of exp(1j*phase)).
        Log normalizers are included: they do not generally cancel for
        an individual forward/backward transition or finite sequence.
        """
        X = self._validate_X(X)
        forward = self.score_samples(X)
        reversed_X = X[::-1]
        if conjugate:
            reversed_X = -reversed_X
        backward = self.score_samples(reversed_X)
        return float(np.mean(forward) - np.mean(backward))

    def effective_df(self, threshold=1e-7):
        """Approximate degrees of freedom: active groups times group dimension.

        This active-parameter count is a plug-in AIC approximation for group Lasso,
        not an exact generalized degrees-of-freedom correction.
        """
        check_is_fitted(self, 'coef_')
        return int(np.count_nonzero(np.linalg.norm(self.coef_, axis=-1) > threshold)
                   * self._n_features)

    def refit_active(self, X, threshold=1e-7):
        """Unpenalized conditional MLE restricted to active Group Lasso edges.

        Does not modify coef_: the Group Lasso solution remains the output.
        The refitted parameters are stored separately as refit_coef_.
        """
        X = self._validate_X(X)
        active = np.linalg.norm(self.coef_, axis=-1) > threshold
        a, b, oc, os = self._get_prepared(X)
        refit = np.zeros_like(self.coef_)
        if not active.any():
            self.refit_coef_ = refit
            self.refit_success_ = True
            self.refit_n_iter_ = 0
            return self

        # One group = last-axis block, so Boolean masking retains all h coefficients.
        def fn(flat):
            theta = np.zeros_like(self.coef_)
            theta[active] = flat.reshape(-1, self._n_features)
            loss, grad = self._loss_grad(theta, a, b, oc, os)
            return loss, grad[active].ravel()

        x0 = self.coef_[active].ravel()
        result = minimize(fn, x0, jac=True, method='L-BFGS-B',
                          options={'maxiter': self.max_iter, 'ftol': self.tol})
        refit[active] = result.x.reshape(-1, self._n_features)
        self.refit_coef_ = refit
        self.refit_success_ = bool(result.success)
        self.refit_n_iter_ = int(result.nit)
        return self

    def aic(self, X, refit=True):
        """Support-refitted conditional AIC (default), or naive penalized-fit AIC.

        With refit=True, performs an unpenalized restricted MLE on X using the
        support selected by coef_, without overwriting coef_. Compute AIC on
        training data, not held-out data. df = number of freely refitted values.
        """
        X = self._validate_X(X)
        if refit:
            self.refit_active(X)
            theta = self.refit_coef_
        else:
            theta = self.coef_
        a, b, _, _ = self._get_prepared(X)
        ec = np.einsum('nljk,lijk->ni', a, theta, optimize=True)
        es = np.einsum('nljk,lijk->ni', b, theta, optimize=True)
        y = X[self.lags:]
        loglik = np.sum(ec * np.cos(y) + es * np.sin(y)
                        - self._log_i0(np.hypot(ec, es)) - np.log(_TWO_PI))
        return float(-2 * loglik + 2 * self.effective_df())

    def edge_strengths(self, aggregate_lags=False):
        """Group L2 norms. (lags,target,source), or (target,source) pooled."""
        check_is_fitted(self, 'coef_')
        if aggregate_lags:
            return np.sqrt(np.sum(self.coef_ ** 2, axis=(0, 3)))
        return np.linalg.norm(self.coef_, axis=-1)


class KTG(_PhaseEstimator):
    """Rotational phase-difference model (2 coefficients per directed edge)."""
    _n_features = 2


class ARTG(_PhaseEstimator):
    """Full phase difference + sum model (4 coefficients per directed edge)."""
    _n_features = 4


def _fit_aic_candidate(template, X, lam):
    """Independent process worker: penalized fit followed by restricted refit AIC."""
    candidate = template.__class__(lags=template.lags, group_lasso=True,
        lambda_=lam, n_lambdas=template.n_lambdas,
        lambda_min_ratio=template.lambda_min_ratio, max_iter=template.max_iter,
        tol=template.tol, initial_step=template.initial_step,
        verbose=False, lambda_jobs=1)
    candidate.fit(X)
    aic = candidate.aic(X, refit=True)
    return lam, aic, candidate
