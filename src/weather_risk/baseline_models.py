"""Training-only zero-mean variance baselines for the Phase 3 experiment."""

import math

import numpy as np
from scipy.optimize import minimize


def _values(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("Returns must be a one-dimensional array containing finite values or NaN")
    return values


def _garch_parameters(params):
    omega, alpha, beta, gamma = (float(params.get(name, 0.0))
                                for name in ("omega", "alpha", "beta", "gamma"))
    initial = float(params["initial_variance"])
    if not all(math.isfinite(v) for v in (omega, alpha, beta, gamma, initial)):
        raise ValueError("GARCH parameters must be finite")
    if min(omega, alpha, beta, gamma) < 0 or initial <= 0:
        raise ValueError("GARCH coefficients must be nonnegative and initial variance positive")
    if alpha + beta + gamma / 2 >= 1:
        raise ValueError("GARCH persistence must be less than one")
    return omega, alpha, beta, gamma, initial


def garch_variances(values, params):
    """Return h[t], before observing y[t], preserving every missing grid row.

    A missing innovation advances by its conditional expectation under the
    symmetric zero-mean innovation distribution used by Gaussian QMLE.
    """
    values = _values(values)
    omega, alpha, beta, gamma, h = _garch_parameters(params)
    persistence = alpha + beta + gamma / 2
    result = np.empty(len(values), dtype=float)
    for i, value in enumerate(values):
        if not math.isfinite(h) or h <= 0:
            raise ValueError("GARCH recursion produced a nonpositive or nonfinite variance")
        result[i] = h
        if math.isnan(value):
            h = omega + persistence * h
        else:
            h = omega + beta * h + (alpha + gamma * (value < 0)) * value * value
    return result


def _garch_objective(coefficients, squared, negative, observed, asymmetric):
    """Average Gaussian quasi-likelihood and analytic recursion derivatives."""
    omega, alpha, beta = coefficients[:3]
    gamma = coefficients[3] if asymmetric else 0.0
    persistence = alpha + beta + gamma / 2
    h = 1.0
    d_omega = d_alpha = d_beta = d_gamma = 0.0
    loss = g_omega = g_alpha = g_beta = g_gamma = 0.0
    for square, neg, valid in zip(squared, negative, observed):
        if valid:
            loss += math.log(h) + square / h
            weight = (1.0 - square / h) / h
            g_omega += weight * d_omega
            g_alpha += weight * d_alpha
            g_beta += weight * d_beta
            g_gamma += weight * d_gamma
            d_omega = 1.0 + beta * d_omega
            d_alpha = square + beta * d_alpha
            d_beta = h + beta * d_beta
            d_gamma = square * neg + beta * d_gamma
            h = omega + beta * h + alpha * square + gamma * square * neg
        else:
            d_omega = 1.0 + persistence * d_omega
            d_alpha = h + persistence * d_alpha
            d_beta = h + persistence * d_beta
            d_gamma = 0.5 * h + persistence * d_gamma
            h = omega + persistence * h
    multiplier = 0.5 / int(observed.sum())
    gradient = np.array([g_omega, g_alpha, g_beta, g_gamma] if asymmetric
                        else [g_omega, g_alpha, g_beta])
    return float(loss * multiplier), gradient * multiplier


def fit_garch(values, asymmetric=False):
    """Fit stationary GARCH(1,1), or GJR-GARCH, by deterministic Gaussian QMLE.

    Three fixed SLSQP starts are compared only when they converge and satisfy
    the constraints. RMS normalization, the initial variance and all fitting
    inputs use training data only. Returned variances use the original units.
    An unsuccessful fit raises; it never silently substitutes another model.
    """
    values = _values(values)
    observed = np.isfinite(values)
    if observed.sum() < 20:
        raise ValueError("GARCH requires at least 20 observed training returns")
    with np.errstate(over="ignore", invalid="ignore"):
        variance = float(np.mean(values[observed] ** 2))
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("GARCH requires a finite positive training second moment")
    scale = math.sqrt(variance)
    normalized = values / scale
    squared = np.where(observed, normalized * normalized, 0.0)
    negative = normalized < 0
    weights = np.array([0.0, 1.0, 1.0, 0.5] if asymmetric else [0.0, 1.0, 1.0])
    starts = ([.05, .05, .85, .10], [.15, .10, .70, .10], [.05, .15, .75, .10]) if asymmetric else (
        [.05, .05, .90], [.15, .15, .70], [.05, .20, .75])
    bounds = [(1e-8, None), (0.0, .999), (0.0, .999)]
    if asymmetric:
        bounds.append((0.0, 1.998))
    diagnostics = []
    candidates = []
    for index, start in enumerate(starts):
        result = minimize(
            _garch_objective, np.asarray(start), args=(squared, negative, observed, asymmetric),
            method="SLSQP", jac=True, bounds=bounds,
            constraints={"type": "ineq", "fun": lambda p: .999 - float(weights @ p),
                         "jac": lambda p: -weights},
            options={"ftol": 1e-9, "maxiter": 200, "disp": False},
        )
        valid = (bool(result.success) and np.isfinite(result.fun) and np.isfinite(result.x).all()
                 and np.min(result.x) >= 0 and float(weights @ result.x) <= .999 + 1e-8)
        diagnostics.append({"start": index, "initial_parameters": list(start),
                            "success": valid, "optimizer_success": bool(result.success),
                            "status": int(result.status), "message": str(result.message),
                            "iterations": int(result.nit),
                            "objective": float(result.fun) if np.isfinite(result.fun) else None})
        if valid:
            candidates.append((float(result.fun), index, result.x))
    if not candidates:
        messages = "; ".join(f"start {d['start']}: {d['message']}" for d in diagnostics)
        raise RuntimeError(f"GARCH optimization failed for every deterministic start: {messages}")
    objective, selected, coefficients = min(candidates, key=lambda candidate: candidate[:2])
    omega, alpha, beta = coefficients[:3]
    gamma = float(coefficients[3]) if asymmetric else 0.0
    return {"omega": float(omega * variance), "alpha": float(alpha), "beta": float(beta),
            "gamma": gamma, "initial_variance": variance, "scale": scale,
            "persistence": float(alpha + beta + gamma / 2), "asymmetric": bool(asymmetric),
            "diagnostics": {"optimizer": "SLSQP", "selected_start": selected,
                            "starts": diagnostics, "observations": int(observed.sum()),
                            "grid_rows": len(values), "initialization": "training_second_moment",
                            "mean_quasi_negative_log_likelihood": objective + math.log(scale)}}


def _features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or np.isinf(X).any():
        raise ValueError("Features must be a two-dimensional array containing finite values or NaN")
    return X


def _design(X, medians, centers, scales):
    expanded = np.column_stack((np.where(np.isnan(X), medians, X), np.isnan(X).astype(float)))
    return np.column_stack((np.ones(len(X)), (expanded - centers) / scales))


def fit_log_variance(X, y, ridge=1.0):
    """Ridge regression of log(y² + floor), with training-only preprocessing.

    Include one missing indicator per feature and leave the intercept
    unpenalized. Duan's in-sample residual smearing corrects retransformation.
    """
    X = _features(X)
    y = _values(y)
    if len(X) != len(y) or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("Log-variance regression needs at least two aligned finite training targets")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("Ridge penalty must be finite and nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        squares = y * y
        second_moment = float(squares.mean())
    if not math.isfinite(second_moment):
        raise ValueError("Training squared targets must have a finite mean")
    floor = max(second_moment * 1e-6, np.finfo(float).tiny) if second_moment > 0 else 1e-12
    medians = np.array([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0
                        for column in X.T])
    expanded = np.column_stack((np.where(np.isnan(X), medians, X), np.isnan(X).astype(float)))
    centers = expanded.mean(axis=0)
    scales = expanded.std(axis=0)
    scales[scales <= np.finfo(float).eps] = 1.0
    design = _design(X, medians, centers, scales)
    target = np.log(squares + floor)
    # Augmented least squares handles redundant or entirely missing columns.
    penalty = np.sqrt(ridge) * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.lstsq(np.vstack((design, penalty)),
                                  np.concatenate((target, np.zeros(design.shape[1]))), rcond=None)[0]
    residuals = target - design @ coefficients
    smearing = float(np.mean(np.exp(residuals)))
    if not np.isfinite(coefficients).all() or not math.isfinite(smearing) or smearing <= 0:
        raise RuntimeError("Log-variance regression produced nonfinite coefficients or smearing")
    return {"coefficients": coefficients.tolist(), "medians": medians.tolist(),
            "centers": centers.tolist(), "scales": scales.tolist(), "log_floor": float(floor),
            "smearing": smearing, "ridge": float(ridge), "n_features": X.shape[1],
            "training_observations": len(y)}


def predict_log_variance(X, params):
    """Predict positive variance using frozen training transforms and smearing.

    Remove the additive target floor after retransformation, then apply that
    floor to the variance. Exponentiation is restricted to machine range.
    """
    X = _features(X)
    if X.shape[1] != params["n_features"]:
        raise ValueError("Prediction feature count differs from training")
    design = _design(X, np.asarray(params["medians"]), np.asarray(params["centers"]),
                     np.asarray(params["scales"]))
    log_prediction = design @ np.asarray(params["coefficients"]) + math.log(params["smearing"])
    limits = (math.log(np.finfo(float).tiny), math.log(np.finfo(float).max) - 1)
    variance = np.exp(np.clip(log_prediction, *limits)) - params["log_floor"]
    return np.maximum(variance, params["log_floor"])
