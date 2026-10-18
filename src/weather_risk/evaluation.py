"""Frozen Phase 5 numerical evaluation on an uncompressed observation grid.

Losses are negatives of the signed targets. NaNs denote unavailable rows, not
zero losses. Coverage tests use strict exceedances, so equality is a non-event.
The likelihood-ratio reference distributions are asymptotic; the binomial test
and its Clopper--Pearson interval assume independent Bernoulli observations.
"""

import hashlib
import json
import math
from numbers import Integral

import numpy as np
from scipy.special import xlogy
from scipy.stats import binomtest, chi2

from .risk import _confidence


def _array(values, name: str) -> np.ndarray:
    try:
        raw = np.asarray(values)
        if raw.dtype.kind not in "iuf" or raw.ndim != 1:
            raise ValueError
        result = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional real numeric array") from exc
    if np.isinf(result).any():
        raise ValueError(f"{name} may contain NaN, but not infinity")
    return result


def _matched_arrays(**arrays) -> list[np.ndarray]:
    result = [_array(value, name) for name, value in arrays.items()]
    if any(value.shape != result[0].shape for value in result[1:]):
        raise ValueError("all arrays must have the same shape")
    return result


def _bernoulli_log_likelihood(successes: int, failures: int, probability: float) -> float:
    # xlogy(0, 0) = 0 handles fitted probabilities of exactly zero or one.
    return float(xlogy(successes, probability) + xlogy(failures, 1 - probability))


def _unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason, "statistic": None, "pvalue": None}


def coverage_tests(losses, var, confidence: float, *, grid_indices=None) -> dict:
    """Return coverage diagnostics with JSON-safe results, including failures.

    Supply rows in original grid order and retain missing rows as NaN. When
    passing an already filtered sample, ``grid_indices`` must contain its
    strictly increasing original integer positions. Independence uses only
    adjacent finite pairs whose original indices differ by one. It is
    unavailable if either previous-state transition row has no observations.

    ``unconditional`` is Kupiec's LR test (one degree of freedom); ``exact``
    contains the two-sided binomial p-value and 95% Clopper--Pearson bounds for
    the exceedance rate. ``conditional`` uses the conventional sum of full
    sample unconditional LR and adjacent-pair independence LR (two degrees of
    freedom); with missing rows this is an asymptotic diagnostic, not an exact
    finite-sample joint likelihood test.
    """
    confidence = _confidence(confidence)
    losses, var = _matched_arrays(losses=losses, var=var)
    if grid_indices is None:
        indices = np.arange(len(losses), dtype=np.int64)
    else:
        indices = np.asarray(grid_indices)
        if (indices.shape != losses.shape or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices[1:] <= indices[:-1])):
            raise ValueError("grid_indices must be matching, strictly increasing nonnegative integers")
    valid = np.isfinite(losses) & np.isfinite(var)
    hits = losses > var
    n = int(valid.sum())
    k = int(hits[valid].sum())
    nominal = 1 - confidence
    adjacent = valid[:-1] & valid[1:] & ((indices[1:] - indices[:-1]) == 1)
    previous, current = hits[:-1][adjacent], hits[1:][adjacent]
    counts = np.bincount(previous.astype(int) * 2 + current.astype(int), minlength=4)
    n00, n01, n10, n11 = [int(value) for value in counts]
    transitions = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
    result = {
        "confidence": confidence, "grid_rows": len(losses), "observations": n,
        "exceedances": k, "nominal_exceedance_rate": nominal,
        "exceedance_rate": k / n if n else None,
        "transitions": transitions, "adjacent_pairs": int(adjacent.sum()),
    }
    if not n:
        result.update(unconditional=_unavailable("no finite loss/VaR pairs"),
                      exact={"available": False, "reason": "no finite loss/VaR pairs",
                             "pvalue": None, "ci_low": None, "ci_high": None})
    else:
        null_ll = _bernoulli_log_likelihood(k, n - k, nominal)
        fitted_ll = _bernoulli_log_likelihood(k, n - k, k / n)
        statistic = max(0., 2 * (fitted_ll - null_ll))
        exact = binomtest(k, n, nominal, alternative="two-sided")
        interval = exact.proportion_ci(confidence_level=.95, method="exact")
        result.update(
            unconditional={"available": True, "reason": None,
                           "statistic": statistic, "pvalue": float(chi2.sf(statistic, 1))},
            exact={"available": True, "reason": None, "pvalue": float(exact.pvalue),
                   "ci_low": float(interval.low), "ci_high": float(interval.high)},
        )
    row0, row1 = n00 + n01, n10 + n11
    if not row0 or not row1:
        reason = "transition row unidentified: no adjacent pairs after " + (
            "either state" if not row0 and not row1 else "non-exceedances" if not row0 else "exceedances")
        result["independence"] = _unavailable(reason)
        result["conditional"] = _unavailable(reason)
    else:
        p01, p11 = n01 / row0, n11 / row1
        pooled = (n01 + n11) / (row0 + row1)
        null_ll = _bernoulli_log_likelihood(n01 + n11, n00 + n10, pooled)
        fitted_ll = (_bernoulli_log_likelihood(n01, n00, p01)
                     + _bernoulli_log_likelihood(n11, n10, p11))
        statistic = max(0., 2 * (fitted_ll - null_ll))
        result["independence"] = {
            "available": True, "reason": None, "statistic": statistic,
            "pvalue": float(chi2.sf(statistic, 1)), "p01": p01, "p11": p11,
        }
        joint_statistic = result["unconditional"]["statistic"] + statistic
        result["conditional"] = {
            "available": True, "reason": None, "statistic": joint_statistic,
            "pvalue": float(chi2.sf(joint_statistic, 2)),
        }
    return result


def _es_arrays(losses, var, es, confidence: float):
    confidence = _confidence(confidence)
    losses, var, es = _matched_arrays(losses=losses, var=var, es=es)
    if np.any(es[np.isfinite(es)] <= 0):
        raise ValueError("ES must be strictly positive for the FZ0 score and relative residual")
    valid = np.isfinite(losses) & np.isfinite(var) & np.isfinite(es)
    return losses, var, es, confidence, valid


def fz0_score(losses, var, es, confidence: float) -> np.ndarray:
    """Upper-loss-tail FZ0 joint VaR/ES score; smaller is better.

    The ES domain is strictly positive. Arrays share the original grid; any
    incomplete row returns NaN. The exact score is
    I(L > q)(L-q)/((1-alpha)e) + q/e + log(e) - 1.
    """
    losses, var, es, confidence, valid = _es_arrays(losses, var, es, confidence)
    result = np.full(losses.shape, np.nan)
    q, e, loss = var[valid], es[valid], losses[valid]
    result[valid] = np.maximum(loss - q, 0) / ((1 - confidence) * e) + q / e + np.log(e) - 1
    if not np.isfinite(result[valid]).all():
        raise ValueError("FZ0 score overflowed for finite input")
    return result


def es_identification_residual(losses, var, es, confidence: float) -> np.ndarray:
    """Relative ES calibration residual; joint calibration implies mean zero.

    For continuous losses this is (q + I(L>q)(L-q)/(1-alpha) - e)/e.
    Interpret jointly with VaR calibration, not as an isolated test of ES.
    """
    losses, var, es, confidence, valid = _es_arrays(losses, var, es, confidence)
    result = np.full(losses.shape, np.nan)
    q, e, loss = var[valid], es[valid], losses[valid]
    result[valid] = (q + np.maximum(loss - q, 0) / (1 - confidence) - e) / e
    if not np.isfinite(result[valid]).all():
        raise ValueError("ES identification residual overflowed for finite input")
    return result


def stable_seed(seed: int, key: str = "") -> int:
    """Derive a stable independent stream key without Python's salted hash()."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(key, str):
        raise ValueError("key must be a string")
    payload = json.dumps([int(seed), key], separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def moving_block_bootstrap(values, *, block_length: int = 20, replications: int = 1000,
                           seed: int = 1729, key: str = "", min_observations: int = 100) -> dict:
    """Bootstrap the finite-row mean while preserving original-grid NaN holes.

    Sample overlapping, non-circular blocks uniformly with replacement,
    concatenate, then truncate to the original grid length. Missing values
    remain in the sampled blocks; finite rows are never compressed first.
    At least ``min_observations`` finite source rows and 95% finite bootstrap
    means are required. The 95% interval is the percentile interval. The
    centered two-sided p-value is (1 + count(|mean* - mean| >= |mean|))/(Bvalid+1).
    This is an approximate weak-dependence diagnostic, not an exact test.
    """
    values = _array(values, "values")
    for name, value in (("block_length", block_length), ("replications", replications),
                        ("min_observations", min_observations)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    stream_seed = stable_seed(seed, key)
    observed = int(np.isfinite(values).sum())
    estimate = float(np.mean(values[np.isfinite(values)])) if observed else None
    if estimate is not None and not math.isfinite(estimate):
        raise ValueError("finite input mean overflowed")
    result = {
        "available": False, "reason": None, "grid_rows": len(values),
        "observations": observed, "mean": estimate, "ci_low": None, "ci_high": None,
        "pvalue": None, "block_length": int(block_length), "replications": int(replications),
        "valid_replications": 0, "seed": stream_seed,
    }
    if observed < min_observations:
        result["reason"] = f"fewer than {min_observations} finite observations"
        return result
    if len(values) < block_length:
        result["reason"] = "original grid shorter than block length"
        return result
    rng = np.random.default_rng(stream_seed)
    block_count = math.ceil(len(values) / block_length)
    starts = rng.integers(0, len(values) - block_length + 1, size=(replications, block_count))
    indices = (starts[..., None] + np.arange(block_length)).reshape(replications, -1)[:, :len(values)]
    samples = values[indices]
    counts = np.isfinite(samples).sum(axis=1)
    means = np.full(replications, np.nan)
    np.divide(np.nansum(samples, axis=1), counts, out=means, where=counts > 0)
    means = means[np.isfinite(means)]
    result["valid_replications"] = len(means)
    if len(means) < math.ceil(.95 * replications):
        result["reason"] = "fewer than 95% of bootstrap replications have finite means"
        return result
    low, high = np.quantile(means, [.025, .975], method="linear")
    pvalue = (1 + int(np.count_nonzero(np.abs(means - estimate) >= abs(estimate)))) / (len(means) + 1)
    result.update(available=True, ci_low=float(low), ci_high=float(high), pvalue=pvalue)
    return result


def holm_adjust(p_values) -> list[float | None]:
    """Holm familywise adjustment, retaining unavailable planned hypotheses.

    None and NaN stay None in the output and count toward the prespecified
    family size (equivalent to a p-value of one for multiplicity counting).
    The family must be assembled before seeing which results are significant.
    """
    values = list(p_values)
    finite = []
    for index, value in enumerate(values):
        if value is None:
            continue
        if isinstance(value, (bool, str, complex)):
            raise ValueError("p-values must lie between zero and one, or be None/NaN")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("p-values must lie between zero and one, or be None/NaN") from exc
        if math.isnan(numeric):
            continue
        if not math.isfinite(numeric) or not 0 <= numeric <= 1:
            raise ValueError("p-values must lie between zero and one, or be None/NaN")
        finite.append((numeric, index))
    adjusted = [None] * len(values)
    previous = 0.
    for rank, (value, index) in enumerate(sorted(finite)):
        previous = max(previous, min(1., (len(values) - rank) * value))
        adjusted[index] = previous
    return adjusted
