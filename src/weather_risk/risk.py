"""Loss-based VaR, expected shortfall, and quantile loss for Phase 3.

Targets are signed returns or price changes; losses are their negatives. Every
result stays in the target's units, and negative risk estimates are not clipped.
"""

import math
from collections.abc import Iterable
from numbers import Real
from statistics import NormalDist


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _confidence(value: float) -> float:
    confidence = _finite_number(value, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    return confidence


def empirical_var_es(targets: Iterable[float], confidence: float) -> tuple[float, float]:
    """Return empirical loss VaR and ES from equally weighted finite targets.

    VaR is the inverted empirical CDF: ordered loss ``ceil(n * confidence)``
    using one-based indexing. ES integrates that quantile function over the
    upper ``1 - confidence`` probability mass. Fractional mass at the boundary
    is included, so ES is not the mean of observations strictly above VaR.
    """
    confidence = _confidence(confidence)
    try:
        losses = sorted(-_finite_number(value, "target") for value in targets)
    except TypeError as exc:
        raise ValueError("targets must be a nonempty iterable of finite real numbers") from exc
    if not losses:
        raise ValueError("targets must be nonempty")
    n = len(losses)
    var = losses[math.ceil(n * confidence) - 1]
    tail_mass = n * (1 - confidence)
    whole = math.floor(tail_mass)
    fractional = tail_mass - whole
    # Normalize before summing to avoid overflow when the finite losses are large.
    weighted = [loss / tail_mass for loss in losses[n - whole:]]
    if fractional:
        weighted.append(losses[n - whole - 1] * (fractional / tail_mass))
    es = math.fsum(weighted)
    return var, es


def gaussian_var_es(mean: float, sigma: float, confidence: float) -> tuple[float, float]:
    """Return loss VaR and ES when the target is Normal(mean, sigma**2)."""
    mean = _finite_number(mean, "mean")
    sigma = _finite_number(sigma, "sigma")
    confidence = _confidence(confidence)
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    normal = NormalDist()
    z = normal.inv_cdf(confidence)
    return -mean + sigma * z, -mean + sigma * normal.pdf(z) / (1 - confidence)


def quantile_loss(loss: float, var: float, confidence: float) -> float:
    """Pinball loss for an upper loss quantile, not for a signed return quantile."""
    loss = _finite_number(loss, "loss")
    var = _finite_number(var, "var")
    confidence = _confidence(confidence)
    # Weight the endpoints before subtraction so a representable penalty does
    # not overflow merely because the unweighted difference is too large.
    if loss >= var:
        return confidence * loss - confidence * var
    weight = 1 - confidence
    return weight * var - weight * loss
