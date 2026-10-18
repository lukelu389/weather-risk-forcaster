"""Fixed-notional CAD loss sensitivities for the existing equity basket.

Each observation starts with an independent CAD notional. A target R is the
basket's log return in percent, giving loss N * (1 - exp(R / 100)). These are
interval risk sensitivities, not a self-financing NAV or a turnover estimate.
"""

import math

import numpy as np
from scipy.special import erfcx, log_ndtr, ndtri

from .risk import _confidence, _finite_number, empirical_var_es


def _budget(notional: float, cost_bps: float) -> tuple[float, float, float]:
    notional = _finite_number(notional, "notional")
    cost_bps = _finite_number(cost_bps, "cost_bps")
    if notional <= 0:
        raise ValueError("notional must be strictly positive")
    if cost_bps < 0:
        raise ValueError("cost_bps must be nonnegative")
    cost = notional * (cost_bps / 10000)
    if not math.isfinite(cost):
        raise ValueError("cost budget exceeds finite numeric range")
    return notional, cost_bps, cost


def _loss_from_log_factor(log_factor: float, notional: float) -> float:
    # expm1 avoids cancellation for small returns. The alternate expression
    # permits a tiny notional to offset an otherwise overflowing exponential.
    try:
        if log_factor > 700:
            loss = notional - math.exp(math.log(notional) + log_factor)
        else:
            loss = -notional * math.expm1(log_factor)
    except OverflowError as exc:
        raise ValueError("CAD loss exceeds finite numeric range") from exc
    if not math.isfinite(loss):
        raise ValueError("CAD loss exceeds finite numeric range")
    return loss


def _result(var: float, es: float, confidence: float, notional: float,
            cost_bps: float, cost: float) -> dict:
    net_var, net_es = var + cost, es + cost
    if not math.isfinite(net_var) or not math.isfinite(net_es):
        raise ValueError("cost-adjusted CAD risk exceeds finite numeric range")
    return {
        "confidence": confidence, "notional_cad": notional,
        "cost_bps": cost_bps, "cost_budget_cad": cost,
        "gross_var_cad": var, "gross_es_cad": es,
        "var_cad": net_var, "es_cad": net_es,
    }


def log_returns_to_cad_losses(targets, *, notional: float = 1_000_000.) -> np.ndarray:
    """Convert finite log-percent targets to fixed-start-notional CAD losses.

    Returns a one-dimensional array and does not mutate the input. No missing
    values are silently dropped. Returns can imply negative losses (gains).
    """
    notional, _, _ = _budget(notional, 0.)
    try:
        values = [_finite_number(value, "target") for value in targets]
    except TypeError as exc:
        raise ValueError("targets must be a one-dimensional iterable of finite real numbers") from exc
    return np.array([_loss_from_log_factor(value / 100, notional) for value in values], dtype=float)


def gaussian_cad_var_es(mean: float, sigma: float, confidence: float, *,
                        notional: float = 1_000_000., cost_bps: float = 0.) -> dict:
    """Exact upper-loss VaR/ES for R ~ Normal(mean, sigma**2), in log percent.

    With z = Phi^-1(confidence), gross VaR is N(1-exp((mean-sigma*z)/100)).
    Gross ES is N[1-exp(mean/100+sigma²/20000)
    * Phi(-z-sigma/100)/(1-confidence)]. The tail expectation is evaluated
    in log space; large sigma uses a scaled complementary error function to
    avoid cancellation of two squared terms. A zero sigma is deterministic.

    The explicit cost budget N*cost_bps/10000 shifts both VaR and ES. It is a
    sensitivity assumption, not an estimate of realized transaction costs.
    Negative VaR and ES are retained.
    """
    mean = _finite_number(mean, "mean")
    sigma = _finite_number(sigma, "sigma")
    confidence = _confidence(confidence)
    notional, cost_bps, cost = _budget(notional, cost_bps)
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    scaled_mean, scaled_sigma = mean / 100, sigma / 100
    z = float(ndtri(confidence))
    var = _loss_from_log_factor(scaled_mean - scaled_sigma * z, notional)
    if sigma == 0:
        es = var
    else:
        shifted_z = z + scaled_sigma
        if scaled_sigma > 10 and shifted_z > 0:
            # exp(t²/2)*Phi(-z-t) = exp(-t*z-z²/2)*erfcx((z+t)/sqrt(2))/2.
            log_tail_factor = (scaled_mean - scaled_sigma * z - z * z / 2
                               + math.log(float(erfcx(shifted_z / math.sqrt(2))))
                               - math.log(2) - math.log1p(-confidence))
        else:
            log_tail_factor = (scaled_mean + scaled_sigma * scaled_sigma / 2
                               + float(log_ndtr(-shifted_z)) - math.log1p(-confidence))
        es = _loss_from_log_factor(log_tail_factor, notional)
    return _result(var, es, confidence, notional, cost_bps, cost)


def empirical_cad_var_es(targets, confidence: float, *,
                         notional: float = 1_000_000., cost_bps: float = 0.) -> dict:
    """Empirical CAD VaR and fractional-tail ES after transforming each return.

    VaR is the inverted empirical loss CDF; ES integrates its upper tail,
    including fractional boundary mass. Transforming a log-return ES itself
    would be incorrect because the CAD loss transform is nonlinear.
    """
    confidence = _confidence(confidence)
    notional, cost_bps, cost = _budget(notional, cost_bps)
    losses = log_returns_to_cad_losses(targets, notional=notional)
    # The shared routine accepts signed targets and negates them to get losses.
    var, es = empirical_var_es(-losses, confidence)
    return _result(var, es, confidence, notional, cost_bps, cost)
