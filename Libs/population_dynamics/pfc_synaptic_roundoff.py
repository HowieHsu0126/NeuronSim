"""Exact interval/error enclosures for the fixed NMDA s/x arithmetic DAG.

The model assumes IEEE754 binary64 nearest/ties-even, gradual underflow, finite
operations, and the stated unfused operation order. It does not verify compiler,
FMA, FTZ/DAZ or native backend behavior. Same-event-history propagation assumes
identical valid refractory-spaced spikes and the same clock. No source confidence,
voltage/feedback or GABA aggregate qualification follows from these bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .pfc_synaptic_trace_bounds import (
    NmdaParameters, nmda_invariance_bounds, refractory_x_ceiling, rk2_decay_factor,
)

ZERO = Fraction(0)
U = Fraction(1, 2 ** 53)
ETA = Fraction(1, 2 ** 1074)
MAX_FINITE = Fraction((2 ** 53 - 1) * 2 ** 971)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("SYNAPTIC_ROUNDOFF_PROOF_CONDITION: " + reason)


def _fraction(value: Fraction, name: str, *, positive=False, nonnegative=False) -> None:
    _require(type(value) is Fraction, name + " must be an exact Fraction")
    if positive:
        _require(value > 0, name + " must be positive")
    if nonnegative:
        _require(value >= 0, name + " must be nonnegative")


@dataclass(frozen=True)
class Interval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self):
        _fraction(self.lower, "interval.lower")
        _fraction(self.upper, "interval.upper")
        _require(self.lower <= self.upper, "interval endpoints are reversed")

    @property
    def magnitude(self) -> Fraction:
        return max(abs(self.lower), abs(self.upper))


@dataclass(frozen=True)
class ErrorPair:
    nominal: Interval
    error: Fraction = ZERO

    def __post_init__(self):
        _require(isinstance(self.nominal, Interval), "an exact nominal interval is required")
        _fraction(self.error, "absolute error", nonnegative=True)
        _require(self.nominal.magnitude + self.error <= MAX_FINITE, "finite binary64 enclosure required")


def constant(value: Fraction) -> ErrorPair:
    return ErrorPair(Interval(value, value))


def negate(value: ErrorPair) -> ErrorPair:
    """Binary64 sign change is exact under the finite arithmetic model."""
    return ErrorPair(Interval(-value.nominal.upper, -value.nominal.lower), value.error)


def _rounded(nominal: Interval, pre_error: Fraction) -> ErrorPair:
    _require(nominal.magnitude + pre_error <= MAX_FINITE, "operation may overflow")
    error = pre_error + U * (nominal.magnitude + pre_error) + ETA
    return ErrorPair(nominal, error)


def add(a: ErrorPair, b: ErrorPair) -> ErrorPair:
    nominal = Interval(a.nominal.lower + b.nominal.lower, a.nominal.upper + b.nominal.upper)
    return _rounded(nominal, a.error + b.error)


def subtract(a: ErrorPair, b: ErrorPair) -> ErrorPair:
    nominal = Interval(a.nominal.lower - b.nominal.upper, a.nominal.upper - b.nominal.lower)
    return _rounded(nominal, a.error + b.error)


def multiply(a: ErrorPair, b: ErrorPair) -> ErrorPair:
    corners = [x * y for x in (a.nominal.lower, a.nominal.upper)
               for y in (b.nominal.lower, b.nominal.upper)]
    pre = a.nominal.magnitude * b.error + b.nominal.magnitude * a.error + a.error * b.error
    return _rounded(Interval(min(corners), max(corners)), pre)


def divide_positive(a: ErrorPair, b: ErrorPair) -> ErrorPair:
    minimum = b.nominal.lower
    _require(minimum > b.error, "division denominator must remain strictly positive")
    corners = [x / y for x in (a.nominal.lower, a.nominal.upper)
               for y in (b.nominal.lower, b.nominal.upper)]
    pre = (a.error / (minimum - b.error)
           + a.nominal.magnitude * b.error / (minimum * (minimum - b.error)))
    return _rounded(Interval(min(corners), max(corners)), pre)


def _binary64_constant(value: Fraction, name: str) -> None:
    _fraction(value, name)
    _require(abs(value) <= MAX_FINITE, name + " exceeds finite binary64")
    denominator, numerator = value.denominator, abs(value.numerator)
    _require(denominator & (denominator - 1) == 0 and denominator <= 2 ** 1074,
             name + " must be a bound source binary64 value")
    if numerator:
        trailing = (numerator & -numerator).bit_length() - 1
        _require((numerator >> trailing).bit_length() <= 53,
                 name + " must be a bound source binary64 value")


@dataclass(frozen=True)
class LocalRoundoff:
    s: ErrorPair
    x_before_spike: ErrorPair
    x_after_spike: ErrorPair
    reciprocal: ErrorPair
    lx: ErrorPair


def nmda_local_roundoff(parameters: NmdaParameters, *,
                        s_box: Interval, x_box: Interval) -> LocalRoundoff:
    """Evaluate the actual source DAG; nominal values use exact-real source constants.

    Input states themselves have zero local error: they are the exact values
    of the machine inputs in the expanded box. Propagation error is separate.
    """
    for name in ("dt_seconds", "tau_x_seconds", "tau_s_seconds", "alpha_per_second"):
        _binary64_constant(getattr(parameters, name), name)
    one, half = constant(Fraction(1)), constant(Fraction(1, 2))
    h, alpha = constant(parameters.dt_seconds), constant(parameters.alpha_per_second)
    s, x = ErrorPair(s_box), ErrorPair(x_box)
    reciprocal = divide_positive(one, constant(parameters.tau_s_seconds))
    lx = divide_positive(negate(h), constant(parameters.tau_x_seconds))
    ks = multiply(h, subtract(multiply(alpha, multiply(x, subtract(one, s))),
                              multiply(reciprocal, s)))
    kx = multiply(lx, x)
    x_mid_s = add(multiply(half, kx), x)
    # Retain (1+0.5*(-ks))-s; replacing it by 1-(s+0.5*ks) changes the float DAG.
    complement = subtract(add(one, multiply(half, negate(ks))), s)
    s_mid = add(multiply(half, ks), s)
    next_s = add(multiply(h, subtract(multiply(alpha, multiply(x_mid_s, complement)),
                                     multiply(reciprocal, s_mid))), s)
    x_mid_x = add(multiply(half, kx), x)  # Source recomputes this expression.
    next_x = add(multiply(lx, x_mid_x), x)
    spike_x = add(next_x, one)  # E reset injects +1 after RK2.
    return LocalRoundoff(next_s, next_x, spike_x, reciprocal, lx)


def _coefficients(parameters: NmdaParameters, x: Fraction) -> tuple[Fraction, Fraction]:
    h, alpha = parameters.dt_seconds, parameters.alpha_per_second
    a0, d = 1 / parameters.tau_s_seconds, 1 - h / (2 * parameters.tau_x_seconds)
    a, am = a0 + alpha * x, a0 + alpha * d * x
    return 1 - h * am * (1 - h * a / 2), h * alpha * x * (d - h * am / 2)


def derivative_x(parameters: NmdaParameters, x: Fraction, s: Fraction) -> Fraction:
    """Exact derivative of the algebraically extended real s midpoint map."""
    _fraction(x, "x")
    _fraction(s, "s")
    h, alpha = parameters.dt_seconds, parameters.alpha_per_second
    a0, d = 1 / parameters.tau_s_seconds, 1 - h / (2 * parameters.tau_x_seconds)
    cp = -h * alpha * d + h * h * alpha * (2 * alpha * d * x + a0 * (d + 1)) / 2
    bp = h * alpha * d - h * h * alpha * (2 * alpha * d * x + a0) / 2
    return cp * s + bp


@dataclass(frozen=True)
class ExtendedDerivatives:
    s_box: Interval
    x_box: Interval
    a_max: Fraction
    a_mid_min: Fraction
    coefficient_min: Fraction
    coefficient_max: Fraction
    q_s: Fraction
    l_x: Fraction


def expanded_derivative_bounds(parameters: NmdaParameters, *, x_ceiling: Fraction,
                                x_radius: Fraction, s_radius: Fraction) -> ExtendedDerivatives:
    """Bound |partial_s Phi| and |partial_x Phi| on the expanded rectangle."""
    for name, value in (("x_ceiling", x_ceiling), ("x_radius", x_radius), ("s_radius", s_radius)):
        _fraction(value, name, nonnegative=True)
    h, alpha = parameters.dt_seconds, parameters.alpha_per_second
    d, a0 = 1 - h / (2 * parameters.tau_x_seconds), 1 / parameters.tau_s_seconds
    _require(d > 0, "expanded comparison requires d > 0")
    s_box, x_box = Interval(-s_radius, 1 + s_radius), Interval(-x_radius, x_ceiling + x_radius)
    a_max = a0 + alpha * x_box.upper
    a_mid_min = a0 + alpha * d * x_box.lower
    _require(a_mid_min > 0 and h * a_max <= 1,
             "expanded comparison requires a_m_min>0 and h*Amax<=1")
    q_s = 1 - h * a_mid_min * (1 - h * a_max / 2)
    points = [x_box.lower, x_box.upper]
    if alpha > 0:
        stationary = (h * alpha * d - h * h * alpha * a0 * (d + 1) / 2) / (h * h * alpha * alpha * d)
        if x_box.lower <= stationary <= x_box.upper:
            points.append(stationary)
    coefficients = [_coefficients(parameters, x)[0] for x in points]
    _require(min(coefficients) >= 0 and max(coefficients) <= q_s < 1,
             "expanded |partial_s Phi| contraction not established")
    # partial_x Phi is bilinear in x,s, so its absolute maximum is at a corner.
    l_x = max(abs(derivative_x(parameters, x, s))
              for x in (x_box.lower, x_box.upper) for s in (s_box.lower, s_box.upper))
    return ExtendedDerivatives(s_box, x_box, a_max, a_mid_min,
                               min(coefficients), max(coefficients), q_s, l_x)


@dataclass(frozen=True)
class SameEventHistoryBound:
    parameters: NmdaParameters
    x_ceiling: Fraction
    derivatives: ExtendedDerivatives
    local: LocalRoundoff
    x_radius: Fraction
    s_radius: Fraction
    initial_x_upper_bound: Fraction
    initial_s_box: Interval
    initial_x_error_bound: Fraction
    initial_s_error_bound: Fraction
    x_slack: Fraction
    s_slack: Fraction
    native_backend_certified: bool = field(default=False, init=False)
    source_confidence_certified: bool = field(default=False, init=False)


def same_event_history_bound(parameters: NmdaParameters, *, refractory_ticks: int,
                              x_radius: Fraction, s_radius: Fraction,
                              initial_x_upper_bound: Fraction = ZERO,
                              initial_s_box: Interval = Interval(ZERO, ZERO),
                              initial_x_error_bound: Fraction = ZERO,
                              initial_s_error_bound: Fraction = ZERO) -> SameEventHistoryBound:
    """Check uniform error radii for identical valid unit-spike histories.

    Defaults state the source's zero initial x/s and zero initial error.
    Broader supplied initial boxes/errors are premises, not source distributions.
    No power or loop depends on the source horizon.
    """
    _fraction(x_radius, "x_radius", positive=True)
    _fraction(s_radius, "s_radius", positive=True)
    for name, value in (("initial_x_upper_bound", initial_x_upper_bound),
                        ("initial_x_error_bound", initial_x_error_bound),
                        ("initial_s_error_bound", initial_s_error_bound)):
        _fraction(value, name, nonnegative=True)
    _require(0 <= initial_s_box.lower <= initial_s_box.upper <= 1,
             "reference initial s box must lie in [0,1]")
    _require(initial_x_error_bound <= x_radius and initial_s_error_bound <= s_radius,
             "initial errors exceed the supplied radii")
    q_x = rk2_decay_factor(dt_seconds=parameters.dt_seconds, tau_seconds=parameters.tau_x_seconds)
    ceiling = refractory_x_ceiling(q_x, refractory_ticks, initial_x_upper_bound=initial_x_upper_bound)
    nmda_invariance_bounds(parameters, x_upper_bound=ceiling)
    extended = expanded_derivative_bounds(parameters, x_ceiling=ceiling,
                                           x_radius=x_radius, s_radius=s_radius)
    local = nmda_local_roundoff(parameters, s_box=extended.s_box, x_box=extended.x_box)
    epsilon_x = max(local.x_before_spike.error, local.x_after_spike.error)
    epsilon_s = local.s.error
    x_slack = x_radius - (q_x * x_radius + epsilon_x)
    s_slack = s_radius - (extended.q_s * s_radius + extended.l_x * x_radius + epsilon_s)
    _require(x_slack >= 0, "q_x*Dx+epsilon_x<=Dx not established")
    _require(s_slack >= 0, "q_s_ext*Ds+L_x_ext*Dx+epsilon_s<=Ds not established")
    return SameEventHistoryBound(parameters, ceiling, extended, local, x_radius, s_radius,
                                 initial_x_upper_bound, initial_s_box, initial_x_error_bound,
                                 initial_s_error_bound, x_slack, s_slack)


def propagated_error_envelope(bound: SameEventHistoryBound) -> tuple[Fraction, Fraction]:
    """Return constant-cost uniform errors from the invariant recurrences.

    Initial errors and worst per-step forcing are retained. This sharper
    envelope covers the initial/pre states and every post-update state.
    """
    q_x = rk2_decay_factor(dt_seconds=bound.parameters.dt_seconds,
                          tau_seconds=bound.parameters.tau_x_seconds)
    q_s, l_x = bound.derivatives.q_s, bound.derivatives.l_x
    _require(0 <= q_x < 1 and 0 <= q_s < 1, "propagated envelope requires contractive coefficients")
    epsilon_x = max(bound.local.x_before_spike.error, bound.local.x_after_spike.error)
    e_x = max(bound.initial_x_error_bound, epsilon_x / (1 - q_x))
    e_s = max(bound.initial_s_error_bound, (l_x * e_x + bound.local.s.error) / (1 - q_s))
    _require(e_x <= bound.x_radius and e_s <= bound.s_radius,
             "propagated envelope exceeds its checked proof box")
    return e_x, e_s


def history_l1_error_bound(bound: SameEventHistoryBound, horizon_ticks: int) -> tuple[Fraction, Fraction]:
    """Same-clock x/s L1 envelope over H pre or post states, including initial error."""
    _require(type(horizon_ticks) is int and horizon_ticks >= 0, "horizon_ticks must be a nonnegative integer")
    duration = bound.parameters.dt_seconds * horizon_ticks
    e_x, e_s = propagated_error_envelope(bound)
    return duration * e_x, duration * e_s
