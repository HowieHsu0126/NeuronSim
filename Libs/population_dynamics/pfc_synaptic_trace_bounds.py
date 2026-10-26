"""Exact-real RK2 synaptic trace bounds, with explicit same-clock inputs.

All timed quantities are SI seconds. Float source constants must be converted
explicitly to Fraction by the caller. These are sufficient premises of a trace
comparison component, not campaign gates, native rounding certification or
source-history confidence. No input sequence is sorted, clipped or resampled.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

ZERO = Fraction(0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("SYNAPTIC_TRACE_PROOF_CONDITION: " + message)


def _fraction(value: Fraction, name: str, *, positive: bool = False) -> None:
    _require(type(value) is Fraction, name + " must be an exact Fraction; convert source binary64 explicitly")
    _require(value > 0 if positive else value >= 0, name + " must be positive" if positive else name + " must be nonnegative")


def _integer(value: int, name: str, *, positive: bool = False) -> None:
    _require(type(value) is int and (value > 0 if positive else value >= 0),
             name + " must be a positive integer" if positive else name + " must be a nonnegative integer")


def _decay(q: Fraction) -> None:
    _fraction(q, "decay_factor")
    _require(q < 1, "finite trace bounds require 0 <= decay_factor < 1")


def validate_spike_ticks(spikes: Sequence[int], *, refractory_ticks: int,
                         horizon_ticks: int | None = None) -> tuple[int, ...]:
    """Validate full per-neuron events; source tick t injects after update t."""
    _integer(refractory_ticks, "refractory_ticks", positive=True)
    if horizon_ticks is not None:
        _integer(horizon_ticks, "horizon_ticks")
    result = tuple(spikes)
    for index, tick in enumerate(result):
        _integer(tick, "spike_tick")
        if horizon_ticks is not None:
            _require(tick < horizon_ticks, "spike_tick lies outside the declared history")
        if index:
            _require(tick - result[index - 1] >= refractory_ticks,
                     "spike ticks must be ordered and satisfy refractory spacing")
    return result


def rk2_decay_factor(*, dt_seconds: Fraction, tau_seconds: Fraction) -> Fraction:
    """Exact-real midpoint decay; no certification of native float operations."""
    _fraction(dt_seconds, "dt_seconds", positive=True)
    _fraction(tau_seconds, "tau_seconds", positive=True)
    z = dt_seconds / tau_seconds
    _require(z < 2, "RK2 trace decay requires 0 < dt_seconds/tau_seconds < 2")
    return 1 - z + z * z / 2


def refractory_x_ceiling(q_x: Fraction, refractory_ticks: int, *,
                         initial_x_upper_bound: Fraction = ZERO) -> Fraction:
    """Bound x_t=q_x*x_(t-1)+unit_spike from an explicit initial ceiling.

    Refractory spacing bounds all post-start pulse contributions by a geometric
    series. The independently bounded initial contribution adds at most its
    initial ceiling; this does not assume unobserved prehistory is zero.
    """
    _decay(q_x)
    _integer(refractory_ticks, "refractory_ticks", positive=True)
    _fraction(initial_x_upper_bound, "initial_x_upper_bound")
    return initial_x_upper_bound + 1 / (1 - q_x ** refractory_ticks)


@dataclass(frozen=True)
class NmdaParameters:
    dt_seconds: Fraction
    tau_x_seconds: Fraction
    tau_s_seconds: Fraction
    alpha_per_second: Fraction

    def __post_init__(self):
        for name in ("dt_seconds", "tau_x_seconds", "tau_s_seconds"):
            _fraction(getattr(self, name), name, positive=True)
        _fraction(self.alpha_per_second, "alpha_per_second")


@dataclass(frozen=True)
class NmdaBounds:
    q_x: Fraction
    d: Fraction
    a0: Fraction
    a_max: Fraction
    q_s: Fraction
    l_x: Fraction
    x_upper_bound: Fraction


def nmda_invariance_bounds(parameters: NmdaParameters, *,
                           x_upper_bound: Fraction) -> NmdaBounds:
    """Check sufficient premises for s in [0,1] and its Lipschitz recurrence.

    For F(x,s)=C(x)s+B(x), the premises give C in [0,q_s] and
    dF/dx in [0,l_x]. They also give F(x,0)>=0 and F(x,1)<=1.
    """
    _fraction(x_upper_bound, "x_upper_bound")
    h, alpha = parameters.dt_seconds, parameters.alpha_per_second
    q_x = rk2_decay_factor(dt_seconds=h, tau_seconds=parameters.tau_x_seconds)
    d, a0 = 1 - h / (2 * parameters.tau_x_seconds), 1 / parameters.tau_s_seconds
    a_max = a0 + alpha * x_upper_bound
    _require(d > 0, "NMDA comparison requires d > 0")
    _require(h * a_max <= 1, "NMDA comparison requires h*Amax <= 1")
    _require(d - h * a0 / 2 - h * alpha * d * x_upper_bound >= 0,
             "NMDA comparison requires d-h*a0/2-h*alpha*d*Xmax >= 0")
    q_s, l_x = 1 - h * a0 * (1 - h * a_max / 2), h * alpha * d
    _require(0 <= q_s < 1 and l_x >= 0, "NMDA contraction coefficients are outside their proof range")
    return NmdaBounds(q_x, d, a0, a_max, q_s, l_x, x_upper_bound)


def nmda_rk2_step(s: Fraction, x: Fraction, parameters: NmdaParameters) -> tuple[Fraction, Fraction]:
    """Source-equivalent exact-real midpoint algebra, before a spike injection."""
    _fraction(s, "s")
    _require(s <= 1, "NMDA s must lie in [0,1]")
    _fraction(x, "x")
    bounds = nmda_invariance_bounds(parameters, x_upper_bound=x)
    h, alpha = parameters.dt_seconds, parameters.alpha_per_second
    a, a_mid = bounds.a0 + alpha * x, bounds.a0 + alpha * bounds.d * x
    coefficient = 1 - h * a_mid * (1 - h * a / 2)
    forcing = h * alpha * x * (bounds.d - h * a_mid / 2)
    return coefficient * s + forcing, bounds.q_x * x


def pulse_translation_l1_exact(*, dt_seconds: Fraction, decay_factor: Fraction,
                                shift_ticks: int) -> Fraction:
    """Exact infinite-history pulse distance; use only for short algebra checks.

    The scalable history path below does not evaluate this large exponent.
    """
    _fraction(dt_seconds, "dt_seconds", positive=True)
    _decay(decay_factor)
    _require(type(shift_ticks) is int, "shift_ticks must be an integer, not bool")
    return 2 * dt_seconds * (1 - decay_factor ** abs(shift_ticks)) / (1 - decay_factor)


def pulse_translation_l1_bound(*, dt_seconds: Fraction, decay_factor: Fraction,
                                shift_ticks: int) -> Fraction:
    """Exact rational upper bound, with no work proportional to time shift."""
    _fraction(dt_seconds, "dt_seconds", positive=True)
    _decay(decay_factor)
    _require(type(shift_ticks) is int, "shift_ticks must be an integer, not bool")
    return min(2 * dt_seconds * abs(shift_ticks), 2 * dt_seconds / (1 - decay_factor))


def pulse_train_l1_bound(left_spikes: Sequence[int], right_spikes: Sequence[int], *,
                         matched_spikes: Sequence[tuple[int, int]],
                         dt_seconds: Fraction, decay_factor: Fraction,
                         refractory_ticks: int, initial_left: Fraction,
                         initial_right: Fraction, horizon_ticks: int | None = None) -> Fraction:
    """Bound same-clock linear traces; unmatched unit spikes pay their full area.

    Initial traces use x_t=q**t*x_0; an event at source tick t first contributes
    to x_(t+1). The infinite-history bound also bounds every finite restriction.
    Matching is explicit and validated against both complete event sequences.
    """
    _fraction(dt_seconds, "dt_seconds", positive=True)
    _decay(decay_factor)
    _fraction(initial_left, "initial_left")
    _fraction(initial_right, "initial_right")
    left = validate_spike_ticks(left_spikes, refractory_ticks=refractory_ticks, horizon_ticks=horizon_ticks)
    right = validate_spike_ticks(right_spikes, refractory_ticks=refractory_ticks, horizon_ticks=horizon_ticks)
    left_set, right_set, used_left, used_right = set(left), set(right), set(), set()
    bound = dt_seconds * abs(initial_left - initial_right) / (1 - decay_factor)
    for pair in matched_spikes:
        _require(isinstance(pair, (tuple, list)) and len(pair) == 2, "each matched spike requires two source ticks")
        a, b = pair
        _integer(a, "matched_left_tick")
        _integer(b, "matched_right_tick")
        _require(a in left_set and b in right_set and a not in used_left and b not in used_right,
                 "matched spikes must exist and may not reuse events")
        used_left.add(a)
        used_right.add(b)
        bound += pulse_translation_l1_bound(dt_seconds=dt_seconds, decay_factor=decay_factor,
                                            shift_ticks=b - a)
    unmatched = len(left) + len(right) - len(used_left) - len(used_right)
    return bound + unmatched * dt_seconds / (1 - decay_factor)


@dataclass(frozen=True)
class SynapticHistory:
    source_neuron_id: int
    dt_seconds: Fraction
    spike_ticks: Sequence[int]
    initial_x: Fraction
    initial_s: Fraction

    def __post_init__(self):
        _integer(self.source_neuron_id, "source_neuron_id")
        _fraction(self.dt_seconds, "dt_seconds", positive=True)
        _fraction(self.initial_x, "initial_x")
        _fraction(self.initial_s, "initial_s")
        _require(self.initial_s <= 1, "initial_s must lie in [0,1]")
        object.__setattr__(self, "spike_ticks", validate_spike_ticks(self.spike_ticks, refractory_ticks=1))


def nmda_history_l1_bound(left: SynapticHistory, right: SynapticHistory, *,
                          matched_spikes: Sequence[tuple[int, int]],
                          parameters: NmdaParameters, refractory_ticks: int,
                          initial_x_upper_bound: Fraction, horizon_ticks: int,
                          state_interval: Literal["post", "pre"] = "post") -> Fraction:
    """Bound an explicit same-clock interval of NMDA state differences.

    "post" returns h*sum_(t=1..H)|delta_s[t]|, preserving the existing interval.
    "pre" returns h*sum_(t=0..H-1)|delta_s[t]| and covers the OLD E NMDA states
    projected at the start of each source tick. Both use pre-update x inputs.
    Geometric sums are bounded by min(H,1/(1-q_s)); q_s**H is never computed.
    The result assumes only the checked scalar trace premises and event records.
    It does not bound changes in the emitted spike sequence.
    """
    _require(state_interval in ("post", "pre"), 'state_interval must be "post" or "pre"')
    _integer(horizon_ticks, "horizon_ticks")
    _fraction(initial_x_upper_bound, "initial_x_upper_bound")
    _require(left.source_neuron_id == right.source_neuron_id, "comparison requires the same source neuron identity")
    _require(left.dt_seconds == right.dt_seconds == parameters.dt_seconds,
             "comparison requires the same clock; no resampling or clock coercion")
    _require(max(left.initial_x, right.initial_x) <= initial_x_upper_bound,
             "initial x exceeds its explicitly supplied upper bound")
    q_x = rk2_decay_factor(dt_seconds=parameters.dt_seconds, tau_seconds=parameters.tau_x_seconds)
    ceiling = refractory_x_ceiling(q_x, refractory_ticks, initial_x_upper_bound=initial_x_upper_bound)
    bounds = nmda_invariance_bounds(parameters, x_upper_bound=ceiling)
    linear = pulse_train_l1_bound(left.spike_ticks, right.spike_ticks, matched_spikes=matched_spikes,
        dt_seconds=parameters.dt_seconds, decay_factor=q_x, refractory_ticks=refractory_ticks,
        initial_left=left.initial_x, initial_right=right.initial_x, horizon_ticks=horizon_ticks)
    h = parameters.dt_seconds
    x_l1 = min(linear, h * horizon_ticks * ceiling)
    gain = min(Fraction(horizon_ticks), 1 / (1 - bounds.q_s))
    initial_factor = bounds.q_s if state_interval == "post" else Fraction(1)
    result = gain * (h * initial_factor * abs(left.initial_s - right.initial_s) + bounds.l_x * x_l1)
    return min(h * horizon_ticks, result)


def absolute_column_sums(neuron_ids: Sequence[int],
                          edges: Iterable[tuple[int, int, Fraction]]) -> dict[int, Fraction]:
    """Stream signed graph edges; store O(neurons) absolute column sums.

    Parallel edges are charged separately by triangle inequality. Source and
    target are source-native neuron IDs; no dense matrix or identity remap occurs.
    """
    ids = tuple(neuron_ids)
    _require(bool(ids) and len(ids) == len(set(ids)), "declare unique nonempty source-neuron identities")
    for neuron in ids:
        _integer(neuron, "source_neuron_id")
    result = dict.fromkeys(ids, ZERO)
    for source, target, weight in edges:
        _integer(source, "edge_source")
        _integer(target, "edge_target")
        _require(source in result and target in result, "edge references an undeclared source-neuron identity")
        _require(type(weight) is Fraction, "graph weights must be exact Fractions in the operator's units")
        result[source] += abs(weight)
    return result


def weighted_graph_input_l1_bound(source_l1_bounds: Mapping[int, Fraction],
                                  column_sums: Mapping[int, Fraction]) -> Fraction:
    """Bound summed target-input L1 by absolute columns and per-source bounds.

    All source bounds must apply to the same physical interval and time units.
    Columns must bound the actual operator; this function does not certify the
    graph, resample clocks, or infer missing source identities.
    """
    _require(bool(source_l1_bounds) and set(source_l1_bounds) == set(column_sums),
             "graph columns and bounds must retain the same complete source identities")
    total = ZERO
    for neuron, value in source_l1_bounds.items():
        _integer(neuron, "source_neuron_id")
        _fraction(value, "source_l1_bound")
        _fraction(column_sums[neuron], "absolute_column_sum")
        total += column_sums[neuron] * value
    return total
