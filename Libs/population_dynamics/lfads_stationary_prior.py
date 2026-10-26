"""Fit the existing LFADS zero-mean, four-channel stationary AR(1) family.

This module fits frozen complete paths; the caller establishes their provenance
and ancestral sampling semantics. It neither fits a mean/cross-channel coupling
nor validates native TensorFlow execution. Boundary optima are reported, never
replaced with clipped interior parameters.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


CHANNELS = 4
LOG_TWO_PI = math.log(2.0 * math.pi)
_MOMENT_NAMES = ("A", "B", "C", "difference_squared", "initial_squared", "terminal_squared")


class StationaryARFitError(ValueError):
    """An inadmissible or numerically unresolved fit, with all channel failures."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        failures = "; ".join(
            f"channel={row['channel']}:{row['reason']}" for row in diagnostics["failures"]
        )
        super().__init__("STATIONARY_AR_FIT_FAILED: " + failures)


class StationaryARMoments:
    """Stream complete float32 [sequence, time, 4] paths in fixed sequence order.

    Pass sequence_length when the native full length is known. Without it, the
    first update establishes the common length; this does not establish that
    the caller supplied complete paths. Updates never join time-axis fragments.
    Only one sequence is converted to float64 at a time.
    """

    def __init__(self, sequence_length: int | None = None):
        if sequence_length is not None and (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, (int, np.integer))
            or sequence_length < 2
        ):
            raise ValueError("FULL_SEQUENCE_LENGTH_MUST_BE_AN_INTEGER_AT_LEAST_TWO")
        self.sequence_length = None if sequence_length is None else int(sequence_length)
        self.sequence_count = 0
        self._sums = np.zeros((len(_MOMENT_NAMES), CHANNELS), dtype=np.float64)
        self._compensation = np.zeros_like(self._sums)

    def update(self, paths: np.ndarray) -> "StationaryARMoments":
        if not isinstance(paths, np.ndarray) or paths.dtype != np.dtype("float32"):
            raise ValueError("NATIVE_FLOAT32_PATH_ARRAY_REQUIRED")
        if paths.ndim != 3 or paths.shape[0] < 1 or paths.shape[1] < 2 or paths.shape[2] != CHANNELS:
            raise ValueError("COMPLETE_SEQUENCE_TIME_FOUR_CHANNEL_PATHS_REQUIRED")
        if self.sequence_length is not None and paths.shape[1] != self.sequence_length:
            raise ValueError("COMPLETE_PATH_LENGTH_CHANGED")
        if not np.isfinite(paths).all():
            raise ValueError("FINITE_NATIVE_CONTROLLER_PATHS_REQUIRED")
        # Validation precedes mutation, including validation of later sequences.
        self.sequence_length = paths.shape[1]
        for path in paths:
            z = path.astype(np.float64)
            squared = np.square(z)
            values = np.stack((
                squared.sum(axis=0),
                (z[1:] * z[:-1]).sum(axis=0),
                squared[1:-1].sum(axis=0),
                np.square(z[1:] - z[:-1]).sum(axis=0),
                squared[0],
                squared[-1],
            ))
            increment = values - self._compensation
            updated = self._sums + increment
            self._compensation = (updated - self._sums) - increment
            self._sums = updated
            self.sequence_count += 1
        return self

    def _checked_values(self) -> tuple[int, int, np.ndarray]:
        if not self.sequence_count or self.sequence_length is None:
            raise ValueError("NONEMPTY_COMPLETE_PATH_MOMENTS_REQUIRED")
        if not np.isfinite(self._sums).all():
            raise ValueError("NONFINITE_ACCUMULATED_AR_MOMENTS")
        return self.sequence_count, self.sequence_length, self._sums.copy()

    def as_dict(self) -> dict[str, Any]:
        sequences, length, values = self._checked_values()
        return dict(
            sequence_count=sequences,
            sequence_length=length,
            channels=CHANNELS,
            scalar_values=sequences * length * CHANNELS,
            accumulation_dtype="float64",
            source_dtype="float32",
            definitions=dict(A="sum z_t^2 including every first sample",
                B="sum z_t*z_(t-1) over transitions",
                C="sum z_t^2 over 1 <= t < T-1",
                difference_squared="sum (z_t-z_(t-1))^2",
                initial_squared="sum z_0^2", terminal_squared="sum z_(T-1)^2"),
            **{name: value.tolist() for name, value in zip(_MOMENT_NAMES, values)},
        )


def _logs(value: np.ndarray, name: str) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.shape != (1, CHANNELS)
            or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(value).all()):
        raise ValueError("FINITE_FLOAT_LOG_PARAMETER_SHAPE_1_BY_4_REQUIRED:" + name)
    return value.astype(np.float64)[0]


def _quadratic(values: np.ndarray, phi: np.ndarray) -> np.ndarray:
    # Q=A-2*B*phi+C*phi^2. The equivalent expression below contains only
    # nonnegative terms for 0 <= phi < 1 and avoids cancellation near one.
    a, _, c, differences = values[:4]
    return (1.0 - phi) * (a - phi * c) + phi * differences


def stationary_ar_nll(
    moments: StationaryARMoments, logatau: np.ndarray, logevars: np.ndarray
) -> dict[str, Any]:
    """Full negative log density, evaluated in float64 from the specified logs.

    Return keys are total, per_channel, phi, innovation_variance,
    stationary_variance, sequence_count and sequence_length. This is the exact
    mathematical source-family density including normalizing constants and the
    stationary first sample; it is not a native float32 TensorFlow comparison.
    """
    if not isinstance(moments, StationaryARMoments):
        raise ValueError("STATIONARY_AR_MOMENT_ACCUMULATOR_REQUIRED")
    sequences, length, values = moments._checked_values()
    logtau, logvar = _logs(logatau, "logatau"), _logs(logevars, "logevars")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        phi = np.exp(-np.exp(-logtau))
        variance = np.exp(logvar)
        tau = np.exp(logtau)
        log_stationarity = np.log1p(-phi) + np.log1p(phi)
        stationary_variance = np.exp(logvar - log_stationarity)
    if (not all(np.isfinite(v).all() for v in (phi, variance, tau, stationary_variance))
            or (phi <= 0).any() or (phi >= 1).any() or (variance <= 0).any()
            or (tau <= 0).any() or (stationary_variance <= 0).any()):
        raise ValueError("FINITE_INTERIOR_STATIONARY_AR_PARAMETERS_REQUIRED")
    q = _quadratic(values, phi)
    with np.errstate(over="ignore", invalid="ignore"):
        per_channel = 0.5 * (
            sequences * length * (LOG_TWO_PI + logvar)
            - sequences * log_stationarity + q / variance
        )
    if not np.isfinite(per_channel).all():
        raise ValueError("NONFINITE_STATIONARY_AR_NEGATIVE_LOG_DENSITY")
    return dict(total=math.fsum(per_channel.tolist()), per_channel=per_channel.tolist(),
        phi=phi.tolist(), innovation_variance=variance.tolist(),
        stationary_variance=stationary_variance.tolist(),
        sequence_count=sequences, sequence_length=length)


def _profile_channel(sequences: int, length: int, values: np.ndarray, channel: int) -> dict[str, Any]:
    a, b, c, differences = (float(x) for x in values[:4, channel])
    base = dict(channel=channel, A=a, B=b, C=c, difference_squared=differences)
    if a == 0.0:
        return dict(base, reason="ZERO_VARIANCE_UNBOUNDED")
    if differences == 0.0:
        return dict(base, reason="PHI_ONE_VARIANCE_ZERO_UNBOUNDED")
    if b <= 0.0:
        return dict(base, reason="NONINTERIOR_PHI_ZERO_OPTIMUM")
    count = sequences * length
    # Normalize both by A and T to avoid scaling the cubic by path amplitude
    # or path length. C=0 at T=2 naturally reduces this to a linear equation.
    bn, cn, dn = b / a, c / a, differences / a
    coefficients = np.array(
        [cn * (1.0 / length - 1.0), bn * (1.0 - 2.0 / length),
         cn + 1.0 / length, -bn], dtype=np.float64)
    roots = np.roots(np.trim_zeros(coefficients, "f"))
    root_records = [dict(real=float(r.real), imaginary=float(r.imag)) for r in roots]

    def profile(phi: float) -> tuple[float, float]:
        q_normalized = (1.0 - phi) * (1.0 - phi * cn) + phi * dn
        log_variance = math.log(a) + math.log(q_normalized) - math.log(count)
        nll = 0.5 * (count * (LOG_TWO_PI + log_variance + 1.0)
            - sequences * (math.log1p(-phi) + math.log1p(phi)))
        return nll, log_variance

    boundary_nll, _ = profile(0.0)
    candidates = []
    numerical_tolerance = 2048.0 * np.finfo(np.float64).eps
    for root in roots:
        if abs(root.imag) > numerical_tolerance * max(1.0, abs(root.real)):
            continue
        phi = float(root.real)
        if not 0.0 < phi < 1.0:
            continue
        # Refine only when a Newton step stays interior and reduces the cubic
        # residual. There is no parameter clipping or change of family.
        for _ in range(4):
            residual = float(np.polyval(coefficients, phi))
            slope = float(np.polyval(np.polyder(coefficients), phi))
            if slope == 0.0:
                break
            refined = phi - residual / slope
            if (not 0.0 < refined < 1.0
                    or abs(np.polyval(coefficients, refined)) >= abs(residual)):
                break
            phi = refined
        scale = float(np.polyval(np.abs(coefficients), abs(phi)))
        scaled_residual = abs(float(np.polyval(coefficients, phi))) / scale
        if scaled_residual > numerical_tolerance:
            continue
        nll, log_variance = profile(phi)
        candidates.append(dict(phi=phi, nll=nll, log_innovation_variance=log_variance,
            scaled_stationarity_residual=scaled_residual))
    common = dict(base, derivative_coefficients=coefficients.tolist(),
        polynomial_roots=root_records, admissible_stationary_points=candidates,
        phi_zero_limit_nll=boundary_nll, phi_one_limit_nll="positive_infinity")
    if not candidates:
        return dict(common, reason="NO_RESOLVED_INTERIOR_OPTIMUM")
    best = min(candidates, key=lambda row: row["nll"])
    if best["nll"] >= boundary_nll:
        return dict(common, reason="INTERIOR_IMPROVEMENT_OVER_BOUNDARY_UNRESOLVED")
    phi = best["phi"]
    logvar = best["log_innovation_variance"]
    return dict(common, phi=phi, logatau=-math.log(-math.log(phi)), logevars=logvar,
        innovation_variance=math.exp(logvar),
        stationary_variance=math.exp(logvar - math.log1p(-phi) - math.log1p(phi)),
        profile_nll=best["nll"], scaled_stationarity_residual=best["scaled_stationarity_residual"])


def _float32_source_range(logatau: np.ndarray, logevars: np.ndarray) -> dict[str, Any]:
    """Check NumPy float32 evaluations of source expressions, not native TF."""
    one, half = np.float32(1.0), np.float32(0.5)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        inverse_tau = np.exp(-logatau)
        phi = np.exp(-inverse_tau)
        innovation_variance = np.exp(logevars)
        innovation_std = np.exp(half * logevars)
        tau = np.exp(logatau)
        logpvar = logevars - np.log(one - phi) - np.log(one + phi)
        stationary_variance = np.exp(logpvar)
        stationary_std = np.exp(half * logpvar)
    derived = dict(inverse_tau=inverse_tau, phi=phi, innovation_variance=innovation_variance,
        innovation_std=innovation_std, autocorrelation_tau_bins=tau,
        stationary_variance=stationary_variance, stationary_std=stationary_std)
    failures = []
    for channel in range(CHANNELS):
        if (not all(np.isfinite(v[0, channel]) and v[0, channel] > 0 for v in derived.values())
                or phi[0, channel] >= 1 or not np.isfinite(logpvar[0, channel])):
            failures.append(dict(channel=channel, reason="CANDIDATE_NOT_FLOAT32_SOURCE_ADMISSIBLE"))
    if failures:
        raise StationaryARFitError(dict(failures=failures))
    return dict(expression_dtype="float32", native_tensorflow_validated=False,
        **{key: value.astype(np.float64)[0].tolist() for key, value in derived.items()},
        stationary_log_variance=logpvar.astype(np.float64)[0].tolist())


def fit_stationary_ar_prior(
    moments: StationaryARMoments,
    initial_logatau: np.ndarray | None = None,
    initial_logevars: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return float32 [1,4] logatau/logevars and JSON-compatible diagnostics.

    Each independent channel is fit by its complete stationary likelihood,
    profiling innovation variance Q/(M*T). All real interior roots of
      C*(1-T)*phi^3 + B*(T-2)*phi^2 + (T*C+A)*phi - T*B
    are compared with the phi=0 limit and the phi->1 limit. No admissible
    finite maximizer means an explicit error, not a boundary approximation.
    initial_* are optional together, solely for before/after density reporting.
    """
    if not isinstance(moments, StationaryARMoments):
        raise ValueError("STATIONARY_AR_MOMENT_ACCUMULATOR_REQUIRED")
    sequences, length, values = moments._checked_values()
    if (initial_logatau is None) != (initial_logevars is None):
        raise ValueError("BOTH_INITIAL_LOG_PARAMETERS_REQUIRED")
    before = None if initial_logatau is None else stationary_ar_nll(
        moments, initial_logatau, initial_logevars)
    channels = [_profile_channel(sequences, length, values, channel) for channel in range(CHANNELS)]
    failures = [row for row in channels if "reason" in row]
    if failures:
        raise StationaryARFitError(dict(failures=failures, channels=channels, moments=moments.as_dict()))
    logatau = np.array([[row["logatau"] for row in channels]], dtype=np.float32)
    logevars = np.array([[row["logevars"] for row in channels]], dtype=np.float32)
    source_range = _float32_source_range(logatau, logevars)
    after = stationary_ar_nll(moments, logatau, logevars)
    continuous = [row["profile_nll"] for row in channels]
    diagnostics = dict(status="FITTED_STATIONARY_AR1_FLOAT32_PARAMETER_CANDIDATE",
        scientific_claim_qualified=False, native_tensorflow_validated=False,
        family="zero-mean independent four-channel stationary AR(1)",
        phi_expression="exp(-exp(-logatau))",
        innovation_variance_expression="exp(logevars)",
        first_sample_variance="exp(logevars)/(1-phi^2)",
        mean_fitted=False, cross_channel_parameters_fitted=False,
        moments=moments.as_dict(), channels=channels,
        continuous_optimum_nll=dict(total=math.fsum(continuous), per_channel=continuous),
        initial_parameter_nll=before, float32_candidate_nll=after,
        float32_source_range=source_range)
    return dict(logatau=logatau, logevars=logevars, diagnostics=diagnostics)

