"""Compare saved full-unit predictions with an actual native pulse response.

The input episodes and saved artifacts must already have passed their provenance
checks. A single native response is a development comparison, not an independent
success-rate or future-noise-pairing qualification.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .trajectory_scores import energy_score

N_UNITS = 100000
N_FIELDS = 128
N_DRAWS = 16
N_BINS = 100
DT_MS = 0.1
BIN_MS = 5.0
BIN_TICKS = 50
DURATION_TICKS = 60000
SOURCE_REVISION = "74fc021cad2b4bd5c011c390b3f95a302b663483"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _counts(value, shape, name):
    array = np.asarray(value)
    _require(array.shape == shape and array.dtype.kind in "iuf" and np.isfinite(array).all()
             and np.all(array >= 0) and np.array_equal(array, np.floor(array)),
             f"{name}: complete nonnegative integer counts required")
    return array


def _validate_episode(native, label):
    _require(native.dt_ms == DT_MS and native.bin_ms == BIN_MS
             and native.bin_ticks == BIN_TICKS and native.duration_ticks == DURATION_TICKS,
             f"{label}: complete 6-second episode on the 0.1 ms clock and 5 ms bins required")
    ids, fields, field_map = map(np.asarray, (native.unit_ids, native.field_ids, native.unit_field_index))
    _require(ids.shape == (N_UNITS,) and ids.dtype.kind in "iu" and len(np.unique(ids)) == N_UNITS,
             f"{label}: all 100000 unique source unit identities required")
    _require(fields.shape == (N_FIELDS,) and len(set(map(str, fields))) == N_FIELDS,
             f"{label}: all 128 unique source field identities required")
    _require(field_map.shape == ids.shape and field_map.dtype.kind in "iu"
             and np.all(field_map >= 0) and np.all(field_map < N_FIELDS),
             f"{label}: source unit-to-field observation map required")
    denominator = np.bincount(field_map.astype(np.int64), minlength=N_FIELDS)
    _require(np.all(denominator > 0), f"{label}: every source field must contain native units")
    identity = native.source_identity
    _require(isinstance(identity, Mapping) and identity.get("dt_ms") == DT_MS
             and identity.get("gpu_source_revision") == SOURCE_REVISION
             and identity.get("native_poisson_rng_family") == "per_edge_philox4_32_10",
             f"{label}: explicit native clock, source revision and Philox identity required")
    digest = identity.get("graph_manifest_sha256")
    _require(isinstance(digest, str) and len(digest) == 64,
             f"{label}: native graph manifest identity required")
    return ids, fields, field_map, denominator


def compare_candidate(native, source_native, prediction_arrays, reference_arrays,
                      start_tick=10000, stop_tick=15000):
    """Return comparison metrics and actual source-indexed response arrays.

    The interval is half open: spikes at ``start_tick`` count in the first bin;
    spikes at ``stop_tick`` do not count. Prefix identity uses events strictly
    before ``start_tick``. Native library digests may differ after the qualified
    live-control patch; graph, clocks, source law and observation maps must match.
    """
    _require(type(start_tick) is int and type(stop_tick) is int
             and 0 <= start_tick < stop_tick <= DURATION_TICKS
             and start_tick % BIN_TICKS == 0 and stop_tick % BIN_TICKS == 0
             and stop_tick - start_tick == N_BINS * BIN_TICKS,
             "an aligned half-open 100-bin response interval within the native episode is required")
    units, fields, field_map, field_units = _validate_episode(native, "native response")
    source_units, source_fields, source_map, _ = _validate_episode(source_native, "source episode")
    _require(np.array_equal(units, source_units) and np.array_equal(fields, source_fields)
             and np.array_equal(field_map, source_map)
             and np.array_equal(native.minimum_interval_ticks, source_native.minimum_interval_ticks),
             "native response unit identities, field order, observation map or unit availability law changed")
    _require(native.source_identity["graph_manifest_sha256"] == source_native.source_identity["graph_manifest_sha256"],
             "native response graph identity differs from the source episode")
    native_prefix = np.searchsorted(native.event_ticks, start_tick, side="left")
    source_prefix = np.searchsorted(source_native.event_ticks, start_tick, side="left")
    _require(np.array_equal(native.event_ticks[:native_prefix], source_native.event_ticks[:source_prefix])
             and np.array_equal(native.event_units[:native_prefix], source_native.event_units[:source_prefix]),
             "native response events strictly before the decision differ from the source prefix")
    _require(isinstance(prediction_arrays, Mapping) and isinstance(reference_arrays, Mapping),
             "parsed prediction and reference array mappings required")
    prediction, reference = prediction_arrays, reference_arrays
    _require(np.array_equal(prediction.get("unit_ids"), units)
             and np.array_equal(prediction.get("field_ids"), fields),
             "saved prediction source unit identity or field order changed")
    predicted_counts = _counts(prediction.get("field_counts"), (N_DRAWS, N_BINS, N_FIELDS),
                               "saved predicted field counts")
    predicted_totals = _counts(prediction.get("unit_total_counts"), (N_DRAWS, N_UNITS),
                               "saved predicted unit totals")
    predicted_hz = np.asarray(prediction.get("field_rate_hz"))
    _require(predicted_hz.shape == predicted_counts.shape and predicted_hz.dtype.kind in "fiu"
             and np.isfinite(predicted_hz).all() and np.all(predicted_hz >= 0)
             and np.array_equal(predicted_hz, predicted_counts / (field_units * (BIN_MS / 1000.0))),
             "saved prediction Hz differs from its full field counts, native denominators or bin duration")
    predicted_field_totals = np.stack([
        predicted_totals[:, field_map == field].sum(-1, dtype=np.int64) for field in range(N_FIELDS)], axis=-1)
    _require(np.array_equal(predicted_field_totals, predicted_counts.sum(axis=1, dtype=np.int64)),
             "saved predicted unit totals disagree with saved field trajectory totals")
    _require(np.array_equal(reference.get("field_ids"), fields),
             "reference source field identity or order changed")
    reference_hz, scale = np.asarray(reference.get("field_rate_hz")), np.asarray(reference.get("scale_hz"))
    blocks = np.asarray(reference.get("block_ids"))
    _require(reference_hz.shape == (14, N_BINS, N_FIELDS) and reference_hz.dtype.kind in "fiu"
             and np.isfinite(reference_hz).all() and np.all(reference_hz >= 0)
             and scale.shape == (N_FIELDS,) and scale.dtype.kind in "fiu"
             and np.isfinite(scale).all() and np.all(scale > 0)
             and blocks.shape == (14,) and tuple(map(str, blocks)) == tuple(str(block) for block in range(14)),
             "14 fitting-only reference blocks, matching finite Hz trajectories and positive frozen field scales required")
    observed, _ = native.window(start_tick, stop_tick)
    observed = _counts(observed, (N_BINS, N_UNITS), "actual native response counts")
    actual_totals = observed.sum(axis=0, dtype=np.int64)
    actual_field_counts = np.stack([
        observed[:, field_map == field].sum(-1, dtype=np.int64) for field in range(N_FIELDS)], axis=-1)
    actual_hz = actual_field_counts / (field_units * (BIN_MS / 1000.0))
    actual_ready = np.asarray(native.ready_at(stop_tick))
    _require(actual_ready.shape == (N_UNITS,) and actual_ready.dtype.kind in "iu"
             and np.all(actual_ready >= 0), "actual terminal native availability for every source unit required")
    discrepancy = predicted_totals.mean(axis=0, dtype=np.float64) - actual_totals
    standardized_actual = actual_hz / scale
    distances = np.sqrt(np.mean((reference_hz / scale - standardized_actual) ** 2, axis=(1, 2)))
    report = dict(
        status="DEVELOPMENT_NATIVE_PULSE_RESPONSE_COMPARED", response_start_tick=start_tick,
        response_stop_tick=stop_tick, response_interval="[start_tick, stop_tick)",
        dt_ms=DT_MS, bin_ms=BIN_MS, source_unit_count=N_UNITS, field_count=N_FIELDS,
        horizon_bins=N_BINS, prediction_draws=N_DRAWS, reference_block_ids=list(map(str, blocks)),
        prefix_events_verified=int(native_prefix), causal_source_prefix_verified=True,
        prediction_native_energy_score=energy_score(predicted_hz / scale, standardized_actual),
        unit_total_ensemble_mean_rmse_spikes=float(np.sqrt(np.mean(discrepancy ** 2))),
        unit_total_ensemble_mean_mae_spikes=float(np.mean(np.abs(discrepancy))),
        unit_total_ensemble_mean_bias_spikes=float(np.mean(discrepancy)),
        native_reference_standardized_rmse_mean=float(distances.mean()),
        native_reference_standardized_rmse_by_block=distances.tolist(),
        reference_distance_definition="RMSE of field Hz divided by frozen per-field scale over all response bins and fields",
        actual_total_spikes=int(actual_totals.sum()), source_graph_manifest_sha256=native.source_identity["graph_manifest_sha256"],
        target_success_tolerance_qualified=False, success_tolerance_qualified=False,
        native_future_noise_pairing_verified=False, independent_intervention_qualified=False,
        inverse_intervention_qualified=False, scientific_claim_qualified=False,
    )
    actual_arrays = dict(field_counts=actual_field_counts, field_rate_hz=actual_hz,
                         unit_total_counts=actual_totals, terminal_ready_in_ticks=actual_ready.copy(),
                         unit_ids=units.copy(), field_ids=fields.copy())
    return report, actual_arrays
