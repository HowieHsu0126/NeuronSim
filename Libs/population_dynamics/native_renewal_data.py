"""Source-bound native episodes with causal renewal exposure and unit identity.

The dead-time rule is specific to the inspected native iaf_psc_exp revision:
a spike sets round(t_ref / dt), followed by that many decrement-only updates.
The next eligible emission is therefore round(t_ref / dt) + 1 ticks later.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .native_count_windows import NativeEpisodeCounts
from .native_trajectory_cohort import load_dataset_index
from .provenance import sha256_file
from .refractory_observation import RenewalWindow, renewal_window

NATIVE_SOURCE_REVISION = "74fc021cad2b4bd5c011c390b3f95a302b663483"
MINIMUM_INTERVAL_POLICY = "native_iaf_psc_exp_round_t_ref_over_dt_plus_one"


class NativeRenewalEpisode(NativeEpisodeCounts):
    """Keep sparse events; materialize counts/exposures only on request."""

    def __init__(self, episode: Path, field_view: Path, *, expected_source=None,
                 expected_unit_ids=None, expected_unit_field_index=None):
        episode, field_view = Path(episode), Path(field_view)
        report = json.loads((episode / "report.json").read_text())
        if report["gpu_source_revision"] != NATIVE_SOURCE_REVISION:
            raise ValueError("minimum-interval policy requires the inspected native source revision")
        self.source_identity = dict(
            gpu_source_revision=report["gpu_source_revision"],
            native_library_sha256=report["native_library_sha256"],
            native_poisson_rng_family=report.get("native_poisson_rng_family", "not_reported_by_native_library"),
            graph_manifest_sha256=report["graph_manifest_sha256"],
            dt_ms=report["dt_ms"],
        )
        if (any(not isinstance(self.source_identity[key], str) or not self.source_identity[key]
                for key in ("gpu_source_revision", "native_poisson_rng_family"))
                or any(len(self.source_identity[key]) != 64 for key in
                       ("native_library_sha256", "graph_manifest_sha256"))):
            raise ValueError("explicit native library and Poisson RNG identity required")
        if expected_source is not None and self.source_identity != expected_source:
            raise ValueError("native source identity changed")
        super().__init__(episode, field_view)
        if (not np.isfinite(self.dt_ms) or self.dt_ms <= 0 or not np.isfinite(self.bin_ms)
                or self.duration_ticks <= 0 or self.duration_ticks % self.bin_ticks):
            raise ValueError("finite positive aligned native clock required")
        if (self.unit_ids.ndim != 1 or self.unit_ids.dtype.kind not in "iu"
                or not len(self.unit_ids) or self.unit_field_index.shape != self.unit_ids.shape
                or self.unit_field_index.dtype.kind not in "iu" or self.field_ids.ndim != 1
                or len(set(map(str, self.field_ids))) != len(self.field_ids)
                or np.any(self.unit_field_index < 0)
                or np.any(self.unit_field_index >= len(self.field_ids))):
            raise ValueError("identified native units and explicit spatial indices required")
        if expected_unit_ids is not None and not np.array_equal(self.unit_ids, expected_unit_ids):
            raise ValueError("source unit identity/order changed")
        if (expected_unit_field_index is not None
                and not np.array_equal(self.unit_field_index, expected_unit_field_index)):
            raise ValueError("source unit-to-field observation map changed")
        if (self.event_ticks.ndim != 1 or self.event_units.shape != self.event_ticks.shape
                or self.event_ticks.dtype.kind not in "iu" or self.event_units.dtype.kind not in "iu"
                or np.any(np.diff(self.event_ticks) < 0) or np.any(self.event_ticks < 0)
                or np.any(self.event_ticks > self.duration_ticks)
                or np.any(self.event_units < 0) or np.any(self.event_units >= len(self.unit_ids))):
            raise ValueError("ordered identified native event ticks required")
        parameters = episode / "initial_parameters.npz"
        parameter_digest = sha256_file(parameters)
        if parameter_digest != report["artifacts"]["initial_parameters.npz"]:
            raise ValueError("source initial parameters changed")
        with np.load(parameters, allow_pickle=False) as source:
            if not np.array_equal(source["unit_ids"], self.unit_ids):
                raise ValueError("source initial-parameter unit identity/order changed")
            reset, threshold, refractory = (source[k].astype(np.float64) for k in
                                            ("V_reset_rel", "Theta_rel", "t_ref"))
        if (any(v.shape != self.unit_ids.shape or not np.isfinite(v).all()
                for v in (reset, threshold, refractory))
                or np.any(reset >= threshold) or np.any(refractory < 0)):
            raise ValueError("finite source reset/threshold/refractory parameters required")
        ratio = refractory / self.dt_ms
        if (not np.isfinite(ratio).all() or np.any(ratio >= np.iinfo(np.int32).max)
                or not np.allclose(ratio, np.rint(ratio), atol=1e-6, rtol=0)):
            raise ValueError("source refractory duration must be an integer native tick count")
        self.minimum_interval_ticks = np.rint(ratio).astype(np.int64) + 1
        view = json.loads((field_view / "summary.json").read_text())
        if (float(view["dt_ms"]) != self.dt_ms
                or view["window_start_tick"] != 0
                or view["window_stop_tick"] != self.duration_ticks):
            raise ValueError("complete episode field view on the same native clock required")
        self.binding = dict(
            episode_report_sha256=sha256_file(episode / "report.json"),
            field_summary_sha256=sha256_file(field_view / "summary.json"),
            initial_parameters_sha256=parameter_digest,
            spikes_sha256=report["artifacts"]["spikes.npz"],
            spatial_fields_sha256=view["artifacts"]["spatial_fields.npz"],
            native_source_identity=self.source_identity,
            minimum_interval_policy=MINIMUM_INTERVAL_POLICY,
        )

    def renewal_window(self, start_tick: int, stop_tick: int) -> RenewalWindow:
        return renewal_window(self, start_tick, stop_tick, self.minimum_interval_ticks)

    def ready_at(self, tick: int) -> np.ndarray:
        """Availability from spikes strictly before the next observation tick."""
        if type(tick) is not int or not 0 <= tick <= self.duration_ticks:
            raise ValueError("causal boundary must be an integer tick inside the episode")
        stop = np.searchsorted(self.event_ticks, tick, side="left")
        units, events = self.event_units[:stop], self.event_ticks[:stop]
        order = np.lexsort((events, units))
        same = np.diff(units[order]) == 0
        if np.any(np.diff(events[order])[same] < self.minimum_interval_ticks[units[order][1:][same]]):
            raise ValueError("observed prefix violates the source minimum interval")
        last = np.full(len(self.unit_ids), -int(self.minimum_interval_ticks.max()), dtype=np.int64)
        np.maximum.at(last, units, events)
        return np.maximum(last + self.minimum_interval_ticks - tick, 0)


