"""Complete source-indexed rate fields and verified deterministic TC controls."""
from __future__ import annotations

import json

import numpy as np

from .native_count_windows import NativeEpisodeCounts
from .native_trajectory_cohort import TrajectoryCohort


def grid_contract(field_ids):
    """Require the source's complete population-major Y/X grid, without sorting."""
    coordinates = [json.loads(str(value)) for value in field_ids]
    populations = list(dict.fromkeys(row[0] for row in coordinates))
    if (not coordinates or any(len(row) != 3 or not isinstance(row[0], str)
            or type(row[1]) is not int or type(row[2]) is not int
            or min(row[1:]) < 0 for row in coordinates)):
        raise ValueError("source population/Y/X field identities required")
    ny = max(row[1] for row in coordinates) + 1
    nx = max(row[2] for row in coordinates) + 1
    expected = [[population, y, x] for population in populations
                for y in range(ny) for x in range(nx)]
    if coordinates != expected:
        raise ValueError("complete source population-major Y/X order required")
    return populations, (ny, nx)


def fitting_scale(episodes):
    """Estimate one source-rate scale per field from complete fitting episodes."""
    values = list(episodes)
    if not values or any(value.shape != values[0].shape or value.ndim != 4
            or not np.isfinite(value).all() or np.any(value < 0) for value in values):
        raise ValueError("aligned finite complete fitting rate episodes required")
    mean = np.mean([value.mean(axis=0, dtype=np.float64) for value in values], axis=0)
    return np.where(mean > 0, mean, 1.).astype(np.float32)


class MemoryFieldCohort(TrajectoryCohort):
    """Reuse the authoritative 56/16 partition; avoid dense 100k-unit tensors.

    Sparse source events independently reconstruct every saved field count.
    Full source identities and the fixed observation map remain bound even
    though this field-only baseline does not emit identified-unit activity.
    """

    def __init__(self, acquisition, fields, config):
        super().__init__(acquisition, fields, config)
        self.populations, self.grid_shape = grid_contract(self.field_ids)
        self._episodes = {}
        for case in self.fit + self.validation:
            self._episodes[case["case_id"]] = self._load_fields(case)
        self.scale = fitting_scale(self._episodes[case["case_id"]][0] for case in self.fit)

    def _load_fields(self, case):
        self._verify_metadata(case)
        raw = self.acquisition / case["case_id"]
        report = json.loads((raw / "report.json").read_text())
        if not report["checks"].get("tc_emissions_exact"):
            raise ValueError("source-verified deterministic exogenous TC relays required")
        native = NativeEpisodeCounts(raw, self.fields / case["case_id"])
        self._verify_episode(native)
        valid = (native.event_ticks >= 0) & (native.event_ticks < native.duration_ticks)
        bins = native.event_ticks[valid] // native.bin_ticks
        fields = native.unit_field_index[native.event_units[valid]]
        reconstructed = np.bincount(bins * len(self.field_ids) + fields,
            minlength=1200 * len(self.field_ids)).reshape(1200, -1)
        pulses = native.pulse_ticks[(native.pulse_ticks >= 0)
                                    & (native.pulse_ticks < native.duration_ticks)]
        control = np.bincount(pulses // native.bin_ticks, minlength=1200).astype(np.float32)[:, None]
        with np.load(self.fields / case["case_id"] / "spatial_fields.npz", allow_pickle=False) as data:
            counts, rates = data["count_sum"], data["observed_rate_hz"]
            occupancy = np.bincount(native.unit_field_index, minlength=len(self.field_ids))
            expected = reconstructed / (self.bin_seconds * occupancy[None])
            if (counts.shape != reconstructed.shape or not np.array_equal(counts, reconstructed)
                    or not np.array_equal(data["total_unit_count"], occupancy)
                    or not np.all(data["complete"]) or np.any(occupancy <= 0)
                    or not np.array_equal(data["observed_unit_count"],
                                          np.broadcast_to(occupancy, counts.shape))
                    or not np.allclose(rates, expected, rtol=1e-12, atol=1e-12)
                    or not np.array_equal(data["bin_edges_ticks"], np.arange(1201) * native.bin_ticks)):
                raise ValueError("complete source-event field counts, rates, occupancy and clock required")
        shape = (1200, len(self.populations), *self.grid_shape)
        return rates.reshape(shape).astype(np.float32), control

    def window(self, case):
        return self._episodes[case["case_id"]]

    def field_binding(self):
        return dict(cohort=self.binding, source_field_ids=self.field_ids.tolist(),
                    source_unit_ids=self.unit_ids.tolist(), grid_shape=list(self.grid_shape),
                    populations=self.populations, field_units="spikes_per_second_per_neuron",
                    control_units="unique_deterministic_TC_relay_pulses_per_bin",
                    control_timing="target_bin_control", bin_seconds=self.bin_seconds)
