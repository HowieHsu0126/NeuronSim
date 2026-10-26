"""Bounded all-cortical-unit count windows on a verified native event clock."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .nest_gpu_network import ticks
from .provenance import sha256_file


class NativeEpisodeCounts:
    """Keep sparse events in memory; materialize only the requested unit window."""

    def __init__(self, episode: Path, field_view: Path):
        report = json.loads((episode / "report.json").read_text())
        view = json.loads((field_view / "summary.json").read_text())
        if (report["status"] != "EXECUTED" or view["status"] != "DESCRIBED"
                or view["episode_report_sha256"] != sha256_file(episode / "report.json")
                or sha256_file(episode / "spikes.npz") != report["artifacts"]["spikes.npz"]):
            raise ValueError("verified native episode and bound spatial view required")
        self.dt_ms = float(report["dt_ms"])
        self.duration_ticks = int(ticks([report["duration_ms"]], self.dt_ms)[0])
        self.bin_ms = float(view["settings"]["bin_ms"])
        self.bin_ticks = int(ticks([self.bin_ms], self.dt_ms)[0])
        if self.bin_ticks < 1:
            raise ValueError("positive native count bin required")
        with np.load(episode / "spikes.npz", allow_pickle=False) as data:
            all_ids = data["unit_ids"]
            labels = data["populations"]
            cortical = labels != "TC"
            self.unit_ids = all_ids[cortical]
            if np.any(np.diff(self.unit_ids) <= 0) or np.any(np.diff(all_ids) <= 0):
                raise ValueError("source-indexed increasing native unit order required")
            event_positions = np.searchsorted(all_ids, data["spike_unit_ids"])
            selected = cortical[event_positions]
            self.event_ticks = data["spike_ticks"][selected]
            self.event_units = np.searchsorted(self.unit_ids, data["spike_unit_ids"][selected])
            # TC relays are deterministic here; use the verified actual control
            # events rather than a second, independently mutable time table.
            self.pulse_ticks = np.unique(data["spike_ticks"][~selected])
        field_path = field_view / "spatial_fields.npz"
        if sha256_file(field_path) != view["artifacts"]["spatial_fields.npz"]:
            raise ValueError("spatial observation map changed")
        with np.load(field_path, allow_pickle=False) as field:
            if not np.array_equal(self.unit_ids, field["source_unit_ids"]):
                raise ValueError("unit order differs from observation map")
            self.unit_field_index = field["unit_field_index"]
            self.field_ids = field["field_ids"]

    def window(self, start_tick: int, stop_tick: int):
        """Return exact [start,stop) counts and known TC pulse counts per bin."""
        if (type(start_tick) is not int or type(stop_tick) is not int
                or start_tick < 0 or stop_tick > self.duration_ticks or stop_tick <= start_tick
                or start_tick % self.bin_ticks or stop_tick % self.bin_ticks):
            raise ValueError("complete aligned native window required")
        n_bins = (stop_tick - start_tick) // self.bin_ticks
        left, right = np.searchsorted(self.event_ticks, [start_tick, stop_tick], side="left")
        count = np.zeros((n_bins, len(self.unit_ids)), dtype=np.float32)
        index = (self.event_ticks[left:right] - start_tick) // self.bin_ticks
        np.add.at(count, (index, self.event_units[left:right]), 1)
        pulse = self.pulse_ticks[(self.pulse_ticks >= start_tick) & (self.pulse_ticks < stop_tick)]
        stimulus = np.bincount((pulse - start_tick) // self.bin_ticks, minlength=n_bins).astype(np.float32)[:, None]
        return count, stimulus
