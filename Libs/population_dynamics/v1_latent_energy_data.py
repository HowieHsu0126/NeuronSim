"""Complete source-ordered V1 counts for the existing dense EAG baseline.

Dense model channels are positions in each circuit's declared source order.
Neuron identities remain case-specific; no correspondence between neurons from
independently generated circuits is assumed. Source admission and original
whole-circuit partitions are shared with the population training cohort.
"""
from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path

import numpy as np

from .v1_population_training_data import V1PopulationTrainingCohort


class V1LatentEnergyCohort(V1PopulationTrainingCohort):
    """Return all count columns and planned controls on the declared EAG clock."""

    def __init__(self, inputs_path: Path, config: dict, *,
                 config_validator: Callable[[dict, int], None]):
        self._channel_count = None
        self._control_features = None
        super().__init__(inputs_path, config, config_validator=config_validator)
        self.unit_ids = np.arange(self._channel_count, dtype=np.int64)
        self.binding = dict(self.binding, count_adapter=dict(
            schema="v1_latent_energy_source_ordered_counts_v1",
            model_channel_identity="zero_based_channel_position",
            source_unit_order="unchanged_per_case_circuit_archive_order",
            source_unit_identity="native_episode_unit_ids_and_existing_source_binding",
            cross_circuit_unit_correspondence="not_assumed",
            unit_count=self._channel_count,
            complete_episode_bins=config["complete_episode_bins"],
            bin_ms=config["bin_ms"], bin_ticks=self.bin_ticks,
            control_features=list(self._control_features)))

    def _episode(self, case):
        episode = super()._episode(case)
        config, model = self.config, self.config["model"]
        units = len(episode.unit_ids)
        if self._channel_count is not None and units != self._channel_count:
            raise ValueError("V1_EAG_EQUAL_SOURCE_CHANNEL_COUNTS_REQUIRED")
        bins = config["complete_episode_bins"]
        if (episode.duration_ticks != bins * self.bin_ticks
                or model["complete_episode_bins"] != bins
                or not math.isclose(episode.dt_ms * self.bin_ticks, config["bin_ms"], rel_tol=1e-12)
                or not math.isclose(model["bin_seconds"] * 1000, config["bin_ms"], rel_tol=1e-12)):
            raise ValueError("V1_EAG_DECLARED_COMPLETE_SOURCE_CLOCK_REQUIRED")
        controls = tuple(episode.model_control_features)
        if (len(controls) != model["input_size"]
                or (self._control_features is not None and controls != self._control_features)):
            raise ValueError("V1_EAG_DECLARED_SOURCE_CONTROL_CHANNELS_REQUIRED")
        self._channel_count, self._control_features = units, controls
        return episode

    def native_episode(self, case):
        """Expose the admitted case's native unit IDs, field map and source binding."""
        if self.active_case != case or self.active_episode is None:
            self.active_case, self.active_episode = None, None
            episode = self._episode(case)
            self.active_case, self.active_episode = case, episode
        return self.active_episode

    def window(self, case):
        """Read complete binned counts without computing unused renewal exposures."""
        episode = self.native_episode(case)
        counts, controls = episode.window(0, episode.duration_ticks)
        bins, width = self.config["complete_episode_bins"], self.config["model"]["input_size"]
        if (counts.shape != (bins, self._channel_count)
                or controls.shape != (bins, width)):
            raise ValueError("V1_EAG_COMPLETE_COUNT_AND_CONTROL_DENOMINATORS_REQUIRED")
        return counts, controls
