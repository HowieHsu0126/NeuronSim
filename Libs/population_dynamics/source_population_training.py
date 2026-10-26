"""Shared population fitting operations for independently admitted source cohorts.

Subclasses supply fit/validation cases and _episode(case). Source admission
remains source-specific. The numerical methods and legacy error identifiers
are retained verbatim so saved recipe comparisons can identify unchanged math.
"""
from __future__ import annotations

import numpy as np
import torch


class SourcePopulationTrainingCohort:
    """Share model parameters across complete episodes with different source maps."""

    def _episode(self, case):
        raise NotImplementedError("source cohort must supply an admitted episode")

    def population_configuration(self, config, seed):
        episode = self._episode(self.fit[0])
        metadata = episode.population_metadata()
        result=dict(unit_ids=list(map(int, episode.unit_ids)), unit_field_index=episode.unit_field_index.tolist(),
            field_names=list(episode.field_names), minimum_interval_ticks=episode.minimum_interval_ticks.tolist(),
            tick_seconds=episode.dt_ms / 1000, bin_seconds=episode.dt_ms / 1000 * self.bin_ticks,
            input_size=len(metadata["control_features"]),
            latent_channels=config["latent_channels"], encoder_size=config["encoder_size"],
            representation=config["representation"], initialization_seed=seed,
            unit_chunk_size=config["unit_chunk_size"], metadata=metadata)
        if config.get("control_geometry","static")!="static":
            result["control_geometry"]=config["control_geometry"]
        return result

    def window(self, case):
        # Release the previous event arrays before loading the next full episode.
        self.active_case, self.active_episode = None, None
        self.active_case, self.active_episode = case, self._episode(case)
        return self.active_episode.renewal_window(0, self.active_episode.duration_ticks)

    def model_for_case(self, canonical, case):
        if self.active_case != case:
            raise ValueError("PFC_TRAINING_ACTIVE_EPISODE_MAP_REQUIRED")
        episode = self.active_episode
        if canonical.tick_seconds != episode.dt_ms / 1000 or canonical.bin_ticks != self.bin_ticks:
            raise ValueError("PFC_TRAINING_COMMON_OBSERVATION_CLOCK_REQUIRED")
        return canonical.training_for_circuit(unit_ids=list(map(int, episode.unit_ids)),
            unit_field_index=episode.unit_field_index.tolist(), field_names=list(episode.field_names),
            minimum_interval_ticks=episode.minimum_interval_ticks.tolist(), metadata=episode.population_metadata())

    def initialize_observation(self, canonical, pseudocount):
        """Exposure-weighted least squares of fit-only unit log hazards.

        Streaming QR retains a small design factor instead of storing all units
        from all circuits. Repeated episodes with the same source target geometry
        aggregate each unit before applying the pseudocount. Different known
        target masks remain separate design groups. Validation observations never enter.
        """
        columns = canonical.unit_features.shape[-1] + 1
        factor, target = np.empty((0, columns)), np.empty(0)
        spikes, exposure, fitted_units, design_rows = 0., 0., 0, 0
        groups = sorted({(case["circuit_id"], case.get("source_target_geometry", ())) for case in self.fit})
        source_sizes = {}
        for circuit, geometry in groups:
            successes = eligible = features = None
            for case in (case for case in self.fit if case["circuit_id"] == circuit
                         and case.get("source_target_geometry", ()) == geometry):
                window = self.window(case)
                view = self.model_for_case(canonical, case)
                current = view._intrinsic_unit_features(0,len(view.unit_ids)).detach().cpu().numpy().astype(np.float64)
                if successes is None:
                    successes = np.zeros(len(current), dtype=np.float64)
                    eligible = np.zeros_like(successes)
                    features = current
                elif not np.array_equal(current, features):
                    raise ValueError("PFC_TRAINING_STATIC_CIRCUIT_FEATURES_CHANGED")
                successes += window.counts.sum(axis=0, dtype=np.float64)
                eligible += window.eligible_ticks.sum(axis=0, dtype=np.float64)
            probability = (successes + pseudocount) / (eligible + 2 * pseudocount)
            if np.any(probability <= 0) or np.any(probability >= 1):
                raise ValueError("PFC_TRAINING_INTERIOR_ELIGIBLE_TICK_PROBABILITY_REQUIRED")
            hazard = np.log(-np.log1p(-probability) / canonical.tick_seconds)
            weights = np.sqrt(eligible + 2 * pseudocount)
            for start in range(0, len(features), canonical.unit_chunk_size):
                stop = start + canonical.unit_chunk_size
                design = np.column_stack((features[start:stop], np.ones(len(features[start:stop]))))
                design = np.concatenate((factor, design * weights[start:stop, None]))
                response = np.r_[target, hazard[start:stop] * weights[start:stop]]
                orthogonal, factor = np.linalg.qr(design, mode="reduced")
                target = orthogonal.T @ response
            spikes += successes.sum(); exposure += eligible.sum(); design_rows += len(features)
            source_sizes[circuit] = len(features)
        fitted_units = sum(source_sizes.values())
        # QR compresses rows but does not increase the original design's rank.
        # Keep the full-design LAPACK tolerance. Using the small factor's row
        # count would promote rounding noise from constant circuit descriptors.
        rank_tolerance = float(np.finfo(np.float64).eps * max(design_rows, columns))
        solution, _, rank, singular = np.linalg.lstsq(factor, target, rcond=rank_tolerance)
        if not np.isfinite(solution).all():
            raise ValueError("PFC_TRAINING_NONFINITE_SHARED_HAZARD_INITIALIZATION")
        with torch.no_grad():
            readout = canonical.shared_unit_readout
            readout.weight[-1].copy_(torch.as_tensor(solution[:-1], device=readout.weight.device,
                                                   dtype=readout.weight.dtype))
            readout.bias[-1].copy_(torch.as_tensor(solution[-1], device=readout.bias.device,
                                                 dtype=readout.bias.dtype))
        self.active_case, self.active_episode = None, None
        return dict(method="fit_only_exposure_weighted_shared_log_hazard_streaming_qr",
            fitting_episodes=len(self.fit), fitting_circuits=len({c["circuit_id"] for c in self.fit}),
            fitted_source_units=fitted_units, pseudocount=pseudocount,
            fitting_source_geometry_groups=len(groups), fitted_source_feature_rows=design_rows,
            total_observed_spikes=float(spikes), total_eligible_ticks=float(exposure),
            descriptor_rank=int(rank), singular_values=singular.tolist(),
            rank_relative_tolerance=rank_tolerance,
            coefficients=solution.tolist(), validation_observations_used=False)
