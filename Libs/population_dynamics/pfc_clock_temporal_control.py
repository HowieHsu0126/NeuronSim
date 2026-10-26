"""A paired temporal-order control for the existing coupled PFC input stream.

Generate the complete original coarse/middle/fine triplet first, then reverse
only the child time order within that coarse tick. This is the same auxiliary
realization, not another independent repeat. No source simulations run here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .pfc_clock_coupling import CoupledClockRows
from .pfc_coupled_drive import CoupledFineDrive, create_coupled_drive


class TemporalReversalRows:
    """Wrap one native coupled row stream without changing its RNG draw order.

    The refinement kernel is symmetric in its two children. Conditional on the
    middle row counts, the two fine refinements are independent with the same
    kernel. Swapping both middle branches and reversing each fine sibling pair
    therefore preserves the complete coupled input law under ideal independent
    auxiliary randomness. It preserves all original per-coarse-tick, per-neuron
    counts at each level, including each fine sibling total assigned to its
    reversed middle parent. Counts need not be conserved between clock levels.

    Neural responses need not be unchanged. This control establishes no neural
    error, variance reduction, or inference-coverage result.
    """

    def __init__(self, rows: CoupledClockRows):
        if not isinstance(rows, CoupledClockRows):
            raise ValueError("PFC_TEMPORAL_CONTROL_ORIGINAL_COUPLED_ROWS_REQUIRED")
        self._original = rows

    @property
    def coarse(self):
        return self._original.coarse

    def take_row(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coarse, middle, fine = self._original.take_row()
        # Advanced indexing copies the child arrays. Neither the I/E unit axis
        # nor the native fine-generation order is changed by the transformation.
        return coarse, middle[[1, 0], :], fine[[3, 2, 1, 0], :]

    def state_arrays(self) -> dict[str, np.ndarray]:
        return self._original.state_arrays()

    def metadata(self) -> dict[str, Any]:
        return dict(self._original.metadata(), input_order_control=dict(
            name="within_coarse_tick_tree_temporal_reversal",
            scope="paired_input_order_control",
            middle_row_order=[1, 0], fine_row_order=[3, 2, 1, 0],
            unit_axis_permuted=False,
            native_triplet_generated_before_reordering=True,
            rng_draw_order_changed=False,
            same_original_auxiliary_realization=True,
            independent_repeat=False, new_independent_blocks=0,
            per_coarse_tick_per_neuron_totals_preserved=True,
            fine_child_totals_follow_reversed_middle_parents=True,
            cross_level_event_conservation_claimed=False,
            joint_input_law_preserved_by_tree_exchangeability=True,
            law_assumption="ideal independent auxiliary randomness",
            neural_error_reduction_established=False,
            variance_reduction_established=False,
            coverage_established=False,
        ))


def create_temporal_reversal_drive(
    config: Mapping[str, Any],
    case: Mapping[str, Any],
    coarse_dt_ms: float,
    refinement_seeds: Sequence[int],
) -> tuple[np.ndarray, CoupledFineDrive]:
    """Build the normal coupled fine driver with within-parent time reversal."""
    return create_coupled_drive(
        config, case, coarse_dt_ms, refinement_seeds,
        row_adapter=TemporalReversalRows,
    )
