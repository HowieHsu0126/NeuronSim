"""Held-out, control-conditioned residual-history witnesses for state selection.

This is a finite predictive witness, not a distributional sufficiency theorem.
Targets may contain joint field statistics and paired control responses, but
must have been declared before selection. Neither model seeds nor overlapping
windows replace the source's independent circuit/initial-noise block identities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .conditional_regression import conditional_prediction


def _matrix(value, label):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not np.isfinite(result).all():
        raise ValueError(f"finite row/feature matrix required: {label}")
    return result


def _ridge_predictions(train_x, train_y, selection_x, penalty):
    center, scale = train_x.mean(0), train_x.std(0)
    scale = np.where(scale > 0, scale, 1.)
    x = (train_x - center) / scale
    query = (selection_x - center) / scale
    mean_y = train_y.mean(0)
    if x.shape[1] <= x.shape[0]:
        weights = np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ (train_y - mean_y))
    else:
        weights = x.T @ np.linalg.solve(x @ x.T + penalty * np.eye(len(x)), train_y - mean_y)
    return query @ weights + mean_y


def _conditioned(features, controls):
    # A history variable can matter only under a particular applied input.
    # Include the same control interaction family in both witness regressions.
    interactions = (features[:, :, None] * controls[:, None, :]).reshape(len(features), -1)
    return np.concatenate((features, controls, interactions), axis=1)


def assess_state(*, fit_state, selection_state, fit_history, selection_history,
                 fit_controls, selection_controls, fit_targets, selection_targets,
                 fit_blocks, selection_blocks, penalty: float, target_names,
                 fit_cells=None, selection_cells=None, witness_kind: str = "linear"):
    if not isinstance(witness_kind, str) or witness_kind not in ("linear", "rbf"):
        raise ValueError("witness_kind must be linear or rbf")
    if witness_kind == "rbf" and (fit_cells is None or selection_cells is None):
        raise ValueError("RBF witness requires explicit fitting and selection factorial cells")
    arrays = {name: _matrix(value, name) for name, value in locals().copy().items()
              if name.startswith(("fit_", "selection_")) and not name.endswith(("blocks", "cells"))}
    fit_blocks, selection_blocks = np.asarray(fit_blocks), np.asarray(selection_blocks)
    if (fit_blocks.ndim != 1 or selection_blocks.ndim != 1
            or len(fit_blocks) != len(arrays["fit_state"])
            or len(selection_blocks) != len(arrays["selection_state"])
            or set(fit_blocks.tolist()) & set(selection_blocks.tolist())):
        raise ValueError("disjoint source blocks required; seeds/windows are not independent blocks")
    for prefix, length in (("fit_", len(fit_blocks)), ("selection_", len(selection_blocks))):
        if any(len(x) != length for key, x in arrays.items() if key.startswith(prefix)):
            raise ValueError("witness rows and source-block identities differ")
    for key in ("state", "history", "controls", "targets"):
        if arrays["fit_" + key].shape[1] != arrays["selection_" + key].shape[1]:
            raise ValueError("fitting/selection feature semantics differ")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("positive predeclared witness penalty required")
    names = tuple(target_names)
    if len(names) != arrays["fit_targets"].shape[1] or len(set(names)) != len(names) or not names:
        raise ValueError("complete named target family required")
    cell_conditioned = fit_cells is not None or selection_cells is not None
    if cell_conditioned:
        fit_cells, selection_cells = np.asarray(fit_cells), np.asarray(selection_cells)
        if (fit_cells.shape != fit_blocks.shape or selection_cells.shape != selection_blocks.shape
                or fit_cells.dtype.kind not in "iu" or selection_cells.dtype.kind not in "iu"
                or np.any(fit_cells < 0) or np.any(selection_cells < 0)
                or not set(selection_cells.tolist()).issubset(set(fit_cells.tolist()))):
            raise ValueError("identified fitting/selection factorial cells required")
        # Each cell identifies the same fixed past/future control program.
        # Reuse unpenalized cell effects without duplicating every unit feature
        # across one-hot interactions. Both arms use the same declared witness family.
        programs = {}
        for cells, controls in ((fit_cells, arrays["fit_controls"]),
                                (selection_cells, arrays["selection_controls"])):
            for cell in np.unique(cells):
                rows = controls[cells == cell]
                if not np.all(rows == rows[0]) or (int(cell) in programs
                        and not np.array_equal(programs[int(cell)], rows[0])):
                    raise ValueError("factorial cell must identify one unchanged control program")
                programs[int(cell)] = rows[0]
        state_train, state_query = arrays["fit_state"], arrays["selection_state"]
        full_train = np.concatenate((state_train, arrays["fit_history"]), 1)
        full_query = np.concatenate((state_query, arrays["selection_history"]), 1)
    else:
        state_train = _conditioned(arrays["fit_state"], arrays["fit_controls"])
        state_query = _conditioned(arrays["selection_state"], arrays["selection_controls"])
        full_train = _conditioned(np.concatenate((arrays["fit_state"], arrays["fit_history"]), 1), arrays["fit_controls"])
        full_query = _conditioned(np.concatenate((arrays["selection_state"], arrays["selection_history"]), 1), arrays["selection_controls"])
    target_scale = arrays["fit_targets"].std(0)
    # An unvarying training target gives no empirical tolerance calibration.
    defined = target_scale > 0
    divisor = np.where(defined, target_scale, 1.)
    y = arrays["fit_targets"] / divisor
    truth = arrays["selection_targets"] / divisor
    if cell_conditioned:
        state_prediction, state_diagnostic = conditional_prediction(state_train, y, fit_cells, state_query,
                                                     selection_cells, witness_kind, penalty)
        full_prediction, augmented_diagnostic = conditional_prediction(full_train, y, fit_cells, full_query,
                                                    selection_cells, witness_kind, penalty)
    else:
        state_prediction = _ridge_predictions(state_train, y, state_query, penalty)
        full_prediction = _ridge_predictions(full_train, y, full_query, penalty)
        state_diagnostic = augmented_diagnostic = None
    state_error, full_error = (state_prediction - truth) ** 2, (full_prediction - truth) ** 2
    blocks = []
    for block in np.unique(selection_blocks):
        mask = selection_blocks == block
        blocks.append(dict(block=str(block), rows=int(mask.sum()),
                           state_mse=state_error[mask].mean(0).tolist(),
                           augmented_mse=full_error[mask].mean(0).tolist(),
                           history_gain=(state_error[mask] - full_error[mask]).mean(0).tolist()))
    # Equal independent-block weighting, regardless of windows per block.
    gains = np.asarray([b["history_gain"] for b in blocks])
    return dict(status="COMPLETE_DEVELOPMENT_CLOSURE_WITNESS", target_names=list(names),
                fitting_blocks=len(np.unique(fit_blocks)), selection_blocks=len(blocks),
                source_block_rows=blocks, mean_history_gain=gains.mean(0).tolist(),
                maximum_observed_block_gain=gains.max(0).tolist(),
                training_target_scale=target_scale.tolist(), defined_targets=defined.tolist(),
                penalty=penalty, witness_kind=witness_kind, control_interactions=not cell_conditioned,
                ridge_diagonal_penalty=penalty * len(fit_blocks) if cell_conditioned else penalty,
                state_predictor_diagnostic=state_diagnostic, augmented_predictor_diagnostic=augmented_diagnostic,
                control_conditioning=(f"fixed_factorial_cells_shared_{witness_kind}_state"
                                      if cell_conditioned else "state_control_interactions"),
                normalization_fitted_on="fit_only",
                full_joint_distribution_certified=False, formal_tolerances_qualified=False,
                confidence_intervals_computed=False, calibration_outcomes_accessed=False,
                test_outcomes_accessed=False)


def select_development_state(reports: dict, state_costs: dict, tolerances: dict):
    """Choose the cheapest passing *development* state, retaining every witness.

    Per-target tolerances must be declared externally. This empirical rule is
    deliberately separate from formal simultaneous scientific qualification.
    """
    if set(reports) != set(state_costs) or not reports:
        raise ValueError("every candidate needs both witness and measured state cost")
    candidates = []
    family = None
    for name, report in reports.items():
        targets = report["target_names"]
        if family is None:
            family = targets
        if targets != family or set(targets) != set(tolerances):
            raise ValueError("state comparisons must retain the complete common target family")
        if (report["status"] != "COMPLETE_DEVELOPMENT_CLOSURE_WITNESS"
                or not all(report["defined_targets"])):
            passed = False
        else:
            if any(v is None or not np.isfinite(v) or v < 0 for v in tolerances.values()):
                raise ValueError("predeclared development tolerances required")
            passed = all(g <= tolerances[t] for t, g in zip(targets, report["maximum_observed_block_gain"]))
        cost = state_costs[name]
        if not np.isfinite(cost) or cost <= 0:
            raise ValueError("positive measured state cost required")
        candidates.append(dict(name=name, passes_empirical_witness=bool(passed), state_cost=float(cost)))
    eligible = [c for c in candidates if c["passes_empirical_witness"]]
    chosen = min(eligible, key=lambda c: (c["state_cost"], c["name"]))["name"] if eligible else None
    return dict(status="SELECTED_DEVELOPMENT_STATE" if chosen else "NO_STATE_PASSES_DEVELOPMENT_WITNESS",
                selected_state=chosen, candidates=candidates, witnesses=reports, tolerances=tolerances,
                scientific_closure_qualified=False, source_fidelity_qualified=False,
                production_episodes=0, test_outcomes_accessed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("scope") != "development_only" or config.get("selection_split") != "validation":
        raise ValueError("witness selection requires the declared development validation split")
    with np.load(args.inputs, allow_pickle=False) as data:
        result = assess_state(**{key: data[key] for key in data.files},
                              penalty=config["penalty"], target_names=config["target_names"])
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
