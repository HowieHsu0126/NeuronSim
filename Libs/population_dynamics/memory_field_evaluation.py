"""Causal free forecasts from delivered MemNO field-mean checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .memory_field_training import make_model, load_committed, write_json
from .memory_field_cohort import MemoryFieldCohort
from .provenance import sha256_file

ORIGINS = (200, 600, 1000)
HORIZONS = (4, 16, 100)


@torch.no_grad()
def forecast(model, observed, controls, bins):
    """Observe bins [0,o); predict bins [o,o+bins) with target-bin controls.

    ``observed`` contains only admissible prefix fields. Controls cover the
    prefix and forecast bins. Future response fields cannot enter this API.
    Reset explicit S4 state at each independent forecast origin.
    """
    origin = observed.shape[1]
    if (observed.ndim != 5 or controls.ndim != 3 or type(bins) is not int or bins < 1
            or origin < 1 or controls.shape[:2] != (observed.shape[0], origin + bins)):
        raise ValueError("complete observed prefix and registered target-bin controls required")
    model.eval()
    state = model.reset_state(observed.shape[0])
    for index in range(origin - 1):
        _, state = model.one_step(observed[:, index], controls[:, index + 1], state)
    current = observed[:, -1]
    outputs = []
    for index in range(origin, origin + bins):
        current, state = model.one_step(current, controls[:, index], state)
        outputs.append(current)
    return torch.stack(outputs, dim=1)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(acquisition, fields, training, output, *, allow_partial=False, replay_from=None):
    acquisition, fields, training, output = map(Path, (acquisition, fields, training, output))
    if output.exists():
        raise FileExistsError("MemNO evaluation needs a new output directory")
    run = json.loads((training / "run.json").read_text())
    if run["status"] != "FITTED_DEVELOPMENT" and not allow_partial:
        raise ValueError("completed MemNO development training required")
    if run["best_checkpoint"] is None:
        raise ValueError("at least one committed MemNO epoch required")
    binding = run["binding"]
    config = binding["config"]
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(config["device"])
    cohort = MemoryFieldCohort(acquisition, fields, config)
    if (cohort.field_binding() != binding["dataset"]
            or not np.array_equal(cohort.scale, np.asarray(binding["rate_scale"], dtype=np.float32))):
        raise ValueError("MemNO evaluation source map or fitting scale changed")
    load_committed(training, binding, device)
    checkpoint = training / run["best_checkpoint"]
    if sha256_file(checkpoint) != run["checkpoint_sha256"][run["best_checkpoint"]]:
        raise ValueError("MemNO selected checkpoint changed")
    model = make_model(cohort, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    scale = torch.as_tensor(cohort.scale, device=device)
    output.mkdir(parents=True)
    rows, costs, artifacts = [], [], {}
    replay = Path(replay_from) if replay_from else None
    for case in cohort.validation:
        rate, controls = cohort.window(case)
        for origin in ORIGINS:
            if origin + max(HORIZONS) > len(rate):
                raise ValueError("complete registered forecast horizon required")
            synchronize(device)
            began = time.monotonic()
            observed = torch.as_tensor(rate[None, :origin], device=device) / scale
            control = torch.as_tensor(controls[None, :origin + max(HORIZONS)], device=device)
            prediction = forecast(model, observed, control, max(HORIZONS)) * scale
            synchronize(device)
            forecast_seconds = time.monotonic() - began
            predicted = prediction[0].cpu().numpy().reshape(max(HORIZONS), -1)
            truth = rate[origin:origin + max(HORIZONS)].reshape(max(HORIZONS), -1)
            path = output / f"{case['case_id']}_origin_{origin:04d}.npz"
            if replay:
                with np.load(replay / path.name, allow_pickle=False) as original:
                    if (not np.array_equal(original["source_field_ids"], cohort.field_ids)
                            or not np.array_equal(original["truth_rate_hz"], truth)
                            or not np.array_equal(original["predicted_rate_hz"], predicted)):
                        raise ValueError("MemNO fresh-process forecast replay changed")
            np.savez_compressed(path, predicted_rate_hz=predicted, truth_rate_hz=truth,
                source_field_ids=cohort.field_ids, source_unit_ids=cohort.unit_ids,
                origin_bin=origin, bin_seconds=cohort.bin_seconds)
            artifacts[path.name] = sha256_file(path)
            cost_start = time.monotonic()
            for horizon in HORIZONS:
                error = predicted[:horizon].astype(np.float64) - truth[:horizon]
                rows.append(dict(seed=run["training_seed"], method="MemNO_S4FFNO_paper",
                    block=case["block"], case_id=case["case_id"], origin_ms=origin * 5.,
                    horizon_ms=horizon * 5., field_rate_mse=float(np.mean(error ** 2)),
                    field_rate_mae=float(np.mean(np.abs(error))), field_rate_units="Hz_per_neuron",
                    complete_fields=len(cohort.field_ids), identified_unit_scores=False))
            costs.append(dict(case_id=case["case_id"], origin_ms=origin * 5.,
                              causal_forecast_seconds=forecast_seconds,
                              score_seconds=time.monotonic() - cost_start))
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    artifacts["scores.csv"] = sha256_file(output / "scores.csv")
    write_json(output / "evaluation.json", dict(status="REPLAYED_DEVELOPMENT" if replay else "EVALUATED_DEVELOPMENT",
        training_seed=run["training_seed"], score_rows=len(rows), origins=len(costs),
        binding=binding, checkpoint_sha256=sha256_file(checkpoint), artifacts=artifacts,
        origin_costs=costs, formal_test_accessed=False, field_mean_only=True,
        scientific_claim_qualified=False, partial_training_diagnostic=allow_partial,
        timing_scope="prefix transfer, causal state warmup and complete field rollout; score cost separate",
        end_to_end_unit_speedup_qualified=False))
    return dict(score_rows=len(rows), origins=len(costs), replayed=bool(replay))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "training", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="software diagnostic only")
    parser.add_argument("--replay-from", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.acquisition, args.fields, args.training, args.output,
                             allow_partial=args.allow_partial, replay_from=args.replay_from)))


if __name__ == "__main__":
    main()
