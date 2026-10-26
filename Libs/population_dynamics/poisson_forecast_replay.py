"""Native-unit sample replay and quadrature convergence for trained LDS forecasts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from .native_count_windows import NativeEpisodeCounts
from .poisson_state_space import ControlledPoissonLDS, LatentRandomStreams
from .poisson_trajectory_evaluation import poisson_lognormal_scores
from .poisson_trajectory_training import partition
from .provenance import sha256_file


@torch.no_grad()
def replay(acquisition, fields, training, forecasts, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    rows, sample_rows = [], []
    for seed in (41,42,43):
        folder = training / f"seed_{seed}"
        run = json.loads((folder/"run.json").read_text())
        if sha256_file(folder/"best.pt") != run["best_checkpoint_sha256"]:
            raise ValueError("training checkpoint changed")
        checkpoint = torch.load(folder/"best.pt", map_location="cuda", weights_only=True)
        binding = checkpoint["binding"]; cfg = binding["config"]
        if (sha256_file(acquisition/"design.json") != binding["acquisition_design_sha256"]
                or sha256_file(fields/"dataset.json") != binding["dataset_sha256"]
                or sha256_file(Path(__file__).with_name("poisson_state_space.py")) != binding["model_implementation_sha256"]):
            raise ValueError("model or data identity changed")
        _, cases = partition(json.loads((acquisition/"design.json").read_text()), cfg)
        forecast = forecasts / f"seed_{seed}"
        evaluation = json.loads((forecast/"evaluation.json").read_text())
        if sha256_file(forecast/"scores.csv") != evaluation["scores_sha256"]:
            raise ValueError("forecast scores changed")
        score_rows = list(csv.DictReader((forecast/"scores.csv").open()))
        score_lookup = {(r["case_id"], float(r["origin_ms"]), float(r["horizon_ms"]), r["method"]): r for r in score_rows}
        forecast_cfg = evaluation["configuration"]
        data = NativeEpisodeCounts(acquisition/cases[0]["case_id"], fields/cases[0]["case_id"])
        model = ControlledPoissonLDS(tuple(map(int, data.unit_ids)), 1, cfg["latent_size"], cfg["encoder_size"],
                                    cfg["bin_ms"]/1000, initialization_seed=seed).cuda()
        model.load_state_dict(checkpoint["model"], strict=True); model.eval(); del checkpoint, data
        for index, case in enumerate(cases):
            data = NativeEpisodeCounts(acquisition/case["case_id"], fields/case["case_id"])
            counts, control = data.window(0, data.duration_ticks)
            y = torch.as_tensor(counts, device="cuda"); u = torch.as_tensor(control, device="cuda")
            for origin_index, origin in enumerate(forecast_cfg["origins_ms"]):
                boundary = round(origin/cfg["bin_ms"]); horizon = round(max(forecast_cfg["horizons_ms"])/cfg["bin_ms"])
                state = model.infer_state({"counts": y[None, :boundary], "stimulus": u[None, :boundary]})
                future = u[None, boundary:boundary+horizon]; actual = y[boundary:boundary+horizon]
                moments = model.forecast_moments(state, future)
                mu = moments.mean[0] @ model.loading.T + model.log_rate_bias + math.log(model.bin_seconds)
                variance = torch.einsum("nd,tde,ne->tn", model.loading, moments.covariance[0], model.loading)
                scores = {k: poisson_lognormal_scores(actual, mu, variance, k) for k in (32,64,128)}
                for ms in forecast_cfg["horizons_ms"]:
                    n = round(ms/cfg["bin_ms"])
                    original = float(score_lookup[(case["case_id"], origin, ms, "stochastic")]["composite_marginal_nll"])
                    if not np.isclose(scores[32][:n].mean(), original, rtol=0, atol=1e-12):
                        raise ValueError("marginal score exact replay differs")
                rows.append(dict(seed=seed, case_id=case["case_id"], origin_ms=origin,
                                 unit_bin_marginals=actual.numel(), maximum_log_variance=float(variance.max()),
                                 maximum_absolute_bin_mean_nll_32_64=float(np.max(np.abs(scores[32]-scores[64]))),
                                 maximum_absolute_bin_mean_nll_64_128=float(np.max(np.abs(scores[64]-scores[128])))))
                # First complete case/origin per seed: fresh process, all 100k
                # units, identical random streams, all declared methods.
                if index == 0 and origin_index == 0:
                    base = forecast_cfg["sampling_seed_base"] + seed*100000
                    mapping = torch.as_tensor(data.unit_field_index, device="cuda")
                    draws = forecast_cfg["joint_samples"]
                    for method in forecast_cfg["methods"]:
                        streams = LatentRandomStreams.from_seeds(base, base+1, base+2, device=model.loading.device)
                        if method == "stochastic":
                            sample = model.simulate(state, future, streams, draws)
                            generated = sample.counts[:, 0]; latent = sample.latent_states[:, 0].cpu().numpy()
                            del sample
                        elif method == "latent_mean":
                            generated = torch.poisson(mu.exp().expand(draws, -1, -1), generator=streams.observation).long()
                            latent = moments.mean[0].cpu().numpy()[None]
                        else:
                            log_intensity = mu + variance.sqrt()*torch.randn((draws,*mu.shape),device=mu.device,dtype=mu.dtype,generator=streams.dynamics)
                            generated = torch.poisson(log_intensity.exp(), generator=streams.observation).long()
                            latent = np.empty(0, np.float32); del log_intensity
                        ref = score_lookup[(case["case_id"], origin, max(forecast_cfg["horizons_ms"]), method)]
                        with np.load(forecast/ref["ensemble_artifact"], allow_pickle=False) as stored:
                            field = torch.zeros((draws, horizon, stored["observed_field_counts"].shape[-1]), device="cuda", dtype=torch.int64)
                            field.index_add_(-1, mapping, generated)
                            if not np.array_equal(field.cpu().numpy(), stored["sampled_field_counts"]):
                                raise ValueError("full-unit generated field replay differs")
                            if not np.array_equal(latent, stored["latent_states"]):
                                raise ValueError("latent sample replay differs")
                            sample_rows.append(dict(seed=seed, method=method, units=len(data.unit_ids),
                                                    unit_counts_generated=generated.numel(), field_values_compared=field.numel(),
                                                    latent_values_compared=latent.size, exact=True))
                        del generated, field
            print(json.dumps(dict(seed=seed, case=case["case_id"], checked_origins=len(rows))), flush=True)
            del y, u
        del model
    report = dict(status="REPLAYED_DEVELOPMENT", marginal_checks=rows, sample_replays=sample_rows,
                  maximum_absolute_bin_mean_nll_32_64=max(r["maximum_absolute_bin_mean_nll_32_64"] for r in rows),
                  maximum_absolute_bin_mean_nll_64_128=max(r["maximum_absolute_bin_mean_nll_64_128"] for r in rows),
                  interpretation="quadrature node convergence of bin-mean marginal scores; not proof of every tail probability",
                  scientific_claim_qualified=False, formal_test_accessed=False)
    (output/"replay.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("acquisition", "fields", "training", "forecasts", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.acquisition, args.fields, args.training, args.forecasts, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("marginal_checks", "sample_replays")}))


if __name__ == "__main__":
    main()
