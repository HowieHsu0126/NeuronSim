"""Shared complete-unit forecast scores and replay for LDNS and EAG."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .latent_spike_diffusion_training import _seed, _write_json
from .provenance import sha256_file
from .trajectory_sample_metrics import unit_count_metrics
from .trajectory_scores import energy_score, field_pairs, variogram_score

HORIZONS_MS = [20., 80., 500.]
DRAWS = 16

def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def sample_scores(counts, truth, field_index, units, mean, sd, interval, bin_ticks, bin_seconds, pairs):
    mapping = torch.as_tensor(field_index, device=counts.device)
    fields = counts.new_zeros((len(counts), counts.shape[1], len(units)))
    fields.index_add_(-1, mapping, counts)
    observed = counts.new_zeros(fields.shape[1:]); observed.index_add_(-1, mapping, truth.to(torch.int64))
    sampled = (fields.cpu().numpy() / (units * bin_seconds) - mean) / sd
    actual = (observed.cpu().numpy() / (units * bin_seconds) - mean) / sd
    rows = []
    for ms in HORIZONS_MS:
        bins = round(ms / (bin_seconds * 1000))
        if bins < 1 or bins > truth.shape[0] or bins * bin_seconds * 1000 != ms:
            raise ValueError("matched horizon outside generated trajectory")
        x, y = sampled[:, :bins], actual[:bins]
        row = dict(horizon_ms=ms, energy_score=energy_score(x, y),
            fitting_standardized_field_rate_mse=float(((x.mean(0)-y)**2-x.var(0,ddof=1)/len(x)).mean()),
            **unit_count_metrics(counts[:, :bins], truth[:bins], interval, bin_ticks, [1, 2, 4]))
        for label, indices in pairs.items():
            row["variogram_" + label] = variogram_score(x, y, indices[:, 0], indices[:, 1])
        for lag in [1, 4, 16]:
            row[f"variogram_temporal_lag_{lag}"] = (variogram_score(x, y, np.arange(len(units)),
                np.arange(len(units)), lag) if lag < bins else None)
        rows.append(row)
    return rows, fields.cpu().numpy(), observed.cpu().numpy()


def fitting_scale(cohort, scale_path):
    with np.load(scale_path, allow_pickle=False) as scale:
        mean, sd, units, ids = [scale[k].copy() for k in
                              ("mean_hz", "standard_deviation_hz", "unit_count", "field_ids")]
    if (not np.array_equal(ids, cohort.field_ids)
            or not np.array_equal(units, np.bincount(cohort.unit_field_index, minlength=len(ids)))
            or not np.isfinite(mean).all() or not np.isfinite(sd).all() or np.any(sd <= 0)):
        raise ValueError("matched source map and fitting field scale required")
    definitions = [json.loads(str(value)) for value in ids]
    pairs = field_pairs([row[0] for row in definitions], [row[2] for row in definitions], [row[1] for row in definitions])
    return mean, sd, units, pairs


def evaluate_counts(acquisition, fields, training, config_path, scale_path, output, seed, *,
                    replay, load, method, infer_state, simulate, evaluator_path, native_episode_factory):
    acquisition, fields, training, config_path, scale_path, output = map(Path,
        (acquisition, fields, training, config_path, scale_path, output))
    if output.exists() and not replay:
        raise FileExistsError("existing joint-count evaluation requires explicit replay")
    model, cohort, config, mean, sd, units, pairs = load(acquisition, fields, training, config_path, scale_path, seed)
    binding = dict(training_committed_sha256=sha256_file(training / "committed.json"),
        cohort=cohort.binding, configuration_sha256=sha256_file(config_path),
        fitting_field_scale_sha256=sha256_file(scale_path), evaluator_sha256=sha256_file(Path(evaluator_path)),
        shared_evaluator_sha256=sha256_file(Path(__file__)), method=method,
        unit_metric_sha256=sha256_file(Path(__file__).with_name("trajectory_sample_metrics.py")),
        field_metric_sha256=sha256_file(Path(__file__).with_name("trajectory_scores.py")),
        origins_bins=config["forecast_origins_bins"], horizons_ms=HORIZONS_MS, draws=DRAWS)
    if replay:
        original = json.loads((output / "evaluation.json").read_text())
        if original["binding"] != binding or original["status"] != "EVALUATED_DEVELOPMENT":
            raise ValueError("completed bound joint-count evaluation required for replay")
        for name, digest in original["artifacts"].items():
            if sha256_file(output / name) != digest:
                raise ValueError("joint-count saved output changed: " + name)
        with (output / "scores.csv").open() as stream:
            reference = list(csv.DictReader(stream))
        if len(reference) != 144:
            raise ValueError("complete 144-row saved score table required for replay")
    else:
        output.mkdir(parents=True)
        _write_json(output / "evaluation.json", dict(status="RUNNING", binding=binding, scientific_claim_qualified=False))
    rows, costs, artifacts = [], [], {}
    started = time.perf_counter()
    for case in cohort.validation:
        load_start = time.perf_counter()
        counts, controls = cohort.window(case)
        native = native_episode_factory(acquisition / case["case_id"], fields / case["case_id"],
            expected_unit_ids=cohort.unit_ids, expected_unit_field_index=cohort.unit_field_index)
        episode_loading = time.perf_counter() - load_start
        for origin in config["forecast_origins_bins"]:
            origin_start = time.perf_counter()
            y = torch.as_tensor(counts[None, :origin], device=config["device"], dtype=torch.float32)
            u = torch.as_tensor(controls[None, origin:origin + config["forecast_bins"]],
                                device=config["device"], dtype=torch.float32)
            _sync(config["device"]); start = time.perf_counter()
            observed_controls = torch.as_tensor(controls[None, :origin], device=config["device"], dtype=torch.float32)
            state = infer_state(model, y, observed_controls)
            _sync(config["device"]); inference_seconds = time.perf_counter() - start
            start = time.perf_counter(); generated, latents, digests = [], [], []
            global_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state_all() if config["device"] == "cuda" else []
            for draw in range(DRAWS):
                result = simulate(model, state, u, config,
                    latent_rng=torch.Generator(device=config["device"]).manual_seed(_seed(seed, "latent", case["case_id"], origin, draw)),
                    observation_rng=torch.Generator(device=config["device"]).manual_seed(_seed(seed, "observation", case["case_id"], origin, draw)))
                digests.append(hashlib.sha256(result["counts"].cpu().numpy().tobytes()).hexdigest())
                generated.append(result["counts"][0].cpu()); latents.append(result["latent"][0].cpu())
                del result
            if (not torch.equal(global_rng, torch.get_rng_state())
                    or any(not torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else []))):
                raise RuntimeError("joint-count evaluation changed global CPU RNG")
            _sync(config["device"]); generation_seconds = time.perf_counter() - start
            ensemble, latent = torch.stack(generated), torch.stack(latents)
            truth = torch.as_tensor(counts[origin:origin + config["forecast_bins"]])
            start = time.perf_counter()
            current, projected, actual = sample_scores(ensemble, truth, cohort.unit_field_index, units,
                mean, sd, native.minimum_interval_ticks, native.bin_ticks, cohort.bin_seconds, pairs)
            scoring_seconds = time.perf_counter() - start
            identity = dict(seed=seed, block=case["block"], case_id=case["case_id"], template_id=case["template_id"],
                            origin_ms=origin * config["bin_ms"], method=method)
            current = [dict(identity, **row) for row in current]
            payload = dict(sampled_field_counts=projected, observed_field_counts=actual,
                latent_states=latent.numpy(), prefix_condition=state.latent.cpu().numpy(), observed_controls=observed_controls.cpu().numpy(),
                sampled_count_sha256=np.asarray(digests), unit_ids=cohort.unit_ids, field_ids=cohort.field_ids)
            name = f"{case['case_id']}_origin_{identity['origin_ms']:g}.npz"
            start = time.perf_counter()
            if replay:
                with np.load(output / name, allow_pickle=False) as saved:
                    for key, value in payload.items(): np.testing.assert_array_equal(saved[key], value)
                for a, b in zip(current, reference[len(rows):len(rows) + len(current)]):
                    for key, value in a.items():
                        if value is None:
                            if b[key] != "": raise ValueError("joint-count replay missing-value differs")
                        elif isinstance(value, str):
                            if value != b[key]: raise ValueError("joint-count replay identity differs")
                        elif value != float(b[key]): raise ValueError("joint-count replay metric differs: " + key)
                if not rows:
                    with np.load(output / "full_unit_samples.npz", allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["sampled_unit_counts"], ensemble.numpy())
                        np.testing.assert_array_equal(saved["unit_ids"], cohort.unit_ids)
                        np.testing.assert_array_equal(saved["observed_unit_counts"], truth.numpy())
            else:
                np.savez_compressed(output / name, **payload); artifacts[name] = sha256_file(output / name)
                if not rows:
                    np.savez_compressed(output / "full_unit_samples.npz", sampled_unit_counts=ensemble.numpy(),
                                        observed_unit_counts=truth.numpy(), unit_ids=cohort.unit_ids)
                    artifacts["full_unit_samples.npz"] = sha256_file(output / "full_unit_samples.npz")
            archive_seconds = time.perf_counter() - start
            rows.extend(current)
            if not replay:
                with (output / "scores.csv").open("a", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    if len(rows) == len(current): writer.writeheader()
                    writer.writerows(current)
            costs.append(dict(case_id=case["case_id"], origin_ms=identity["origin_ms"],
                episode_loading_seconds=episode_loading, inference_seconds=inference_seconds,
                complete_unit_generation_seconds=generation_seconds, projection_and_scoring_seconds=scoring_seconds,
                serialization_and_hash_seconds=archive_seconds, origin_total_seconds=time.perf_counter()-origin_start,
                generated_count_values=ensemble.numel(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
            del ensemble, generated, latent, payload, y, u
    if len(rows) != 144 or len(costs) != 48:
        raise ValueError("complete joint-count 48-origin/144-score denominator required")
    if replay:
        report = dict(status="PASS", count_values_digest_replayed=sum(row["generated_count_values"] for row in costs),
            source_unit_count=len(cohort.unit_ids),
            saved_full_unit_origins_compared=1, all_origin_count_digests_fields_latents_metrics_exact=True,
            evaluation_sha256=sha256_file(output/"evaluation.json"), scientific_claim_qualified=False)
        _write_json(output / "replay.json", report)
    else:
        with (output / "costs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(costs[0])); writer.writeheader(); writer.writerows(costs)
        artifacts.update({name:sha256_file(output/name) for name in ("scores.csv", "costs.csv")})
        report = dict(status="EVALUATED_DEVELOPMENT", binding=binding, artifacts=artifacts, score_rows=144,
            complete_origins=48, source_unit_count=len(cohort.unit_ids), method=method, elapsed_seconds=time.perf_counter()-started, seed=seed,
            observation_law="source_softplus_poisson", native_refractory_qualified=False,
            scientific_claim_qualified=False, hardware=dict(device=config["device"],torch=torch.__version__))
        _write_json(output / "evaluation.json", report)
    return report

