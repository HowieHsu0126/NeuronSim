"""Replay field scores and summarize development forecasts without pseudoreplication.

The four realization blocks also selected training epochs. These summaries are
descriptive, with no independent confidence interval or hypothesis test. A
covariance diagnostic compares repeated-template distributions at matched times;
it is not a conditional covariance estimate from one observed continuation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist

from .provenance import sha256_file


def _write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def pair_sets(field_ids):
    definitions = [json.loads(str(value)) for value in field_ids]
    spatial, cross = [], []
    nx = max(v[2] for v in definitions) + 1
    ny = max(v[1] for v in definitions) + 1
    for a, b in itertools.combinations(range(len(definitions)), 2):
        p, y, x = definitions[a]; q, v, u = definitions[b]
        if p == q and ((x == u and (y-v) % ny in (1, ny-1)) or
                       (y == v and (x-u) % nx in (1, nx-1))):
            spatial.append((a, b))
        if p != q and (x, y) == (u, v):
            cross.append((a, b))
    if nx <= 2 or ny <= 2:
        raise ValueError("independent adjacency replay currently requires at least 3x3 cells")
    return {"spatial": np.asarray(spatial), "cross_population": np.asarray(cross)}


def replay_scores(samples, truth, pairs, lags):
    """Independent U-statistic implementation, avoiding the evaluator helpers."""
    m, t, f = samples.shape
    flat = samples.reshape(m, -1)
    energy = (cdist(flat, truth.reshape(1, -1)).mean() - .5 * pdist(flat).mean()) / np.sqrt(t*f)
    result = {"energy_score": float(energy)}
    requests = [("variogram_"+name, ij[:, 0], ij[:, 1], 0) for name, ij in pairs.items()]
    requests += [(f"variogram_temporal_lag_{lag}", np.arange(f), np.arange(f), lag) for lag in lags if lag < t]
    for name, left, right, lag in requests:
        observed = np.abs(truth[:t-lag, left] - truth[lag:, right])
        values = np.abs(samples[:, :t-lag, left] - samples[:, lag:, right])
        # Average products over distinct ensemble draws gives an unbiased
        # estimate of the square of the model variogram expectation.
        first = values.sum(0)
        cross = (first*first - (values*values).sum(0)) / (m*(m-1))
        result[name] = float((observed*observed - 2*observed*first/m + cross).mean())
    return result


def covariance_components(samples, truth, left, right, lag=0):
    """Between-block + within-ensemble covariance at each matched phase/pair.

    samples: [block, draw, time, field]; truth: [block, time, field].
    Correct the between-block mean covariance for Monte Carlo variance. Each
    block contributes one observed realization, regardless of ensemble size.
    """
    b, m, t, _ = samples.shape
    if b < 2 or m < 2 or truth.shape != (b, t, samples.shape[-1]):
        raise ValueError("matching independent blocks and repeated draws required")
    x, y = samples[:, :, :t-lag, left], samples[:, :, lag:, right]
    mx, my = x.mean(1), y.mean(1)
    within = ((x-mx[:, None]) * (y-my[:, None])).sum(1) / (m-1)
    between = ((mx-mx.mean(0)) * (my-my.mean(0))).sum(0) / (b-1)
    predicted = between + (1 - 1/m) * within.mean(0)
    ox, oy = truth[:, :t-lag, left], truth[:, lag:, right]
    observed = ((ox-ox.mean(0)) * (oy-oy.mean(0))).sum(0) / (b-1)
    return observed, predicted, within.mean(0), between


def analyze(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    all_rows, cov_rows, comparison = [], [], []
    cov_payload = {}; max_difference = 0.; counts = 0; bindings = {}
    expected_keys = None
    for seed in (41, 42, 43):
        folder = root / f"seed_{seed}"
        report = json.loads((folder / "evaluation.json").read_text())
        if report["status"] != "EVALUATED_DEVELOPMENT":
            raise ValueError("all three completed forecasts are required")
        if sha256_file(folder / "scores.csv") != report["scores_sha256"]:
            raise ValueError("forecast scores changed")
        cfg = report["configuration"]
        rows = list(csv.DictReader((folder / "scores.csv").open()))
        keys = [(r["block"], r["case_id"], r["origin_ms"], r["horizon_ms"], r["method"]) for r in rows]
        if len(keys) != len(set(keys)) or len(rows) != report["score_rows"]:
            raise ValueError("duplicate or missing forecast rows")
        if expected_keys is None: expected_keys = set(keys)
        if set(keys) != expected_keys: raise ValueError("unmatched seed coverage")
        blocks = sorted({int(r["block"]) for r in rows})
        if blocks != report["evaluation_blocks"]: raise ValueError("block coverage differs")
        with np.load(folder / "fitting_field_scale.npz", allow_pickle=False) as scale:
            mean, sd, units = scale["mean_hz"], scale["standard_deviation_hz"], scale["unit_count"]
            pairs = pair_sets(scale["field_ids"])
        bindings[str(seed)] = {"scores_sha256": report["scores_sha256"],
                               "checkpoint_sha256": report["checkpoint_sha256"]}
        ensembles = {}; grouped = defaultdict(list)
        for row in rows:
            artifact = row["ensemble_artifact"]
            if artifact not in ensembles:
                with np.load(folder / artifact, allow_pickle=False) as array:
                    raw, obs = array["sampled_field_counts"], array["observed_field_counts"]
                    if (raw < 0).any() or (obs < 0).any() or not np.equal(raw, np.floor(raw)).all():
                        raise ValueError("nonnegative field counts required")
                    ensembles[artifact] = ((raw/(units*.005)-mean)/sd, (obs/(units*.005)-mean)/sd)
            sample, truth = ensembles[artifact]
            h = int(float(row["horizon_ms"]) / 5)
            for name, value in replay_scores(sample[:, :h], truth[:h], pairs, cfg["temporal_lags_bins"]).items():
                delta = abs(value-float(row[name])); max_difference = max(max_difference, delta); counts += 1
                if not np.isclose(value, float(row[name]), rtol=1e-10, atol=1e-12):
                    raise ValueError(f"field score replay mismatch: {artifact} {name}")
            grouped[(row["template_id"], row["origin_ms"], row["horizon_ms"], row["method"])].append(row)
        for key, members in grouped.items():
            if sorted(int(r["block"]) for r in members) != blocks:
                raise ValueError("covariance group does not contain each block exactly once")
            h = int(float(key[2])/5)
            sample = np.stack([ensembles[r["ensemble_artifact"]][0][:, :h] for r in members])
            truth = np.stack([ensembles[r["ensemble_artifact"]][1][:h] for r in members])
            requests = [(name, ij[:, 0], ij[:, 1], 0) for name, ij in pairs.items()]
            requests += [(f"temporal_lag_{lag}", np.arange(truth.shape[-1]), np.arange(truth.shape[-1]), lag)
                         for lag in cfg["temporal_lags_bins"] if lag < h]
            for name, left, right, lag in requests:
                observed, predicted, within, between = covariance_components(sample, truth, left, right, lag)
                index = f"g{len(cov_rows)}"
                cov_payload[index] = np.stack([observed, predicted, within, between])
                cov_rows.append(dict(seed=seed, template_id=key[0], origin_ms=key[1], horizon_ms=key[2], method=key[3],
                                     statistic=name, independent_blocks=len(blocks), pairs=len(left), phases=h-lag,
                                     observed_covariance_mean=float(observed.mean()), predicted_covariance_mean=float(predicted.mean()),
                                     covariance_mse=float(((predicted-observed)**2).mean()),
                                     covariance_mae=float(np.abs(predicted-observed).mean()), array_key=index))
        all_rows.extend(rows)
    _write_rows(output / "forecast_all_seeds.csv", all_rows)
    excluded = {"seed", "block", "case_id", "template_id", "origin_ms", "horizon_ms", "method", "ensemble_artifact"}
    metrics = [key for key in all_rows[0] if key not in excluded]
    block_groups = defaultdict(list)
    for row in all_rows: block_groups[(row["seed"], row["block"], row["method"], row["horizon_ms"])].append(row)
    block_rows = []
    for key, members in sorted(block_groups.items()):
        row = dict(zip(["seed", "block", "method", "horizon_ms"], key)); row["episodes_origins"] = len(members)
        for metric in metrics:
            values = [float(r[metric]) for r in members if r[metric] != ""]
            row[metric] = float(np.mean(values)) if values else None
        block_rows.append(row)
    _write_rows(output / "forecast_by_block.csv", block_rows)
    summary = []
    for method in sorted({r["method"] for r in all_rows}):
        for h in sorted({r["horizon_ms"] for r in all_rows}, key=float):
            for metric in metrics:
                members = [r for r in block_rows if r["method"] == method and r["horizon_ms"] == h and r[metric] is not None]
                if not members: continue
                seeds = [np.mean([r[metric] for r in members if int(r["seed"]) == s]) for s in (41,42,43)]
                summary.append(dict(method=method, horizon_ms=h, metric=metric, mean=float(np.mean(seeds)),
                                    minimum_seed_mean=float(min(seeds)), maximum_seed_mean=float(max(seeds)),
                                    independent_blocks=4, training_seeds=3, interpretation="development_descriptive"))
    for h in sorted({r["horizon_ms"] for r in all_rows}, key=float):
        for method in ("latent_mean", "independent_marginals"):
            for metric in metrics:
                for block in blocks:
                    diffs = []
                    for s in (41,42,43):
                        r = [r for r in block_rows if int(r["seed"]) == s and int(r["block"]) == block and r["horizon_ms"] == h]
                        selected = {v["method"]: v[metric] for v in r}
                        if selected[method] is not None: diffs.append(selected["stochastic"]-selected[method])
                    if diffs: comparison.append(dict(block=block, horizon_ms=h, comparator=method, metric=metric,
                                                       stochastic_minus_comparator=float(np.mean(diffs)), training_seeds=3))
    _write_rows(output / "forecast_summary.csv", summary)
    _write_rows(output / "forecast_paired_block_differences.csv", comparison)
    _write_rows(output / "covariance_diagnostic.csv", cov_rows)
    np.savez_compressed(output / "covariance_arrays.npz", **cov_payload)
    report = dict(status="REPLAYED_DEVELOPMENT", seeds=[41,42,43], score_rows=len(all_rows),
                  field_scores_recomputed=counts, maximum_absolute_replay_difference=max_difference,
                  independent_blocks=4, seed_checkpoint_bindings=bindings,
                  covariance_scope="across realization blocks at fixed template, origin and phase; total predictive covariance",
                  covariance_array_axis=["observed", "predicted_total", "mean_within", "between_sample_means"],
                  covariance_comparison="descriptive noisy covariance estimates from four source realizations",
                  uncertainty="no inferential CI or p-value on epoch-selection blocks",
                  costs="generation includes all unit outputs and field projection; excludes archive I/O and source matching",
                  scientific_claim_qualified=False, formal_test_accessed=False)
    (output / "analysis.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.forecasts, args.output)))


if __name__ == "__main__":
    main()
