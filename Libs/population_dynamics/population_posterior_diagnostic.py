"""Compare observed-history reconstruction with conditional one-step decoding.

Both paths evaluate latent means, not latent-integrated predictive laws. The
current-bin posterior reads its observed counts. Observed eligible exposure is
used for the event-time likelihood and an exposure-weighted probability sum;
that sum is not a generative expected count. No samples or fits are produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .refractory_observation import observation_nll
from .native_renewal_data import NativeRenewalEpisode
from .population_forecast_analysis import (CanonicalFields, _delivered_population_analysis,
                                           _population_budget_pair, _write_rows)
from .population_trajectory_evaluation import _load_model
from .provenance import sha256_file


def diagnostic_boundary_vector(values, model):
    """Use the existing witness's float64 normalized-readiness representation."""
    mean = np.asarray(values["boundary_mean"], dtype=np.float64)
    variance = np.asarray(values["boundary_variance"], dtype=np.float64)
    ready = np.asarray(values["boundary_ready_in_ticks"])
    model._ready(ready, 1)
    if mean.shape != variance.shape or mean.shape != (1, len(model.field_names), model.latent_channels):
        raise ValueError("complete boundary moments required")
    interval = model.minimum_interval_ticks.cpu().numpy()
    return np.concatenate((mean.reshape(1, -1), variance.reshape(1, -1), ready / interval), axis=1)


def transition_delta_variance(model, mean, variance, controls):
    """First-order diagonal variance J diag(variance) J^T at the latent mean.

    This includes both local-channel and cross-field derivatives. It excludes
    process variance. The nonlinear prior mean is evaluated at the previous
    posterior mean; neither returned moment is an exact marginal prior moment.
    """
    if (mean.shape != variance.shape or mean.shape[-2:] !=
            (len(model.field_names), model.latent_channels)
            or controls.shape != (*mean.shape[:-2], model.input_size)
            or not torch.isfinite(mean).all() or not torch.isfinite(variance).all()
            or torch.any(variance < 0) or not torch.isfinite(controls).all()):
        raise ValueError("aligned finite latent moments and controls required")
    mixing = model.field_mix_logits.softmax(-1)
    decay = model.decay_logit.sigmoid()
    activation = (mean @ model.local_weight.T
                  + torch.einsum("fg,...gk->...fk", mixing, mean)
                  + torch.einsum("...c,fck->...fk", controls, model.control_weight)
                  + model.transition_bias)
    slope = (1 - decay) * (1 - activation.tanh().square())
    identity = torch.eye(model.latent_channels, dtype=mean.dtype, device=mean.device)
    local = (decay[..., :, None] * identity
             + slope[..., :, None]
             * (model.local_weight + mixing.diag()[:, None, None] * identity))
    off_diagonal = mixing.square() - torch.diag_embed(mixing.diag().square())
    return (torch.einsum("...fkl,...fl->...fk", local.square(), variance)
            + slope.square() * torch.einsum("fg,...gk->...fk", off_diagonal, variance))


@torch.no_grad()
def diagnose_native_plan(model, native, origin_tick: int, stop_tick: int):
    """Score every source unit after a complete observed causal prefix.

    Returned arrays have response-bin or response-bin/field axes. The prior at
    bin t uses posterior[t-1] and control[t], as in sequence_loss. Readiness and
    eligible ticks come from the unmodified native history, with half-open bins.
    """
    if (model.unit_loading.device.type != "cpu"
            or type(origin_tick) is not int or type(stop_tick) is not int
            or origin_tick < native.bin_ticks or stop_tick <= origin_tick
            or origin_tick % native.bin_ticks or stop_tick % native.bin_ticks
            or stop_tick > native.duration_ticks
            or tuple(map(int, native.unit_ids)) != model.unit_ids
            or tuple(map(str, native.field_ids)) != model.field_names
            or not np.array_equal(native.unit_field_index, model.unit_field_index.cpu().numpy())
            or not np.array_equal(native.minimum_interval_ticks, model.minimum_interval_ticks.cpu().numpy())
            or native.dt_ms / 1000 != model.tick_seconds
            or native.bin_ms / 1000 != model.bin_seconds):
        raise ValueError("CPU model and complete source-aligned causal window required")
    window = native.renewal_window(0, stop_tick)
    first = origin_tick // native.bin_ticks
    counts, controls = window.counts[None], window.stimulus[None]
    mean, variance = model.encode(counts, controls)
    previous = mean[:, first - 1:-1]
    u = model._tensor(controls[:, first:])
    posterior = mean[:, first:]
    prior = model.transition_mean(previous, u)
    propagated = transition_delta_variance(model, previous, variance[:, first - 1:-1], u)
    residual = (posterior - prior).double().square()
    process = model.process_log_variance.exp().double()
    residual_variance = variance[:, first:].double() + process + propagated.double()
    if not torch.isfinite(residual_variance).all() or torch.any(residual_variance <= 0):
        raise FloatingPointError("positive finite diagnostic residual variance required")
    bins, fields = posterior.shape[1:3]
    output = {
        "bin_start_tick": np.arange(origin_tick, stop_tick, native.bin_ticks, dtype=np.int64),
        "controls": window.stimulus[first:].copy(),
        "observed_total_spikes": window.counts[first:].sum(-1, dtype=np.float64),
        "eligible_tick_total": window.eligible_ticks[first:].sum(-1, dtype=np.int64),
        "latent_transition_residual_rms": residual.mean((-1, -2)).sqrt()[0].numpy(),
        "process_standardized_residual_rms": (residual / process).mean((-1, -2)).sqrt()[0].numpy(),
        "delta_standardized_residual_rms": (residual / residual_variance).mean((-1, -2)).sqrt()[0].numpy(),
        "posterior_variance_mean": variance[:, first:].mean((-1, -2))[0].numpy(),
        "propagated_previous_variance_mean": propagated.mean((-1, -2))[0].numpy(),
        "process_variance_mean": np.full(bins, float(process.mean())),
        "boundary_mean": mean[:, first - 1].numpy().copy(),
        "boundary_variance": variance[:, first - 1].numpy().copy(),
        "boundary_ready_in_ticks": native.ready_at(origin_tick)[None],
    }
    field_units = model.field_unit_counts.double()
    if torch.any(field_units <= 0):
        raise ValueError("this diagnostic requires nonempty source fields")
    for name, latent in (("posterior", posterior), ("one_step", prior)):
        nll = torch.zeros(bins, dtype=torch.float64)
        probabilities = torch.zeros((bins, fields), dtype=torch.float64)
        rates = torch.zeros_like(probabilities)
        for start in range(0, len(model.unit_ids), model.unit_chunk_size):
            end = min(start + model.unit_chunk_size, len(model.unit_ids))
            y = torch.as_tensor(window.counts[first:, start:end], dtype=torch.float64)
            exposure = torch.as_tensor(window.eligible_ticks[first:, start:end], dtype=torch.float64)
            log_rate = model.log_hazard(latent, start, end)[0].double()
            nll += observation_nll(y, exposure, log_rate, model.tick_seconds)
            rate = log_rate.exp()
            probability = -torch.expm1(-rate * model.tick_seconds)
            index = model.unit_field_index[start:end]
            probabilities.index_add_(-1, index, exposure * probability)
            rates.index_add_(-1, index, rate)
        output[name + "_event_nll"] = nll.numpy()
        output[name + "_event_nll_per_unit_bin"] = nll.numpy() / len(model.unit_ids)
        output[name + "_observed_exposure_probability_sum"] = probabilities.sum(-1).numpy()
        output[name + "_field_exposure_probability_sum"] = probabilities.numpy()
        output[name + "_field_point_rate_hz"] = (rates / field_units).numpy()
    if any(not np.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("nonfinite native posterior diagnostic output")
    return output


POINT_DIAGNOSTIC_CONFIG = dict(
    schema="neural_population_point_transition_diagnostic_v1", training_seeds=[41, 42, 43],
    roles=["parent", "continuation"], training_ceilings=[20, 80], selection_blocks=[14, 15, 16, 17],
    templates=["familiar_0", "familiar_1", "familiar_2", "familiar_3"],
    origins_ms=[1000., 3000., 5000.], horizons_ms=[20., 80., 500.], window_ms=500.,
    core_neurons=100000, source_dt_ms=.1, bin_ms=5., device="cpu", cpu_threads=2,
    conditioning="future_observation_assisted", latent_evaluation="means_not_integrated",
    formal_test_accessed=False, samples_generated=0,
)


def load_point_config(path):
    """Load the fixed source-native diagnostic protocol without changing gates."""
    config = yaml.safe_load(Path(path).read_text())
    if (config != POINT_DIAGNOSTIC_CONFIG
            or any(type(value) is not int for name in ("training_seeds", "training_ceilings", "selection_blocks") for value in config[name])
            or type(config["samples_generated"]) is not int or type(config["formal_test_accessed"]) is not bool):
        raise ValueError("source-native point transition diagnostic configuration changed; PFC metadata-conditioned diagnostics are unsupported")
    return config


def _point_summary(values, observed_rate, scale, bins):
    """Describe point decoders at latent means on the requested native bins."""
    summary = {name: float(np.asarray(values[name])[:bins].mean()) for name in (
        "latent_transition_residual_rms", "process_standardized_residual_rms", "delta_standardized_residual_rms",
        "posterior_variance_mean", "propagated_previous_variance_mean", "process_variance_mean")}
    summary["observed_total_spikes"] = float(values["observed_total_spikes"][:bins].sum())
    summary["eligible_tick_total"] = int(values["eligible_tick_total"][:bins].sum())
    summary["observed_field_rate_hz_mean"] = float(observed_rate[:bins].mean())
    for decoder in ("posterior", "one_step"):
        rate = values[decoder + "_field_point_rate_hz"][:bins]
        error = rate - observed_rate[:bins]
        summary[decoder + "_event_nll"] = float(values[decoder + "_event_nll"][:bins].sum())
        summary[decoder + "_event_nll_per_unit_bin"] = float(values[decoder + "_event_nll_per_unit_bin"][:bins].mean())
        summary[decoder + "_field_point_rate_hz_mean"] = float(rate.mean())
        summary[decoder + "_field_point_rate_mse_hz2"] = float(np.square(error).mean())
        summary[decoder + "_fitting_scale_standardized_field_point_rate_mse"] = float(np.square(error / scale).mean())
        summary[decoder + "_observed_exposure_probability_sum"] = float(
            values[decoder + "_observed_exposure_probability_sum"][:bins].sum())
    if any(not np.isfinite(value) for value in summary.values()):
        raise FloatingPointError("finite point diagnostic summaries required")
    return summary


def diagnose_population_campaign(analyses, acquisition, fields, config_path, output):
    """Describe both committed budgets using future-observation-assisted points.

    All source and selected-checkpoint gates run before output creation. The
    existing checkpoint loader must execute beside the checkpoint-bound
    numerical source files. This caller never changes or bypasses those hashes.
    """
    config = load_point_config(config_path)
    if len(analyses) != 2:
        raise ValueError("point diagnostic requires exactly two complete recurrent population analyses")
    acquisition, fields, output = map(Path, (acquisition, fields, output))
    if output.exists():
        raise FileExistsError("point diagnostic output must be new")
    if acquisition.is_file():
        raise ValueError("PFC metadata-conditioned posterior diagnostics are unsupported; source-native cohort directories required")
    parent, continuation = _population_budget_pair([
        _delivered_population_analysis(Path(path)) for path in analyses])
    items = {"parent": parent, "continuation": continuation}
    if [item[0]["common_training_budget"]["maximum_epochs"] for item in items.values()] != config["training_ceilings"]:
        raise ValueError("point diagnostic requires the registered 20/80 training ceilings")
    source = CanonicalFields(acquisition, fields)
    if source.binding != parent[0]["source_binding"]:
        raise ValueError("point diagnostic canonical source differs from the paired aggregates")
    cases = sorted(source.validation, key=lambda case: case["case_id"])
    expected = {(block, template) for block in config["selection_blocks"] for template in config["templates"]}
    if (len(cases) != 16 or {(case["block"], case["template_id"]) for case in cases} != expected
            or len(source.unit_ids) != config["core_neurons"] or len(source.field_ids) != 128
            or source.bin_ms != config["bin_ms"]):
        raise ValueError("all sixteen source-defined 100000-unit selection cases required")
    native_cases = {}
    interval = None
    for case in cases:
        case_id = case["case_id"]
        native = NativeRenewalEpisode(acquisition / case_id, fields / case_id, expected_source=source.source_identity,
            expected_unit_ids=source.unit_ids, expected_unit_field_index=source.unit_field_index)
        if (native.dt_ms != config["source_dt_ms"] or native.bin_ms != config["bin_ms"]
                or native.duration_ticks * native.dt_ms != 6000.
                or not np.array_equal(native.field_ids, source.field_ids)
                or (interval is not None and not np.array_equal(native.minimum_interval_ticks, interval))):
            raise ValueError("point diagnostic complete source clock, fields or minimum interval changed")
        interval = native.minimum_interval_ticks
        native_cases[case_id] = native
    torch.set_num_threads(config["cpu_threads"])
    first = native_cases[cases[0]["case_id"]]
    models, checkpoints = {}, {}
    for role, item in items.items():
        campaign = Path(item[0]["inputs"]["forecasts"])
        for seed in config["training_seeds"]:
            model, binding, checkpoint = _load_model(campaign / f"seed_{seed}/training", acquisition, fields,
                                                     "population", torch.device("cpu"), first)
            bound = item[0]["model_seed_bindings"][f"population_{seed}"]
            if (binding["seed"] != seed or binding["config"] != bound["training_config"]
                    or binding["config"].get("representation") != "recurrent"
                    or model.representation != "recurrent" or checkpoint != bound["checkpoint_sha256"]):
                raise ValueError("point diagnostic selected checkpoint or recurrent training binding changed")
            models[role, seed] = model
            checkpoints[role + "_" + str(seed)] = dict(checkpoint_sha256=checkpoint,
                training_report_sha256=bound["training_report_sha256"], training_config=bound["training_config"])
    output.mkdir(parents=True, exist_ok=False)
    map_name = "observation_map.npz"
    np.savez_compressed(output / map_name, source_unit_ids=source.unit_ids, source_field_ids=source.field_ids,
        unit_field_index=source.unit_field_index, field_unit_count=source.unit_count,
        minimum_interval_ticks=interval, source_dt_ms=config["source_dt_ms"], bin_ms=config["bin_ms"])
    artifacts = {map_name: sha256_file(output / map_name)}
    rows = []
    for role, item in items.items():
        for seed in config["training_seeds"]:
            model = models[role, seed]
            for case in cases:
                case_id = case["case_id"]
                native = native_cases[case_id]
                for origin_ms in config["origins_ms"]:
                    origin_tick = int(origin_ms / native.dt_ms)
                    stop_tick = int((origin_ms + config["window_ms"]) / native.dt_ms)
                    values = diagnose_native_plan(model, native, origin_tick, stop_tick)
                    starts = np.arange(origin_tick, stop_tick, native.bin_ticks, dtype=np.int64)
                    start_bin, stop_bin = origin_tick // native.bin_ticks, stop_tick // native.bin_ticks
                    truth = source.truth[case_id][start_bin:stop_bin]
                    if (not np.array_equal(values["bin_start_tick"], starts)
                            or not np.array_equal(values["observed_total_spikes"], truth.sum(-1))
                            or values["boundary_ready_in_ticks"].shape != (1, config["core_neurons"])):
                        raise ValueError("point diagnostic output bins, source spikes or complete boundary unit order changed")
                    observed_rate = truth / (source.unit_count * config["bin_ms"] / 1000)
                    name = f"point_arrays/{role}/seed_{seed}/{case_id}_origin_{origin_ms:g}.npz"
                    path = output / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **values, observed_field_counts=truth,
                        observed_field_rate_hz=observed_rate, source_observation_map_sha256=artifacts[map_name])
                    artifacts[name] = sha256_file(path)
                    for horizon_ms in config["horizons_ms"]:
                        bins = int(horizon_ms / config["bin_ms"])
                        rows.append(dict(role=role, training_ceiling_epochs=item[0]["common_training_budget"]["maximum_epochs"],
                            seed=seed, block=case["block"], case_id=case_id, template_id=case["template_id"],
                            origin_ms=origin_ms, origin_tick=origin_tick, horizon_ms=horizon_ms,
                            horizon_stop_tick=origin_tick + bins * native.bin_ticks, diagnostic_array=name,
                            diagnostic_array_sha256=artifacts[name], observation_map_sha256=artifacts[map_name],
                            checkpoint_sha256=checkpoints[role + "_" + str(seed)]["checkpoint_sha256"],
                            future_observation_assisted=True, diagnostic_only=True,
                            **_point_summary(values, observed_rate, source.sd, bins)))
    if len(rows) != 864 or len(artifacts) != 289:
        raise ValueError("complete two-role three-seed 48-origin point diagnostic denominator required")
    table = "point_transition_by_condition.csv"
    _write_rows(output / table, rows)
    artifacts[table] = sha256_file(output / table)
    report = dict(status="COMPLETE_FUTURE_OBSERVATION_ASSISTED_POINT_TRANSITION_DIAGNOSTIC",
        config=config, config_sha256=sha256_file(Path(config_path)), roles=config["roles"], seeds=config["training_seeds"],
        training_ceilings=config["training_ceilings"], selection_cases=16, origins_per_role_seed=48,
        diagnostic_windows=288, score_rows=864, core_neurons=100000, source_fields=128,
        source_binding=source.binding, source_observation_map_sha256=artifacts[map_name],
        input_analysis_bindings=[parent[4], continuation[4]], checkpoint_bindings=checkpoints,
        scope="future_observation_assisted oracle debug; posterior and one-step latent means, not integrated predictive laws",
        point_rate_units="spikes_per_second_per_neuron; source-defined field mean",
        exposure_probability_sum_interpretation="observed eligible ticks times point tick probability; not a generative expected count",
        residual_variance_interpretation="first-order diagonal propagation plus posterior/process variance; not an exact marginal prior variance",
        future_observation_assisted=True, latent_evaluation="means_not_integrated", diagnostic_only=True,
        scientific_claim_qualified=False, source_fidelity_qualified=False, formal_test_accessed=False,
        samples_generated=0, source_simulation_performed=False, training_performed=False,
        state_selection_enabled=False, confidence_interval_or_significance=False, artifacts=artifacts)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    for name in ("acquisition", "fields", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if len(args.analysis) != 2:
        parser.error("point diagnostic requires exactly two complete recurrent population analyses")
    report = diagnose_population_campaign(args.analysis, args.acquisition, args.fields, args.config, args.output)
    print(json.dumps({key: report[key] for key in ("status", "diagnostic_windows", "score_rows")}))


if __name__ == "__main__":
    main()
