"""Shared factual forecasting and independent all-unit replay for source cohorts.

Source entry modules own checkpoint admission and forecast protocols. Numeric
kernels retain historical identifiers for delivered-recipe compatibility.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .circuit_data import _control_bins
from .population_state import ControlledPopulationState
from .population_trajectory_evaluation import _draw, _evaluation_hardware, _peak_process_rss_bytes, _synchronize
from .population_forecast_analysis import COST_METRICS, validate_evaluation_costs
from .provenance import freeze_json, sha256_file
from .trajectory_sample_metrics import unit_count_metrics
from .trajectory_scores import energy_score


def evaluator_sources(entry: Path) -> dict:
    """Bind the actual execution and scoring implementations, including shared code."""
    names = (entry.name, Path(__file__).name, "population_trajectory_evaluation.py",
             "trajectory_sample_metrics.py", "trajectory_scores.py", "population_forecast_replay.py")
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def restore_source_model(checkpoint, run, cohort, device, *, error_prefix):
    """Authenticate the source cohort, fitted implementation and every source buffer."""
    config = checkpoint["binding"]["config"]
    for name, value in cohort.binding.items():
        if name != "circuit_inputs_sha256" and checkpoint["binding"].get(name) != value:
            raise ValueError(error_prefix + "_EVALUATION_SOURCE_COHORT_CHANGED:" + name)
    for name, digest in checkpoint["trainer_sources"].items():
        path = (Path(__file__).resolve().parents[1] / "configs" / name if name.endswith(".yaml")
                else Path(__file__).with_name(name))
        if sha256_file(path) != digest:
            raise ValueError(error_prefix + "_EVALUATION_FITTED_IMPLEMENTATION_CHANGED:" + name)
    constructor = cohort.population_configuration(config, checkpoint["binding"]["seed"])
    if constructor != checkpoint["model_configuration"]:
        raise ValueError(error_prefix + "_EVALUATION_CANONICAL_SOURCE_MAP_CHANGED")
    model = ControlledPopulationState(**constructor)
    for name, expected in model.named_buffers():
        if not torch.equal(checkpoint["model"][name], expected):
            raise ValueError(error_prefix + "_EVALUATION_SOURCE_BUFFER_CHANGED:" + name)
    model = model.to(dtype=checkpoint["model"]["posterior_mean.weight"].dtype)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model

def validate_config(value, *, schema, evaluation_scope):
    if (value.get("schema") != schema
            or value.get("formal_evaluation") is not False
            or value.get("evaluation_scope") != evaluation_scope
            or value.get("field_units") != "source_field_mean_unit_rate_hz"
            or value.get("causal_history") != "all_observed_bins_before_origin"
            or value.get("finite_ensemble_corrections") is not True):
        raise ValueError("EXPLICIT_PFC_CHECKPOINT_FORECAST_PROTOCOL_REQUIRED")
    for name in ("origins_ticks", "horizons_bins", "support_windows"):
        items = value.get(name)
        if (not isinstance(items, list) or not items or items != sorted(set(items))
                or any(type(x) is not int or x < 1 for x in items)):
            raise ValueError("PFC_FORECAST_POSITIVE_UNIQUE_TIMES_REQUIRED:" + name)
    if type(value.get("joint_samples")) is not int or value["joint_samples"] < 2:
        raise ValueError("PFC_FORECAST_MULTIPLE_JOINT_DRAWS_REQUIRED")
    if type(value.get("sampling_seed_base")) is not int or value["sampling_seed_base"] < 0:
        raise ValueError("PFC_FORECAST_EXPLICIT_SAMPLING_SEED_REQUIRED")
    if value.get("random_seed_policy")=="exact_registered_stream_seeds":
        if value.get("sampling_seed_base")!=41:
            raise ValueError("PFC_FORECAST_STREAM_SEEDS_MUST_BE_41_42_43")
    return value


def source_controls(data, start_tick: int, stop_tick: int):
    """Read authenticated source commands without reading spike outcomes.

    Preserve source segment boundaries and declared command features. Bin
    values are time averages of planned commands. Realized drive events and
    microscopic outcomes are excluded. Source-specific intervention admission
    remains in the source entry module.
    """
    if (type(start_tick) is not int or type(stop_tick) is not int
            or not 0 <= start_tick < stop_tick <= data.duration_ticks
            or start_tick % data.bin_ticks or stop_tick % data.bin_ticks):
        raise ValueError("PFC_RESPONSE_COMPLETE_ALIGNED_CONTROL_WINDOW_REQUIRED")
    row = data.data.episodes[data.episode_id]
    shard = data.data.shards[row["shard"]]
    with h5py.File(data.data._verified(shard["path"], shard["sha256"]), "r") as archive:
        group = archive[data.episode_id]
        metadata = json.loads(group.attrs["metadata_json"])
        if any(metadata[name] != row[name] for name in
               ("episode_id", "circuit_id", "block_id", "split", "duration_ticks", "dt_ms")):
            raise ValueError("PFC_RESPONSE_CONTROL_EPISODE_IDENTITY_CHANGED")
        if metadata["control_kind"] != "piecewise_constant":
            raise ValueError("PFC_RESPONSE_SOURCE_CURRENT_COMMANDS_REQUIRED")
        ticks, values = group["control_ticks"][:], group["control_values"][:]
    stops = np.r_[ticks[1:], data.duration_ticks]
    selected = (ticks < stop_tick) & (stops > start_tick)
    return dict(known_controls=_control_bins(ticks, values, "piecewise_constant",
                                             start_tick, stop_tick, data.bin_ticks),
                control_features=data.model_control_features,
                source_control_features=metadata["control_features"],
                segment_start_ticks=ticks[selected], segment_stop_ticks=stops[selected],
                segment_values=values[selected], start_tick=start_tick, stop_tick_exclusive=stop_tick,
                control_kind="piecewise_constant", source_binding=data.binding)


@torch.no_grad()
def prepare_factual_response(view, data, origin: int, horizon_bins: int):
    """Infer a causal boundary and bind its scheduled future source controls.

    Complete source admission can authenticate future outcomes. This prediction
    preparation reads only the observed prefix and the known current program.
    Independent source execution is still required for intervention evidence.
    """
    if (type(origin) is not int or type(horizon_bins) is not int or origin < 1 or horizon_bins < 1
            or origin % data.bin_ticks or origin + horizon_bins * data.bin_ticks > data.duration_ticks):
        raise ValueError("PFC_FORECAST_COMPLETE_ALIGNED_PREFIX_AND_FUTURE_REQUIRED")
    if (tuple(map(int, data.unit_ids)) != view.unit_ids or data.field_names != view.field_names
            or not np.array_equal(data.unit_field_index, view.unit_field_index.cpu().numpy())
            or not np.array_equal(data.minimum_interval_ticks, view.minimum_interval_ticks.cpu().numpy())
            or data.dt_ms / 1000 != view.tick_seconds or data.bin_ticks != view.bin_ticks
            or view.metadata is None
            or view.metadata["source_identity"]["circuit_id"] != data.binding["circuit_id"]):
        raise ValueError("PFC_RESPONSE_MODEL_SOURCE_MAP_OR_CLOCK_CHANGED")
    counts, control = data.window(0, origin)
    scheduled_prefix = source_controls(data, 0, origin)
    future = source_controls(data, origin, origin + horizon_bins * data.bin_ticks)
    if (scheduled_prefix["control_features"] != view.metadata["control_features"]
            or future["control_features"] != view.metadata["control_features"]
            or not np.array_equal(control, scheduled_prefix["known_controls"])):
        raise ValueError("PFC_RESPONSE_SOURCE_CONTROL_SEMANTICS_CHANGED")
    if (counts.shape != (origin // data.bin_ticks, len(view.unit_ids))
            or control.shape != (origin // data.bin_ticks, view.input_size)
            or future["known_controls"].shape != (horizon_bins, view.input_size)):
        raise ValueError("PFC_RESPONSE_COMPLETE_OBSERVATION_AND_CONTROL_DENOMINATOR_REQUIRED")
    boundary = view.infer_state(dict(counts=counts[None], stimulus=control[None],
                                    ready_in_ticks=data.ready_at(origin)[None]))
    return boundary, future


@torch.no_grad()
def _forecast(canonical, cohort, case, origin, config, seed_base, device, *, prepared=None, costs=None):
    if prepared is None:
        data = cohort._episode(case)
        cohort.active_case, cohort.active_episode = case, data
        view = cohort.model_for_case(canonical, case).eval()
    else:
        view, data = prepared
        cohort.active_case, cohort.active_episode = case, data
    if costs is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device); start = time.perf_counter()
    boundary, future = prepare_factual_response(view, data, origin, max(config["horizons_bins"]))
    _synchronize(device); inference_seconds = time.perf_counter() - start
    start = time.perf_counter()
    generated, latent, initial, terminal = _draw(view, "population", boundary, future["known_controls"][None], seed_base,
                                                config["joint_samples"], device)
    _synchronize(device); generation_seconds = time.perf_counter() - start
    start = time.perf_counter()
    truth, observed_controls = data.window(origin, origin + max(config["horizons_bins"]) * cohort.bin_ticks)
    if not np.array_equal(observed_controls, future["known_controls"]):
        raise ValueError("PFC_FORECAST_FACTUAL_CONTROL_READS_DIFFER")
    # Fixed source-ordered reductions also work on CUDA without scatter atomics.
    fields = torch.stack([generated.index_select(-1, getattr(view, f"field_units_{f}")).sum(-1)
                          for f in range(len(view.field_names))], -1).cpu().numpy()
    observed = np.column_stack([truth[:, view.unit_field_index.cpu().numpy() == f].sum(-1)
                                for f in range(len(view.field_names))])
    _synchronize(device)
    if costs is not None:
        costs.update(inference_seconds=inference_seconds, full_unit_generation_seconds=generation_seconds,
                     field_projection_seconds=time.perf_counter() - start,
                     peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    return view, data, truth, generated, fields, observed, latent, initial, terminal


def _scores(view, truth, generated, fields, observed, config, *, costs=None):
    sizes = view.field_unit_counts.cpu().numpy()
    present = sizes > 0
    if not present.any():
        raise ValueError("PFC_FORECAST_NO_OBSERVED_SOURCE_FIELD")
    rates = fields[..., present] / (sizes[present] * view.bin_seconds)
    actual = observed[:, present] / (sizes[present] * view.bin_seconds)
    if costs is not None:
        costs["field_metric_dtype"] = str(rates.dtype)
    rows = []
    for horizon in config["horizons_bins"]:
        metrics = unit_count_metrics(generated[:, :horizon], torch.as_tensor(truth[:horizon],device=generated.device),
                                    view.minimum_interval_ticks, view.bin_ticks, config["support_windows"])
        rows.append(dict(horizon_bins=horizon, field_energy_score_hz=energy_score(rates[:, :horizon], actual[:horizon]),
                         **metrics))
    return rows


def _case_cost_contract(case, view, data):
    return dict(block=data.binding["block_id"], unit_count=len(view.unit_ids), field_count=len(view.field_names),
                bin_ms=data.dt_ms * data.bin_ticks, source_dt_ms=data.dt_ms,
                episode_bins=data.duration_ticks // data.bin_ticks, full_unit_retention="all_origins",
                episode_inputs_materialized=False)


def validate_forecast_costs(report, rows, *, validation, case_cost_contracts):
    """Normalize PFC clocks and source blocks for the shared measured-cost validator.

    Source contracts come from admitted episodes and model maps, not report fields.
    Legacy reports retain unmeasured costs. The PFC training contract requires a
    common native clock and bin across its distinct source circuits.
    """
    if not any(name in report for name in ("origin_costs", "episode_costs")):
        return None
    clocks = {(c["source_dt_ms"], c["bin_ms"]) for c in case_cost_contracts.values()}
    if len(clocks) != 1:
        raise ValueError("PFC_COST_COMMON_SOURCE_CLOCK_REQUIRED")
    dt_ms, bin_ms = next(iter(clocks))
    config = dict(report["configuration"], origins_ms=[v * dt_ms for v in report["configuration"]["origins_ticks"]],
                  horizons_ms=[v * bin_ms for v in report["configuration"]["horizons_bins"]], field_metric_dtype="float64")
    normalized = []
    for row in rows:
        value = dict(row, origin_ms=float(row["origin_tick"]) * dt_ms,
                     horizon_ms=float(row["horizon_bins"]) * bin_ms)
        for name in COST_METRICS:
            raw = row.get(name)
            value[name] = None if raw is None or raw == "" else raw if isinstance(raw, bool) else float(raw)
        normalized.append(value)
    return validate_evaluation_costs(dict(report, origins=len(report["origins"])), normalized, config,
        validation=[dict(case, block=case_cost_contracts[case["case_id"]]["block"]) for case in validation],
        unit_count=None, field_count=None, bin_ms=None, source_dt_ms=None, episode_bin_counts=None,
        case_cost_contracts=case_cost_contracts)



def _prepare_case(model, cohort, case):
    data = cohort._episode(case)
    cohort.active_case, cohort.active_episode = case, data
    return cohort.model_for_case(model, case).eval(), data


def _forecast_arrays(view, truth, generated, fields, observed, latent, initial, terminal):
    """Retain actual model outputs; count-only generators have no renewal state."""
    if (initial is None) != (terminal is None):
        raise ValueError("SOURCE_FORECAST_INITIAL_AND_TERMINAL_STATE_PAIR_REQUIRED")
    values = dict(generated_unit_counts=generated.cpu().numpy(), observed_unit_counts=truth,
        generated_field_counts=fields, observed_field_counts=observed,
        unit_ids=np.asarray(view.unit_ids), unit_field_index=view.unit_field_index.cpu().numpy(),
        field_names=np.asarray(view.field_names), latent_states=latent.cpu().numpy())
    if initial is not None:
        values.update(initial_latent_states=initial.cpu().numpy(),
                      terminal_ready_in_ticks=terminal.cpu().numpy())
    return values


def evaluate(inputs: Path, training: Path, config_path: Path, output: Path, device_name="cpu",
             *, load, read_config, evaluator_path: Path, evaluation_status: str,
             forecast=None, prepare_case=None):
    if output.exists():
        raise FileExistsError("PFC_FORECAST_REQUIRES_NEW_OUTPUT")
    function_started = time.perf_counter()
    forecast = _forecast if forecast is None else forecast
    prepare_case = _prepare_case if prepare_case is None else prepare_case
    config = read_config(config_path)
    device = torch.device(device_name)
    torch.set_num_threads(1)
    model, cohort, run = load(inputs, training, device)
    output.mkdir(parents=True)
    report = dict(status=evaluation_status, configuration=config,
        inputs_sha256=sha256_file(inputs), original_training_inputs_sha256=run["binding"]["circuit_inputs_sha256"],
        checkpoint_sha256=run["best_checkpoint_sha256"], training_status=run["status"],
        training_seed=run["training_seed"],
        **(run["evaluation_training_metadata"] if "evaluation_training_metadata" in run else
           dict(committed_training_epochs=run["completed_epochs"])),
        field_units=config["field_units"],
        empty_field_policy="retained_in_count_artifact; undefined_per_unit_rate_excluded_from_energy_score",
        hardware=_evaluation_hardware(model, device), evaluator_sha256=sha256_file(evaluator_path),
        evaluator_sources=evaluator_sources(evaluator_path),
        cost_scope=dict(
            function_total_seconds="function entry through score hash, report assembly and cost validation; final evaluation.json write excluded",
            excluded_costs=["process imports", "launcher input/software staging", "launcher archival", "training"],
            elapsed_seconds="evaluation loop including episode preparation, all origins and NPZ/hash/CSV writes; setup and final score hash excluded",
            setup_seconds="function entry through configuration, checkpoint/source validation, model loading and report setup",
            episode_costs="source adapter and model map loaded once per episode; complete input tensors not materialized; prefix/control reads and transfers charged at each origin",
            origin_total_seconds="fresh observed-prefix inference, full-unit generation, truth read/projection, scoring, all-unit NPZ/hash/CSV writes and cleanup",
            horizon_reuse="one longest-horizon ensemble per origin; all horizon scores share this origin cost",
            peak_process_rss_bytes="cumulative process-lifetime resident-memory high-water mark, including imports and prior work",
            speedup_qualified=False),
        scientific_claim_qualified=False, formal_evaluation=False, speedup_qualified=False)
    rows, artifacts, origins, episode_costs, origin_costs, contracts = [], {}, [], [], [], {}
    _synchronize(device); started = time.perf_counter()
    report["setup_seconds"] = started - function_started
    for index, case in enumerate(cohort.validation):
        episode_started = time.perf_counter()
        view, data = prepare_case(model, cohort, case)
        _synchronize(device); loading_seconds = time.perf_counter() - episode_started
        contract = _case_cost_contract(case, view, data)
        contracts[case["case_id"]] = contract
        episode_costs.append(dict(case_id=case["case_id"], block=contract["block"],
            native_duration_ticks=data.duration_ticks, native_bins=contract["episode_bins"], unit_count=len(view.unit_ids),
            origins_ms=[v * data.dt_ms for v in config["origins_ticks"]], episode_inputs_materialized=False,
            count_dtype=None, stimulus_dtype=None, count_tensor_dtype=None, stimulus_tensor_dtype=None,
            count_tensor_device=None, stimulus_tensor_device=None, loading_seconds=loading_seconds,
            window_seconds=0., transfer_seconds=0., episode_preparation_seconds=time.perf_counter() - episode_started))
        for origin_index, origin in enumerate(config["origins_ticks"]):
            origin_started = time.perf_counter()
            seed_base = (41 if config.get("random_seed_policy")=="exact_registered_stream_seeds" else
                         config["sampling_seed_base"] + run["training_seed"] * 100000 + index * 10000 + origin_index * 3)
            cost = {}
            view, data, truth, generated, fields, observed, latent, initial, terminal = forecast(
                model, cohort, case, origin, config, seed_base, device, prepared=(view, data), costs=cost)
            start = time.perf_counter()
            scored = _scores(view, truth, generated, fields, observed, config, costs=cost)
            first_row = len(rows)
            rows.extend(dict(case_id=case["case_id"], circuit_id=case["circuit_id"],
                             origin_tick=origin, training_seed=run["training_seed"], **row,
                             **{name: cost[name] for name in COST_METRICS}) for row in scored)
            _synchronize(device); cost["scoring_seconds"] = time.perf_counter() - start
            name = f"{case['case_id']}_origin_{origin}.npz"
            start = time.perf_counter()
            np.savez_compressed(output / name, **_forecast_arrays(
                view, truth, generated, fields, observed, latent, initial, terminal))
            cost["serialization_seconds"] = time.perf_counter() - start
            start = time.perf_counter(); artifacts[name] = sha256_file(output / name)
            cost["hash_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            with (output / "scores.csv").open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                if first_row == 0: writer.writeheader()
                writer.writerows(rows[first_row:])
            cost["csv_seconds"] = time.perf_counter() - start
            cost.update(case_id=case["case_id"], block=contract["block"], origin_ms=origin * data.dt_ms,
                horizons_ms=[v * contract["bin_ms"] for v in config["horizons_bins"]],
                generated_horizon_ms=max(config["horizons_bins"]) * contract["bin_ms"],
                generated_bins=max(config["horizons_bins"]), draws=config["joint_samples"],
                unit_count=len(view.unit_ids), field_count=len(view.field_names), score_rows=len(scored),
                generated_unit_count_values=generated.numel(), generated_unit_count_dtype=str(generated.dtype),
                generated_unit_count_device=str(generated.device), latent_dtype=str(latent.dtype),
                latent_device=str(latent.device), field_count_dtype=str(fields.dtype),
                stored_full_unit_counts=True)
            origins.append(dict(case_id=case["case_id"], origin_tick=origin, seed_base=seed_base,
                source_binding=data.binding, artifact=name, units=len(view.unit_ids), fields=len(view.field_names),
                source_metadata_identity=view.source_metadata_identity))
            del generated, fields, observed, truth, latent, initial, terminal
            cost["origin_total_seconds"] = time.perf_counter() - origin_started
            cost["peak_process_rss_bytes"] = _peak_process_rss_bytes()
            origins[-1]["seconds"] = cost["origin_total_seconds"]
            origin_costs.append(cost)
        del view, data
    report.update(origins=origins, score_rows=len(rows), episodes=len(episode_costs),
                  artifacts=artifacts, episode_costs=episode_costs, origin_costs=origin_costs,
                  generated_unit_count_values=sum(v["generated_unit_count_values"] for v in origin_costs),
                  elapsed_seconds=time.perf_counter() - started)
    start = time.perf_counter(); artifacts["scores.csv"] = sha256_file(output / "scores.csv")
    report["score_hash_seconds"] = time.perf_counter() - start
    report["peak_process_rss_bytes"] = _peak_process_rss_bytes()
    report["function_total_seconds"] = time.perf_counter() - function_started
    validate_forecast_costs(report, rows, validation=cohort.validation, case_cost_contracts=contracts)
    report["function_total_seconds"] = time.perf_counter() - function_started
    freeze_json(output / "evaluation.json", report)
    return report


def replay(inputs: Path, training: Path, forecasts: Path, output: Path, device_name="cpu",
           *, load, evaluator_path: Path, evaluation_status: str, forecast=None):
    from .population_forecast_replay import numpy_unit_metrics, _project_units, METRIC_RTOL, METRIC_ATOL
    forecast = _forecast if forecast is None else forecast
    if Path(output).exists():
        raise FileExistsError("existing replay evidence will not be overwritten")
    report = json.loads((forecasts / "evaluation.json").read_text())
    if (report["status"] != evaluation_status
            or report["evaluator_sha256"] != sha256_file(evaluator_path)
            or report.get("evaluator_sources") != evaluator_sources(evaluator_path)):
        raise ValueError("PFC_REPLAY_EVALUATOR_OR_STATUS_CHANGED")
    for name, digest in report["artifacts"].items():
        if sha256_file(forecasts / name) != digest:
            raise ValueError("PFC_REPLAY_FORECAST_ARTIFACT_CHANGED:" + name)
    device = torch.device(device_name)
    torch.set_num_threads(1)
    model, cohort, run = load(inputs, training, device)
    if run["best_checkpoint_sha256"] != report["checkpoint_sha256"]:
        raise ValueError("PFC_REPLAY_CHECKPOINT_CHANGED")
    with (forecasts / "scores.csv").open() as stream:
        scores = list(csv.DictReader(stream))
    expected = [(c["case_id"], origin) for c in cohort.validation for origin in report["configuration"]["origins_ticks"]]
    if [(r["case_id"], r["origin_tick"]) for r in report["origins"]] != expected:
        raise ValueError("PFC_REPLAY_ORIGIN_DENOMINATOR_CHANGED")
    checked, consumed, contracts = [], 0, {}
    if len(scores) != len(expected) * len(report["configuration"]["horizons_bins"]):
        raise ValueError("PFC_REPLAY_COMPLETE_SCORE_DENOMINATOR_CHANGED")
    for origin in report["origins"]:
        case = next(c for c in cohort.validation if c["case_id"] == origin["case_id"])
        view, data, truth, generated, fields, observed, latent, initial, terminal = forecast(
            model, cohort, case, origin["origin_tick"], report["configuration"], origin["seed_base"], device)
        contracts[case["case_id"]] = _case_cost_contract(case, view, data)
        if origin["units"] != len(view.unit_ids) or origin["fields"] != len(view.field_names):
            raise ValueError("PFC_REPLAY_SOURCE_MAP_SIZE_CHANGED")
        actual = _forecast_arrays(view, truth, generated, fields, observed, latent, initial, terminal)
        values = actual["generated_unit_counts"]
        with np.load(forecasts / origin["artifact"], allow_pickle=False) as stored:
            if set(stored.files) != set(actual) or any(not np.array_equal(stored[k],v) for k,v in actual.items()):
                raise ValueError("PFC_REPLAY_ALL_UNIT_OR_SOURCE_MAP_MISMATCH")
        independent, reference = _project_units(values,truth,view.unit_field_index.cpu().numpy(),len(view.field_names))
        if not np.array_equal(independent,fields) or not np.array_equal(reference,observed):
            raise ValueError("PFC_REPLAY_INDEPENDENT_FIELD_PROJECTION_MISMATCH")
        metrics, _, used = numpy_unit_metrics(values,truth,view.minimum_interval_ticks.cpu().numpy(),view.bin_ticks,
            report["configuration"]["horizons_bins"],report["configuration"]["support_windows"])
        selected = [r for r in scores if r["case_id"] == case["case_id"] and int(r["origin_tick"]) == origin["origin_tick"]]
        if [int(r["horizon_bins"]) for r in selected] != report["configuration"]["horizons_bins"]:
            raise ValueError("PFC_REPLAY_SCORE_DENOMINATOR_CHANGED")
        for row in selected:
            for key,value in metrics[int(row["horizon_bins"])].items():
                if value is None:
                    if row[key] != "": raise ValueError("PFC_REPLAY_UNDEFINED_SUPPORT_METRIC_CHANGED")
                elif not np.isclose(float(row[key]),value,rtol=METRIC_RTOL,atol=METRIC_ATOL):
                    raise ValueError("PFC_REPLAY_INDEPENDENT_UNIT_METRIC_MISMATCH:"+key)
            horizon = int(row["horizon_bins"])
            sizes = view.field_unit_counts.cpu().numpy(); present = sizes > 0
            draws = independent[:, :horizon, present] / (sizes[present] * view.bin_seconds)
            target = reference[:horizon, present] / (sizes[present] * view.bin_seconds)
            if not np.isclose(float(row["field_energy_score_hz"]),energy_score(draws,target),rtol=0,atol=1e-12):
                raise ValueError("PFC_REPLAY_FIELD_SCORE_MISMATCH")
        if data.binding != origin["source_binding"] or view.source_metadata_identity != origin["source_metadata_identity"]:
            raise ValueError("PFC_REPLAY_SOURCE_BINDING_CHANGED")
        consumed += used; checked.append(dict(case_id=case["case_id"],origin_tick=origin["origin_tick"],unit_values=used))
    validate_forecast_costs(report, scores, validation=cohort.validation, case_cost_contracts=contracts)
    receipt=dict(status="PASS_ALL_UNIT_AND_INDEPENDENT_METRIC_REPLAY",origins=checked,
        unit_values_consumed=consumed,checkpoint_sha256=report["checkpoint_sha256"],scientific_claim_qualified=False)
    freeze_json(output,receipt)
    return receipt


