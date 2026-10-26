"""Deploy one registered development seed from accepted source-bound inputs.

Run only inside the companion Slurm allocation. Fit, evaluation and exhaustive
replay use fresh child processes and one accepted local software snapshot.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import yaml

SEEDS = (41, 42, 43)
SOFTWARE = "Output/neural_population_dynamics/model/trajectory_software/run_606023"
EPISODE = "Output/neural_population_dynamics/model/full_episode_check/run_606074"
COHORT = "Output/neural_population_dynamics/mechanistic_reference/spatial_connectivity/stimulus_trajectory_cohort/philox_development"
CONFIGS = ("neural_population_state_trajectory.yaml", "neural_population_matched_forecast.yaml")
PFC_CONFIGS = ("neural_population_pfc_task_training.yaml", "neural_population_pfc_task_forecast.yaml")
PFC_CAMPAIGN = "Output/neural_population_dynamics/model/pfc_task_development"
V1_CONFIGS = ("neural_population_v1_source_training.yaml", "neural_population_v1_forecast.yaml")
V1_CAMPAIGN = "Output/neural_population_dynamics/model/v1_native_observation"
COST_POINT_CASE = "train_block_014_familiar_0"
ACCEPTED_SEED41 = "Output/neural_population_dynamics/model/trajectory_development/run_606139/seed_41"


class ResourceBlocked(ValueError):
    """Retain an admission failure as resource evidence before child execution."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__(",".join(self.reasons))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def manifest_rows(receipt):
    rows = [dict(row, bytes=row.get("bytes", row.get("destination_bytes")))
            for row in receipt["files"]]
    if any(type(row["bytes"]) is not int or row["bytes"] < 0 for row in rows):
        raise ValueError("explicit source manifest byte lengths required")
    index = {row["path"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate source manifest paths")
    return index


def copy_bound(source, target, row):
    if (source.stat().st_size != row["bytes"] or digest(source) != row["sha256"]):
        raise ValueError("source manifest identity changed: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
        raise ValueError("staged copy differs: " + str(target))


def prerequisite(repo, relative, job):
    root = repo / relative
    receipt = json.loads((root / "delivery.json").read_text())
    if receipt["status"] != "PASS" or receipt["exit_code"] != 0 or receipt["job_id"] != job:
        raise ValueError("accepted prerequisite receipt required: " + relative)
    return root, receipt, manifest_rows(receipt)


def software_closure(repo, modules):
    """Freeze the shared entry points and their package-relative dependencies."""
    pending, sources = list(modules), set()
    while pending:
        name = pending.pop(0)
        if name in sources:
            continue
        path = repo / "Libs/population_dynamics" / (name + ".py")
        tree = ast.parse(path.read_text())
        sources.add(name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                pending.extend([node.module] if node.module else [item.name for item in node.names])
    return ["Libs/__init__.py", "Libs/population_dynamics/__init__.py"] + [
        "Libs/population_dynamics/" + name + ".py" for name in sorted(sources)]


def input_copy(source, target, sha256, size=None):
    """Describe one already declared input; authenticate payloads during staging."""
    target = Path(target)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError("PFC_INPUT_RELATIVE_STAGING_PATH_REQUIRED")
    source = Path(source)
    return dict(source=str(source), path=str(target),
                bytes=source.stat().st_size if size is None else size, sha256=sha256)


def pfc_output_storage_bytes(validation_episodes, forecast, latent_channels, encoder_size):
    """Conservative uncompressed all-origin arrays, checkpoint space and reserve."""
    origins = validation_episodes * len(forecast["origins_ticks"])
    draws, bins, units, fields = forecast["joint_samples"], max(forecast["horizons_bins"]), 100000, 17
    # int64 counts/eligibility/maps and float64 latent upper bounds.
    per_origin = 8 * ((draws + 1) * bins * (units + fields)
        + draws * units + 2 * units + draws * (bins + 1) * latent_channels * encoder_size)
    return origins * per_origin + 1024**3 + 20 * 1024**3


def pfc_input_inventory(inputs, spec, source_entry_sha256):
    """Resolve only declared catalog/source files without reading payloads."""
    SOURCE_ENTRY_SHA256 = source_entry_sha256
    copies = [input_copy(inputs, "inputs/cohort.json", digest(inputs))]
    source_entry = Path(spec["source_entry"]).resolve()
    if digest(source_entry) != SOURCE_ENTRY_SHA256:
        raise ValueError("PFC_PINNED_SOURCE_ENTRY_CHANGED")
    entry_target = "inputs/source_entry/wm_model.py"
    copies.append(input_copy(source_entry, entry_target, SOURCE_ENTRY_SHA256))
    paths = {str(source_entry): dict(path=entry_target, sha256=SOURCE_ENTRY_SHA256)}
    for i, episode in enumerate(spec["episodes"]):
        dataset = Path(episode["dataset"]).resolve()
        source = Path(episode["source_run"]).resolve()
        source = source if source.name == "source_run" else source / "source_run"
        catalog_path, delivery_path = dataset / "dataset.json", source.parent / "delivery.json"
        catalog = json.loads(catalog_path.read_text())
        if (catalog["complete_episodes"] != 1 or len(catalog["shards"]) != 1
                or len(catalog["circuits"]) != 1
                or next(iter(catalog["circuits"].values()))["units"] != 100000):
            raise ValueError("PFC_STAGING_COMPLETE_100000_UNIT_EPISODE_REQUIRED")
        selected = catalog["shards"][0]["episodes"]
        if (len(selected) != 1 or selected[0]["episode_id"] != episode["episode_id"]
                or selected[0]["duration_ticks"] != 100000 or selected[0]["dt_ms"] != .05):
            raise ValueError("PFC_STAGING_EPISODE_IDENTITY_OR_CLOCK_CHANGED")
        dataset_target = f"inputs/datasets/{i:03d}"
        source_target = f"inputs/sources/{i:03d}"
        catalog_sha, delivery_sha = digest(catalog_path), digest(delivery_path)
        paths[str(dataset)] = dict(path=dataset_target, sha256=catalog_sha)
        paths[str(source)] = dict(path=source_target + "/source_run", sha256=delivery_sha)
        copies.append(input_copy(catalog_path, dataset_target + "/dataset.json", catalog_sha))
        for row in [*catalog["circuits"].values(), *catalog["shards"]]:
            copies.append(input_copy(dataset / row["path"], dataset_target + "/" + row["path"], row["sha256"]))
        proof = json.loads(delivery_path.read_text())
        if (proof.get("source_execution_complete") is not True or proof.get("exit_code") != 0
                or proof.get("case_id") != episode["episode_id"]):
            raise ValueError("PFC_STAGING_SUCCESSFUL_SOURCE_REQUIRED")
        index = manifest_rows(proof)
        report = json.loads((source / "report.json").read_text())
        reference = next(row for row in catalog["binding"]["source_bindings"]
                         if row["case_id"] == episode["episode_id"])
        if (reference["delivery_sha256"] != delivery_sha
                or reference["report_sha256"] != digest(source / "report.json")):
            raise ValueError("PFC_STAGING_SOURCE_BINDING_CHANGED")
        names = ["source_run/report.json", *["source_run/" + name for name in report["outputs"]]]
        names += ["Libs/population_dynamics/" + name for name in (
            "pfc_source_execution.py", "pfc_deterministic_replay.py", "pfc_native_drive.py")]
        if "kernel_parameterization_sha256" in report["code"]:
            names.append("Libs/population_dynamics/pfc_kernel_parameters.py")
        names += ["qualification/delivery.json", "qualification/comparison/qualification.json"]
        copies.append(input_copy(delivery_path, source_target + "/delivery.json", delivery_sha))
        for name in names:
            row = index[name]
            copies.append(input_copy(source.parent / name, source_target + "/" + name,
                                     row["sha256"], row["bytes"]))
    return copies, paths


def prepare_pfc(repo, inputs, coverage_path, output, *, training_config=None, forecast_config=None):
    """Freeze a complete authenticated development cohort; never launch training."""
    if output.exists():
        raise FileExistsError("PFC_PREPARATION_REQUIRES_NEW_OUTPUT")
    # Use the maintained intake, including its original plan/receipt checks.
    sys.path.insert(0, str(repo / "Libs"))
    from population_dynamics.pfc_training_inputs import assemble
    from population_dynamics.pfc_population_training_data import validate_config
    from population_dynamics.pfc_population_evaluation import _config
    from population_dynamics.pfc_scale_analysis import SOURCE_ENTRY_SHA256
    coverage = json.loads(coverage_path.read_text())
    spec, current = assemble(Path(coverage["baseline_inputs"]), Path(coverage["baseline_validation"]),
        Path(coverage["plan"]).parent, Path(coverage["plan"]), coverage["reader_jobs"])
    saved_input = coverage.pop("inputs", None)
    saved_digest = coverage.pop("inputs_sha256", None)
    if (saved_input is None or Path(saved_input).resolve() != inputs.resolve()
            or saved_digest != digest(inputs)):
        raise ValueError("PFC_COVERAGE_OUTPUT_INPUT_BINDING_CHANGED")
    if (current != coverage or spec != json.loads(inputs.read_text())
            or current["fit_validation_coverage_ready"] is not True
            or (current["fitting_episodes"], current["validation_episodes"]) != (80, 30)):
        raise ValueError("PFC_COMPLETE_AUTHENTICATED_80_FIT_30_VALIDATION_REQUIRED")
    configs=(training_config.name if training_config else PFC_CONFIGS[0],
             forecast_config.name if forecast_config else PFC_CONFIGS[1])
    for path in (training_config,forecast_config):
        if path is not None and path.resolve().parent != (repo/"Libs/configs").resolve():
            raise ValueError("PFC_CANONICAL_CONFIG_DIRECTORY_REQUIRED")
    config = yaml.safe_load((repo / "Libs/configs" / configs[0]).read_text())
    for seed in SEEDS:
        validate_config(config, seed)
    forecast = _config(repo / "Libs/configs" / configs[1])
    if (forecast["origins_ticks"] != [2000, 8000, 48000, 80000]
            or forecast["horizons_bins"] != [4, 16, 100] or forecast["joint_samples"] != 16):
        raise ValueError("PFC_REGISTERED_FULL_DEVELOPMENT_FORECAST_REQUIRED")
    copies, paths = pfc_input_inventory(inputs, spec, SOURCE_ENTRY_SHA256)
    # The trainer hashes modules that are not necessarily imported eagerly.
    modules = ["population_trajectory_training", "population_trajectory_evaluation",
        "population_forecast_replay", "pfc_population_evaluation", "pfc_training_inputs", "resource_probe",
        "poisson_trajectory_training", "poisson_state_space", "native_count_windows", "native_trajectory_cohort",
        "renewal_trajectory_training", "population_state", "renewal_state_space", "refractory_observation",
        "native_renewal_data", "pfc_population_training_data", "pfc_renewal_data", "circuit_data",
        "pfc_scale_analysis", "pfc_source_dataset", "pfc_kernel_parameters",
        "pfc_recurrent_gain_comparison", "benchmark_tasks"]
    declared = software_closure(repo, modules) + ["Libs/configs/" + name for name in (*configs,
        "neural_population_pfc_recurrent_gain.yaml")]
    declared += ["Libs/scripts/run_population_trajectory_campaign.py", "Libs/scripts/run_population_trajectory_campaign.sbatch"]
    files = [dict(path=name, bytes=(repo / name).stat().st_size, sha256=digest(repo / name)) for name in declared]
    if len({row["path"] for row in copies}) != len(copies):
        raise ValueError("PFC_STAGING_DUPLICATE_DESTINATION")
    output.mkdir(parents=True)
    for row in files:
        copy_bound(repo / row["path"], output / row["path"], row)
    copy_bound(coverage_path, output / "input_coverage.json",
               dict(bytes=coverage_path.stat().st_size, sha256=digest(coverage_path)))
    write_json(output / "preparation.json", dict(schema="pfc_population_campaign_preparation_v1",
        status="PREPARED_COMPLETE_DEVELOPMENT_COHORT", files=files, input_files=copies, paths=paths,
        training_config=configs[0],forecast_config=configs[1],control_geometry=config.get("control_geometry","static"),
        input_sha256=digest(inputs), input_coverage_sha256=digest(coverage_path),
        training_seeds=list(SEEDS), fitting_episodes=80, validation_episodes=30,
        source_dt_ms=.05, episode_ms=5000, core_neurons=100000,
        input_bytes=sum(row["bytes"] for row in copies),
        output_storage_bytes=pfc_output_storage_bytes(30, forecast, config["latent_channels"], config["encoder_size"]),
        training_submitted=False, scientific_claim_qualified=False,
        formal_evaluation=False, calibration_or_test_accessed=False))


def v1_output_storage_bytes(validation_cases, dimensions, forecast, latent_channels):
    """Account for every native unit and saved origin, including terminal state."""
    draws, bins = forecast["joint_samples"], max(forecast["horizons_bins"])
    total = 0
    for case in validation_cases:
        size = dimensions[case["circuit_id"]]
        units, fields = size["units"], size["fields"]
        per_origin = 8 * ((draws + 1) * bins * (units + fields)
            + draws * units + 2 * units + draws * (bins + 1) * fields * latent_channels)
        total += len(forecast["origins_ticks"]) * per_origin
    return total + 21 * 1024**3


def prepare_v1(repo, inputs, output, *, training_config=None, forecast_config=None):
    """Freeze original V1 catalog identities and selected complete source episodes."""
    if output.exists():
        raise FileExistsError("V1_PREPARATION_REQUIRES_NEW_OUTPUT")
    if os.environ.get("NEURONSIM_EXECUTION_PATH_MAP"):
        raise ValueError("V1_PREPARATION_REQUIRES_ORIGINAL_SOURCE_PATHS")
    sys.path.insert(0, str(repo / "Libs"))
    from population_dynamics.v1_population_training_data import inspect_inputs, V1PopulationTrainingCohort
    from population_dynamics.v1_population_evaluation import _config
    configs = (training_config.name if training_config else V1_CONFIGS[0],
               forecast_config.name if forecast_config else V1_CONFIGS[1])
    for path in (training_config, forecast_config):
        if path is not None and path.resolve().parent != (repo / "Libs/configs").resolve():
            raise ValueError("V1_CANONICAL_CONFIG_DIRECTORY_REQUIRED")
    config = yaml.safe_load((repo / "Libs/configs" / configs[0]).read_text())
    inspected = inspect_inputs(inputs, config)
    forecast = _config(repo / "Libs/configs" / configs[1])
    selected = set(config["fit_circuit_ids"]) | set(config["validation_circuit_ids"])
    episodes = [row for row in inspected.data.episodes.values() if row["circuit_id"] in selected]
    episode_ids = {row["episode_id"] for row in episodes}
    for row in episodes:
        if (row["duration_ticks"] % config["bin_ticks"]
                or any(origin % config["bin_ticks"] or
                       origin + max(forecast["horizons_bins"]) * config["bin_ticks"] > row["duration_ticks"]
                       for origin in forecast["origins_ticks"])):
            raise ValueError("V1_PREPARATION_COMPLETE_ALIGNED_FORECAST_REQUIRED")
    catalog_path = inspected.data.root / "dataset.json"
    copies = [input_copy(inputs, "inputs/cohort.json", hashlib.sha256(inspected.raw).hexdigest()),
              input_copy(catalog_path, "inputs/dataset/dataset.json", digest(catalog_path))]
    declared = [inspected.data.catalog["circuits"][cid] for cid in sorted(selected)]
    for shard in inspected.data.catalog["shards"]:
        if any(row["episode_id"] in episode_ids for row in shard["episodes"]):
            if any(row["episode_id"] not in episode_ids for row in shard["episodes"]):
                raise ValueError("V1_STAGING_SHARD_CONTAINS_UNSELECTED_EPISODES")
            declared.append(shard)
    for row in declared:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("V1_STAGING_RELATIVE_CATALOG_PATH_REQUIRED")
        source = (inspected.data.root / relative).resolve()
        if not source.is_relative_to(inspected.data.root.resolve()):
            raise ValueError("V1_STAGING_SOURCE_PATH_ESCAPES_DATASET")
        copies.append(input_copy(source, Path("inputs/dataset") / relative, row["sha256"]))
    # Admit one source representative per circuit. The existing adapter admits
    # every complete episode when fitting or scoring consumes it.
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("V1_SOURCE_PREPARATION_REQUIRES_COMPUTE_ALLOCATION")
    cohort = V1PopulationTrainingCohort(inputs, config)
    paths = dict(dataset=dict(path="inputs/dataset", sha256=copies[1]["sha256"]), native_parameters={})
    for index, cid in enumerate(sorted(selected)):
        target = f"inputs/parameters/circuit_{index:03d}.json"
        copies.append(input_copy(cohort.parameters[cid], target, cohort.parameter_digests[cid]))
        paths["native_parameters"][cid] = dict(path=target, sha256=cohort.parameter_digests[cid])
    if len({row["path"] for row in copies}) != len(copies):
        raise ValueError("V1_STAGING_DUPLICATE_DESTINATION")
    modules = ["population_trajectory_training", "population_trajectory_evaluation",
        "population_forecast_replay", "v1_population_training_data", "v1_population_evaluation",
        "resource_probe", "poisson_trajectory_training", "poisson_state_space", "native_count_windows",
        "native_trajectory_cohort", "renewal_trajectory_training", "population_state",
        "renewal_state_space", "refractory_observation", "native_renewal_data"]
    names = software_closure(repo, modules) + ["Libs/configs/" + name for name in
        (*configs, "neural_population_pfc_recurrent_gain.yaml")]
    names += ["Libs/scripts/run_population_trajectory_campaign.py", "Libs/scripts/run_population_trajectory_campaign.sbatch"]
    files = [dict(path=name, bytes=(repo / name).stat().st_size, sha256=digest(repo / name)) for name in names]
    durations = {row["dt_ms"] * row["duration_ticks"] for row in episodes}
    units = {row["units"] for row in cohort.source_dimensions.values()}
    output.mkdir(parents=True)
    for row in files:
        copy_bound(repo / row["path"], output / row["path"], row)
    write_json(output / "preparation.json", dict(schema="v1_population_campaign_preparation_v1",
        status="PREPARED_COMPLETE_DEVELOPMENT_COHORT", training_seeds=list(SEEDS),
        files=files, input_files=copies, runtime_paths=paths,
        input_sha256=hashlib.sha256(inspected.raw).hexdigest(), cohort_binding=cohort.binding,
        training_config=configs[0], forecast_config=configs[1], forecast_configuration=forecast,
        fitting_episodes=len(cohort.fit), validation_episodes=len(cohort.validation),
        validation_cases=cohort.validation, source_dimensions=cohort.source_dimensions,
        core_neurons=next(iter(units)) if len(units) == 1 else None,
        episode_ms=next(iter(durations)) if len(durations) == 1 else None,
        input_bytes=sum(row["bytes"] for row in copies),
        output_storage_bytes=v1_output_storage_bytes(cohort.validation, cohort.source_dimensions, forecast, config["latent_channels"]),
        training_submitted=False, scientific_claim_qualified=False, formal_evaluation=False,
        calibration_or_test_accessed=False))


def staging_probe(work):
    spec = importlib.util.spec_from_file_location("population_resource_probe", work / "Libs/population_dynamics/resource_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.probe


def stage_pfc(preparation, work):
    return stage_source(preparation, work, source="pfc")


def stage_v1(preparation, work):
    return stage_source(preparation, work, source="v1")


def stage_source(preparation, work, *, source):
    """Authenticate a frozen preparation and copy only declared local inputs."""
    prefix = source.upper() + "_"
    proof = json.loads((preparation / "preparation.json").read_text())
    if (source not in ("pfc", "v1") or proof.get("schema") != source + "_population_campaign_preparation_v1"
            or proof.get("status") != "PREPARED_COMPLETE_DEVELOPMENT_COHORT"
            or proof.get("training_seeds") != list(SEEDS)
            or (source == "pfc" and (proof.get("fitting_episodes"), proof.get("validation_episodes")) != (80, 30))):
        raise ValueError(prefix + "FROZEN_COMPLETE_PREPARATION_REQUIRED")
    if source == "v1":
        if (any(type(proof.get(key)) is not int or proof[key] < 1 for key in ("fitting_episodes", "validation_episodes"))
                or any(proof[key] != proof["cohort_binding"][key] for key in ("fitting_episodes", "validation_episodes"))
                or len(proof["validation_cases"]) != proof["validation_episodes"]
                or proof["input_sha256"] != proof["cohort_binding"]["circuit_inputs_sha256"]):
            raise ValueError("V1_FROZEN_COMPLETE_PREPARATION_REQUIRED")
        inventory = manifest_rows(dict(files=proof["input_files"]))
        roles = [dict(proof["runtime_paths"]["dataset"],
                      path=proof["runtime_paths"]["dataset"]["path"] + "/dataset.json"),
                 *proof["runtime_paths"]["native_parameters"].values()]
        if (set(proof["runtime_paths"]["native_parameters"]) != set(proof["cohort_binding"]["native_parameter_sha256"])
                or any(row["path"] not in inventory or row["sha256"] != inventory[row["path"]]["sha256"] for row in roles)
                or {cid: row["sha256"] for cid, row in proof["runtime_paths"]["native_parameters"].items()}
                   != proof["cohort_binding"]["native_parameter_sha256"]):
            raise ValueError("V1_FROZEN_RUNTIME_MAP_IDENTITY_CHANGED")
    files = manifest_rows(proof)
    caller = "Libs/scripts/run_population_trajectory_campaign.py"
    if caller not in files or digest(Path(__file__)) != files[caller]["sha256"]:
        raise ValueError(prefix + "FROZEN_CAMPAIGN_CALLER_CHANGED")
    free = shutil.disk_usage(work).free
    if free < proof["input_bytes"] + proof["output_storage_bytes"]:
        raise ResourceBlocked([prefix + "NODE_LOCAL_INPUT_AND_ALL_UNIT_OUTPUT_SPACE_REQUIRED:"
            + f"required_bytes={proof['input_bytes'] + proof['output_storage_bytes']};free_bytes={free}"])
    for row in proof["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(prefix + "RELATIVE_SOFTWARE_PATH_REQUIRED")
        copy_bound(preparation / relative, work / relative, row)
    probe = staging_probe(work)
    health = probe(work / "phases/staging_preflight.json", required_memory_gib=32)
    if health["status"] != "PASS":
        raise ResourceBlocked(health["reasons"])
    for row in proof["input_files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "inputs":
            raise ValueError(prefix + "RELATIVE_INPUT_PATH_REQUIRED")
        copy_bound(Path(row["source"]), work / relative, row)
    if source == "v1":
        def local(row):
            return dict(path=str(work / row["path"]), sha256=row["sha256"])
        mapping = dict(schema="v1_population_execution_path_map_v1", inputs_sha256=proof["input_sha256"],
            dataset=local(proof["runtime_paths"]["dataset"]),
            native_parameters={cid: local(row) for cid, row in proof["runtime_paths"]["native_parameters"].items()})
    else:
        mapping = dict(schema="pfc_population_execution_path_map_v1", inputs_sha256=proof["input_sha256"],
            paths={original: dict(path=str(work / row["path"]), sha256=row["sha256"])
                   for original, row in proof["paths"].items()})
    write_json(work / "execution_path_map.json", mapping)
    write_json(work / "source_snapshot.json", dict(preparation=str(preparation),
        preparation_sha256=digest(preparation / "preparation.json"), files=proof["files"]))
    write_json(work / "inputs.json", dict(input_sha256=proof["input_sha256"],
        files=proof["input_files"], input_bytes=proof["input_bytes"]))
    return probe, proof


def stage_software(repo, work):
    software, proof, index = prerequisite(repo, SOFTWARE, "606023")
    if proof["tests"] != dict(tests=96, failures=0, errors=0, skipped=0):
        raise ValueError("complete accepted 96-test prerequisite required")
    episode, execution, episode_index = prerequisite(repo, EPISODE, "606074")
    for name, expected in (("results/run.json", "ENGINEERING_EXECUTED"),
                           ("results/replay.json", "PASS_EXACT_FULL_UNIT_EVENT_FIELD_AND_LATENT_REPLAY")):
        copy_bound(episode / name, work / "prerequisites" / "episode" / name, episode_index[name])
        if json.loads((work / "prerequisites" / "episode" / name).read_text())["status"] != expected:
            raise ValueError("real full-episode execution/replay prerequisite differs")
    package = software / "Libs/population_dynamics"
    pending = ["population_trajectory_training", "population_trajectory_evaluation",
               "population_forecast_replay", "resource_probe"]
    sources = set()
    while pending:
        name = pending.pop(0)
        if name in sources:
            continue
        relative = "Libs/population_dynamics/" + name + ".py"
        source = software / relative
        row = index[relative]
        if digest(source) != row["sha256"]:
            raise ValueError("accepted source changed: " + relative)
        sources.add(name)
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                pending.extend([node.module] if node.module else [item.name for item in node.names])
    # Include the strict training-source dictionary even if an import is lazy.
    sources.update(("population_state", "renewal_state_space", "poisson_state_space",
                    "native_count_windows", "native_renewal_data", "refractory_observation",
                    "renewal_trajectory_training", "poisson_trajectory_training", "provenance"))
    declared = ["Libs/__init__.py", "Libs/population_dynamics/__init__.py"]
    declared += ["Libs/population_dynamics/" + name + ".py" for name in sorted(sources)]
    declared += ["Libs/configs/" + name for name in CONFIGS]
    for relative in declared:
        if relative in episode_index and episode_index[relative]["sha256"] != index[relative]["sha256"]:
            raise ValueError("software and real episode source snapshots differ: " + relative)
        copy_bound(software / relative, work / relative, index[relative])
    for name, root in (("software", software), ("episode", episode)):
        shutil.copy2(root / "delivery.json", work / "prerequisites" / (name + "_delivery.json"))
    shutil.copy2(Path(__file__), work / "deployment.py")
    shutil.copy2(Path(__file__).with_suffix(".sbatch"), work / "launcher.sbatch")
    write_json(work / "source_snapshot.json", dict(
        software_job_id="606023", episode_job_id="606074",
        software_delivery_sha256=digest(software / "delivery.json"),
        episode_delivery_sha256=digest(episode / "delivery.json"),
        files=[index[relative] for relative in declared]))
    spec = importlib.util.spec_from_file_location("population_resource_probe", work / "Libs/population_dynamics/resource_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.probe


def stage_inputs(repo, work):
    source = repo / COHORT
    receipt = json.loads((source / "delivery.json").read_text())
    required = dict(status="VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE",
                    completed_episodes=72, cortical_units=100000, episode_duration_ms=6000.0,
                    source_dt_ms=.1, field_bin_ms=5., native_rng_family="per_edge_philox4_32_10")
    if any(receipt[key] != value for key, value in required.items()):
        raise ValueError("complete accepted native Philox development cohort required")
    index = manifest_rows(receipt)
    names = ["train/design.json", "train/acquisition.json", "train_fields/dataset.json", "train_fields/episodes.csv"]
    copy_bound(source / names[0], work / "inputs" / names[0], index[names[0]])
    design = json.loads((work / "inputs" / names[0]).read_text())
    cases = design["cases"]
    if (design["acquired_split"] != "train" or len(cases) != 72
            or len({case["case_id"] for case in cases}) != 72
            or any(case["case_id"] != f"train_block_{case['block']:03d}_familiar_{i}"
                   for case in cases for i in [int(case["template_id"].removeprefix("familiar_"))])
            or {case["block"] for case in cases} != set(range(18))
            or any({case["template_id"] for case in cases if case["block"] == block}
                   != {f"familiar_{i}" for i in range(4)} for block in range(18))):
        raise ValueError("72 unique complete familiar training identities required")
    for case in cases:
        identity = case["case_id"]
        names += [f"train/{identity}/{name}" for name in ("report.json", "spikes.npz", "initial_parameters.npz")]
        names += [f"train_fields/{identity}/{name}" for name in ("summary.json", "spatial_fields.npz", "unit_rates.npz")]
    for relative in names[1:]:
        copy_bound(source / relative, work / "inputs" / relative, index[relative])
    shutil.copy2(source / "delivery.json", work / "prerequisites/cohort_delivery.json")
    write_json(work / "inputs.json", dict(
        delivery_sha256=digest(source / "delivery.json"), source_root=str(source),
        source_identity=receipt["independent_field_replay"], files=[index[name] for name in names]))


def phase(work, name, command, probe):
    directory = work / "phases" / name
    directory.mkdir(parents=True)
    receipt = dict(phase=name, command=command, status="PREFLIGHT", command_started=False,
                   started_utc=datetime.now(timezone.utc).isoformat())
    health = probe(directory / "preflight.json", required_memory_gib=32)
    if health["status"] != "PASS":
        receipt.update(status="RESOURCE_BLOCKED", reasons=health["reasons"], exit_code=None)
        write_json(directory / "receipt.json", receipt)
        return 3
    environment = dict(os.environ, PYTHONPATH=str(work / "Libs"), TMPDIR=str(work / "scratch"),
                       PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    if (work / "execution_path_map.json").is_file():
        environment["NEURONSIM_EXECUTION_PATH_MAP"] = str(work / "execution_path_map.json")
    with (directory / "command.log").open("w") as log:
        child = subprocess.Popen(command, cwd=work, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(status="RUNNING", command_started=True, pid=child.pid)
        write_json(directory / "receipt.json", receipt)
        stopped = None
        try:
            with (directory / "watchdog.jsonl").open("a") as telemetry:
                while child.poll() is None:
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        health = probe(directory / "watchdog_latest.json", required_memory_gib=32)
                        telemetry.write(json.dumps(health, allow_nan=False) + "\n")
                        telemetry.flush()
                        if health["status"] != "PASS" and child.poll() is None:
                            stopped = health["reasons"]
                            os.killpg(child.pid, signal.SIGTERM)
                            child.wait(timeout=30)
                            break
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    receipt.update(status="STOP_REQUESTED_CHILD_UNRESOLVED", pid=child.pid)
                    write_json(directory / "receipt.json", receipt)
            raise
        receipt.update(status="RESOURCE_BLOCKED" if stopped else "PASS" if child.returncode == 0 else "FAILED",
                       reasons=stopped or [], exit_code=child.returncode,
                       finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 3 if stopped else 0 if child.returncode == 0 else 2


def archive(work, target, record):
    target.mkdir(parents=True, exist_ok=False)
    rows = []
    for source in sorted(work.rglob("*")):
        relative = source.relative_to(work)
        if not source.is_file() or set(relative.parts) & {"inputs", "scratch", "__pycache__", ".pytest_cache"}:
            continue
        row = dict(path=str(relative), bytes=source.stat().st_size, sha256=digest(source))
        copy_bound(source, target / relative, row)
        rows.append(row)
    record.update(files=rows, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(target / "delivery.json", record)
    print(json.dumps({key: value for key, value in record.items() if key != "files"}), flush=True)


def execution_identity(device, seed=None, campaign_array_id=None, replaces_job=None):
    """Keep campaign/seed identity when relocating an unstarted seed job."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise ValueError("population campaign requires an allocated compute node")
    node, partition = os.environ["SLURMD_NODENAME"], os.environ["SLURM_JOB_PARTITION"]
    gpu = os.environ.get("SLURM_JOB_GPUS", "")
    if ((device == "cpu" and ((node, partition) not in (("cpu03", "gpu"),("cpu04", "cpu")) or gpu))
            or (device == "cuda" and (node not in ("cpu03", "cpu04", "cpu05", "cpu06", "gpu03")
                                     or not set(partition.split(",")) <= {"cpu", "gpu", "gpu1"}
                                     or not gpu or len(gpu.split(",")) != 1))):
        raise ValueError("allocated CPU03/CPU04 CPU or one GPU on an authorized node required")
    if seed is None and campaign_array_id is None and replaces_job is None:
        task = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if task not in range(3):
            raise ValueError("registered seed array indices 0, 1 and 2 required")
        seed = SEEDS[task]
        campaign_array_id = os.environ["SLURM_ARRAY_JOB_ID"]
    else:
        if seed not in SEEDS or type(seed) is not int or campaign_array_id is None or replaces_job is None:
            raise ValueError("relocation requires a registered seed, original campaign array and replaced job")
        if os.environ.get("SLURM_ARRAY_JOB_ID"):
            raise ValueError("relocation must run as an explicit separate job")
        campaign_array_id = str(campaign_array_id)
        if not campaign_array_id.isdecimal() or int(campaign_array_id) <= 0:
            raise ValueError("positive original campaign array identity required")
        task = SEEDS.index(seed)
        if replaces_job != campaign_array_id + "_" + str(task):
            raise ValueError("replaced job must match the original campaign seed task")
    return dict(seed=seed, task=task, array_id=campaign_array_id, node=node,
                partition=partition, replaces_job=replaces_job)


def budget_options(parent_campaign, total_epochs, *, representation="recurrent", pfc_preparation=None,
                   recovery_from=None):
    """Reject incomplete or incompatible continuation requests before writes."""
    if (parent_campaign is None) != (total_epochs is None):
        raise ValueError("budget continuation requires both parent campaign and total epochs")
    if parent_campaign is not None:
        if type(total_epochs) is not int or total_epochs <= 20:
            raise ValueError("prospective total epochs must exceed the frozen 20-epoch budget")
        if representation != "recurrent" or pfc_preparation is not None or recovery_from is not None:
            raise ValueError("budget continuation requires recurrent Mesocircuit execution without resource recovery")


def inspect_budget_parent(repo, parent_campaign, seed, device, total_epochs):
    """Authenticate one complete parent and its declared bindings before staging."""
    parent = Path(parent_campaign).resolve() / ("seed_" + str(seed))
    path = parent / "delivery.json"
    if not path.is_file():
        raise ValueError("budget continuation requires an existing terminal parent delivery")
    proof = json.loads(path.read_text())
    required = dict(status="PASS", exit_code=0, training_seed=seed,
                    device=device, core_neurons=100000, complete_episodes=72, episode_ms=6000,
                    fit_episodes=56, validation_episodes=16, verified_forecast_origins=48, score_rows=144,
                    calibration_or_test_accessed=False, scientific_claim_qualified=False)
    # Original recurrent deliveries predate the optional representation field.
    # The authenticated configuration and report binding below must still declare it.
    if (any(proof.get(key) != value for key, value in required.items())
            or proof.get("representation", "recurrent") != "recurrent"):
        raise ValueError("complete same-seed same-device recurrent parent campaign required")
    rows = manifest_rows(proof)

    def bound_json(name):
        if name not in rows:
            raise ValueError("declared budget parent artifact missing: " + name)
        source = parent / name
        if source.stat().st_size != rows[name]["bytes"] or digest(source) != rows[name]["sha256"]:
            raise ValueError("budget parent artifact identity changed: " + name)
        return json.loads(source.read_text())

    config = bound_json("training_configuration.json")
    report = bound_json("training/run.json")
    snapshot = bound_json("source_snapshot.json")
    inputs = bound_json("inputs.json")
    binding = report.get("binding", {})
    if (report.get("status") != "FITTED_DEVELOPMENT" or report.get("training_seed") != seed
            or binding.get("seed") != seed or binding.get("config") != config
            or config.get("maximum_epochs") != 20 or config.get("representation") != "recurrent"
            or config.get("device") != device or config.get("early_stopping_patience") != 4
            or type(report.get("completed_epochs")) is not int or report["completed_epochs"] != 20
            or report.get("effective_maximum_epochs") != 20 or report.get("continuation") is not None
            or report.get("calibration_outcomes_accessed") is not False
            or report.get("test_outcomes_accessed") is not False or total_epochs <= report["completed_epochs"]):
        raise ValueError("terminal frozen 20-epoch population training binding required")
    training_names = ("training/run.json", "training/latest.pt", "training/best.pt", "training/epochs.json",
                      "training/optimization.csv", "training/validation.csv")
    for name in training_names:
        if name not in rows:
            raise ValueError("declared budget parent artifact missing: " + name)
        source = parent / name
        if source.stat().st_size != rows[name]["bytes"] or digest(source) != rows[name]["sha256"]:
            raise ValueError("budget parent artifact identity changed: " + name)
    for name, key in (("training/latest.pt", "latest_checkpoint_sha256"),
                      ("training/best.pt", "best_checkpoint_sha256")):
        if report.get(key) != rows[name]["sha256"]:
            raise ValueError("budget parent checkpoint receipt missing or changed: " + name)
    software, _, accepted = prerequisite(repo, SOFTWARE, "606023")
    episode, _, _ = prerequisite(repo, EPISODE, "606074")
    if (snapshot.get("software_job_id") != "606023" or snapshot.get("episode_job_id") != "606074"
            or snapshot.get("software_delivery_sha256") != digest(software / "delivery.json")
            or snapshot.get("episode_delivery_sha256") != digest(episode / "delivery.json")):
        raise ValueError("budget parent accepted software binding changed")
    sources = report.get("trainer_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("budget parent checkpoint-recorded trainer source receipts required")
    for name, sha256 in sources.items():
        relative = "Libs/population_dynamics/" + name
        if (Path(name).name != name or relative not in rows or relative not in accepted
                or rows[relative]["sha256"] != sha256 or accepted[relative]["sha256"] != sha256
                or digest(parent / relative) != sha256):
            raise ValueError("budget parent numerical source binding changed: " + name)
    cohort = repo / COHORT
    if inputs.get("delivery_sha256") != digest(cohort / "delivery.json"):
        raise ValueError("budget parent original cohort binding changed")
    expected_inputs = {"acquisition_design_sha256": "train/design.json",
                       "acquisition_ledger_sha256": "train/acquisition.json",
                       "dataset_sha256": "train_fields/dataset.json",
                       "episode_table_sha256": "train_fields/episodes.csv"}
    input_rows = {row["path"]: row for row in inputs["files"]}
    for key, name in expected_inputs.items():
        if name not in input_rows or binding.get(key) != input_rows[name]["sha256"]:
            raise ValueError("budget parent training data binding changed: " + key)
    return dict(parent=str(parent), parent_job_id=proof["job_id"], parent_delivery_sha256=digest(path),
                parent_completed_epochs=report["completed_epochs"], base_maximum_epochs=20,
                total_epochs=total_epochs, parent_checkpoint_sha256=rows["training/latest.pt"]["sha256"],
                config=config, trainer_sources=sources, training_files=[rows[name] for name in training_names],
                parent_metadata=[rows[name] for name in ("training_configuration.json", "source_snapshot.json", "inputs.json")])


def stage_budget_parent(parent, work):
    """Copy only authenticated parent training artifacts into separate local input."""
    source = Path(parent["parent"])
    local = work / "prerequisites/budget_parent"
    if parent["config"] != json.loads((work / "training_configuration.json").read_text()):
        raise ValueError("budget continuation training configuration changed")
    for name, sha256 in parent["trainer_sources"].items():
        if digest(work / "Libs/population_dynamics" / name) != sha256:
            raise ValueError("budget continuation numerical source changed: " + name)
    for row in [*parent["training_files"], *parent["parent_metadata"]]:
        copy_bound(source / row["path"], local / row["path"], row)
    path = source / "delivery.json"
    copy_bound(path, local / "delivery.json", dict(bytes=path.stat().st_size,
               sha256=parent["parent_delivery_sha256"]))
    return local / "training"


def stage_recovery(parent, work, seed, array_id, device, representation):
    """Restore one resource-stopped attempt without changing its training budget."""
    proof_path = parent / "delivery.json"
    proof = json.loads(proof_path.read_text())
    if (proof.get("status") != "RESOURCE_BLOCKED" or proof.get("exit_code") != 3
            or proof.get("training_seed") != seed or str(proof.get("array_job_id")) != array_id
            or proof.get("device") != device or proof.get("representation", "recurrent") != representation
            or proof.get("calibration_or_test_accessed") is not False
            or proof.get("scientific_claim_qualified") is not False):
        raise ValueError("same-seed, same-device resource-stopped campaign delivery required")
    rows = manifest_rows(proof)
    copy_bound(parent / "training_configuration.json", work / "prerequisites/recovery_configuration.json",
               rows["training_configuration.json"])
    config = json.loads((work / "prerequisites/recovery_configuration.json").read_text())
    if config != json.loads((work / "training_configuration.json").read_text()):
        raise ValueError("recovery training configuration changed")
    for name, row in rows.items():
        if name.startswith("Libs/population_dynamics/") and name.endswith(".py"):
            current = work / name
            if not current.is_file() or digest(current) != row["sha256"]:
                raise ValueError("recovery numerical source changed: " + name)
    for name in ("training/latest.pt", "training/run.json"):
        if name not in rows:
            raise ValueError("committed recovery artifact missing: " + name)
    for name, row in rows.items():
        if name.startswith("training/"):
            copy_bound(parent / name, work / name, row)
    report = json.loads((work / "training/run.json").read_text())
    maximum = config["maximum_epochs"]
    lineage = proof.get("optimization_budget_continuation")
    continuation = report.get("continuation")
    if lineage is not None or continuation is not None:
        if (not isinstance(lineage, dict) or not isinstance(continuation, dict)
                or continuation.get("schema") != "neural_population_trajectory_budget_continuation_v1"
                or lineage.get("base_maximum_epochs") != maximum
                or continuation.get("base_maximum_epochs") != maximum
                or type(lineage.get("total_epochs")) is not int or lineage["total_epochs"] <= maximum
                or any(continuation.get(key) != lineage.get(key) for key in
                       ("total_epochs", "parent_completed_epochs", "parent_checkpoint_sha256"))):
            raise ValueError("resource recovery requires the original budget continuation lineage")
        maximum = lineage["total_epochs"]
        prefix = "prerequisites/budget_parent/"
        for name, expected in (("delivery.json", lineage.get("parent_delivery_sha256")),
                               ("training/latest.pt", lineage.get("parent_checkpoint_sha256")),
                               ("training/run.json", continuation.get("parent_artifact_sha256", {}).get("run.json"))):
            if not expected or rows.get(prefix + name, {}).get("sha256") != expected:
                raise ValueError("budget recovery parent prerequisite missing or changed: " + name)
        for name, row in rows.items():
            if name.startswith(prefix):
                copy_bound(parent / name, work / name, row)
    if (report["training_seed"] != seed or type(report["completed_epochs"]) is not int
            or report["completed_epochs"] < 0
            or report["completed_epochs"] >= maximum
            or report.get("effective_maximum_epochs") != maximum
            or report.get("calibration_outcomes_accessed") is not False
            or report.get("test_outcomes_accessed") is not False):
        raise ValueError("incomplete committed training with the original epoch ceiling required")
    if report["completed_epochs"] > 0:
        for name in ("training/best.pt", "training/epochs.json"):
            if name not in rows:
                raise ValueError("committed recovery artifact missing: " + name)
    elif report.get("best_epoch") is not None:
        raise ValueError("zero-epoch recovery cannot declare a selected epoch")
    copy_bound(proof_path, work / "prerequisites/recovery_delivery.json",
               dict(bytes=proof_path.stat().st_size, sha256=digest(proof_path)))
    return dict(parent=str(parent), parent_job_id=proof["job_id"],
                parent_delivery_sha256=digest(proof_path),
                committed_epochs=report["completed_epochs"],
                latest_checkpoint_sha256=rows["training/latest.pt"]["sha256"],
                maximum_epochs=maximum)


def stage_pfc_recovery(parent, work, seed, array_id, device, representation, preparation_sha256):
    return stage_source_recovery(parent, work, seed, array_id, device, representation, preparation_sha256)


def stage_source_recovery(parent, work, seed, array_id, device, representation, preparation_sha256, *,
                          source_contract=None):
    """Resume a frozen source attempt and reuse completed evaluation phases."""
    contract = source_contract or dict(input_family="pfc_source_reproduction",
        evaluation_status="EVALUATED_PFC_DEVELOPMENT_CHECKPOINT", score_rows=360, origins=120)
    path = parent / "delivery.json"
    proof = json.loads(path.read_text())
    if (proof.get("status") != "RESOURCE_BLOCKED" or proof.get("exit_code") != 3
            or proof.get("input_family") != contract["input_family"]
            or proof.get("training_seed") != seed or str(proof.get("array_job_id")) != array_id
            or proof.get("device") != device or proof.get("representation") != representation
            or proof.get("preparation_sha256") != preparation_sha256
            or proof.get("calibration_or_test_accessed") is not False
            or proof.get("scientific_claim_qualified") is not False):
        raise ValueError("PFC_SAME_IDENTITY_FROZEN_RESOURCE_RECOVERY_REQUIRED")
    rows = manifest_rows(proof)
    for name, row in rows.items():
        if name.startswith("Libs/") and name.endswith((".py", ".yaml")):
            target = work / name
            if not target.is_file() or digest(target) != row["sha256"]:
                raise ValueError("PFC_RECOVERY_SOFTWARE_OR_CONFIGURATION_CHANGED:" + name)
    if "training_configuration.json" in rows:
        copy_bound(parent / "training_configuration.json", work / "prerequisites/recovery_configuration.json",
                   rows["training_configuration.json"])
        if json.loads((work / "prerequisites/recovery_configuration.json").read_text()) != json.loads(
                (work / "training_configuration.json").read_text()):
            raise ValueError("PFC_RECOVERY_TRAINING_CONFIGURATION_CHANGED")
    completed = []
    for phase_name in ("fit", "evaluate", "replay"):
        name = "phases/" + phase_name + "/receipt.json"
        if name not in rows:
            break
        copy_bound(parent / name, work / "prerequisites" / ("recovery_" + phase_name + ".json"), rows[name])
        receipt = json.loads((work / "prerequisites" / ("recovery_" + phase_name + ".json")).read_text())
        if receipt.get("status") != "PASS" or receipt.get("exit_code") != 0:
            break
        completed.append(phase_name)
    can_resume = "training/latest.pt" in rows
    if completed and not can_resume:
        raise ValueError("PFC_RECOVERY_COMPLETED_FIT_CHECKPOINT_REQUIRED")
    if can_resume:
        if "training/run.json" not in rows or "training_configuration.json" not in rows:
            raise ValueError("PFC_RECOVERY_COMMITTED_TRAINING_METADATA_REQUIRED")
        for name, row in rows.items():
            if name.startswith("training/"):
                copy_bound(parent / name, work / name, row)
        report = json.loads((work / "training/run.json").read_text())
        maximum = json.loads((work / "training_configuration.json").read_text())["maximum_epochs"]
        if (report.get("training_seed") != seed or type(report.get("completed_epochs")) is not int
                or not 0 <= report["completed_epochs"] <= maximum
                or report.get("effective_maximum_epochs") != maximum
                or report.get("calibration_outcomes_accessed") is not False
                or report.get("test_outcomes_accessed") is not False):
            raise ValueError("PFC_RECOVERY_COMMITTED_EPOCH_BOUNDARY_REQUIRED")
        if report["completed_epochs"] > 0 and "training/epochs.json" not in rows:
            raise ValueError("PFC_RECOVERY_COMMITTED_EPOCH_HISTORY_REQUIRED")
        if report.get("best_epoch") is not None and "training/best.pt" not in rows:
            raise ValueError("PFC_RECOVERY_SELECTED_CHECKPOINT_REQUIRED")
        if "fit" in completed and report.get("status") != "FITTED_DEVELOPMENT":
            raise ValueError("PFC_RECOVERY_COMPLETED_FIT_REPORT_REQUIRED")
    else:
        # Before initialization commits, retain diagnostics outside fresh output.
        for name, row in rows.items():
            if name.startswith("training/"):
                copy_bound(parent / name, work / "prerequisites/uncommitted_attempt" / name, row)
    if "evaluate" in completed:
        if "forecasts/evaluation.json" not in rows:
            raise ValueError("PFC_RECOVERY_COMPLETED_EVALUATION_REPORT_REQUIRED")
        for name, row in rows.items():
            if name.startswith("forecasts/"):
                copy_bound(parent / name, work / name, row)
        evaluation = json.loads((work / "forecasts/evaluation.json").read_text())
        if (evaluation.get("status") != contract["evaluation_status"]
                or evaluation.get("score_rows") != contract["score_rows"] or len(evaluation.get("origins", [])) != contract["origins"]
                or evaluation.get("checkpoint_sha256") != digest(work / "training/best.pt")):
            raise ValueError("PFC_RECOVERY_COMPLETE_EVALUATION_DENOMINATOR_REQUIRED")
    if "replay" in completed:
        copy_bound(parent / "replay.json", work / "replay.json", rows["replay.json"])
    for phase_name in completed:
        receipt = json.loads((work / "prerequisites" / ("recovery_" + phase_name + ".json")).read_text())
        receipt.update(artifacts_reused_from=str(parent), original_phase_command_not_repeated=True)
        write_json(work / "phases" / phase_name / "receipt.json", receipt)
    copy_bound(path, work / "prerequisites/recovery_delivery.json",
               dict(bytes=path.stat().st_size, sha256=digest(path)))
    return dict(parent=str(parent), parent_job_id=proof["job_id"], parent_delivery_sha256=digest(path),
                resume_training=can_resume and "fit" not in completed, completed_phases=completed)


def run(repo, work, device, seed=None, campaign_array_id=None, replaces_job=None, representation="recurrent", recovery_from=None,
        pfc_preparation=None, budget_parent_campaign=None, total_epochs=None, v1_preparation=None):
    if pfc_preparation is not None and v1_preparation is not None:
        raise ValueError("select one source preparation")
    source_preparation = pfc_preparation or v1_preparation
    budget_options(budget_parent_campaign, total_epochs, representation=representation,
                   pfc_preparation=source_preparation, recovery_from=recovery_from)
    if representation not in ("fixed", "recurrent"):
        raise ValueError("registered fixed or recurrent representation required")
    identity = execution_identity(device, seed, campaign_array_id, replaces_job)
    if Path("/tmp/neuronsim") not in work.parents:
        raise ValueError("node-local /tmp/neuronsim campaign scratch required")
    seed, task, array_id = identity["seed"], identity["task"], identity["array_id"]
    node, partition = identity["node"], identity["partition"]
    budget_parent = (inspect_budget_parent(repo, budget_parent_campaign, seed, device, total_epochs)
                     if budget_parent_campaign is not None else None)
    budget_lineage = ({key: value for key, value in budget_parent.items()
                      if key not in ("config", "trainer_sources", "training_files", "parent_metadata")}
                     if budget_parent is not None else None)
    if recovery_from is not None and not source_preparation:
        budget_lineage = json.loads((recovery_from / "delivery.json").read_text()).get(
            "optimization_budget_continuation")
    campaign_root = repo / (V1_CAMPAIGN if v1_preparation else PFC_CAMPAIGN if pfc_preparation
                            else "Output/neural_population_dynamics/model/trajectory_development")
    prepared_config = json.loads((source_preparation / "preparation.json").read_text()) if source_preparation else None
    if pfc_preparation:
        if prepared_config.get("control_geometry","static")=="actuation_only":
            campaign_root=campaign_root/"control_actuation"
    if representation == "fixed":
        campaign_root = campaign_root / "representation_fixed"
    if budget_lineage is not None:
        campaign_root = campaign_root / "optimization_budget_continuation"
    target = campaign_root / ("run_" + array_id) / ("seed_" + str(seed))
    if (source_preparation or budget_lineage is not None) and recovery_from is not None:
        target = campaign_root / ("run_" + array_id) / "resource_attempts" / (
            "seed_" + str(seed) + "_job_" + os.environ["SLURM_JOB_ID"])
    if target.exists():
        raise FileExistsError("existing campaign evidence requires an explicit separate recovery path")
    record = dict(status="RUNNING", exit_code=None, job_id=os.environ["SLURM_JOB_ID"],
                  array_job_id=array_id, array_task_id=task, training_seed=seed, representation=representation,
                  node=node, partition=partition, device=device, replaces_unstarted_job=identity["replaces_job"], core_neurons=100000,
                  complete_episodes=72, episode_ms=6000, fit_episodes=56, validation_episodes=16,
                  formal_training_completed=False, source_fidelity_qualified=False,
                  generative_fidelity_qualified=False, scientific_claim_qualified=False,
                  calibration_or_test_accessed=False, started_utc=datetime.now(timezone.utc).isoformat())
    if pfc_preparation:
        record.update(input_family="pfc_source_reproduction", complete_episodes=110,
                      episode_ms=5000, fit_episodes=80, validation_episodes=30, formal_evaluation=False)
    if v1_preparation:
        record.update(input_family="v1_qualified_source_episodes",
            complete_episodes=prepared_config["fitting_episodes"] + prepared_config["validation_episodes"],
            fit_episodes=prepared_config["fitting_episodes"], validation_episodes=prepared_config["validation_episodes"],
            episode_ms=prepared_config["episode_ms"], core_neurons=prepared_config["core_neurons"],
            source_dimensions=prepared_config["source_dimensions"], formal_evaluation=False)
    if budget_lineage is not None:
        record["optimization_budget_continuation"] = budget_lineage
    code = 2
    try:
        configs = V1_CONFIGS if v1_preparation else PFC_CONFIGS if pfc_preparation else CONFIGS
        if pfc_preparation:
            record["preparation_sha256"] = digest(pfc_preparation / "preparation.json")
            probe, preparation = stage_pfc(pfc_preparation, work)
            configs=(preparation.get("training_config",PFC_CONFIGS[0]),
                     preparation.get("forecast_config",PFC_CONFIGS[1]))
            record.update(input_family="pfc_source_reproduction", complete_episodes=110,
                          episode_ms=5000, fit_episodes=80, validation_episodes=30,
                          preparation_sha256=digest(pfc_preparation / "preparation.json"),
                          formal_evaluation=False)
        elif v1_preparation:
            record["preparation_sha256"] = digest(v1_preparation / "preparation.json")
            probe, preparation = stage_v1(v1_preparation, work)
            configs = (preparation["training_config"], preparation["forecast_config"])
        else:
            probe = stage_software(repo, work)
        (work / "scratch").mkdir()
        preflight = "inputs_ready_preflight.json" if source_preparation else "staging_preflight.json"
        health = probe(work / "phases" / preflight, required_memory_gib=32)
        if health["status"] != "PASS":
            record.update(status="RESOURCE_BLOCKED", reasons=health["reasons"], exit_code=3)
            archive(work, target, record)
            (work / "archive_complete").write_text(str(target))
            return 3
        if not source_preparation:
            stage_inputs(repo, work)
        config = yaml.safe_load((work / "Libs/configs" / configs[0]).read_text())
        config.update(device=device, cpu_threads=2, representation=representation)
        write_json(work / "training_configuration.json", config)
        (work / "Libs/configs/neural_population_state_trajectory_execution.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        forecast = yaml.safe_load((work / "Libs/configs" / configs[1]).read_text())
        if pfc_preparation:
            if (forecast["joint_samples"] != 16 or forecast["origins_ticks"] != [2000, 8000, 48000, 80000]
                    or forecast["horizons_bins"] != [4, 16, 100]):
                raise ValueError("registered full PFC development evaluation required")
        elif v1_preparation:
            if forecast != preparation["forecast_configuration"]:
                raise ValueError("V1_FROZEN_FORECAST_CONFIGURATION_CHANGED")
        elif forecast["joint_samples"] != 16 or forecast["origins_ms"] != [1000., 3000., 5000.] or forecast["horizons_ms"] != [20., 80., 500.]:
            raise ValueError("fixed 16-draw 3-origin development evaluation required")
        if recovery_from is not None:
            if recovery_from.resolve() == target.resolve():
                raise ValueError("recovery must preserve its archived parent")
            if pfc_preparation:
                record["resource_recovery"] = stage_pfc_recovery(recovery_from.resolve(), work, seed,
                    array_id, device, representation, record["preparation_sha256"])
            elif v1_preparation:
                origins = preparation["validation_episodes"] * len(forecast["origins_ticks"])
                record["resource_recovery"] = stage_source_recovery(recovery_from.resolve(), work, seed,
                    array_id, device, representation, record["preparation_sha256"], source_contract=dict(
                        input_family="v1_qualified_source_episodes", evaluation_status="EVALUATED_V1_DEVELOPMENT_CHECKPOINT",
                        origins=origins, score_rows=origins * len(forecast["horizons_bins"])))
            else:
                record["resource_recovery"] = stage_recovery(
                    recovery_from.resolve(), work, seed, array_id, device, representation)
        local_budget_parent = stage_budget_parent(budget_parent, work) if budget_parent is not None else None
        common = (["--acquisition", "inputs/cohort.json"] if source_preparation else
                  ["--acquisition", "inputs/train", "--fields", "inputs/train_fields"])
        commands = [("fit", [sys.executable, "-m", "population_dynamics.population_trajectory_training",
            *common, "--config", "Libs/configs/neural_population_state_trajectory_execution.yaml",
            "--output", "training", "--seed", str(seed)]),
            ("evaluate", [sys.executable, "-m", "population_dynamics.population_trajectory_evaluation",
            *common, "--training", "training", "--config", "Libs/configs/" + configs[1],
            "--output", "forecasts", "--model-type", "population", "--device", device]),
            ("replay", [sys.executable, "-m", "population_dynamics.population_forecast_replay",
            *common, "--training", "training", "--forecasts", "forecasts", "--output", "replay.json",
            "--device", device, "--all-origins"])]
        if local_budget_parent is not None:
            commands[0][1].extend(["--resume-from", str(local_budget_parent), "--total-epochs", str(total_epochs)])
        recovery = record.get("resource_recovery", {})
        if recovery_from is not None and (not source_preparation or recovery["resume_training"]):
            commands[0][1].append("--resume")
        if source_preparation and recovery_from is not None:
            commands = [(name, command) for name, command in commands if name not in recovery["completed_phases"]]
        for name, command in commands:
            code = phase(work, name, command, probe)
            if code:
                break
        if code == 0:
            fit = json.loads((work / "training/run.json").read_text())
            evaluation = json.loads((work / "forecasts/evaluation.json").read_text())
            replay = json.loads((work / "replay.json").read_text())
            fit_count, validation_count = ((preparation["fitting_episodes"], preparation["validation_episodes"])
                if v1_preparation else (80, 30) if pfc_preparation else (56, 16))
            expected_origins = validation_count * len(forecast["origins_ticks"] if source_preparation else forecast["origins_ms"])
            expected_rows = expected_origins * len(forecast["horizons_bins"] if source_preparation else forecast["horizons_ms"])
            evaluation_status = ("EVALUATED_V1_DEVELOPMENT_CHECKPOINT" if v1_preparation else
                                 "EVALUATED_PFC_DEVELOPMENT_CHECKPOINT" if pfc_preparation else "EVALUATED_DEVELOPMENT")
            replay_status = "PASS_ALL_UNIT_AND_INDEPENDENT_METRIC_REPLAY" if source_preparation else "PASS"
            verified_origins = len(replay["origins"]) if source_preparation else replay["verified_origins"]
            if (fit["status"] != "FITTED_DEVELOPMENT" or fit["training_seed"] != seed
                    or fit["fitting_episodes"] != fit_count or fit["validation_episodes"] != validation_count
                    or evaluation["status"] != evaluation_status or evaluation["score_rows"] != expected_rows
                    or replay["status"] != replay_status or verified_origins != expected_origins):
                raise ValueError("complete fitting/evaluation/replay denominator not achieved")
            if v1_preparation:
                if any(fit["binding"].get(key) != value for key, value in preparation["cohort_binding"].items()):
                    raise ValueError("V1_FITTED_SOURCE_COHORT_DIFFERS_FROM_PREPARATION")
                expected_values = sum(preparation["source_dimensions"][case["circuit_id"]]["units"]
                    for case in preparation["validation_cases"]) * len(forecast["origins_ticks"]) * forecast["joint_samples"] * max(forecast["horizons_bins"])
                if (replay["unit_values_consumed"] != expected_values
                        or evaluation["generated_unit_count_values"] != expected_values):
                    raise ValueError("V1_COMPLETE_UNIT_REPLAY_DENOMINATOR_CHANGED")
            if budget_lineage is not None:
                continuation = fit.get("continuation", {})
                if (fit.get("effective_maximum_epochs") != budget_lineage["total_epochs"]
                        or continuation.get("total_epochs") != budget_lineage["total_epochs"]
                        or continuation.get("base_maximum_epochs") != 20
                        or continuation.get("parent_completed_epochs") != budget_lineage["parent_completed_epochs"]
                        or continuation.get("parent_checkpoint_sha256") != budget_lineage["parent_checkpoint_sha256"]
                        or fit.get("binding", {}).get("config") != config
                        or not budget_lineage["parent_completed_epochs"] <= fit["completed_epochs"] <= budget_lineage["total_epochs"]):
                    raise ValueError("budget continuation effective ceiling or parent provenance changed")
                record["effective_maximum_epochs"] = fit["effective_maximum_epochs"]
            record.update(status="PASS", completed_epochs=fit["completed_epochs"],
                          stop_reason=fit["stop_reason"], verified_forecast_origins=expected_origins, score_rows=expected_rows)
        else:
            record["status"] = "RESOURCE_BLOCKED" if code == 3 else "FAILED"
    except ResourceBlocked as error:
        record.update(status="RESOURCE_BLOCKED", reasons=error.reasons)
        code = 3
    except BaseException as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        code = 2
    record["exit_code"] = code
    archive(work, target, record)
    (work / "archive_complete").write_text(str(target))
    return code


def stage_cost_point(repo, work, software):
    """Stage one accepted checkpoint and one real history without training."""
    software_proof = json.loads((software/"delivery.json").read_text())
    if software_proof["status"] != "PASS" or software_proof["exit_code"] != 0:
        raise ValueError("accepted cost-point software required")
    software_rows = manifest_rows(software_proof)
    for name, row in software_rows.items():
        if name.startswith("Libs/") and name.endswith((".py", ".yaml")):
            copy_bound(software/name, work/name, row)
    seed41 = repo/ACCEPTED_SEED41
    seed_proof = json.loads((seed41/"delivery.json").read_text())
    if (seed_proof["status"] != "PASS" or seed_proof["exit_code"] != 0
            or seed_proof["training_seed"] != 41 or seed_proof["verified_forecast_origins"] != 48):
        raise ValueError("complete accepted seed41 checkpoint/evaluation/replay required")
    seed_rows = manifest_rows(seed_proof)
    for name in ("training/run.json", "training/best.pt", "forecasts/fitting_field_scale.npz"):
        copy_bound(seed41/name, work/name, seed_rows[name])
    source = repo/COHORT
    source_proof = json.loads((source/"delivery.json").read_text())
    if (source_proof["status"] != "VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE"
            or source_proof["completed_episodes"] != 72 or source_proof["cortical_units"] != 100000):
        raise ValueError("complete original source cohort required")
    source_rows = manifest_rows(source_proof)
    names = ["train/design.json", "train_fields/dataset.json"]
    names += ["train/"+COST_POINT_CASE+"/"+name for name in ("report.json", "spikes.npz", "initial_parameters.npz")]
    names += ["train_fields/"+COST_POINT_CASE+"/"+name for name in ("summary.json", "spatial_fields.npz")]
    for name in names:
        copy_bound(source/name, work/"inputs"/name, source_rows[name])
    binding = dict(software_delivery_sha256=digest(software/"delivery.json"),
                   checkpoint_delivery_sha256=digest(seed41/"delivery.json"),
                   cohort_delivery_sha256=digest(source/"delivery.json"),
                   checkpoint_sha256=digest(work/"training/best.pt"), case_id=COST_POINT_CASE,
                   training_performed=False, complete_native_unit_count=100000,
                   fitting_scale_sha256=digest(work/"forecasts/fitting_field_scale.npz"))
    write_json(work/"cost_point_binding.json", binding)
    return binding


def run_cost_point(repo, work, target, software):
    """Measure a new same-output posterior draw set on the allocated GPU."""
    if (os.environ.get("SLURMD_NODENAME") != "gpu03" or os.environ.get("SLURM_JOB_PARTITION") != "gpu1"
            or not os.environ.get("SLURM_JOB_GPUS") or not os.environ.get("SLURM_JOB_ID")):
        raise ValueError("single GPU03 cost-point allocation required")
    work.mkdir(parents=True, exist_ok=False)
    (work/"scratch").mkdir()
    started = time.perf_counter()
    record = dict(status="RUNNING", job_id=os.environ["SLURM_JOB_ID"],
                  started_utc=datetime.now(timezone.utc).isoformat(), training_performed=False,
                  previous_forecast_streams_repeated=False, observed_history_posterior_cost_point=True,
                  native_fixed_microstate_law_matched=False, end_to_end_cost_qualified=False,
                  matched_fidelity_qualified=False, speedup_qualified=False, scientific_claim_qualified=False)
    code = 2
    try:
        record["binding"] = stage_cost_point(repo, work, software)
        record["software_and_input_staging_seconds"] = time.perf_counter()-started
        env = dict(os.environ, PYTHONPATH=str(work)+":"+str(work/"Libs"),
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", TMPDIR=str(work/"scratch"))
        health_command = [sys.executable, "-m", "Libs.population_dynamics.resource_probe", "--output", str(work/"resource.json"),
                          "--required-memory-gib", "16"]
        with (work/"resource.log").open("w") as log:
            health = subprocess.run(health_command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        if health.returncode:
            raise RuntimeError("population cost-point resource gate failed")
        command = [sys.executable, "-m", "Libs.population_dynamics.population_trajectory_evaluation",
                   "--acquisition", str(work/"inputs/train"), "--fields", str(work/"inputs/train_fields"),
                   "--training", str(work/"training"), "--config", str(work/"Libs/configs/neural_population_population_cost_point.yaml"),
                   "--fitting-scale", str(work/"forecasts/fitting_field_scale.npz"), "--output", str(work/"cost_point"),
                   "--model-type", "population", "--device", "cuda", "--cost-point"]
        process_start = time.perf_counter()
        with (work/"execution.log").open("w") as log:
            execution = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(command=command, child_exit_code=execution.returncode,
                      process_seconds=time.perf_counter()-process_start)
        if execution.returncode:
            raise RuntimeError("population cost-point execution failed")
        result = json.loads((work/"cost_point/cost_point.json").read_text())
        if (result["status"] != "EVALUATED_OBSERVED_HISTORY_POSTERIOR_COST_POINT"
                or result["generated_unit_count_values"] != 160000000
                or result["all_unit_outputs_saved"] is not True
                or result["checkpoint_sha256"] != record["binding"]["checkpoint_sha256"]):
            raise ValueError("complete accepted-checkpoint posterior cost point required")
        record.update(status="PASS", result_status=result["status"], generated_unit_count_values=160000000,
                      full_unit_counts_saved=True, hardware=result["hardware"],
                      cost_scope="one new16draw observed-history posterior cost point; process imports, software/input staging and archive measured; training amortization and native conditional equivalence unqualified")
        code = 0
    except BaseException as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        record.update(exit_code=code, pipeline_before_archive_seconds=time.perf_counter()-started,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        archive_start = time.perf_counter()
        archive(work, target, record)
        record.update(archive_seconds=time.perf_counter()-archive_start, task_through_archive_seconds=time.perf_counter()-started)
        write_json(target/"delivery.json", record)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, help="Allocated node-local execution directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--representation", choices=("fixed", "recurrent"), default="recurrent",
                        help="Shared generator state representation; fixed output has its own campaign subtree")
    parser.add_argument("--prepare-pfc", action="store_true", help="Freeze a complete accepted PFC cohort without submission")
    parser.add_argument("--pfc-inputs", type=Path, help="Original PFC input manifest, preserved byte-identically")
    parser.add_argument("--pfc-coverage", type=Path, help="Maintained intake coverage receipt")
    parser.add_argument("--pfc-training-config",type=Path,help="Canonical training config for a new complete preparation")
    parser.add_argument("--pfc-forecast-config",type=Path,help="Canonical forecast config for a new complete preparation")
    parser.add_argument("--pfc-preparation", type=Path, help="Execute a frozen complete PFC preparation")
    parser.add_argument("--prepare-v1", action="store_true", help="Freeze original V1 train/validation inputs without submission")
    parser.add_argument("--v1-inputs", type=Path)
    parser.add_argument("--v1-training-config", type=Path)
    parser.add_argument("--v1-forecast-config", type=Path)
    parser.add_argument("--v1-preparation", type=Path, help="Execute a frozen V1 source preparation")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Relocate an unstarted seed into the original campaign output")
    parser.add_argument("--campaign-array-id", help="Original campaign identity for explicit seed relocation")
    parser.add_argument("--replaces-job", help="Original unstarted array task, recorded in the delivery")
    parser.add_argument("--recovery-from", type=Path, help="Archived resource-stopped attempt; restore committed epochs with the same configuration")
    parser.add_argument("--budget-parent-campaign", type=Path, help="Complete recurrent campaign; select the registered seed parent")
    parser.add_argument("--total-epochs", type=int, help="Prospective budget continuation ceiling; base configuration stays unchanged")
    parser.add_argument("--cost-point", action="store_true", help="Measure one new draw set from an accepted seed41 checkpoint")
    parser.add_argument("--producer-software", type=Path, help="Accepted cost-point software snapshot")
    parser.add_argument("--output", type=Path, help="Explicit cost-point archive")
    args = parser.parse_args()
    try:
        budget_options(args.budget_parent_campaign, args.total_epochs, representation=args.representation,
                       pfc_preparation=args.pfc_preparation or args.v1_preparation, recovery_from=args.recovery_from)
    except ValueError as error:
        parser.error(str(error))
    if args.budget_parent_campaign is not None and (args.prepare_pfc or args.prepare_v1 or args.cost_point):
        parser.error("budget continuation cannot prepare PFC inputs or execute a cost point")
    if args.pfc_preparation and args.v1_preparation:
        parser.error("select one source preparation")
    if args.prepare_v1:
        if (not args.v1_inputs or not args.output or args.work or args.prepare_pfc or args.pfc_preparation
                or args.v1_preparation or args.recovery_from or args.cost_point or args.seed
                or args.pfc_inputs or args.pfc_coverage or args.pfc_training_config or args.pfc_forecast_config):
            parser.error("--prepare-v1 requires V1 inputs and output; no execution or PFC arguments")
        prepare_v1(args.repo.resolve(), args.v1_inputs.resolve(), args.output.resolve(),
                   training_config=args.v1_training_config, forecast_config=args.v1_forecast_config)
        return
    if args.v1_inputs or args.v1_training_config or args.v1_forecast_config:
        parser.error("V1 input/config arguments apply only to --prepare-v1")
    if args.prepare_pfc:
        if (not args.pfc_inputs or not args.pfc_coverage or not args.output
                or args.cost_point or args.pfc_preparation or args.v1_preparation or args.recovery_from):
            parser.error("--prepare-pfc requires inputs, coverage and output; no execution/recovery arguments")
        prepare_pfc(args.repo.resolve(), args.pfc_inputs.resolve(), args.pfc_coverage.resolve(), args.output.resolve(),
                    training_config=args.pfc_training_config,forecast_config=args.pfc_forecast_config)
        return
    if args.pfc_inputs or args.pfc_coverage or args.pfc_training_config or args.pfc_forecast_config:
        parser.error("PFC inputs/coverage apply only to --prepare-pfc")
    if args.work is None:
        parser.error("execution requires --work")
    if args.cost_point:
        if (not args.producer_software or not args.output or args.device != "cuda"
                or args.seed or args.campaign_array_id or args.replaces_job or args.recovery_from or args.pfc_preparation or args.v1_preparation or args.representation != "recurrent"):
            parser.error("--cost-point requires GPU device, software and output; no training seed/campaign arguments")
        raise SystemExit(run_cost_point(args.repo.resolve(), args.work.resolve(), args.output.resolve(), args.producer_software.resolve()))
    raise SystemExit(run(args.repo.resolve(), args.work.resolve(), args.device,
                         args.seed, args.campaign_array_id, args.replaces_job, args.representation, args.recovery_from,
                         args.pfc_preparation.resolve() if args.pfc_preparation else None,
                         args.budget_parent_campaign.resolve() if args.budget_parent_campaign else None, args.total_epochs,
                         args.v1_preparation.resolve() if args.v1_preparation else None))


if __name__ == "__main__":
    main()
