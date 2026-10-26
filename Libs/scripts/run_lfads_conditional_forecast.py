"""Run a fitted stationary or conditional-mean LFADS prior and its replay."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import yaml


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_file(root, name, entries):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("ARCHIVE_RELATIVE_PATH_REQUIRED")
    path = root / relative
    row = entries[str(relative)]
    if path.is_symlink() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
        raise ValueError("ARCHIVED_INPUT_CHANGED:" + str(path))
    return path


def stationary_input_plan(fit_run, reference):
    """Extend the completed fit's verified plan with selection data and its cache."""
    completed = [row for row in reference.get("commands", []) if row["label"] == "posterior_fit"]
    if (reference.get("status") != "FITTED_PRIOR_ONLY" or reference.get("action") != "fit"
            or len(completed) != 1 or completed[0].get("status") != "PASS"
            or completed[0].get("exit_code") != 0
            or reference["staged_inputs"]["status"] != "VERIFIED"
            or reference["training_seed"] not in (41, 42, 43)
            or reference["network_variant"] not in ("budget_20", "budget_80")):
        raise ValueError("COMPLETED_STATIONARY_FIT_REQUIRED")
    entries = {row["path"]: row for row in reference["files"]}
    config_path = checked_file(fit_run, "Libs/configs/neural_population_lfads_prior_refit.yaml", entries)
    config = yaml.safe_load(config_path.read_text())
    cache_path = checked_file(fit_run, "cache/posterior_cache.json", entries)
    cache = json.loads(cache_path.read_text())
    cases = [f"train_block_{block:03d}_familiar_{template}"
             for block in config["fit_blocks"] for template in range(4)]
    if (cache.get("status") != "FITTED_FROZEN_NETWORK_AR_PRIOR"
            or cache.get("fitting_cohort_extracted") is not True
            or cache.get("model_parameters_updated") is not False
            or cache.get("formal_test_accessed") is not False
            or len(cases) != config["complete_episodes"]
            or cache.get("complete_episodes") != len(cases)
            or cache.get("posterior_paths") != len(cases) * config["posterior_draws_per_episode"]
            or cache["binding"].get("case_ids") != cases
            or cache["binding"].get("training_seed") != reference["training_seed"]
            or type(cache.get("frozen_global_variables_verified")) is not int
            or cache["frozen_global_variables_verified"] <= 0):
        raise ValueError("COMPLETE_FROZEN_STATIONARY_CACHE_REQUIRED")
    episodes = cache.get("episodes", [])
    if [row["case_id"] for row in episodes] != cases:
        raise ValueError("COMPLETE_FITTING_CASES_REQUIRED")
    for case, row in zip(cases, episodes):
        if row["path"] != case + ".npz":
            raise ValueError("FITTING_CACHE_PATH_CHANGED")
    plan = [dict(row) for row in reference["staged_inputs"]["files"]]
    by_path = {row["path"]: row for row in plan}
    source_manifests = {}

    def add_selection(anchor, filename):
        original = by_path[anchor]
        manifest_path = Path(original["delivery"])
        if manifest_path not in source_manifests:
            if digest(manifest_path) != reference["staged_inputs"]["source_deliveries"][str(manifest_path)]:
                raise ValueError("SELECTION_SOURCE_DELIVERY_CHANGED")
            source_manifests[manifest_path] = {
                row["path"]: row for row in json.loads(manifest_path.read_text())["files"]}
        source = Path(original["source"]).with_name(filename)
        relative = str(source.relative_to(manifest_path.parent))
        declared = source_manifests[manifest_path][relative]
        # Native copyback receipts use destination_bytes; staged fit receipts
        # use bytes. Preserve the same two formats as the existing fit runner.
        size = declared.get("bytes", declared.get("destination_bytes"))
        if type(size) is not int or size < 0 or len(declared.get("sha256", "")) != 64:
            raise ValueError("ARCHIVE_SIZE_AND_DIGEST_REQUIRED:" + str(source))
        destination = str(Path(anchor).with_name(filename))
        if destination in by_path:
            raise ValueError("DUPLICATE_SELECTION_INPUT")
        row = dict(source=str(source), path=destination, delivery=str(manifest_path),
                   bytes=size, sha256=declared["sha256"])
        plan.append(row)
        by_path[destination] = row

    for block in config["excluded_selection_blocks"]:
        for template in range(4):
            case = f"train_block_{block:03d}_familiar_{template}"
            for name in ("spikes.npz", "initial_parameters.npz"):
                add_selection(f"inputs/train/{case}/report.json", name)
            add_selection(f"inputs/train_fields/{case}/summary.json", "spatial_fields.npz")
    for name in ["posterior_cache.json", "prior_parameters.npz", *[case + ".npz" for case in cases]]:
        row = entries["cache/" + name]
        plan.append(dict(source=str(fit_run / "cache" / name), path="inputs/prior_refit/" + name,
                         bytes=row["bytes"], sha256=row["sha256"], delivery=str(fit_run / "delivery.json")))
    return plan




def configure_execution_environment(work):
    """Apply CPU isolation and qualified precision before parent native imports."""
    values = dict(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(work / "Libs"),
        TMPDIR=str(work / "scratch"), PYTHONDONTWRITEBYTECODE="1", NVIDIA_TF32_OVERRIDE="0",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.environ.update(values)
    return dict(os.environ)


def activate_staged_software(work):
    """Bind both supported package names to the digest-verified local code."""
    work = Path(work).resolve()
    roots = {"Libs": work / "Libs", "population_dynamics": work / "Libs/population_dynamics"}
    for root in roots.values():
        if not (root / "__init__.py").is_file():
            raise ValueError("COMPLETE_STAGED_PACKAGE_REQUIRED:" + str(root))
    # Changing sys.path cannot replace a package already cached by an outer
    # launcher. Reject conflicting origins without resetting unrelated modules.
    for name, module in tuple(sys.modules.items()):
        root = next((path for package, path in roots.items()
                     if name == package or name.startswith(package + ".")), None)
        if root is None:
            continue
        origins = list(getattr(module, "__path__", ()))
        if getattr(module, "__file__", None):
            origins.append(module.__file__)
        if not origins or any(not Path(path).resolve().is_relative_to(root) for path in origins):
            raise ValueError("CONFLICTING_STAGED_PACKAGE_IMPORT:" + name)
    paths = [str(work), str(work / "Libs")]
    sys.path[:] = paths + [path for path in sys.path if path not in paths]
    importlib.invalidate_caches()
    for package, root in roots.items():
        module = importlib.import_module(package)
        if Path(module.__file__).resolve() != root / "__init__.py":
            raise ValueError("STAGED_PACKAGE_ORIGIN_CHANGED:" + package)


def validate_stationary_inputs(work):
    # Reuse the native reader for every episode, parameter and checkpoint check.
    from population_dynamics.lfads_prior_refit import load_prior_parameters
    from population_dynamics.lfads_trajectory_evaluation import _verified_training

    inputs = work / "inputs"
    cohort, run, _, selected, migration = _verified_training(
        inputs / "train", inputs / "train_fields", inputs / "training",
        inputs / "migration/inference_migration.json")
    cache = inputs / "prior_refit"
    load_prior_parameters(cache, run=run, selected=selected, cohort_binding=cohort.binding,
                          training_run_sha256=digest(inputs / "training/run.json"))
    document = json.loads((cache / "posterior_cache.json").read_text())
    if document["binding"].get("inference_migration") != migration:
        raise ValueError("STATIONARY_PRIOR_INFERENCE_MIGRATION_CHANGED")
    return dict(status="VERIFIED", complete_episodes=len(document["episodes"]),
                frozen_global_variables_verified=document["frozen_global_variables_verified"],
                training_updates_before_and_after=document["training_updates_before_and_after"])


def validate_execution_allocation():
    """Use the two CPU nodes permitted by the cluster policy, without GPUs."""
    if (os.environ.get("SLURMD_NODENAME") not in {"cpu03", "cpu04"}
            or not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS")):
        raise ValueError("AUTHORIZED_CPU_ONLY_SLURM_ALLOCATION_REQUIRED")


def stage_completed_forecasts(parent, output):
    """Reuse a completed forecast phase; the independent replay remains mandatory."""
    delivery = json.loads((parent / "delivery.json").read_text())
    if delivery["status"] != "RESOURCE_BLOCKED" or delivery["exit_code"] != 3:
        raise ValueError("CLOSED_RESOURCE_STOPPED_PARENT_REQUIRED")
    entries = {row["path"]: row for row in delivery["files"]}
    if len(entries) != len(delivery["files"]):
        raise ValueError("UNIQUE_PARENT_DELIVERY_PATHS_REQUIRED")
    prefix = "candidate/forecasts/"
    evaluation_path = checked_file(parent, prefix + "evaluation.json", entries)
    evaluation = json.loads(evaluation_path.read_text())
    phase = json.loads(checked_file(parent, "phases/forecast/receipt.json", entries).read_text())
    if (evaluation["status"] != "EVALUATED_DEVELOPMENT" or not evaluation["completed_origins"]
            or phase["status"] != "PASS" or phase["exit_code"] != 0 or not phase["command_started"]):
        raise ValueError("COMPLETED_FORECAST_PHASE_REQUIRED_FOR_REPLAY_ONLY")
    names = ["evaluation.json", "scores.csv", *evaluation["artifacts"]]
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError("UNIQUE_FORECAST_ARTIFACT_NAMES_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    for name in names:
        source = checked_file(parent, prefix + name, entries)
        expected = (evaluation["scores_sha256"] if name == "scores.csv"
                    else evaluation["artifacts"].get(name, entries[prefix + name]["sha256"]))
        if entries[prefix + name]["sha256"] != expected:
            raise ValueError("PARENT_FORECAST_EVALUATION_BINDING_CHANGED")
        shutil.copy2(source, output / name)
        if digest(output / name) != expected:
            raise ValueError("COPIED_COMPLETED_FORECAST_CHANGED")
    return dict(parent_job_id=delivery["job_id"], parent_node=delivery.get("node"),
        parent_delivery_sha256=digest(parent / "delivery.json"),
        evaluation_sha256=digest(output / "evaluation.json"),
        reused_origins=evaluation["completed_origins"], reused_score_rows=evaluation["score_rows"],
        forecast_origins_generated=0, replay_progress_reused=0, source_forecast_phase=phase,
        cost_scope="Forecast timings and continuation lineage belong to the preserved source attempts; this attempt runs only replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-run", type=Path)
    source.add_argument("--fit-run", type=Path, help="Completed stationary-prior fit; forecast only the fitted arm")
    parser.add_argument("--mean-run", type=Path)
    for name in ("software", "guard", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--continuation-software", type=Path)
    parser.add_argument("--replay-only", action="store_true",
                        help="Reuse a completed parent forecast phase and run only independent replay")
    args = parser.parse_args()
    source_fit = args.fit_run is not None
    stationary = args.mean_run is None
    if stationary and not source_fit:
        raise ValueError("STATIONARY_FORECAST_REQUIRES_COMPLETED_FIT")
    if (args.continue_from is None) != (args.continuation_software is None):
        raise ValueError("CONTINUATION_REQUIRES_ITS_VALIDATED_SOFTWARE")
    if args.replay_only and args.continue_from is None:
        raise ValueError("REPLAY_ONLY_REQUIRES_COMPLETED_PARENT")
    validate_execution_allocation()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference_root = args.fit_run if source_fit else args.reference_run
    reference = json.loads((reference_root / "delivery.json").read_text())
    record = dict(status="RUNNING", job_id=os.environ["SLURM_JOB_ID"], node=os.environ["SLURMD_NODENAME"], command=sys.argv,
        started_utc=datetime.now(timezone.utc).isoformat(),
        training_seed=reference["training_seed"] if source_fit else 41,
        network_budget_epochs={"budget_20": 20, "budget_80": 80}[reference["network_variant"]] if source_fit else 20,
        reference_run=str(reference_root), mean_run=str(args.mean_run) if args.mean_run is not None else None,
        software=str(args.software), prior_variant="stationary" if stationary else "conditional_mean",
        reference_delivery_sha256=digest(reference_root / "delivery.json"),
        mean_delivery_sha256=digest(args.mean_run / "delivery.json") if args.mean_run is not None else None,
        software_delivery_sha256=digest(args.software / "delivery.json"),
        scientific_claim_qualified=False, formal_test_accessed=False, production_episodes=0,
        original_reference_reexecuted=False, neural_training_updates=0, replay_only=args.replay_only, commands=[])
    code = 1
    with tempfile.TemporaryDirectory(prefix="neuronsim-conditional-forecast-" + record["job_id"] + ".", dir="/tmp") as runtime:
        work = Path(runtime)
        (work / "scratch").mkdir()
        (work / "phases").mkdir()
        def save():
            (work / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
        try:
            env = configure_execution_environment(work)
            software = json.loads((args.software / "delivery.json").read_text())
            accepted_software = (software["status"] == "ARCHIVED_STATIONARY_PRIOR_SOFTWARE_TESTS"
                                 and software.get("validation_mode") == "consumers"
                                 and software.get("requested_test_device") == "cpu") if stationary else software["status"] == "PASS"
            accepted_software = accepted_software or (
                software["status"] == "ARCHIVED_STATIONARY_PRIOR_SOFTWARE_TESTS"
                and software.get("validation_mode") == "conditional_mean_matrix"
                and software.get("requested_test_device") == "cpu")
            if not accepted_software or software["exit_code"] != 0:
                raise ValueError("PASSED_CURRENT_SOFTWARE_RECEIPT_REQUIRED")
            rows = {row["path"]: row for row in software["files"]}
            def copy_software(name):
                target = work / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.software / name, target)
                if digest(target) != rows[name]["sha256"]:
                    raise ValueError("VALIDATED_SOURCE_CHANGED:" + name)
            copy_software("Libs/__init__.py")
            for name in ("__init__.py", "resource_probe.py"):
                copy_software("Libs/population_dynamics/" + name)
            activate_staged_software(work)
            from population_dynamics.resource_probe import probe
            def health(label):
                admission = probe(work / (label + ".json"), required_memory_gib=64)
                if admission["status"] != "PASS":
                    record.update(status="RESOURCE_BLOCKED", reasons=admission["reasons"])
                    raise RuntimeError("CONDITIONAL_FORECAST_RESOURCE_ADMISSION_FAILED")
            health("admission")
            for name in rows:
                if name.startswith("Libs/"):
                    copy_software(name)
            if source_fit:
                fit_entries = {row["path"]: row for row in reference["files"]}
                for name in ("neural_population_dynamics.yaml", "neural_population_lfads_trajectory.yaml",
                             "neural_population_lfads_prior_refit.yaml", "neural_population_matched_forecast.yaml"):
                    relative = "Libs/configs/" + name
                    src = checked_file(reference_root, relative, fit_entries)
                    destination = work / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, destination)
            if args.continuation_software is not None:
                proof = json.loads((args.continuation_software / "delivery.json").read_text())
                if proof["status"] != "PASS" or proof["exit_code"] != 0:
                    raise ValueError("PASSED_CONTINUATION_SOFTWARE_REQUIRED")
                required = {"lfads_trajectory_evaluation.py", "lfads_forecast_continuation.py",
                            "lfads_evaluator_compatibility.py"}
                for row in proof["files"]:
                    path = Path(row["path"])
                    if str(path.parent) == "Libs/population_dynamics" and path.name in required:
                        destination = work / path
                        shutil.copy2(args.continuation_software / path, destination)
                        if digest(destination) != row["sha256"]:
                            raise ValueError("CONTINUATION_SOFTWARE_CHANGED")
                        required.remove(path.name)
                if required:
                    raise ValueError("COMPLETE_VALIDATED_CONTINUATION_SOURCES_REQUIRED")
                record["continuation_software"] = str(args.continuation_software)
                record["continuation_software_delivery_sha256"] = digest(args.continuation_software / "delivery.json")
            guard = work / "run_guarded.py"
            guard.write_text(args.guard.read_text().replace("required_memory_gib=1", "required_memory_gib=64"))
            shutil.copy2(Path(__file__), work / "launcher.py")
            if source_fit:
                input_plan = stationary_input_plan(reference_root, reference)
            else:
                if reference["status"] != "COMPARED_PRIORS_DEVELOPMENT" or reference["staged_inputs"]["status"] != "VERIFIED":
                    raise ValueError("COMPLETED_REFERENCE_INPUT_MANIFEST_REQUIRED")
                input_plan = reference["staged_inputs"]["files"]
            if shutil.disk_usage(work).free < sum(row.get("bytes", 0) for row in input_plan) + 8 * 1024**3:
                raise RuntimeError("INSUFFICIENT_NODE_LOCAL_FORECAST_STORAGE")
            record["staged_files"] = 0
            last_check = time.monotonic()
            # Reuse the exact input plan of the completed two-arm comparison.
            # The evaluator independently checks its scientific bindings again.
            for row in input_plan:
                name = Path(row["path"])
                if name.is_absolute() or ".." in name.parts or name.parts[0] != "inputs":
                    raise ValueError("REFERENCE_STAGING_PATH_INVALID")
                destination = work / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = Path(row["source"])
                if "link_target" in row:
                    target = Path(row["link_target"])
                    if target.is_absolute() or ".." in target.parts or not source.is_symlink() or source.readlink() != target:
                        raise ValueError("REFERENCE_INPUT_ALIAS_CHANGED")
                    destination.symlink_to(target)
                else:
                    if source.is_symlink() or source.stat().st_size != row["bytes"]:
                        raise ValueError("REFERENCE_INPUT_SIZE_OR_TYPE_CHANGED")
                    shutil.copy2(source, destination)
                    if digest(destination) != row["sha256"]:
                        raise ValueError("REFERENCE_INPUT_CONTENT_CHANGED:" + str(source))
                record["staged_files"] += 1
                if time.monotonic() - last_check >= 30:
                    health("staging_resources"); last_check = time.monotonic()
            means = None
            if source_fit:
                record["stationary_prior_intake"] = validate_stationary_inputs(work)
            if not stationary:
                mean_delivery = json.loads((args.mean_run / "delivery.json").read_text())
                if mean_delivery["status"] != "FITTED_TEMPLATE_TIME_CONTROLLER_MEAN":
                    raise ValueError("COMPLETED_MEAN_FIT_REQUIRED")
                means = work / "inputs/conditional_mean"
                means.mkdir()
                entries = {row["path"]: row for row in mean_delivery["files"]}
                for name in ("mean_prior.json", "controller_means.npz"):
                    shutil.copy2(args.mean_run / "mean" / name, means / name)
                    if digest(means / name) != entries["mean/" + name]["sha256"]:
                        raise ValueError("FITTED_MEAN_CHANGED")
            continuation_args = []
            if args.continue_from is not None:
                parent = work / "inputs/parent"
                parent.mkdir()
                shutil.copy2(args.continue_from / "delivery.json", parent / "delivery.json")
                parent_delivery = json.loads((parent / "delivery.json").read_text())
                if parent_delivery["status"] != "RESOURCE_BLOCKED" or parent_delivery["exit_code"] != 3:
                    raise ValueError("CLOSED_RESOURCE_STOPPED_PARENT_REQUIRED")
                for row in parent_delivery["files"]:
                    name = Path(row["path"])
                    if name.is_absolute() or ".." in name.parts:
                        raise ValueError("PARENT_DELIVERY_PATH_INVALID")
                    if (str(name).startswith("candidate/forecasts/")
                            or str(name) == "phases/forecast/receipt.json"):
                        destination = parent / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(args.continue_from / name, destination)
                        if digest(destination) != row["sha256"]:
                            raise ValueError("PARENT_FORECAST_CHANGED")
                old = json.loads((parent / "candidate/forecasts/evaluation.json").read_text())
                for key in ("continuation_software", "software"):
                    if key not in parent_delivery:
                        continue
                    source = Path(parent_delivery[key]) / "Libs/population_dynamics/lfads_trajectory_evaluation.py"
                    if source.is_file() and digest(source) == old["evaluator_implementation_sha256"]:
                        shutil.copy2(source, parent / "source_evaluator.py")
                        break
                else:
                    raise ValueError("RECORDED_PARENT_EVALUATOR_SOURCE_REQUIRED")
                record["continued_from"] = str(args.continue_from)
                record["parent_delivery_sha256"] = digest(parent / "delivery.json")
                continuation_args = ["--continue-from", str(parent)]
            common = ["--acquisition", str(work / "inputs/train"), "--fields", str(work / "inputs/train_fields"),
                "--training", str(work / "inputs/training"), "--device", "cpu",
                "--inference-migration", str(work / "inputs/migration/inference_migration.json"),
                "--prior-refit", str(work / "inputs/prior_refit")]
            if means is not None:
                common += ["--conditional-prior", str(means)]
            forecast = work / "candidate/forecasts"
            phases = [
                ("forecast", [sys.executable, "-B", "-m", "Libs.population_dynamics.lfads_trajectory_evaluation", *common,
                    "--config", str(work / "Libs/configs/neural_population_matched_forecast.yaml"), "--output", str(forecast), *continuation_args]),
                ("replay", [sys.executable, "-B", "-m", "Libs.population_dynamics.lfads_forecast_replay", *common,
                    "--forecasts", str(forecast), "--output", str(work / "candidate/replay.json")])]
            if args.replay_only:
                record["reused_forecast"] = stage_completed_forecasts(parent, forecast)
                phases = [phase for phase in phases if phase[0] == "replay"]
            for label, command in phases:
                record.update(status="RUNNING_" + label.upper())
                record["commands"].append(command);save()
                result = subprocess.run([sys.executable, "-B", str(guard), label, *command], cwd=work, env=env)
                code = result.returncode
                if code:
                    record.update(status="RESOURCE_BLOCKED" if code == 3 else "FAILED", exit_code=code)
                    break
            else:
                evaluation = json.loads((forecast / "evaluation.json").read_text())
                replay = json.loads((work / "candidate/replay.json").read_text())
                if evaluation["status"] != "EVALUATED_DEVELOPMENT" or replay["status"] != "PASS":
                    raise ValueError("COMPLETED_CANDIDATE_AND_REPLAY_REQUIRED")
                record.update(status="REPLAYED_STATIONARY_PRIOR_CANDIDATE" if stationary else "REPLAYED_CONDITIONAL_MEAN_CANDIDATE",
                              completed_origins=evaluation["completed_origins"], exit_code=0)
        except Exception as error:
            code = 1
            record.update(error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
            if record["status"].startswith("RUNNING"):
                record["status"] = "FAILED"
        finally:
            record["completed_utc"] = datetime.now(timezone.utc).isoformat();save()
            args.output.mkdir(parents=True, exist_ok=False)
            files = []
            for source in sorted(work.rglob("*")):
                relative = source.relative_to(work)
                if not source.is_file() or {"inputs", "scratch", "Libs", "__pycache__"} & set(relative.parts):
                    continue
                target = args.output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                checksum = digest(source)
                if digest(target) != checksum:
                    raise ValueError("CANDIDATE_DELIVERY_CHANGED:" + str(relative))
                files.append(dict(path=str(relative), bytes=source.stat().st_size, sha256=checksum))
            (args.output / "delivery.json").write_text(json.dumps(dict(record, files=files), indent=2) + "\n")
            print(json.dumps(record), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
