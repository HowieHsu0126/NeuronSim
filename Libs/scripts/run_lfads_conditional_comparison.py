"""Stage the completed LFADS arms and run their prospective conditional-mean comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def accepted_software(receipt):
    if receipt.get("exit_code") != 0:
        return False
    return receipt.get("status") == "PASS" or (
        receipt.get("status") == "ARCHIVED_STATIONARY_PRIOR_SOFTWARE_TESTS"
        and receipt.get("validation_mode") in {"conditional_mean_matrix", "conditional_mean_comparison"}
        and receipt.get("requested_test_device") == "cpu")


def reference_prefix(receipt):
    if receipt.get("status") == "COMPARED_PRIORS_DEVELOPMENT":
        return "fitted_prior"
    if receipt.get("status") == "REPLAYED_STATIONARY_PRIOR_CANDIDATE" and receipt.get("exit_code") == 0:
        return "candidate"
    raise ValueError("COMPLETED_STATIONARY_REFERENCE_REQUIRED")


def stationary_readout_plan(fit_run, reference, evaluation):
    """Select checkpoint/map metadata from the existing authenticated fit plan."""
    from Libs.scripts.run_lfads_conditional_forecast import stationary_input_plan

    if fit_run is None:
        raise ValueError("FIT_RUN_REQUIRED_WITHOUT_REFERENCE_READOUT")
    fit_run = Path(fit_run)
    delivery_path = fit_run / "delivery.json"
    if digest(delivery_path) != reference.get("reference_delivery_sha256"):
        raise ValueError("REFERENCE_PARENT_FIT_DELIVERY_CHANGED")
    fit = json.loads(delivery_path.read_text())
    entries = {row["path"]: row for row in fit["files"]}
    if len(entries) != len(fit["files"]):
        raise ValueError("DUPLICATE_FIT_DELIVERY_PATH")
    prior = evaluation["prior_refit"]
    if (fit.get("training_seed") != evaluation["training_seed"]
            or {"budget_20": 20, "budget_80": 80}.get(fit.get("network_variant")) != evaluation["effective_training_epoch_budget"]
            or entries["cache/posterior_cache.json"]["sha256"] != prior["posterior_cache_sha256"]
            or entries["cache/prior_parameters.npz"]["sha256"] != prior["prior_parameters_sha256"]):
        raise ValueError("REFERENCE_FIT_CELL_OR_PRIOR_BINDING_CHANGED")
    checkpoint = Path(evaluation["selected_checkpoint"])
    artifacts = evaluation["selected_model_artifacts"]
    if (checkpoint.is_absolute() or ".." in checkpoint.parts or not checkpoint.parts
            or not {"model.json", "weights.npz"}.issubset(artifacts)
            or any(Path(name).is_absolute() or ".." in Path(name).parts for name in artifacts)):
        raise ValueError("BOUND_SELECTED_CHECKPOINT_REQUIRED")
    selected = {str(Path("inputs/training") / checkpoint / name): checksum for name, checksum in artifacts.items()}
    field_prefix = "inputs/train_fields/train_block_014_familiar_0/"
    required = set(selected) | {field_prefix + "summary.json"}
    plan = stationary_input_plan(fit_run, fit)
    rows = [row for row in plan if row["path"] in required]
    if len(rows) != len(required) or {row["path"] for row in rows} != required:
        raise ValueError("COMPLETE_SELECTED_READOUT_INPUTS_REQUIRED")
    for row in rows:
        if row["path"] in selected and row["sha256"] != selected[row["path"]]:
            raise ValueError("SELECTED_READOUT_CHECKPOINT_CHANGED")
    # Fitting stages field summaries, while readout also needs their bound unit map.
    summary_row = next(row for row in rows if row["path"] == field_prefix + "summary.json")
    summary_path = Path(summary_row["source"])
    expected_summary = evaluation["training_binding"]["episode_bindings"]["train_block_014_familiar_0"]["field_summary_sha256"]
    if digest(summary_path) != summary_row["sha256"] or summary_row["sha256"] != expected_summary:
        raise ValueError("READOUT_SOURCE_FIELD_SUMMARY_CHANGED")
    summary = json.loads(summary_path.read_text())
    map_source = summary_path.with_name("spatial_fields.npz")
    if not map_source.is_file():
        raise ValueError("BOUND_SOURCE_FIELD_MAP_MISSING")
    rows.append(dict(source=str(map_source), path=field_prefix + "spatial_fields.npz",
        bytes=map_source.stat().st_size, sha256=summary["artifacts"]["spatial_fields.npz"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-run", "candidate-run", "mean-run",
                 "software", "updated-software", "guard", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-readout", type=Path)
    parser.add_argument("--fit-run", type=Path)
    args = parser.parse_args()
    if args.reference_readout is None and args.fit_run is None:
        raise ValueError("FIT_RUN_REQUIRED_WITHOUT_REFERENCE_READOUT")
    if (os.environ.get("SLURMD_NODENAME") not in {"cpu03", "cpu04"}
            or not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS")):
        raise ValueError("AUTHORIZED_CPU_ONLY_SLURM_ALLOCATION_REQUIRED")
    if args.output.exists():
        raise FileExistsError(args.output)
    record = dict(status="RUNNING", job_id=os.environ["SLURM_JOB_ID"], node=os.environ["SLURMD_NODENAME"],
        command=sys.argv, started_utc=datetime.now(timezone.utc).isoformat(),
        scientific_claim_qualified=False, formal_test_accessed=False, production_episodes=0,
        new_forecasts=0, new_posterior_draws=0, reference_readout_recomputed=args.reference_readout is None,
        cross_node_runtime_comparison_qualified=False,
        software_deliveries={str(path): digest(path / "delivery.json") for path in (args.software, args.updated_software)})
    code = 1
    with tempfile.TemporaryDirectory(prefix="neuronsim-conditional-comparison-" + record["job_id"] + ".", dir="/tmp") as name:
        work = Path(name)
        (work / "scratch").mkdir()
        (work / "phases").mkdir()
        def save():
            (work / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
        try:
            baseline = json.loads((args.software / "delivery.json").read_text())
            update = json.loads((args.updated_software / "delivery.json").read_text())
            if not all(accepted_software(x) for x in (baseline, update)):
                raise ValueError("PASSED_SOFTWARE_RECEIPTS_REQUIRED")
            def stage(root, receipt, names, destination):
                entries = {row["path"]: row for row in receipt["files"]}
                if len(entries) != len(receipt["files"]):
                    raise ValueError("DUPLICATE_DELIVERY_PATH")
                for name in names:
                    relative = Path(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("DECLARED_RELATIVE_PATH_REQUIRED")
                    row = entries[name]
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(root / relative, target)
                    if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                        raise ValueError("DECLARED_INPUT_CHANGED:" + str(root / relative))
            stage(args.software, baseline, ["Libs/population_dynamics/__init__.py",
                                           "Libs/population_dynamics/resource_probe.py"], work)
            sys.path[:0] = [str(work), str(work / "Libs")]
            from population_dynamics.resource_probe import probe
            admission = probe(work / "admission.json", required_memory_gib=8)
            if admission["status"] != "PASS":
                record.update(status="RESOURCE_BLOCKED", reasons=admission["reasons"])
                raise RuntimeError("COMPARISON_RESOURCE_ADMISSION_FAILED")
            stage(args.software, baseline, [row["path"] for row in baseline["files"] if row["path"].startswith("Libs/")], work)
            stage(args.updated_software, update, [row["path"] for row in update["files"] if row["path"].startswith("Libs/")], work)
            shutil.copy2(Path(__file__), work / "launcher.py")
            guard = work / "run_guarded.py"
            guard.write_text(args.guard.read_text().replace("required_memory_gib=1", "required_memory_gib=8"))
            references = {}
            for label, root, status in (
                ("reference", args.reference_run, "COMPARED_PRIORS_DEVELOPMENT"),
                ("candidate", args.candidate_run, "REPLAYED_CONDITIONAL_MEAN_CANDIDATE"),
                ("readout", args.reference_readout, "RECONSTRUCTED_REPLAYED_CONDITIONAL_MEANS"),
                ("mean", args.mean_run, "FITTED_TEMPLATE_TIME_CONTROLLER_MEAN")):
                if root is None:
                    continue
                receipt = json.loads((root / "delivery.json").read_text())
                if label == "reference":
                    ref_prefix = reference_prefix(receipt)
                elif receipt["status"] != status:
                    raise ValueError("COMPLETED_INPUT_REQUIRED:" + label)
                references[label] = receipt
                record[label + "_delivery"] = dict(path=str(root / "delivery.json"), sha256=digest(root / "delivery.json"))
            if args.reference_readout is not None and references["readout"]["source_delivery_sha256"] != digest(args.reference_run / "delivery.json"):
                raise ValueError("REFERENCE_READOUT_SOURCE_DELIVERY_CHANGED")
            if shutil.disk_usage(work).free < 8 * 1024**3:
                raise RuntimeError("INSUFFICIENT_LOCAL_COMPARISON_STORAGE")
            plan = (
                ("reference", args.reference_run, [row["path"] for row in references["reference"]["files"]
                                                  if row["path"].startswith(ref_prefix + "/")]),
                ("candidate", args.candidate_run, [row["path"] for row in references["candidate"]["files"]
                                                  if row["path"].startswith("candidate/")]),
                ("readout", args.reference_readout, ["analysis/analysis.json", "analysis/field_error_by_origin.csv"]),
                ("mean", args.mean_run, ["mean/mean_prior.json", "mean/controller_means.npz"]),
            )
            for label, root, names in plan:
                if root is None:
                    continue
                target = work / "inputs" / label
                target.mkdir(parents=True)
                shutil.copy2(root / "delivery.json", target / "delivery.json")
                stage(root, references[label], names, target)
                resource = probe(work / ("staging_" + label + ".json"), required_memory_gib=8)
                if resource["status"] != "PASS":
                    record.update(status="RESOURCE_BLOCKED", reasons=resource["reasons"])
                    raise RuntimeError("COMPARISON_STAGING_RESOURCE_BLOCKED")
            reference_evaluation = json.loads((work / "inputs/reference" / ref_prefix / "forecasts/evaluation.json").read_text())
            if args.reference_readout is None:
                readout_inputs = stationary_readout_plan(args.fit_run, references["reference"], reference_evaluation)
                record["fit_delivery"] = dict(path=str(args.fit_run / "delivery.json"), sha256=digest(args.fit_run / "delivery.json"))
                record["checkpoint_and_field_inputs"] = readout_inputs
            else:
                readout_inputs = references["readout"]["checkpoint_and_field_inputs"]
            checkpoint = field_view = None
            for row in readout_inputs:
                relative = Path(row["path"])
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "inputs":
                    raise ValueError("READOUT_INPUT_PATH_INVALID")
                target = work / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(row["source"], target)
                if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                    raise ValueError("READOUT_CHECKPOINT_OR_MAP_CHANGED")
                if relative.name == "model.json":
                    checkpoint = target.parent
                if relative.name == "spatial_fields.npz":
                    field_view = target.parent
            if checkpoint is None or field_view is None:
                raise ValueError("BOUND_READOUT_INPUTS_REQUIRED")
            candidate_evaluation = json.loads((work / "inputs/candidate/candidate/forecasts/evaluation.json").read_text())
            record["execution_environment"] = dict(python=sys.version, numpy=importlib.metadata.version("numpy"),
                reference_tensorflow_versions=sorted({row["execution_device"].get("tensorflow_version", "unrecorded")
                    for row in reference_evaluation["graph_lifetimes"]}),
                candidate_tensorflow_versions=sorted({row["execution_device"].get("tensorflow_version", "unrecorded")
                    for row in candidate_evaluation["graph_lifetimes"]}),
                thread_settings={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
            lineage = candidate_evaluation.get("forecast_continuation", {})
            record["attempt_provenance"] = dict(candidate_node=references["candidate"].get("node"),
                inherited_origins=lineage.get("inherited_origins", 0),
                newly_computed_origins=candidate_evaluation["completed_origins"] - lineage.get("inherited_origins", 0),
                parent_job_id=lineage.get("parent_job_id"),
                parent_attempt_started_utc=lineage.get("parent_attempt_started_utc"),
                parent_attempt_completed_utc=lineage.get("parent_attempt_completed_utc"),
                inherited_origin_artifacts=sorted(name for name in lineage.get("inherited_artifacts", {}) if name.startswith("train_block_")),
                candidate_parent=references["candidate"].get("continued_from"),
                candidate_continuation_software=references["candidate"].get("continuation_software"),
                note="Inherited and new origins keep their original costs. This comparison makes no cross-node runtime claim.")
            command = [sys.executable, "-B", "-m", "population_dynamics.lfads_conditional_comparison",
                "--reference", str(work / "inputs/reference" / ref_prefix),
                "--candidate", str(work / "inputs/candidate/candidate"),
                "--mean", str(work / "inputs/mean/mean"),
                "--checkpoint", str(checkpoint), "--field-view", str(field_view), "--output", str(work / "analysis")]
            if args.reference_readout is not None:
                command.extend(["--reference-readout", str(work / "inputs/readout")])
            record.update(status="ANALYZING", analysis_command=command);save()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.pathsep.join((str(work), str(work / "Libs"))), TMPDIR=str(work / "scratch"),
                PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            result = subprocess.run([sys.executable, "-B", str(guard), "comparison", *command], cwd=work, env=env)
            code = result.returncode
            record.update(status="RESOURCE_BLOCKED" if code == 3 else "FAILED" if code else
                          "DESCRIBED_REPLAYED_CONDITIONAL_PRIOR_COMPARISON", exit_code=code)
            if code == 0:
                analysis = json.loads((work / "analysis/analysis.json").read_text())
                record["route"] = analysis["route"]
                record["paired_origins"] = analysis["paired_origins"]
        except Exception as error:
            record.update(error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
            if record["status"] in {"RUNNING", "ANALYZING"}:
                record["status"] = "FAILED"
            code = 3 if record["status"] == "RESOURCE_BLOCKED" else 1
        finally:
            record["completed_utc"] = datetime.now(timezone.utc).isoformat();save()
            args.output.mkdir(parents=True, exist_ok=False)
            files = []
            for source in sorted(work.rglob("*")):
                relative = source.relative_to(work)
                if not source.is_file() or {"inputs", "Libs", "scratch", "__pycache__"} & set(relative.parts):
                    continue
                target = args.output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                checksum = digest(source)
                if digest(target) != checksum:
                    raise ValueError("COMPARISON_DELIVERY_CHANGED:" + str(relative))
                files.append(dict(path=str(relative), bytes=source.stat().st_size, sha256=checksum))
            (args.output / "delivery.json").write_text(json.dumps(dict(record, files=files), indent=2) + "\n")
            print(json.dumps({k:v for k,v in record.items() if k != "analysis_command"}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
