"""Stage a tested boundary witness and reuse archived native history targets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

from run_population_trajectory_campaign import (
    COHORT, SOFTWARE, archive, copy_bound, digest, manifest_rows, phase, prerequisite, stage_inputs,
)


def stage_software(repo, work, software, *, cached=False):
    proof = json.loads((software / "delivery.json").read_text())
    counts = proof["tests"]
    if (proof["status"] != "PASS" or proof["exit_code"] != 0 or counts["tests"] <= 0
            or any(counts[key] for key in ("failures", "errors", "skipped"))):
        raise ValueError("complete accepted boundary software tests required")
    rows = manifest_rows(proof)
    required = ("test_population_boundary_closure.py", "test_closure_cells.py", "test_population_state.py")
    for name in required:
        if "Libs/population_dynamics/" + name not in rows:
            raise ValueError("boundary, cell and existing-state tests required")
    for name, row in rows.items():
        if name.startswith("Libs/") and "/test_" not in name and name.endswith((".py", ".yaml")):
            copy_bound(software / name, work / name, row)
    accepted, _, source_rows = prerequisite(repo, SOFTWARE, "606023")
    name = "Libs/population_dynamics/resource_probe.py"
    copy_bound(accepted / name, work / name, source_rows[name])
    # Existing checkpoints compare these implementation hashes at load time.
    for name in (() if cached else ("population_state", "refractory_observation", "native_renewal_data",
                 "poisson_state_space", "native_count_windows")):
        relative = "Libs/population_dynamics/" + name + ".py"
        if rows[relative]["sha256"] != source_rows[relative]["sha256"]:
            raise ValueError("checkpoint numerical implementation differs from tested source")
    for source, destination in ((Path(__file__), "deployment.py"),
                                (Path(__file__).with_suffix(".sbatch"), "launcher.sbatch"),
                                (Path(__file__).with_name("run_population_trajectory_campaign.py"), "campaign_utilities.py")):
        copy_bound(source, work / destination, dict(bytes=source.stat().st_size, sha256=digest(source)))
    receipt = software / "delivery.json"
    copy_bound(receipt, work / "prerequisites/software_delivery.json",
               dict(bytes=receipt.stat().st_size, sha256=digest(receipt)))
    spec = importlib.util.spec_from_file_location("boundary_resource_probe", work / "Libs/population_dynamics/resource_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.probe


def stage_history(history, work):
    proof = json.loads((history / "delivery.json").read_text())
    if (proof["status"] != "ARCHIVED_INDEPENDENTLY_REPLAYED_DEVELOPMENT_PROBE"
            or proof["calibration_outcomes_accessed"] is not False or proof["test_outcomes_accessed"] is not False
            or digest(history / "payload_manifest.json") != proof["payload_manifest_sha256"]):
        raise ValueError("accepted development history delivery required")
    manifest = json.loads((history / "payload_manifest.json").read_text())
    rows = manifest_rows(manifest)
    for name in ("features/features.json", "features/features.npz"):
        copy_bound(history / name, work / "inputs/history" / name, rows[name])
    for name in ("delivery.json", "payload_manifest.json"):
        source = history / name
        copy_bound(source, work / "inputs/history" / name,
                   dict(bytes=source.stat().st_size, sha256=digest(source)))
    copy_bound(history / "delivery.json", work / "prerequisites/history_delivery.json",
               dict(bytes=(history / "delivery.json").stat().st_size, sha256=digest(history / "delivery.json")))


def stage_checkpoints(campaign, work):
    for seed in (41, 42, 43):
        root = campaign / f"seed_{seed}"
        proof = json.loads((root / "delivery.json").read_text())
        if (proof["status"] != "PASS" or proof["exit_code"] != 0 or proof["training_seed"] != seed
                or proof["calibration_or_test_accessed"] is not False):
            raise ValueError("all three complete development campaign cells required")
        rows = manifest_rows(proof)
        for name in ("training/run.json", "training/best.pt"):
            copy_bound(root / name, work / "inputs/campaign" / f"seed_{seed}" / name, rows[name])
        source = root / "delivery.json"
        copy_bound(source, work / "prerequisites" / f"seed_{seed}_delivery.json",
                   dict(bytes=source.stat().st_size, sha256=digest(source)))


def stage_cached_boundaries(boundary_run, work):
    proof = json.loads((boundary_run / "delivery.json").read_text())
    if proof["status"] != "PASS" or proof["exit_code"] != 0 or proof["mode"] != "inference_only":
        raise ValueError("accepted original boundary extraction required")
    rows = manifest_rows(proof)
    names = ["analysis/analysis.json"]
    for seed in (41, 42, 43):
        names.extend(f"analysis/seed_{seed}/{name}" for name in ("boundaries.npz", "witness.json"))
    for name in names:
        copy_bound(boundary_run / name, work / "inputs/boundary_run" / name, rows[name])
    source = boundary_run / "delivery.json"
    row = dict(bytes=source.stat().st_size, sha256=digest(source))
    copy_bound(source, work / "inputs/boundary_run/delivery.json", row)
    copy_bound(source, work / "prerequisites/boundary_delivery.json", row)


def run(args):
    cached = args.cached_boundary_run is not None
    if cached and (args.prepare_only or args.campaign is not None):
        raise ValueError("cached readout excludes boundary preparation and neural campaign")
    if not cached and args.campaign is None:
        raise ValueError("boundary inference requires its neural campaign")
    node, partition = os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION")
    expected = (("cpu04", "cpu"),) if args.prepare_only else (("cpu03", "gpu"),("cpu04", "cpu"))
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS") or (node, partition) not in expected:
        raise ValueError("CPU04 preparation or allocated CPU03/CPU04 inference-only execution required")
    if Path("/tmp/neuronsim") not in args.work.parents:
        raise ValueError("node-local boundary scratch required")
    job = os.environ["SLURM_JOB_ID"]
    target = args.repo / "Output/neural_population_dynamics/model/boundary_closure" / ("run_" + job)
    if target.exists():
        raise FileExistsError("existing boundary evidence must be preserved")
    record = dict(status="RUNNING", exit_code=None, job_id=job, node=node,
                  mode="cached_rbf_readout" if cached else "prepare" if args.prepare_only else "inference_only",
                  campaign=str(args.campaign) if args.campaign else None,
                  cached_boundary_run=str(args.cached_boundary_run) if cached else None,
                  history=str(args.history), software=str(args.software), training_seeds=[41, 42, 43],
                  training_or_generation_repeated=False, scientific_claim_qualified=False,
                  formal_tolerances_qualified=False, state_selection_enabled=False,
                  calibration_or_test_accessed=False, started_utc=datetime.now(timezone.utc).isoformat())
    code = 2
    try:
        base_probe = stage_software(args.repo, args.work, args.software, cached=cached)
        def probe(path, required_memory_gib=32):
            return base_probe(path, required_memory_gib=4 if cached else required_memory_gib)
        (args.work / "scratch").mkdir()
        health = probe(args.work / "staging_preflight.json", required_memory_gib=32)
        if health["status"] != "PASS":
            record.update(status="RESOURCE_BLOCKED", reasons=health["reasons"])
            code = 3
        else:
            stage_history(args.history, args.work)
            command = [sys.executable, "-m", "population_dynamics.population_boundary_closure",
                       "--acquisition", "inputs/train", "--fields", "inputs/train_fields",
                       "--history", "inputs/history", "--config", "Libs/configs/neural_population_boundary_closure.yaml"]
            if args.prepare_only or cached:
                source = args.repo / COHORT
                rows = manifest_rows(json.loads((source / "delivery.json").read_text()))
                for name in ("train/design.json", "train_fields/dataset.json"):
                    copy_bound(source / name, args.work / "inputs" / name, rows[name])
                if cached:
                    stage_cached_boundaries(args.cached_boundary_run, args.work)
                    command.extend(("--cached-boundary-run", "inputs/boundary_run", "--output", "analysis"))
                else:
                    command.append("--validate-only")
            else:
                stage_checkpoints(args.campaign, args.work)
                stage_inputs(args.repo, args.work)
                command.extend(("--campaign", "inputs/campaign", "--output", "analysis"))
            code = phase(args.work, "cached_readout" if cached else "validate_inputs" if args.prepare_only else "boundary_inference", command, probe)
            record["status"] = ("PREPARED" if args.prepare_only else "PASS") if code == 0 else "RESOURCE_BLOCKED" if code == 3 else "FAILED"
    except Exception as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error))
    record["exit_code"] = code
    archive(args.work, target, record)
    (args.work / "archive_complete").write_text(str(target))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "work", "software", "history"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--cached-boundary-run", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
