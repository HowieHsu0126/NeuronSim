"""Stage a tested population analysis and reuse delivered baseline aggregates.

Run inside Slurm. Preparation checks small metadata and software only. Complete
campaign analysis requires all three deliveries and reads the same native cohort.
Representation, budget and prior-objective comparisons reuse two delivered
analysis wrappers without cohort reads. Analysis does not fit or generate samples
and does not access calibration or test outcomes.
"""
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


def stage_analysis(repo: Path, work: Path, software: Path, baselines: list[Path]):
    proof = json.loads((software / "delivery.json").read_text())
    if (proof["status"] != "PASS" or proof["exit_code"] != 0
            or proof["tests"]["tests"] < 35
            or any(proof["tests"][name] != 0 for name in ("failures", "errors", "skipped"))):
        raise ValueError("complete accepted analysis software with at least the existing35-test suite required")
    index = manifest_rows(proof)
    sources = [name for name in index if name.startswith("Libs/")
               and "/test_" not in name and name.endswith((".py", ".yaml"))]
    for name in sources:
        copy_bound(software / name, work / name, index[name])
    for source, destination in ((Path(__file__), "deployment.py"),
                                (Path(__file__).with_name("run_population_forecast_analysis.sbatch"), "launcher.sbatch"),
                                (Path(__file__).with_name("run_population_trajectory_campaign.py"), "campaign_utilities.py")):
        copy_bound(source, work / destination, dict(bytes=source.stat().st_size, sha256=digest(source)))
    accepted, _, source_rows = prerequisite(repo, SOFTWARE, "606023")
    name = "Libs/population_dynamics/resource_probe.py"
    copy_bound(accepted / name, work / name, source_rows[name])
    (work / "prerequisites").mkdir()
    copy_bound(software / "delivery.json", work / "prerequisites/software_delivery.json",
               dict(bytes=(software / "delivery.json").stat().st_size,
                    sha256=digest(software / "delivery.json")))
    for i, baseline in enumerate(baselines):
        proof = json.loads((baseline / "delivery.json").read_text())
        if (proof["status"] != "ARCHIVED_VERIFIED_DEVELOPMENT"
                or proof["training_seeds"] != [41, 42, 43]
                or proof["calibration_outcomes_accessed"] is not False
                or proof["test_outcomes_accessed"] is not False):
            raise ValueError("delivered development-only baseline required")
        rows = manifest_rows(proof)
        for name in ("analysis/analysis.json", "analysis/forecast_all_seeds.csv",
                     "analysis/costs_by_origin.csv"):
            copy_bound(baseline / name, work / "baselines" / str(i) / name, rows[name])
        copy_bound(baseline / "delivery.json", work / "baselines" / str(i) / "delivery.json",
                   dict(bytes=(baseline / "delivery.json").stat().st_size,
                        sha256=digest(baseline / "delivery.json")))
    module_path = work / "Libs/population_dynamics/resource_probe.py"
    spec = importlib.util.spec_from_file_location("analysis_resources", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.probe


def stage_population_analyses(work: Path, roots: list[Path]):
    """Stage completed development aggregates; preserve their source bindings."""
    if len(roots) not in (1, 2):
        raise ValueError("one or two completed population analyses required")
    for i, root in enumerate(roots):
        receipt = root / "delivery.json"
        proof = json.loads(receipt.read_text())
        if (proof["status"] != "PASS" or proof["exit_code"] != 0
                or proof["mode"] != "complete" or proof["training_seeds"] != [41, 42, 43]
                or proof["scientific_claim_qualified"] is not False
                or proof["calibration_or_test_accessed"] is not False):
            raise ValueError("complete development-only population analysis required")
        rows = manifest_rows(proof)
        for name in ("analysis/analysis.json", "analysis/forecast_all_seeds.csv",
                     "analysis/costs_by_origin.csv"):
            copy_bound(root / name, work / "comparisons" / str(i) / name, rows[name])
        copy_bound(receipt, work / "comparisons" / str(i) / "delivery.json",
                   dict(bytes=receipt.stat().st_size, sha256=digest(receipt)))


def run(args):
    comparisons = getattr(args, "population_analysis", [])
    campaign = getattr(args, "campaign", None)
    budget_comparison = getattr(args, "budget_comparison", False)
    prior_objective_comparison = getattr(args, "prior_objective_comparison", False)
    if budget_comparison and prior_objective_comparison:
        raise ValueError("choose budget comparison or prior-objective comparison")
    if (budget_comparison or prior_objective_comparison) and (len(comparisons) != 2 or campaign is not None
                              or args.prepare_only or args.baseline_analysis):
        raise ValueError("budget or prior-objective comparison requires exactly two saved population analyses")
    if comparisons:
        if (campaign is not None or args.prepare_only
                or not (len(comparisons) == 2 and not args.baseline_analysis
                        or len(comparisons) == 1 and args.baseline_analysis)):
            raise ValueError("two population analyses or one analysis with baselines required")
    elif campaign is None:
        raise ValueError("campaign or two population analyses required")
    node, partition = os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION")
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS")
            or ((node, partition) != ("cpu04", "cpu") if args.prepare_only
                else (node, partition) not in (("cpu03", "gpu"),("cpu04", "cpu")))):
        raise ValueError("CPU04 preparation or allocated CPU03/CPU04 complete analysis required")
    if Path("/tmp/neuronsim") not in args.work.parents:
        raise ValueError("node-local analysis scratch required")
    job = os.environ["SLURM_JOB_ID"]
    target = args.repo / "Output/neural_population_dynamics/model/forecast_analysis" / ("run_" + job)
    if target.exists():
        raise FileExistsError("existing analysis evidence must be preserved")
    record = dict(status="RUNNING", exit_code=None, job_id=job, node=node,
                  mode="comparison" if comparisons else "prepare" if args.prepare_only else "complete",
                  campaign=str(campaign) if campaign is not None else None, software=str(args.software),
                  training_seeds=[41, 42, 43], scientific_claim_qualified=False,
                  source_fidelity_qualified=False, calibration_or_test_accessed=False,
                  started_utc=datetime.now(timezone.utc).isoformat())
    if comparisons:
        record["population_analysis"] = [str(root) for root in comparisons]
        record["budget_comparison"] = budget_comparison
        record["prior_objective_comparison"] = prior_objective_comparison
    code = 2
    try:
        probe = stage_analysis(args.repo, args.work, args.software, args.baseline_analysis)
        (args.work / "scratch").mkdir()
        health = probe(args.work / "staging_preflight.json", required_memory_gib=32)
        if health["status"] != "PASS":
            record.update(status="RESOURCE_BLOCKED", exit_code=3, reasons=health["reasons"])
            archive(args.work, target, record)
            (args.work / "archive_complete").write_text(str(target))
            return 3
        if comparisons:
            stage_population_analyses(args.work, comparisons)
            command = [sys.executable, "-m", "population_dynamics.population_forecast_analysis",
                       "--population-analysis", "comparisons/0"]
            if len(comparisons) == 2:
                command += ["--population-analysis", "comparisons/1"]
                if budget_comparison:
                    command.append("--budget-comparison")
                if prior_objective_comparison:
                    command.append("--prior-objective-comparison")
            else:
                for i in range(len(args.baseline_analysis)):
                    command += ["--baseline-analysis", "baselines/" + str(i)]
            command += ["--output", "analysis"]
            code = phase(args.work, "analysis", command, probe)
            record["status"] = "PASS" if code == 0 else "RESOURCE_BLOCKED" if code == 3 else "FAILED"
        elif args.prepare_only:
            source = args.repo / COHORT
            proof = json.loads((source / "delivery.json").read_text())
            rows = manifest_rows(proof)
            if (proof["status"] != "VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE"
                    or proof["completed_episodes"] != 72 or proof["cortical_units"] != 100000):
                raise ValueError("complete accepted native development cohort required")
            for name in ("train/design.json", "train_fields/dataset.json"):
                copy_bound(source / name, args.work / "inputs" / name, rows[name])
            command = [sys.executable, "-m", "population_dynamics.population_forecast_analysis", "--help"]
            code = phase(args.work, "entry_point", command, probe)
            record.update(status="PREPARED" if code == 0 else "FAILED",
                          native_input_files=len(rows), native_input_bytes=sum(row["bytes"] for row in rows.values()),
                          campaign_execution_repeated=False)
        else:
            # The analysis loader checks every seed delivery and exact source
            # bindings. Read completed archived cells; do not regenerate samples.
            for seed in (41, 42, 43):
                receipt = args.campaign / f"seed_{seed}/delivery.json"
                if not receipt.is_file() or json.loads(receipt.read_text())["status"] != "PASS":
                    raise ValueError("three completed campaign deliveries required before staging")
            stage_inputs(args.repo, args.work)
            command = [sys.executable, "-m", "population_dynamics.population_forecast_analysis",
                       "--forecasts", str(args.campaign), "--acquisition", "inputs/train",
                       "--fields", "inputs/train_fields", "--output", "analysis", "--population-campaign"]
            for i in range(len(args.baseline_analysis)):
                command.extend(("--baseline-analysis", "baselines/" + str(i)))
            code = phase(args.work, "analysis", command, probe)
            record["status"] = "PASS" if code == 0 else "RESOURCE_BLOCKED" if code == 3 else "FAILED"
    except Exception as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error))
    record["exit_code"] = code
    archive(args.work, target, record)
    (args.work / "archive_complete").write_text(str(target))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "work", "software"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--campaign", type=Path)
    mode.add_argument("--population-analysis", type=Path, action="append", default=[])
    parser.add_argument("--baseline-analysis", type=Path, action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--budget-comparison", action="store_true",
                        help="Pair an original recurrent result and its bound budget continuation")
    parser.add_argument("--prior-objective-comparison", action="store_true",
                        help="Compare a prior refit with its parent, or two prior objectives with the same parent; include inherited training costs")
    args = parser.parse_args()
    if args.budget_comparison and args.prior_objective_comparison:
        parser.error("choose budget comparison or prior-objective comparison")
    if (args.budget_comparison or args.prior_objective_comparison) and (len(args.population_analysis) != 2 or args.campaign is not None
                                  or args.prepare_only or args.baseline_analysis):
        parser.error("budget or prior-objective comparison requires exactly two saved population analyses")
    if args.population_analysis and (args.prepare_only or not (
            len(args.population_analysis) == 2 and not args.baseline_analysis
            or len(args.population_analysis) == 1 and args.baseline_analysis)):
        parser.error("two population analyses or one analysis with baselines required")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
