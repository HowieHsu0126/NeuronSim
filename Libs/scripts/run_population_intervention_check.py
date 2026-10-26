"""Stage a bounded real-source pulse-search check on CPU04."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from run_population_boundary_closure import stage_checkpoints

from run_population_trajectory_campaign import (
    COHORT, EPISODE, SOFTWARE, archive, copy_bound, digest, manifest_rows, phase, prerequisite,
)


def run(args):
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURMD_NODENAME") != "cpu04"
            or os.environ.get("SLURM_JOB_PARTITION") != "cpu" or os.environ.get("SLURM_JOB_GPUS")
            or Path("/tmp/neuronsim") not in args.work.parents):
        raise ValueError("CPU04 allocation and node-local scratch required")
    job = os.environ["SLURM_JOB_ID"]
    kind = "intervention_development" if args.campaign else "intervention_replay" if args.replay_from else "intervention_check"
    target = args.repo / "Output/neural_population_dynamics/model" / kind / ("run_" + job)
    if target.exists():
        raise FileExistsError("existing intervention evidence must be preserved")
    record = dict(status="RUNNING", exit_code=None, job_id=job, node="cpu04", training_performed=False,
                  mechanistic_simulation_performed=False, scientific_claim_qualified=False,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  purpose="development_prepare" if args.prepare_only else "development_search" if args.campaign
                  else "saved_array_replay" if args.replay_from else "engineering_search")
    code = 2
    try:
        proof = json.loads((args.software / "delivery.json").read_text())
        if (proof["status"] != "PASS" or proof["exit_code"] != 0
                or proof["tests"] != dict(tests=26, failures=0, errors=0, skipped=0)):
            raise ValueError("accepted 11-test plus 15-subtest intervention software required")
        rows = manifest_rows(proof)
        for name, row in rows.items():
            if name.startswith("Libs/") and name.endswith(".py") and ("/test_" not in name
                    or (args.replay_from or args.campaign) and name.endswith("test_population_intervention.py")):
                copy_bound(args.software / name, args.work / name, row)
        # This check runner is exercised through its real CLI below.
        search_rows = None
        if args.replay_from:
            search = json.loads((args.replay_from / "delivery.json").read_text())
            if (search["status"] != "PASS" or search["exit_code"] != 0
                    or search["training_performed"] or search["mechanistic_simulation_performed"]):
                raise ValueError("accepted inference-only pulse-search evidence required")
            search_rows = manifest_rows(search)
        prepared_rows = None
        if args.development_preparation:
            prepared = json.loads((args.development_preparation / "delivery.json").read_text())
            if (prepared["status"] != "PREPARED" or prepared["exit_code"] != 0
                    or prepared["purpose"] != "development_prepare" or prepared["training_seeds"] != [41, 42, 43]):
                raise ValueError("accepted three-seed intervention preparation required")
            prepared_rows = manifest_rows(prepared)
            source = args.development_preparation / "delivery.json"
            copy_bound(source, args.work / "prerequisites/development_preparation.json",
                       dict(bytes=source.stat().st_size, sha256=digest(source)))
            record["development_preparation_job"] = prepared["job_id"]
        config_name = ("neural_population_intervention_development.yaml" if args.campaign
                       else "neural_population_intervention_check.yaml")
        for name in ("Libs/population_dynamics/population_intervention_check.py",
                     "Libs/configs/" + config_name):
            source_root = args.development_preparation or args.replay_from or args.repo
            source = source_root / name
            index = prepared_rows or search_rows
            if args.replay_from and name.endswith("population_intervention_check.py"):
                # The current replay also supports development configuration.
                # Reuse its tested validator module while original result and
                # checkpoint bindings remain tied to the source search receipt.
                source_root = args.repo / "Output/neural_population_dynamics/model/intervention_replay_software/run_606514"
                proof = json.loads((source_root / "delivery.json").read_text())
                if proof["status"] != "PASS" or proof["exit_code"] != 0:
                    raise ValueError("accepted current replay validator software required")
                source = source_root / name
                index = manifest_rows(proof)
            row = index[name] if index else dict(bytes=source.stat().st_size, sha256=digest(source))
            copy_bound(source, args.work / name, row)
        if args.campaign:
            if args.prepare_only:
                relative = "Libs/configs/neural_population_intervention_check.yaml"
                copy_bound(args.repo / relative, args.work / relative,
                           dict(bytes=(args.repo / relative).stat().st_size, sha256=digest(args.repo / relative)))
                relative = "Libs/population_dynamics/test_population_intervention_development.py"
                source = args.repo / relative
                copy_bound(source, args.work / relative, dict(bytes=source.stat().st_size, sha256=digest(source)))
            if not args.prepare_only:
                stage_checkpoints(args.campaign, args.work)
                replay_root = args.repo / "Output/neural_population_dynamics/model/intervention_replay_software/run_606514"
                replay_proof = json.loads((replay_root / "delivery.json").read_text())
                replay_rows = manifest_rows(replay_proof)
                if replay_proof["status"] != "PASS" or replay_proof["exit_code"] != 0:
                    raise ValueError("accepted development saved-array replay software required")
                copy_bound(replay_root / "tests.xml", args.work / "prerequisites/replay_tests.xml", replay_rows["tests.xml"])
                suites = ET.parse(args.work / "prerequisites/replay_tests.xml").getroot().iter("testsuite")
                counts = {key:0 for key in ("tests", "failures", "errors", "skipped")}
                for suite in suites:
                    for key in counts:
                        counts[key] += int(suite.get(key,0))
                if counts != dict(tests=40, failures=0, errors=0, skipped=0):
                    raise ValueError("complete accepted 17-test plus 23-subtest replay required")
                relative = "Libs/population_dynamics/population_intervention_replay.py"
                copy_bound(replay_root / relative, args.work / relative, replay_rows[relative])
                for relative in ("Libs/population_dynamics/population_intervention_check.py",
                                 "Libs/configs/neural_population_intervention_development.yaml"):
                    if digest(args.work / relative) != replay_rows[relative]["sha256"]:
                        raise ValueError("prepared search differs from tested development replay:" + relative)
                source = replay_root / "delivery.json"
                copy_bound(source, args.work / "prerequisites/development_replay_software.json",
                           dict(bytes=source.stat().st_size, sha256=digest(source)))
                record["development_replay_software_job"] = replay_proof["job_id"]
            record["training_seeds"] = [41, 42, 43]
            record["campaign"] = str(args.campaign)
        if args.replay_from:
            for name in ("population_intervention_replay.py", "test_population_intervention_replay.py"):
                relative = "Libs/population_dynamics/" + name
                source = args.repo / relative
                copy_bound(source, args.work / relative, dict(bytes=source.stat().st_size, sha256=digest(source)))
            result_names = [name for name in search_rows if name.startswith("results/")]
            if set(result_names) != {"results/" + name for name in (
                    "analysis.json", "candidates.csv", "reference.npz", "omit.npz",
                    "first_future_pulse.npz", "original_future_pulses.npz")}:
                raise ValueError("complete saved pulse-search artifacts required")
            for name in result_names:
                copy_bound(args.replay_from / name, args.work / "inputs/search" / name, search_rows[name])
            source = args.replay_from / "delivery.json"
            copy_bound(source, args.work / "prerequisites/search_delivery.json",
                       dict(bytes=source.stat().st_size, sha256=digest(source)))
            record["replayed_search_delivery_sha256"] = digest(source)
            record["replayed_search_job"] = search["job_id"]
        accepted, _, accepted_rows = prerequisite(args.repo, SOFTWARE, "606023")
        for module in ("native_renewal_data", "poisson_trajectory_evaluation", "poisson_trajectory_training",
                       "population_trajectory_training", "population_trajectory_evaluation",
                       "renewal_trajectory_training", "resource_probe"):
            # Stage each module and its relative imports from the accepted
            # full-episode snapshot; checkpoint-bound files stay byte-identical.
            pending = [module]
            while pending:
                current = pending.pop(0)
                name = "Libs/population_dynamics/" + current + ".py"
                if (args.work / name).is_file():
                    if digest(args.work / name) != accepted_rows[name]["sha256"]:
                        raise ValueError("checkpoint source differs from tested intervention snapshot:" + name)
                    continue
                copy_bound(accepted / name, args.work / name, accepted_rows[name])
                for node in ast.walk(ast.parse((args.work / name).read_text())):
                    if isinstance(node, ast.ImportFrom) and node.level == 1:
                        pending.extend([node.module] if node.module else [v.name for v in node.names])
        episode, receipt, episode_rows = prerequisite(args.repo, EPISODE, "606074")
        for name in ("results/model.pt", "results/run.json"):
            copy_bound(episode / name, args.work / "inputs/episode" / name, episode_rows[name])
        source = args.repo / COHORT
        cohort = json.loads((source / "delivery.json").read_text())
        if cohort["status"] != "VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE" or cohort["completed_episodes"] != 72:
            raise ValueError("accepted native development cohort required")
        inputs = manifest_rows(cohort)
        names = ["train/design.json", "train_fields/dataset.json"]
        for block in range(15 if args.campaign else 4):
            case = f"train_block_{block:03d}_familiar_0"
            names += [f"train/{case}/{name}" for name in ("report.json", "spikes.npz", "initial_parameters.npz")]
            names += [f"train_fields/{case}/{name}" for name in ("summary.json", "spatial_fields.npz")]
        for name in names:
            copy_bound(source / name, args.work / "inputs" / name, inputs[name])
        native_binding = json.loads((args.work / "inputs/episode/results/run.json").read_text())["binding"]["native"]
        graph = args.repo / "Output/neural_population_dynamics/mechanistic_reference/spatial_connectivity/source_graph_100000/manifest.json"
        copy_bound(graph, args.work / "inputs/graph_manifest.json",
                   dict(bytes=graph.stat().st_size, sha256=native_binding["native_source_identity"]["graph_manifest_sha256"]))
        for source, name in ((Path(__file__), "deployment.py"),
                              (Path(__file__).with_suffix(".sbatch"), "launcher.sbatch"),
                              (Path(__file__).with_name("run_population_trajectory_campaign.py"), "campaign_utilities.py"),
                              (Path(__file__).with_name("run_population_boundary_closure.py"), "checkpoint_staging.py")):
            copy_bound(source, args.work / name, dict(bytes=source.stat().st_size, sha256=digest(source)))
        for source, name in ((args.software / "delivery.json", "software_delivery.json"),
                             (episode / "delivery.json", "episode_delivery.json"),
                             (args.repo / COHORT / "delivery.json", "cohort_delivery.json")):
            copy_bound(source, args.work / "prerequisites" / name, dict(bytes=source.stat().st_size, sha256=digest(source)))
        spec = importlib.util.spec_from_file_location("intervention_resource_probe", args.work / "Libs/population_dynamics/resource_probe.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        (args.work / "scratch").mkdir()
        common = ["--acquisition", "inputs/train", "--fields", "inputs/train_fields",
                  "--graph-manifest", "inputs/graph_manifest.json", "--config", "Libs/configs/" + config_name]
        if args.campaign:
            command = [sys.executable, "-m", "population_dynamics.population_intervention_check", *common,
                       "--campaign", "inputs/campaign", "--output", "results"]
            if args.prepare_only:
                command.append("--validate-only")
            commands = []
            if args.prepare_only:
                commands.append(("development_software", [sys.executable, "-m", "pytest", "-q",
                    "Libs/population_dynamics/test_population_intervention_development.py", "--junitxml=development_tests.xml"]))
            commands.append(("validate_native_development_inputs" if args.prepare_only else "trained_native_pulse_search", command))
            if not args.prepare_only:
                commands.append(("trained_saved_array_replay", [sys.executable, "-m",
                    "population_dynamics.population_intervention_replay", *common, "--campaign", "inputs/campaign",
                    "--results", "results", "--output", "replay.json"]))
        else:
            common.extend(("--checkpoint", "inputs/episode/results/model.pt", "--model-receipt", "inputs/episode/results/run.json"))
        if args.replay_from:
            commands = [("replay_software", [sys.executable, "-m", "pytest", "-q",
                "Libs/population_dynamics/test_population_intervention_replay.py", "--junitxml=replay_tests.xml"]),
                ("saved_array_replay", [sys.executable, "-m", "population_dynamics.population_intervention_replay",
                 *common, "--results", "inputs/search/results", "--output", "replay.json"])]
        elif not args.campaign:
            commands = [("native_pulse_search", [sys.executable, "-m", "population_dynamics.population_intervention_check",
                         *common, "--output", "results"])]
        for name, command in commands:
            code = phase(args.work, name, command, module.probe)
            if code:
                break
        if args.campaign and not args.prepare_only and code == 0:
            searched = json.loads((args.work / "results/analysis.json").read_text())
            replayed = json.loads((args.work / "replay.json").read_text())
            if (searched["status"] != "THREE_SEED_DEVELOPMENT_NATIVE_PULSE_SEARCH_EXECUTED"
                    or searched["completed_seeds"] != [41, 42, 43] or searched["candidate_results"] != 9
                    or replayed["status"] != "INDEPENDENT_THREE_SEED_SAVED_ARRAY_PULSE_SEARCH_REPLAYED"
                    or replayed["completed_seeds"] != [41, 42, 43]):
                raise ValueError("complete three-seed search and saved-array replay required")
            record["saved_array_replay_completed"] = True
        record["status"] = ("PREPARED" if args.prepare_only else "PASS") if code == 0 else "RESOURCE_BLOCKED" if code == 3 else "FAILED"
        record.update(native_input_files=len(names), native_input_bytes=sum(inputs[name]["bytes"] for name in names),
                      independent_native_intervention_executed=False)
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay-from", type=Path,
                      help="Replay every saved array from a completed engineering search; do not repeat search")
    mode.add_argument("--campaign", type=Path, help="Use all three completed development checkpoints")
    parser.add_argument("--prepare-only", action="store_true", help="Validate new development software and native inputs without inference")
    parser.add_argument("--development-preparation", type=Path,
                        help="Reuse accepted code/config from the completed development preparation")
    args = parser.parse_args()
    if args.development_preparation and (not args.campaign or args.prepare_only):
        parser.error("--development-preparation requires --campaign execution")
    if args.campaign and not args.prepare_only and not args.development_preparation:
        parser.error("campaign execution requires --development-preparation")
    if args.prepare_only and not args.campaign:
        parser.error("--prepare-only requires --campaign")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
