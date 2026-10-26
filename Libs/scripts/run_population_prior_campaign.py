"""Execute the native population prior refit with the existing phase runner."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

CONFIG = "Libs/configs/neural_population_prior_refit.yaml"
MIXTURE_CONFIG = "Libs/configs/neural_population_prior_predictive_mixture.yaml"
OBSERVATION_CONFIG = "Libs/configs/neural_population_native_observation_prior.yaml"
CONFIGURATIONS = (CONFIG, MIXTURE_CONFIG, OBSERVATION_CONFIG)
PRIOR_PROFILES = {
    OBSERVATION_CONFIG: ("neural_population_native_observation_prior_v1",
                         "fixed_readout_native_event_path_mixture_nll", "validation_native_event_path_mixture_nll"),
    CONFIG: ("neural_population_prior_refit_v1", "fixed_posterior_multistep_prior_kl",
             "validation_multistep_prior_kl"),
    MIXTURE_CONFIG: ("neural_population_prior_predictive_mixture_v1",
                     "fixed_posterior_joint_predictive_mixture_kl", "validation_joint_predictive_mixture_kl"),
}
TESTS = ["test_population_prior_objective.py", "test_population_prior_refit.py",
         "test_renewal_trajectory_training.py", "test_population_trajectory_training.py",
         "test_population_prior_campaign.py"]
ALLOWED_NODES = {"cpu03", "cpu04", "cpu05", "cpu06", "gpu03"}


def script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_names(configuration):
    if configuration == OBSERVATION_CONFIG:
        return TESTS + ["test_population_observation_prior.py", "test_population_observation_prior_refit.py"]
    return TESTS + (["test_population_predictive_mixture.py", "test_population_predictive_mixture_refit.py"]
                    if configuration == MIXTURE_CONFIG else [])


def prepare(repo, destination, *, configuration=CONFIG):
    if configuration not in CONFIGURATIONS:
        raise ValueError("registered prior configuration required")
    helper = script(repo / "Libs/scripts/run_population_trajectory_campaign.py", "prior_prepare_helper")
    modules = ["population_trajectory_training", "population_trajectory_evaluation", "population_forecast_replay", "resource_probe",
               *[Path(name).stem for name in test_names(configuration)]]
    names = helper.software_closure(repo, modules)
    names += [configuration, "Libs/configs/neural_population_state_trajectory.yaml",
        "Libs/configs/neural_population_matched_forecast.yaml",
        "Libs/scripts/run_population_trajectory_campaign.py", "Libs/scripts/run_population_prior_campaign.py"]
    if destination.exists():
        raise FileExistsError("reuse the existing immutable prior preparation")
    destination.mkdir(parents=True)
    rows = []
    for name in sorted(set(names)):
        source = repo / name
        row = dict(path=name, bytes=source.stat().st_size, sha256=helper.digest(source))
        helper.copy_bound(source, destination / name, row)
        rows.append(row)
    result = dict(status="PREPARED_FOR_VALIDATION", files=rows, configuration_path=configuration, training_seeds=[41, 42, 43],
                  created_utc=datetime.now(timezone.utc).isoformat(), scientific_claim_qualified=False)
    helper.write_json(destination / "preparation.json", result)
    return result


def stage_budget_parent(helper, parent, work, seed, original_parent_digest, total_epochs, *, configuration=CONFIG):
    """Stage one accepted same-objective parent; keep its original delivery identity."""
    if configuration not in PRIOR_PROFILES:
        raise ValueError("registered prior configuration required")
    schema, objective, selection = PRIOR_PROFILES[configuration]
    parent = Path(parent).resolve()
    proof = json.loads((parent / "delivery.json").read_text())
    rows = helper.manifest_rows(proof)
    name = "training/run.json"
    helper.copy_bound(parent / name, work / "budget_training/run.json", rows[name])
    report = json.loads((work / "budget_training/run.json").read_text())
    config = report["binding"]["config"]
    if (proof.get("status") != "PASS" or proof.get("exit_code") != 0
            or proof.get("training_seed") != seed
            or proof.get("parent_delivery_sha256") != original_parent_digest
            or report.get("training_seed") != seed
            or report.get("status") != "FITTED_DEVELOPMENT"
            or config.get("schema") != schema
            or config.get("observation_objective") != objective
            or report.get("training_objective") != objective
            or report.get("selection_metric") != selection
            or report.get("continuation") is not None
            or report.get("calibration_outcomes_accessed") is not False
            or report.get("test_outcomes_accessed") is not False
            or not report.get("frozen_parameters_unchanged_in_attempt")
            or type(total_epochs) is not int
            or total_epochs <= max(report["completed_epochs"], report["effective_maximum_epochs"])):
        raise ValueError("accepted same-seed prior parent and increased prospective budget required")
    required = {"training/" + name for name in
                ("run.json", "latest.pt", "best.pt", "epochs.json", "optimization.csv", "validation.csv")}
    caches = {f"training/posterior_cache/train_block_{block:03d}_familiar_{template}.pt"
              for block in config["fit_blocks"] + config["validation_blocks"] for template in range(4)}
    if (len(caches) != 72 or not required <= set(rows)
            or {name for name in rows if name.startswith("training/posterior_cache/")} != caches):
        raise ValueError("complete prior checkpoint/history and72 posterior caches required")
    for name, row in rows.items():
        if name.startswith("training/") and name != "training/run.json":
            helper.copy_bound(parent / name, work / "budget_training" / name.removeprefix("training/"), row)
    for name in ("delivery.json", "training/run.json"):
        source = parent / name
        row = rows[name] if name in rows else dict(bytes=source.stat().st_size, sha256=helper.digest(source))
        helper.copy_bound(source, work / "prerequisites/budget_parent" / name, row)
    return dict(parent=str(parent), parent_delivery_sha256=helper.digest(parent / "delivery.json"),
        parent_checkpoint_sha256=rows["training/latest.pt"]["sha256"],
        parent_completed_epochs=report["completed_epochs"], base_maximum_epochs=config["maximum_epochs"],
        total_epochs=total_epochs, reused_posterior_cache_files=len(caches))


def run(repo, preparation, output, seed, parent_cell, resume_from=None, stop_after_epochs=None, validate=False,
        *, budget_parent=None, total_epochs=None, prior_cache_source=None):
    if ((budget_parent is None) != (total_epochs is None)
            or (budget_parent is not None and resume_from is not None)):
        raise ValueError("choose explicit budget parent/total epochs or resource recovery")
    new_budget = budget_parent is not None
    if (seed not in (41, 42, 43) or os.environ.get("SLURMD_NODENAME") not in ALLOWED_NODES
            or not os.environ.get("SLURM_JOB_GPUS")):
        raise ValueError("registered seed and authorized Slurm GPU allocation required")
    helper = script(preparation / "Libs/scripts/run_population_trajectory_campaign.py", "prior_run_helper")
    receipt = json.loads((preparation / "preparation.json").read_text())
    configuration = receipt.get("configuration_path", CONFIG)
    if configuration not in CONFIGURATIONS:
        raise ValueError("registered frozen prior configuration required")
    uses_fixed_cache = configuration in (MIXTURE_CONFIG, OBSERVATION_CONFIG)
    if prior_cache_source is not None and (resume_from is not None or new_budget):
        raise ValueError("posterior cache source is only for a new refit")
    if uses_fixed_cache and resume_from is None and not new_budget and prior_cache_source is None:
        raise ValueError("predictive-mixture or observation refit requires reuse of the accepted posterior caches")
    preparation_digest = helper.digest(preparation / "preparation.json")
    parent_receipt = json.loads((parent_cell / "delivery.json").read_text())
    if parent_receipt["status"] != "PASS" or parent_receipt["training_seed"] != seed:
        raise ValueError("same-seed accepted population parent required")
    parent_rows = helper.manifest_rows(parent_receipt)
    if output.exists():
        raise FileExistsError("prior resource attempts must retain separate outputs")
    scratch = Path("/tmp") / os.environ["USER"]
    scratch.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"neuronsim-prior-{os.environ['SLURM_JOB_ID']}-", dir=scratch))
    record = dict(status="PREPARING", exit_code=2, job_id=os.environ["SLURM_JOB_ID"],
        training_seed=seed, preparation_sha256=preparation_digest, preparation=str(preparation),
        configuration_path=configuration,
        parent_cell=str(parent_cell.resolve()), parent_delivery_sha256=helper.digest(parent_cell / "delivery.json"),
        work=str(work), node=os.environ["SLURMD_NODENAME"], started_utc=datetime.now(timezone.utc).isoformat(),
        source_units=100000, fit_episodes=56, validation_episodes=16, scientific_claim_qualified=False)
    if prior_cache_source is not None:
        record["posterior_cache_source"] = str(Path(prior_cache_source).resolve())
    print(json.dumps(record), flush=True)
    try:
        for row in receipt["files"]:
            helper.copy_bound(preparation / row["path"], work / row["path"], row)
        (work / "prerequisites").mkdir()
        (work / "scratch").mkdir()
        probe = script(work / "Libs/population_dynamics/resource_probe.py", "prior_probe").probe
        health = probe(work / "input_preflight.json", required_memory_gib=32)
        if health["status"] != "PASS":
            raise helper.ResourceBlocked(health["reasons"])
        source = json.loads((repo / helper.COHORT / "delivery.json").read_text())
        required = sum(row["bytes"] for row in helper.manifest_rows(source).values()) + 24 * 1024**3
        free = shutil.disk_usage(work).free
        helper.write_json(work / "storage_reserve.json", dict(required_free_bytes=required, free_bytes=free))
        if free < required:
            raise helper.ResourceBlocked(["prior input/cache/output storage reserve unavailable"])
        entry = work / "entry.py"
        entry.write_text("import sys,runpy\nfrom pathlib import Path\np=Path(__file__).parent\nsys.path[:0]=[str(p),str(p/'Libs')]\nmodule=sys.argv.pop(1)\nrunpy.run_module(module,run_name='__main__')\n")
        os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", NVIDIA_TF32_OVERRIDE="0",
                          OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                          PYTHONDONTWRITEBYTECODE="1")
        for key, name in (("XDG_CACHE_HOME", "cache"), ("CUDA_CACHE_PATH", "cuda"),
                          ("TORCH_EXTENSIONS_DIR", "extensions")):
            path = work / "scratch" / name
            path.mkdir()
            os.environ[key] = str(path)
        if validate:
            tests = [sys.executable, str(entry), "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(work / "scratch/pytest"),
                     *[str(work / "Libs/population_dynamics" / name) for name in test_names(configuration)]]
            code = helper.phase(work, "focused_tests", tests, probe)
            if code:
                raise RuntimeError("focused prior/native training tests failed")
            record["focused_tests_passed"] = True
        helper.stage_inputs(repo, work)
        for name in ("training/run.json", "training/best.pt"):
            helper.copy_bound(parent_cell / name, work / "parent_training" / Path(name).name, parent_rows[name])
        helper.write_json(work / "prerequisites/prior_parent.json", dict(parent_cell=str(parent_cell.resolve()),
            delivery_sha256=helper.digest(parent_cell / "delivery.json"),
            files=[parent_rows[name] for name in ("training/run.json", "training/best.pt")]))
        complete = dict(train=False, evaluate=False, replay=False)
        inherited_budget = None
        if resume_from is not None:
            inherited_budget = json.loads((resume_from / "delivery.json").read_text()).get("optimization_budget_continuation")
            if inherited_budget is not None:
                budget_parent = Path(inherited_budget["parent"])
                total_epochs = inherited_budget["total_epochs"]
        if budget_parent is not None:
            lineage = stage_budget_parent(helper, budget_parent, work, seed,
                                          record["parent_delivery_sha256"], total_epochs,
                                          configuration=configuration)
            if inherited_budget is not None and inherited_budget != lineage:
                raise ValueError("resource recovery prior budget ancestry changed")
            record["optimization_budget_continuation"] = lineage
        if resume_from is not None:
            previous = json.loads((resume_from / "delivery.json").read_text())
            if (previous["training_seed"] != seed or previous["preparation_sha256"] != preparation_digest
                    or previous["status"] == "PASS"
                    or previous["parent_delivery_sha256"] != record["parent_delivery_sha256"]):
                raise ValueError("same-seed incomplete prior refit resource parent required")
            rows = helper.manifest_rows(previous)
            if "training/latest.pt" not in rows:
                raise ValueError("committed prior checkpoint missing")
            for phase_name in complete:
                name = "phases/" + phase_name + "/receipt.json"
                if name in rows:
                    helper.copy_bound(resume_from / name, work / "prerequisites" / ("parent_" + phase_name + ".json"), rows[name])
                    phase_receipt = json.loads((resume_from / name).read_text())
                    complete[phase_name] = phase_receipt.get("status") == "PASS" and phase_receipt.get("exit_code") == 0
            helper.copy_bound(resume_from / "training/run.json", work / "prerequisites/parent_training.json", rows["training/run.json"])
            previous_training = json.loads((work / "prerequisites/parent_training.json").read_text())
            complete["train"] = previous_training["status"] == "FITTED_DEVELOPMENT"
            if (complete["evaluate"] or complete["replay"]) and not complete["train"]:
                raise ValueError("accepted forecasts require complete prior refit training")
            if complete["replay"] and not complete["evaluate"]:
                raise ValueError("accepted replay requires accepted forecasts")
            for name, row in rows.items():
                if (name.startswith("training/") or complete["evaluate"] and name.startswith("forecasts/")
                        or complete["replay"] and name == "replay.json"):
                    helper.copy_bound(resume_from / name, work / name, row)
            helper.write_json(work / "prerequisites/resource_parent.json", dict(parent=str(resume_from.resolve()),
                delivery_sha256=helper.digest(resume_from / "delivery.json"), parent_job=previous["job_id"],
                completed_phases=complete))
        common = ["--acquisition", str(work / "inputs/train"), "--fields", str(work / "inputs/train_fields")]
        train = [sys.executable, str(entry), "Libs.population_dynamics.population_trajectory_training", *common,
                 "--config", str(work / configuration), "--seed", str(seed), "--output", str(work / "training")]
        if resume_from is not None:
            train += ["--resume"]
        elif new_budget:
            train += ["--resume-from", str(work / "budget_training"), "--total-epochs", str(total_epochs),
                      "--parent-checkpoint-sha256", record["optimization_budget_continuation"]["parent_checkpoint_sha256"]]
        else:
            train += ["--prior-parent", str(work / "parent_training")]
        if prior_cache_source is not None:
            train += ["--prior-cache-source", str(prior_cache_source)]
        if stop_after_epochs is not None:
            train += ["--stop-after-epochs", str(stop_after_epochs)]
        commands = [("train", train)]
        if stop_after_epochs is None:
            evaluation = [sys.executable, str(entry), "Libs.population_dynamics.population_trajectory_evaluation", *common,
                "--training", str(work / "training"), "--config", str(work / "Libs/configs/neural_population_matched_forecast.yaml"),
                "--output", str(work / "forecasts"), "--model-type", "population", "--device", "cuda"]
            replay = [sys.executable, str(entry), "Libs.population_dynamics.population_forecast_replay", *common,
                "--training", str(work / "training"), "--forecasts", str(work / "forecasts"),
                "--output", str(work / "replay.json"), "--device", "cuda", "--all-origins"]
            commands += [("evaluate", evaluation), ("replay", replay)]
        for name, command in commands:
            if complete[name]:
                continue
            code = helper.phase(work, name, command, probe)
            if code:
                record.update(status="RESOURCE_BLOCKED" if code == 3 else "FAILED", exit_code=code, failed_phase=name)
                break
        else:
            training = json.loads((work / "training/run.json").read_text())
            if not training.get("frozen_parameters_unchanged_in_attempt"):
                raise ValueError("fixed encoder/readout verification missing")
            if stop_after_epochs is None:
                evaluated = json.loads((work / "forecasts/evaluation.json").read_text())
                replayed = json.loads((work / "replay.json").read_text())
                if (training["status"] != "FITTED_DEVELOPMENT" or training["fitting_episodes"] != 56
                        or training["validation_episodes"] != 16 or evaluated["status"] != "EVALUATED_DEVELOPMENT"
                        or evaluated["score_rows"] != 144 or replayed["status"] != "PASS"
                        or replayed["verified_origins"] != 48):
                    raise ValueError("complete prior training/48-origin/144-score replay required")
                record.update(complete_origins=48, score_rows=144, verified_origins=48)
            record.update(status="COMMITTED_TRAINING_SLICE" if stop_after_epochs is not None else "PASS", exit_code=0,
                committed_epochs=training["completed_epochs"], best_epoch=training["best_epoch"],
                frozen_parameters_unchanged=True, training_resumed=training["resumed"],
                effective_maximum_epochs=training["effective_maximum_epochs"])
    except helper.ResourceBlocked as error:
        record.update(status="RESOURCE_BLOCKED", exit_code=3, reasons=error.reasons)
    except BaseException as error:
        record.update(status="FAILED", exit_code=2, error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
    if (work / "budget_training").exists():
        shutil.rmtree(work / "budget_training")
    helper.archive(work, output, record)
    shutil.rmtree(work)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--configuration", choices=CONFIGURATIONS, help="configuration for preparation only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, choices=[41, 42, 43])
    parser.add_argument("--parent-cell", type=Path)
    parser.add_argument("--prior-cache-source", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--budget-parent", type=Path)
    parser.add_argument("--total-epochs", type=int)
    parser.add_argument("--stop-after-epochs", type=int)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        result = prepare(args.repo, args.preparation, configuration=args.configuration or CONFIG)
    else:
        if args.configuration is not None:
            parser.error("--configuration applies only to --prepare-only")
        if args.output is None or args.seed is None or args.parent_cell is None:
            parser.error("execution requires output, seed and accepted parent cell")
        result = run(args.repo, args.preparation, args.output, args.seed, args.parent_cell,
                     args.resume_from, args.stop_after_epochs, args.validate,
                     budget_parent=args.budget_parent, total_epochs=args.total_epochs,
                      prior_cache_source=args.prior_cache_source)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}), flush=True)
    raise SystemExit(result.get("exit_code", 0))


if __name__ == "__main__":
    main()
