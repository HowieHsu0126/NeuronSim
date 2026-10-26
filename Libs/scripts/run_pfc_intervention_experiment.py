"""Stage the bound PFC development search and its independent replay on CPU03."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import getpass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import yaml

SEEDS = (41, 42, 43)
SCRATCH_ROOT = Path("/tmp") / getpass.getuser()
FROZEN_HELPER = "Libs/scripts/run_population_trajectory_campaign.py"
RESOURCE_PROBE = "Libs/population_dynamics/resource_probe.py"
WRAPPER = "Libs/scripts/run_pfc_intervention_experiment.py"


def require(condition, message):
    if not condition:
        raise ValueError("PFC_INTERVENTION_LAUNCHER_" + message)


def read(path):
    return json.loads(Path(path).read_text())


def relative_path(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and bool(path.parts), "SAFE_RELATIVE_PATH_REQUIRED")
    return path


def load_frozen_helper(preparation):
    """Authenticate the existing helper before executing its module body."""
    proof = read(preparation / "preparation.json")
    rows = proof["files"]
    index = {row["path"]: row for row in rows}
    require(len(index) == len(rows) and FROZEN_HELPER in index, "UNIQUE_FROZEN_HELPER_REQUIRED")
    path, row = preparation / FROZEN_HELPER, index[FROZEN_HELPER]
    require(path.stat().st_size == row["bytes"]
            and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "FROZEN_HELPER_CHANGED")
    spec = importlib.util.spec_from_file_location("neuronsim_frozen_campaign_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, proof


def validated_software(helper, software):
    proof = read(software / "source_snapshot.json")
    index = helper.manifest_rows(proof)
    for name, row in index.items():
        source = software / relative_path(name)
        require(source.stat().st_size == row["bytes"] and helper.digest(source) == row["sha256"],
                "TESTED_SOFTWARE_CHANGED:" + name)
    return index


def validated_caller(software):
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(software / "Libs"))
        caller = importlib.import_module("population_dynamics.pfc_intervention_experiment")
    finally:
        sys.path[:] = previous
    require(Path(caller.__file__).resolve() ==
            (software / "Libs/population_dynamics/pfc_intervention_experiment.py").resolve(),
            "CALLER_MUST_LOAD_FROM_TESTED_SOFTWARE")
    return caller


def protected_sources(helper, proof, software_index, admitted):
    frozen = helper.manifest_rows(proof)
    protected = {}
    for seed in SEEDS:
        for name, digest in admitted[seed]["run"]["trainer_sources"].items():
            require(Path(name).name == name, "PLAIN_TRAINER_SOURCE_NAME_REQUIRED")
            relative = ("Libs/configs/" if name.endswith(".yaml") else "Libs/population_dynamics/") + name
            require(relative not in protected or protected[relative] == digest, "TRAINER_SEEDS_DISAGREE:" + relative)
            protected[relative] = digest
    # stage_pfc loads this probe before the overlay. Preserve its bytes and the
    # helper so preflight and watchdog behavior remain bound to the preparation.
    for name in (FROZEN_HELPER, RESOURCE_PROBE):
        require(name in frozen, "FROZEN_EXECUTION_SOURCE_REQUIRED:" + name)
        protected[name] = frozen[name]["sha256"]
    for name, digest in protected.items():
        require(name in frozen and frozen[name]["sha256"] == digest
                and name in software_index and software_index[name]["sha256"] == digest,
                "PROTECTED_SOURCE_DIFFERENCE:" + name)
    return protected


def bound_row(helper, path, name):
    return dict(path=name, bytes=path.stat().st_size, sha256=helper.digest(path))


def compose_sources(helper, proof, software, software_index, protected, config, work):
    index = helper.manifest_rows(proof)
    closure = helper.software_closure(software, ["pfc_intervention_experiment", "pfc_intervention_outcomes"])
    for name in closure:
        require(name in software_index, "UNTESTED_RUNTIME_DEPENDENCY:" + name)
        row = software_index[name]
        if name not in protected:
            helper.copy_bound(software / relative_path(name), work / name, row)
        index[name] = row
    config_name = "Libs/configs/" + config.name
    require(config_name not in protected, "EXPERIMENT_CONFIG_COLLIDES_WITH_PROTECTED_CONFIG")
    row = bound_row(helper, config, config_name)
    if config_name in software_index:
        require(row["sha256"] == software_index[config_name]["sha256"], "TESTED_EXPERIMENT_CONFIG_CHANGED")
    helper.copy_bound(config, work / config_name, row)
    index[config_name] = row
    for source, name in ((Path(__file__), WRAPPER), (Path(__file__).with_suffix(".sbatch"), WRAPPER.removesuffix(".py") + ".sbatch")):
        row = bound_row(helper, source, name)
        if name in software_index:
            require(row["sha256"] == software_index[name]["sha256"], "TESTED_WRAPPER_CHANGED")
        helper.copy_bound(source, work / name, row)
        index[name] = row
    for name, row in index.items():
        require((work / name).stat().st_size == row["bytes"] and helper.digest(work / name) == row["sha256"],
                "COMPOSED_RUNTIME_CHANGED:" + name)
        if name in software_index:
            require(row["sha256"] == software_index[name]["sha256"], "COMPOSED_RUNTIME_DIFFERS_FROM_TESTED_BUNDLE:" + name)
    helper.write_json(work / "source_snapshot.json", dict(
        schema="pfc_intervention_composed_runtime_v1",
        frozen_preparation_sha256=helper.digest(work / "preparation.json"),
        tested_software_snapshot_sha256=helper.digest(software / "source_snapshot.json"),
        protected_sources=protected, files=[index[name] for name in sorted(index)]))
    return work / config_name


def completed_summary(path, replay=False):
    report = read(path / "analysis.json")
    require(report.get("status") == ("PASS_PFC_INTERVENTION_EXPERIMENT_REPLAY" if replay
                                    else "COMPLETE_PFC_INTERVENTION_EXPERIMENT")
            and report.get("candidate_rows") == 90 and report.get("seed_block_cases") == 9
            and report.get("generated_unit_count_values_across_completed_cases") == 141120000000
            and all(report.get(name) is False for name in (
                "per_bin_full_unit_counts_saved", "per_bin_full_unit_counts_exact_replay_claimed",
                "training_performed", "native_execution_performed", "success_tolerance_qualified",
                "memory_recovery_qualified", "scientific_claim_qualified")),
            "COMPLETE_UNQUALIFIED_NINE_CASE_RESULT_REQUIRED")
    if replay:
        require(report.get("saved_candidate_array_values_compared") ==
                90 * (2 * 16 * 980 * 17 + 2 * 16 * 100000 + 100000 + 17),
                "COMPLETE_RETAINED_ARRAY_REPLAY_REQUIRED")
    return report


def completed_outcomes_summary(path):
    report = read(path / "analysis.json")
    expected = dict(status="COMPLETE_PFC_INTERVENTION_SOURCE_OUTCOMES",
        candidate_rows=90, native_source_episodes=30, native_source_blocks=3,
        native_circuits=1, policy_rows=3, program_rows=10,
        observed_unit_bin_values_aggregated=2940000000, newly_generated_samples=0,
        all_candidate_outcomes_retained=True, observed_prefixes_equal=True,
        native_future_noise_pairing_verified=False, independent_intervention_qualified=False,
        success_tolerance_qualified=False, memory_recovery_qualified=False,
        scientific_claim_qualified=False, native_execution_performed=False)
    require(all(type(report.get(name)) is type(value) and report[name] == value
                for name, value in expected.items()), "COMPLETE_BOUND_NATIVE_SOURCE_READBACK_REQUIRED")
    return report


def launch(preparation, software, reference, config, seed_cells, work, output, *, reuse_completed_from=None):
    preparation, software, reference, config, work, output = map(
        Path, (preparation, software, reference, config, work, output))
    require(set(seed_cells) == set(SEEDS), "EXACT_THREE_SEED_CELLS_REQUIRED")
    require(os.environ.get("SLURMD_NODENAME") == "cpu03" and os.environ.get("SLURM_JOB_PARTITION") == "gpu"
            and os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_CPUS_PER_TASK") == "4"
            and not any(os.environ.get(name) for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")),
            "CPU03_FOUR_CPU_NO_GPU_ALLOCATION_REQUIRED")
    require(work.resolve().is_relative_to(SCRATCH_ROOT) and work.resolve() != SCRATCH_ROOT
            and (not work.exists() or not any(work.iterdir())), "FRESH_NODE_LOCAL_WORK_REQUIRED")
    require(not output.exists(), "NEW_DELIVERY_DIRECTORY_REQUIRED")
    require(not output.resolve().is_relative_to(work.resolve()), "DELIVERY_MUST_BE_OUTSIDE_SCRATCH")
    helper, proof = load_frozen_helper(preparation)
    work.mkdir(parents=True, exist_ok=True)
    (work / "scratch").mkdir()
    record = dict(status="PREPARING", exit_code=2, job_id=os.environ["SLURM_JOB_ID"],
        node="cpu03", partition="gpu", device="cpu", training_seeds=list(SEEDS),
        preparation=str(preparation), preparation_sha256=helper.digest(preparation / "preparation.json"),
        software=str(software), reference=str(reference), config=str(config),
        started_utc=datetime.now(timezone.utc).isoformat(), training_performed=False,
        native_execution_performed=False, scientific_claim_qualified=False,
        per_bin_full_unit_counts_exact_replay_claimed=False,
        generation_and_replay_complete=False, source_outcome_readback_performed=False)
    try:
        index = validated_software(helper, software)
        caller = validated_caller(software)
        configuration = caller.validate_config(yaml.safe_load(config.read_text()))
        require(helper.digest(reference) == configuration["reference_sha256"], "REFERENCE_CONFIG_BINDING_CHANGED")
        admitted = caller.admit_seed_cells(seed_cells)
        require(all(admitted[seed]["cell"].proof["preparation_sha256"] == record["preparation_sha256"]
                    for seed in SEEDS), "DELIVERED_PREPARATION_CHANGED")
        protected = protected_sources(helper, proof, index, admitted)
        admission = dict(seed_cells={str(seed): dict(path=str(Path(seed_cells[seed]).resolve()),
                                                   binding=deepcopy(admitted[seed]["binding"])) for seed in SEEDS},
            protected_sources=protected, preparation_sha256=record["preparation_sha256"],
            software_snapshot_sha256=helper.digest(software / "source_snapshot.json"),
            reference_sha256=configuration["reference_sha256"], config_sha256=helper.digest(config))
        helper.write_json(work / "admission.json", admission)
        probe, staged_proof = helper.stage_pfc(preparation, work)
        require(staged_proof == proof, "PREPARATION_CHANGED_DURING_STAGING")
        helper.copy_bound(preparation / "preparation.json", work / "preparation.json",
                          bound_row(helper, preparation / "preparation.json", "preparation.json"))
        require(helper.digest(work / "preparation.json") == record["preparation_sha256"], "STAGED_PREPARATION_CHANGED")
        config_local = compose_sources(helper, proof, software, index, protected, config, work)
        helper.copy_bound(software / "source_snapshot.json", work / "software_snapshot.json",
                          bound_row(helper, software / "source_snapshot.json", "software_snapshot.json"))
        helper.copy_bound(reference, work / "reference.json", bound_row(helper, reference, "reference.json"))
        seed_arguments = []
        for seed in SEEDS:
            cell = admitted[seed]["cell"]
            local = work / "inputs/seed_cells" / str(seed)
            for name in caller.SEED_CELL_FILES:
                helper.copy_bound(cell.path(name), local / name, cell.files[name])
            delivery_row = bound_row(helper, cell.root / "delivery.json", "delivery.json")
            require(delivery_row["sha256"] == admitted[seed]["binding"]["delivery_sha256"],
                    "SEED_DELIVERY_CHANGED_AFTER_ADMISSION")
            helper.copy_bound(cell.root / "delivery.json", local / "delivery.json", delivery_row)
            seed_arguments.extend(["--seed-cell", str(seed), str(local)])
        helper.write_json(work / "staged_seed_cells.json", dict(
            files_per_seed=["delivery.json", *caller.SEED_CELL_FILES],
            bindings={str(seed): deepcopy(admitted[seed]["binding"]) for seed in SEEDS},
            completed_forecast_payloads_copied=False))
        common = [sys.executable, "-m", "population_dynamics.pfc_intervention_experiment",
                  "--inputs", str(work / "inputs/cohort.json"), "--preparation", str(work),
                  "--config", str(config_local), "--reference", str(work / "reference.json"), *seed_arguments]
        search = [*common, "--output", str(work / "results")]
        if reuse_completed_from is not None:
            search.extend(["--reuse-completed-from", str(Path(reuse_completed_from))])
        code = helper.phase(work, "intervention", search, probe)
        record.update(exit_code=code, status="RESOURCE_BLOCKED" if code == 3 else "FAILED", failed_phase="intervention")
        if code == 0:
            completed_summary(work / "results")
            replay = [*common, "--output", str(work / "replay"), "--replay-from", str(work / "results")]
            code = helper.phase(work, "intervention_replay", replay, probe)
            record.update(exit_code=code, status="RESOURCE_BLOCKED" if code == 3 else "FAILED", failed_phase="intervention_replay")
            if code == 0:
                summary = completed_summary(work / "replay", replay=True)
                record.update(generation_and_replay_complete=True, candidate_rows=90, seed_block_cases=9,
                    generated_unit_count_values=141120000000, independently_regenerated_unit_count_values=141120000000,
                    retained_candidate_array_values_compared=summary["saved_candidate_array_values_compared"])
                outcomes = [sys.executable, "-m", "population_dynamics.pfc_intervention_outcomes",
                    "--inputs", str(work / "inputs/cohort.json"), "--preparation", str(work),
                    "--config", str(config_local), "--reference", str(work / "reference.json"),
                    "--search", str(work / "results"), "--replay", str(work / "replay"),
                    "--output", str(work / "source_outcomes")]
                code = helper.phase(work, "source_outcomes", outcomes, probe)
                record.update(exit_code=code, status="RESOURCE_BLOCKED" if code == 3 else "FAILED",
                              failed_phase="source_outcomes")
                if code == 0:
                    native = completed_outcomes_summary(work / "source_outcomes")
                    record.update(status="PASS", failed_phase=None, source_outcome_readback_performed=True,
                        native_source_episodes_reused=native["native_source_episodes"],
                        observed_unit_bin_values_aggregated=native["observed_unit_bin_values_aggregated"],
                        native_future_noise_pairing_verified=False, independent_intervention_qualified=False)
    except helper.ResourceBlocked as error:
        record.update(status="RESOURCE_BLOCKED", exit_code=3, reasons=error.reasons)
    except BaseException as error:
        record.update(status="FAILED", exit_code=(int(error.code) if isinstance(error, SystemExit)
                                                and isinstance(error.code, int) and error.code != 0 else 2),
                      error_type=type(error).__name__, error=str(error))
        helper.write_json(work / "failure.json", dict(error_type=type(error).__name__, error=str(error),
                                                     traceback=traceback.format_exc()))
    helper.write_json(work / "execution.json", record)
    helper.archive(work, output, record)
    (work / "archive_complete").write_text("verified archive complete\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation", "software", "reference", "config", "work", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed-cell", nargs=2, action="append", required=True, metavar=("SEED", "PATH"))
    parser.add_argument("--reuse-completed-from", type=Path)
    args = parser.parse_args(argv)
    pairs = [(int(seed), Path(path)) for seed, path in args.seed_cell]
    require(len(pairs) == 3 and len(dict(pairs)) == 3, "EXACT_THREE_SEED_CELLS_REQUIRED")
    record = launch(args.preparation, args.software, args.reference, args.config, dict(pairs),
                    args.work, args.output, reuse_completed_from=args.reuse_completed_from)
    return record["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
