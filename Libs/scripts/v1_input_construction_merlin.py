#!/usr/bin/env python3
"""Run one full V1 input construction and independent replay on a Merlin CPU allocation.

The nine-module canary snapshot remains the numerical implementation. The two
listed runtime replacements only add Merlin CPU placement/provenance. The staged
resource probe retains its original resource checks and replaces Slurm identity
handling. This runner writes an isolated delivery; canonical intake remains a
separate consumer operation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import os
import traceback

import yaml

from population_dynamics.provenance import cpu_execution_provenance, freeze_json, sha256_file, versions


QUALIFIED_MODULE_SHA256 = {
    "v1_circuit_build": "7279be38ff9179ec52e2eaf95a2cf48556885faebdf9b8e748e97ef5a79d6563",
    "v1_circuit_source": "bf400fcdd482693e9bdcadf40e71bfe29515031a40efef7edb9a732eb8a4c808",
    "v1_graph_identity": "c9afa5a301d22bf16280dc2ce74dc7512079831fa49c97bb85f841ec89c12ad9",
    "v1_source_conversion": "57f46ac773e1cdcd13289032a047afda546f0a638f9b6c855d587d5f57c879b3",
    "provenance": "57e4ed6178a2f655cc81e4b943e5934d5d58b226133ede2efdcf451c49d5d43f",
    "resource_probe": "39e734be94d3d22ac15f895d710b933259794eaeffeef36f39f66cac1594405b",
    "v1_circuit_inputs": "682b07f308dab5f1050b6199dc362a895ae04ec8e099bceb3644a5a996f85242",
    "v1_recurrent_runtime": "56d08b656c54b9dbc9d3726f361f2123303933ff3989ecd0d36ef31843b3a060",
    "v1_input_runtime": "55eaa3c261a867f0730e29919937e34f38e78c006a894b254ee85fc777a29ed7",
}
PROVENANCE_RUNTIME_SHA256 = {
    "provenance": "2dcb5122d1a13cf3dde9e5a02a9f9fa55cb4015541b510b5b973f38b8adfa2b6",
    "v1_circuit_inputs": "e64ee9bb5958ca57809a04f96ca2f590aa9c96611b8a1038e4d7d845fa553ba1",
}
SOURCE_NAMES = (
    "original_build_network", "original_node_functions", "original_edge_functions",
    "original_node_parameters", "original_edge_parameters", "original_connection_parameters",
    "original_lgn_connection_parameters", "original_glif_node_parameters", "original_bio_to_glif_map",
    "weight_semantics/legacy_glif_converter", "weight_semantics/legacy_recurrent_table",
    "original_lgn_node_parameters", "original_lgn_edge_parameters", "original_bkg_edge_parameters",
    "original_glif_lgn_parameters", "original_glif_bkg_parameters",
)
RECURRENT_FILES = ("delivery.json", "primary/report.json", *(
    "primary/" + kind + "/" + name for kind in ("biophysical", "glif")
    for name in ("v1_nodes.h5", "v1_node_types.csv")))
QUALIFICATION_REPORTS = ("validation.json", "canary/report.json", "comparison/report.json")
NUMERICAL_PACKAGES = ("python", "numpy", "pandas", "scipy", "h5py", "PyYAML")
FULL_REQUIRED_MEMORY_GIB = 64


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def repo_path(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def bound_request(repo: Path, config_path: Path, recurrent_record: Path) -> tuple[dict, dict, Path]:
    config = yaml.safe_load(config_path.read_text())
    record = read_json(recurrent_record)
    candidate = config["benchmark_circuit_candidate"]
    require(record["status"] == "COMPLETE_SOURCE_CONSTRUCTION" and record["exit_code"] == 0
            and record["stage"] == "recurrent"
            and record["circuit"]["config_sha256"] == sha256_file(config_path)
            and repo_path(repo, record["circuit"]["config"]) == config_path.resolve()
            and record["circuit"]["circuit_index"] == candidate["circuit_index"]
            and record["circuit"]["circuit_id"] == candidate["circuit_id"]
            and record["circuit"]["connectivity_seed"] == config["builder_seed"]
            and record["circuit"]["glif_assignment_seed"] == config["glif_assignment_seed"],
            "V1_INPUT_PREPARED_RECURRENT_PARENT_REQUIRED")
    require(candidate["scenario_id"] == "v1_visual" and candidate["construction_scope"] == "production"
            and candidate["production_eligible"] is True
            and config["expected_full_nodes"] == 230924 and config["full_fraction"] == 1.0,
            "V1_INPUT_FULL_PREPARED_PRODUCTION_CIRCUIT_REQUIRED")
    return config, record, repo_path(repo, record["archive"])


def require_canary_recipe(config: dict, canary_config: dict) -> None:
    realization_keys = {"builder_seed", "glif_assignment_seed", "benchmark_circuit_candidate",
                        "source_root", "bmtk_wheel"}
    require({key: value for key, value in config.items() if key not in realization_keys}
            == {key: value for key, value in canary_config.items() if key not in realization_keys}
            and {key: value for key, value in config["bmtk_wheel"].items() if key != "path"}
            == {key: value for key, value in canary_config["bmtk_wheel"].items() if key != "path"},
            "V1_INPUT_RECIPE_CHANGED_SINCE_CANARY")


def fit_paths(source_root: Path) -> tuple[Path, ...]:
    report = read_json(source_root / "lgn_fitted_models_archive/report.json")
    require(report["status"] == "LGN_FITTED_ARCHIVE_ACQUIRED", "V1_INPUT_COMPLETE_FITS_REQUIRED")
    paths = [Path("source.json"), Path("report.json")]
    for row in report["fitted_files"]:
        path = Path(row["path"])
        require(not path.is_absolute() and ".." not in path.parts, "V1_INPUT_FIT_PATH_INVALID")
        paths.append(path)
    return tuple(paths)


def input_dependencies(repo: Path, config_path: Path, recurrent_record: Path,
                       qualification: Path) -> tuple[Path, ...]:
    """Declare exact files without traversing directories or reading large payloads."""
    config, _, recurrent = bound_request(repo, config_path, recurrent_record)
    source_root = repo_path(repo, config["source_root"])
    paths = {Path(__file__).resolve(), config_path.resolve(), recurrent_record.resolve(),
             repo / "Libs/scripts/run_v1_input_construction.sbatch",
             repo_path(repo, config["bmtk_wheel"]["path"])}
    paths.update(qualification / relative for relative in QUALIFICATION_REPORTS)
    for name in QUALIFIED_MODULE_SHA256:
        paths.add(qualification / "Libs/population_dynamics" / (name + ".py"))
        # Current copies support the focused tests; numerical children use the snapshot.
        paths.add(repo / "Libs/population_dynamics" / (name + ".py"))
    for test in ("test_v1_circuit_inputs.py", "test_v1_input_construction_merlin.py"):
        paths.add(repo / "Libs/population_dynamics" / test)
    for name in SOURCE_NAMES:
        for leaf in ("source.json", "source_asset.txt"):
            paths.add(source_root / name / leaf)
            paths.add(qualification / "reference/sources" / name / leaf)
    for path in fit_paths(source_root):
        paths.add(source_root / "lgn_fitted_models_archive" / path)
        paths.add(qualification / "reference/sources/lgn_fitted_models_archive" / path)
    paths.update(recurrent / relative for relative in RECURRENT_FILES)
    return tuple(sorted(paths))


def merlin_resource_probe(source: str) -> str:
    """Replace only admission and identity fields in the pinned resource probe."""
    replacements = (
        ("import time\n", "import time\n\nfrom .provenance import cpu_execution_provenance\n"),
        ('    if not os.environ.get("SLURM_JOB_ID"):\n'
         '        raise ValueError("NODE_RESOURCE_PROBE_REQUIRES_SLURM")\n',
         '    execution = cpu_execution_provenance()\n'
         '    if execution["backend"] != "merlin":\n'
         '        raise ValueError("NODE_RESOURCE_PROBE_REQUIRES_MERLIN_CPU")\n'),
        ('                  hostname=socket.gethostname(), slurm_job_id=os.environ["SLURM_JOB_ID"],\n'
         '                  cpus_requested=os.environ.get("SLURM_CPUS_PER_TASK"), samples=samples,\n',
         '                  hostname=socket.gethostname(), execution_context=execution,\n'
         '                  cpus_requested=str(execution["allocated_cpus"]), samples=samples,\n'),
    )
    for before, after in replacements:
        require(source.count(before) == 1, "V1_INPUT_RESOURCE_PROBE_IDENTITY_ANCHOR_CHANGED")
        source = source.replace(before, after, 1)
    return source


def copy_verified(source: Path, target: Path, expected_sha256: str) -> None:
    require(sha256_file(source) == expected_sha256, "V1_INPUT_BOUND_FILE_CHANGED:" + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    require(sha256_file(target) == expected_sha256, "V1_INPUT_STAGED_FILE_CHANGED:" + str(target))


def stage_runtime(repo: Path, qualification: Path, output: Path) -> dict:
    package = output / "Libs/population_dynamics"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    bindings = {}
    for name, original_sha in QUALIFIED_MODULE_SHA256.items():
        original = qualification / "Libs/population_dynamics" / (name + ".py")
        require(sha256_file(original) == original_sha, "V1_INPUT_QUALIFIED_MODULE_CHANGED:" + name)
        selected = repo / "Libs/population_dynamics" / (name + ".py") if name in PROVENANCE_RUNTIME_SHA256 else original
        executed_sha = PROVENANCE_RUNTIME_SHA256.get(name, original_sha)
        change = "Merlin CPU placement and execution provenance" if name in PROVENANCE_RUNTIME_SHA256 else "unchanged"
        target = package / (name + ".py")
        if name == "resource_probe":
            body = merlin_resource_probe(original.read_text()).encode()
            executed_sha = hashlib.sha256(body).hexdigest()
            with target.open("xb") as stream:
                stream.write(body)
            change = "Merlin CPU admission and resource identity; original thresholds and sampling retained"
        else:
            copy_verified(selected, target, executed_sha)
        bindings[name] = dict(qualified_source=str(original), qualified_sha256=original_sha,
                              executed_source=str(selected), executed_sha256=executed_sha, change=change)
    return bindings


def resolved_config(config: dict, original_sha256: str, work: Path) -> dict:
    return dict(config, source_root=str(work / "sources"),
                bmtk_wheel=dict(config["bmtk_wheel"], path=str(work / "bmtk.whl")),
                original_config=config, original_config_sha256=original_sha256)


def stage_inputs(repo: Path, config_path: Path, recurrent_record: Path, qualification: Path,
                 output: Path, receipt: dict) -> None:
    config, parent, recurrent = bound_request(repo, config_path, recurrent_record)
    proof = read_json(qualification / "validation.json")
    comparison = read_json(qualification / "comparison/report.json")
    canary = read_json(qualification / "canary/report.json")
    require(proof["status"] == "PASS" and proof["exit_code"] == 0
            and comparison["status"] == "ORIGINAL_CIRCUIT_INPUT_REPLAY_EQUAL"
            and comparison["all_native_files_equal"] and comparison["bound_reference_and_rng_states_equal"],
            "V1_INPUT_PASSED_SOURCE_CANARY_REQUIRED")
    require_canary_recipe(config, canary["original_config"])
    original_versions = canary["runtime"]["versions"]
    actual_versions = versions()
    differences = {name: dict(qualified=original_versions.get(name), actual=actual_versions.get(name))
                   for name in NUMERICAL_PACKAGES if original_versions.get(name) != actual_versions.get(name)}
    receipt.update(circuit=parent["circuit"], parent_job_id=parent["job_id"], parent_job_id_namespace="slurm",
                   recurrent_reference=str(recurrent), canary_reference=str(qualification),
                   qualified_versions=original_versions, actual_versions=actual_versions,
                   numerical_version_differences=differences,
                   mpi4py_version=importlib.metadata.version("mpi4py"),
                   qualified_mpi4py_version=None)
    require(not differences, "V1_INPUT_QUALIFIED_RUNTIME_VERSION_MISMATCH:" + json.dumps(differences, sort_keys=True))
    receipt["runtime_modules"] = stage_runtime(repo, qualification, output)
    receipt["qualification_bindings"] = {}
    for relative in QUALIFICATION_REPORTS:
        path = qualification / relative
        value = sha256_file(path)
        copy_verified(path, output / "qualification" / relative, value)
        receipt["qualification_bindings"][relative] = dict(path=str(path), sha256=value)
    source_root = repo_path(repo, config["source_root"])
    for name in SOURCE_NAMES:
        for leaf in ("source.json", "source_asset.txt"):
            reference = qualification / "reference/sources" / name / leaf
            copy_verified(source_root / name / leaf, output / "sources" / name / leaf, sha256_file(reference))
    for relative in fit_paths(source_root):
        reference = qualification / "reference/sources/lgn_fitted_models_archive" / relative
        copy_verified(source_root / "lgn_fitted_models_archive" / relative,
                      output / "sources/lgn_fitted_models_archive" / relative, sha256_file(reference))
    require(sha256_file(recurrent / "delivery.json") == parent["delivery_sha256"],
            "V1_INPUT_RECURRENT_CONSTRUCTION_DELIVERY_CHANGED")
    for relative in RECURRENT_FILES:
        path = recurrent / relative
        copy_verified(path, output / "recurrent" / relative, sha256_file(path))
    native = read_json(output / "recurrent/primary/report.json")
    require(native["mode"] == "full" and native["original_config"] == config
            and native["original_config_sha256"] == sha256_file(config_path),
            "V1_INPUT_MODE_OR_CONFIG_DIFFERS_FROM_RECURRENT")
    wheel = repo_path(repo, config["bmtk_wheel"]["path"])
    copy_verified(wheel, output / "bmtk.whl", config["bmtk_wheel"]["sha256"])
    original_sha = sha256_file(config_path)
    copy_verified(config_path, output / "original_config.yaml", original_sha)
    copy_verified(recurrent_record, output / "recurrent_construction.json", sha256_file(recurrent_record))
    launcher = repo / "Libs/scripts/run_v1_input_construction.sbatch"
    copy_verified(launcher, output / "launcher.sbatch", sha256_file(launcher))
    copy_verified(Path(__file__), output / "merlin_launcher.py", sha256_file(Path(__file__)))
    with (output / "resolved_config.yaml").open("x") as stream:
        yaml.safe_dump(resolved_config(config, original_sha, output), stream, sort_keys=False)


def phase_commands(output: Path) -> tuple[tuple[str, list[str]], ...]:
    phases = [("resources", [sys.executable, "-B", "-m", "population_dynamics.resource_probe",
                            "--output", str(output / "resources.json"),
                            "--required-memory-gib", str(FULL_REQUIRED_MEMORY_GIB)])]
    for label in ("primary", "replay"):
        phases.append((label, [sys.executable, "-B", "-m", "population_dynamics.v1_input_runtime",
                              "--config", str(output / "resolved_config.yaml"),
                              "--recurrent", str(output / "recurrent"), "--output", str(output / label)]))
    phases.append(("compare", [sys.executable, "-B", "-m", "population_dynamics.v1_circuit_inputs", "compare",
                              "--left", str(output / "primary"), "--right", str(output / "replay"),
                              "--output", str(output / "comparison")]))
    return tuple(phases)


def run(repo: Path, config_path: Path, recurrent_record: Path, qualification: Path, output: Path) -> dict:
    execution = cpu_execution_provenance()
    require(execution["backend"] == "merlin", "V1_INPUT_MERLIN_CPU_EXECUTION_REQUIRED")
    config, _, recurrent = bound_request(repo, config_path, recurrent_record)
    for protected in (qualification, config_path.parent, recurrent, recurrent_record.parent,
                      repo_path(repo, config["source_root"])):
        require(not output.is_relative_to(protected), "V1_INPUT_ISOLATED_DELIVERY_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="PREPARING", exit_code=1, mode="full",
                   job_id="merlin:" + execution["run_id"] + ":" + execution["trial_id"],
                   job_id_namespace="merlin_run_trial", node=execution["hostname"], execution_context=execution,
                   scientific_qualification=False, production_episodes=0, source_simulations=0,
                   training_seeds_used=[], commands=[], phases=[],
                   started_utc=datetime.now(timezone.utc).isoformat())
    code = 1
    try:
        stage_inputs(repo, config_path, recurrent_record, qualification, output, receipt)
        environment = dict(os.environ, PYTHONPATH=str(output / "Libs"), TMPDIR=str(output),
                           PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
        for label, command in phase_commands(output):
            receipt["commands"].append(command)
            print(json.dumps(dict(job_id=receipt["job_id"], phase=label, status="STARTED")), flush=True)
            with (output / (label + ".log")).open("x") as log:
                code = subprocess.run(command, env=environment, cwd=output,
                                      stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            receipt["phases"].append(dict(phase=label, command=command, exit_code=code))
            print(json.dumps(dict(job_id=receipt["job_id"], phase=label, exit_code=code)), flush=True)
            if code:
                break
        if code == 0:
            comparison = read_json(output / "comparison/report.json")
            require(comparison["status"] == "ORIGINAL_CIRCUIT_INPUT_REPLAY_EQUAL"
                    and comparison["all_native_files_equal"] and comparison["bound_reference_and_rng_states_equal"],
                    "V1_INPUT_COMPLETE_REPLAY_REQUIRED")
            for label in ("primary", "replay"):
                report = read_json(output / label / "report.json")
                require(report["status"] == "ORIGINAL_CIRCUIT_INPUTS_COMPLETE" and report["exit_code"] == 0
                        and report["original_config"] == config
                        and report["original_config_sha256"] == sha256_file(config_path)
                        and report["execution_context"] == execution
                        and report["lgn_execution_implementation_sha256"] == QUALIFIED_MODULE_SHA256["v1_input_runtime"],
                        "V1_INPUT_EXECUTED_QUALIFIED_RUNTIME_OR_IDENTITY_DIFFERS")
            receipt["status"] = "PASS_SOURCE_INPUT_BUILD_REPEATABILITY"
        else:
            receipt["status"] = "FAILED_SOURCE_INPUT_BUILD"
    except BaseException as error:
        code = code or 1
        if isinstance(error, KeyboardInterrupt):
            code = 130
        receipt.update(status="FAILED_SOURCE_INPUT_BUILD", error_type=type(error).__name__, error=str(error))
        (output / "failure.txt").write_text(traceback.format_exc())
    finally:
        receipt["child_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        files = []
        for path in sorted(output.rglob("*")):
            relative = path.relative_to(output)
            if not path.is_file() or "__pycache__" in relative.parts or relative.name == "bmtk.whl":
                continue
            files.append(dict(path=str(relative), bytes=path.stat().st_size, sha256=sha256_file(path)))
        receipt.update(files=files, exit_code=code, recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output / "delivery.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "dependencies"):
        command = actions.add_parser(action)
        for name in ("repo", "config", "recurrent-record", "qualification"):
            command.add_argument("--" + name, type=Path, required=True)
        if action == "run":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = dict(repo=args.repo.resolve(), config_path=args.config.resolve(),
                  recurrent_record=args.recurrent_record.resolve(), qualification=args.qualification.resolve())
    if args.action == "dependencies":
        canary = read_json(values["qualification"] / "canary/report.json")
        print(json.dumps(dict(files=[str(path) for path in input_dependencies(**values)],
                              qualified_versions=canary["runtime"]["versions"],
                              required_available_memory_gib=FULL_REQUIRED_MEMORY_GIB,
                              required_extra_python_packages=["mpi4py"], system_programs=["vmstat"],
                              qualified_module_sha256=QUALIFIED_MODULE_SHA256,
                              provenance_runtime_sha256=PROVENANCE_RUNTIME_SHA256), indent=2))
        return 0
    report = run(**values, output=args.output.resolve())
    print(json.dumps({key: value for key, value in report.items() if key not in ("files", "commands")}), flush=True)
    return report["exit_code"] if report["exit_code"] >= 0 else 128 - report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
