"""Qualify V1 source launchers on a bulk compute node against frozen inputs.

Canaries do not count as mechanistic episodes. Full-rate and full-spike
comparisons must pass before a node can be used for production source work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import h5py
import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_nest_input_replay import bound_json
from .v1_circuit_stimulus import rate_config_without_noise
from .v1_execution_recipe import require_same_recipe, visual_recipe


def _delivered_file(archive: Path, relative: str, destination: Path) -> Path:
    delivery = json.loads((archive / "delivery.json").read_text())
    if delivery.get("exit_code") != 0:
        raise ValueError("NODE_QUALIFICATION_REFERENCE_NOT_COMPLETE")
    entries = [row for row in delivery["files"] if row["path"] == relative]
    if len(entries) != 1:
        raise ValueError("NODE_QUALIFICATION_REFERENCE_FILE_MISSING")
    row = entries[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive / relative, destination)
    if destination.stat().st_size != row["bytes"] or sha256_file(destination) != row["sha256"]:
        raise ValueError("NODE_QUALIFICATION_REFERENCE_FILE_CHANGED")
    return destination


def compare_rates(reference: Path, current: Path, work: Path, *, complete: bool) -> dict:
    before, _ = bound_json(reference, "probe/probe.json")
    after, _ = bound_json(current, "probe/probe.json")
    if before["input_construction_binding"] != after["input_construction_binding"]:
        raise ValueError("NODE_QUALIFICATION_INPUT_CONSTRUCTION_CHANGED")
    config = deepcopy(after["config"])
    config["unit_coverage"] = before["config"]["unit_coverage"]
    if rate_config_without_noise(before["config"]) != rate_config_without_noise(config):
        raise ValueError("NODE_QUALIFICATION_RATE_CONFIG_CHANGED")
    count = 17400 if complete else 14
    if after["source_units"] != count or before["source_units"] != 17400:
        raise ValueError("NODE_QUALIFICATION_RATE_DENOMINATOR_CHANGED")
    row_by_id = {value: index for index, value in enumerate(before["source_unit_ids"])}
    selected = [row_by_id[value] for value in after["source_unit_ids"]]
    left = _delivered_file(reference, "probe/firing_rates_f_tot.h5", work / "reference_rates.h5")
    right = _delivered_file(current, "probe/firing_rates_f_tot.h5", work / "current_rates.h5")
    mismatches = values = 0
    maximum = 0.0
    with h5py.File(left, "r") as a, h5py.File(right, "r") as b:
        clock_exact = bool(np.array_equal(a["time"][:], b["time"][:]))
        if len(set(selected)) != count or b["firing_rates_Hz"].shape != (count, before["rate_shape"][1]):
            raise ValueError("NODE_QUALIFICATION_RATE_SHAPE_CHANGED")
        for start in range(0, count, 256):
            indices = selected[start:start + 256]
            # HDF5 fancy indexing requires increasing indices; preserve source order.
            expected = np.stack([a["firing_rates_Hz"][index] for index in indices])
            observed = b["firing_rates_Hz"][start:start + len(indices)]
            if expected.dtype != observed.dtype:
                raise ValueError("NODE_QUALIFICATION_RATE_DTYPE_CHANGED")
            mismatches += int(np.count_nonzero(expected != observed))
            maximum = max(maximum, float(np.max(np.abs(expected - observed))))
            values += expected.size
    if not clock_exact or mismatches:
        raise ValueError(f"NODE_QUALIFICATION_RATE_VALUES_DIFFER:{mismatches}:{maximum}")
    return dict(source_units=count, compared_rate_values=values, clock_exact=clock_exact,
                mismatched_rate_values=mismatches, maximum_absolute_difference_hz=maximum,
                reference=str(reference), current=str(current))


def compare_spikes(reference: Path, current: Path, work: Path) -> dict:
    before, _ = bound_json(reference, "visual_inputs/report.json")
    after, _ = bound_json(current, "visual_inputs/report.json")
    if before["config"] != after["config"] or before["input_construction_binding"] != after["input_construction_binding"]:
        raise ValueError("NODE_QUALIFICATION_SPIKE_INPUTS_CHANGED")
    compared = {}
    for population in ("lgn", "bkg"):
        relative = f"visual_inputs/{population}/trial_0000.npz"
        left = _delivered_file(reference, relative, work / ("reference_" + population + ".npz"))
        right = _delivered_file(current, relative, work / ("current_" + population + ".npz"))
        with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError("NODE_QUALIFICATION_SPIKE_FIELDS_CHANGED")
            for name in a.files:
                if a[name].dtype != b[name].dtype or not np.array_equal(a[name], b[name]):
                    raise ValueError("NODE_QUALIFICATION_SPIKES_DIFFER:" + population + ":" + name)
            compared[population] = list(a.files)
    if after["complete_lgn_population"] is not True:
        raise ValueError("NODE_QUALIFICATION_FULL_SPIKES_REQUIRED")
    return dict(reference=str(reference), current=str(current), all_arrays_exact=True,
                population_fields=compared, complete_lgn_population=True)


def execute(manifest_path: Path, phase: str, case_index: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    root, repo = manifest_path.parent, Path(manifest["repo"])
    if os.environ.get("SLURMD_NODENAME") != manifest["node"] or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("NODE_QUALIFICATION_CPU_PLACEMENT_CHANGED")
    case = manifest["cases"][case_index]
    if case["case_index"] != case_index:
        raise ValueError("NODE_QUALIFICATION_CASE_INDEX_CHANGED")
    role = "rates" if phase.startswith("rates_") else "spikes"
    launcher = Path(manifest["launchers"][role]["path"])
    if sha256_file(launcher) != manifest["launchers"][role]["sha256"]:
        raise ValueError("NODE_QUALIFICATION_LAUNCHER_CHANGED")
    job = os.environ["SLURM_JOB_ID"]
    output = root / phase / f"case_{case_index:02d}" / ("run_" + job)
    output.mkdir(parents=True, exist_ok=False)
    family = repo / manifest["source_family"]
    archive = family / (role + "_" + job)
    report = dict(status="RUNNING", phase=phase, case_index=case_index, job_id=job,
                  node=manifest["node"], source_archive=str(archive), source_simulations=0,
                  production_episodes=0, recorded_utc=datetime.now(timezone.utc).isoformat())
    locator = root / phase / f"case_{case_index:02d}" / "execution.json"
    if locator.exists():
        raise ValueError("NODE_QUALIFICATION_PRIOR_EXECUTION_REQUIRES_RECONCILIATION")
    locator.write_text(json.dumps(dict(job_id=job, report=str(output / "report.json"),
                                      source_archive=str(archive)), indent=2) + "\n")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("NRNSIM_")}
    start = time.monotonic()
    try:
        if role == "rates":
            code_reference = case["reference_rate"]
            if phase == "rates_full":
                prior = json.loads((root / "rates_canary" / f"case_{case_index:02d}" / "execution.json").read_text())
                passed = json.loads(Path(prior["report"]).read_text())
                if passed["status"] != "PASS_NODE_SOURCE_QUALIFICATION":
                    raise ValueError("NODE_QUALIFICATION_RATE_CANARY_NOT_PASSED")
                code_reference = prior["source_archive"]
            environment.update(NRNSIM_V1_LGN_CONFIG=case["config"],
                NRNSIM_V1_LGN_INPUT_CONSTRUCTION=case["input_source"],
                NRNSIM_V1_LGN_PREPARED_ARCHIVE=case["reference_rate"],
                NRNSIM_V1_LGN_CODE_REFERENCE=code_reference,
                NRNSIM_V1_LGN_UNIT_COVERAGE="all_source_units" if phase == "rates_full" else "model_representatives")
            command = ["bash", str(launcher), str(repo)]
        else:
            rate = case["reference_rate"]
            prior_spikes = case["reference_spike_canary"]
            mode = "canary"
            if phase == "spikes_canary":
                prior = json.loads((root / "rates_canary" / f"case_{case_index:02d}" / "execution.json").read_text())
                if json.loads(Path(prior["report"]).read_text())["status"] != "PASS_NODE_SOURCE_QUALIFICATION":
                    raise ValueError("NODE_QUALIFICATION_RATE_CANARY_NOT_PASSED")
                rate = prior["source_archive"]
                command = ["bash", str(launcher), str(repo), mode, rate, case["input_source"], prior_spikes]
            else:
                prior = json.loads((root / "spikes_canary" / f"case_{case_index:02d}" / "execution.json").read_text())
                if json.loads(Path(prior["report"]).read_text())["status"] != "PASS_NODE_SOURCE_QUALIFICATION":
                    raise ValueError("NODE_QUALIFICATION_SPIKE_CANARY_NOT_PASSED")
                command = ["bash", str(launcher), str(repo), "full", rate, case["input_source"],
                           prior["source_archive"], case["config"]]
        report["command"] = command
        with (output / "launcher.log").open("x") as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        report["launcher_exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError("NODE_QUALIFICATION_SOURCE_LAUNCHER_FAILED")
        delivery = json.loads((archive / "delivery.json").read_text())
        if delivery["exit_code"] != 0:
            raise ValueError("NODE_QUALIFICATION_SOURCE_DELIVERY_NOT_PASSED")
        with tempfile.TemporaryDirectory(prefix="neuronsim-node-source-comparison-", dir="/tmp") as temporary:
            work = Path(temporary)
            if role == "rates":
                report["comparison"] = compare_rates(Path(case["reference_rate"]), archive, work, complete=phase == "rates_full")
            elif phase == "spikes_full":
                report["comparison"] = compare_spikes(Path(case["reference_spike"]), archive, work)
            else:
                if delivery["status"] != "PASS_CONSTRUCTED_VISUAL_INPUT_CANARY":
                    raise ValueError("NODE_QUALIFICATION_SPIKE_CANARY_DELIVERY_INVALID")
                report["consumer_readback"] = delivery["consumer_readback"]
        report["status"] = "PASS_NODE_SOURCE_QUALIFICATION"
        return report
    except BaseException as error:
        report.update(status="FAILED_NODE_SOURCE_QUALIFICATION", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - start, finished_utc=datetime.now(timezone.utc).isoformat())
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")



def collect_qualification(manifest_path: Path, output: Path) -> dict:
    """Collect completed source comparisons without changing their evidence."""
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    if (manifest["node"] not in ("cpu05", "cpu06")
            or manifest["numerical_source_modules_changed"]
            or manifest["canonical_launchers_changed"]
            or manifest["full_rate_cases"] != [0, 8, 9]
            or manifest["full_spike_cases"] != list(range(10))
            or len(manifest["cases"]) != 17):
        raise ValueError("NODE_QUALIFICATION_MANIFEST_SCOPE_CHANGED")
    bindings = {}

    def bind(path, expected=None):
        path = Path(path)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("NODE_QUALIFICATION_EVIDENCE_CHANGED:" + str(path))
        bindings[str(path)] = digest

    bind(manifest_path)
    bind(manifest["source_qualification"], manifest["source_qualification_sha256"])
    for path, digest in manifest["software_bindings"].items():
        bind(path, digest)
    for launcher in manifest["launchers"].values():
        bind(launcher["path"], launcher["sha256"])
        bind(launcher["original_path"], launcher["original_sha256"])
    expected = {"rates_canary": list(range(17)), "spikes_canary": list(range(17)),
                "rates_full": [0, 8, 9], "spikes_full": list(range(10))}
    evidence, references = [], {}
    for phase, indices in expected.items():
        for index in indices:
            locator_path = root / phase / f"case_{index:02d}" / "execution.json"
            locator = json.loads(locator_path.read_text())
            path = Path(locator["report"])
            report = json.loads(path.read_text())
            archive = Path(locator["source_archive"])
            if (report["status"] != "PASS_NODE_SOURCE_QUALIFICATION"
                    or report["phase"] != phase or report["case_index"] != index
                    or report["node"] != manifest["node"]
                    or report["job_id"] != locator["job_id"]
                    or report["source_archive"] != str(archive)
                    or report["launcher_exit_code"] != 0 or report["production_episodes"] != 0):
                raise ValueError("NODE_QUALIFICATION_PHASE_NOT_PASSED:" + str(path))
            bind(locator_path)
            bind(path)
            bind(archive / "delivery.json")
            if phase.startswith("rates_"):
                comparison = report["comparison"]
                units = 17400 if phase == "rates_full" else 14
                if (comparison["source_units"] != units or not comparison["clock_exact"]
                        or comparison["mismatched_rate_values"] != 0
                        or comparison["maximum_absolute_difference_hz"] != 0.0):
                    raise ValueError("NODE_QUALIFICATION_RATE_COMPARISON_NOT_EXACT")
            elif phase == "spikes_full":
                if (report["comparison"]["all_arrays_exact"] is not True
                        or report["comparison"]["complete_lgn_population"] is not True):
                    raise ValueError("NODE_QUALIFICATION_SPIKE_COMPARISON_NOT_EXACT")
            if phase.endswith("_canary"):
                role = "rate_code_reference" if phase == "rates_canary" else "spike_canary"
                references.setdefault(index, {})[role] = str(archive)
                relative = "probe/probe.json" if phase == "rates_canary" else "visual_inputs/report.json"
                source, delivery = bound_json(archive, relative)
                if delivery["exit_code"] != 0:
                    raise ValueError("NODE_QUALIFICATION_CANARY_DELIVERY_FAILED")
                bind(archive / relative)
                if phase == "rates_canary":
                    references[index]["config"] = source["config"]
                else:
                    require_same_recipe(source["config"], references[index]["config"], family="visual")
            evidence.append(dict(phase=phase, case_index=index, job_id=report["job_id"],
                                 report=str(path), elapsed_seconds=report["elapsed_seconds"]))
    result = dict(status="PASS_V1_SOURCE_NODE_QUALIFICATION", node=manifest["node"],
                  partition="cpu", manifest=str(manifest_path),
                  source_qualification_sha256=manifest["source_qualification_sha256"],
                  launchers=manifest["launchers"], cases=[references[i] for i in range(17)],
                  completed_cases={phase: len(indices) for phase, indices in expected.items()},
                  evidence=evidence, source_bindings=bindings, production_episodes=0,
                  recorded_utc=datetime.now(timezone.utc).isoformat())
    freeze_json(output, result)
    return result



def source_node_export_launcher(repo: Path, output: Path, node: str) -> Path:
    """Reuse the exporter with only its permitted CPU node changed."""
    if node not in ("cpu05", "cpu06"):
        raise ValueError("NODE_QUALIFICATION_EXPORT_CPU_INVALID")
    original = (repo / "Libs/scripts/run_v1_production_block_export.sbatch").read_text()
    guard = '[[ "${SLURMD_NODENAME:-}" == cpu03 &&'
    if original.count(guard) != 1:
        raise ValueError("NODE_QUALIFICATION_EXPORT_NODE_GUARD_CHANGED")
    current = original.replace(guard, '[[ "${SLURMD_NODENAME:-}" =~ ^(cpu03|' + node + ')$ &&', 1)
    path = output / ("run_v1_production_block_export_" + node + ".sbatch")
    if path.exists():
        if path.read_text() != current:
            raise ValueError("NODE_QUALIFICATION_EXPORT_LAUNCHER_CHANGED")
    else:
        with path.open("x") as stream:
            stream.write(current)
    return path


def load_source_node_qualification(path: Path, source_qualification_sha256: str,
                                   bindings: dict) -> dict:
    """Reject missing, changed or incomplete node evidence before submission."""
    report = json.loads(path.read_text())
    if (report["status"] != "PASS_V1_SOURCE_NODE_QUALIFICATION"
            or report["node"] not in ("cpu05", "cpu06") or report["partition"] != "cpu"
            or report["source_qualification_sha256"] != source_qualification_sha256
            or report["completed_cases"] !=
            dict(rates_canary=17, spikes_canary=17, rates_full=3, spikes_full=10)
            or len(report["cases"]) != 17):
        raise ValueError("NODE_QUALIFICATION_COMPLETE_SOURCE_EVIDENCE_REQUIRED")
    for name, digest in report["source_bindings"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("NODE_QUALIFICATION_EVIDENCE_CHANGED:" + name)
    bindings.update(report["source_bindings"])
    bindings[str(path)] = sha256_file(path)
    return report


def qualified_source_case(qualification: dict, recipe: dict) -> dict:
    matches = [row for row in qualification["cases"]
               if visual_recipe(row["config"]) == visual_recipe(recipe)]
    if len(matches) != 1:
        raise ValueError("NODE_QUALIFICATION_UNIQUE_CONDITION_REQUIRED")
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("rates_canary", "spikes_canary", "rates_full", "spikes_full"))
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--aggregate-output", type=Path)
    args = parser.parse_args(argv)
    if args.aggregate_output is not None:
        if args.phase is not None or args.case_index is not None:
            parser.error("--aggregate-output cannot be combined with an execution phase")
        print(json.dumps(collect_qualification(args.manifest, args.aggregate_output)))
        return
    if args.phase is None:
        parser.error("--phase or --aggregate-output is required")
    index = args.case_index if args.case_index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(json.dumps(execute(args.manifest, args.phase, index)))


if __name__ == "__main__":
    main()
