"""Resume committed native observation cases with a separate execution adapter.

The pinned source, reference executor and observer remain unchanged. The adapter
records its own implementation. Only complete source and observation pairs can
be reused; unfinished case files stay in the preserved failed input archive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import FunctionType
from unittest.mock import patch

import h5py
import numpy as np
import yaml

from . import pfc_reference as reference
from .pfc_input_capture import _bind_native_case, _bound
from .pfc_input_observation import NativeInputObserver
from .circuit_data import CircuitDataset, CircuitUnits
from .provenance import freeze_json, sha256_file
from .pfc_factorized_source import SOURCE_ENTRY_SHA256


KINDS = ("terminal_state", "numpy_rng", "python_rng", "generated_code")


def restore_observations(previous: Path, root: Path, observer, config: dict, recovered: dict) -> list[str]:
    """Bind each reused observation to an already validated source episode."""
    resolved = reference.case_settings(config)
    expected = {case["case_id"]:case for case in resolved}
    if not recovered or not set(recovered) <= set(expected):
        raise ValueError("PFC_OBSERVATION_RECOVERY_CASE_SET_INVALID")
    summaries, copied = [], []
    native_report = dict(resolved_config=config, source_entry_sha256=SOURCE_ENTRY_SHA256)
    for case in resolved:
        name = case["case_id"]
        if name not in recovered:
            continue
        origin = previous / "inputs" / name / "observation.json"
        receipt = json.loads(origin.read_text())
        if (receipt.get("status") != "NATIVE_INPUT_OBSERVATION_CASE_COMPLETE"
                or receipt.get("mode") != observer.mode or receipt.get("case") != case
                or receipt.get("case_id") != name or receipt.get("source_entry_sha256") != SOURCE_ENTRY_SHA256):
            raise ValueError("PFC_OBSERVATION_RECOVERY_RECEIPT_CHANGED")
        summary = dict(case_id=name, mode=observer.mode)
        for kind in KINDS:
            filename, digest = receipt[kind+"_file"], receipt[kind+"_sha256"]
            _bound(previous / "inputs", filename, digest)
            summary[kind+"_file"], summary[kind+"_sha256"] = filename, digest
        _bind_native_case(previous, summary, receipt, native_report, recovered[name]["case"], case)
        snapshots = [(receipt[kind+"_file"], receipt[kind+"_sha256"]) for kind in KINDS]
        if observer.mode == "capture":
            for provider in receipt["providers"]:
                path = _bound(previous / "inputs", provider["counts_file"], provider["counts_sha256"])
                counts = np.load(path, allow_pickle=False, mmap_mode="r")
                if (counts.dtype != np.uint16 or list(counts.shape) != provider["counts_shape"]
                        or int(counts.sum(dtype=np.uint64)) != provider["captured_external_events"]):
                    raise ValueError("PFC_OBSERVATION_RECOVERY_CAPTURE_CHANGED")
                snapshots.append((provider["counts_file"], provider["counts_sha256"]))
        snapshots.append((name+"/observation.json", sha256_file(origin)))
        for filename, digest in snapshots:
            source = _bound(previous / "inputs", filename, digest)
            relative = Path(filename)
            if relative.parts[0] != name:
                raise ValueError("PFC_OBSERVATION_RECOVERY_CASE_FILE_PATH_CHANGED")
            target = observer.output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256_file(target) != digest:
                raise ValueError("PFC_OBSERVATION_RECOVERY_STAGE_CHANGED")
            copied.append(dict(path=filename, sha256=digest))
        summary.update(receipt_file=name+"/observation.json", receipt_sha256=sha256_file(origin))
        summaries.append(summary)
    if len(summaries) != len(recovered):
        raise ValueError("PFC_OBSERVATION_RECOVERY_DENOMINATOR_CHANGED")
    observer._cases.extend(summaries)
    freeze_json(root / "reused_observations.json", dict(
        reused_case_ids=[row["case_id"] for row in summaries], previous=str(previous.resolve()),
        files=copied, recovery_implementation_sha256=sha256_file(Path(__file__)),
        source_simulations_repeated=0, production_episodes=0))
    return [row["case_id"] for row in summaries]


def observed_source_function(module, observer, cases: list[dict], config: dict):
    """Observe only missing calls; the returned function keeps source globals."""
    original = module.simulate_wm
    position = [0]

    def intercept(*args, **kwargs):
        if position[0] >= len(cases):
            raise ValueError("PFC_OBSERVATION_RECOVERY_UNDECLARED_SOURCE_CALL")
        case = cases[position[0]]
        observer.begin_case(case, config)
        result = original(*args, **kwargs)
        observer.finish_case(case, result)
        position[0] += 1
        return result

    return FunctionType(intercept.__code__, original.__globals__, original.__name__,
                        intercept.__defaults__, intercept.__closure__)


def record(source: Path, config_path: Path, previous: Path, output: Path, mode: str) -> dict:
    if output.exists() or mode not in ("audit", "capture"):
        raise ValueError("PFC_OBSERVATION_RECOVERY_FRESH_OUTPUT_AND_MODE_REQUIRED")
    config = yaml.safe_load(config_path.read_text())
    dataset = CircuitDataset(previous / "native/dataset", splits=("development",))
    catalog = dataset.catalog
    binding = catalog["binding"]
    if (binding.get("resolved_source_config") != config
            or binding.get("config_sha256") != sha256_file(config_path)
            or not catalog["shards"]):
        raise ValueError("PFC_OBSERVATION_RECOVERY_ORIGINAL_CONFIG_CHANGED")
    output.mkdir(parents=True)
    native = output / "native"
    (native / "dataset").mkdir(parents=True)
    # Stage committed files only. Preserve any uncommitted source and observer files.
    copied = []

    def stage(relative, digest):
        src = previous / "native" / relative
        if sha256_file(src) != digest:
            raise ValueError("PFC_OBSERVATION_RECOVERY_SOURCE_FILE_CHANGED:"+relative)
        dst = native / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256_file(dst) != digest:
            raise ValueError("PFC_OBSERVATION_RECOVERY_SOURCE_STAGE_CHANGED")
        copied.append(dict(path=relative, sha256=digest))

    stage("dataset/dataset.json", sha256_file(previous / "native/dataset/dataset.json"))
    for row in catalog["circuits"].values():
        stage("dataset/"+row["path"],row["sha256"])
    for shard in catalog["shards"]:
        stage("dataset/"+shard["path"], shard["sha256"])
        with h5py.File(native / "dataset" / shard["path"], "r") as stored:
            for row in shard["episodes"]:
                metadata = json.loads(stored[row["episode_id"]].attrs["metadata_json"])
                source_record = metadata["source_record"]
                stage(row["episode_id"]+"_native.npz",source_record["raw_sha256"])
                if "initial_voltage_file" in source_record:
                    stage(source_record["initial_voltage_file"], source_record["initial_voltage_sha256"])
    n_e, n_i = config["excitatory_neurons"], config["inhibitory_neurons"]
    units = CircuitUnits("pfc_source_reference", np.arange(n_e+n_i),
        np.asarray(["E"]*n_e+["I"]*n_i), np.r_[np.arange(n_e)*config["excitatory_preference_bins"]//n_e,
            np.full(n_i, config["excitatory_preference_bins"])],
        tuple([f"E_preference_{i*360/config['excitatory_preference_bins']:g}_{(i+1)*360/config['excitatory_preference_bins']:g}_deg"
               for i in range(config["excitatory_preference_bins"])] + ["I_population"]),
        "E angle=360*source_index/N_E degrees; I has no preference coordinate", binding["source_revision"],
        circuit_provenance=reference.kernel_parameters.circuit_provenance(config,binding["source_entry_sha256"]))
    recovered = reference.resume_cases(native,binding,units,reference.case_settings(config),round(config["duration_ms"]/config["dt_ms"]))
    observer = NativeInputObserver(output / "inputs", mode=mode)
    reused = restore_observations(previous,output,observer,config,recovered)
    missing = [case for case in reference.case_settings(config) if case["case_id"] not in recovered]
    original_load = reference.load_source

    def observe_module(source_path):
        module, manifest, entry = original_load(source_path)
        observer.install(module)
        module.simulate_wm = observed_source_function(module,observer,missing,config)
        return module,manifest,entry

    try:
        with patch.object(reference,"load_source",side_effect=observe_module):
            native_report = reference.execute(source,config_path,native,resume=True)
        inputs = observer.finalize()
        if [case["case_id"] for case in inputs["cases"]] != [case["case_id"] for case in native_report["cases"]]:
            raise ValueError("PFC_OBSERVATION_RECOVERY_COMPLETE_CASE_ORDER_CHANGED")
        report = dict(status="RECORDED_NATIVE_PFC_INPUT_OBSERVATION",mode=mode,
            recorded_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha256_file(config_path),
            native_report_sha256=sha256_file(native/"report.json"),
            observation_report_sha256=sha256_file(output/"inputs/report.json"),cases=len(inputs["cases"]),
            implementation_sha256=sha256_file(Path(__file__)),
            observer_implementation_sha256=sha256_file(Path(reference.__file__).with_name("pfc_input_observation.py")),
            reference_implementation_sha256=sha256_file(Path(reference.__file__)),
            recovery_adapter_implementation_sha256=sha256_file(Path(__file__)),
            native_source_function_intercepted_for_observation=True,native_equations_altered=False,
            reused_case_ids=reused,new_source_cases=len(missing),source_simulations_repeated=0,
            original_partial_output_preserved=str(previous.resolve()),source_parity_verified=False,
            gpu_parity_qualified=False,scientific_fidelity_qualified=False,production_episodes=0)
        freeze_json(output/"execution.json",report)
        return report
    finally:
        observer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","config","previous","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--mode",choices=("audit","capture"),required=True)
    args=parser.parse_args()
    print(json.dumps(record(args.source,args.config,args.previous,args.output,args.mode),indent=2))


if __name__ == "__main__":
    main()
