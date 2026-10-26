"""Validate streamed source inputs and both source RNG policies before scaling."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import shutil
import numpy as np
from .pfc_native_drive import create_drive, verify
from . import pfc_native_drive, pfc_deterministic_replay
from .pfc_deterministic_replay import SOURCE_ENTRY_SHA256, _array_comparison, _save_arrays, replay, simulate
from .provenance import freeze_json, sha256_file
from . import pfc_kernel_parameters as kernel_parameters


def _native_kernel_parameters(report):
    parameters = kernel_parameters.resolve_kernel_parameters(report['resolved_config'])
    if (parameters != kernel_parameters.DEFAULT_KERNEL_PARAMETERS
            or 'resolved_kernel_parameters' in report or 'kernel_parameterization_sha256' in report):
        if (report.get('resolved_kernel_parameters') != parameters
                or report.get('kernel_parameterization_sha256') != sha256_file(Path(kernel_parameters.__file__))):
            raise ValueError('PFC_QUALIFICATION_NATIVE_KERNEL_BINDING_INVALID')
    return parameters


def native_clock_scope(validation, input_result, global_config):
    # Consume the exact native report already checked by the input replayer.
    name = 'capture/native/report.json'
    rows = [row for row in input_result['consumed_files'] if row['path'] == name]
    path = Path(validation)/name
    if (input_result['status'] != 'PASS_NATIVE_NUMPY_DRIVE_REPLAY' or len(rows) != 1
            or path.stat().st_size != rows[0]['bytes'] or sha256_file(path) != rows[0]['sha256']):
        raise ValueError('PFC_QUALIFICATION_NATIVE_CLOCK_BINDING_INVALID')
    native = json.loads(path.read_text())
    clock = native['resolved_config']['dt_ms']
    if (type(clock) not in (int,float) or not np.isfinite(clock) or clock <= 0
            or clock != global_config['dt_ms']):
        raise ValueError('PFC_QUALIFICATION_NATIVE_CLOCKS_DIFFER')
    if _native_kernel_parameters(native) != kernel_parameters.resolve_kernel_parameters(global_config):
        raise ValueError('PFC_QUALIFICATION_NATIVE_KERNELS_DIFFER')
    return [float(clock)]


def reuse_completed_phases(validation, pairing, previous, output, backend):
    """Reuse terminal phase evidence after resource interruption, with exact bindings."""
    previous, output, validation, pairing = map(Path, (previous, output, validation, pairing))
    input_path = previous / "input_replay/comparison.json"
    recorded_path = previous / "recorded_regression/comparison.json"
    inputs = json.loads(input_path.read_text())
    recorded = json.loads(recorded_path.read_text())
    delivery = json.loads((validation / "delivery.json").read_text())
    pairing_report = json.loads(pairing.read_text())
    dsha, psha = sha256_file(validation / "delivery.json"), sha256_file(pairing)
    if (inputs.get("status") != "PASS_NATIVE_NUMPY_DRIVE_REPLAY"
            or inputs.get("completed_comparisons") != 24 or inputs.get("passed_comparisons") != 24
            or inputs.get("numpy_version") != np.__version__
            or inputs.get("implementation_sha256") != sha256_file(Path(pfc_native_drive.__file__))
            or inputs.get("source_entry_sha256") != SOURCE_ENTRY_SHA256
            or recorded.get("status") != "PASS_PFC_DETERMINISTIC_REPLAY"
            or recorded.get("backend") != backend or recorded.get("completed_cases") != 8
            or recorded.get("passed_cases") != 8 or recorded.get("not_completed_cases") != 0
            or recorded.get("implementation_sha256") != sha256_file(Path(pfc_deterministic_replay.__file__))
            or delivery.get("status") != "ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY"
            or pairing_report.get("status") != "PASS_NATIVE_INPUT_PAIRING"
            or pairing_report.get("delivery_sha256") != dsha):
        raise ValueError("PFC_COMPLETED_PHASE_STATUS_OR_PRODUCER_CHANGED")
    manifests = {row["path"]: row for row in delivery["files"]}
    if len(manifests) != len(delivery["files"]):
        raise ValueError("PFC_COMPLETED_PHASE_DUPLICATE_INPUT")
    copied = []
    checked_inputs = set()

    def copy_bound(origin, relative, digest):
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts or not name.parts:
            raise ValueError("PFC_COMPLETED_PHASE_OUTPUT_PATH_INVALID")
        source = origin / name
        if sha256_file(source) != digest:
            raise ValueError("PFC_COMPLETED_PHASE_OUTPUT_CHANGED:" + str(name))
        target = output / origin.name / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(dict(path=str(target.relative_to(output)), sha256=digest))

    for phase, report in ((input_path.parent, inputs), (recorded_path.parent, recorded)):
        if report.get("delivery_sha256") != dsha or report.get("pairing_sha256") != psha:
            raise ValueError("PFC_COMPLETED_PHASE_INPUT_BINDING_CHANGED")
        for row in report["consumed_files"]:
            if row["path"] in checked_inputs:
                continue
            bound = manifests.get(row["path"])
            if (bound is None or any(row[key] != bound[key] for key in ("sha256", "bytes"))
                    or (validation / row["path"]).stat().st_size != row["bytes"]
                    or sha256_file(validation / row["path"]) != row["sha256"]):
                raise ValueError("PFC_COMPLETED_PHASE_SOURCE_LINEAGE_CHANGED:" + row["path"])
            checked_inputs.add(row["path"])
        copy_bound(phase, "comparison.json", sha256_file(phase / "comparison.json"))
    expected_cases = [case["case_id"] for case in recorded["resolved_config"]["cases"]]
    expected_input = {(case, block) for case in expected_cases for block in (1, 37, 256)}
    observed_input = [(case.get("case_id"), case.get("block_rows")) for case in inputs.get("cases", [])]
    observed_recorded = [case.get("case_id") for case in recorded.get("cases", [])]
    if (len(expected_cases) != 8 or len(set(expected_cases)) != 8
            or len(observed_input) != 24 or set(observed_input) != expected_input
            or observed_recorded != expected_cases):
        raise ValueError("PFC_COMPLETED_PHASE_DENOMINATOR_CHANGED")
    for case in inputs["cases"]:
        if case.get("passed") is not True:
            raise ValueError("PFC_COMPLETED_PHASE_INPUT_CASE_FAILED")
        copy_bound(input_path.parent, case["receipt_file"], case["receipt_sha256"])
        receipt = json.loads((input_path.parent / case["receipt_file"]).read_text())
        if any(receipt.get(key) != value for key, value in case.items()
               if key not in ("receipt_file", "receipt_sha256")):
            raise ValueError("PFC_COMPLETED_PHASE_INPUT_CASE_CHANGED")
        copy_bound(input_path.parent, case["generated_rng_file"], case["generated_rng_sha256"])
    for case in recorded["cases"]:
        if case.get("status") != "PASS_NUMERICAL_SOURCE_COMPARISON":
            raise ValueError("PFC_COMPLETED_PHASE_RECORDED_CASE_FAILED")
        copy_bound(recorded_path.parent, case["report_file"], case["report_sha256"])
        receipt = json.loads((recorded_path.parent / case["report_file"]).read_text())
        if (receipt.get("case_id") != case["case_id"]
                or receipt.get("status") != "PASS_NUMERICAL_SOURCE_COMPARISON"
                or receipt.get("implementation_sha256") != recorded["implementation_sha256"]
                or receipt.get("pairing_sha256") != psha or receipt.get("delivery_sha256") != dsha
                or not all(row.get("passed") is True for row in receipt["comparisons"])):
            raise ValueError("PFC_COMPLETED_PHASE_RECORDED_CASE_CHANGED")
        parent = Path(case["report_file"]).parent
        copy_bound(recorded_path.parent, parent / receipt["raw_file"], receipt["raw_sha256"])
        copy_bound(recorded_path.parent, parent / receipt["terminal_state_file"], receipt["terminal_state_sha256"])
    with (input_path.parent / "drive_comparisons.csv").open(newline="") as stream:
        input_rows = list(csv.DictReader(stream))
    expected_rows = {(case, block, population) for case, block in expected_input for population in ("E", "I")}
    observed_rows = [(row.get("case_id"), int(row.get("block_rows", 0)), row.get("population")) for row in input_rows]
    if (len(observed_rows) != 48 or set(observed_rows) != expected_rows
            or any(row.get("passed") != "True" for row in input_rows)):
        raise ValueError("PFC_COMPLETED_PHASE_INPUT_TABLE_CHANGED")
    with (recorded_path.parent / "array_comparisons.csv").open(newline="") as stream:
        recorded_rows = list(csv.DictReader(stream))
    if (not recorded_rows or {row.get("case_id") for row in recorded_rows} != set(expected_cases)
            or any(row.get("passed") != "True" for row in recorded_rows)):
        raise ValueError("PFC_COMPLETED_PHASE_RECORDED_TABLE_CHANGED")
    native_clock_scope(validation, inputs, recorded["resolved_config"])
    for origin, report, name, key in ((input_path.parent, inputs, "drive_comparisons.csv", "comparison_table_sha256"),
                                      (recorded_path.parent, recorded, "array_comparisons.csv", "array_comparisons_sha256")):
        copy_bound(origin, name, report[key])
    return inputs, recorded, dict(previous=str(previous), input_report_sha256=sha256_file(input_path),
                                  recorded_report_sha256=sha256_file(recorded_path), reused_outputs=copied,
                                  native_input_payload_rehashed=True, previous_phase_simulations_repeated=0)


def validate(validation, pairing, reference, output, backend, *, reuse_completed=None):
    validation,pairing,reference,output=map(Path,(validation,pairing,reference,output))
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='FAILED_NATIVE_STREAM_SOURCE_EXECUTION',backend=backend,numpy_version=np.__version__,
        source_entry_sha256=SOURCE_ENTRY_SHA256,executor_sha256=sha256_file(Path(__file__).with_name('pfc_deterministic_replay.py')),
        drive_sha256=sha256_file(Path(__file__).with_name('pfc_native_drive.py')),
        kernel_parameterization_sha256=sha256_file(Path(kernel_parameters.__file__)),
        input_replay_status='NOT_RUN',input_replay_passed_comparisons=0,global_cases_passed=0,global_arrays_passed=0,
        global_cases=[],scientific_fidelity_qualified=False,full_scale_qualified=False,production_episodes=0)
    freeze_json(output/'run.json',report)
    try:
        if reuse_completed is None:
            input_result=verify(validation,pairing,output/'input_replay')
            recorded=None
        else:
            input_result,recorded,reuse=reuse_completed_phases(
                validation,pairing,reuse_completed,output,backend)
            report['completed_phase_reuse']=reuse
        report.update(input_replay_status=input_result['status'],input_replay_passed_comparisons=input_result['passed_comparisons'],
                      input_replay_file='input_replay/comparison.json',input_replay_sha256=sha256_file(output/'input_replay/comparison.json'))
        if input_result['status']!='PASS_NATIVE_NUMPY_DRIVE_REPLAY':raise ValueError('native input sequence did not reproduce')
        if recorded is None:
            recorded=replay(validation,pairing,output/'recorded_regression',backend)
        report.update(recorded_regression_status=recorded['status'],recorded_regression_cases=recorded['passed_cases'])
        if recorded['status']!='PASS_PFC_DETERMINISTIC_REPLAY':raise ValueError('recorded-input regression failed')
        delivery_path=reference/'delivery.json';delivery=json.loads(delivery_path.read_text())
        entries={r['path']:r for r in delivery['files']}
        def bound(name):
            path=reference/name;entry=entries[name]
            if path.stat().st_size!=entry['bytes'] or sha256_file(path)!=entry['sha256']:raise ValueError('legacy source changed:'+name)
            return path
        native_path=bound('plain/report.json');native=json.loads(native_path.read_text());config=native['resolved_config']
        if (native['status']!='NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS' or native['source_entry_sha256']!=SOURCE_ENTRY_SHA256
            or config.get('random_stream_policy','source_global')!='source_global' or len(native['cases'])!=2
            or native['source_revision']!=config['expected_source_revision']):raise ValueError('complete pinned legacy source required')
        report.update(global_reference_delivery_sha256=sha256_file(delivery_path),global_reference_report_sha256=sha256_file(native_path))
        report['validated_dt_ms'] = native_clock_scope(validation, input_result, config)
        report['validated_kernel_parameters'] = [_native_kernel_parameters(native)]
        if [r['case_id'] for r in config['cases']]!=[r['case_id'] for r in native['cases']]:raise ValueError('legacy case denominator changed')
        for case,record in zip(config['cases'],native['cases']):
            initial,drive=create_drive(config,case)
            result=simulate(config,case,initial,backend=backend,drive=drive)
            folder=output/case['case_id'];folder.mkdir()
            _save_arrays(folder/'raw.npz',result['raw']);_save_arrays(folder/'terminal_state.npz',result['terminal_state'])
            source=bound('plain/'+record['raw_file'])
            if sha256_file(source)!=record['raw_sha256']:raise ValueError('native raw receipt changed')
            with np.load(source,allow_pickle=False) as truth:
                comparisons=[_array_comparison(case['case_id'],'raw',name,result['raw'].get(name),truth[name]) for name in truth.files]
                if set(truth.files)!=set(result['raw']):raise ValueError('raw source fields changed')
            passed=all(r['passed'] for r in comparisons)
            row=dict(case_id=case['case_id'],passed=passed,comparisons=comparisons,cost=result['cost'],
                     inputs_used=result['inputs_used'],driver=drive.metadata(),raw_sha256=sha256_file(folder/'raw.npz'))
            freeze_json(folder/'comparison.json',row);report['global_cases'].append(row)
            report['global_cases_passed']+=int(passed);report['global_arrays_passed']+=sum(r['passed'] for r in comparisons)
        if report['global_cases_passed']!=2 or report['global_arrays_passed']!=10:raise ValueError('global source execution mismatch')
        report['status']='PASS_NATIVE_STREAM_SOURCE_EXECUTION'
    except Exception as error:
        report['failure']=dict(type=type(error).__name__,message=str(error))
    freeze_json(output/'qualification.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('validation','pairing','reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--backend',choices=('numpy','cuda'),default='numpy')
    parser.add_argument('--reuse-completed',type=Path,help='Reuse completed input and recorded-regression phases after interruption.')
    a=parser.parse_args();r=validate(a.validation,a.pairing,a.reference,a.output,a.backend,reuse_completed=a.reuse_completed)
    print(json.dumps({k:v for k,v in r.items() if k!='global_cases'},indent=2))
    return 0 if r['status']=='PASS_NATIVE_STREAM_SOURCE_EXECUTION' else 1

if __name__=='__main__':sys.exit(main())
