"""Contract tests; the standalone C++ fixture is not a neural simulation."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
try:
    from . import ca3_native_observation as obs
except ImportError:
    import ca3_native_observation as obs


def header_bytes(groups=None):
    groups = groups or [
        (0, 2, 3, 100, 0, 2, 3, 0, "pyramidal"),
        (1, 0, 2, 100, 1, 0, 2, 0, "inhibitory"),
    ]
    result = obs.HEADER.pack(obs.MAGIC, len(groups), 3)
    for *values, name in groups:
        encoded = name.encode()
        result += obs.GROUP.pack(*values, len(encoded))+encoded
    return result


def record(kind, start, stop, group, values):
    dtype = "<i4" if kind == b"C" else "<f4"
    return obs.RECORD.pack(kind, start, stop, group, len(values))+np.asarray(values, dtype=dtype).tobytes()


def fixture_records():
    maximum = np.iinfo(np.int32).max
    return [
        record(b"R", 0, 0, 0, [7]),
        record(b"R", 0, 0, 1, [1, -0.0]),
        record(b"A", 0, 1, 0, [7]),
        record(b"A", 0, 1, 1, [1, -0.0]),
        record(b"C", 0, 1, 0, [5]),
        record(b"C", 0, 1, 1, [maximum, 0]),
        # Only this group changes; the first group's requested current persists.
        record(b"R", 1, 1, 1, [2, 3]),
        record(b"A", 1, 2, 0, [7]),
        record(b"A", 1, 2, 1, [2, 3]),
        record(b"C", 1, 2, 0, [3]),
        record(b"C", 1, 2, 1, [maximum, 1]),
        obs.RECORD.pack(b"E", 2, 2, -1, 0),
    ]


def write_fixture(tmp_path, records=None, header=None):
    path = tmp_path/"observation.bin"
    path.write_bytes((header if header is not None else header_bytes())+
                     b"".join(fixture_records() if records is None else records))
    return path


def test_nontrivial_global_mapping_hold_semantics_and_int64_accumulation(tmp_path):
    path = write_fixture(tmp_path)
    result = obs.summarize(path, expected_stop_ms=2)
    assert result["unit_counts"].dtype == np.int64
    assert result["unit_counts"].tolist() == [4294967294, 1, 8]
    assert result["population_counts_per_ms"].tolist() == [[5, 2147483647], [3, 2147483648]]
    intervals = list(obs.iter_intervals(path, expected_stop_ms=2))
    assert [(i.start_ms, i.stop_ms) for i in intervals] == [(0, 1), (1, 2)]
    assert intervals[1].effective_current.tolist() == [2, 3, 7]
    assert not result["independently_reconciled_with_aer"]
    assert not result["current_units"]["numeric_conversion_applied"]
    assert result["current_units"]["model_specific_physical_unit"] == "unresolved"


@pytest.mark.parametrize("values", [[1, 0.0], [2, -0.0]])
def test_effective_current_must_equal_requested_bits(tmp_path, values):
    records = fixture_records()
    records[3] = record(b"A", 0, 1, 1, values)
    with pytest.raises(ValueError, match="REQUEST_RUNTIME_CURRENT_MISMATCH"):
        obs.summarize(write_fixture(tmp_path, records), expected_stop_ms=2)


@pytest.mark.parametrize("change, error", [
    ("missing_request", "MISSING_REQUEST"),
    ("missing_counter", "INTERVAL_ORDER"),
    ("duplicate_group", "INTERVAL_ORDER"),
    ("bad_count", "VALUES_INVALID"),
    ("request_during_step", "INTERVAL_ORDER"),
    ("nonfinite", "CURRENT_NONFINITE"),
    ("bad_record_units", "RECORD_UNIT_COUNT"),
])
def test_incomplete_or_invalid_observation_is_rejected(tmp_path, change, error):
    records = fixture_records()
    if change == "missing_request":
        del records[0]
    elif change == "missing_counter":
        del records[4]
    elif change == "duplicate_group":
        records[3] = records[2]
    elif change == "bad_count":
        records[4] = record(b"C", 0, 1, 0, [-1])
    elif change == "request_during_step":
        records.insert(3, record(b"R", 0, 0, 0, [7]))
    elif change == "nonfinite":
        records[0] = record(b"R", 0, 0, 0, [float("nan")])
    else:
        records[0] = record(b"R", 0, 0, 0, [7, 8])
    with pytest.raises(ValueError, match=error):
        obs.summarize(write_fixture(tmp_path, records), expected_stop_ms=2)


@pytest.mark.parametrize("mode", ["truncated", "no_footer", "trailing", "wrong_stop"])
def test_completion_requires_exact_duration_and_final_record(tmp_path, mode):
    path = write_fixture(tmp_path)
    data = path.read_bytes()
    if mode == "truncated":
        path.write_bytes(data[:-1])
    elif mode == "no_footer":
        path.write_bytes(data[:-obs.RECORD.size])
    elif mode == "trailing":
        path.write_bytes(data+b"extra")
    with pytest.raises(ValueError):
        obs.summarize(path, expected_stop_ms=3 if mode == "wrong_stop" else 2)


def test_group_coverage_must_be_exact(tmp_path):
    groups = [(0, 1, 2, 100, 0, 0, 1, 0, "a"), (1, 0, 2, 100, 1, 2, 4, 0, "b")]
    with pytest.raises(ValueError, match="GROUP_COVERAGE"):
        obs.summarize(write_fixture(tmp_path, header=header_bytes(groups)), expected_stop_ms=2)


@pytest.fixture
def pinned_files():
    source = Path(os.environ.get("CARLSIM_OBSERVATION_SOURCE",
                               str(Path(__file__).resolve().parents[1]/"external/carlsim_ca3")))
    if not all((source/name).is_file() for name in obs.PINNED_FILES):
        pytest.skip("pinned CARLsim sources are not installed")
    return {name: (source/name).read_bytes() for name in obs.PINNED_FILES}


def copied_tree(path, files):
    for name, data in files.items():
        target = path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def test_pinned_overlay_hooks_and_original_preservation(tmp_path, pinned_files):
    original, staging = tmp_path/"original", tmp_path/"staging"
    copied_tree(original, pinned_files)
    copied_tree(staging, pinned_files)
    receipt = obs.apply_overlay(original, staging)
    manager = (staging/"carlsim/kernel/src/snn_manager.cpp").read_text()
    assert "\tca3ObserveCurrents();\n\tdoCurrentUpdate();" in manager
    assert "\t\tfetchNeuronSpikeCount(ALL);\n\t\tca3ObserveCounts();" in manager
    assert manager.index("CA3_OBSERVATION_REQUIRES_ONE_MS_CALLS") < manager.index("resetSpikeCnt(ALL)")
    assert "cudaMemcpyDeviceToHost" in manager
    assert "managerRuntimeData.nSpikeCnt+m.gStartN" in manager
    assert not receipt["native_build_verified"]
    for name, data in pinned_files.items():
        assert (original/name).read_bytes() == data
    with pytest.raises(ValueError, match="PINNED_SOURCE_CHANGED"):
        obs.apply_overlay(original, staging)


def test_changed_source_fails_before_any_overlay_write(tmp_path, pinned_files):
    original, staging = tmp_path/"original", tmp_path/"staging"
    copied_tree(original, pinned_files)
    copied_tree(staging, pinned_files)
    name = "carlsim/kernel/src/snn_cpu_module.cpp"
    (staging/name).write_bytes(pinned_files[name]+b"\n")
    before = {p: (staging/p).read_bytes() for p in pinned_files}
    with pytest.raises(ValueError, match="PINNED_SOURCE_CHANGED"):
        obs.apply_overlay(original, staging)
    assert not (staging/obs.GENERATED_HEADER).exists()
    assert before == {p: (staging/p).read_bytes() for p in pinned_files}


def test_original_and_hardlink_aliases_are_rejected(tmp_path, pinned_files):
    original, staging = tmp_path/"original", tmp_path/"staging"
    copied_tree(original, pinned_files)
    copied_tree(staging, pinned_files)
    with pytest.raises(ValueError, match="SEPARATE_STAGING"):
        obs.apply_overlay(original, original)
    name = next(iter(pinned_files))
    (staging/name).unlink()
    os.link(original/name, staging/name)
    with pytest.raises(ValueError, match="ALIASES_SOURCE"):
        obs.apply_overlay(original, staging)


def test_generated_writer_and_cpu_hooks_compile_and_roundtrip(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ unavailable for the standalone serialization contract")
    (tmp_path/"ca3_native_observation.h").write_text(obs.CPP_WRITER)
    # This fake runtime tests the injected C++ interface and serialized values,
    # not a mechanistic simulator or GPU execution.
    scaffold = r'''
#include "ca3_native_observation.h"
#include <map>
#include <climits>
#define __NO_CUDA__
const int CPU_RUNTIME_BASE=100;
struct GroupConfigMD { int gStartN,gEndN,netId,lGrpId,lStartN,lEndN; };
struct GroupConfig { std::string grpName; };
struct Runtime { float* extCurrent; int* nSpikeCnt; };
class SNN {
public:
    nrnsim_ca3_observation::Writer ca3Observation_;
    int numGroups=2, time=0;
    struct { int numN=3; } glbNetworkConfig;
    std::map<int,GroupConfigMD> groupConfigMDMap;
    std::map<int,GroupConfig> groupConfigMap;
    Runtime runtimeData[101],managerRuntimeData;
    int getSimTime() { return time; }
    bool isPoissonGroup(int) { return false; }
    void ca3ObserveInitialize();
    void ca3ObserveCurrents();
    void ca3ObserveCounts();
};
'''
    main = r'''
int main() {
  try {
    SNN s;
    s.groupConfigMDMap[0]={2,2,100,0,2,2};
    s.groupConfigMDMap[1]={0,1,100,1,0,1};
    s.groupConfigMap[0]={"pyramidal"}; s.groupConfigMap[1]={"inhibitory"};
    float effective[]={1,-0.0f,7};
    int counters[]={INT_MAX,0,5};
    s.runtimeData[100].extCurrent=effective;
    s.managerRuntimeData.nSpikeCnt=counters;
    s.ca3ObserveInitialize();
    s.ca3Observation_.requested(0,0,{7});
    s.ca3Observation_.requested(0,1,{1,-0.0f});
    s.ca3ObserveCurrents(); s.time=1; s.ca3ObserveCounts();
    effective[0]=2; effective[1]=3;
    counters[1]=1; counters[2]=3;
    s.ca3Observation_.requested(1,1,{2,3});
    s.ca3ObserveCurrents(); s.time=2; s.ca3ObserveCounts();
    s.ca3Observation_.finish(2);
    return 0;
  } catch (const std::exception& e) { std::fprintf(stderr,"%s\\n",e.what()); return 2; }
}
'''
    source = tmp_path/"contract.cpp"
    source.write_text(scaffold+obs.CPP_METHODS+main)
    binary = tmp_path/"contract"
    subprocess.run([compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    output = tmp_path/"observations.bin"
    env = dict(os.environ, **{obs.ENVIRONMENT_VARIABLE: str(output)})
    subprocess.run([str(binary)], env=env, cwd=tmp_path, check=True, capture_output=True)
    result = obs.summarize(output, expected_stop_ms=2)
    assert result["unit_counts"].tolist() == [4294967294, 1, 8]
    assert result["population_counts_per_ms"].tolist() == [[5, 2147483647], [3, 2147483648]]
    original = output.read_bytes()
    repeated = subprocess.run([str(binary)], env=env, cwd=tmp_path, capture_output=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == original

