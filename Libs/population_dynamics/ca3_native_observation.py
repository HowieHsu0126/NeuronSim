"""Opt-in observation of the pinned native CA3 runtime, without changing dynamics.

Apply only to a fresh copied CARLsim tree, before its CMake build. Set
NRNSIM_CA3_OBSERVATION_FILE to a new binary path when running that build.
The source program must advance with runNetwork(0, 1). Requested setter vectors,
effective runtime extCurrent immediately before doCurrentUpdate, and native
nSpikeCnt after the existing fetch are separate records. GPU current is copied
device-to-host into an observation-only vector.

Current floats are preserved exactly: the API documents mA, while the author's
model-specific physical unit remains unresolved. Counts reset at every native
runNetwork call; this reader accumulates the one-ms snapshots using int64.
Observation adds transfers and I/O, so its execution cost is not a baseline cost.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import BinaryIO, Iterator

import numpy as np

REVISION = "c4955998b950b9c561b3373a71b611aedc705be9"
ENVIRONMENT_VARIABLE = "NRNSIM_CA3_OBSERVATION_FILE"
CURRENT_UNIT = {
    "api_documented_unit": "mA",
    "model_specific_physical_unit": "unresolved",
    "numeric_conversion_applied": False,
}
PINNED_FILES = {
    "carlsim/kernel/inc/snn.h": "44f7cadc204b83b2cccd2ac2ae4fb48b7e88c2b9d64c5223aa5b1651a94bfa76",
    "carlsim/kernel/inc/snn_datastructures.h": "50c71ccea154bae8c5da5e04dd0fef93ea603f2e3106abe44bf05dfec944d0b9",
    "carlsim/kernel/src/snn_manager.cpp": "d14315a63715c4a0a722a22d6e0acfa542f3946ce40cba64c86c91aca8904a1d",
    "carlsim/kernel/src/gpu_module/snn_gpu_module.cu": "4e25494a89427f5479838a17dc48cff5b255954061689146068dd85e16c44b4d",
    "carlsim/kernel/src/snn_cpu_module.cpp": "238111d64b7f672f7198a3eed5749a5a68e1c8a7f06770917a2e4486fd616e50",
}
GENERATED_HEADER = "carlsim/kernel/inc/ca3_native_observation.h"
MAGIC = b"NSCA3OB1"
HEADER = struct.Struct("<8sII")
GROUP = struct.Struct("<8iI")
RECORD = struct.Struct("<cqqiI")

# Standalone standard C++11 serialization, also compiled by the focused test.
CPP_WRITER = r"""
#ifndef NRNSIM_CA3_NATIVE_OBSERVATION_H
#define NRNSIM_CA3_NATIVE_OBSERVATION_H
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace nrnsim_ca3_observation {
struct Group {
    int id, start, stop, net, local_group, local_start, local_stop, gpu;
    std::string name;
};
class Writer {
    FILE* file_;
    std::string path_;
    bool initialized_;
    static void fail(const char* message) { throw std::runtime_error(message); }
    void bytes(const void* data, size_t size) {
        if (size && std::fwrite(data, 1, size, file_) != size)
            fail("CA3_OBSERVATION_WRITE_FAILED");
    }
    template<class T> void scalar(T value) { bytes(&value, sizeof(value)); }
    void record(char type, int64_t start, int64_t stop, int group,
                const void* values, uint32_t count) {
        scalar(type); scalar(start); scalar(stop); scalar(int32_t(group));
        scalar(count); bytes(values, size_t(count)*4);
    }
public:
    Writer() : file_(NULL), initialized_(false) {
        const char* value = std::getenv("NRNSIM_CA3_OBSERVATION_FILE");
        if (value) path_ = value;
    }
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    ~Writer() { if (file_) std::fclose(file_); }
    bool enabled() const { return !path_.empty(); }
    bool initialized() const { return initialized_; }
    void initialize(const std::vector<Group>& groups, int units, int64_t time) {
        if (!enabled() || initialized_) return;
        static_assert(sizeof(float)==4 && sizeof(int)==4, "32-bit native values required");
        static_assert(std::numeric_limits<float>::is_iec559, "IEEE float required");
        uint32_t endian = 1;
        if (*reinterpret_cast<unsigned char*>(&endian)!=1 || time!=0)
            fail("CA3_OBSERVATION_PLATFORM_OR_START_INVALID");
        file_ = std::fopen(path_.c_str(), "wbx");
        if (!file_) fail("CA3_OBSERVATION_OPEN_FAILED_OR_PATH_EXISTS");
        bytes("NSCA3OB1",8); scalar(uint32_t(groups.size())); scalar(uint32_t(units));
        for (size_t i=0; i<groups.size(); ++i) {
            const Group& g = groups[i];
            scalar(int32_t(g.id)); scalar(int32_t(g.start)); scalar(int32_t(g.stop));
            scalar(int32_t(g.net)); scalar(int32_t(g.local_group));
            scalar(int32_t(g.local_start)); scalar(int32_t(g.local_stop));
            scalar(int32_t(g.gpu)); scalar(uint32_t(g.name.size()));
            bytes(g.name.data(),g.name.size());
        }
        initialized_ = true;
    }
    void requested(int64_t time, int group, const std::vector<float>& values) {
        record('R',time,time,group,values.data(),uint32_t(values.size()));
    }
    void applied(int64_t time, int group, const std::vector<float>& values) {
        record('A',time,time+1,group,values.data(),uint32_t(values.size()));
    }
    void counts(int64_t stop, int group, const int* values, int count) {
        record('C',stop-1,stop,group,values,uint32_t(count));
    }
    void finish(int64_t time) {
        if (!file_) return;
        record('E',time,time,-1,NULL,0);
        FILE* closing = file_; file_ = NULL;
        if (std::fclose(closing)!=0) fail("CA3_OBSERVATION_CLOSE_FAILED");
    }
};
}
#endif
"""

CPP_METHODS = r"""
// Observation buffers never replace runtime currents or neuronal state.
void SNN::ca3ObserveInitialize() {
    if (!ca3Observation_.enabled() || ca3Observation_.initialized()) return;
    std::vector<nrnsim_ca3_observation::Group> groups;
    for (int g=0; g<numGroups; ++g) {
        if (isPoissonGroup(g))
            throw std::runtime_error("CA3_OBSERVATION_REQUIRES_REGULAR_NEURONS");
        const GroupConfigMD& m = groupConfigMDMap.at(g);
        nrnsim_ca3_observation::Group out = {
            g,m.gStartN,m.gEndN+1,m.netId,m.lGrpId,m.lStartN,m.lEndN+1,
            m.netId<CPU_RUNTIME_BASE ? 1 : 0,groupConfigMap.at(g).grpName};
        groups.push_back(out);
    }
    ca3Observation_.initialize(groups,glbNetworkConfig.numN,getSimTime());
}
void SNN::ca3ObserveCurrents() {
    if (!ca3Observation_.enabled()) return;
    ca3ObserveInitialize();
    for (int g=0; g<numGroups; ++g) {
        const GroupConfigMD& m = groupConfigMDMap.at(g);
        const int count = m.lEndN-m.lStartN+1;
        std::vector<float> effective(count);
        if (m.netId<CPU_RUNTIME_BASE) {
#ifndef __NO_CUDA__
            checkAndSetGPUDevice(m.netId);
            cudaError_t result = cudaMemcpy(effective.data(),
                runtimeData[m.netId].extCurrent+m.lStartN,
                size_t(count)*sizeof(float),cudaMemcpyDeviceToHost);
            if (result!=cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(result));
#else
            throw std::runtime_error("CA3_OBSERVATION_GPU_UNAVAILABLE");
#endif
        } else {
            std::memcpy(effective.data(),runtimeData[m.netId].extCurrent+m.lStartN,
                        size_t(count)*sizeof(float));
        }
        ca3Observation_.applied(getSimTime(),g,effective);
    }
}
void SNN::ca3ObserveCounts() {
    if (!ca3Observation_.enabled()) return;
    for (int g=0; g<numGroups; ++g) {
        const GroupConfigMD& m = groupConfigMDMap.at(g);
        ca3Observation_.counts(getSimTime(),g,
            managerRuntimeData.nSpikeCnt+m.gStartN,m.gEndN-m.gStartN+1);
    }
}
"""


def _replace_once(data: bytes, old: bytes, new: bytes) -> bytes:
    if data.count(old) != 1:
        raise ValueError("CA3_OBSERVATION_SOURCE_ANCHOR_CHANGED")
    return data.replace(old, new, 1)


def overlay_bytes(files: dict[str, bytes]) -> dict[str, bytes]:
    """Validate original bytes, then return just the three generated/changed files."""
    for name, digest in PINNED_FILES.items():
        if name not in files or hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError("CA3_OBSERVATION_PINNED_SOURCE_CHANGED: " + name)
    header_name, manager_name = "carlsim/kernel/inc/snn.h", "carlsim/kernel/src/snn_manager.cpp"
    header = _replace_once(files[header_name], b"#include <map>",
                           b'#include "ca3_native_observation.h"\n#include <map>')
    declaration = b"""private:
    nrnsim_ca3_observation::Writer ca3Observation_;
    void ca3ObserveInitialize();
    void ca3ObserveCurrents();
    void ca3ObserveCounts();
"""
    header = _replace_once(header, b"private:\n", declaration)
    manager = _replace_once(files[manager_name], b"#include <snn.h>\n",
                            b"#include <snn.h>\n#ifndef __NO_CUDA__\n#include <cuda_runtime_api.h>\n#endif\n")
    manager = _replace_once(manager, b"SNN::~SNN() {\n",
                           b"SNN::~SNN() {\n\tca3Observation_.finish(getSimTime());\n")
    anchor = b"\tint runDurationMs = _nsec*1000 + _nmsec;\n"
    manager = _replace_once(manager, anchor, anchor + b"""    if (ca3Observation_.enabled() && runDurationMs!=1)
        throw std::runtime_error("CA3_OBSERVATION_REQUIRES_ONE_MS_CALLS");
""")
    manager = _replace_once(manager, b"\tdoCurrentUpdate();\n",
                           b"\tca3ObserveCurrents();\n\tdoCurrentUpdate();\n")
    manager = _replace_once(manager, b"\t\tfetchNeuronSpikeCount(ALL);\n",
                           b"\t\tfetchNeuronSpikeCount(ALL);\n\t\tca3ObserveCounts();\n")
    # The complete setter tail uniquely identifies the existing CPU/GPU assignment.
    anchor = b"""        copyExternalCurrent(netId, lGrpId, &runtimeData[netId], false);
    }
}""".replace(b"        ", b"\t\t").replace(b"    ", b"\t")
    replacement = anchor[:-1] + b"""    if (ca3Observation_.enabled()) {
        ca3ObserveInitialize();
        ca3Observation_.requested(getSimTime(),grpId,current);
    }
}"""
    manager = _replace_once(manager, anchor, replacement)
    manager += CPP_METHODS.encode()
    return {header_name: header, manager_name: manager,
            GENERATED_HEADER: CPP_WRITER.encode()}


def apply_overlay(original_root: Path, staging_root: Path) -> dict:
    """Modify only an explicitly supplied, distinct, fresh copied source tree.

    All five original and staged files are checked before writing. Symlinks or
    hard links back to original files are rejected. Reapplication requires a
    fresh copy; this function does not overwrite a previously generated header.
    """
    original_root, staging_root = Path(original_root).resolve(), Path(staging_root).resolve()
    if staging_root == original_root or original_root in staging_root.parents:
        raise ValueError("CA3_OBSERVATION_REQUIRES_SEPARATE_STAGING")
    inputs = {}
    for relative, digest in PINNED_FILES.items():
        source, target = original_root/relative, staging_root/relative
        if (target.is_symlink() or staging_root not in target.resolve().parents
                or source.samefile(target)):
            raise ValueError("CA3_OBSERVATION_STAGING_ALIASES_SOURCE")
        original, staged = source.read_bytes(), target.read_bytes()
        if original != staged or hashlib.sha256(original).hexdigest() != digest:
            raise ValueError("CA3_OBSERVATION_PINNED_SOURCE_CHANGED: " + relative)
        inputs[relative] = staged
    generated = staging_root/GENERATED_HEADER
    if generated.exists() or generated.is_symlink():
        raise FileExistsError(generated)
    outputs = overlay_bytes(inputs)
    for relative, data in outputs.items():
        (staging_root/relative).write_bytes(data)
    return dict(framework_revision=REVISION, enabled_by=ENVIRONMENT_VARIABLE,
                original_sha256=PINNED_FILES.copy(),
                overlay_sha256={p: hashlib.sha256(b).hexdigest() for p, b in outputs.items()},
                current_units=CURRENT_UNIT.copy(), counter_source="native nSpikeCnt",
                count_interval_ms=1, accumulated_count_dtype="int64",
                current_source="runtime extCurrent; GPU device-to-host or CPU direct",
                native_build_verified=False, native_execution_verified=False)


@dataclass(frozen=True)
class Group:
    group_id: int
    start: int
    stop: int
    net_id: int
    local_group: int
    local_start: int
    local_stop: int
    gpu: int
    name: str

    @property
    def units(self) -> int:
        return self.stop-self.start


@dataclass(frozen=True)
class Interval:
    start_ms: int
    stop_ms: int
    native_counts: np.ndarray
    requested_current: np.ndarray
    effective_current: np.ndarray


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("CA3_OBSERVATION_TRUNCATED")
    return value


def _header(stream: BinaryIO) -> tuple[int, tuple[Group, ...]]:
    magic, count, units = HEADER.unpack(_read_exact(stream, HEADER.size))
    if magic != MAGIC or not 0 < count <= units:
        raise ValueError("CA3_OBSERVATION_HEADER_INVALID")
    groups = []
    for i in range(count):
        values = GROUP.unpack(_read_exact(stream, GROUP.size))
        *indices, name_size = values
        if not 0 < name_size <= 4096:
            raise ValueError("CA3_OBSERVATION_GROUP_NAME_INVALID")
        group = Group(*indices, _read_exact(stream, name_size).decode("utf-8"))
        if (group.group_id != i or not 0 <= group.start < group.stop <= units
                or min(group.net_id, group.local_group, group.local_start) < 0
                or group.local_stop-group.local_start != group.units
                or group.gpu not in (0, 1)):
            raise ValueError("CA3_OBSERVATION_GROUP_INVALID")
        groups.append(group)
    ordered = sorted(groups, key=lambda g: g.start)
    if (ordered[0].start != 0 or ordered[-1].stop != units
            or any(a.stop != b.start for a, b in zip(ordered, ordered[1:]))
            or len({g.name for g in groups}) != count
            or len({(g.net_id, g.local_group) for g in groups}) != count):
        raise ValueError("CA3_OBSERVATION_GROUP_COVERAGE_INVALID")
    return units, tuple(groups)


def read_header(path: Path) -> tuple[int, tuple[Group, ...]]:
    with Path(path).open("rb") as stream:
        return _header(stream)


def _record(stream: BinaryIO, groups: tuple[Group, ...]):
    kind, start, stop, gid, count = RECORD.unpack(_read_exact(stream, RECORD.size))
    if kind == b"E":
        if gid != -1 or count:
            raise ValueError("CA3_OBSERVATION_FOOTER_INVALID")
        return kind, start, stop, gid, b""
    if kind not in (b"R", b"A", b"C") or not 0 <= gid < len(groups):
        raise ValueError("CA3_OBSERVATION_RECORD_INVALID")
    if count != groups[gid].units:
        raise ValueError("CA3_OBSERVATION_RECORD_UNIT_COUNT_INVALID")
    return kind, start, stop, gid, _read_exact(stream, count*4)


def _intervals(stream: BinaryIO, units: int, groups: tuple[Group, ...],
               expected_stop_ms: int) -> Iterator[Interval]:
    requested = np.empty(units, dtype="<f4")
    known = set()
    time = 0
    while True:
        record = _record(stream, groups)
        while record[0] == b"R":
            _, start, stop, gid, payload = record
            if start != time or stop != time:
                raise ValueError("CA3_OBSERVATION_REQUEST_TIME_INVALID")
            group = groups[gid]
            values = np.frombuffer(payload, dtype="<f4")
            if not np.isfinite(values).all():
                raise ValueError("CA3_OBSERVATION_CURRENT_NONFINITE")
            requested[group.start:group.stop] = values
            known.add(gid)
            record = _record(stream, groups)
        if record[0] == b"E":
            if record[1:3] != (time, time) or time != expected_stop_ms or stream.read(1):
                raise ValueError("CA3_OBSERVATION_DURATION_OR_FOOTER_INVALID")
            return
        if time >= expected_stop_ms or len(known) != len(groups):
            raise ValueError("CA3_OBSERVATION_MISSING_REQUEST_OR_EXTRA_INTERVAL")
        effective = np.empty(units, dtype="<f4")
        counts = np.empty(units, dtype=np.int64)
        for kind, target in ((b"A", effective), (b"C", counts)):
            for group in groups:
                actual, start, stop, gid, payload = record
                if (actual, start, stop, gid) != (kind, time, time+1, group.group_id):
                    raise ValueError("CA3_OBSERVATION_INTERVAL_ORDER_OR_COVERAGE_INVALID")
                values = np.frombuffer(payload, dtype="<f4" if kind == b"A" else "<i4")
                if (kind == b"A" and not np.isfinite(values).all()) or (kind == b"C" and (values < 0).any()):
                    raise ValueError("CA3_OBSERVATION_VALUES_INVALID")
                target[group.start:group.stop] = values
                if not (kind == b"C" and group == groups[-1]):
                    record = _record(stream, groups)
        # Compare bit patterns: signed zero and every original float bit matter.
        if not np.array_equal(requested.view("<u4"), effective.view("<u4")):
            raise ValueError("CA3_OBSERVATION_REQUEST_RUNTIME_CURRENT_MISMATCH")
        yield Interval(time, time+1, counts, requested.copy(), effective)
        time += 1


def iter_intervals(path: Path, *, expected_stop_ms: int) -> Iterator[Interval]:
    """Stream complete, reconciled intervals; exhaust iteration to validate footer.

    Yields all units in global-ID order. One interval is buffered; the entire
    current tensor is never materialized. A missing final record is an error.
    """
    if type(expected_stop_ms) is not int or expected_stop_ms <= 0:
        raise ValueError("CA3_OBSERVATION_EXPECTED_DURATION_INVALID")
    with Path(path).open("rb") as stream:
        units, groups = _header(stream)
        yield from _intervals(stream, units, groups, expected_stop_ms)


def summarize(path: Path, *, expected_stop_ms: int) -> dict:
    """Accumulate int64 native counts; AER reconciliation is a separate check."""
    if type(expected_stop_ms) is not int or expected_stop_ms <= 0:
        raise ValueError("CA3_OBSERVATION_EXPECTED_DURATION_INVALID")
    with Path(path).open("rb") as stream:
        units, groups = _header(stream)
        counts = np.zeros(units, dtype=np.int64)
        population = np.empty((expected_stop_ms, len(groups)), dtype=np.int64)
        for interval in _intervals(stream, units, groups, expected_stop_ms):
            if np.any(counts > np.iinfo(np.int64).max-interval.native_counts):
                raise ValueError("CA3_OBSERVATION_COUNT_OVERFLOW")
            counts += interval.native_counts
            for group in groups:
                population[interval.start_ms, group.group_id] = interval.native_counts[group.start:group.stop].sum()
    return dict(groups=groups, unit_counts=counts, population_counts_per_ms=population,
                duration_ms=expected_stop_ms, current_units=CURRENT_UNIT.copy(),
                host_request_runtime_current_equal=True,
                independently_reconciled_with_aer=False,
                scientific_fidelity_qualified=False, production_episodes=0)

