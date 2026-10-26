"""Host allocation contract tests; fake CUDA is not native GPU qualification."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from . import ca3_conductance_memory as memory
from . import ca3_native_observation as observation


@pytest.fixture(scope="module")
def source_root():
    path = Path(os.environ.get("NEURONSIM_CARLSIM_SOURCE_ROOT",
                              str(Path(__file__).resolve().parents[1]/"external/carlsim_ca3")))
    if not (path/memory.GPU_FILE).is_file():
        pytest.skip("pinned CA3 source is not installed")
    return path


@pytest.fixture(scope="module")
def source_bytes(source_root):
    data = (source_root/memory.GPU_FILE).read_bytes()
    assert hashlib.sha256(data).hexdigest() == memory.PINNED_SHA256
    return data


def copied_gpu(root, data):
    path = root/memory.GPU_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def host_sections(source):
    """Extract actual host sections independently of overlay replacement constants."""
    allocation_start = source.index(
        b"#if JK_CA3_SNN\n\tif(allocateMem) {\n\t\tnetworkConfigs[netId].syn_gLength")
    allocation_stop = source.index(b"\n#endif", allocation_start) + len(b"\n#endif")
    free_start = source.index(
        b"#if JK_CA3_SNN\n\tCUDA_CHECK_ERRORS( cudaFree(runtimeData[netId].AMPA_syn_g)")
    free_stop = source.index(b"\n#endif", free_start) + len(b"\n#endif")
    return source[allocation_start:allocation_stop], source[free_start:free_stop]


def test_overlay_preserves_original_and_every_byte_outside_host_sections(tmp_path, source_bytes):
    original, staging = tmp_path/"original", tmp_path/"staging"
    source = copied_gpu(original, source_bytes)
    target = copied_gpu(staging, source_bytes)
    untouched = staging/"untouched.txt"
    untouched.write_bytes(b"unchanged")
    receipt = memory.apply_overlay(original, staging)
    output = target.read_bytes()
    before, after = source_bytes, output
    for section in host_sections(source_bytes):
        before = before.replace(section, b"", 1)
    for section in host_sections(output):
        after = after.replace(section, b"", 1)
    assert before == after  # Includes every kernel, accessor, and host caller.
    assert source.read_bytes() == source_bytes
    assert untouched.read_bytes() == b"unchanged"
    assert receipt["input_sha256"] == receipt["original_sha256"]
    assert receipt["overlay_sha256"][memory.GPU_FILE] == hashlib.sha256(output).hexdigest()
    assert receipt["initialization_correction"]["corrected_extent"] == "syn_gPitch * numNReg"
    assert receipt["kernel_arithmetic_unchanged"]
    assert receipt["precision_unchanged"]
    assert not receipt["native_build_verified"]
    assert not receipt["native_execution_verified"]
    assert not receipt["native_qualification"]
    with pytest.raises(ValueError, match="STAGED_SOURCE_CHANGED"):
        memory.apply_overlay(original, staging)
    assert target.read_bytes() == output


@pytest.mark.parametrize("changed", ["original", "staging"])
def test_source_identity_rejection_writes_nothing(tmp_path, source_bytes, changed):
    original, staging = tmp_path/"original", tmp_path/"staging"
    copied_gpu(original, source_bytes)
    target = copied_gpu(staging, source_bytes)
    path = (original if changed == "original" else staging)/memory.GPU_FILE
    path.write_bytes(source_bytes + b"\n")
    before = target.read_bytes()
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        memory.apply_overlay(original, staging)
    assert target.read_bytes() == before


@pytest.mark.parametrize("alias", ["same", "nested", "hardlink", "symlink", "parent_symlink"])
def test_source_aliases_are_rejected(tmp_path, source_bytes, alias):
    original, staging = tmp_path/"original", tmp_path/"staging"
    source = copied_gpu(original, source_bytes)
    if alias in ("same", "nested"):
        staging = original if alias == "same" else original/"nested"
    elif alias == "parent_symlink":
        staging.mkdir()
        (staging/"carlsim").symlink_to(original/"carlsim", target_is_directory=True)
    else:
        target = staging/memory.GPU_FILE
        target.parent.mkdir(parents=True)
        if alias == "hardlink":
            os.link(source, target)
        else:
            target.symlink_to(source)
    with pytest.raises(ValueError, match="SEPARATE_STAGING|ALIASES_SOURCE"):
        memory.apply_overlay(original, staging)
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize("change", ["allocation", "free", "duplicate"])
def test_unexpected_source_shape_is_rejected_before_write(tmp_path, source_bytes, change):
    allocation, cleanup = host_sections(source_bytes)
    if change == "duplicate":
        changed = source_bytes + allocation
    else:
        old = allocation if change == "allocation" else cleanup
        changed = source_bytes.replace(old, old.replace(b"CUDA_CHECK_ERRORS", b"CHANGED", 1), 1)
    original, staging = tmp_path/"original", tmp_path/"staging"
    copied_gpu(original, source_bytes)
    target = copied_gpu(staging, changed)
    preceding = {"original_sha256": {memory.GPU_FILE: memory.PINNED_SHA256},
                 "overlay_sha256": {memory.GPU_FILE: hashlib.sha256(changed).hexdigest()}}
    with pytest.raises(ValueError, match="SOURCE_ANCHOR_CHANGED"):
        memory.apply_overlay(original, staging, preceding_overlay=preceding)
    assert target.read_bytes() == changed


def test_actual_observer_composition_keeps_all_other_files_exact(tmp_path, source_root):
    original, staging = tmp_path/"original", tmp_path/"staging"
    names = list(observation.PINNED_FILES)
    for name in names:
        data = (source_root/name).read_bytes()
        for root in (original, staging):
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    preceding = observation.apply_overlay(original, staging)
    names.append(observation.GENERATED_HEADER)
    before = {name: (staging/name).read_bytes() for name in names}
    receipt = memory.apply_overlay(original, staging, preceding_overlay=preceding)
    assert receipt["input_sha256"][memory.GPU_FILE] == memory.PINNED_SHA256
    for name in names:
        if name != memory.GPU_FILE:
            assert (staging/name).read_bytes() == before[name]


@pytest.mark.parametrize("identity", ["exact", "undeclared", "wrong_digest", "wrong_original"])
def test_declared_preceding_gpu_identity_is_exact(tmp_path, source_bytes, identity):
    original, staging = tmp_path/"original", tmp_path/"staging"
    copied_gpu(original, source_bytes)
    changed = b"// preceding GPU observation hook fixture\n" + source_bytes
    target = copied_gpu(staging, changed)
    preceding = {"original_sha256": {memory.GPU_FILE: memory.PINNED_SHA256},
                 "overlay_sha256": {memory.GPU_FILE: hashlib.sha256(changed).hexdigest()}}
    if identity == "undeclared":
        preceding["overlay_sha256"].clear()
    elif identity == "wrong_digest":
        preceding["overlay_sha256"][memory.GPU_FILE] = memory.PINNED_SHA256
    elif identity == "wrong_original":
        preceding["original_sha256"][memory.GPU_FILE] = "0"*64
    if identity == "exact":
        receipt = memory.apply_overlay(original, staging, preceding_overlay=preceding)
        assert receipt["input_sha256"][memory.GPU_FILE] == hashlib.sha256(changed).hexdigest()
        assert target.read_bytes() == memory.transform_source(changed)
    else:
        with pytest.raises(ValueError, match="STAGED_SOURCE_CHANGED|PRECEDING_OVERLAY_INVALID"):
            memory.apply_overlay(original, staging, preceding_overlay=preceding)
        assert target.read_bytes() == changed


CPP_SCAFFOLD = r'''
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#ifdef CSTP_DOUBLES
using Value = double;
#else
using Value = float;
#endif
struct NetworkConfig {
    int numNReg, maxNumPreSynN, syn_gLength;
    size_t syn_gPitch;
    bool sim_with_NMDA_rise, sim_with_GABAb_rise;
} networkConfigs[1];
struct RuntimeData {
    Value *AMPA_syn_g, *NMDA_d_syn_g, *NMDA_r_syn_g;
    Value *GABAa_syn_g, *GABAb_d_syn_g, *GABAb_r_syn_g;
} runtimeData[1];
std::map<void*, size_t> allocations;
size_t liveBytes=0, allocationCalls=0, freeCalls=0, memsetCalls=0, infoCalls=0;
std::string info;
void require(bool condition, const char* error) {
    if (!condition) throw std::runtime_error(error);
}
int cudaMallocPitch(void** output, size_t* pitch, size_t width, size_t rows) {
    *pitch = ((width+31)/32)*32+16; // Deliberately padded, precision-dependent rows.
    size_t bytes = *pitch * rows;
    *output = std::malloc(bytes);
    require(*output!=NULL, "fake allocation failed");
    std::memset(*output, 0xa5, bytes);
    require(allocations.emplace(*output, bytes).second, "duplicate allocation");
    liveBytes += bytes; ++allocationCalls;
    return 0;
}
int cudaMemset(void* output, int value, size_t bytes) {
    auto found = allocations.find(output);
    require(output!=NULL && found!=allocations.end(), "null or unknown memset");
    require(value==0 && bytes<=found->second, "out-of-allocation memset");
    std::memset(output, value, bytes); ++memsetCalls;
    return 0;
}
int cudaFree(void* output) {
    auto found = allocations.find(output);
    require(output!=NULL && found!=allocations.end(), "null or unmatched free");
    liveBytes -= found->second;
    allocations.erase(found); std::free(output); ++freeCalls;
    return 0;
}
void recordInfo(const char* format, ...) {
    char buffer[1024];
    va_list args; va_start(args, format);
    int length = std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    require(length>0 && size_t(length)<sizeof(buffer), "allocation diagnostic invalid");
    info=buffer; ++infoCalls;
}
#define CUDA_CHECK_ERRORS(call) do { require((call)==0, "CUDA failure"); } while (0)
#define KERNEL_INFO(...) recordInfo(__VA_ARGS__)
'''

CPP_MAIN = r'''
size_t inspectBytes(unsigned char sentinel) {
    size_t tail=0;
    for (const auto& allocation : allocations) {
        const unsigned char* bytes = static_cast<const unsigned char*>(allocation.first);
#ifdef LEGACY
        size_t cleared = networkConfigs[0].syn_gPitch * networkConfigs[0].syn_gLength;
#else
        size_t cleared = allocation.second;
#endif
        for (size_t i=0; i<allocation.second; ++i) {
            require(bytes[i] == (i<cleared ? 0 : sentinel), "wrong initialization extent");
            if (i>=cleared) ++tail;
        }
        require(bytes[allocation.second-1] == (cleared<allocation.second ? sentinel : 0),
                "last allocated row not checked");
    }
    return tail;
}
int main(int argc, char** argv) {
    try {
        require(argc==5, "expected NMDA GABAb rows synapses");
        NetworkConfig& config = networkConfigs[0];
        config.sim_with_NMDA_rise = std::atoi(argv[1])!=0;
        config.sim_with_GABAb_rise = std::atoi(argv[2])!=0;
        config.numNReg=std::atoi(argv[3]); config.maxNumPreSynN=std::atoi(argv[4]);
        Value* invalid = reinterpret_cast<Value*>(size_t(1));
        runtimeData[0] = {invalid,invalid,invalid,invalid,invalid,invalid};
        allocate(true);
#ifdef LEGACY
        const size_t active=6;
        require(infoCalls==0, "unexpected legacy diagnostic");
#else
        const size_t active=4+size_t(config.sim_with_NMDA_rise)+size_t(config.sim_with_GABAb_rise);
        require((runtimeData[0].NMDA_r_syn_g!=NULL)==config.sim_with_NMDA_rise, "NMDA inactive pointer");
        require((runtimeData[0].GABAb_r_syn_g!=NULL)==config.sim_with_GABAb_rise, "GABAb inactive pointer");
        require(infoCalls==1, "allocation diagnostic count");
#endif
        const size_t expectedPitch=((sizeof(Value)*size_t(config.maxNumPreSynN)+31)/32)*32+16;
        require(config.syn_gPitch==expectedPitch, "pitch mismatch");
        require(allocations.size()==active && allocationCalls==active, "wrong active channels");
        require(liveBytes==active*expectedPitch*size_t(config.numNReg), "wrong active bytes");
        require(memsetCalls==active, "wrong initialization calls");
        const size_t tail=inspectBytes(0xa5);
        for (const auto& allocation : allocations)
            std::memset(allocation.first, 0x3c, allocation.second);
        allocate(false);
        require(allocationCalls==active && memsetCalls==2*active, "reset reallocated or skipped");
        require(inspectBytes(0x3c)==tail, "reset extent differs");
#ifndef LEGACY
        require(infoCalls==1, "reset repeated allocation diagnostic");
        std::puts(info.c_str());
#endif
        std::printf("precision_bytes=%zu arrays=%zu live_bytes=%zu legacy_tail_bytes=%zu\n",
                    sizeof(Value),active,liveBytes,tail);
        release();
        require(allocations.empty() && liveBytes==0 && freeCalls==active, "unmatched allocation/free");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"%s\n",error.what()); return 2;
    }
}
'''


def compile_sections(directory, source, precision, *, legacy=False):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ unavailable for host allocation contract")
    allocation, cleanup = host_sections(source)
    text = (CPP_SCAFFOLD + "\nvoid allocate(bool allocateMem) {\n"
            "const int netId=0; RuntimeData* dest=&runtimeData[0];\n"
            + allocation.decode() + "\n}\nvoid release() { const int netId=0;\n"
            + cleanup.decode() + "\n}\n" + CPP_MAIN)
    path, binary = directory/"allocation.cpp", directory/"allocation"
    path.write_text(text)
    defines = ["-DJK_CA3_SNN=1"]
    if precision == "double":
        defines.append("-DCSTP_DOUBLES")
    if legacy:
        defines.append("-DLEGACY")
    subprocess.run([compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", *defines,
                    str(path), "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


@pytest.fixture(scope="module", params=["float", "double"])
def compiled_overlay(request, tmp_path_factory, source_bytes):
    directory = tmp_path_factory.mktemp("ca3_conductance_" + request.param)
    binary = compile_sections(directory, memory.transform_source(source_bytes), request.param)
    return binary, 8 if request.param == "double" else 4


@pytest.mark.parametrize("nmda,gabab", [(0, 0), (1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("rows,synapses", [(11, 5), (3, 7)])
def test_actual_changed_sections_initialize_and_free_active_buffers(
        compiled_overlay, nmda, gabab, rows, synapses):
    binary, precision_bytes = compiled_overlay
    result = subprocess.run([str(binary), str(nmda), str(gabab), str(rows), str(synapses)],
                            check=True, capture_output=True, text=True)
    pitch = ((precision_bytes*synapses+31)//32)*32+16
    arrays = 4+nmda+gabab
    active_bytes = arrays*pitch*rows
    assert f"precision_bytes={precision_bytes} arrays={arrays} live_bytes={active_bytes} legacy_tail_bytes=0" in result.stdout
    assert f"numNReg={rows} maxNumPreSynN={synapses} NMDA_rise={nmda} GABAb_rise={gabab}" in result.stdout
    assert f"active_arrays={arrays} pitch_bytes={pitch} bytes_per_array={pitch*rows} planned_total_bytes={active_bytes}" in result.stdout


def test_actual_legacy_extent_leaves_sentinel_rows_uncleared(tmp_path, source_bytes):
    binary = compile_sections(tmp_path, source_bytes, "double", legacy=True)
    result = subprocess.run([str(binary), "0", "0", "11", "5"],
                            check=True, capture_output=True, text=True)
    assert "precision_bytes=8 arrays=6 live_bytes=5280 legacy_tail_bytes=2880" in result.stdout
    # The opposite aspect ratio exposes an attempted write beyond allocation.
    overflow = subprocess.run([str(binary), "0", "0", "3", "7"],
                              capture_output=True, text=True)
    assert overflow.returncode == 2
    assert "out-of-allocation memset" in overflow.stderr
