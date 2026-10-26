"""Opt-in memory overlay for the pinned CA3 conductance buffers.

Only a separate copied GPU source is written. Inactive rise buffers become null;
active buffers are initialized across their allocated neuron rows. The latter
corrects an initial-state defect, so unchanged kernel arithmetic does not imply
trajectory equivalence to the original incompletely initialized allocation.
Native compilation, execution, and scientific qualification remain separate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

GPU_FILE = "carlsim/kernel/src/gpu_module/snn_gpu_module.cu"
PINNED_SHA256 = "4e25494a89427f5479838a17dc48cff5b255954061689146068dd85e16c44b4d"
OVERLAY_ID = "ca3_conductance_memory"
_CHANNELS = ("AMPA", "NMDA_d", "NMDA_r", "GABAa", "GABAb_d", "GABAb_r")
_RISE_FLAGS = {"NMDA_r": "sim_with_NMDA_rise", "GABAb_r": "sim_with_GABAb_rise"}


def _malloc_line(channel: str, precision: str) -> bytes:
    return (f"\t\t\tCUDA_CHECK_ERRORS(cudaMallocPitch((void**)&dest->{channel}_syn_g, "
            f"&networkConfigs[netId].syn_gPitch, sizeof({precision}) * "
            "networkConfigs[netId].syn_gLength, networkConfigs[netId].numNReg));\n").encode()


def _memset_line(channel: str, rows: str) -> bytes:
    trailing = "\t" if channel == "GABAb_r" else ""
    return (f"\tCUDA_CHECK_ERRORS(cudaMemset(dest->{channel}_syn_g, 0, "
            f"networkConfigs[netId].syn_gPitch * networkConfigs[netId].{rows}));"
            f"{trailing}\n").encode()


def _free_line(channel: str) -> bytes:
    trailing = "\t" if channel in ("NMDA_d", "GABAa") else ""
    return (f"\tCUDA_CHECK_ERRORS( cudaFree(runtimeData[netId].{channel}_syn_g) );"
            f"{trailing}\n").encode()


_ALLOCATION_BLOCK = (
    b"#if JK_CA3_SNN\n\tif(allocateMem) {\n"
    b"\t\tnetworkConfigs[netId].syn_gLength = networkConfigs[netId].maxNumPreSynN;\n"
    b"\t\t#ifdef CSTP_DOUBLES\n"
    + b"".join(_malloc_line(channel, "double") for channel in _CHANNELS)
    + b"\t\t#else\n"
    + b"".join(_malloc_line(channel, "float") for channel in _CHANNELS)
    + b"\t\t#endif\n\t}\n"
    b"\tassert(networkConfigs[netId].syn_gPitch > 0 || networkConfigs[netId].maxNumPreSynN == 0);\n"
    + b"".join(_memset_line(channel, "syn_gLength") for channel in _CHANNELS)
    + b"#endif"
)
_FREE_BLOCK = (b"#if JK_CA3_SNN\n"
               + b"".join(_free_line(channel) for channel in _CHANNELS)
               + b"#endif")
_ALLOCATION_INFO = b"""\t\t\tconst int activeSynGArrays = 4 + int(networkConfigs[netId].sim_with_NMDA_rise) + int(networkConfigs[netId].sim_with_GABAb_rise);
\t\t\tconst size_t synGBytes = networkConfigs[netId].syn_gPitch * networkConfigs[netId].numNReg;
\t\t\tKERNEL_INFO("CA3 conductance allocation: numNReg=%d maxNumPreSynN=%d NMDA_rise=%d GABAb_rise=%d active_arrays=%d pitch_bytes=%zu bytes_per_array=%zu planned_total_bytes=%zu",
\t\t\t\tnetworkConfigs[netId].numNReg, networkConfigs[netId].maxNumPreSynN,
\t\t\t\tint(networkConfigs[netId].sim_with_NMDA_rise), int(networkConfigs[netId].sim_with_GABAb_rise),
\t\t\t\tactiveSynGArrays, networkConfigs[netId].syn_gPitch, synGBytes, size_t(activeSynGArrays) * synGBytes);
"""


def transform_source(source: bytes) -> bytes:
    """Replace two exact host blocks; reject changed, missing, or repeated anchors.

    Source identity is checked by ``apply_overlay`` so an explicitly declared
    preceding observation overlay can compose without changing these anchors.
    All bytes outside allocation/initialization and cleanup remain unchanged.
    """
    if not isinstance(source, bytes):
        raise TypeError("CA3_CONDUCTANCE_MEMORY_REQUIRES_BYTES")
    if source.count(_ALLOCATION_BLOCK) != 1 or source.count(_FREE_BLOCK) != 1:
        raise ValueError("CA3_CONDUCTANCE_MEMORY_SOURCE_ANCHOR_CHANGED")
    allocation, cleanup = _ALLOCATION_BLOCK, _FREE_BLOCK
    for channel, flag in _RISE_FLAGS.items():
        for precision in ("double", "float"):
            line = _malloc_line(channel, precision)
            guarded = (f"\t\t\tif (networkConfigs[netId].{flag}) {{\n".encode()
                       + b"\t" + line + b"\t\t\t} else {\n"
                       + f"\t\t\t\tdest->{channel}_syn_g = NULL;\n".encode()
                       + b"\t\t\t}\n")
            allocation = allocation.replace(line, guarded, 1)
        cleanup = cleanup.replace(
            _free_line(channel),
            f"\tif (networkConfigs[netId].{flag})\n".encode() + b"\t" + _free_line(channel), 1)
    for channel in _CHANNELS:
        corrected = _memset_line(channel, "numNReg")
        if channel in _RISE_FLAGS:
            corrected = (f"\tif (networkConfigs[netId].{_RISE_FLAGS[channel]})\n".encode()
                         + b"\t" + corrected)
        allocation = allocation.replace(_memset_line(channel, "syn_gLength"), corrected, 1)
    # Log after the first successful pitched allocation, before later requests
    # can fail. Only one precision branch exists in the compiled program.
    for precision in ("double", "float"):
        first = _malloc_line("AMPA", precision)
        allocation = allocation.replace(first, first + _ALLOCATION_INFO, 1)
    return source.replace(_ALLOCATION_BLOCK, allocation, 1).replace(_FREE_BLOCK, cleanup, 1)


def apply_overlay(original_root: Path, staging_root: Path, *,
                  preceding_overlay: dict | None = None) -> dict:
    """Validate pinned original and exact staged identity before the sole write.

    ``preceding_overlay`` uses the receipt from ``ca3_native_observation``. Its
    GPU output hash, when present, must match the staged GPU file exactly;
    otherwise that file must still match the pinned original. Roots cannot nest,
    and staging symlinks or hard links back to original are rejected.
    """
    original_root = Path(original_root).resolve()
    staging_root = Path(staging_root).resolve()
    if (staging_root == original_root or original_root in staging_root.parents
            or staging_root in original_root.parents):
        raise ValueError("CA3_CONDUCTANCE_MEMORY_REQUIRES_SEPARATE_STAGING")
    source, target = original_root/GPU_FILE, staging_root/GPU_FILE
    if (target.is_symlink() or staging_root not in target.resolve().parents
            or source.samefile(target)):
        raise ValueError("CA3_CONDUCTANCE_MEMORY_STAGING_ALIASES_SOURCE")
    original, staged = source.read_bytes(), target.read_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    input_hash = hashlib.sha256(staged).hexdigest()
    if original_hash != PINNED_SHA256:
        raise ValueError("CA3_CONDUCTANCE_MEMORY_PINNED_SOURCE_CHANGED")
    expected_input = PINNED_SHA256
    if preceding_overlay is not None:
        if not isinstance(preceding_overlay, dict):
            raise ValueError("CA3_CONDUCTANCE_MEMORY_PRECEDING_OVERLAY_INVALID")
        prior_original = preceding_overlay.get("original_sha256")
        prior_output = preceding_overlay.get("overlay_sha256")
        if (not isinstance(prior_original, dict) or not isinstance(prior_output, dict)
                or prior_original.get(GPU_FILE) != PINNED_SHA256):
            raise ValueError("CA3_CONDUCTANCE_MEMORY_PRECEDING_OVERLAY_INVALID")
        expected_input = prior_output.get(GPU_FILE, PINNED_SHA256)
    if input_hash != expected_input:
        raise ValueError("CA3_CONDUCTANCE_MEMORY_STAGED_SOURCE_CHANGED")
    output = transform_source(staged)
    output_hash = hashlib.sha256(output).hexdigest()
    target.write_bytes(output)
    return {
        "schema_version": 1,
        "overlay_id": OVERLAY_ID,
        "original_sha256": {GPU_FILE: original_hash},
        "input_sha256": {GPU_FILE: input_hash},
        "overlay_sha256": {GPU_FILE: output_hash},
        "rise_buffer_allocation": _RISE_FLAGS.copy(),
        "initialization_correction": {
            "previous_extent": "syn_gPitch * syn_gLength",
            "corrected_extent": "syn_gPitch * numNReg",
            "allocation_rows": "numNReg",
            "applies_to": [channel + "_syn_g" for channel in _CHANNELS],
            "previously_uninitialized_rows_are_now_zeroed": True,
        },
        "kernel_arithmetic_unchanged": True,
        "precision_unchanged": True,
        "native_build_verified": False,
        "native_execution_verified": False,
        "native_qualification": False,
    }
