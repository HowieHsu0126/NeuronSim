"""Install the official CUDA 11.8 components needed by the native CA3 source.

This explicit compute-node operation installs no driver or system service.
The upstream redistribution manifest supplies every archive identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import requests

from .provenance import freeze_json

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
MANIFEST = BASE + "redistrib_11.8.0.json"
COMPONENTS = ("cuda_nvcc", "cuda_cudart", "libcurand", "cuda_cccl")
PROBE = r"""
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
texture<float, 1, cudaReadModeElementType> legacy_texture;
__global__ void probe(float *x) {
    curandState_t state;
    curand_init(10, 0, 0, &state);
    x[0] = curand_uniform(&state);
}
int main() {
    float *x;
    if (cudaMalloc(&x, sizeof(float)) != cudaSuccess) return 1;
    if (cudaBindTexture(nullptr, legacy_texture, x, sizeof(float)) != cudaSuccess) return 2;
    probe<<<1,1>>>(x);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    float value;
    if (cudaMemcpy(&value, x, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    cudaUnbindTexture(legacy_texture);
    cudaFree(x);
    std::printf("legacy_texture_and_curand=%g\n", value);
    return value > 0.0f && value <= 1.0f ? 0 : 5;
}
"""


def install(output: Path, cuda_arch: str, *, compile_only=False):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("CUDA_REFERENCE_INSTALL_REQUIRES_COMPUTE_ALLOCATION")
    if not compile_only and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("CUDA_REFERENCE_PROBE_REQUIRES_ALLOCATED_GPU")
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="INSTALLING", manifest_url=MANIFEST, components={},
                   slurm_job_id=os.environ["SLURM_JOB_ID"], driver_installed=False,
                   gpu_probe_executed=False)
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(MANIFEST, timeout=(15, 45))
        response.raise_for_status()
        manifest = response.json()
        (output / "redistribution.json").write_bytes(response.content)
        receipt["manifest_sha256"] = hashlib.sha256(response.content).hexdigest()
        for name in COMPONENTS:
            entry = manifest[name]["linux-x86_64"]
            size = int(entry["size"])
            receipt["available_scratch_bytes"] = shutil.disk_usage(output).free
            if receipt["available_scratch_bytes"] < 20 * 1024**3 + 4 * size:
                raise RuntimeError("CUDA_REFERENCE_INSTALL_INSUFFICIENT_ROOT_HEADROOM")
            archive = output / (name + ".tar.xz")
            digest, actual = hashlib.sha256(), 0
            with session.get(BASE + entry["relative_path"], stream=True, timeout=(15, 90)) as source:
                source.raise_for_status()
                with archive.open("xb") as stream:
                    for chunk in source.iter_content(1024 * 1024):
                        actual += len(chunk)
                        if actual > size:
                            raise ValueError("CUDA_ARCHIVE_SIZE_EXCEEDED:" + name)
                        digest.update(chunk)
                        stream.write(chunk)
            if actual != size or digest.hexdigest() != entry["sha256"]:
                raise ValueError("CUDA_ARCHIVE_IDENTITY_MISMATCH:" + name)
            extract = output / "components" / name
            extract.mkdir(parents=True)
            with tarfile.open(archive) as bundle:
                # Only verified official archives are extracted. Python's data
                # filter also rejects escaping paths and special device files.
                bundle.extractall(extract, filter="data")
            roots = list(extract.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise ValueError("CUDA_COMPONENT_LAYOUT_CHANGED:" + name)
            notices = output / "component_notices" / name
            notices.mkdir(parents=True)
            for child in roots[0].iterdir():
                if child.is_dir():
                    shutil.copytree(child, output / child.name, dirs_exist_ok=True, symlinks=True)
                elif child.is_file():
                    shutil.copy2(child, notices / child.name)
            archive.unlink()
            shutil.rmtree(extract)
            receipt["components"][name] = dict(version=manifest[name]["version"], **entry)
        (output / "lib64").symlink_to("lib", target_is_directory=True)
        required = ("bin/nvcc", "bin/ptxas", "bin/fatbinary", "nvvm/bin/cicc",
                    "nvvm/libdevice/libdevice.10.bc", "include/curand_kernel.h",
                    "include/cuda_runtime.h", "lib/libcudart.so")
        if any(not (output / name).is_file() for name in required):
            raise ValueError("CUDA_REFERENCE_COMPONENT_LAYOUT_INCOMPLETE")
        receipt["nvcc_version"] = subprocess.check_output([str(output / "bin/nvcc"), "--version"], text=True)
        probe = output / "legacy_probe.cu"
        probe.write_text(PROBE)
        command = [str(output / "bin/nvcc"), "-ccbin", "/usr/bin/g++", "-arch=sm_" + cuda_arch,
                   str(probe), "-o", str(output / "legacy_probe")]
        receipt["probe_command"] = command
        with (output / "probe_build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt["status"] = "LEGACY_CUDA_AND_CURAND_COMPILE_PASSED_GPU_UNTESTED"
        if not compile_only:
            receipt["probe_output"] = subprocess.check_output([str(output / "legacy_probe")], text=True)
            receipt.update(status="LEGACY_CUDA_AND_CURAND_GPU_PROBE_PASSED", gpu_probe_executed=True)
    except Exception as error:
        receipt.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        session.close()
        freeze_json(output / "toolchain.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-arch", required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(install(args.output.resolve(), args.cuda_arch, compile_only=args.compile_only)))


if __name__ == "__main__":
    main()
