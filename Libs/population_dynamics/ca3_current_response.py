"""Test the pinned native nine-parameter current path against a passive-capacitance oracle."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import traceback

from .provenance import sha256_file

REVISION = "c4955998b950b9c561b3373a71b611aedc705be9"
# Float32 voltage accumulation near -60 mV, with five RK4 substeps per ms.
VOLTAGE_INCREMENT_ATOL = 1e-4
SOURCES = (
    "carlsim/interface/inc/carlsim.h", "carlsim/interface/src/carlsim.cpp",
    "carlsim/kernel/src/snn_manager.cpp", "carlsim/kernel/src/snn_cpu_module.cpp",
    "carlsim/kernel/src/gpu_module/snn_gpu_module.cu",
)


def verify_response(path: Path) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(solver, float(capacitance), phase, neuron, sample)
                for solver in ("euler1", "rk4_5") for capacitance in (100, 200)
                for phase in range(3) for neuron in range(3) for sample in range(3)}
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        solver, capacitance = row["solver"], float(row["capacitance"])
        phase, neuron, sample = (int(row[key]) for key in ("phase", "neuron", "sample"))
        identity = solver, capacitance, phase, neuron, sample
        if identity not in expected or identity in seen:
            raise ValueError("Duplicate or unexpected native response identity")
        seen.add(identity)
        current = float(row["requested_current"])
        declared = (0., 25., 100.)[neuron] if phase == 0 else 50. if phase == 1 else 0.
        voltage, recovery = float(row["voltage"]), float(row["recovery"])
        if current != declared or not math.isfinite(voltage) or recovery != 0.:
            raise ValueError("Invalid native current, voltage or intrinsic recovery state")
        groups[identity[:-1]].append((sample, voltage, current))
    if seen != expected:
        raise ValueError("Native response did not cover both capacitances, solvers and setter phases")
    checks = []
    for (solver, capacitance, phase, neuron), samples in groups.items():
        ordered = sorted(samples)
        for left, right in zip(ordered[:-1], ordered[1:]):
            expected_increment = left[2] / capacitance
            measured = right[1] - left[1]
            error = abs(measured - expected_increment)
            if error > VOLTAGE_INCREMENT_ATOL:
                raise ValueError(f"Native current response differs from I/C: {solver, capacitance, phase, neuron, error}")
            checks.append(dict(solver=solver, capacitance=capacitance, phase=phase, neuron=neuron,
                               sample=left[0], expected_increment=expected_increment,
                               measured_increment=measured, absolute_error=error))
    return dict(native_voltage_samples=len(rows), native_increment_checks=len(checks), checks=checks,
                max_absolute_increment_error=max(row["absolute_error"] for row in checks))


def run(framework: Path, build: Path, cuda: Path, manifest: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="STARTED", created_utc=datetime.now(timezone.utc).isoformat(),
                  source_revision=REVISION, backend="native_CPU", biological_network=False,
                  production_episodes=0, scientific_fidelity_qualified=False,
                  voltage_increment_atol=VOLTAGE_INCREMENT_ATOL)
    try:
        registered = json.loads(manifest.read_text())
        if registered["revision"] != REVISION:
            raise ValueError("Unexpected native framework revision")
        inventory = {row["path"]: row for row in registered["files"]}
        bindings = {}
        for name in SOURCES:
            path = framework / name
            if sha256_file(path) != inventory[name]["sha256"]:
                raise ValueError("Native source differs from pinned manifest: " + name)
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            bindings[name] = inventory[name]["sha256"]
        library = build / "libcarlsim.so.6.0.0"
        if sha256_file(library) != "8eb1da0fb3832aaccc918fd4212096cd0d320372163d968f766829af53a0d078":
            raise ValueError("Expected the already verified native library")
        generated_flags = build / "projects/author_formation/CMakeFiles/ca3_full_net_lognormalI_09_05_23_assem_275.dir/flags.make"
        flags = {}
        for line in generated_flags.read_text().splitlines():
            if " = " in line:
                name, value = line.split(" = ", 1)
                flags[name] = shlex.split(value)
        source = Path(__file__).with_suffix(".cpp")
        shutil.copy2(source, output / source.name)
        binary = output / "native_current_response"
        command = ["/usr/bin/g++", *flags["CXX_DEFINES"], *flags["CXX_INCLUDES"],
                   *flags["CXX_FLAGS"], str(source), "-o", str(binary),
                   str(library), str(build / "libcarlsim-cuda.a"),
                   str(cuda / "lib64/libcudart_static.a"), str(cuda / "lib64/libcurand.so"),
                   "-ldl", "-lrt", "-pthread", f"-Wl,-rpath,{build}:{cuda / 'lib64'}"]
        report.update(compile_command=command, source_bindings=bindings,
                      library_sha256=sha256_file(library), framework_manifest_sha256=sha256_file(manifest),
                      implementation_sha256=sha256_file(Path(__file__)), probe_sha256=sha256_file(source))
        with (output / "compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        (output / "results").mkdir()
        response = output / "response.csv"
        with (output / "native.log").open("w") as log:
            subprocess.run([str(binary), str(response)], cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
        report.update(verify_response(response), status="PASS_NATIVE_NINE_PARAMETER_CURRENT_RESPONSE",
                      response_sha256=sha256_file(response), no_numeric_API_rescaling=True,
                      oracle="With k=a=b=d=0, each 1 ms voltage difference is raw external current / C",
                      physical_interpretation="pA when the model uses pF, mV and ms; paper units must be bound separately",
                      GPU_runtime_response_checked=False, model_physical_unit_qualified=False)
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("framework", "build", "cuda", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
