#!/usr/bin/env python3
"""Build only the pinned CA3 shared backend in an actual Merlin CPU trial."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile

from ca3_merlin_remote import digest, download_inputs, memory_evidence, persist, stage_inputs, write_json

GIB = 1024 ** 3
COMPILER_PACKAGES = {"gcc-11", "g++-11", "cpp-11", "gcc-11-base", "libgcc-11-dev",
                     "libstdc++-11-dev", "libasan6", "libtsan0"}
COMPILER_PACKAGE_VERSION = "11.3.0-12"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def extract_cuda(components: dict, cuda: Path) -> None:
    """Merge verified NVIDIA archives into one toolkit, retaining each license."""
    cuda.mkdir(exist_ok=False)
    for component, row in components.items():
        archive = Path(row["path"])
        prefix = archive.name.removesuffix(".tar.xz")
        with tarfile.open(archive, "r|xz") as source:
            for member in source:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != prefix:
                    raise ValueError("CA3_CUDA_ARCHIVE_PATH_INVALID")
                if len(relative.parts) == 1:
                    if not member.isdir():
                        raise ValueError("CA3_CUDA_ARCHIVE_ROOT_INVALID")
                    continue
                tail = Path(*relative.parts[1:])
                if tail.parts[0] not in {"LICENSE", "bin", "include", "lib", "nvvm", "pkg-config"}:
                    raise ValueError("CA3_CUDA_ARCHIVE_COMPONENT_UNEXPECTED:" + str(tail))
                target = cuda / (Path("licenses") / component / tail if tail.parts[0] == "LICENSE" else tail)
                if cuda not in target.resolve().parents:
                    raise ValueError("CA3_CUDA_ARCHIVE_LINK_ESCAPE")
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.isdir():
                    target.mkdir(exist_ok=True)
                elif member.issym():
                    # The five pinned archives use local filename-only library links.
                    link = PurePosixPath(member.linkname)
                    if link.is_absolute() or len(link.parts) != 1 or link.name in {".", ".."}:
                        raise ValueError("CA3_CUDA_LIBRARY_LINK_INVALID")
                    if target.is_symlink() and os.readlink(target) == member.linkname:
                        continue
                    target.symlink_to(member.linkname)
                elif member.isfile():
                    if target.is_symlink():
                        raise ValueError("CA3_CUDA_REGULAR_FILE_LINK_CONFLICT")
                    incoming = source.extractfile(member)
                    if incoming is None:
                        raise ValueError("CA3_CUDA_ARCHIVE_FILE_UNREADABLE")
                    temporary = target.with_name(target.name + ".incoming")
                    with incoming, temporary.open("xb") as stream:
                        shutil.copyfileobj(incoming, stream, 1024 * 1024)
                    if temporary.stat().st_size != member.size:
                        raise ValueError("CA3_CUDA_EXTRACTED_SIZE_DIFFERS")
                    temporary.chmod(member.mode & 0o777)
                    if target.exists():
                        if digest(target) != digest(temporary):
                            raise ValueError("CA3_CUDA_COMPONENT_FILE_CONFLICT:" + str(tail))
                        temporary.unlink()
                    else:
                        temporary.rename(target)
                else:
                    raise ValueError("CA3_CUDA_ARCHIVE_MEMBER_TYPE_INVALID")


def build_commands(backend: dict, source: Path, build: Path, cuda: Path,
                   cc: str = "/usr/bin/gcc", cxx: str = "/usr/bin/g++") -> tuple[list[list[str]], dict]:
    original = backend["build_commands"]
    if len(original) != 2:
        raise ValueError("CA3_ORIGINAL_TWO_BUILD_COMMANDS_REQUIRED")
    configure, compile_command = [list(command) for command in original]
    if (configure[:2] != ["cmake", "-S"] or configure[3] != "-B"
            or compile_command[:2] != ["cmake", "--build"]
            or compile_command[3] != "--target" or compile_command[5:] != ["--parallel", "2"]):
        raise ValueError("CA3_ORIGINAL_BUILD_COMMAND_SHAPE_DIFFERS")
    arch = "-DCARLSIM_CUDA_GENCODE=-gencode arch=compute_89,code=sm_89"
    if configure.count(arch) != 1 or configure.count("-DCUDA_TOOLKIT_ROOT_DIR=" + backend["cuda_root"]) != 1:
        raise ValueError("CA3_ORIGINAL_CUDA_PATH_OR_ARCHITECTURE_DIFFERS")
    bindings = {configure[2]: str(source), configure[4]: str(build), backend["cuda_root"]: str(cuda)}
    configure[2], configure[4] = str(source), str(build)
    for key, old, new in (("-DCMAKE_C_COMPILER=", "/usr/bin/gcc", cc),
                          ("-DCMAKE_CXX_COMPILER=", "/usr/bin/g++", cxx),
                          ("-DCUDA_HOST_COMPILER=", "/usr/bin/g++", cxx)):
        if configure.count(key + old) != 1:
            raise ValueError("CA3_ORIGINAL_COMPILER_ARGUMENT_DIFFERS:" + key)
        configure[configure.index(key + old)] = key + new
        bindings[old] = new
    configure[configure.index(arch)] = arch.replace("compute_89", "compute_80").replace("sm_89", "sm_80")
    for key, path in (("-DCUDA_TOOLKIT_ROOT_DIR=", cuda), ("-DCUDA_NVCC_EXECUTABLE=", cuda / "bin/nvcc")):
        indices = [index for index, value in enumerate(configure) if value.startswith(key)]
        if len(indices) != 1:
            raise ValueError("CA3_ORIGINAL_CUDA_BINDING_MISSING:" + key)
        configure[indices[0]] = key + str(path)
    compile_command[2], compile_command[4] = str(build), "carlsim"
    return [configure, compile_command], bindings


def prepare_source(job: dict, work: Path, output: Path) -> tuple[dict, Path, Path, Path]:
    inputs = job["build_inputs"]
    sys.path.insert(0, str(Path(inputs["repo"]) / "Libs"))
    # These modules become available only after the verified input archive is staged.
    from population_dynamics.ca3_reference import stage
    from population_dynamics.circuit_sources import materialize_build_inputs
    from population_dynamics.ca3_native_observation import apply_overlay as observe
    from population_dynamics.ca3_conductance_memory import apply_overlay as correct_conductance

    original = read(Path(inputs["original_build_report"]))
    backend = original["backend_build_report"]
    for name, key in (("ca3_native_observation.py", "observer_implementation_sha256"),
                      ("ca3_conductance_memory.py", "conductance_memory_implementation_sha256")):
        if digest(Path(inputs["repo"]) / "Libs/population_dynamics" / name) != backend[key]:
            raise ValueError("CA3_OVERLAY_IMPLEMENTATION_CHANGED:" + name)
    source = work / "carlsim"
    source_root = Path(inputs["framework"])
    if any(not (source_root / name).is_file() for name in inputs["deferred_templates"]):
        raise ValueError("CA3_MATERIALIZED_TEMPLATE_INPUT_REQUIRED")
    manifest = stage(source_root, source)
    if manifest["revision"] != backend["framework_revision"]:
        raise ValueError("CA3_FRAMEWORK_REVISION_DIFFERS")
    if any(not (source / name).is_file() for name in inputs["deferred_templates"]):
        raise ValueError("CA3_STAGE_DID_NOT_COPY_BUILD_TEMPLATES")
    templates = materialize_build_inputs(source, inputs["deferred_templates"])
    observation = observe(source_root, source)
    conductance = correct_conductance(source_root, source, preceding_overlay=observation)
    for actual, key in ((observation, "native_observation_overlay"),
                        (conductance, "conductance_memory_overlay")):
        for field in ("original_sha256", "overlay_sha256"):
            if actual[field] != backend[key][field]:
                raise ValueError("CA3_ORIGINAL_OVERLAY_IDENTITY_DIFFERS:" + key)
    write_json(output / "source_preparation.json", dict(
        framework_revision=manifest["revision"], templates=templates,
        native_observation_overlay=observation, conductance_memory_overlay=conductance))
    cuda = work / "cuda-11.8"
    extract_cuda(inputs["components"], cuda)
    samples = Path(inputs["samples"])
    sample_manifest = read(samples / "source.json")
    if sample_manifest["revision"] != backend["cuda_samples_revision"]:
        raise ValueError("CA3_CUDA_SAMPLES_REVISION_DIFFERS")
    for row in sample_manifest["files"]:
        if not row["path"].startswith("Common/"):
            continue
        target = cuda / "samples/common/inc" / row["path"].removeprefix("Common/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(samples / row["path"], target)
    curand = cuda / "lib/libcurand.so.10"
    if digest(curand) != original["backend_libraries"]["libcurand.so.10"]:
        raise ValueError("CA3_ORIGINAL_CURAND_LIBRARY_DIFFERS")
    shutil.copy2(Path(inputs["original_build_report"]), output / "original_build_report.json")
    return original, source, work / "build", cuda


def install_candidate_compiler(job: dict, work: Path, output: Path) -> tuple[str, str]:
    candidate = job["candidate_compiler"]
    if (candidate["profile"] != "debian_bookworm_gcc11"
            or set(candidate["packages"]) != COMPILER_PACKAGES
            or set(candidate["packages"].values()) != set(candidate["input_files"])
            or set(candidate["input_files"]) & set(job["input_files"])):
        raise ValueError("CA3_CANDIDATE_COMPILER_INPUTS_INVALID")
    compiler_work = work / "compiler_inputs"
    compiler_work.mkdir(exist_ok=False)
    archive = download_inputs(candidate, compiler_work)
    stage_inputs(archive, candidate["input_files"])
    package_metadata = {}
    for name, path in candidate["packages"].items():
        values = {key: command_output(["dpkg-deb", "--field", path, key]).strip()
                  for key in ("Package", "Version", "Architecture")}
        if values != dict(Package=name, Version=COMPILER_PACKAGE_VERSION, Architecture="amd64"):
            raise ValueError("CA3_CANDIDATE_COMPILER_PACKAGE_DIFFERS:" + name)
        package_metadata[name] = values
    command = (["sudo", "-n"] if os.geteuid() else []) + [
        "dpkg", "--install", *[candidate["packages"][name] for name in sorted(COMPILER_PACKAGES)]]
    record = dict(profile=candidate["profile"], package_metadata=package_metadata,
                  input_files=candidate["input_files"], install_command=command,
                  original_compiler_equivalence=False, candidate_qualified=False)
    write_json(output / "candidate_compiler_preparation.json", record)
    with (output / "candidate_compiler_install.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    installed = command_output(["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\t${Status}\n",
                                *sorted(COMPILER_PACKAGES)])
    rows = [line.split("\t") for line in installed.splitlines()]
    if (len(rows) != len(COMPILER_PACKAGES) or {row[0] for row in rows} != COMPILER_PACKAGES
            or any(row[1:] != [COMPILER_PACKAGE_VERSION, "amd64", "install ok installed"] for row in rows)):
        raise ValueError("CA3_INSTALLED_CANDIDATE_COMPILER_PACKAGES_DIFFER")
    cc, cxx = "/usr/bin/gcc-11", "/usr/bin/g++-11"
    for compiler in (cc, cxx):
        if command_output([compiler, "-dumpfullversion"]).strip() != "11.3.0":
            raise ValueError("CA3_CANDIDATE_COMPILER_VERSION_DIFFERS")
    record.update(installed_packages=rows, compiler_paths=dict(cc=cc, cxx=cxx), install_exit_code=0)
    write_json(output / "candidate_compiler_installed.json", record)
    return cc, cxx


def execute_build(job: dict, work: Path, output: Path, receipt: dict) -> None:
    inputs = job["build_inputs"]
    recorded = read(Path(inputs["compiler_locations"]))
    compiler_candidate = job.get("candidate_compiler")
    cc, cxx = (install_candidate_compiler(job, work, output) if compiler_candidate is not None
               else ("/usr/bin/gcc", "/usr/bin/g++"))
    versions = {"gcc": command_output([cc, "--version"]),
                "gxx": command_output([cxx, "--version"]),
                "cmake": command_output(["cmake", "--version"]), "python": sys.version}
    write_json(output / "compiler_preflight.json", dict(actual=versions, recorded=recorded,
        candidate_profile=compiler_candidate["profile"] if compiler_candidate else None,
        compiler_paths=dict(cc=cc, cxx=cxx), original_compiler_equivalence=False,
        comparison_scope="Explicit candidate package binding" if compiler_candidate else "Original GCC driver version lines"))
    if compiler_candidate is None:
        for name, field in (("gcc", "gcc_version"), ("gxx", "gxx_version")):
            if versions[name].splitlines()[0] != recorded[field].splitlines()[0]:
                raise ValueError("CA3_HOST_COMPILER_VERSION_DIFFERS:" + name)
    original, source, build, cuda = prepare_source(job, work, output)
    backend = original["backend_build_report"]
    versions["nvcc"] = command_output([str(cuda / "bin/nvcc"), "--version"])
    if versions["nvcc"] != backend["nvcc_version"]:
        raise ValueError("CA3_ORIGINAL_NVCC_VERSION_DIFFERS")
    versions["cc1"] = command_output([cc, "-print-prog-name=cc1"]).strip()
    versions["cc1plus"] = command_output([cxx, "-print-prog-name=cc1plus"]).strip()
    commands, bindings = build_commands(backend, source, build, cuda, cc, cxx)
    receipt.update(status="BUILDING_CA3_SM80_BACKEND", build_commands=commands, path_bindings=bindings)
    write_json(output / "merlin_execution.json", receipt)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for index, command in enumerate(commands):
        with (output / f"build_{index}.log").open("x") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    library = build / "libcarlsim.so.6.0.0"
    if not library.is_file() or library.is_symlink():
        raise ValueError("CA3_SHARED_BACKEND_BUILD_OUTPUT_MISSING")
    (output / "lib").mkdir()
    candidate = output / "lib" / library.name
    shutil.copy2(library, candidate)
    dynamic = command_output(["readelf", "-d", str(candidate)])
    comments = command_output(["readelf", "-p", ".comment", str(candidate)])
    cubins = command_output([str(cuda / "bin/cuobjdump"), "--list-elf", str(candidate)])
    (output / "library_dynamic.txt").write_text(dynamic)
    (output / "library_compiler_comments.txt").write_text(comments)
    (output / "library_cuda_objects.txt").write_text(cubins)
    architectures = sorted(set(re.findall(r"sm_[0-9]+", cubins)))
    if architectures != ["sm_80"]:
        raise ValueError("CA3_CANDIDATE_SM80_OBJECTS_REQUIRED")
    shutil.copy2(build / "CMakeCache.txt", output / "CMakeCache.txt")
    result = write_build_report(original, versions, commands, bindings, output,
                                compiler_candidate["profile"] if compiler_candidate else None)
    receipt.update(status=result["status"], candidate_library=result["library"], build_exit_code=0)


def write_build_report(original: dict, versions: dict, commands: list, bindings: dict,
                       output: Path, compiler_profile: str | None) -> dict:
    backend = original["backend_build_report"]
    candidate = output / "lib/libcarlsim.so.6.0.0"
    dynamic = (output / "library_dynamic.txt").read_text()
    architectures = sorted(set(re.findall(r"sm_[0-9]+", (output / "library_cuda_objects.txt").read_text())))
    if architectures != ["sm_80"] or not candidate.is_file() or candidate.is_symlink():
        raise ValueError("CA3_CANDIDATE_SM80_LIBRARY_REQUIRED")
    result = dict(status="BUILT_UNQUALIFIED_CA3_SM80_BACKEND", target="carlsim",
        library=dict(path="lib/" + candidate.name, bytes=candidate.stat().st_size, sha256=digest(candidate)),
        original_library_sha256=original["backend_libraries"][candidate.name],
        original_curand_sha256=original["backend_libraries"]["libcurand.so.10"],
        build_commands=commands, original_build_commands=backend["build_commands"], path_bindings=bindings,
        environment_versions=versions, cuda_architectures=architectures,
        candidate_compiler_profile=compiler_profile,
        original_compiler_equivalence=False,
        needed_libraries=re.findall(r"\(NEEDED\).*\[([^]]+)\]", dynamic),
        framework_revision=backend["framework_revision"], cuda_samples_revision=backend["cuda_samples_revision"],
        original_sm89_binaries_preserved=True, rebuilt_host_program=False,
        candidate_qualified=False, scientific_fidelity_qualified=False, simulations_run=0, production_episodes=0)
    write_json(output / "backend_build_report.json", result)
    return result


def recover_build_report(job: dict, work: Path, output: Path, receipt: dict) -> None:
    """Recover the report-only failure from its authenticated completed library."""
    recovery = job["completed_build"]
    original_receipt = recovery["source_receipt"]
    if (original_receipt.get("persisted") is not True or original_receipt.get("exit_code") != 1
            or original_receipt.get("error_type") != "TypeError"
            or original_receipt.get("error") != "'PosixPath' object is not subscriptable"
            or set(recovery["input_files"]) & set(job["input_files"])):
        raise ValueError("CA3_COMPLETED_BUILD_REPORT_FAILURE_REQUIRED")
    staging = work / "completed_build_inputs"
    staging.mkdir(exist_ok=False)
    archive = download_inputs(recovery, staging)
    stage_inputs(archive, recovery["input_files"])
    previous = Path(original_receipt["output"])
    terminal = read(previous / "merlin_terminal.json")
    for key in ("merlin_job_run_id", "merlin_trial_id", "error", "exit_code", "build_commands"):
        if terminal[key] != original_receipt[key]:
            raise ValueError("CA3_COMPLETED_BUILD_TERMINAL_DIFFERS:" + key)
    if "[100%] Built target carlsim" not in (previous / "build_1.log").read_text():
        raise ValueError("CA3_COMPLETED_CARLSIM_TARGET_REQUIRED")
    original = read(Path(job["build_inputs"]["original_build_report"]))
    if original != read(previous / "original_build_report.json"):
        raise ValueError("CA3_COMPLETED_BUILD_ORIGINAL_PROVENANCE_DIFFERS")
    backend = original["backend_build_report"]
    preflight = read(previous / "compiler_preflight.json")
    if preflight["candidate_profile"] != "debian_bookworm_gcc11":
        raise ValueError("CA3_COMPLETED_BUILD_COMPILER_PROFILE_DIFFERS")
    old_work = previous.parent
    commands, bindings = build_commands(backend, old_work / "carlsim", old_work / "build",
        old_work / "cuda-11.8", preflight["compiler_paths"]["cc"], preflight["compiler_paths"]["cxx"])
    if commands != terminal["build_commands"] or bindings != terminal["path_bindings"]:
        raise ValueError("CA3_COMPLETED_BUILD_COMMANDS_DIFFER")
    for name in ("original_build_report.json", "source_preparation.json", "compiler_preflight.json",
                 "candidate_compiler_installed.json", "candidate_compiler_preparation.json",
                 "candidate_compiler_install.log", "build_0.log", "build_1.log", "CMakeCache.txt",
                 "library_dynamic.txt", "library_compiler_comments.txt", "library_cuda_objects.txt"):
        shutil.copy2(previous / name, output / name)
    (output / "lib").mkdir()
    candidate = output / "lib/libcarlsim.so.6.0.0"
    shutil.copy2(previous / "lib/libcarlsim.so.6.0.0", candidate)
    if command_output(["readelf", "-d", str(candidate)]) != (output / "library_dynamic.txt").read_text():
        raise ValueError("CA3_RECOVERED_LIBRARY_DYNAMIC_METADATA_DIFFERS")
    cuda = work / "cuda-11.8"
    extract_cuda(job["build_inputs"]["components"], cuda)
    if command_output([str(cuda / "bin/cuobjdump"), "--list-elf", str(candidate)]) != (output / "library_cuda_objects.txt").read_text():
        raise ValueError("CA3_RECOVERED_LIBRARY_CUDA_OBJECTS_DIFFER")
    nvcc = command_output([str(cuda / "bin/nvcc"), "--version"])
    if nvcc != backend["nvcc_version"]:
        raise ValueError("CA3_RECOVERY_NVCC_VERSION_DIFFERS")
    versions = dict(preflight["actual"], nvcc=nvcc)
    result = write_build_report(original, versions, commands, bindings, output, preflight["candidate_profile"])
    write_json(output / "original_failed_receipt.json", original_receipt)
    receipt.update(status=result["status"], candidate_library=result["library"],
        scope="completed_ca3_build_report_recovery", builds_run=0, recovered_build_exit_code=0,
        original_job_exit_code=1, original_build_run=original_receipt["merlin_job_run_id"],
        original_build_trial=original_receipt["merlin_trial_id"],
        original_result_sha256=original_receipt["result_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    run, trial, checkpoint = (os.environ.get(name) for name in
                             ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS"))
    if not run or not trial or not checkpoint or not re.fullmatch(r"[A-Za-z0-9_.-]+", trial):
        raise ValueError("CA3_ACTUAL_MERLIN_BUILD_RUN_TRIAL_AND_CHECKPOINT_REQUIRED")
    job = read(args.job_json)
    work = args.job_json.resolve().parent
    output = work / "output"
    output.mkdir(exist_ok=False)
    target = checkpoint.rstrip("/") + "/ca3/backend_build/" + trial
    receipt = dict(status="PREPARING_CA3_BACKEND_BUILD", execution_backend="merlin",
        merlin_job_run_id=run, merlin_trial_id=trial, hostname=socket.gethostname(), output=str(output),
        started_utc=datetime.now(timezone.utc).isoformat(), scope="library_only_ca3_sm80_cpu_build",
        candidate_qualified=False, scientific_fidelity_qualified=False, simulations_run=0,
        production_episodes=0, persisted=False, package_identity=job["package_identity"])
    ready, exit_code = False, 1

    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        ready = True
        shape = job["resource_shape"]
        if (shape["gpu"] != 0 or shape["gpu_type"] != "CPU_ONLY" or shape["cpu"] < 2
                or int(os.environ.get("ARNOLD_APPLIED_GPU_NUM", "0")) != 0):
            raise ValueError("CA3_CPU_ONLY_BACKEND_BUILD_REQUIRED")
        receipt["memory_evidence"] = memory_evidence(shape, required_available_gib=4)
        filesystem = command_output(["stat", "-f", "-c", "%T", str(work)]).strip()
        if filesystem not in {"ext2/ext3", "ext4", "xfs", "btrfs", "tmpfs", "overlayfs", "zfs"}:
            raise ValueError("CA3_LOCAL_BUILD_SCRATCH_REQUIRED")
        receipt.update(scratch_filesystem=filesystem, scratch_free_bytes=shutil.disk_usage(work).free,
                       cpu_affinity_count=len(os.sched_getaffinity(0)), resource_shape=shape)
        write_json(output / "merlin_resource_preflight.json", receipt)
        archive = download_inputs(job, work)
        stage_inputs(archive, job["input_files"])
        if job.get("completed_build") is not None:
            recover_build_report(job, work, output, receipt)
        else:
            execute_build(job, work, output, receipt)
        exit_code = 0
    except BaseException as error:
        receipt.update(status="FAILED_CA3_SM80_BACKEND_BUILD", error_type=type(error).__name__,
                       error=str(error), build_exit_code=getattr(error, "returncode", None))
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        receipt.update(exit_code=exit_code, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "merlin_terminal.json", receipt)
        if ready:
            try:
                persist(output, work, target, receipt, chunk_bytes=256 * 1024)
            except BaseException as error:
                receipt.update(persisted=False, persistence_error_type=type(error).__name__,
                               persistence_error=str(error))
                write_json(output / "merlin_terminal.json", receipt)
                exit_code = exit_code or 1
        print(json.dumps(dict(status=receipt["status"], merlin_job_run_id=run, merlin_trial_id=trial,
                              persisted=receipt["persisted"], exit_code=exit_code, production_episodes=0)), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
