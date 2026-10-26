"""Build pinned CA3 formation/retrieval projects; execute formation on a GPU.

The default prefix stops after4.45s. Explicit full-formation mode runs the
unchanged70s author program with all20 saved networks. These saved networks
are not full-state restarts. Retrieval is explicitly build-only and retains its
original normalized65p input binding; native retrieval requires a separate,
qualified normalized-input execution path.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import time

from .circuit_sources import fetch_source, materialize_build_inputs
from .provenance import freeze_json, sha256_file
from .carlsim_records import network_header
from .ca3_native_observation import apply_overlay
from .ca3_coupled_extension import extend_source
from .config import load_config

PROJECT = "formation/project/ca3_full_net_lognormalI_09_05_23_assem_275"
MAIN = "src/main_ca3_full_net_lognormalI_09_05_23_assem_275.cpp"
PATTERNS = "formation/build/ca3_full_net_lognormalI_09_05_23_assem_275/trainPat.csv"
TARGET = "ca3_full_net_lognormalI_09_05_23_assem_275"
PREFIX_DURATION_MS = 4450
FORMATION_DURATION_MS = 70000
FORMATION_SCOPES = ("prefix", "full-formation")
SCOPES = (*FORMATION_SCOPES, "retrieval")
RETRIEVAL_PROJECT = "retrieval/project/ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275"
RETRIEVAL_MAIN = "src/main_ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275.cpp"
RETRIEVAL_PATTERNS = "retrieval/build/ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275/trainPat.csv"
RETRIEVAL_TARGET = "ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275"
RETRIEVAL_DURATION_MS = 1001


@dataclass(frozen=True)
class SourceProject:
    project: str
    main: str
    patterns: str
    target: str
    local_name: str


def select_project(scope: str) -> SourceProject:
    if scope not in SCOPES:
        raise ValueError("CA3_UNSUPPORTED_EXECUTION_SCOPE")
    if scope == "retrieval":
        return SourceProject(RETRIEVAL_PROJECT, RETRIEVAL_MAIN, RETRIEVAL_PATTERNS,
                             RETRIEVAL_TARGET, "author_retrieval")
    return SourceProject(PROJECT, MAIN, PATTERNS, TARGET, "author_formation")


def prepare_retrieval_program(path: Path, config: Path, *, local_input: bool = False,
                              population_multiplier: int = 1, cue_omission: bool = False) -> dict:
    """Inspect the source contract; optionally replace only its input-path literal."""
    if population_multiplier not in (1, 2) or type(population_multiplier) is not int:
        raise ValueError("CA3_EXTENSION_MULTIPLIER_MUST_BE_ONE_OR_TWO")
    if population_multiplier == 2 and not local_input:
        raise ValueError("CA3_COUPLED_RETRIEVAL_REQUIRES_LOCAL_INPUT")
    if type(cue_omission) is not bool or (cue_omission and not local_input):
        raise ValueError("CA3_OMISSION_REQUIRES_EXPLICIT_LOCAL_INPUT")
    input_name = f"network65p_assem{275 * population_multiplier}_divnorm.dat"
    main, header = active(path.read_text()), active(config.read_text())
    bindings = re.findall(r'fopen\(\s*"([^"\n]*network65p_assem275_divnorm\.dat)"\s*,\s*"rb"\s*\)', main)
    if len(bindings) != 1 or not re.search(r'sim\.loadSimulation\(\s*fId\s*\)', main):
        raise ValueError("CA3_RETRIEVAL_NORMALIZED65P_BINDING_CHANGED")
    if (not re.search(r'\bint\s+randSeed\s*=\s*10\s*;', main)
            or not re.search(r'\bint\s+myrandom\s*=\s*3\s*;', main)
            or not re.search(r'std::srand\(\s*myrandom\s*\)', main)):
        raise ValueError("CA3_RETRIEVAL_SOURCE_RANDOM_STREAM_CHANGED")
    if not re.search(r'sim\.setESTDP\(\s*CA3_Pyramidal\s*,\s*CA3_Pyramidal\s*,\s*true\b', header):
        raise ValueError("CA3_RETRIEVAL_SOURCE_ESTDP_CHANGED")
    effective_binding = bindings[0]
    if local_input:
        original = path.read_bytes()
        old_literal = ('"' + bindings[0] + '"').encode('utf-8')
        new_literal = ('"' + input_name + '"').encode("utf-8")
        if original.count(old_literal) != 1:
            raise ValueError("CA3_RETRIEVAL_PATH_LITERAL_NOT_UNIQUE")
        path.write_bytes(original.replace(old_literal, new_literal, 1))
        effective_binding = input_name
    result = dict(required_checkpoint=input_name,
                original_source_input_binding=bindings[0],
                effective_input_binding=effective_binding,
                source_path_binding_only=local_input and population_multiplier == 1,
                fresh_coupled_checkpoint_required=population_multiplier == 2,
                raw_network65p_substitution_allowed=False,
                normalization_source_revision="8c6c185994f8d54b7a3e027badcfaa7d15f232d1",
                normalized_input_qualified=False, native_retrieval_executed=False,
                 source_carlsim_seed=10, source_host_srand_seed=3, source_estdp_enabled=True)
    if cue_omission:
        # Frozen historical source bundles do not contain this opt-in helper.
        from .ca3_cue_omission import contract, omission_program

        path.write_bytes(omission_program(path.read_bytes(), population_multiplier=population_multiplier))
        result.update(source_path_binding_only=False, cue_omission=contract(population_multiplier))
    return result



def formation_program(original: str, scope: str) -> str:
    """Apply only the declared early exit; full formation preserves source bytes."""
    if scope not in FORMATION_SCOPES:
        raise ValueError("CA3_UNSUPPORTED_EXECUTION_SCOPE")
    needle = 'sim.saveSimulation("network5p.dat", true);'
    if original.count(needle) != 1:
        raise ValueError("CA3_FIRST_CHECKPOINT_BOUNDARY_CHANGED")
    if scope == "full-formation":
        return original
    return original.replace(needle, needle + '\n                return 0; // declared source-prefix stop; destructors flush native monitors\n')


def prepare_formation_program(path: Path, scope: str) -> None:
    """Keep full-source file bytes, including CRLF; retain legacy prefix overlay."""
    transformed = formation_program(path.read_text(), scope)
    if scope == 'prefix':
        path.write_text(transformed)


def verify_checkpoints(run: Path, scope: str, *, expected_units: int) -> list[dict]:
    if scope not in FORMATION_SCOPES:
        raise ValueError("CA3_UNSUPPORTED_EXECUTION_SCOPE")
    checkpoints = range(5, 101, 5) if scope == "full-formation" else (5,)
    records = []
    groups = None
    for cycle in checkpoints:
        checkpoint = run / f"network{cycle}p.dat"
        header = network_header(checkpoint)
        if header["units"] != expected_units or (groups is not None and header["groups"] != groups):
            raise ValueError("CA3_CHECKPOINT_UNIT_IDENTITY_DIFFERS")
        groups = header["groups"]
        expected_ms = 1000 + cycle * 3 * (20 + 210)
        if header["simulation_seconds"] != expected_ms / 1000:
            # Header stores float32 seconds; allow only its storage rounding.
            expected_seconds = struct.unpack("<f", struct.pack("<f", expected_ms / 1000))[0]
            if header["simulation_seconds"] != expected_seconds:
                raise ValueError("CA3_CHECKPOINT_SIMULATION_TIME_DIFFERS")
        records.append(dict(path=checkpoint.name, bytes=checkpoint.stat().st_size,
                            simulation_ms=expected_ms, units=header["units"]))
    return records


def active(text):
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", text, flags=re.S))


def stage(source: Path, target: Path):
    manifest = json.loads((source / "source.json").read_text())
    target.mkdir(parents=True, exist_ok=True)
    for row in [*manifest["files"], {"path":"source.json"}]:
        old, new = source / row["path"], target / row["path"]
        if "sha256" in row and sha256_file(old) != row["sha256"]:
            raise ValueError("CA3_SOURCE_CHANGED:" + row["path"])
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, new)
    # Reuse previously materialized, pinned CMake inputs instead of depending on
    # an external request during every native build. The materializer below
    # compares their Git blob identities with this same source manifest.
    for row in manifest["deferred_blobs"]:
        if row["path"].endswith(".in") and (source / row["path"]).is_file():
            new = target / row["path"]
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / row["path"], new)
    return manifest


def prepare_coupled_project(project: Path, local_project: Path, patterns: Path,
                            selected: SourceProject, *, multiplier: int, role: str) -> tuple[dict, dict, Path]:
    """Transform a copied project; retain the acquired author tree as its reference."""
    header_name = "generateCONFIGStateSTP.h"
    original_header = (project / header_name).read_bytes()
    original_main = (project / selected.main).read_bytes()
    original_pattern = patterns.read_bytes()
    header, main, pattern, metadata = extend_source(
        original_header.decode("utf-8"), original_main.decode("utf-8"),
        original_pattern.decode("utf-8"), multiplier=multiplier, role=role)
    expected = {header_name: header.encode("utf-8"), selected.main: main.encode("utf-8")}
    for relative, content in expected.items():
        (local_project / relative).write_bytes(content)
    effective_pattern = local_project / "trainPat.csv"
    effective_pattern.write_bytes(pattern.encode("utf-8"))
    metadata.update(source_pattern_sha256=sha256_file(patterns),
                    derived_pattern_sha256=sha256_file(effective_pattern))
    return metadata, expected, effective_pattern


def execute(source_root: Path, workdir: Path, cuda_arch: str, native_run_dir: Path | None = None,
            cuda_root: Path | None = None, *, build_only=False, boost_include: Path | None = None,
            scope: str = "prefix", local_retrieval_input: bool = False,
             native_observation: bool = False, coupled_extension_config: Path | None = None,
            conductance_memory: bool = False, cue_omission: bool = False,
            neuron_substeps: int | None = None, source_realization_config: Path | None = None):
    selected = select_project(scope)
    realization = None
    if source_realization_config is not None:
        # Optional for preserved source bundles; historical builds stay byte-identical.
        from .ca3_realization import load_realization

        realization = load_realization(source_realization_config)
        expected_phase = 'retrieval' if scope == 'retrieval' else 'formation'
        if (scope not in SCOPES or not build_only or (scope == 'retrieval' and not local_retrieval_input)
                or not conductance_memory or coupled_extension_config is None or cue_omission
                or neuron_substeps is not None or realization.phase != expected_phase):
            raise ValueError('CA3_REALIZATION_REQUIRES_CORRECTED_COUPLED_SOURCE_CLOCK_BUILD')
    if neuron_substeps is not None:
        from .ca3_neuron_substeps import validate_count

        validate_count(neuron_substeps)
        if (scope != 'retrieval' or not build_only or not local_retrieval_input
                or not native_observation or not conductance_memory
                or coupled_extension_config is None or cue_omission):
            raise ValueError('CA3_REFINEMENT_REQUIRES_OBSERVED_CORRECTED_COUPLED_RETRIEVAL')
    if type(cue_omission) is not bool or (cue_omission and (
            scope != 'retrieval' or not local_retrieval_input or not native_observation)):
        raise ValueError('CA3_OMISSION_REQUIRES_OBSERVED_LOCAL_RETRIEVAL_BUILD')
    population_multiplier = 1
    extension_options = None
    if coupled_extension_config is not None:
        extension_options = load_config(coupled_extension_config, validate_training_seeds=False).get("ca3_coupled_extension")
        if (not isinstance(extension_options, dict)
                or set(extension_options) != {"population_multiplier"}
                or type(extension_options["population_multiplier"]) is not int
                or extension_options["population_multiplier"] != 2):
            raise ValueError("CA3_COUPLED_EXTENSION_CONFIG_REQUIRED")
        population_multiplier = extension_options["population_multiplier"]
    if population_multiplier == 2 and scope == "retrieval" and not local_retrieval_input:
        raise ValueError("CA3_COUPLED_RETRIEVAL_REQUIRES_LOCAL_INPUT")
    if local_retrieval_input and scope != "retrieval":
        raise ValueError("CA3_LOCAL_INPUT_BINDING_IS_RETRIEVAL_ONLY")
    if scope == "retrieval" and not build_only:
        raise ValueError("CA3_RETRIEVAL_BUILD_ONLY_REQUIRES_QUALIFIED_NORMALIZED65P_RUNNER")
    if native_observation and not build_only:
        raise ValueError("CA3_OBSERVATION_BUILD_REQUIRES_EXPLICIT_NATIVE_RUNNER")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("CA3_BUILD_REQUIRES_COMPUTE_ALLOCATION")
    if not build_only and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("CA3_NATIVE_PREFIX_REQUIRES_ALLOCATED_GPU")
    if workdir.exists():
        raise FileExistsError("CA3_NATIVE_WORKDIR_ALREADY_EXISTS")
    workdir.mkdir(parents=True)
    source_scope, duration = {
        "prefix": ("native_source_prefix", PREFIX_DURATION_MS),
        "full-formation": ("native_source_formation", FORMATION_DURATION_MS),
        "retrieval": ("native_source_retrieval", RETRIEVAL_DURATION_MS),
    }[scope]
    if population_multiplier == 2:
        source_scope = source_scope.replace("native_source_", "coupled_extension_")
    report = dict(status="PREPARING", scope=source_scope,
                  expected_duration_ms=duration, build_target=selected.target,
                  source_project=selected.project, source_main=selected.main,
                  source_patterns=selected.patterns,
                  source_full_formation_ms=70000, complete_production_episodes=0,
                  full_scale_qualified=False, scientific_fidelity_qualified=False,
                  slurm_job_id=os.environ["SLURM_JOB_ID"])
    if scope == "prefix":
        report["expected_prefix_ms"] = PREFIX_DURATION_MS
    report["complete_formation_retrieval"] = False
    report["full_state_restart_available"] = False
    if extension_options is not None:
        report["coupled_extension_config"] = dict(path=str(coupled_extension_config),
            sha256=sha256_file(coupled_extension_config), resolved=extension_options)
    report_path = workdir / "report.json"
    if cue_omission:
        report['retrieval_condition'] = 'cue_omission'
    try:
        framework = workdir / "carlsim"
        framework_manifest = stage(source_root / "carlsim_ca3", framework)
        author = workdir / "author"
        author_manifest = stage(source_root / "ca3_formation_retrieval", author)
        extra = materialize_build_inputs(framework, ["cmake/carlsim-config.cmake.in", "carlsim/interface/inc/carlsim_conf.h.in"])
        freeze_json(workdir / "materialized_build_inputs.json", extra)
        project = author / selected.project
        header = active((project / "generateCONFIGStateSTP.h").read_text())
        main = active((project / selected.main).read_text())
        cells = re.findall(r'createGroup\("([^\"]+)",\s*(\d+)', header)
        if len(cells) != 8 or "setIntegrationMethod(RUNGE_KUTTA4, 5)" not in main:
            raise ValueError("CA3_NATIVE_MODEL_OR_INTEGRATOR_CHANGED")
        if set(re.findall(r"RangeDelay\((\d+)\)", header)) != {"1"} or "setConductances(" in header + main:
            raise ValueError("CA3_UNSUPPORTED_SOURCE_DELAY_OR_CONDUCTANCE")
        if len(re.findall(r"sim\.setSTP\([^;]+?\btrue\b", header, re.S)) != len(re.findall(r"sim\.connect\(", header)):
            raise ValueError("CA3_CONNECTION_SPECIFIC_STP_INCOMPLETE")
        text = (author / selected.patterns).read_text()
        if re.fullmatch(r"\s*\d+(?:\s+\d+)*\s*", text) is None:
            raise ValueError("CA3_PATTERN_INPUT_IS_NOT_WHITESPACE_INTEGER_FORMAT")
        pattern = list(map(int, text.split()))
        if not pattern or len(pattern) % 20 or min(pattern) < 0 or max(pattern) > 274:
            raise ValueError("CA3_PATTERN_INPUT_OUTSIDE_SOURCE_SUPPORT")
        local_project = framework / "projects" / selected.local_name
        shutil.copytree(project, local_project)
        path = local_project / selected.main
        effective_pattern = author / selected.patterns
        extension_expected = {}
        if population_multiplier == 2:
            extension, extension_expected, effective_pattern = prepare_coupled_project(
                project, local_project, effective_pattern, selected, multiplier=2,
                role="retrieval" if scope == "retrieval" else "formation")
            report["coupled_extension"] = extension
            cells = list(extension["derived_populations"].items())
            pattern = list(map(int, effective_pattern.read_text().split()))
        if scope == "retrieval":
            report["retrieval_input_contract"] = prepare_retrieval_program(
                path, local_project / "generateCONFIGStateSTP.h", local_input=local_retrieval_input,
                population_multiplier=population_multiplier, cue_omission=cue_omission)
        else:
            prepare_formation_program(path, scope)
        if neuron_substeps is not None:
            from .ca3_neuron_substeps import apply_overlay as apply_substep_overlay

            report['numerical_refinement'] = apply_substep_overlay(
                path, local_project/'generateCONFIGStateSTP.h',
                local_project/'generateSETUPStateSTP.h', framework, neuron_substeps)
            report['numerical_refinement_implementation_sha256'] = sha256_file(
                Path(__file__).with_name('ca3_neuron_substeps.py'))
        file_checks = []
        for name in (selected.main, "generateCONFIGStateSTP.h", "generateSETUPStateSTP.h"):
            original_sha, copied_sha = sha256_file(project / name), sha256_file(local_project / name)
            preserved = original_sha == copied_sha
            path_only = False
            extension_verified = False
            omission_verified = False
            if population_multiplier == 2:
                expected = extension_expected.get(name, (project / name).read_bytes())
                if name == selected.main and scope == "prefix":
                    expected = formation_program(expected.decode("utf-8").replace("\r\n", "\n"), scope).encode("utf-8")
                if name == selected.main and scope == "retrieval":
                    binding = report["retrieval_input_contract"]
                    old = ('"' + binding["original_source_input_binding"] + '"').encode("utf-8")
                    new = ('"' + binding["effective_input_binding"] + '"').encode("utf-8")
                    expected = expected.replace(old, new, 1)
                    if cue_omission:
                        from .ca3_cue_omission import omission_program

                        expected = omission_program(expected, population_multiplier=population_multiplier)
                        omission_verified = (local_project / name).read_bytes() == expected
                    if neuron_substeps is not None:
                        from .ca3_neuron_substeps import program as substep_program

                        expected = substep_program(expected, neuron_substeps)
                extension_verified = (local_project / name).read_bytes() == expected
                if not extension_verified:
                    raise ValueError("CA3_UNDECLARED_COUPLED_SOURCE_CHANGE:" + name)
            elif scope == "prefix" and name == selected.main:
                expected = formation_program((project / name).read_text(), scope).encode("utf-8")
                if (local_project / name).read_bytes() != expected:
                    raise ValueError("CA3_UNDECLARED_NATIVE_PREFIX_SOURCE_CHANGE")
            elif scope == "retrieval" and local_retrieval_input and name == selected.main:
                binding = report["retrieval_input_contract"]
                old_literal = ('"' + binding['original_source_input_binding'] + '"').encode('utf-8')
                new_literal = ('"' + binding['effective_input_binding'] + '"').encode('utf-8')
                expected = (project/name).read_bytes().replace(old_literal, new_literal, 1)
                if cue_omission:
                    from .ca3_cue_omission import omission_program

                    expected = omission_program(expected, population_multiplier=population_multiplier)
                    omission_verified = (local_project/name).read_bytes() == expected
                else:
                    path_only = (local_project/name).read_bytes() == expected
                if not path_only and not omission_verified:
                    raise ValueError("CA3_RETRIEVAL_CHANGED_BEYOND_INPUT_PATH")
            if not preserved and not path_only and not extension_verified and not omission_verified and (scope != "prefix" or name != selected.main):
                raise ValueError("CA3_STAGED_SOURCE_BYTES_CHANGED:" + name)
            file_checks.append(dict(path=name, source_sha256=original_sha,
                                    staged_sha256=copied_sha, source_bytes_preserved=preserved,
                                     source_path_binding_only=path_only,
                                     coupled_extension_transformation_verified=extension_verified))
            if cue_omission and name == selected.main:
                file_checks[-1]['cue_omission_transformation_verified'] = omission_verified
            if neuron_substeps is not None and name == selected.main:
                file_checks[-1]['neuron_substep_transformation_verified'] = extension_verified
        report["source_file_byte_checks"] = file_checks
        if realization is not None:
            from .ca3_realization import (program as realization_program, contract as realization_contract,
                                          source_reset_contract)

            original = path.read_bytes()
            path.write_bytes(realization_program(original, realization))
            main_row = next(row for row in file_checks if row['path'] == selected.main)
            main_row['pre_realization_staged_sha256'] = main_row['staged_sha256']
            main_row['staged_sha256'] = sha256_file(path)
            main_row['source_realization_transformation_verified'] = True
            report['source_realization'] = realization_contract(realization)
            report['source_initial_state_contract'] = source_reset_contract(
                (local_project/'generateCONFIGStateSTP.h').read_bytes(),
                (framework/'carlsim/kernel/src/snn_manager.cpp').read_bytes())
            report['source_realization_config'] = dict(path=str(source_realization_config),
                sha256=sha256_file(source_realization_config), resolved=realization.record())
            report['source_realization_implementation_sha256'] = sha256_file(
                Path(__file__).with_name('ca3_realization.py'))
        if native_observation:
            report["native_observation_overlay"] = apply_overlay(source_root / "carlsim_ca3", framework)
            report["observation_source_execution_equivalence_qualified"] = False
            report["observer_implementation_sha256"] = sha256_file(
                Path(__file__).with_name("ca3_native_observation.py"))
        if conductance_memory:
            # Existing staged reference bundles do not contain this opt-in helper.
            from .ca3_conductance_memory import apply_overlay as apply_conductance_memory_overlay

            report["conductance_memory_overlay"] = apply_conductance_memory_overlay(
                source_root / "carlsim_ca3", framework,
                preceding_overlay=report.get("native_observation_overlay"))
            report["conductance_memory_implementation_sha256"] = sha256_file(
                Path(__file__).with_name("ca3_conductance_memory.py"))
        with (framework / "CMakeLists.txt").open("a") as stream:
            stream.write("\nadd_subdirectory(projects/" + selected.local_name + ")\n")
        report.update(author_revision=author_manifest["revision"], framework_revision=framework_manifest["revision"],
                      core_neurons=sum(int(n) for _, n in cells), populations={name:int(n) for name,n in cells},
                      pattern_integer_count=len(pattern), pattern_sha256=sha256_file(effective_pattern),
                      effective_pattern_relative_path=str(effective_pattern.relative_to(workdir)),
                      integration="RUNGE_KUTTA4_" + str(neuron_substeps or 5), delay_ms=1,
                      overlay_source_sha256=sha256_file(path),
                      # Author-program preservation is separate from the backend's
                      # corrected initialization and its trajectory-equivalence claim.
                      author_program_semantics_preserved_before_stop=(population_multiplier == 1 and not cue_omission),
                      source_code_semantics_preserved_before_stop=(population_multiplier == 1 and not conductance_memory and not cue_omission),
                      original_author_main_sha256=sha256_file(project / selected.main),
                      complete_author_formation_program_unchanged=(scope == "full-formation" and population_multiplier == 1),
                      complete_author_retrieval_program_unchanged=(scope == "retrieval" and not local_retrieval_input and population_multiplier == 1),
                      retrieval_program_changed_only_in_input_path=(scope == "retrieval" and local_retrieval_input and population_multiplier == 1 and not cue_omission),
                      source_rng="CARLsim_seed10_and_author_default_host_streams_not_independent_stream_qualified")
        if realization is not None:
            report['source_rng'] = ('explicit_post_setup_noise_block_on_source_reset' if realization.phase == 'retrieval'
                                    else 'explicit_circuit_seed_with_source_coupled_formation_streams')
            report['author_program_semantics_preserved_before_stop'] = False
            report['source_code_semantics_preserved_before_stop'] = False

        build = workdir / "build"
        if cuda_root is not None:
            helpers = cuda_root / "samples_source"
            helper_manifest = fetch_source("cuda_samples", helpers)
            copied_helpers = 0
            for row in helper_manifest["files"]:
                if not row["path"].startswith("Common/"):
                    continue
                target = cuda_root / "samples/common/inc" / row["path"].removeprefix("Common/")
                if target.is_file() and sha256_file(target) == row["sha256"]:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(helpers / row["path"], target)
                copied_helpers += 1
            report["cuda_helper_files_materialized"] = copied_helpers
            report["cuda_samples_revision"] = helper_manifest["revision"]
            freeze_json(workdir / "cuda_samples_source.json", helper_manifest)
        commands = [["cmake", "-S", str(framework), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                     "-DCMAKE_C_COMPILER=/usr/bin/gcc", "-DCMAKE_CXX_COMPILER=/usr/bin/g++",
                     "-DCUDA_HOST_COMPILER=/usr/bin/g++",
                     "-DCARLSIM_SAMPLES=OFF", "-DCARLSIM_PYCARL=OFF", "-DCARLSIM_TUTORIALS=OFF",
                     "-DCARLSIM_TEST=OFF", "-DCARLSIM_TOOLS=ON", "-DCARLSIM_NO_CUDA=OFF",
                     "-DCARLSIM_SHARED=ON", "-DCARLSIM_JK_CA3_SNN=ON", "-DCARLSIM_CSTP_DOUBLES=ON",
                     "-DCARLSIM_CUDA_GENCODE=-gencode arch=compute_"+cuda_arch+",code=sm_"+cuda_arch],
                    ["cmake", "--build", str(build), "--target", selected.target, "--parallel", os.environ.get("SLURM_CPUS_PER_TASK", "1")]]
        if cuda_root is not None:
            commands[0].extend(["-DCUDA_TOOLKIT_ROOT_DIR=" + str(cuda_root),
                                "-DCUDA_NVCC_EXECUTABLE=" + str(cuda_root / "bin/nvcc")])
            report["cuda_root"] = str(cuda_root)
        if boost_include is not None:
            boost_header = boost_include / "boost/iterator/counting_iterator.hpp"
            if not boost_header.is_file():
                raise ValueError("CA3_BOOST_ITERATOR_HEADER_MISSING")
            commands[0].append("-DCMAKE_CXX_FLAGS=-idirafter " + str(boost_include))
            report["boost_include"] = str(boost_include)
            report["boost_version_header_sha256"] = sha256_file(boost_include / "boost/version.hpp")
        report["build_commands"] = commands
        for index, command in enumerate(commands):
            with (workdir / f"build_{index}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = build / "projects" / selected.local_name / selected.target
        report["binary_relative_path"] = str(binary.relative_to(workdir))
        report["linked_libraries"] = subprocess.check_output(["ldd", str(binary)], text=True)
        report["nvcc_version"] = subprocess.check_output([str(cuda_root / "bin/nvcc") if cuda_root else "nvcc", "--version"], text=True)
        report["binary_sha256"] = sha256_file(binary)
        if native_observation:
            report["native_observation_overlay"]["native_build_verified"] = True
            report["source_code_semantics_preserved_before_stop"] = False
        if build_only:
            report.update(status=("NATIVE_RETRIEVAL_SOURCE_BUILD_PASSED_NORMALIZED65P_EXECUTION_PENDING"
                                  if scope == "retrieval" else "NATIVE_SOURCE_BUILD_PASSED_GPU_RUN_PENDING"),
                          native_run_executed=False)
            if population_multiplier == 2:
                report["status"] = "COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED"
            return report
        run = native_run_dir if native_run_dir is not None else workdir / "native_run"
        run.mkdir()
        report["native_run_directory"] = str(run)
        shutil.copy2(effective_pattern, run / "trainPat.csv")
        started = time.perf_counter()
        with (workdir / "native.log").open("w") as log:
            subprocess.run([str(binary)], cwd=run, stdout=log, stderr=subprocess.STDOUT, check=True)
        checkpoint_records = verify_checkpoints(run, scope, expected_units=report["core_neurons"])
        checkpoint = run / checkpoint_records[-1]["path"]
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError("CA3_DECLARED_NATIVE_CHECKPOINT_MISSING")
        status = "NATIVE_PREFIX_EXIT_ZERO" if scope == "prefix" else "NATIVE_FORMATION_EXIT_ZERO"
        if population_multiplier == 2:
            status = "COUPLED_EXTENSION_PREFIX_EXIT_ZERO" if scope == "prefix" else "COUPLED_EXTENSION_FORMATION_EXIT_ZERO"
        report.update(status=status,
                      native_run_executed=True, exit_code=0, complete_formation_executed=(scope == "full-formation"),
                      checkpoint_relative_path="native_run/" + checkpoint.name, checkpoints=checkpoint_records,
                      native_elapsed_seconds=time.perf_counter()-started,
                      checkpoint_bytes=checkpoint.stat().st_size, checkpoint_sha256=sha256_file(checkpoint))
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if realization is not None and report.get('status') == 'COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED':
            # Native consumers require the explicit matching configuration.
            # Software support is separate from actual native qualification.
            report['status'] = 'CA3_REALIZATION_BUILD_PASSED_NATIVE_QUALIFICATION_PENDING'
            report['source_realization_native_execution_qualified'] = False
        freeze_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--cuda-arch", required=True)
    parser.add_argument("--native-run-dir", type=Path)
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--scope", choices=SCOPES, default="prefix",
                        help="retrieval requires --build-only; its original normalized65p binding is retained")
    parser.add_argument("--local-retrieval-input", action="store_true",
                        help="Replace only the author normalized65p path literal with its basename")
    parser.add_argument("--boost-include", type=Path)
    parser.add_argument("--native-observation", action="store_true",
                        help="Build explicit per-ms native counters/current observation; requires --build-only")
    parser.add_argument("--coupled-extension-config", type=Path,
                        help="Opt in to the unqualified 2x source extension declared in the campaign YAML")
    parser.add_argument("--conductance-memory", action="store_true",
                        help="Use guarded rise-buffer allocation and complete zero initialization in the copied GPU source")
    parser.add_argument('--cue-omission', action='store_true',
                        help='Development retrieval contrast: retain recall background instead of all cue overwrites')
    parser.add_argument('--neuron-substeps', type=int, choices=(5, 10, 20, 40),
                        help='Explicit observed coupled retrieval sensitivity experiment; source network clock stays1ms')
    parser.add_argument('--source-realization-config', type=Path,
                        help='Build an explicit post-setup noise realization; native qualification is separate')
    args = parser.parse_args()
    print(json.dumps(execute(args.source_root.resolve(), args.workdir.resolve(), args.cuda_arch,
                             args.native_run_dir.resolve() if args.native_run_dir else None,
                             args.cuda_root.resolve() if args.cuda_root else None, build_only=args.build_only,
                             boost_include=args.boost_include.resolve() if args.boost_include else None, scope=args.scope,
                             local_retrieval_input=args.local_retrieval_input, native_observation=args.native_observation,
                             coupled_extension_config=args.coupled_extension_config,
                             conductance_memory=args.conductance_memory, cue_omission=args.cue_omission,
                             neuron_substeps=args.neuron_substeps, source_realization_config=args.source_realization_config)))


if __name__ == "__main__":
    main()
