"""Explicit neuronal integration refinement of the pinned coupled CA3 retrieval.

The source uses five RK4 neuron updates per1ms network step. This experiment
changes only that count. Conductance/STP decay, spike delivery, plasticity and
event timestamps retain their source1ms clocks; their values may still change.
It tests retrieval conditional on a fixed learned network, not formation or
whole-simulator discretization convergence.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from .provenance import sha256_file

COUNTS = (5, 10, 20, 40)
# The archived helper implements only the frozen5/10/20 contracts. It cannot
# authorize the prospective40 setting, which requires this current helper.
LEGACY_IMPLEMENTATION_SHA256 = '438831106fe1e638826283a4f527859099410abee57289aedd0d75075deae240'
MAIN_SHA256 = {
    5: 'cf88ebeff742ce83893a8ced810bd0aca1bda5cfc058c125b88251b06ab51bfb',
    10: 'b7b3db337bf499e12e18516c873c153e834a82293e28b1e304ad0f0a61ae1a09',
    20: '7ecf554f98aed820504b35fc48371a42893712d4f0c1c6f655f4a952bdb834d2',
    40: 'e37c7580ed50f16826b4df690488d94e18d2c11b14525fcfd20892e61dc5ff92',
}
CONFIG_SHA256 = '1852603fdebe0bf38c4b343063dba99fadc8ac9e82a545f26833e685ffe6acba'
SETUP_SHA256 = 'b6dc5216827055124226f907795857c81e4997ee0b9cf49ce3422d54d701c3ef'
# These are the actual pinned source files used to trace the setter, macrostep,
# disabled homeostasis default and checkpoint reader. The build verifies them
# before the separately declared observer/memory overlays are applied.
FRAMEWORK_SHA256 = {
    'carlsim/kernel/inc/snn_datastructures.h': '50c71ccea154bae8c5da5e04dd0fef93ea603f2e3106abe44bf05dfec944d0b9',
    'carlsim/kernel/src/snn_manager.cpp': 'd14315a63715c4a0a722a22d6e0acfa542f3946ce40cba64c86c91aca8904a1d',
    'carlsim/interface/src/carlsim.cpp': 'c44eb03331c8ef8e4cf891a2083bab5d0e0a3ff0528e0bf5a0ce3803ba5b2545',
    'carlsim/kernel/src/gpu_module/snn_gpu_module.cu': '4e25494a89427f5479838a17dc48cff5b255954061689146068dd85e16c44b4d',
}


def validate_count(count: int) -> None:
    if type(count) is not int or count not in COUNTS:
        raise ValueError('CA3_NEURON_SUBSTEPS_MUST_BE_5_10_20_OR_40')


def program(original: bytes, count: int) -> bytes:
    """Replace only the unique active source count, retaining all other bytes."""
    validate_count(count)
    if hashlib.sha256(original).hexdigest() != MAIN_SHA256[5]:
        raise ValueError('CA3_NEURON_REFINEMENT_REQUIRES_PINNED_COUPLED_RETRIEVAL')
    text = original.decode('utf-8')
    masked = re.sub(r'/\*.*?\*/|//[^\r\n]*',
                    lambda match: re.sub(r'[^\r\n]', ' ', match.group()), text, flags=re.S)
    matches = list(re.finditer(r'\bsim\.setIntegrationMethod\(\s*RUNGE_KUTTA4\s*,\s*(5)\s*\)', masked))
    if len(matches) != 1:
        raise ValueError('CA3_UNIQUE_SOURCE_RK4_5_REQUIRED')
    start, stop = matches[0].span(1)
    result = (text[:start] + str(count) + text[stop:]).encode('utf-8')
    if hashlib.sha256(result).hexdigest() != MAIN_SHA256[count]:
        raise ValueError('CA3_NEURON_REFINEMENT_OUTPUT_DIFFERS')
    return result


def contract(count: int) -> dict:
    validate_count(count)
    return dict(schema='ca3_neuron_substep_refinement_v1',
                scope='fixed_checkpoint_coupled_retrieval', source_neuron_substeps=5,
                effective_neuron_substeps=count, neuron_step_ms=1.0/count,
                network_step_ms=1, event_timestamp_resolution_ms=1,
                input_step_ms=1, homeostasis_enabled=False,
                checkpoint_restores_neuronal_or_rng_state=False,
                formation_refined=False, whole_simulator_convergence_qualified=False,
                scientific_fidelity_qualified=False,
                reference_main_sha256=MAIN_SHA256[5], effective_main_sha256=MAIN_SHA256[count],
                source_framework_sha256=dict(FRAMEWORK_SHA256))


def apply_overlay(main: Path, config: Path, setup: Path, framework: Path, count: int) -> dict:
    """Verify the traced configuration before touching a copied author program."""
    validate_count(count)
    if sha256_file(config) != CONFIG_SHA256 or sha256_file(setup) != SETUP_SHA256:
        raise ValueError('CA3_REFINEMENT_PINNED_CONFIGURATION_REQUIRED')
    for name, expected in FRAMEWORK_SHA256.items():
        if sha256_file(framework/name) != expected:
            raise ValueError('CA3_REFINEMENT_TRACED_FRAMEWORK_CHANGED:' + name)
    # The pinned headers and main never enable homeostasis. The pinned reader
    # (snn_manager.cpp:6810-7121) loads graph/weights, not homeostasis settings.
    rewritten = program(main.read_bytes(), count)
    main.write_bytes(rewritten)
    return contract(count)


def validate_report(report: dict, *, count: int | None) -> None:
    declared = report.get('numerical_refinement')
    if count is None:
        if declared is not None or report.get('integration', 'RUNGE_KUTTA4_5') != 'RUNGE_KUTTA4_5':
            raise ValueError('CA3_REFINEMENT_REQUIRES_EXPLICIT_MATCHING_SUBSTEPS')
        return
    expected = contract(count)
    allowed_implementations = {sha256_file(Path(__file__))}
    if count in (5, 10, 20):
        allowed_implementations.add(LEGACY_IMPLEMENTATION_SHA256)
    if (declared != expected or type(declared.get('effective_neuron_substeps')) is not int
            or report.get('integration') != 'RUNGE_KUTTA4_' + str(count)
            or report.get('scope') != 'coupled_extension_retrieval'
            or report.get('overlay_source_sha256') != MAIN_SHA256[count]
            or report.get('retrieval_condition', 'source') != 'source'
            or not report.get('native_observation_overlay')
            or not report.get('conductance_memory_overlay')
            or report.get('numerical_refinement_implementation_sha256') not in allowed_implementations):
        raise ValueError('CA3_REFINEMENT_BUILD_OR_REQUEST_DIFFERS')
