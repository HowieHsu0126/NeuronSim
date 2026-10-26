"""Explicit CA3 development intervention: omit all three source cue overwrites.

The formed checkpoint, initialisation, distribution objects, draws, restoration
calls and clock remain unchanged. This is a whole-schedule contrast, not three
independent interventions or a qualified biological result.
"""
from __future__ import annotations

import hashlib

SOURCE_MAIN_SHA256 = {
    1: '84e65574399bfcd71b48f74d96bb3d66cf3bb5c6e389e7eab3d3dd1de7703f4a',
    2: 'cf88ebeff742ce83893a8ced810bd0aca1bda5cfc058c125b88251b06ab51bfb',
}
OMISSION_MAIN_SHA256 = {
    1: '7eb0be21fdb6b30cf88738ccdc7446ebf79ca5c4f51ace10308883c5de4c8b6f',
    2: '6dd9836493bb5935f668e086c2572f734fccb8058a7c35b6fcf91933cf5ed024',
}
SOURCE_ASSIGNMENT = b'pc_current.at(randPyrCell) = 10000.0f;'
OMITTED_ASSIGNMENT = b'(void)randPyrCell; // cue_omission: retain realized recall background'


def omission_program(source: bytes, *, population_multiplier: int) -> bytes:
    if type(population_multiplier) is not int or population_multiplier not in (1, 2):
        raise ValueError('CA3_OMISSION_SOURCE_SCALE_REQUIRED')
    if (hashlib.sha256(source).hexdigest() != SOURCE_MAIN_SHA256[population_multiplier]
            or source.count(SOURCE_ASSIGNMENT) != 1):
        raise ValueError('CA3_OMISSION_PINNED_LOCAL_INPUT_PROGRAM_REQUIRED')
    result = source.replace(SOURCE_ASSIGNMENT, OMITTED_ASSIGNMENT, 1)
    if hashlib.sha256(result).hexdigest() != OMISSION_MAIN_SHA256[population_multiplier]:
        raise ValueError('CA3_OMISSION_UNDECLARED_SOURCE_CHANGE')
    return result


def contract(population_multiplier: int) -> dict:
    if type(population_multiplier) is not int or population_multiplier not in (1, 2):
        raise ValueError('CA3_OMISSION_SOURCE_SCALE_REQUIRED')
    return dict(condition='cue_omission', scope='development_whole_schedule_contrast',
        source_main_sha256=SOURCE_MAIN_SHA256[population_multiplier],
        omission_main_sha256=OMISSION_MAIN_SHA256[population_multiplier],
        omitted_cue_starts_ms=[520, 750, 980], cue_duration_ms=20, duration_ms=1001,
        source_statement_replacements=1, current_assignment_semantics='retain_realized_recall_background',
        random_draws_changed=False, current_distributions_changed=False,
        formation_changed=False, independent_rng_streams_qualified=False,
        independent_cue_effects_claimed=False, scientific_fidelity_qualified=False,
        production_episodes=0)


def validate_report(report: dict, *, population_multiplier: int) -> None:
    expected = contract(population_multiplier)
    if (report.get('retrieval_condition') != 'cue_omission'
            or report.get('retrieval_input_contract', {}).get('cue_omission') != expected
            or report.get('overlay_source_sha256') != expected['omission_main_sha256']
            or report.get('source_code_semantics_preserved_before_stop') is not False
            or report.get('retrieval_program_changed_only_in_input_path') is not False):
        raise ValueError('CA3_OMISSION_SOURCE_CONTRACT_REQUIRED')
    main_rows = [row for row in report.get('source_file_byte_checks', [])
                 if row['path'].startswith('src/main_ca3_')]
    if (len(main_rows) != 1 or main_rows[0].get('staged_sha256') != expected['omission_main_sha256']
            or main_rows[0].get('cue_omission_transformation_verified') is not True
            or main_rows[0].get('source_path_binding_only') is not False):
        raise ValueError('CA3_OMISSION_BOUND_MAIN_REQUIRED')
