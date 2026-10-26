"""Source-byte and declaration tests; no native counterfactual results are implied."""
import hashlib

import pytest

from . import ca3_cue_omission as omission
from .ca3_coupled_extension import extend_source
from .ca3_reference import prepare_retrieval_program
from .test_ca3_coupled_extension import source_fixture


@pytest.mark.parametrize('multiplier', [1, 2])
def test_only_the_cue_assignment_changes_and_default_is_preserved(tmp_path, multiplier):
    header, main, pattern = source_fixture('retrieval')
    if multiplier == 2:
        header, main, pattern, _ = extend_source(header, main, pattern, multiplier=2, role='retrieval')
    config = tmp_path/'config.h'
    config.write_bytes(header.encode())
    original = tmp_path/'source.cpp'
    original.write_bytes(main.encode())
    source_contract = prepare_retrieval_program(original, config, local_input=True, population_multiplier=multiplier)
    raw = original.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == omission.SOURCE_MAIN_SHA256[multiplier]
    assert 'cue_omission' not in source_contract
    variant = tmp_path/'omission.cpp'
    variant.write_bytes(main.encode())
    actual = prepare_retrieval_program(variant, config, local_input=True,
                                       population_multiplier=multiplier, cue_omission=True)
    changed = variant.read_bytes()
    assert changed == omission.omission_program(raw, population_multiplier=multiplier)
    assert changed.replace(omission.OMITTED_ASSIGNMENT, omission.SOURCE_ASSIGNMENT, 1) == raw
    assert actual['cue_omission'] == omission.contract(multiplier)
    assert config.read_bytes() == header.encode()
    assert original.read_bytes() == raw
    for token in (b'drand48()', b'runNetwork', b'lognormal_distribution', b'srand', b'setExternalCurrent'):
        assert changed.count(token) == raw.count(token)
    with pytest.raises(ValueError, match='PINNED_LOCAL_INPUT_PROGRAM'):
        omission.omission_program(raw+b'changed', population_multiplier=multiplier)
    with pytest.raises(ValueError, match='PINNED_LOCAL_INPUT_PROGRAM'):
        omission.omission_program(changed, population_multiplier=multiplier)
