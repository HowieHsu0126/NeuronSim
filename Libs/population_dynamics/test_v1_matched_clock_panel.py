import copy
import pytest

from .v1_matched_clock_panel import combine_tables


def tables():
    cases, jobs, added, digests = [], [], [], {}
    for condition in range(10):
        old = dict(episode_id=f"old_{condition}", circuit_id="c0", block_id="c0_b0",
                   block_index=0, condition_index=condition, connectivity_seed=100000000,
                   background_seed=120000000, stimulus_seed=130000000, split="train")
        new = dict(old, episode_id=f"new_{condition}", block_id="c0_b8",
                   block_index=8, background_seed=120000008, stimulus_seed=130000008)
        cases.append(dict(identity=old, reference=f"/old/coarse/{condition}", delivery_sha256="old"))
        jobs.append(dict(identity=old, native_archive=f"/old/fine/{condition}", native_job=100+condition))
        added.append(dict(identity=new, matched_identity=old,
            matched_reference=f"/old/coarse/{condition}", matched_fine=f"/old/fine/{condition}",
            native_archive=f"/new/coarse/{condition}", fine_archive=f"/new/fine/{condition}",
            native_job=200+condition, fine_job=300+condition, comparison_job=400+condition,
            comparison_output=f"/new/comparison/{condition}"))
        digests[new["episode_id"]] = "new-reference"
    return cases, jobs, dict(cases=added, planned_new_development_episodes=10), digests


def test_fine_job_mapping_and_complete_old_blocks_are_preserved():
    cases, jobs, extension, digests = tables()
    before = copy.deepcopy(cases)
    combined, submitted = combine_tables(cases, jobs, extension, digests)
    assert cases == before and combined[:10] == before
    assert len(combined) == len(submitted) == 20
    assert combined[10]["reference"] == "/new/coarse/0"
    assert submitted[10]["native_job"] == 300
    assert submitted[10]["native_archive"] == "/new/fine/0"


@pytest.mark.parametrize("mutation,error", [
    ("missing", "COVERAGE"),
    ("wrong_condition", "CONTROLS"),
    ("reused", "FRESH_NATIVE"),
    ("duplicate", "DUPLICATE"),
])
def test_invalid_extension_is_rejected(mutation, error):
    cases, jobs, extension, digests = tables()
    if mutation == "missing":
        extension["cases"].pop()
    elif mutation == "wrong_condition":
        extension["cases"][0]["identity"]["condition_index"] = 1
    elif mutation == "reused":
        extension["cases"][0]["native_archive"] = "/old/coarse/0"
    else:
        extension["cases"].append(copy.deepcopy(extension["cases"][0]))
    with pytest.raises(ValueError, match=error):
        combine_tables(cases, jobs, extension, digests)
