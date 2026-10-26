import copy
import numpy as np
import pytest

from .v1_clock_noise_comparison import change_summary, select_blocks


def test_unsigned_decreases_and_silent_elements_are_retained():
    a = np.array([4, 0, 2], dtype=np.uint64)
    fa = np.array([5, 0, 2], dtype=np.uint64)
    b = np.array([7, 0, 1], dtype=np.uint64)
    fb = np.array([8, 0, 1], dtype=np.uint64)
    result = change_summary(a, fa, b, fb)
    assert result["elements"] == result["common_defined_elements"] == 3
    assert result["reference_noise_mean"] == pytest.approx(2/3)
    assert result["reference_noise_mean_absolute"] == pytest.approx(4/3)
    assert result["mean_clock_to_reference_noise_l1_ratio"] == pytest.approx(0.25)


def test_zero_noise_difference_does_not_hide_nonzero_clock_difference():
    result = change_summary(np.array([0, 1]), np.array([1, 2]),
                            np.array([0, 1]), np.array([2, 3]))
    assert result["reference_noise_l1_zero"]
    assert result["mean_clock_to_reference_noise_l1_ratio"] is None
    assert result["clock_a_mean_absolute"] == 1
    assert result["clock_b_mean_absolute"] == 2


def test_undefined_selectivity_uses_one_common_set_and_retains_denominators():
    result = change_summary(np.array([1., np.nan, .5]), np.array([.9, .3, .5]),
                            np.array([.8, .2, np.nan]), np.array([.7, .4, .6]),
                            allow_undefined=True)
    assert result["elements"] == 3 and result["common_defined_elements"] == 1
    assert result["excluded_undefined_elements"] == 2
    assert result["reference_a_defined_elements"] == 2
    assert result["fine_b_defined_elements"] == 3
    assert result["mean_clock_to_reference_noise_l1_ratio"] == pytest.approx(.5)
    empty = change_summary(*[np.array([np.nan])] * 4, allow_undefined=True)
    assert empty["common_defined_elements"] == 0 and empty["clock_a_mean"] is None


def test_nonfinite_counts_and_mismatched_shapes_fail():
    for values in ((np.array([np.inf]),) * 4, (np.array([np.nan]),) * 4,
                   (np.array([1]), np.array([1, 2]), np.array([1]), np.array([1]))):
        with pytest.raises(ValueError):
            change_summary(*values)


def cases():
    return [dict(identity=dict(episode_id=f"c1_b{block}_k{condition}", circuit_id="c1",
            scenario_id="v1_visual", split="train", block_index=block, condition_index=condition,
            connectivity_seed=100000001, background_seed=120001000+block,
            stimulus_seed=130001000+block))
            for block in (1, 9) for condition in range(10)]


def test_matched_complete_blocks_require_distinct_driver_seeds():
    selected = select_blocks(cases(), "c1", 1, 9)
    assert [len(block) for block in selected] == [10, 10]
    invalid = copy.deepcopy(cases())
    invalid[-1]["identity"]["background_seed"] = 120001001
    with pytest.raises(ValueError, match="DISTINCT_DRIVER"):
        select_blocks(invalid, "c1", 1, 9)


@pytest.mark.parametrize("first,second", [(1, 1), (1, 2), (-7, 1)])
def test_mismatched_rotating_low_contrast_conditions_fail(first, second):
    with pytest.raises(ValueError, match="MATCHED_LOW_CONTRAST"):
        select_blocks(cases(), "c1", first, second)


def test_incomplete_or_nontraining_blocks_fail():
    with pytest.raises(ValueError, match="COMPLETE_TRAIN"):
        select_blocks(cases()[:-1], "c1", 1, 9)
    invalid = cases()
    invalid[0]["identity"]["split"] = "test"
    with pytest.raises(ValueError, match="COMPLETE_TRAIN"):
        select_blocks(invalid, "c1", 1, 9)

def metadata_panel(tmp_path, count=60):
    """Metadata fixtures exercise coverage admission, not mechanistic simulation."""
    import json
    from .provenance import sha256_file
    cases = [dict(identity=dict(episode_id=f"case_{i}")) for i in range(count)]
    plan = tmp_path/"cases.json"
    plan.write_text(json.dumps(dict(cases=cases, full_units=230924)))
    jobs = tmp_path/"envelope_submission.json"
    jobs.write_text(json.dumps(dict(cases=cases)))
    binding = dict(cases_sha256=sha256_file(plan), submissions_sha256=sha256_file(jobs))
    task = dict(status="COMPLETE_V1_BENCHMARK_CLOCK_SUMMARY", exit_code=0,
                complete_pairs=count, evidence=[dict(episode_id=c["identity"]["episode_id"]) for c in cases], **binding)
    joint = dict(status="COMPLETE_V1_CLOCK_JOINT_ACTIVITY_PANEL", pairs=count, planned_pairs=count,
                 reports=[dict(episode_id=c["identity"]["episode_id"]) for c in cases], **binding)
    task_path = tmp_path/"task_summary/analysis/report.json"
    joint_path = tmp_path/"joint_activity/report.json"
    for path, value in ((task_path, task), (joint_path, joint)):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    acceptance = dict(status="PASS_COMPLETE_CLOCK_PANEL_DELIVERY", verified_pairs=count,
        task_summary_report_sha256=sha256_file(task_path), joint_activity_report_sha256=sha256_file(joint_path))
    (tmp_path/"panel_acceptance.json").write_text(json.dumps(acceptance))
    return joint_path


@pytest.mark.parametrize("count", [20, 40, 60])
def test_declared_complete_panel_sizes_are_supported(tmp_path, count):
    from .v1_clock_noise_comparison import load_accepted_panel
    metadata_panel(tmp_path, count)
    acceptance, plan, task, joint = load_accepted_panel(tmp_path)
    assert acceptance["verified_pairs"] == len(plan["cases"]) == task["complete_pairs"] == joint["pairs"] == count


def test_changed_and_incomplete_panel_reports_are_rejected(tmp_path):
    import json
    from .provenance import sha256_file
    from .v1_clock_noise_comparison import load_accepted_panel
    path = metadata_panel(tmp_path)
    joint = json.loads(path.read_text())
    joint["reports"].pop()
    path.write_text(json.dumps(joint))
    with pytest.raises(ValueError, match="REPORT_CHANGED"):
        load_accepted_panel(tmp_path)
    acceptance_path = tmp_path/"panel_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    acceptance["joint_activity_report_sha256"] = sha256_file(path)
    acceptance_path.write_text(json.dumps(acceptance))
    with pytest.raises(ValueError, match="COVERAGE_DIFFERS"):
        load_accepted_panel(tmp_path)
