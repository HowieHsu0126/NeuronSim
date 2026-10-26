"""Physical sampling-frame identity and batched conditional-reference parity."""
from copy import deepcopy
import json
from unittest.mock import Mock

import numpy as np
import pytest
from . import pfc_first_passage_history as history
from . import pfc_first_passage_reference as reference
from . import pfc_deterministic_replay as source


def fixture():
    config = dict(dt_ms=.05, duration_ms=5., excitatory_neurons=32, inhibitory_neurons=8,
        excitatory_preference_bins=16, weight_scaling_factor=64., stimulus_start_ms=0.,
        stimulus_duration_ms=0., stimulus_center_deg=120., stimulus_width_deg=30., stimulus_strength_nA=0.,
        distractor_start_ms=0., distractor_duration_ms=0., distractor_center_deg=90., distractor_width_deg=40.)
    case = dict(case_id="history_fixture", stimulus_strength_nA=0., distractor_strength_nA=0.)
    initial = np.full(40, -60.); initial[:2] = -50.000001
    e = np.zeros((100, 32), dtype=np.uint16); e[0,:2] = 20; e[1,:2] = 3
    i = np.full((100, 8), 10, dtype=np.uint16)
    tape = reference.validate_tape(config, source.simulate(config, case, initial, e, i)["raw"])
    return config, case, e, i, tape


def test_sampling_physical_ids_match_clocks_and_census_covers_frame():
    config, *_ = fixture()
    kwargs = dict(field=5, phase="cue", window_ms=5., sample_size=64, seed_components=[20261009,95001,5,1])
    a, ma = history.sample_frame(config, **kwargs)
    b, mb = history.sample_frame(dict(config, dt_ms=.025), **kwargs)
    keys = ("frame_index", "unit_index", "window_start_ms", "window_stop_ms")
    assert [[r[k] for k in keys] for r in a] == [[r[k] for k in keys] for r in b]
    assert ma == mb
    rows, frame = history.sample_frame(config, **dict(kwargs, sample_size=80))
    assert sorted(r["frame_index"] for r in rows) == list(range(frame["frame_size"]))


def test_classification_keeps_zeros_and_multispike_failure_rows():
    config, _, _, _, tape = fixture()
    events = np.array([[1,0],[60,0],[90,1]], dtype=np.int64)
    t = reference.SpikeTape(events,np.bincount(events[:,1],minlength=40),32,8,.00005,100)
    rows = [dict(sample_index=u,unit_index=u,window_start_tick=0,window_stop_tick=100) for u in range(3)]
    actual = history.classify(config,t,rows)
    assert [r["category"] for r in actual] == ["suppressed_network_reference_required", "refractory_covers_remaining_window", "no_first_spike"]
    assert len(actual) == len(rows)
    with pytest.raises(ValueError,match="SUPPRESSED_NETWORK_REPLAY_REQUIRED_NO_ROWS_DROPPED"):
        history.extract(config,{},t,actual)


def test_batched_references_equal_individual_validated_reconstruction():
    config, case, e, i, tape = fixture()
    rows = [dict(sample_index=u,unit_index=u,window_start_tick=0,window_stop_tick=100) for u in range(3)]
    rows = history.classify(config,tape,rows)
    assert [r["category"] for r in rows] == ["conditional_integration_required"]*2+["no_first_spike"]
    actual, _ = history.extract(config,case,tape,rows,counts_E=e,counts_I=i)
    assert len(actual)==2 and len(rows)==3
    for row in rows[:2]:
        selected = reference.FirstSpikeWindow(row["unit_index"],row["first_spike_tick"],0,100,40)
        frames=[]
        def capture(stage,tick,state):
            if stage=="pre_rk2" and selected.first_spike_tick<tick<100:
                frames.append({k:state["E__"+k][row["unit_index"]] for k in ("s_GABA","s_NMDA_total","I_stim","s_NMDA","x")})
        expected=reference.reconstruct(config,case,tape,stop_tick=100,counts_E=e,counts_I=i,
                                       selection=selected,on_snapshot=capture)
        initial,forcing=actual[row["sample_index"]]
        assert initial==expected["conditional_initial"]
        for key in frames[0]: np.testing.assert_array_equal(forcing[key],[f[key] for f in frames])



def integration_fixture():
    config, case, e, i, tape = fixture()
    rows = [dict(sample_index=u, frame_index=100+u, unit_index=u,
                 window_start_tick=0, window_stop_tick=100) for u in range(3)]
    rows = history.classify(config, tape, rows)
    refs, _ = history.extract(config, case, tape, rows, counts_E=e, counts_I=i)
    settings = dict(source_rng_seed=95001, field_index=5, source="fixture_source",
                    importance_sampling=dict(samples=8))
    return config, settings, rows, refs


def integration_result(unit):
    return dict(stopped_prefix=dict(estimate=.125*(unit+1), empirical_score_variance=.5*(unit+1)),
                proposal_hits=unit+2)


def test_integrate_rows_preserves_legacy_seeds_paths_reports_and_row_mutation(tmp_path, monkeypatch):
    config, settings, original_rows, refs = integration_fixture()
    calls = []

    def integrate(actual_config, conditional, arrays, inner, output):
        assert actual_config is config
        unit = conditional["unit_index"]
        assert conditional is refs[unit][0] and arrays is refs[unit][1]
        calls.append(deepcopy(inner))
        return integration_result(unit)

    monkeypatch.setattr(history.importance, "run", integrate)
    for name, context in (("absent", None), ("empty_list", []), ("empty_tuple", ())):
        actual_settings = deepcopy(settings)
        if context is not None:
            actual_settings["integration_seed_context"] = context
        original_settings = deepcopy(actual_settings)
        rows = deepcopy(original_rows)
        output = tmp_path/name
        output.mkdir()
        values, variances = history.integrate_rows(config, actual_settings, rows, refs, output)
        assert values == [.125, .25, 0.] and variances == [.0625, .125, 0.]
        assert actual_settings == original_settings
        expected_rows = deepcopy(original_rows)
        for index, row in enumerate(expected_rows):
            row.update(conditional_estimate=0., inner_monte_carlo_variance=0., proposal_hits=0)
            if index not in refs:
                continue
            result = integration_result(index)
            report_row = dict(row)
            row.update(conditional_estimate=result["stopped_prefix"]["estimate"],
                       inner_monte_carlo_variance=result["stopped_prefix"]["empirical_score_variance"]/8,
                       proposal_hits=result["proposal_hits"],
                       integration_report=f"sample_{index:03d}/integration/report.json")
            expected_report = dict(result,
                reference_archive_role="accepted_source_provenance_for_extracted_reference",
                extracted_reference="../reference", sample_frame_row=report_row)
            base = output/f"sample_{index:03d}"
            assert (base/"integration/report.json").read_text() == json.dumps(expected_report, indent=2)+"\n"
            assert (base/"reference/conditional_initial.json").read_text() == json.dumps(refs[index][0], indent=2)+"\n"
            with np.load(base/"reference/reference_window.npz", allow_pickle=False) as archive:
                assert set(archive.files) == set(refs[index][1])
                for key, value in refs[index][1].items():
                    np.testing.assert_array_equal(archive[key], value)
        assert rows == expected_rows
        assert (output/"integrated_frame.json").read_text() == json.dumps(expected_rows, indent=2)+"\n"
        assert not (output/"sample_002").exists()
    expected_calls = [dict(settings["importance_sampling"],
                           seed_components=[20261009, 95001, 5, 100+index, 50, 2],
                           reference="fixture_source") for index in (0, 1)]
    assert calls == expected_calls*3


def test_integrate_rows_appends_phase_and_auxiliary_context_without_replacing_legacy_seeds(tmp_path, monkeypatch):
    config, settings, rows, refs = integration_fixture()
    settings["integration_seed_namespace"] = 5
    calls = []

    def integrate(actual_config, conditional, arrays, inner, output):
        calls.append(tuple(inner["seed_components"]))
        return integration_result(conditional["unit_index"])

    monkeypatch.setattr(history.importance, "run", integrate)
    contexts = ([1, 0, 0], [2, 0, 0], [1, 0, 1])
    for index, context in enumerate(contexts):
        output = tmp_path/str(index)
        output.mkdir()
        scoped = dict(settings, integration_seed_context=context)
        history.integrate_rows(config, scoped, deepcopy(rows), refs, output)
        assert calls[-2:] == [(20261009, 95001, 5, 100+unit, 50, 5, *context) for unit in (0, 1)]
        assert scoped["integration_seed_context"] == context
    assert len(set(calls[::2])) == len(contexts)


@pytest.mark.parametrize("context", [None, "phase", [True], [1.0], [-1], [2**32], [[0]]])
def test_integrate_rows_rejects_invalid_context_before_output_or_integration(tmp_path, monkeypatch, context):
    integration = Mock()
    monkeypatch.setattr(history.importance, "run", integration)
    with pytest.raises(ValueError, match="INTEGRATION_SEED_CONTEXT"):
        history.integrate_rows({}, dict(integration_seed_context=context), [], {}, tmp_path)
    integration.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_integrate_rows_retains_all_structural_zero_rows(tmp_path, monkeypatch):
    integration = Mock()
    monkeypatch.setattr(history.importance, "run", integration)
    rows = [dict(sample_index=0, frame_index=20, category="no_first_spike"),
            dict(sample_index=1, frame_index=21, category="refractory_covers_remaining_window")]
    identities = [(row["sample_index"], row["frame_index"], row["category"]) for row in rows]
    assert history.integrate_rows({}, {}, rows, {}, tmp_path) == ([0., 0.], [0., 0.])
    assert [(row["sample_index"], row["frame_index"], row["category"]) for row in rows] == identities
    assert all(row["conditional_estimate"] == row["inner_monte_carlo_variance"] == row["proposal_hits"] == 0 for row in rows)
    assert json.loads((tmp_path/"integrated_frame.json").read_text()) == rows
    integration.assert_not_called()


def test_integrate_rows_writes_supported_empty_frame(tmp_path, monkeypatch):
    integration = Mock()
    monkeypatch.setattr(history.importance, "run", integration)
    rows = []
    assert history.integrate_rows({}, {}, rows, {}, tmp_path) == ([], [])
    assert rows == [] and (tmp_path/"integrated_frame.json").read_text() == "[]\n"
    integration.assert_not_called()


def test_integrate_rows_propagates_importance_failure_and_keeps_complete_row_identity(tmp_path, monkeypatch):
    config, settings, rows, refs = integration_fixture()
    original_rows = deepcopy(rows)
    calls = []
    failure = RuntimeError("integration fixture failure")

    def integrate(actual_config, conditional, arrays, inner, output):
        calls.append(conditional["unit_index"])
        if conditional["unit_index"] == 1:
            raise failure
        return integration_result(conditional["unit_index"])

    monkeypatch.setattr(history.importance, "run", integrate)
    with pytest.raises(RuntimeError, match="integration fixture failure") as caught:
        history.integrate_rows(config, settings, rows, refs, tmp_path)
    assert caught.value is failure and calls == [0, 1]
    assert len(rows) == 3 and [row["sample_index"] for row in rows] == [0, 1, 2]
    assert rows[0]["conditional_estimate"] == .125
    assert rows[2] == original_rows[2]
    saved = json.loads((tmp_path/"integrated_frame.json").read_text())
    assert len(saved) == 3 and saved[0]["conditional_estimate"] == .125
    assert saved[1:] == original_rows[1:]
    assert (tmp_path/"sample_000/integration/report.json").is_file()
    assert (tmp_path/"sample_001/reference/conditional_initial.json").is_file()
    assert not (tmp_path/"sample_002").exists()
