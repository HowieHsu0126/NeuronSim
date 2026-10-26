"""Native source identity, complete readback, and descriptive score contracts."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from . import pfc_intervention_outcomes as outcomes
from .pfc_intervention_experiment import _reference_arrays, _save_npz, _write_csv
from .population_intervention import ReferenceTarget
from .provenance import freeze_json, sha256_file


class SmallNative:
    def __init__(self):
        self.episode_id = "fixture"
        self.unit_ids = np.array([10, 20, 30, 40])
        self.unit_field_index = np.array([0, 0, 1, 1])
        self.field_names = ("a", "b")
        self.bin_ticks, self.dt_ms, self.duration_ticks = 100, .05, 600
        self.event_ticks = np.array([0, 100, 199, 200, 299, 300, 400, 499], dtype=np.int64)
        self.event_units = np.array([0, 0, 1, 2, 3, 0, 1, 3], dtype=np.int64)
        self.calls = []
        self.data = SimpleNamespace(_read=self.read)

    def read(self, episode, start, stop, width, first, last):
        assert episode == self.episode_id and (first, last) == (0, 4)
        self.calls.append((start, stop))
        counts = np.zeros(((stop-start)//width, 4), dtype=np.int64)
        for tick, unit in zip(self.event_ticks, self.event_units):
            if start <= tick < stop:
                counts[(tick-start)//width, unit] += 1
        return dict(counts=counts, field_counts=np.stack((counts[:, :2].sum(1), counts[:, 2:].sum(1)), 1),
            unit_ids=self.unit_ids, unit_field_index=self.unit_field_index, field_names=self.field_names,
            bin_seconds=.005)

    def ready_at(self, tick):
        assert tick == 500
        return np.array([0, 0, 0, 19], dtype=np.int64)


def test_native_chunks_preserve_half_open_window_final_partial_chunk_and_all_units():
    native = SmallNative()
    result = outcomes.read_native_outcome(native, 100, 4, chunk_bins=3)
    assert native.calls == [(100, 400), (400, 500)]
    np.testing.assert_array_equal(result["unit_total_counts"], [2, 2, 1, 2])
    np.testing.assert_array_equal(result["field_counts"], [[2, 0], [0, 2], [1, 0], [1, 1]])
    np.testing.assert_array_equal(result["field_rate_hz"], np.array([[2, 0], [0, 2], [1, 0], [1, 1]]) / .01)
    np.testing.assert_array_equal(result["terminal_ready_in_ticks"], [0, 0, 0, 19])
    again = outcomes.read_native_outcome(native, 100, 4, chunk_bins=100)
    for name in result:
        np.testing.assert_array_equal(result[name], again[name])


def test_prefix_uses_tick_unit_multisets_and_excludes_post_origin_events():
    factual, other = SmallNative(), SmallNative()
    factual.event_ticks = other.event_ticks = np.array([0, 100, 100, 200])
    factual.event_units = np.array([0, 1, 2, 3])
    other.event_units = np.array([0, 2, 1, 0])
    outcomes.require_same_prefix(factual, other, 200)
    other.event_units[0] = 1
    with pytest.raises(ValueError, match="OBSERVED_NATIVE_SPIKE_PREFIX_DIFFERS"):
        outcomes.require_same_prefix(factual, other, 200)


def test_prefix_cannot_relabel_unit_identity():
    factual, other = SmallNative(), SmallNative()
    other.unit_ids = other.unit_ids[::-1].copy()
    with pytest.raises(ValueError, match="NATIVE_OBSERVATION_MAP_CHANGED"):
        outcomes.require_same_prefix(factual, other, 200)


@pytest.mark.parametrize("fault", ["dtype", "map", "missing_event"])
def test_reader_rejects_invalid_counts_maps_and_incomplete_event_denominator(fault):
    native = SmallNative()
    original = native.data._read
    def changed(*args):
        value = original(*args)
        if fault == "dtype":value["counts"] = value["counts"].astype(np.float32)
        elif fault == "map":value["unit_ids"] = value["unit_ids"][::-1]
        else:
            value["counts"][:] = 0
            value["field_counts"][:] = 0
        return value
    native.data._read = changed
    with pytest.raises(ValueError):
        outcomes.read_native_outcome(native, 100, 4)


def test_single_native_path_reference_distance_is_not_duplicated_into_an_ensemble():
    target = ReferenceTarget(np.array([[[0.], [0.]], [[2.], [2.]]]), ("a",), np.array([2.]), ("x", "y"), .005)
    assert outcomes.realized_reference_distance(np.array([[0.], [2.]]), target) == pytest.approx(1/np.sqrt(2))


def test_incomplete_search_rejects_before_native_source_admission(tmp_path, monkeypatch):
    search, replay = tmp_path/"search", tmp_path/"replay"
    for root in (search, replay):
        root.mkdir();freeze_json(root/"analysis.json", dict(status="RUNNING"))
    monkeypatch.setattr(outcomes, "_adapters", lambda *args: (_ for _ in ()).throw(AssertionError("native read")))
    with pytest.raises(ValueError, match="COMPLETE_THREE_SEED_SEARCH_AND_REPLAY_REQUIRED"):
        outcomes.admit_search(search, replay, {}, None)


def test_duplicate_seed_block_case_cannot_replace_missing_case():
    rows = [dict(training_seed=s, factual_episode_id=e, path=str(Path(f"seed_{s}")/e))
            for s in outcomes.SEEDS for e in outcomes.FACTUAL_EPISODES]
    rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError, match="NINE_DISTINCT_SEED_BLOCK_CASES_REQUIRED"):
        outcomes._case_index(dict(cases=rows))


@pytest.fixture
def complete_readback(tmp_path, monkeypatch):
    inputs, preparation, config_path, reference = (tmp_path/"inputs.json", tmp_path/"preparation", tmp_path/"config.yaml", tmp_path/"reference.json")
    inputs.write_text("{}");preparation.mkdir();(preparation/"preparation.json").write_text("{}");reference.write_text("{}")
    (preparation/"source_snapshot.json").write_text('{"fixture_runtime":1}')
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/neural_population_pfc_intervention_development.yaml").read_text())
    config["reference_sha256"] = sha256_file(reference);config_path.write_text(yaml.safe_dump(config))
    ids, fields = np.arange(100000), tuple(f"f{i}" for i in range(17))
    target = ReferenceTarget(np.ones((8,980,17)), fields, np.ones(17), tuple(f"reference{i}" for i in range(8)), .005)
    monkeypatch.setattr(outcomes, "load_reference", lambda path: (target,dict(inputs_sha256=sha256_file(inputs))))
    monkeypatch.setattr(outcomes, "FrozenInputs", lambda path: SimpleNamespace(proof=dict(input_sha256=sha256_file(inputs))))
    def source_config(episode):
        return dict(benchmark_episode_candidate=dict(identity=dict(episode_id=episode,condition_index=int(episode[-2:]))))
    def source_binding(episode):
        block=int(episode.split("_block_")[1][:3])
        return dict(episode_id=episode,circuit_id="circuit074",block_id=f"block{block}")
    native_loads=[]
    def load(episode):
        native_loads.append(episode)
        return SimpleNamespace(episode_id=episode,source_config=source_config(episode),binding=source_binding(episode),
            unit_ids=ids,unit_field_index=np.r_[np.repeat(np.arange(16),5000),np.repeat(16,20000)],field_names=fields,
            bin_ticks=100,dt_ms=.05,event_ticks=np.array([0,100,2000]),event_units=np.array([0,1,2]))
    monkeypatch.setattr(outcomes,"_adapters",lambda *args:load)
    monkeypatch.setattr(outcomes,"compile_candidates",lambda native,conditions:tuple(SimpleNamespace(name=f"program{c:02d}",
        source_config=source_config(outcomes.candidate_episode(native.episode_id,c))) for c in conditions))
    def observed(native,*args,**kwargs):
        block=int(native.episode_id.split("_block_")[1][:3]);condition=int(native.episode_id[-2:])
        return dict(field_rate_hz=np.full((980,17),1+block+condition/10),field_counts=np.zeros((980,17),dtype=np.int64),
            unit_total_counts=np.zeros(100000,dtype=np.int64),terminal_ready_in_ticks=np.zeros(100000,dtype=np.int64),
            unit_ids=ids,field_ids=np.array(fields))
    monkeypatch.setattr(outcomes,"read_native_outcome",observed)
    experiment=dict(configuration=config,inputs_sha256=sha256_file(inputs),preparation_sha256=sha256_file(preparation/"preparation.json"),
        config_sha256=sha256_file(config_path),reference_sha256=sha256_file(reference),
        seed_cells={str(s):dict(consumed_files={"training/best.pt":f"{s:064x}"}) for s in outcomes.SEEDS})
    roots=[tmp_path/"search",tmp_path/"replay"]
    for root in roots:root.mkdir()
    template=tmp_path/"prediction.npz"
    _save_npz(template,dict(field_rate_hz=np.ones((16,980,17)),field_counts=np.zeros((16,980,17),dtype=np.int64),
        unit_total_counts=np.zeros((16,100000),dtype=np.int64),terminal_ready_in_ticks=np.zeros((16,100000),dtype=np.int64),unit_ids=ids,field_ids=np.array(fields)))
    entries=[[],[]];all_rows=[[],[]]
    for seed in outcomes.SEEDS:
        for episode in outcomes.FACTUAL_EPISODES:
            relative=Path(f"seed_{seed}")/episode
            search,replay=(root/relative for root in roots);search.mkdir(parents=True);replay.mkdir(parents=True)
            rows=[]
            for condition in range(10):
                name=f"program{condition:02d}";cfg=source_config(outcomes.candidate_episode(episode,condition))
                rows.append(dict(candidate=name,candidate_identity=cfg["benchmark_episode_candidate"]["identity"],source_config=cfg,objective=0.))
                shutil.copyfile(template,search/(name+".npz"))
            selected=f"program{seed-41:02d}"
            binding=dict(experiment=experiment,training_seed=seed,factual_episode_id=episode,checkpoint_sha256=f"{seed:064x}")
            _save_npz(search/"reference.npz",_reference_arrays(target));_write_csv(search/"candidates.csv",[dict(candidate=r["candidate"],objective=0.) for r in rows])
            report=dict(status="RANKED_DEVELOPMENT_COMPILED_PFC_CANDIDATES",schema="pfc_population_intervention_case_v1",
                source_unit_count=100000,joint_draws=16,horizon_bins=980,origin_tick=2000,sampling_stream_seeds=[41,42,43],
                scientific_claim_qualified=False,native_execution_performed=False,experiment_binding=binding,candidates=rows,
                selected_candidate=selected,source_binding=source_binding(episode),
                selected_action=dict(source_config=rows[seed-41]["source_config"],candidate_case_id=rows[seed-41]["candidate_identity"]["episode_id"]),
                artifacts={name:sha256_file(search/name) for name in ["reference.npz","candidates.csv",*[r["candidate"]+".npz" for r in rows]]})
            freeze_json(search/"analysis.json",report)
            replay_rows=[dict(candidate=r["candidate"],objective=0.,reference_block_scores="[]",retained_arrays_exact=True) for r in rows]
            _write_csv(replay/"candidates.csv",replay_rows)
            repeated=dict(status="PASS_PFC_INTERVENTION_RETAINED_ARRAY_REPLAY",experiment_binding=binding,
                search_analysis_sha256=sha256_file(search/"analysis.json"),candidates=replay_rows,
                regenerated_unit_count_values=15680000000,per_bin_full_unit_counts_exact_replay_claimed=False,
                scientific_claim_qualified=False,native_execution_performed=False,retained_arrays_exact=True,
                selected_candidate=selected,artifacts={"candidates.csv":sha256_file(replay/"candidates.csv")})
            freeze_json(replay/"analysis.json",repeated)
            for i,path in enumerate((search,replay)):
                entries[i].append(dict(training_seed=seed,factual_episode_id=episode,path=str(relative),analysis_sha256=sha256_file(path/"analysis.json"),selected_candidate=selected))
                all_rows[i].extend(dict(training_seed=seed,factual_episode_id=episode,candidate=r["candidate"]) for r in rows)
    for i,root in enumerate(roots):
        _write_csv(root/"candidates.csv",all_rows[i])
        freeze_json(root/"analysis.json",dict(status=("COMPLETE_PFC_INTERVENTION_EXPERIMENT" if i==0 else "PASS_PFC_INTERVENTION_EXPERIMENT_REPLAY"),
            candidate_rows=90,seed_block_cases=9,generated_unit_count_values_across_completed_cases=141120000000,
            scientific_claim_qualified=False,native_execution_performed=False,per_bin_full_unit_counts_exact_replay_claimed=False,
            artifacts={"candidates.csv":sha256_file(root/"candidates.csv")},experiment_binding=experiment,cases=entries[i],saved_candidate_array_values_compared=344982330))
    # Reuse identical pure-score calculations across the repeated fixture payloads.
    original=outcomes._scores;cache={}
    def cached(arrays,target):
        key=(arrays["field_rate_hz"].shape,tuple(arrays["field_rate_hz"][:,0,0]))
        if key not in cache:cache[key]=original(arrays,target)
        return cache[key]
    monkeypatch.setattr(outcomes,"_scores",cached)
    return SimpleNamespace(args=(inputs,preparation,config_path,reference,*roots,tmp_path/"outcomes"),loads=native_loads,roots=roots)


def test_complete_join_reads_each_native_once_and_keeps_model_seeds_separate(complete_readback, monkeypatch):
    fixture=complete_readback
    report=outcomes.run_outcome_analysis(*fixture.args)
    assert report["native_source_episodes"]==30 and len(fixture.loads)==len(set(fixture.loads))==30
    assert (report["candidate_rows"],report["policy_rows"],report["program_rows"])==(90,3,10)
    assert report["observed_unit_bin_values_aggregated"]==2940000000
    assert report["newly_generated_samples"]==0 and not report["independent_intervention_qualified"]
    assert not report["scientific_claim_qualified"] and not report["native_future_noise_pairing_verified"]
    monkeypatch.setattr(outcomes,"_adapters",lambda *args:(_ for _ in ()).throw(AssertionError("repeated native read")))
    assert outcomes.run_outcome_analysis(*fixture.args)==report
    (fixture.args[1]/"source_snapshot.json").write_text('{"fixture_runtime":2}')
    with pytest.raises(ValueError,match="EXISTING_NATIVE_READBACK_BINDING_CHANGED"):
        outcomes.run_outcome_analysis(*fixture.args)


def test_changed_selected_native_program_is_rejected_before_readback(complete_readback):
    fixture=complete_readback;root=fixture.roots[0]
    summary=json.loads((root/"analysis.json").read_text());entry=summary["cases"][0];p=root/entry["path"]/"analysis.json"
    report=json.loads(p.read_text());report["selected_action"]["candidate_case_id"]="different_episode"
    p.write_text(json.dumps(report));entry["analysis_sha256"]=sha256_file(p)
    (root/"analysis.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):outcomes.run_outcome_analysis(*fixture.args)
    assert not fixture.loads
