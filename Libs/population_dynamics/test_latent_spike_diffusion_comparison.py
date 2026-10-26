"""Complete paired denominator and independent-delivery rejection checks."""
import csv
import itertools
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import latent_spike_diffusion_comparison as comparison
from .test_braid_population_comparison import fixture,delivery,write_json,write_rows
from .provenance import sha256_file


@pytest.fixture
def inputs(tmp_path):
    fixture(tmp_path)
    ldns=tmp_path/"ldns"
    for seed in (41,42,43):
        cell=ldns/f"seed_{seed}";pop=tmp_path/"population"/f"seed_{seed}"
        (cell/"forecasts").mkdir(parents=True)
        pe=json.loads((pop/"forecasts/evaluation.json").read_text())
        pr=json.loads((pop/"training/run.json").read_text())
        source={k:v for k,v in pr["binding"].items() if k!="config"}
        write_json(cell/"training/run.json",dict(status="FITTED_DEVELOPMENT",training_seed=seed,stage="complete",
            units=100000,elapsed_seconds=200.,parameter_count=20))
        fields=np.array(["field_A","field_B"])
        with (pop/"forecasts/scores.csv").open() as stream:rows=list(csv.DictReader(stream))
        for row in rows:
            row["template_id"]=row["case_id"].rsplit("_",2)[1]+"_"+row["case_id"].rsplit("_",1)[1]
            for metric in comparison.METRICS:row[metric]=(-.01 if "mse" in metric else 1.)
            row["observed_spikes_per_unit"] = 2 * (float(row["horizon_ms"]) / 5.) / 100000
            if float(row["horizon_ms"])==20.:
                name=f"{row['case_id']}_origin_{float(row['origin_ms']):g}.npz"
                with np.load(pop/"forecasts"/name,allow_pickle=False) as a:
                    np.savez(cell/"forecasts"/name,sampled_field_counts=a["sampled_field_counts"],
                        observed_field_counts=a["observed_field_counts"],field_ids=fields,unit_ids=np.arange(100000))
        write_rows(pop/"forecasts/scores.csv",rows)
        pe["scores_sha256"]=sha256_file(pop/"forecasts/scores.csv")
        write_json(pop/"forecasts/evaluation.json",pe)
        replay=json.loads((pop/"replay.json").read_text());replay["evaluation_sha256"]=sha256_file(pop/"forecasts/evaluation.json")
        write_json(pop/"replay.json",replay);delivery(pop,seed)
        for row in rows:row["method"]="ldns"
        write_rows(cell/"forecasts/scores.csv",rows)
        le=dict(status="EVALUATED_DEVELOPMENT",score_rows=144,complete_origins=48,
            binding=dict(cohort=source,draws=16,fitting_field_scale_sha256=sha256_file(pop/"forecasts/fitting_field_scale.npz")),
            hardware=dict(device="cuda"),elapsed_seconds=100.,
            artifacts={"scores.csv":sha256_file(cell/"forecasts/scores.csv")})
        cost_rows=[dict(case_id=f"train_block_{block:03d}_familiar_{template}",origin_ms=origin,
            episode_loading_seconds=.5,inference_seconds=.1,complete_unit_generation_seconds=.2,
            projection_and_scoring_seconds=.3,serialization_and_hash_seconds=.1,
            origin_total_seconds=1.,generated_count_values=160000000,peak_rss_bytes=1000000)
            for block,template,origin in itertools.product(range(14,18),range(4),(1000.,3000.,5000.))]
        write_rows(cell/"forecasts/costs.csv",cost_rows)
        le["artifacts"]["costs.csv"]=sha256_file(cell/"forecasts/costs.csv")
        write_json(cell/"forecasts/evaluation.json",le)
        write_json(cell/"forecasts/replay.json",dict(status="PASS",evaluation_sha256=sha256_file(cell/"forecasts/evaluation.json"),
            all_origin_count_digests_fields_latents_metrics_exact=True))
        delivery(cell,seed)
    return ldns,tmp_path/"population",tmp_path/"comparison"


def test_complete_all_blocks_and_cost_limits(inputs):
    # Reordered measured rows retain the same case/origin charges through the CLI.
    cell=inputs[0]/"seed_41";cost=cell/"forecasts/costs.csv"
    with cost.open() as stream:rows=list(csv.DictReader(stream))
    write_rows(cost,list(reversed(rows)))
    evaluation=json.loads((cell/"forecasts/evaluation.json").read_text())
    evaluation["artifacts"]["costs.csv"]=sha256_file(cost)
    write_json(cell/"forecasts/evaluation.json",evaluation)
    replay=json.loads((cell/"forecasts/replay.json").read_text())
    replay["evaluation_sha256"]=sha256_file(cell/"forecasts/evaluation.json")
    write_json(cell/"forecasts/replay.json",replay);delivery(cell,41)
    child=subprocess.run([sys.executable,"-B","-m","Libs.population_dynamics.latent_spike_diffusion_comparison",
        "--ldns-root",str(inputs[0]),"--population-root",str(inputs[1]),"--output",str(inputs[2])],
        capture_output=True,text=True)
    assert child.returncode==0,child.stderr
    report=json.loads((inputs[2]/"comparison.json").read_text())
    assert report["matched_rows"]==432 and report["block_rows"]==36
    assert not report["same_training_budget"] and not report["same_hardware"]
    with (inputs[2]/"joint_comparison_by_block.csv").open() as stream:rows=list(csv.DictReader(stream))
    assert {int(r["block"]) for r in rows}=={14,15,16,17}
    assert all(float(r["population_minus_ldns_unit_bin_mean_mse_corrected"])==0 for r in rows)
    with (inputs[2]/"costs_by_origin.csv").open() as stream:costs=list(csv.DictReader(stream))
    assert len(costs)==144
    assert sum(float(row["ldns_charged_episode_loading_seconds"] or 0) for row in costs)==24.
    assert sum(float(row["ldns_accounted_origin_total_seconds"]) for row in costs)==168.
    assert all(row["population_accounted_origin_total_seconds"]=="" for row in costs)
    assert report["population_measured_evaluation_cost_seeds"]==[]
    assert report["paired_cost_origins"]==144 and report["speedup_qualified"] is False


def test_population_budget_label_uses_recorded_continuation_and_selection():
    run = dict(binding=dict(config=dict(maximum_epochs=20)), effective_maximum_epochs=80,
               completed_epochs=36, best_epoch=31, stop_reason="validation_patience",
               continuation=dict(schema="neural_population_trajectory_budget_continuation_v1",
                                 base_maximum_epochs=20, total_epochs=80))
    assert comparison._population_budget_label(run) == "80epochs_maximum;36completed;best_epoch31;base20"
    with pytest.raises(ValueError, match="bound continuation"):
        comparison._population_budget_label(dict(run, continuation=None))
    with pytest.raises(ValueError, match="ceiling"):
        comparison._population_budget_label(dict(run, completed_epochs=81))
    legacy = dict(binding=dict(config={}), completed_epochs=20, best_epoch=19,
                  stop_reason="epoch_budget")
    assert comparison._population_budget_label(legacy) == "20epochs_maximum;20completed;best_epoch19;base20"


def test_population_budget_label_is_written_to_complete_comparison(inputs):
    for seed in (41, 42, 43):
        cell = inputs[1] / f"seed_{seed}"
        path = cell / "training/run.json"
        run = json.loads(path.read_text())
        run["binding"]["config"]["maximum_epochs"] = 80
        run.update(effective_maximum_epochs=80, completed_epochs=80, best_epoch=79)
        write_json(path, run)
        delivery(cell, seed)
    comparison.compare(*inputs)
    with (inputs[2] / "training_costs.csv").open() as stream:
        population = [row for row in csv.DictReader(stream) if row["method"] == "population"]
    assert len(population) == 3
    assert {row["training_budget"] for row in population} == {"80epochs_maximum;80completed;best_epoch79;base80"}


def test_missing_seed_and_changed_truth_reject_before_output(inputs):
    cell=inputs[0]/"seed_43";proof=cell/"delivery.json";saved=proof.read_bytes();proof.unlink()
    with pytest.raises(FileNotFoundError):comparison.compare(*inputs)
    assert not inputs[2].exists();proof.write_bytes(saved)
    path=inputs[0]/"seed_41/forecasts/train_block_014_familiar_0_origin_1000.npz"
    with np.load(path,allow_pickle=False) as a:payload={k:a[k].copy() for k in a.files}
    payload["observed_field_counts"][0,0]+=1;np.savez(path,**payload);delivery(inputs[0]/"seed_41",41)
    with pytest.raises(ValueError,match="source truth"):comparison.compare(*inputs)
    assert not inputs[2].exists()


@pytest.mark.parametrize("failure",["duplicate","foreign","missing","nonfinite","negative","count_values",
                                    "short_origin_total","short_loop_total","loading_mismatch","digest"])
def test_invalid_ldns_costs_reject_before_output(inputs,failure):
    cell=inputs[0]/"seed_41";cost=cell/"forecasts/costs.csv"
    with cost.open() as stream:rows=list(csv.DictReader(stream))
    evaluation=json.loads((cell/"forecasts/evaluation.json").read_text())
    if failure=="duplicate":rows[-1]=rows[0]
    elif failure=="foreign":rows[0]["case_id"]="foreign"
    elif failure=="missing":rows.pop()
    elif failure=="nonfinite":rows[0]["inference_seconds"]="nan"
    elif failure=="negative":rows[0]["inference_seconds"]=-1
    elif failure=="count_values":rows[0]["generated_count_values"]=10000000
    elif failure=="short_origin_total":rows[0]["origin_total_seconds"]=.1
    elif failure=="short_loop_total":evaluation["elapsed_seconds"]=48.
    elif failure=="loading_mismatch":rows[1]["episode_loading_seconds"]=.6
    write_rows(cost,rows)
    if failure!="digest":evaluation["artifacts"]["costs.csv"]=sha256_file(cost)
    else:cost.write_text(cost.read_text()+"\n")
    write_json(cell/"forecasts/evaluation.json",evaluation)
    replay=json.loads((cell/"forecasts/replay.json").read_text())
    replay["evaluation_sha256"]=sha256_file(cell/"forecasts/evaluation.json")
    write_json(cell/"forecasts/replay.json",replay);delivery(cell,41)
    with pytest.raises(ValueError):comparison.compare(*inputs)
    assert not inputs[2].exists()


def test_measured_population_costs_join_once_and_preserve_missing_setup(inputs):
    from .test_population_forecast_analysis import _measured_report
    pop=inputs[1]/"seed_41";ldns=inputs[0]/"seed_41"
    run=json.loads((pop/"training/run.json").read_text())
    evaluation=json.loads((pop/"forecasts/evaluation.json").read_text())
    scores=comparison._score_table(pop/"forecasts/scores.csv",41,"population")
    validation=[dict(block=block,case_id=f"train_block_{block:03d}_familiar_{template}")
                for block,template in itertools.product(range(14,18),range(4))]
    run["binding"]["config"]["training_sequence"]="complete_six_second_episode"
    run["binding"]["native_source_identity"]["dt_ms"]=.1
    run["binding"]["episode_bindings"]={case["case_id"]:{} for case in validation}
    source=SimpleNamespace(validation=validation,bin_ms=5.,unit_ids=range(100000),
        field_ids=["field_A","field_B"],source_identity=dict(dt_ms=.1),
        truth={case["case_id"]:range(1200) for case in validation})
    numeric=[dict(case_id=case,block=block,origin_ms=origin,horizon_ms=horizon,
        inference_seconds=1.,full_unit_generation_seconds=2.,field_projection_seconds=3.,
        peak_gpu_allocated_bytes=None) for block,case,origin,horizon in scores]
    evaluation=_measured_report(source,evaluation["configuration"],numeric,evaluation)
    evaluation["configuration"]["field_metric_dtype"]="float64"
    for origin in evaluation["origin_costs"]:origin["origin_total_seconds"]=7.
    evaluation["elapsed_seconds"]=400.;evaluation["function_total_seconds"]=402.
    with (pop/"forecasts/scores.csv").open() as stream:score_rows=list(csv.DictReader(stream))
    for row in score_rows:row["peak_gpu_allocated_bytes"]=""
    write_rows(pop/"forecasts/scores.csv",score_rows)
    evaluation["scores_sha256"]=sha256_file(pop/"forecasts/scores.csv")
    write_json(pop/"training/run.json",run);write_json(pop/"forecasts/evaluation.json",evaluation)
    replay=json.loads((pop/"replay.json").read_text());replay["evaluation_sha256"]=sha256_file(pop/"forecasts/evaluation.json")
    write_json(pop/"replay.json",replay);delivery(pop,41)
    le=json.loads((ldns/"forecasts/evaluation.json").read_text())
    le["binding"]["cohort"]={k:v for k,v in run["binding"].items() if k!="config"}
    write_json(ldns/"forecasts/evaluation.json",le)
    replay=json.loads((ldns/"forecasts/replay.json").read_text());replay["evaluation_sha256"]=sha256_file(ldns/"forecasts/evaluation.json")
    write_json(ldns/"forecasts/replay.json",replay);delivery(ldns,41)
    report=comparison.compare(*inputs)
    assert report["population_measured_evaluation_cost_seeds"]==[41]
    with (inputs[2]/"costs_by_origin.csv").open() as stream:rows=list(csv.DictReader(stream))
    measured=[row for row in rows if int(row["seed"])==41]
    assert sum(float(row["population_episode_preparation_seconds"] or 0) for row in measured)==pytest.approx(6.4)
    assert sum(float(row["population_accounted_origin_total_seconds"]) for row in measured)==pytest.approx(342.4)
    assert all(row["population_accounted_origin_total_seconds"]=="" for row in rows if int(row["seed"])!=41)
    with (inputs[2]/"evaluation_costs.csv").open() as stream:costs=list(csv.DictReader(stream))
    assert all(row["setup_seconds"]=="" for row in costs if row["method"]=="ldns")
    assert all("unmeasured" in row["cost_scope"] for row in costs if row["method"]=="ldns")

@pytest.mark.parametrize("mixture", [False, True])
def test_dedicated_ldns_prior_comparison_retains_method_budget_and_parent_cost(inputs, mixture):
    from .test_braid_population_comparison import prior_population_fixture
    prior_population_fixture(inputs[1].parent)
    if mixture:
        from .test_population_forecast_analysis import _set_prior_fixture_profile
        for seed in (41, 42, 43):
            _set_prior_fixture_profile(inputs[1] / f"seed_{seed}", mixture=True)
    method = "population_recurrent_predictive_mixture" if mixture else "population_recurrent_prior_refit"
    objective = "fixed_posterior_joint_predictive_mixture_kl" if mixture else "fixed_posterior_multistep_prior_kl"
    prefix = "predictive_mixture;" if mixture else "prior_refit;"
    report = comparison.compare(*inputs)
    assert report["candidate_method"] == method
    assert report["matched_rows"] == 432 and report["speedup_qualified"] is False
    assert report["population_training_bindings"]["42"]["prior_refit_lineage"]["parent_completed_epochs"] == 80
    with (inputs[2] / "training_costs.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["method"] == method]
    assert len(rows) == 3 and all(float(row["recorded_training_seconds"]) == 190. for row in rows)
    assert all(float(row["parent_training_elapsed_seconds"]) == 180. for row in rows)
    assert all(prefix in row["training_budget"] and "parent80completed" in row["training_budget"] for row in rows)
    assert all(row["training_objective"] == objective for row in rows)
    if mixture:
        assert all(int(row["prediction_particles"]) == 8 and int(row["posterior_target_draws"]) == 4 for row in rows)
    with (inputs[2] / "joint_comparison_all_seeds.csv").open() as stream:
        scores = list(csv.DictReader(stream))
    assert len(scores) == 432 and all(row["candidate_method"] == report["candidate_method"] for row in scores)
    assert all(float(row["population_minus_ldns_observed_spikes_per_unit"]) == 0. for row in scores)


def test_observed_total_check_reuses_frozen_replay_tolerance_for_native_rounding():
    from .population_forecast_replay import METRIC_ATOL, METRIC_RTOL
    observed = np.array([[4447], [0], [0], [0]], dtype=np.int64)
    reference = comparison._validate_observed_total_scores(
        0.04447000101208687, 0.04446999728679657, observed, 4, 100000)
    assert reference == 4447 / 100000
    assert comparison.METRIC_ATOL == METRIC_ATOL and comparison.METRIC_RTOL == METRIC_RTOL
    with pytest.raises(ValueError, match="matched observed unit count totals"):
        comparison._validate_observed_total_scores(0.04447, 0.044479, observed, 4, 100000)


@pytest.mark.parametrize("genuine_mismatch", [False, True])
def test_saved_observed_total_rounding_retained_and_wrong_total_rejected(inputs, genuine_mismatch):
    cell = inputs[1] / "seed_41"
    scores_path = cell / "forecasts/scores.csv"
    with scores_path.open() as stream:
        rows = list(csv.DictReader(stream))
    original = float(rows[0]["observed_spikes_per_unit"])
    changed = (original + .001 if genuine_mismatch else
               float(np.nextafter(np.float32(original), np.float32(np.inf))))
    rows[0]["observed_spikes_per_unit"] = changed
    write_rows(scores_path, rows)
    evaluation_path = cell / "forecasts/evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["scores_sha256"] = sha256_file(scores_path)
    write_json(evaluation_path, evaluation)
    replay = json.loads((cell / "replay.json").read_text())
    replay["evaluation_sha256"] = sha256_file(evaluation_path)
    write_json(cell / "replay.json", replay)
    delivery(cell, 41)
    if genuine_mismatch:
        with pytest.raises(ValueError, match="matched observed unit count totals"):
            comparison.compare(*inputs)
        assert not inputs[2].exists()
    else:
        report = comparison.compare(*inputs)
        with (inputs[2] / "joint_comparison_all_seeds.csv").open() as stream:
            actual = next(csv.DictReader(stream))
        assert float(actual["population_observed_spikes_per_unit"]) == changed
        assert float(actual["ldns_observed_spikes_per_unit"]) == original
        assert float(actual["population_minus_ldns_observed_spikes_per_unit"]) == changed - original != 0
        assert report["observed_total_validation"]["saved_scalar_differences_retained"] is True
