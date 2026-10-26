"""Reconcile declared full-population V1 driver seed vectors on saved episodes.

The source primary sampler uses Python random.seed/random. Its Bernoulli branch
leaves NumPy state unchanged. The outer NumPy RandomState chooses each primary
seed in source order. A single background generator uses the Poisson branch.
This verifies the implemented seed contract; it does not prove independent
biological units or scientific equivalence to an independent noise hypothesis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from .benchmark_tasks import condition_for_episode
from .circuit_campaign import campaign_config, episode_identities
from .provenance import freeze_json, sha256_file
from .v1_nest_input_replay import bound_json
from .v1_configured_inputs import validate_plan


def expected_bernoulli_seed_vector(starting_seed: int, units: int) -> np.ndarray:
    if type(starting_seed) is not int or not 0 <= starting_seed < 2**32 or type(units) is not int or units < 1:
        raise ValueError("V1_SEED_CONTRACT_INTEGER_DOMAIN_REQUIRED")
    return np.random.RandomState(starting_seed).randint(10000, size=units)


def verify_seed_plan(plan: dict, starting_seed: int, *, branch: str) -> dict:
    validate_plan(plan)
    n = len(plan["source_id"])
    if branch == "bernoulli":
        expected = expected_bernoulli_seed_vector(starting_seed, n)
        valid = np.array_equal(plan["source_primary_seed"], expected) and not plan["source_sampler_code"].any() and np.all(plan["source_poisson_seed"] == -1)
    elif branch == "single_shared_background_poisson":
        rng = np.random.RandomState(starting_seed)
        primary, poisson = int(rng.randint(10000)), int(rng.randint(10000))
        valid = n == 1 and plan["source_primary_seed"].tolist() == [primary] and plan["source_poisson_seed"].tolist() == [poisson] and plan["source_sampler_code"].tolist() == [1]
    else:
        raise ValueError("V1_SEED_CONTRACT_SUPPORTED_BRANCH_REQUIRED")
    if not valid:
        raise ValueError("V1_SEED_CONTRACT_ACTUAL_VECTOR_DIFFERS:" + branch)
    return dict(source_units=n, events=int(plan["source_counts"].sum()), seed_vector_exact=True, branch=branch,
                starting_seed=starting_seed, primary_seed_domain=[0,9999])


def load_bound_plan(source: Path, relative: str) -> dict:
    delivery = json.loads((source / "delivery.json").read_text())
    rows = [row for row in delivery["files"] if row["path"] == relative]
    path = source / relative
    if delivery.get("exit_code") != 0 or len(rows) != 1 or path.stat().st_size != rows[0]["bytes"] or sha256_file(path) != rows[0]["sha256"]:
        raise ValueError("V1_SEED_CONTRACT_BOUND_PLAN_CHANGED")
    with np.load(path, allow_pickle=False) as arrays:
        return {name:arrays[name] for name in arrays.files}


def analyze(config: Path, inputs: Path, project: Path, output: Path) -> dict:
    if not os.environ.get("SLURM_JOB_ID") or output.exists():
        raise ValueError("V1_SEED_CONTRACT_FRESH_SLURM_OUTPUT_REQUIRED")
    campaign = campaign_config(config)
    entries = json.loads(inputs.read_text())["blocks"]
    identities = {(x["circuit_index"],x["block_index"],x["condition_index"]):x for x in episode_identities(campaign,"v1_visual")}
    results = []
    for block in entries:
        path = Path(block["acceptance"])
        if not path.is_absolute():path = project / path
        accepted = json.loads(path.read_text())
        if (sha256_file(path) != block["acceptance_sha256"] or accepted["status"] != "PASS_AUTHENTICATED_COMPLETE_V1_DEVELOPMENT_BLOCK"
                or accepted["exit_code"] != 0 or accepted["complete_development_episodes"] != 10
                or accepted["circuit_index"] != block["circuit_index"] or accepted["block_index"] != block["block_index"]):
            raise ValueError("V1_SEED_CONTRACT_ACCEPTED_BLOCK_REQUIRED")
        reference, background_reference, rows = None, None, []
        for index, entry in enumerate(block["episodes"]):
            identity = identities[block["circuit_index"],block["block_index"],index]
            condition = condition_for_episode(campaign["scenarios"]["v1_visual"]["task_protocol"],"v1_visual",identity)
            source = Path(entry["source"])
            native = Path(entry["native_source"])
            bundle, delivery = bound_json(source,"visual_inputs/report.json")
            primary, _ = bound_json(native,"primary/report.json")
            if (accepted["episodes"][index]["native_job"] != entry["native_job"]
                    or delivery["exit_code"] != 0 or bundle["lgn_noise_seed"] != identity["stimulus_seed"]
                    or bundle["background_noise_seed"] != identity["background_seed"]
                    or primary["bindings"]["spike_inputs"]["delivery_sha256"] != sha256_file(source/"delivery.json")
                    or primary["bindings"]["spike_inputs"]["config"] != bundle["config"]
                    or bundle["config"]["stimulus"]["direction_deg"] != condition["direction_deg"]
                    or bundle["config"]["stimulus"]["contrast_percent"] != condition["contrast_percent"]):
                raise ValueError("V1_SEED_CONTRACT_SOURCE_IDENTITY_DIFFERS")
            lgn = load_bound_plan(source,"visual_inputs/lgn/trial_0000.npz")
            background = load_bound_plan(source,"visual_inputs/bkg/trial_0000.npz")
            if len(lgn["source_id"]) != 17400:
                raise ValueError("V1_SEED_CONTRACT_FULL_LGN_REQUIRED")
            lgn_result = verify_seed_plan(lgn,identity["stimulus_seed"],branch="bernoulli")
            background_result = verify_seed_plan(background,identity["background_seed"],branch="single_shared_background_poisson")
            if reference is not None:
                if not np.array_equal(reference["source_id"],lgn["source_id"]) or not np.array_equal(reference["source_primary_seed"],lgn["source_primary_seed"]):
                    raise ValueError("V1_SEED_CONTRACT_CONDITION_PAIRING_DIFFERS")
                if set(background) != set(background_reference) or not all(np.array_equal(background[k],background_reference[k]) for k in background):
                    raise ValueError("V1_SEED_CONTRACT_SHARED_BACKGROUND_DIFFERS")
            else:reference,background_reference = lgn,background
            rows.append(dict(condition_index=index,condition=condition,lgn=lgn_result,background=background_result,
                spike_delivery_sha256=sha256_file(source/"delivery.json"),native_report_sha256=sha256_file(native/"primary/report.json")))
        results.append(dict(circuit_index=block["circuit_index"],block_index=block["block_index"],conditions=rows,
            all_lgn_primary_seed_vectors_paired=True,all_background_events_paired=True))
    if len({(x["circuit_index"],x["block_index"]) for x in results}) != len(results):
        raise ValueError("V1_SEED_CONTRACT_DUPLICATE_BLOCK")
    report=dict(status="PASS_SAVED_FULL_V1_DRIVER_SEED_CONTRACT",blocks=results,complete_source_episodes_read=10*len(results),
        full_lgn_units=17400,shared_background_generators=1,actual_seed_vectors_replayed=True,
        scientific_qualification=False,probabilistic_independence_established=False,new_source_simulations=0,production_episodes=0,
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        interpretation="Exact implemented seed vectors, independent RNG namespaces and condition pairing. No significance, noise-law equivalence or effective-sample-size claim.")
    freeze_json(output,report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","inputs","project","output"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args(argv)
    report=analyze(args.config,args.inputs,args.project,args.output)
    print(json.dumps({k:v for k,v in report.items() if k!="blocks"}))


if __name__=="__main__":main()
