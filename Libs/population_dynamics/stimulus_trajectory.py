"""Prospectively split continuous stimulus trajectories and factorial controls.

All realizations sharing initial/background seeds stay in one cohort. The
resolved plan is written before acquisition; template repetitions and motifs
are nested observations, not independent replication blocks.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import random

import yaml

from .nest_gpu_network import ticks
from .nest_gpu_spatial import Execution
from .provenance import sha256_file
from .pulse_history import acquire_cases, conditions

SCHEMA = "mesocircuit_stimulus_trajectory_cohort_v1"
COHORTS = ("train", "calibration", "test")


def templates(config, dt):
    """Balance four local conditions at every familiar nominal motif position."""
    rng = random.Random(config["template_seed"])
    labels = list(itertools.product((False, True), (False, True)))
    rng.shuffle(labels)
    result = []
    offsets = config["conditioning_offsets_ms"]
    if not offsets or any(v >= 0 for v in offsets) or sorted(set(offsets)) != offsets:
        raise ValueError("ordered distinct negative local conditioning offsets required")
    for family in ("familiar", "calibration_timing", "test_timing"):
        specification = config["templates"][family]
        count = specification["count"]
        if type(count) is not int or count < 1 or (family == "familiar" and count != 4):
            raise ValueError("four familiar templates and positive timing-template counts required")
        centers = specification["focal_arrivals_ms"]
        if sorted(set(centers)) != centers or not centers:
            raise ValueError("increasing nominal motif times required")
        jitter_ticks = int(ticks([specification["jitter_ms"]], dt)[0])
        if jitter_ticks < 0:
            raise ValueError("nonnegative timing jitter required")
        for template in range(count):
            motifs, arrivals = [], []
            for position, center in enumerate(centers):
                focal_tick = int(ticks([center], dt)[0]) + rng.randint(-jitter_ticks, jitter_ticks)
                focal = float(focal_tick * dt)
                conditioning, present = labels[(position + template) % 4]
                local = [focal + v for v in offsets] if conditioning else []
                if present:
                    local.append(focal)
                if focal + offsets[0] < config["stimulus_free_until_ms"]:
                    raise ValueError("motif violates the initial stimulus-free interval")
                if motifs and focal + offsets[0] <= motifs[-1]["focal_arrival_ms"] + config["response_curve_ms"]:
                    raise ValueError("local motif response windows overlap")
                if focal + config["response_curve_ms"] >= config["duration_ms"]:
                    raise ValueError("incomplete trajectory response support")
                motifs.append(dict(position=position, focal_arrival_ms=focal,
                                   local_conditioning=conditioning, focal_present=present,
                                   pulse_arrivals_ms=local))
                arrivals.extend(local)
            positions = config["state_observation_positions"]
            if (not positions or any(type(i) is not int or i < 0 or i >= len(motifs) for i in positions)
                    or sorted(set(positions)) != positions):
                raise ValueError("valid ordered state-observation motif positions required")
            result.append(dict(template_id=f"{family}_{template}", family=family,
                               motifs=motifs, pulse_arrivals_ms=arrivals))
    if len({tuple(t["pulse_arrivals_ms"]) for t in result}) != len(result):
        raise ValueError("duplicate stimulus templates")
    return result


def build_plan(config, graph, library, native_library=None, expected_poisson_rng_family=None):
    config = copy.deepcopy(config)
    if (native_library is None) != (expected_poisson_rng_family is None):
        raise ValueError("native runtime binding requires its declared Poisson RNG family")
    if native_library is not None:
        config["execution"]["expected_poisson_rng_family"] = expected_poisson_rng_family
    if config["schema"] != SCHEMA or config["formal_training_seeds"] != [41, 42, 43]:
        raise ValueError("supported cohort schema and formal seeds 41/42/43 required")
    source = json.loads((graph / "manifest.json").read_text())
    dt = source["source_config"]["simulation"]["sim_resolution"]
    delay = source["resolved_source_network"]["th_delay_pulse_generator"]
    library = str(library.resolve())
    resolved_templates = templates(config, dt)
    blocks, cases = [], []
    seen_state, seen_noise = set(), set()
    forbidden = config["previous_development_seeds"]
    for split in COHORTS:
        partition = config["cohorts"][split]
        initial = partition["initial_state_seeds"]
        background = partition["background_seeds"]
        if len(initial) != len(background) or len(initial) != partition["expected_blocks"]:
            raise ValueError("cohort denominator differs from declared independent blocks")
        for state_seed, noise_seed in zip(initial, background):
            if (type(state_seed) is not int or type(noise_seed) is not int
                    or state_seed in seen_state or noise_seed in seen_noise
                    or state_seed in forbidden["initial"] or noise_seed in forbidden["background"]):
                raise ValueError("cohort/development realizations must be disjoint")
            seen_state.add(state_seed); seen_noise.add(noise_seed)
            block = len(blocks)
            blocks.append(dict(block=block, split=split, initial_state_seed=state_seed,
                               background_seed=noise_seed))
            for template in resolved_templates:
                if template["family"] not in ("familiar", split + "_timing"):
                    continue
                emission = [float(v - delay) for v in template["pulse_arrivals_ms"]]
                observation = [int(ticks([template["motifs"][i]["focal_arrival_ms"] - delay], dt)[0]) - 1
                               for i in config["state_observation_positions"]]
                settings = dict(config["execution"], duration_ms=config["duration_ms"],
                                initial_state_seed=state_seed, background_seed=noise_seed,
                                source_rng_library=library, state_observation_ticks=observation,
                                pulse_control={"initial_emission_ms": emission})
                Execution(**settings).validate()
                ticks([config["duration_ms"], *emission], dt)
                if any(v <= 0 for v in emission):
                    raise ValueError("pulse emissions must follow time zero")
                cases.append(dict(case_id=f"{split}_block_{block:03d}_{template['template_id']}",
                                  block=block, split=split, family="trajectory",
                                  template_id=template["template_id"],
                                  initial_state_seed=state_seed, background_seed=noise_seed,
                                  settings=settings))
        if split == "train":
            continue
        # Keep the existing eight-condition scalar effect independent of the
        # unmatched long trajectories. Reuse its timing/control implementation.
        factorial = copy.deepcopy(config["factorial"])
        factorial.update(schema="mesocircuit_pulse_history_development_v1",
                         initial_state_seeds=initial, background_seeds=background,
                         expected_episodes=len(initial) * 8, execution=config["execution"],
                         capture_focal_state=True)
        global_blocks = {r["initial_state_seed"]: r["block"] for r in blocks}
        for row in conditions(factorial, source, library):
            row["block"] = global_blocks[row["initial_state_seed"]]
            row.update(split=split, family="factorial", template_id=None)
            row["case_id"] = (f"{split}_block_{row['block']:03d}_focal_{row['focal_arrival_ms']:g}"
                              f"_history_{int(row['history'])}_pulse_{int(row['present'])}")
            cases.append(row)
        # A delayed first pulse and its omission separate elapsed time from
        # cumulative prior stimulation at the later part of a long episode.
        delayed = config["delayed_first_arrival_ms"]
        if delayed + config["response_curve_ms"] >= config["duration_ms"]:
            raise ValueError("delayed-first-pulse response outside support")
        for state_seed, noise_seed in zip(initial, background):
            block = global_blocks[state_seed]
            for present in (False, True):
                settings = dict(config["execution"], duration_ms=config["duration_ms"],
                                initial_state_seed=state_seed, background_seed=noise_seed,
                                source_rng_library=library,
                                state_observation_ticks=[int(ticks([delayed - delay], dt)[0]) - 1],
                                pulse_control={"initial_emission_ms": [delayed - delay] if present else []})
                Execution(**settings).validate()
                cases.append(dict(case_id=f"{split}_block_{block:03d}_delayed_first_pulse_{int(present)}",
                                  block=block, split=split, family="delayed_first", template_id=None,
                                  focal_arrival_ms=delayed, history=False, present=present,
                                  initial_state_seed=state_seed, background_seed=noise_seed,
                                  settings=settings))
    counts = {s: sum(c["split"] == s for c in cases) for s in COHORTS}
    if counts != config["expected_episodes"]:
        raise ValueError(f"resolved cohort episode denominator differs: {counts}")
    random.Random(config["run_order_seed"]).shuffle(cases)
    result = dict(schema=SCHEMA, config=config, blocks=blocks, templates=resolved_templates, cases=cases,
                graph_manifest_sha256=sha256_file(graph / "manifest.json"),
                source_rng_sha256=sha256_file(Path(library)), expected_episodes=counts,
                independent_replication="joint initial/background block conditional on one graph",
                source_future_noise_pairing_verified=False, source_fidelity_qualified=False,
                formal_evaluation_protocol_frozen=False, formal_training_completed=False,
                formal_training_seeds=[41, 42, 43])
    if native_library is not None:
        result["native_library_sha256"] = sha256_file(Path(native_library))
    return result


def freeze(config_path, graph, library, output, native_library=None, expected_poisson_rng_family=None):
    result = build_plan(yaml.safe_load(config_path.read_text()), graph, library,
                        native_library, expected_poisson_rng_family)
    text = json.dumps(result, indent=2) + "\n"
    if output.exists() and output.read_text() != text:
        raise ValueError("cannot overwrite a different prospective cohort")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return result


def acquire(design, split, graph, library, output, resume=False):
    plan = json.loads(design.read_text())
    if (plan["schema"] != SCHEMA or plan["graph_manifest_sha256"] != sha256_file(graph / "manifest.json")
            or plan["source_rng_sha256"] != sha256_file(library)):
        raise ValueError("prospective cohort source binding changed")
    if split not in COHORTS:
        raise ValueError("unknown acquisition split")
    if ("native_library_sha256" in plan
            and sha256_file(Path(os.environ["NESTGPU_LIB"])) != plan["native_library_sha256"]):
        raise ValueError("prospective cohort native runtime changed")
    cases = [copy.deepcopy(c) for c in plan["cases"] if c["split"] == split]
    if len(cases) != plan["expected_episodes"][split]:
        raise ValueError("prospective split denominator differs")
    for case in cases:
        case["settings"]["source_rng_library"] = str(library.resolve())
    declaration = {k: v for k, v in plan.items() if k != "cases"}
    declaration.update(cases=cases, acquired_split=split, prospective_design_sha256=sha256_file(design))
    return acquire_cases(graph, output, declaration, resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "acquire"):
        p = sub.add_parser(name)
        p.add_argument("--graph", type=Path, required=True)
        p.add_argument("--source-rng-library", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "freeze":
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--native-library", type=Path)
            p.add_argument("--expected-poisson-rng-family", choices=["per_edge_xorwow", "per_edge_philox4_32_10"])
        else:
            p.add_argument("--design", type=Path, required=True)
            p.add_argument("--split", choices=COHORTS, required=True)
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.config, args.graph, args.source_rng_library, args.output,
                        args.native_library, args.expected_poisson_rng_family)
        print(json.dumps(dict(status="COHORT_DECLARED", expected_episodes=result["expected_episodes"])))
    else:
        raise SystemExit(acquire(args.design, args.split, args.graph, args.source_rng_library, args.output, args.resume))


if __name__ == "__main__":
    main()
