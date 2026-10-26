"""Build source-bound cortical fields for one completed trajectory cohort split.

Raw all-unit events remain authoritative and available to observation models.
The field view uses the existing spatial observer without a time-by-unit array.
No normalization, threshold selection, or model fitting occurs here.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import yaml

from .nest_gpu_network import ticks
from .spatial_connectivity import native_delay_ticks
from .provenance import sha256_file
from .pulse_history_analysis import _verified_events
from .spatial_activity import Activity, analyze
from .stimulus_trajectory import SCHEMA


def verify_controls(events, indices, case, source, dt):
    """Independently enumerate the source circular TC mask and pulse arrivals."""
    net = source["resolved_source_network"]
    is_tc = events["populations"] == "TC"
    active = is_tc & (np.sum(events["positions_mm"] ** 2, axis=1) <= float(net["th_radius"]) ** 2)
    control = case["settings"]["pulse_control"]
    if set(control) != {"initial_emission_ms"}:
        raise ValueError("this acquisition requires one complete preloaded pulse program")
    emission = ticks(control["initial_emission_ms"], dt)
    delay = int(native_delay_ticks([net["th_delay_pulse_generator"]], dt)[0])
    expected_ticks = emission + delay
    expected_ids = np.repeat(events["unit_ids"][active], len(expected_ticks))
    expected_times = np.tile(expected_ticks, int(active.sum()))
    tc_events = is_tc[indices]
    actual_ids = events["spike_unit_ids"][tc_events]
    actual_times = events["spike_ticks"][tc_events]
    a = np.lexsort((actual_ids, actual_times))
    b = np.lexsort((expected_ids, expected_times))
    if (not np.array_equal(actual_ids[a], expected_ids[b])
            or not np.array_equal(actual_times[a], expected_times[b])):
        raise ValueError("native TC events differ from source mask and preloaded controls")
    return dict(generator_emission_ticks=emission.tolist(), tc_arrival_ticks=expected_ticks.tolist(),
                active_tc_units=int(active.sum()), tc_events=len(actual_ids),
                physical_generator_to_tc_delay_ms=float(net["th_delay_pulse_generator"]))


def build(acquisition: Path, graph: Path, output: Path, settings: Activity):
    design = json.loads((acquisition / "design.json").read_text())
    ledger = json.loads((acquisition / "acquisition.json").read_text())
    split = design["acquired_split"]
    if (design["schema"] != SCHEMA or ledger["status"] != "ACQUIRED"
            or len(design["cases"]) != design["expected_episodes"][split]
            or ledger["completed_episodes"] != len(design["cases"])):
        raise ValueError("complete predeclared split acquisition required")
    source = json.loads((graph / "manifest.json").read_text())
    graph_digest = sha256_file(graph / "manifest.json")
    if graph_digest != design["graph_manifest_sha256"]:
        raise ValueError("source graph differs from cohort declaration")
    if sha256_file(graph / "layout.npz") != source["layout_sha256"]:
        raise ValueError("source layout changed")
    with np.load(graph / "layout.npz", allow_pickle=False) as f:
        layout = {key: f[key] for key in ("unit_ids", "populations", "positions_mm")}
    binding = dict(schema="mesocircuit_trajectory_fields_v1", split=split, settings=asdict(settings),
                   design_sha256=sha256_file(acquisition / "design.json"),
                   graph_manifest_sha256=graph_digest,
                   raw_acquisition=str(acquisition.resolve()),
                   source_fidelity_qualified=False, formal_training_completed=False)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "binding.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("output belongs to a different cohort or observation map")
    path.write_text(json.dumps(binding, indent=2) + "\n")
    rows = []
    for case in design["cases"]:
        if case["split"] != split:
            raise ValueError("mixed cohort splits")
        episode = acquisition / case["case_id"]
        report, events, indices = _verified_events(episode, case, graph_digest)
        for key in layout:
            if not np.array_equal(events[key], layout[key]):
                raise ValueError(f"native event layout differs: {key}")
        controls = verify_controls(events, indices, case, source, report["dt_ms"])
        del events, indices
        target = output / case["case_id"]
        if target.exists():
            result = json.loads((target / "summary.json").read_text())
            if (result["status"] != "DESCRIBED" or result["settings"] != asdict(settings)
                    or result["episode_report_sha256"] != sha256_file(episode / "report.json")):
                raise ValueError("existing field view differs or is incomplete")
            for name, digest in result["artifacts"].items():
                if sha256_file(target / name) != digest:
                    raise ValueError("field view changed")
        else:
            result = analyze(graph, episode, target, settings)
            if result["status"] != "DESCRIBED":
                raise RuntimeError(result.get("error", "field observer failed"))
        if result["cortical_refractory_violations"]:
            raise ValueError("native cortical refractory violation")
        rows.append(dict(case_id=case["case_id"], block=case["block"], split=split,
                         template_id=case["template_id"], family=case["family"],
                         initial_state_seed=case["initial_state_seed"],
                         background_seed=case["background_seed"],
                         cortical_units=result["cortical_units"],
                         cortical_events=result["cortical_events_in_window"],
                         tc_events=controls["tc_events"],
                         refractory_violations=result["cortical_refractory_violations"],
                         native_duration_ms=report["duration_ms"],
                         native_simulation_seconds=report["simulation_seconds"],
                         native_elapsed_seconds=report["elapsed_seconds"],
                         field_summary_sha256=sha256_file(target / "summary.json"),
                         raw_report_sha256=sha256_file(episode / "report.json")))
        (target / "controls.json").write_text(json.dumps(controls, indent=2) + "\n")
        print(json.dumps(dict(status="BUILDING", completed=len(rows), expected=len(design["cases"]),
                              last_case=case["case_id"])), flush=True)
    with (output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = dict(binding, status="BUILT", completed_episodes=len(rows),
                  independent_blocks=len({r["block"] for r in rows}),
                  cortical_units=sorted({r["cortical_units"] for r in rows}),
                  cortical_events=sum(r["cortical_events"] for r in rows),
                  control_delivery_violations=0, refractory_violations=0,
                  episode_table_sha256=sha256_file(output / "episodes.csv"),
                  raw_events_retained=True, unit_count_matrix_materialized=False,
                  test_outcomes_accessed=split == "test")
    (output / "dataset.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.acquisition, args.graph, args.output,
                   Activity(**yaml.safe_load(args.config.read_text())))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
