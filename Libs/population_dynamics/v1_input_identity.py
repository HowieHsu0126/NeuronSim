"""Inspect original V1 input identities and timing before native execution.

This preserves raw events and edge attributes. It diagnoses the pinned author's
input mapping; it does not resample inputs or qualify a native simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import zipfile

import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_graph_identity import _finite_float32, _integers
from .v1_parameter_reconciliation import _delivered_json

INPUT_MEMBER = "GLIF_network2/input_dat.pkl"
AUTHOR_LGN_INPUTS = 17_400
AUTHOR_WINDOW_START = 1000.0
AUTHOR_WINDOW_DURATION = 1000.0
AUTHOR_WINDOW_DT = 1.0


def _spike_summary(ids: np.ndarray, trains, duration: float, dt: float) -> dict:
    if len(trains) != len(ids):
        raise ValueError("V1_INPUT_SPIKE_ID_LENGTH_DIFFERS")
    total = negative = non_grid = unordered = author_count = collisions = 0
    boundaries = {str(t): 0 for t in (0.0, AUTHOR_WINDOW_START,
                                     AUTHOR_WINDOW_START + AUTHOR_WINDOW_DURATION, duration)}
    minimum = maximum = None
    for train in trains:
        times = np.asarray(train)
        if times.ndim != 1 or times.dtype.kind not in "iuf" or not np.isfinite(times).all():
            raise ValueError("V1_INPUT_SPIKE_TIMES_INVALID")
        total += len(times)
        if not len(times):
            continue
        minimum = float(times.min()) if minimum is None else min(minimum, float(times.min()))
        maximum = float(times.max()) if maximum is None else max(maximum, float(times.max()))
        negative += int(np.count_nonzero(times < 0))
        unordered += int(np.any(np.diff(times.astype(np.float64)) < 0))
        non_grid += int(np.count_nonzero(times / dt != np.floor(times / dt)))
        for key in boundaries:
            boundaries[key] += int(np.count_nonzero(times == float(key)))
        selected = times[(times > AUTHOR_WINDOW_START)
                         & (times < AUTHOR_WINDOW_START + AUTHOR_WINDOW_DURATION)]
        bins = np.floor((selected - AUTHOR_WINDOW_START) / AUTHOR_WINDOW_DT)
        author_count += len(selected)
        collisions += len(bins) - len(np.unique(bins))
    return dict(raw_events=total, minimum_time=minimum, maximum_time=maximum,
                negative_time_events=negative, unordered_trains=unordered,
                events_not_exactly_on_source_config_dt=non_grid,
                exact_boundary_event_counts=boundaries,
                author_interval_events=author_count,
                author_within_unit_bin_collisions=collisions,
                multiplicity_preserved=True,
                rule="Strict author interval; flooring is diagnostic only. Raw spike times are unchanged.")


def inspect_inputs(raw, cortical_neurons: int, *, duration: float, dt: float) -> dict:
    if (not isinstance(cortical_neurons, int) or cortical_neurons <= 0
            or not np.isfinite([duration, dt]).all() or duration <= 0 or dt <= 0):
        raise ValueError("V1_INPUT_CLOCK_OR_CORTICAL_COUNT_INVALID")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("V1_INPUT_POPULATIONS_REQUIRED")
    populations = []
    concerns = []
    for number, population in enumerate(raw):
        if not isinstance(population, (list, tuple)) or len(population) != 2:
            raise ValueError("V1_INPUT_POPULATION_SCHEMA_INVALID")
        nodes, edges = population
        ids = _integers(nodes["ids"], "input_ids")
        n = len(ids)
        if not n or not np.array_equal(np.sort(ids), np.arange(n)):
            raise ValueError("V1_AUTHOR_INPUT_ID_INDEX_ASSUMPTION_NOT_MET")
        timing = _spike_summary(ids, nodes["spikes"], duration, dt)
        if not isinstance(edges, (list, tuple)) or not edges:
            raise ValueError("V1_INPUT_EDGE_GROUPS_REQUIRED")
        sources, targets, weights, delays, receptors = [], [], [], [], []
        for edge in edges:
            src = _integers(edge["source"], "input_edge_source")
            dst = _integers(edge["target"], "input_edge_target")
            size = len(src)
            if (len(dst) != size or np.any(src < 0) or np.any(src >= n)
                    or np.any(dst < 0) or np.any(dst >= cortical_neurons)):
                raise ValueError("V1_INPUT_EDGE_ENDPOINT_INVALID")
            params = edge["params"]
            receptor = np.asarray(params["receptor_type"])
            if receptor.shape != () or receptor.dtype.kind not in "iu" or not 1 <= int(receptor) <= 4:
                raise ValueError("V1_INPUT_RECEPTOR_INVALID")
            _finite_float32(params["weight"], (size,), "input_weight")
            delay = np.asarray(params["delay"])
            if delay.shape not in ((), (size,)):
                raise ValueError("V1_INPUT_DELAY_SHAPE_INVALID")
            _finite_float32(delay, delay.shape, "input_delay")
            if np.any(delay < 0):
                raise ValueError("V1_INPUT_NEGATIVE_DELAY")
            # Normalize only after integer and endpoint validation. Mixing uint64
            # with signed IDs/receptors can otherwise promote packed indices and
            # sort keys to float, including during concatenation across groups.
            sources.append(src.astype(np.int64, copy=False))
            targets.append(dst.astype(np.int64, copy=False))
            receptors.append(np.full(size, int(receptor), dtype=np.int64))
            weights.append(np.asarray(params["weight"], dtype=np.float64))
            delays.append(np.broadcast_to(delay, (size,)).astype(np.float64))
        src, dst, receptor, weight, delay = [np.concatenate(items) for items in
                                           (sources, targets, receptors, weights, delays)]
        packed = 4 * dst + receptor - 1
        count = len(src)
        if count:
            maximum_index = max(int(packed.max()), int(src.max())) + 1
            if int(packed.max()) * maximum_index + int(src.max()) > np.iinfo(np.int64).max:
                raise ValueError("V1_INPUT_AUTHOR_SORT_KEY_OVERFLOW")
            order = np.argsort(packed * maximum_index + src)
        else:
            order = np.array([], dtype=np.int64)
        mismatch = np.flatnonzero(delay != delay[order])
        examples = [
            dict(sorted_position=int(i), raw_edge_index=int(order[i]),
                 source=int(src[order[i]]), target=int(dst[order[i]]),
                 receptor=int(receptor[order[i]]), weight=float(weight[order[i]]),
                 author_returned_delay=float(delay[i]), edge_associated_delay=float(delay[order[i]]))
            for i in mismatch[:8]
        ]
        row = dict(population_index=number,
                   author_role_assumption=("LGN" if number == 0 else "background" if number == 1 else None),
                   input_units=n, ids_already_sorted=bool(np.array_equal(ids, np.arange(n))),
                   raw_edges=count, nonzero_edges=int(np.count_nonzero(weight)),
                   connected_input_units=len(np.unique(src[weight != 0])),
                   reached_cortical_units=len(np.unique(dst[weight != 0])),
                   delay_min=float(delay.min()) if count else None,
                   delay_max=float(delay.max()) if count else None,
                   delays_not_exactly_on_source_config_dt=int(np.count_nonzero(delay / dt != np.floor(delay / dt))),
                   spikes=timing, source_population_identity_qualified=False,
                   loader_delay_alignment=dict(mismatches=len(mismatch), examples=examples,
                       status="MISMATCH" if len(mismatch) else "ALIGNED_FOR_THIS_PAYLOAD"))
        if len(mismatch):
            concerns.append(f"population_{number}:AUTHOR_SORTED_DELAY_ASSOCIATION_DIFFERS")
        if number == 1:
            # Execute the pinned caller's array assignment, without modifying raw
            # edges. Multiple presynaptic rows can address one postsynaptic slot.
            slots, multiplicity = np.unique(packed, return_counts=True)
            repeated = slots[multiplicity > 1]
            assigned = np.zeros(4 * cortical_neurons, dtype=np.float32)
            assigned[packed[order]] = weight[order]
            row["author_flat_background_assignment"] = dict(
                rows=count, unique_target_receptor_slots=len(slots),
                overwritten_rows=count-len(slots), collision_slots=len(repeated),
                examples=[dict(target_receptor_slot=int(slot),
                               assigned_float32_weight=float(assigned[slot]),
                               raw_rows=int(np.count_nonzero(packed == slot)))
                          for slot in repeated[:8]],
                source_spike_times_and_delays_used_by_this_assignment=False)
            if len(repeated):
                concerns.append("background:FLAT_ASSIGNMENT_COLLAPSES_EDGE_ROWS")
        populations.append(row)
    if len(populations) != 2:
        concerns.append("AUTHOR_LGN_BACKGROUND_POPULATION_COUNT_DIFFERS")
    if populations[0]["input_units"] != AUTHOR_LGN_INPUTS:
        concerns.append("AUTHOR_UNREDUCED_LGN_INPUT_COUNT_DIFFERS")
    return dict(status="SOURCE_INPUT_INSPECTED", cortical_neurons=cortical_neurons,
                source_config_clock=dict(duration=duration, dt=dt),
                author_convenience_clock=dict(start=AUTHOR_WINDOW_START,
                    duration=AUTHOR_WINDOW_DURATION, dt=AUTHOR_WINDOW_DT),
                raw_inputs_modified=False, populations=populations,
                reconciliation_concerns=concerns,
                population_roles="Position-based assumptions in author code, not independently established identities.",
                original_pointnet_input_equivalence_qualified=False,
                native_dynamics_qualified=False, source_scientific_fidelity_qualified=False,
                production_episodes=0)


def inspect_archive(source_acquisition_archive: Path, graph_inspection_archive: Path,
                    point_config: Path, point_config_source: Path, output: Path) -> dict:
    source, delivered = _delivered_json(source_acquisition_archive, "asset/source.json")
    graph, graph_delivery = _delivered_json(graph_inspection_archive, "graph_identity.json")
    if (delivered["status"] != "ARCHIVED_V1_SOURCE_ASSET" or delivered["exit_code"] != 0
            or source["status"] != "SOURCE_ASSET_ACQUIRED" or source["asset_body_acquired"] is not True
            or graph_delivery["status"] not in ("INSPECTED", "INSPECTED_REQUIRES_RECONCILIATION")
            or graph["status"] != "SOURCE_GRAPH_INSPECTED"
            or graph["native_metadata_identity_verified"] is not True):
        raise ValueError("V1_COMPLETE_SOURCE_AND_GRAPH_INSPECTION_REQUIRED")
    identity = source["source"]
    if Path(identity["artifact"]).name != identity["artifact"]:
        raise ValueError("V1_INPUT_ARCHIVE_NAME_INVALID")
    archive = source_acquisition_archive / "asset" / identity["artifact"]
    if (archive.stat().st_size != identity["expected_bytes"]
            or identity["expected_bytes"] != graph["source_archive_bytes"]
            or identity["expected_md5"] != graph["source_archive_md5"]):
        raise ValueError("V1_INPUT_GRAPH_ARCHIVE_IDENTITY_DIFFERS")
    digest = hashlib.md5()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != identity["expected_md5"]:
        raise ValueError("V1_INPUT_COMPLETE_ARCHIVE_MD5_REQUIRED")
    config_receipt = json.loads(point_config_source.read_text())
    if sha256_file(point_config) != config_receipt["sha256"]:
        raise ValueError("V1_ORIGINAL_POINTNET_CONFIG_CHANGED")
    config = json.loads(point_config.read_text())
    if config["target_simulator"] != "NEST":
        raise ValueError("V1_ORIGINAL_POINTNET_CONFIG_REQUIRED")
    with tempfile.TemporaryDirectory(prefix="v1-input-") as temporary:
        payload = Path(temporary) / "input_dat.pkl"
        member_digest = hashlib.sha256()
        with zipfile.ZipFile(archive) as package:
            if package.namelist().count(INPUT_MEMBER) != 1:
                raise ValueError("V1_INPUT_MEMBER_NOT_UNIQUE")
            with package.open(INPUT_MEMBER) as src, payload.open("xb") as dst:
                for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    member_digest.update(block)
                    dst.write(block)
        with payload.open("rb") as stream:
            raw = pickle.load(stream)
        report = inspect_inputs(raw, int(graph["cortical_neurons"]),
                                duration=config["run"]["duration"], dt=config["run"]["dt"])
    report.update(source_acquisition_archive=str(source_acquisition_archive),
                  source_archive_md5=digest.hexdigest(),
                  source_receipt_sha256=sha256_file(source_acquisition_archive / "asset/source.json"),
                  graph_inspection_sha256=sha256_file(graph_inspection_archive / "graph_identity.json"),
                  point_config_sha256=sha256_file(point_config), input_member=INPUT_MEMBER,
                  input_member_sha256=member_digest.hexdigest())
    freeze_json(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-acquisition-archive", "graph-inspection-archive",
                 "point-config", "point-config-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    report = inspect_archive(**vars(parser.parse_args(argv)))
    print(json.dumps(report))
    # Inspection completion is separate from the reported reconciliation needs.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
