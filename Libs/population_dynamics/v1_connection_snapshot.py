"""Recover complete effective connection tuples from a verified native reference.

The pinned BMTK code computes its original connection batches. This transport
writes those batches instead of constructing NEST cortical synapses. It calls
source module initialization but never advances simulator time. Every stored
batch must reproduce the digest captured during the completed native run.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import h5py
import numpy as np

from . import v1_pointnet_native as native
from .provenance import freeze_json, sha256_file
from .v1_glif_component_reference import require

FIELDS = ("source", "target", "receptor_type", "weight", "requested_delay", "delay")
INTEGER_FIELDS = frozenset(FIELDS[:3])
CHUNK_ROWS = 65536


def update_digest(digest, population: str, values: dict) -> None:
    digest.update(population.encode())
    for key in FIELDS:
        array = values[key]
        digest.update(key.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())


class ConnectionSnapshot:
    def __init__(self, path: Path, expected: dict, digest: str):
        self.path, self.expected, self.expected_digest = path, expected, digest
        self.rows = {name: 0 for name in expected}
        self.digest = hashlib.sha256()
        self.handle = h5py.File(path, "x")
        self.handle.attrs["schema"] = "native_connection_batches_v1"
        for field in FIELDS:
            self.handle.create_dataset(field, shape=(0,), maxshape=(None,),
                dtype=np.int64 if field in INTEGER_FIELDS else np.float64,
                chunks=(CHUNK_ROWS,), compression="lzf")
        self.batches = []

    def append(self, population: str, values: dict) -> None:
        require(population in self.rows and tuple(values) == FIELDS, "SNAPSHOT_BATCH_FIELDS")
        count = len(values["source"])
        require(count > 0 and all(x.shape == (count,) and np.isfinite(x).all() for x in values.values()),
                "SNAPSHOT_BATCH_SHAPE")
        for field in FIELDS:
            require(values[field].dtype == (np.int64 if field in INTEGER_FIELDS else np.float64),
                    "SNAPSHOT_BATCH_DTYPE:" + field)
        require(np.all(values["delay"] > 0) and np.all((values["receptor_type"] >= 1)
                & (values["receptor_type"] <= 4)), "SNAPSHOT_BATCH_SYNAPSIS")
        start = len(self.handle["source"])
        for field in FIELDS:
            target = self.handle[field]
            target.resize((start+count,))
            target[start:start+count] = values[field]
        self.batches.append((population, start, count))
        self.rows[population] += count
        update_digest(self.digest, population, values)

    @contextmanager
    def capture(self, nest, graph):
        original = graph._nest_connect
        def emit(source, target, conn_spec="one_to_one", syn_spec=None):
            native.execution_clock(nest, native.V1_RESOLUTION_MS)
            require(conn_spec == "one_to_one" and syn_spec is not None
                    and syn_spec["synapse_model"] == "static_synapse", "SNAPSHOT_SOURCE_CONNECTION_MODE")
            count = len(source)
            src = native.native_vector(source, count, "source", True)
            dst = native.native_vector(target, count, "target", True)
            pool = graph.gid_map if int(src[0]) in graph.gid_map._gid2pop_id else graph._virtual_gids
            _, source_pop = pool.get_node_ids(src)
            _, target_pop = graph.gid_map.get_node_ids(dst)
            names = set(source_pop)
            require(len(names) == 1 and np.all(target_pop == "v1"), "SNAPSHOT_SOURCE_POPULATIONS")
            values = dict(source=src, target=dst,
                receptor_type=native.native_vector(syn_spec["receptor_type"], count, "receptor_type", True),
                weight=native.native_vector(syn_spec["weight"], count, "weight"),
                requested_delay=native.native_vector(syn_spec["delay"], count, "delay"))
            values["delay"] = np.floor(values["requested_delay"]/native.V1_RESOLUTION_MS+.5)*native.V1_RESOLUTION_MS
            self.append(next(iter(names)), values)
            # This export transport does not invoke the NEST connection sink.
            return None
        graph._nest_connect = emit
        try:
            yield self
        finally:
            graph._nest_connect = original

    def verify(self, cortical_ids=None) -> dict:
        require(self.rows == {name: row["edge_rows"] for name, row in self.expected.items()},
                "SNAPSHOT_SOURCE_ROW_COUNTS_DIFFER")
        require(self.digest.hexdigest() == self.expected_digest, "SNAPSHOT_NATIVE_BATCH_DIGEST_DIFFERS")
        return dict(source_batch_sha256=self.digest.hexdigest(), rows_by_source_population=self.rows,
                    transport="source_tuple_export_without_native_cortical_synapses")

    def close(self):
        if self.handle:
            self.handle.create_dataset("batch_population", data=np.asarray([x[0] for x in self.batches], dtype="S3"))
            self.handle.create_dataset("batch_start", data=np.asarray([x[1] for x in self.batches], dtype=np.int64))
            self.handle.create_dataset("batch_count", data=np.asarray([x[2] for x in self.batches], dtype=np.int64))
            self.handle.close()
            self.handle = None


def verify_snapshot(path: Path, expected_digest: str, expected_rows: int) -> dict:
    digest = hashlib.sha256()
    cursor, populations = 0, {}
    with h5py.File(path, "r") as file:
        require(file.attrs["schema"] == "native_connection_batches_v1", "SNAPSHOT_SCHEMA")
        require(all(file[field].shape == (expected_rows,) for field in FIELDS), "SNAPSHOT_STORED_ROW_COUNT")
        metadata = (file["batch_population"][:], file["batch_start"][:], file["batch_count"][:])
        require(len({len(x) for x in metadata}) == 1, "SNAPSHOT_BATCH_METADATA_SHAPE")
        for raw, start, count in zip(*metadata):
            name = raw.decode()
            start, count = int(start), int(count)
            require(start == cursor and count > 0 and start+count <= expected_rows
                    and name in ("v1", "lgn", "bkg"), "SNAPSHOT_STORED_BATCH_RANGE")
            digest.update(name.encode())
            for field in FIELDS:
                require(file[field].dtype == (np.int64 if field in INTEGER_FIELDS else np.float64),
                        "SNAPSHOT_STORED_DTYPE")
                digest.update(field.encode())
                digest.update(np.asarray([count], dtype=np.int64).tobytes())
                for lo in range(start, start+count, CHUNK_ROWS):
                    digest.update(file[field][lo:min(lo+CHUNK_ROWS,start+count)].tobytes())
            populations[name] = populations.get(name, 0)+count
            cursor += count
        require(cursor == expected_rows and digest.hexdigest() == expected_digest,
                "SNAPSHOT_STORED_NATIVE_DIGEST_DIFFERS")
    return dict(rows=cursor, batches=len(metadata[0]), rows_by_source_population=populations,
                source_batch_sha256=digest.hexdigest())


def export_snapshot(archive: Path, output: Path) -> dict:
    require(bool(os.environ.get("SLURM_JOB_ID")) and not output.exists(), "SNAPSHOT_FRESH_COMPUTE_OUTPUT_REQUIRED")
    output.mkdir(parents=True)
    report = dict(status="FAILED_NATIVE_CONNECTION_SNAPSHOT", job_id=os.environ["SLURM_JOB_ID"],
        benchmark_episodes=0, production_episodes=0, native_simulation_calls=0,
        native_cortical_synapses_constructed=False, scientific_qualification=False,
        code_sha256=sha256_file(Path(__file__)))
    started, writer = time.monotonic(), None
    try:
        relatives = ("original_config.yaml", "primary/report.json", "primary/native/parameters.json",
                     "primary/native/cortical_state.npz", "primary/native/input_events.npz")
        delivery, binding = native.bound_files(archive, relatives)
        original = json.loads((archive/"primary/report.json").read_text())
        require(original["exit_code"] == 0 and original["native_nest_simulation"]
                and original["execution_clock"]["resolution_ms"] == native.V1_RESOLUTION_MS,
                "SNAPSHOT_COMPLETE_NATIVE_REFERENCE_REQUIRED")
        report.update(native_reference=str(archive), reference_binding=binding,
            native_report_sha256=binding["files"]["primary/report.json"]["sha256"],
            expected_source_batch_sha256=original["connections"]["source_batch_sha256"])
        config, snapshots = native.core.load_config(archive/"original_config.yaml")
        config_sha = snapshots["original_config_sha256"] or snapshots["input_config_sha256"]
        require(config_sha == original["original_config_sha256"], "SNAPSHOT_SOURCE_CONFIG_DIFFERS")
        require(native.core.verify_wheel(config)["sha256"] == original["wheel"]["sha256"],
                "SNAPSHOT_BMTK_WHEEL_DIFFERS")
        with tempfile.TemporaryDirectory(prefix="neuronsim-native-snapshot-",dir="/tmp") as scratch:
            work = Path(scratch)
            rec, inp, source_binding = native.stage_network(Path(delivery["recurrent_reference"]),
                Path(delivery["input_reference"]),work,config_sha)
            bodies, _ = native.bind_sources(Path(config["source_root"]), (rec, inp))
            data = {name:native.node_data(work/"network",name) for name in ("v1","lgn","bkg")}
            refs, _ = native.stage_parameters(Path(delivery["parameter_reference"]),data["v1"],work)
            plan, spike_files = None, None
            if delivery["spike_inputs_reference"] is not None:
                _, _, expected = native.read_input_construction_cells(Path(delivery["input_reference"]))
                plan, _, spike_files = native.load_spike_inputs(Path(delivery["spike_inputs_reference"]),expected,data)
            cfg_path, _ = native.stage_point_config(Path(delivery["source_config"]),work,data,bodies,
                spike_files=spike_files,resolution_ms=native.V1_RESOLUTION_MS)
            expected = {name:native.edge_inventory(work/"network",name,
                rec["validation"] if name=="v1" else inp["validation"][name],config["edge_chunk_size"])
                for name in ("v1","lgn","bkg")}
            require(expected == original["source_denominators"], "SNAPSHOT_SOURCE_DENOMINATORS_DIFFER")
            (work/"native").mkdir()
            with native.core.original_runtime(work,Path(config["bmtk_wheel"]["path"])):
                import nest
                from bmtk.simulator import pointnet
                from bmtk.simulator.pointnet import glif_utils, pointnetwork, pointsimulator
                require(nest.__version__ == native.NEST_VERSION, "SNAPSHOT_NATIVE_VERSION_DIFFERS")
                for module, expected_sha in ((glif_utils,native.BMTK_CONVERTER_SHA256),
                    (pointnetwork,native.POINTNETWORK_SHA256),(pointsimulator,native.POINTSIMULATOR_SHA256)):
                    require(sha256_file(Path(module.__file__)) == expected_sha, "SNAPSHOT_IMPORTED_SOURCE_DIFFERS")
                native.core.require_serial_runtime()
                rules, changes = native.extract_functions(bodies["weight_semantics/legacy_pointnet_runner"].decode(),native.RULE_NAMES)
                require(not changes,"SNAPSHOT_SOURCE_WEIGHT_RULE_CHANGED")
                for name in native.RULE_NAMES:pointnet.synaptic_weight(rules[name])
                cfg = pointnet.Config.from_json(str(cfg_path),validate=True)
                cfg.build_env()
                graph = pointnet.PointNetwork.from_config(cfg)
                writer = ConnectionSnapshot(output/"connections.h5",expected,original["connections"]["source_batch_sha256"])
                with writer.capture(nest,graph):
                    sim = pointnet.PointSimulator.from_config(cfg,graph)
                    native_ids = np.asarray(graph.gid_map.get_nestids("v1",data["v1"]["sorted_ids"]),dtype=np.int64)
                    initial = native.verify_parameters(nest,data["v1"],native_ids,refs,work)
                    # This is precisely the initialization loop in the pinned run()
                    # method. No call to run(), Simulate(), or finalization follows.
                    for module in sim._mods:module.initialize(sim)
                    native.InputReadback(nest,data,writer,native_ids,work,report,input_plan=plan).initialize(sim)
                    require(float(nest.GetKernelStatus("biological_time")) == 0.0, "SNAPSHOT_SIMULATOR_TIME_ADVANCED")
                    output_connections = int(nest.GetKernelStatus("num_connections"))
                    require(output_connections == original["native_total_connections_including_output_devices"]
                            - original["connections"]["native_connections_to_cortex"], "SNAPSHOT_UNEXPECTED_NATIVE_SYNAPSES")
                    report.update(native_biological_time_ms=0.0,native_output_device_connections=output_connections)
                with np.load(archive/"primary/native/cortical_state.npz",allow_pickle=False) as state:
                    require(np.array_equal(state["native_id"],native_ids)
                        and np.array_equal(state["source_id"],data["v1"]["sorted_ids"])
                        and np.array_equal(state["node_type_id"],data["v1"]["sorted_types"])
                        and np.array_equal(state["initial_vm_mV"],initial),"SNAPSHOT_NATIVE_CELL_STATE_DIFFERS")
                require(json.loads((work/"native/parameters.json").read_text()) ==
                        json.loads((archive/"primary/native/parameters.json").read_text()),"SNAPSHOT_PARAMETERS_DIFFER")
                with np.load(work/"native/input_events.npz",allow_pickle=False) as a,np.load(archive/"primary/native/input_events.npz",allow_pickle=False) as b:
                    require(set(a.files)==set(b.files) and all(np.array_equal(a[key],b[key]) for key in a.files),
                            "SNAPSHOT_NATIVE_INPUT_EVENTS_DIFFER")
                report.update(units=len(native_ids),parameters_and_initial_state_exact=True,input_events_exact=True)
            writer.close()
            checked = verify_snapshot(output/"connections.h5",original["connections"]["source_batch_sha256"],
                                      original["connections"]["native_connections_to_cortex"])
            report.update(snapshot=checked,status="PASS_COMPLETE_NATIVE_CONNECTION_SNAPSHOT",
                          connections_file="connections.h5",file_bytes=(output/"connections.h5").stat().st_size)
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if writer is not None:writer.close()
        report.update(elapsed_seconds=time.monotonic()-started,recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/"report.json",report)
        shutil.copy2(__file__,output/Path(__file__).name)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export_snapshot(args.reference,args.output)))


if __name__=="__main__":
    main()
