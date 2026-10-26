"""Replay the source edge converter on bound real witnesses, without graph data."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

from .provenance import freeze_json, runtime_manifest, sha256_file


DIAGNOSTIC_ORDER = (2, 0, 1, 1, 2, 0, 0, 2, 1)
BLOCK_SIZE = 4


def read_source(directory: Path, expected: dict) -> tuple[str, dict]:
    receipt_path = directory / "source.json"
    receipt = json.loads(receipt_path.read_text())
    path = directory / receipt["artifact"]
    body = path.read_bytes()
    actual = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                  receipt_sha256=sha256_file(receipt_path))
    if any(actual[key] != expected[key] for key in actual) or any(
            actual[key] != receipt[key] for key in ("bytes", "sha256")):
        raise ValueError("CONVERSION_BOUND_SOURCE_CHANGED:" + directory.name)
    return body.decode(), actual


def node_dataframe(nodes: Path, types: Path) -> pd.DataFrame:
    with h5py.File(nodes, "r") as handle:
        native = handle["nodes/v1"]
        ids, kinds = native["node_id"][...], native["node_type_id"][...]
    if (ids.ndim != 1 or kinds.shape != ids.shape or ids.dtype.kind not in "ui"
            or kinds.dtype.kind not in "ui" or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(kinds < 0)):
        raise ValueError("CONVERSION_UNIQUE_NATIVE_IDS_REQUIRED")
    with types.open() as handle:
        rows = list(csv.DictReader(handle, delimiter=" "))
    lookup = {}
    for row in rows:
        kind = int(row["node_type_id"])
        if kind in lookup or not row["pop_name"]:
            raise ValueError("CONVERSION_UNIQUE_POPULATION_TYPES_REQUIRED")
        lookup[kind] = row["pop_name"]
    if any(int(kind) not in lookup for kind in np.unique(kinds)):
        raise ValueError("CONVERSION_NODE_TYPE_DECLARATION_MISSING")
    return pd.DataFrame({"node_type_id": kinds,
                         "pop_name": [lookup[int(kind)] for kind in kinds]},
                        index=pd.Index(ids, name="node_id"))


class IdMetadataFile:
    """Explicit adapter for this replay; this is not BMTK SONATA validation."""
    def __init__(self, data_files, data_type_files):
        frame = node_dataframe(Path(data_files), Path(data_type_files))
        self.nodes = {"v1": SimpleNamespace(to_dataframe=lambda: frame.copy())}


def source_reader():
    if importlib.util.find_spec("bmtk") is None:
        return IdMetadataFile, dict(name="direct_native_id_metadata_adapter",
            bmtk_installed=False, sonata_file_to_dataframe_dynamically_tested=False,
            boundary="Only native node_id/node_type_id and source-declared pop_name are read; BMTK File/to_dataframe is not dynamically tested.")
    # BMTK is optional; this import is reachable only for an installed dependency.
    from bmtk.utils.sonata import File
    import inspect
    path = Path(inspect.getfile(File))
    return File, dict(name="installed_bmtk_sonata_File", bmtk_installed=True,
        sonata_file_to_dataframe_dynamically_tested=True,
        source_path=str(path), source_sha256=sha256_file(path))


def extract_converter(source: str, reader=IdMetadataFile):
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "convert_edges"]
    if len(functions) != 1 or functions[0].decorator_list:
        raise ValueError("CONVERSION_SINGLE_UNDECORATED_SOURCE_FUNCTION_REQUIRED")
    function = functions[0]
    original = ast.dump(function, include_attributes=False)
    original_text = ast.get_source_segment(source, function)

    class IntCompatibility(ast.NodeTransformer):
        def __init__(self):
            self.lines = []

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr == "int":
                self.lines.append(node.lineno)
                return ast.copy_location(ast.Name(id="int", ctx=ast.Load()), node)
            return self.generic_visit(node)

    compatibility = IntCompatibility()
    converted = compatibility.visit(function)
    module = ast.fix_missing_locations(ast.Module(body=[converted], type_ignores=[]))
    namespace = dict(os=os, shutil=shutil, pd=pd, np=np, h5py=h5py, File=reader)
    exec(compile(module, "<bound-source-convert_edges>", "exec"), namespace)
    return namespace["convert_edges"], dict(
        first_line=function.lineno, last_line=function.end_lineno,
        original_function_ast_sha256=hashlib.sha256(original.encode()).hexdigest(),
        executed_function_ast_sha256=hashlib.sha256(ast.dump(converted, include_attributes=False).encode()).hexdigest(),
        compatibility=[dict(line=line, before="np.int", after="int") for line in compatibility.lines],
        original_function_text=original_text)


def query_value(text: str, column: str):
    expression = ast.parse(text, mode="eval").body
    if not (isinstance(expression, ast.Compare) and isinstance(expression.left, ast.Name)
            and expression.left.id == column and len(expression.ops) == 1
            and isinstance(expression.ops[0], ast.Eq) and len(expression.comparators) == 1
            and isinstance(expression.comparators[0], ast.Constant)):
        raise ValueError("CONVERSION_INVALID_DECLARED_QUERY:" + text)
    value = expression.comparators[0].value
    return int(value) if column == "node_type_id" else str(value)


def edge_lookup(table: str) -> dict:
    result, ids = {}, set()
    for row in csv.DictReader(io.StringIO(table), delimiter=" "):
        identity = int(row["edge_type_id"])
        key = (query_value(row["source_query"], "pop_name"),
               query_value(row["target_query"], "node_type_id"))
        if identity in ids or key in result:
            raise ValueError("CONVERSION_UNIQUE_EDGE_TYPE_KEYS_REQUIRED")
        ids.add(identity)
        result[key] = identity
    return result


def witness_rows(arithmetic: dict, sonata: dict) -> list[dict]:
    if (arithmetic.get("status") != "SOURCE_WEIGHT_ARITHMETIC_DIAGNOSTIC_COMPLETE"
            or sonata.get("status") != "SOURCE_SONATA_WITNESS_QUERY_COMPLETE"
            or sonata.get("completed_selectors") != 3 or len(arithmetic.get("cases", [])) != 3
            or len(sonata.get("cases", [])) != 3):
        raise ValueError("CONVERSION_THREE_COMPLETE_WITNESSES_REQUIRED")
    cases = {row["name"]: row for row in sonata["cases"]}
    if len(cases) != 3:
        raise ValueError("CONVERSION_DUPLICATE_WITNESS")
    rows = []
    for witness in arithmetic["cases"]:
        case = cases[witness["name"]]
        if (case.get("matches_complete") is not True or case.get("matching_row_count") != 1
                or len(case.get("matching_rows", [])) != 1
                or any(case[key] != witness[key] for key in
                       ("source_id", "target_id", "raw_group", "raw_offset"))):
            raise ValueError("CONVERSION_BOUND_WITNESS_IDENTITY_REQUIRED")
        edge = case["matching_rows"][0]
        nsyns = edge["properties"]["nsyns"]
        if (any(edge[key + "_node_id"] != witness[key + "_id"] for key in ("source", "target"))
                or nsyns.get("status") != "RECORDED" or type(nsyns.get("value")) is not int
                or nsyns["value"] < 1):
            raise ValueError("CONVERSION_RECORDED_SOURCE_ROW_REQUIRED")
        rows.append(dict(name=witness["name"], source_node_id=edge["source_node_id"],
            target_node_id=edge["target_node_id"], observed_edge_type_id=edge["edge_type_id"],
            source_global_row_id=edge["global_row_id"], nsyns=nsyns["value"],
            raw_group=witness["raw_group"], raw_offset=witness["raw_offset"]))
    return rows


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def replay_slice(converter, nodes: Path, types: Path, table: str,
                 rows: list[dict], output: Path, block_size: int = BLOCK_SIZE) -> dict:
    metadata = node_dataframe(nodes, types)
    lookup = edge_lookup(table)
    expected = []
    for row in rows:
        if any(row[key] not in metadata.index for key in ("source_node_id", "target_node_id")):
            raise ValueError("CONVERSION_WITNESS_NODE_ID_MISSING")
        source = metadata.loc[row["source_node_id"], "pop_name"]
        target = int(metadata.loc[row["target_node_id"], "node_type_id"])
        if (source, target) not in lookup:
            raise ValueError("CONVERSION_WITNESS_EDGE_TYPE_KEY_MISSING")
        expected.append(lookup[(source, target)])
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="v1-converter-replay-") as temporary:
        root = Path(temporary)
        bio, glif = root / "network_bio", root / "network_glif"
        bio.mkdir(); glif.mkdir(); (root / "glif_props").mkdir()
        shutil.copy2(nodes, glif / "v1_nodes.h5")
        shutil.copy2(types, glif / "v1_node_types.csv")
        (root / "glif_props/v1_v1_edge_types.csv").write_text(table)
        source_path = bio / "v1_v1_edges.h5"
        unchanged = {
            "source_node_id": np.asarray([row["source_node_id"] for row in rows], dtype=np.uint64),
            "target_node_id": np.asarray([row["target_node_id"] for row in rows], dtype=np.uint64),
            "edge_group_id": np.zeros(len(rows), dtype=np.uint32),
            "edge_group_index": np.arange(len(rows), dtype=np.int64),
            "diagnostic_row_id": np.arange(len(rows), dtype=np.int64),
            "source_global_row_id": np.asarray([row["source_global_row_id"] for row in rows], dtype=np.int64),
            "0/nsyns": np.asarray([row["nsyns"] for row in rows], dtype=np.uint32),
        }
        with h5py.File(source_path, "w") as handle:
            group = handle.create_group("edges/v1_to_v1")
            for key, values in unchanged.items():
                group.create_dataset(key, data=values)
            group.create_dataset("edge_type_id", data=[row["observed_edge_type_id"] for row in rows])
        before = sha256_file(source_path)
        with working_directory(root):
            converter("v1", bio_network_dir=str(bio), glif_network_dir=str(glif), block_size=block_size)
        converted_path = glif / "v1_v1_edges.h5"
        with h5py.File(converted_path, "r") as handle:
            group = handle["edges/v1_to_v1"]
            preserved = {key: bool(np.array_equal(group[key][...], value) and group[key].dtype == value.dtype)
                         for key, value in unchanged.items()}
            generated = group["edge_type_id"][...].tolist()
        if sha256_file(source_path) != before or not all(preserved.values()):
            raise ValueError("CONVERSION_SOURCE_IDENTITIES_OR_NSYNS_CHANGED")
        shutil.copy2(source_path, output / "diagnostic_input_edges.h5")
        shutil.copy2(converted_path, output / "diagnostic_converted_edges.h5")
        shutil.copy2(glif / "v1_v1_edge_types.csv", output / "converted_edge_types.csv")
    comparisons = [dict(row, diagnostic_row_id=index, expected_edge_type_id=int(expect),
                        generated_edge_type_id=int(actual), agrees_with_independent_lookup=actual == expect,
                        agrees_with_published_source=actual == row["observed_edge_type_id"])
                   for index, (row, expect, actual) in enumerate(zip(rows, expected, generated))]
    if len(generated) != len(rows):
        raise ValueError("CONVERSION_OUTPUT_ROW_COUNT_DIFFERS")
    return dict(metadata_nodes=len(metadata), diagnostic_rows=len(rows), block_size=block_size,
                block_intervals=[[start, min(start + block_size, len(rows))] for start in range(0, len(rows), block_size)],
                preserved_fields=preserved, input_edges_unchanged=True, rows=comparisons,
                independent_lookup_matches=sum(row["agrees_with_independent_lookup"] for row in comparisons),
                published_source_matches=sum(row["agrees_with_published_source"] for row in comparisons))


def replay(source_root: Path, nodes: Path, types: Path, node_comparison: Path,
           arithmetic_report: Path, sonata_report: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_SOURCE_CONVERTER_REPLAY", exit_code=1,
        scientific_qualification=False, production_episodes=0, full_graph_recompared=False,
        scope="Three bound real source witnesses repeated in nine diagnostic rows; original edge converter only, no node conversion or random realization.",
        started_utc=datetime.now(timezone.utc).isoformat())
    try:
        arithmetic = json.loads(arithmetic_report.read_text())
        sonata = json.loads(sonata_report.read_text())
        comparison = json.loads(node_comparison.read_text())
        hashes = {key: sha256_file(path) for key, path in
                  (("nodes", nodes), ("types", types), ("node_comparison", node_comparison),
                   ("arithmetic", arithmetic_report), ("sonata", sonata_report))}
        bindings = comparison["input_sha256"]
        if (comparison.get("status") != "COMPLETE_NODE_METADATA_COMPARISON"
                or comparison["legacy_asset"].get("complete") is not True
                or nodes.stat().st_size != comparison["legacy_asset"]["bytes"]
                or hashes["nodes"] != comparison["legacy_asset"]["sha256"]
                or hashes["nodes"] != bindings["current_nodes.h5"]
                or hashes["nodes"] != arithmetic["input_sha256"]["nodes"]
                or hashes["types"] != bindings["current_types.txt"]
                or hashes["types"] != arithmetic["input_sha256"]["types"]
                or hashes["arithmetic"] != bindings["arithmetic.json"]
                or hashes["sonata"] != bindings["sonata.json"]
                or sonata.get("witness_report_sha256") != hashes["arithmetic"]
                or sonata.get("comparison_report_sha256") != arithmetic["comparison_report_sha256"]):
            raise ValueError("CONVERSION_BOUND_METADATA_OR_WITNESS_CHANGED")
        source_hashes = {}
        bodies = {}
        for name in ("legacy_glif_converter", "legacy_recurrent_table"):
            relative = "weight_semantics/" + name
            bodies[name], source_hashes[relative] = read_source(source_root / relative, arithmetic["source_sha256"][relative])
        selected = witness_rows(arithmetic, sonata)
        rows = [dict(selected[index], witness_index=index) for index in DIAGNOSTIC_ORDER]
        reader, reader_report = source_reader()
        converter, function_report = extract_converter(bodies["legacy_glif_converter"], reader)
        function_text = function_report.pop("original_function_text")
        (output / "original_convert_edges.py").write_text(function_text + "\n")
        result = replay_slice(converter, nodes, types, bodies["legacy_recurrent_table"], rows, output / "slice")
        if result["metadata_nodes"] != comparison["comparison"]["nodes"]:
            raise ValueError("CONVERSION_COMPLETE_METADATA_COUNT_DIFFERS")
        if result["independent_lookup_matches"] != len(rows):
            status = "SOURCE_CONVERTER_REPLAY_POSITIONAL_MISMATCH"
        elif result["published_source_matches"] != len(rows):
            status = "SOURCE_CONVERTER_REPLAY_DIFFERS_FROM_PUBLISHED_WITNESSES"
        else:
            status = "SOURCE_CONVERTER_REPLAY_MATCHES_PUBLISHED_WITNESSES"
        report.update(status=status, exit_code=0 if result["independent_lookup_matches"] == len(rows) else 2,
            input_sha256=hashes, source_sha256=source_hashes,
            reader=reader_report, function=function_report, result=result, unique_witnesses=len(selected),
            diagnostic_order=list(DIAGNOSTIC_ORDER), historical_pandas_runtime_reproduced=False,
            historical_producer_identified=False, numpy_version=np.__version__, pandas_version=pd.__version__)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["runtime"] = runtime_manifest({}, list(sys.argv), report["exit_code"])
        freeze_json(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "nodes", "types", "node-comparison", "arithmetic-report", "sonata-report", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    report = replay(**vars(parser.parse_args(argv)))
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "unique_witnesses")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
