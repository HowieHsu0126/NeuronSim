"""Acquire Allen GLIF3 references for the V1 source's specimen identities.

These are candidate cell-parameter references, not proof that the V1 network
uses identical parameters. No training or evaluation response is requested.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np

from .provenance import freeze_json, sha256_file

API = "https://api.brain-map.org"
GLIF3_TEMPLATE = 395310475
BMTK_CONVERTER_SHA256 = "29b9073479cd791d51703361d196804a3501afb1b040e2eccf55b2426c1c68ea"


def select_models(records: list[dict], expected_specimens: list[int]) -> list[dict]:
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records) or set(by_id) != set(expected_specimens):
        raise ValueError("V1_GLIF_SPECIMEN_COHORT_INCOMPLETE_OR_DUPLICATE")
    selected = []
    for specimen in expected_specimens:
        models = [model for model in by_id[specimen]["neuronal_models"]
                  if model["neuronal_model_template"]["id"] == GLIF3_TEMPLATE]
        if len(models) != 1:
            raise ValueError("V1_UNIQUE_GLIF3_MODEL_REQUIRED:" + str(specimen))
        model = models[0]
        files = [item for item in model["well_known_files"]
                 if item["well_known_file_type"]["name"] == "NeuronalModelParameters"
                 and item["attachable_type"] == "NeuronalModel"
                 and item["attachable_id"] == model["id"]]
        if len(files) != 1:
            raise ValueError("V1_UNIQUE_MODEL_PARAMETER_FILE_REQUIRED:" + str(specimen))
        item = files[0]
        link = item["download_link"]
        if link != "/api/v2/well_known_file_download/" + str(item["id"]):
            raise ValueError("V1_PARAMETER_DOWNLOAD_IDENTITY_DIFFERS")
        selected.append(dict(specimen_id=specimen, neuronal_model_id=model["id"],
            neuronal_model_template_id=GLIF3_TEMPLATE, well_known_file_id=item["id"],
            download_url=API + link))
    return selected


def _fetch(url: str, output: Path, limit: int) -> tuple[dict, dict]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=30) as response:
        body = response.read(limit + 1)
        if len(body) > limit:
            raise ValueError("V1_PARAMETER_JSON_EXCEEDS_BOUND")
        receipt = dict(url=url, resolved_url=response.url, http_status=response.status,
                       bytes=len(body), etag=response.headers.get("ETag"), transport="direct")
    output.write_bytes(body)
    receipt.update(path=output.name, sha256=sha256_file(output))
    return json.loads(body), receipt


def _converter(path: Path):
    if sha256_file(path) != BMTK_CONVERTER_SHA256:
        raise ValueError("V1_BMTK_CONVERSION_SOURCE_CHANGED")
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "converter_builtin"]
    if len(functions) != 1:
        raise ValueError("V1_BMTK_CONVERSION_FUNCTION_MISSING")
    # Reuse the publisher's exact conversion without importing its NEST runtime.
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["converter_builtin"]


def acquire(node_types: Path, node_report: Path, converter: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="ACQUIRING", started_utc=datetime.now(timezone.utc).isoformat(),
        reference_model_family="Allen GLIF3 / LIF-ASC", references=[],
        source_network_parameter_equality_verified=False,
        native_dynamics_qualified=False, source_scientific_fidelity_qualified=False,
        production_episodes=0)
    try:
        native = json.loads(node_report.read_text())
        if (native["status"] != "NATIVE_NODE_IDENTITY_AND_CELL_TYPE_JOIN_PASS"
                or sha256_file(node_types) != native["node_types_sha256"]):
            raise ValueError("V1_VERIFIED_SOURCE_TYPE_TABLE_REQUIRED")
        with node_types.open() as stream:
            rows = list(csv.DictReader(stream, delimiter=" "))
        specimens = sorted(int(row["node_type_id"]) for row in rows)
        if len(specimens) != len(set(specimens)) or len(specimens) != native["source_cell_models"]:
            raise ValueError("V1_SOURCE_SPECIMEN_COUNT_DIFFERS")
        report.update(expected_specimens=len(specimens), specimen_ids=specimens,
                      node_types_sha256=native["node_types_sha256"],
                      bmtk_converter_sha256=BMTK_CONVERTER_SHA256)
        convert = _converter(converter)
        criteria = ("model::Specimen,rma::criteria,[id$in" + ",".join(map(str, specimens)) + "]"
            ",rma::include,neuronal_models(neuronal_model_template,well_known_files(well_known_file_type))"
            ",rma::options[num_rows$eq'all'][count$eqfalse]")
        url = API + "/api/v2/data/query.json?" + urllib.parse.urlencode({"criteria": criteria})
        data, receipt = _fetch(url, output / "specimen_models.json", 8 * 1024 * 1024)
        report["metadata_request"] = receipt
        if data["success"] is not True:
            raise ValueError("V1_GLIF_METADATA_QUERY_FAILED")
        selected = select_models(data["msg"], specimens)
        for identity in selected:
            name = f"specimen_{identity['specimen_id']}_model_{identity['neuronal_model_id']}.json"
            config, receipt = _fetch(identity["download_url"], output / name, 1024 * 1024)
            template, params = convert("nest:glif_lif_asc_psc", config)
            if (template != "nest:glif_psc" or params["after_spike_currents"] is not True
                    or params["spike_dependent_threshold"] is not False
                    or params["adapting_threshold"] is not False):
                raise ValueError("V1_GLIF3_CONVERSION_FAMILY_DIFFERS")
            converted = {key: value.tolist() if isinstance(value, np.ndarray) else value
                         for key, value in params.items()}
            if any(not np.isfinite(np.asarray(value)).all() for value in converted.values()):
                raise ValueError("V1_NONFINITE_CONVERTED_REFERENCE")
            report["references"].append(dict(identity, source=receipt,
                converted_template=template, converted_parameters=converted))
            print(json.dumps({"acquired_references": len(report["references"]),
                              "specimen_id": identity["specimen_id"]}), flush=True)
        report["status"] = "REFERENCE_PARAMETERS_ACQUIRED"
    except Exception as error:
        report.update(status="INCOMPLETE", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["acquired_references"] = len(report["references"])
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        freeze_json(output / "intake.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("node-types", "node-report", "converter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = acquire(**vars(args))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("references", "specimen_ids")}))


if __name__ == "__main__":
    main()
