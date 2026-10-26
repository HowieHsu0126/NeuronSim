#!/usr/bin/env python3
"""Package the existing V1 collection's authenticated metadata and static maps."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import socket
import tarfile

from stage_v1_merlin_block_inputs import module_files

PUBLIC_ROOT = Path("/tmp/neuronsim-v1-collection-public")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--new-readback", type=Path, action="append", default=[])
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise ValueError("COLLECTION_PACKAGING_REQUIRES_RECEIVE_HOST")
    root = args.collection.resolve()
    catalog = json.loads((root / "dataset.json").read_text())
    if catalog["collection"]["kind"] != "accepted_production_blocks":
        raise ValueError("ACCEPTED_COLLECTION_REQUIRED")
    files = {root / name for name in ("dataset.json", "frozen_campaign.yaml",
                                     "qualification.json", "identities.csv", "identities.json")}
    reports = []
    canonical = catalog["circuits"]
    files.update(root / row["path"] for row in canonical.values())
    for row in catalog["collection"]["sources"]:
        reports.append(str(root / row["readback_report"]))
        files.update((root / row["readback_report"], root / row["dataset_catalog"]))
        source_path = root / row["dataset_catalog"]
        source = json.loads(source_path.read_text())
        for key, static in source["circuits"].items():
            if ({k: v for k, v in static.items() if k != "path"}
                    != {k: v for k, v in canonical[key].items() if k != "path"}):
                files.add(source_path.parent / static["path"])
    for report_path in args.new_readback:
        report_path = report_path.resolve()
        report = json.loads(report_path.read_text())
        source_path = Path(report["dataset"]) / "dataset.json"
        source = json.loads(source_path.read_text())
        files.update((report_path, source_path))
        files.update(source_path.parent / row["path"] for row in source["circuits"].values())
        reports.append(str(report_path))
    if any(not path.resolve().is_relative_to(root) for path in files):
        raise ValueError("COLLECTION_INPUT_OUTSIDE_ROOT")
    mapping = {path: path for path in files}
    modules = module_files(args.repo / "Libs/population_dynamics", ["benchmark_collection"])
    mapping.update({path: PUBLIC_ROOT / "population_dynamics" / path.name for path in modules})
    args.record.mkdir(parents=True, exist_ok=True)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with args.archive.open("xb") as target, tarfile.open(fileobj=target, mode="w:gz", compresslevel=1) as archive:
        for source, destination in sorted(mapping.items()):
            raw = source.read_bytes()
            manifest[str(destination)] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            member = tarfile.TarInfo(str(destination).lstrip("/"))
            member.size, member.mode = len(raw), 0o644
            archive.addfile(member, io.BytesIO(raw))
    plan = dict(collection=str(root), public_root=str(PUBLIC_ROOT), readback_reports=reports,
                prior_episodes=catalog["complete_episodes"],
                prior_blocks=len(catalog["collection"]["sources"]),
                added_blocks=len(args.new_readback),
                prior_catalog_sha256=manifest[str(root / "dataset.json")]["sha256"],
                input_files=manifest, episode_payloads_included=False)
    with (args.record / "inputs.json").open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(files=len(manifest), archive_bytes=args.archive.stat().st_size,
                         prior_episodes=plan["prior_episodes"], prior_blocks=plan["prior_blocks"])))


if __name__ == "__main__":
    main()
