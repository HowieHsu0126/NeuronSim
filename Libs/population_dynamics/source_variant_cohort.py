"""Acquire a declared source-RNG diagnostic using existing coupled episodes.

The initial-state RNG library and the native simulation library are separate
dependencies. Both are bound, as are the graph and the original cohort. A
shared numerical seed across RNG families does not establish shared noise.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil

import yaml

from .nest_gpu_spatial import Execution
from .provenance import sha256_file
from .pulse_history import acquire_cases
from .stimulus_trajectory import SCHEMA as TRAJECTORY_SCHEMA

SCHEMA = "mesocircuit_rng_neural_comparison_v1"


def declare(config: dict, original: Path, reference_acquisition: Path,
            candidate_library: Path) -> dict:
    plan = json.loads(original.read_text())
    if (config["schema"] != SCHEMA or plan["schema"] != TRAJECTORY_SCHEMA
            or config["split"] != "train"
            or config["formal_training_seeds"] != [41, 42, 43]):
        raise ValueError("training-only source diagnosis with seeds 41/42/43 required")
    reference_design = json.loads((reference_acquisition / "design.json").read_text())
    reference_ledger = json.loads((reference_acquisition / "acquisition.json").read_text())
    if (reference_design["prospective_design_sha256"] != sha256_file(original)
            or reference_ledger["status"] != "ACQUIRED"):
        raise ValueError("complete source-bound reference acquisition required")
    selected = sorted((copy.deepcopy(c) for c in plan["cases"]
                       if c["split"] == "train" and c["template_id"] == config["template_id"]),
                      key=lambda c: c["block"])
    if (len(selected) != config["expected_blocks"]
            or len({c["block"] for c in selected}) != len(selected)
            or len(selected) < 3):
        raise ValueError("one declared template for every independent training block required")
    reference_libraries = set()
    for case in selected:
        report = json.loads((reference_acquisition / case["case_id"] / "report.json").read_text())
        if (report["status"] != "EXECUTED"
                or asdict(Execution(**report["config"])) != asdict(Execution(**case["settings"]))
                or report["graph_manifest_sha256"] != plan["graph_manifest_sha256"]):
            raise ValueError("reference episode differs from the prospective cohort")
        reference_libraries.add(report["native_library_sha256"])
        case["settings"]["expected_poisson_rng_family"] = config["candidate_rng_family"]
        Execution(**case["settings"]).validate()
    if len(reference_libraries) != 1:
        raise ValueError("reference episodes mix native libraries")
    if not set(config["previously_observed_candidate_cases"]) <= {c["case_id"] for c in selected}:
        raise ValueError("previously observed candidate identity is outside the declaration")
    return dict(schema=SCHEMA, config=config, cases=selected,
                templates=[t for t in plan["templates"] if t["template_id"] == config["template_id"]],
                original_design_sha256=sha256_file(original),
                reference_acquisition=str(reference_acquisition.resolve()),
                reference_native_library_sha256=reference_libraries.pop(),
                native_library_sha256=sha256_file(candidate_library),
                graph_manifest_sha256=plan["graph_manifest_sha256"],
                source_rng_sha256=plan["source_rng_sha256"],
                expected_episodes=len(selected), independent_blocks=len(selected),
                common_initial_states=True, common_noise_across_rng_families=False,
                independent_replication="initial-state/background-seed block conditional on one graph and template",
                scientific_reference_qualified=False, neuroscience_finding_qualified=False,
                formal_training_completed=False, formal_training_seeds=[41, 42, 43])


def freeze(config_path: Path, original: Path, reference: Path,
           candidate_library: Path, output: Path) -> dict:
    result = declare(yaml.safe_load(config_path.read_text()), original, reference, candidate_library)
    encoded = json.dumps(result, indent=2) + "\n"
    if output.exists() and output.read_text() != encoded:
        raise ValueError("cannot overwrite a different source-variant declaration")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded)
    return result


def verify_episode(path: Path, case: dict, declaration: dict) -> dict:
    report = json.loads((path / "report.json").read_text())
    if (report["status"] != "EXECUTED"
            or asdict(Execution(**report["config"])) != asdict(Execution(**case["settings"]))
            or report["graph_manifest_sha256"] != declaration["graph_manifest_sha256"]
            or report["initial_state_rng_library_sha256"] != declaration["source_rng_sha256"]
            or report["native_library_sha256"] != declaration["native_library_sha256"]
            or report["native_poisson_rng_family"] != declaration["config"]["candidate_rng_family"]
            or not report["checks"] or not all(report["checks"].values())):
        raise ValueError("reused candidate has different source, settings, or failed checks")
    for name, digest in report["artifacts"].items():
        if sha256_file(path / name) != digest:
            raise ValueError("reused candidate artifact changed")
    return report


def acquire(design: Path, graph: Path, output: Path, resume: bool = False,
            reuse: tuple[str, Path] | None = None) -> int:
    declaration = json.loads(design.read_text())
    if (declaration["schema"] != SCHEMA
            or sha256_file(graph / "manifest.json") != declaration["graph_manifest_sha256"]
            or sha256_file(Path(os.environ["NESTGPU_LIB"])) != declaration["native_library_sha256"]):
        raise ValueError("declared graph or native runtime changed")
    for case in declaration["cases"]:
        if sha256_file(Path(case["settings"]["source_rng_library"])) != declaration["source_rng_sha256"]:
            raise ValueError("initial-state RNG library changed")
    if output.exists():
        if not resume or json.loads((output / "design.json").read_text()) != declaration:
            raise ValueError("resume requires the exact source-variant declaration")
    else:
        output.mkdir(parents=True)
        (output / "design.json").write_text(json.dumps(declaration, indent=2) + "\n")
    if reuse is not None:
        identity, source = reuse
        cases = [c for c in declaration["cases"] if c["case_id"] == identity]
        if len(cases) != 1:
            raise ValueError("reuse identity outside declared cases")
        report = verify_episode(source, cases[0], declaration)
        target = output / identity
        if not target.exists():
            target.mkdir()
            for name in [*report["artifacts"], "report.json"]:
                shutil.copy2(source / name, target / name)
            (output / "reused_episode.json").write_text(json.dumps(dict(
                case_id=identity, source=str(source.resolve()),
                report_sha256=sha256_file(source / "report.json")), indent=2) + "\n")
        verify_episode(target, cases[0], declaration)
    for case in declaration["cases"]:
        path = output / case["case_id"]
        if path.exists():
            verify_episode(path, case, declaration)
    return acquire_cases(graph, output, declaration, resume=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    frozen = sub.add_parser("freeze")
    frozen.add_argument("--config", type=Path, required=True)
    frozen.add_argument("--original-design", type=Path, required=True)
    frozen.add_argument("--reference-acquisition", type=Path, required=True)
    frozen.add_argument("--candidate-library", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("acquire")
    run.add_argument("--design", type=Path, required=True)
    run.add_argument("--graph", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--reuse-case")
    run.add_argument("--reuse-episode", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.config, args.original_design, args.reference_acquisition,
                        args.candidate_library, args.output)
        print(json.dumps(dict(status="SOURCE_VARIANT_DECLARED", cases=result["expected_episodes"])))
    else:
        if bool(args.reuse_case) != bool(args.reuse_episode):
            parser.error("--reuse-case and --reuse-episode are required together")
        reuse = (args.reuse_case, args.reuse_episode) if args.reuse_case else None
        raise SystemExit(acquire(args.design, args.graph, args.output, args.resume, reuse))


if __name__ == "__main__":
    main()
