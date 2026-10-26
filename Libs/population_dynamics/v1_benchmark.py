"""Join full V1 native execution, delivered visual controls and episode export.

Reuse the source movie witness, visual command verifier and circuit exporter.
This runner reads complete development evidence; it never runs NEST or promotes
the exposed circuit to a production split.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .benchmark_tasks import condition_for_episode
from .circuit_campaign import campaign_config, episode_identities
from .provenance import freeze_json, sha256_file
from .v1_source_dataset import COORDINATE_SOURCES, export_pointnet
from .v1_circuit_stimulus import rate_config_without_noise


def _source_consumer_report(module: str, arguments: list[str], output: Path) -> dict:
    """Keep each archived LGN module namespace in its own source process."""
    environment = dict(os.environ)
    module_root = str(Path(__file__).resolve().parent.parent)
    inherited_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = module_root + (os.pathsep + inherited_path if inherited_path else "")
    subprocess.run([sys.executable, "-B", "-m", "population_dynamics." + module, *arguments,
                    "--output", str(output.resolve())], env=environment, check=True)
    return json.loads((output / "report.json").read_text())


def inspect_source_movie(rate_archive: Path, output: Path) -> dict:
    return _source_consumer_report("v1_lgn_runtime", ["inspect-stimulus", "--rate-archive",
                                   str(rate_archive.resolve())], output)


def export_source_controls(rate_archive: Path, movie_archive: Path,
                           spike_archive: Path, output: Path) -> dict:
    return _source_consumer_report("v1_visual_controls", [
        "--rate-archive", str(rate_archive.resolve()),
        "--movie-archive", str(movie_archive.resolve()),
        "--constructed-input-archive", str(spike_archive.resolve())], output)


def stage_declared(source: Path, target: Path, names: tuple[str, ...]) -> None:
    """Copy only consumed declared files; require successful, unique delivery."""
    delivery = json.loads((source / "delivery.json").read_text())
    rows = {row["path"]: row for row in delivery["files"]}
    if delivery.get("exit_code") != 0 or len(rows) != len(delivery["files"]):
        raise ValueError("V1_BENCHMARK_COMPLETE_UNIQUE_DELIVERY_REQUIRED:" + str(source))
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "delivery.json", target / "delivery.json")
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name not in rows:
            raise ValueError("V1_BENCHMARK_DECLARED_INPUT_REQUIRED:" + name)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        row = rows[name]
        if destination.stat().st_size != row["bytes"] or sha256_file(destination) != row["sha256"]:
            raise ValueError("V1_BENCHMARK_STAGED_INPUT_CHANGED:" + name)


def local_delivery(root: Path, subdirectory: str) -> None:
    """Publish the generated witness for the existing bound-file consumer."""
    files = [dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                  sha256=sha256_file(path)) for path in sorted((root / subdirectory).iterdir()) if path.is_file()]
    freeze_json(root / "delivery.json", dict(exit_code=0, files=files))


def bind_candidate(campaign: dict, recurrent: dict, inputs: dict, native: dict,
                   rate_config: dict, *, circuit_index: int, block_index: int,
                   condition_index: int, spike_config: dict | None = None) -> dict | None:
    """Match a prospective identity to the actually constructed circuit and noise.

    Historical condition exports keep their original development identity and
    seeds. Candidate exports must use their own source construction and declared
    noise block. Deterministic rates can come from another noise block of the
    same circuit and condition. Neither case receives a production split.
    """
    if (type(circuit_index) is not int or not 0 <= circuit_index < campaign["sampling"]["circuit_instances"]
            or type(block_index) is not int or not 0 <= block_index < 100):
        raise ValueError("V1_BENCHMARK_CIRCUIT_OR_BLOCK_INDEX_INVALID")
    constructed = [report.get("original_config", {}) for report in (recurrent, inputs)]
    candidates = [config.get("benchmark_circuit_candidate") for config in constructed]
    if candidates == [None, None]:
        if circuit_index != 0 or block_index != 0:
            raise ValueError("V1_BENCHMARK_NEW_BLOCK_REQUIRES_CANDIDATE_CIRCUIT")
        return None
    identity = next(row for row in episode_identities(campaign, "v1_visual")
                    if row["circuit_index"] == circuit_index and row["block_index"] == block_index
                    and row["condition_index"] == condition_index)
    expected = dict(status="PREPARED_BENCHMARK_CIRCUIT_CONSTRUCTION_INPUT",
                    scenario_id="v1_visual", circuit_id=identity["circuit_id"],
                    circuit_index=circuit_index, intended_split=identity["split"],
                    construction_scope="development", connectivity_seed=identity["connectivity_seed"],
                    complete_episodes=0, production_eligible=False)
    if candidates != [expected, expected] or any(
            config.get("builder_seed") != identity["connectivity_seed"]
            or config.get("glif_assignment_seed") != identity["connectivity_seed"] + 1000000
            for config in constructed):
        raise ValueError("V1_BENCHMARK_CANDIDATE_REQUIRES_ITS_OWN_CONSTRUCTED_CIRCUIT")
    if (recurrent.get("rng", {}).get("builder_seed") != identity["connectivity_seed"]
            or recurrent["rng"].get("glif_assignment_seed") != identity["connectivity_seed"] + 1000000
            or inputs.get("rng", {}).get("lgn_selection_python_seed") != identity["connectivity_seed"]
            or any(report.get("original_config_sha256") != native["original_config_sha256"]
                   for report in (recurrent, inputs))):
        raise ValueError("V1_BENCHMARK_ACTUAL_CONSTRUCTION_SEEDS_DIFFER")
    realized_config = rate_config if spike_config is None else spike_config
    if rate_config_without_noise(rate_config) != rate_config_without_noise(realized_config):
        raise ValueError("V1_BENCHMARK_CANDIDATE_RATE_RECIPE_DIFFERS")
    if (realized_config.get("spike_generation", {}).get("starting_noise_seed") != identity["stimulus_seed"]
            or realized_config.get("background", {}).get("starting_noise_seed") != identity["background_seed"]
            or native["bindings"]["spike_inputs"]["config"] != realized_config):
        raise ValueError("V1_BENCHMARK_CANDIDATE_NOISE_BLOCK_DIFFERS")
    return identity


def export(config: Path, rate_source: Path, spike_source: Path, native_source: Path,
           authority_source: Path, output: Path, *, block_index: int, condition_index: int,
           circuit_index: int = 0) -> dict:
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURMD_NODENAME") != "cpu03"
            or os.environ.get("SLURM_JOB_GPUS") or output.exists()):
        raise ValueError("V1_BENCHMARK_FRESH_CPU03_EXPORT_REQUIRED")
    campaign = campaign_config(config)
    protocol = campaign["scenarios"]["v1_visual"]["task_protocol"]
    condition = condition_for_episode(protocol, "v1_visual",
                                      dict(block_index=block_index, condition_index=condition_index))
    rate = json.loads((rate_source / "probe/probe.json").read_text())
    stimulus = rate["config"]["stimulus"]
    expected = dict(duration_s=protocol["duration_s"], gray_screen_s=protocol["gray_screen_s"],
        spatial_frequency_cycles_per_degree=protocol["spatial_frequency_cycles_per_degree"],
        temporal_frequency_hz=protocol["temporal_frequency_hz"], direction_deg=condition["direction_deg"],
        contrast_percent=condition["contrast_percent"])
    if stimulus != expected or rate.get("source_units") != 17400:
        raise ValueError("V1_BENCHMARK_FULL_TASK_RATE_CONDITION_DIFFERS")
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(status="RUNNING_V1_BENCHMARK_DEVELOPMENT_EXPORT", condition=condition,
        actual_split="development", complete_episodes=0, production_episodes=0,
        scientific_fidelity_qualified=False, source_simulations=0,
        source_archives=dict(rates=str(rate_source), spikes=str(spike_source), native=str(native_source),
                             coordinates=str(authority_source)))
    scratch = output.parent / (output.name + "_inputs")
    scratch.mkdir(exist_ok=False)
    try:
        prepared = json.loads((rate_source / "prepared/prepared.json").read_text())
        rate_names = ("probe/probe.json", "probe/firing_rates_f_tot.h5", "prepared/prepared.json")
        rate_names += tuple("prepared/" + row["path"] for row in prepared["files"])
        stage_declared(rate_source, scratch / "rates", rate_names)
        spike_names = ("visual_inputs/report.json", "visual_inputs/lgn/report.json")
        spike_names += tuple("visual_inputs/" + population + "/trial_0000." + suffix
                             for population in ("lgn", "bkg") for suffix in ("npz", "h5"))
        stage_declared(spike_source, scratch / "spikes", spike_names)
        native_names = ("primary/report.json", "replay/report.json", "comparison/report.json",
            "primary/network/v1_nodes.h5", "primary/network/v1_node_types.csv",
            "recurrent/primary/biophysical/v1_nodes.h5", "recurrent/primary/report.json",
            "inputs_reference/primary/report.json")
        native_names += tuple(prefix + "/native/" + leaf for prefix in ("primary", "replay")
            for leaf in ("cortical_state.npz", "cortical_spikes.npz", "input_events.npz", "parameters.json"))
        stage_declared(native_source, scratch / "native", native_names)
        native = json.loads((scratch / "native/primary/report.json").read_text())
        if native.get("mode") != "full" or native.get("cortical_units") != 230924:
            raise ValueError("V1_BENCHMARK_COMPLETE_COUPLED_CORTEX_REQUIRED")
        construction = []
        for prefix, role in (("recurrent", "recurrent"), ("inputs_reference", "inputs")):
            path = scratch / "native" / prefix / "primary/report.json"
            bound = native["bindings"][role]["files"]["primary/report.json"]
            if path.stat().st_size != bound["bytes"] or sha256_file(path) != bound["sha256"]:
                raise ValueError("V1_BENCHMARK_NATIVE_CONSTRUCTION_BINDING_DIFFERS:" + role)
            construction.append(json.loads(path.read_text()))
        spikes = json.loads((scratch / "spikes/visual_inputs/report.json").read_text())
        report["candidate_identity"] = bind_candidate(campaign, *construction, native, rate["config"],
            circuit_index=circuit_index, block_index=block_index, condition_index=condition_index,
            spike_config=spikes["config"])
        authority_names = ("authority.json", "model_page.html", "model_page.txt", "model_readme.txt")
        authority_names += tuple("sources/" + role + "/" + leaf for role in COORDINATE_SOURCES
                                 for leaf in ("source.json", "source_asset.txt"))
        stage_declared(authority_source, scratch / "coordinates", authority_names)
        # Each existing consumer requires an immutable delivery manifest. Keep
        # the movie and control deliveries distinct throughout the join.
        movie_archive = output / "movie_evidence"
        controls_archive = output / "control_evidence"
        movie = inspect_source_movie(scratch / "rates", movie_archive / "movie")
        if movie["status"] != "SOURCE_MOVIE_CONTROL_WITNESS_PASS":
            raise ValueError("V1_BENCHMARK_SOURCE_MOVIE_WITNESS_FAILED")
        local_delivery(movie_archive, "movie")
        export_source_controls(scratch / "rates", movie_archive, scratch / "spikes",
                               controls_archive / "controls")
        local_delivery(controls_archive, "controls")
        exported = export_pointnet(scratch / "native", scratch / "coordinates", output / "export", controls_archive)
        if exported["status"] != "EXPORTED_VERIFIED_POINTNET_SOURCE_OBSERVATION":
            raise ValueError("V1_BENCHMARK_COMPLETE_EPISODE_EXPORT_FAILED")
        report.update(status="COMPLETE_V1_BENCHMARK_DEVELOPMENT_EPISODE", complete_episodes=1,
                      units=exported["source_units"], episode_id=exported["episode_id"],
                      dataset_catalog_sha256=exported["dataset_catalog_sha256"],
                      source_command_representation_verified=True, independent_native_replay_verified=True)
    except BaseException as error:
        report.update(status="FAILED_V1_BENCHMARK_DEVELOPMENT_EXPORT", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        freeze_json(output / "report.json", report)
        shutil.rmtree(scratch)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "rate-source", "spike-source", "native-source", "authority-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--condition-index", type=int, required=True)
    parser.add_argument("--circuit-index", type=int, default=0)
    args = parser.parse_args()
    export(args.config, args.rate_source, args.spike_source, args.native_source, args.authority_source,
           args.output, block_index=args.block_index, condition_index=args.condition_index,
           circuit_index=args.circuit_index)


if __name__ == "__main__":
    main()
