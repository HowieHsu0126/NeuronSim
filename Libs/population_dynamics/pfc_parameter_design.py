"""Prepare distinct source coupling operators for PFC development qualification.

The output binds prospective circuit aliases to actual numeric kernels. It does
not qualify the parameter domain, release production, or create source episodes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from itertools import islice
from pathlib import Path

import numpy as np
import yaml

from .benchmark_tasks import pfc_working_memory_config
from .circuit_campaign import campaign_config, episode_identities
from .pfc_kernel_parameters import circuit_provenance, source_kernel
from .pfc_native_drive import SOURCE_ENTRY_SHA256
from .pfc_source_execution import _case_config
from .provenance import freeze_json, sha256_file

SCENARIO = "pfc_working_memory"


def parameter_grid(settings: dict, expected_count: int) -> list[dict]:
    if (not isinstance(settings, dict) or settings.get("schema") != "pfc_parameter_design_v1"
            or settings.get("scope") != "development_candidate" or settings.get("frozen") is not False):
        raise ValueError("PFC_PARAMETER_DESIGN_CANDIDATE_REQUIRED")
    axes = []
    for name in ("sigma_weight_profile", "Jpos_excit2excit"):
        axis = settings.get(name)
        if (not isinstance(axis, dict) or set(axis) != {"lower", "upper", "count"}
                or type(axis["count"]) is not int or axis["count"] < 2
                or any(type(axis[k]) not in (int, float) or not np.isfinite(axis[k]) for k in ("lower", "upper"))
                or axis["lower"] >= axis["upper"]):
            raise ValueError("PFC_PARAMETER_DESIGN_AXIS_INVALID:" + name)
        axes.append(np.linspace(axis["lower"], axis["upper"], axis["count"]))
    if len(axes[0]) * len(axes[1]) != expected_count:
        raise ValueError("PFC_PARAMETER_DESIGN_CIRCUIT_COUNT_MISMATCH")
    seed = settings.get("circuit_assignment_seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("PFC_PARAMETER_DESIGN_ASSIGNMENT_SEED_INVALID")
    grid = [dict(sigma_weight_profile=float(sigma), Jpos_excit2excit=float(gain))
            for sigma in axes[0] for gain in axes[1]]
    # Whole-circuit splits must not follow a sorted parameter axis. This seed
    # assigns the design; it is neither a training seed nor a source noise draw.
    order = np.random.RandomState(seed).permutation(expected_count)
    return [grid[int(index)] for index in order]


def describe_kernels(base: dict, parameters: list[dict]) -> list[dict]:
    n_e, n_i = base["excitatory_neurons"], base["inhibitory_neurons"]
    if (type(n_e) is not int or type(n_i) is not int or n_e + n_i < 100000
            or n_i <= 0 or base.get("synapse_execution") != "factorized_uniform_projections"):
        raise ValueError("PFC_PARAMETER_DESIGN_FULL_COUPLED_SOURCE_REQUIRED")
    rows, seen = [], {}
    for index, values in enumerate(parameters):
        config = dict(base, **values)
        kernel = source_kernel(n_e, config)
        digest = hashlib.sha256(kernel.tobytes(order="C")).hexdigest()
        if digest in seen:
            raise ValueError(f"PFC_PARAMETER_DESIGN_DUPLICATE_EFFECTIVE_KERNEL:{seen[digest]}:{index}")
        seen[digest] = index
        rows.append(dict(**values, effective_kernel_sha256=digest,
                         kernel_units=n_e, kernel_min=float(kernel.min()), kernel_max=float(kernel.max()),
                         discrete_kernel_mean=float(kernel.mean()),
                         **circuit_provenance(config, SOURCE_ENTRY_SHA256)))
    return rows


def prepare(campaign_path: Path, base_path: Path, settings_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("PFC_PARAMETER_DESIGN_OUTPUT_EXISTS")
    campaign = campaign_config(campaign_path)
    settings = yaml.safe_load(settings_path.read_text())
    base = yaml.safe_load(base_path.read_text())
    if not isinstance(base, dict) or base.get("scope") != "source_reproduction":
        raise ValueError("PFC_PARAMETER_DESIGN_SOURCE_BASE_REQUIRED")
    protocol = campaign["scenarios"][SCENARIO]["task_protocol"]
    base.update(dt_ms=protocol["dt_ms"], duration_ms=protocol["duration_ms"])
    count = campaign["sampling"]["circuit_instances"]
    parameters = parameter_grid(settings, count)
    rows = describe_kernels(base, parameters)
    inputs = dict(campaign_sha256=sha256_file(campaign_path), base_sha256=sha256_file(base_path),
                  parameter_design_sha256=sha256_file(settings_path),
                  kernel_implementation_sha256=sha256_file(Path(__file__).with_name("pfc_kernel_parameters.py")),
                  source_entry_sha256=SOURCE_ENTRY_SHA256)
    stride = campaign["sampling"]["initial_noise_blocks_per_circuit"] * campaign["sampling"]["conditions_per_block"]
    identities = list(islice(episode_identities(campaign, SCENARIO), 0, None, stride))
    configs, bindings = [], []
    for identity, row in zip(identities, rows):
        binding = dict(status="PREPARED_PFC_PARAMETER_CIRCUIT_CANDIDATE",
                       scenario_id=SCENARIO, circuit_id=identity["circuit_id"], circuit_index=identity["circuit_index"],
                       intended_split=identity["split"], construction_scope="development",
                       parameters={name: row[name] for name in parameters[0]},
                       effective_kernel_sha256=row["effective_kernel_sha256"],
                       parameter_identity=row["parameter_identity"], connectivity_identity=row["connectivity_identity"],
                       inputs=inputs, scientific_parameter_domain_qualified=False,
                       complete_episodes=0, production_eligible=False)
        config = copy.deepcopy(base)
        config.update(binding["parameters"])
        config["benchmark_circuit_candidate"] = binding
        # Reuse the real compiler for every declared condition. Keep the base
        # circuit free of condition-specific controls; prepare_episode owns them.
        for offset in range(campaign["sampling"]["conditions_per_block"]):
            episode = dict(identity, condition_index=offset,
                           episode_id=identity["block_id"] + f"_condition_{offset:02d}")
            compiled = pfc_working_memory_config(config, protocol, episode)
            _case_config(compiled, episode["episode_id"])
            if compiled["duration_ms"] != 5000.0 or compiled["dt_ms"] != protocol["dt_ms"]:
                raise ValueError("PFC_PARAMETER_DESIGN_COMPLETE_TASK_REQUIRED")
        configs.append(config)
        bindings.append(dict(circuit_id=identity["circuit_id"], circuit_index=identity["circuit_index"],
                             intended_split=identity["split"], **row))
    output.mkdir(parents=True)
    for config in configs:
        directory = output / config["benchmark_circuit_candidate"]["circuit_id"]
        directory.mkdir()
        (directory / "circuit.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    with (output / "circuits.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bindings[0]))
        writer.writeheader(); writer.writerows(bindings)
    report = dict(status="PREPARED_PFC_DISTINCT_KERNEL_CANDIDATES", settings=settings, inputs=inputs,
                  candidate_circuits=len(bindings), distinct_effective_kernels=len(bindings),
                  coupled_core_units=base["excitatory_neurons"] + base["inhibitory_neurons"],
                  compiled_condition_configs=len(bindings) * campaign["sampling"]["conditions_per_block"],
                  intended_split_counts=campaign["circuit_split_counts"], scientific_parameter_domain_qualified=False,
                  production_eligible=False, complete_episodes=0,
                  next_required_evidence=["exact-tuple native parity", "full-scale clock and population coverage",
                                          "complete ten-condition task and observation-map qualification"])
    freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "base-config", "parameter-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.config, args.base_config, args.parameter_config, args.output))


if __name__ == "__main__":
    main()
