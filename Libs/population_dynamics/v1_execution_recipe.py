"""Separate verified V1 recipes from independent realization seeds.

These comparisons qualify reuse of executable code. They do not qualify a
dataset, establish biological diversity, or permit reuse of generated outputs.
"""
from __future__ import annotations

from copy import deepcopy

UINT32_MAX = 2**32 - 1


def require_seed(value, field: str) -> None:
    if type(value) is not int or not 0 <= value <= UINT32_MAX:
        raise ValueError("V1_EXECUTION_UINT32_SEED_REQUIRED:" + field)


def circuit_recipe(config: dict) -> dict:
    """Compare all circuit settings except seeds, candidate metadata and paths."""
    if not isinstance(config, dict):
        raise ValueError("V1_EXECUTION_CIRCUIT_MAPPING_REQUIRED")
    for field in ("builder_seed", "glif_assignment_seed"):
        require_seed(config.get(field), field)
    if config["builder_seed"] == config["glif_assignment_seed"]:
        raise ValueError("V1_EXECUTION_SEPARATE_CIRCUIT_SEEDS_REQUIRED")
    recipe = deepcopy(config)
    for field in ("builder_seed", "glif_assignment_seed", "benchmark_circuit_candidate", "source_root"):
        recipe.pop(field, None)
    wheel = recipe.get("bmtk_wheel")
    if not isinstance(wheel, dict) or not wheel.get("sha256"):
        raise ValueError("V1_EXECUTION_PINNED_WHEEL_REQUIRED")
    wheel.pop("path", None)
    return recipe


def visual_recipe(config: dict) -> dict:
    """Compare one condition while admitting independent LGN/background draws."""
    if not isinstance(config, dict):
        raise ValueError("V1_EXECUTION_VISUAL_MAPPING_REQUIRED")
    recipe = deepcopy(config)
    coverage = recipe.pop("unit_coverage", None)
    if coverage not in ("model_representatives", "all_source_units"):
        raise ValueError("V1_EXECUTION_SOURCE_COVERAGE_REQUIRED")
    for field in ("spike_generation", "background"):
        section = recipe.get(field)
        if not isinstance(section, dict):
            raise ValueError("V1_EXECUTION_SOURCE_NOISE_SECTION_REQUIRED:" + field)
        require_seed(section.get("starting_noise_seed"), field + ".starting_noise_seed")
        section.pop("starting_noise_seed")
    return recipe


def require_same_recipe(current: dict, reference: dict, *, family: str) -> None:
    if family == "circuit":
        compare = circuit_recipe
    elif family == "visual":
        compare = visual_recipe
    else:
        raise ValueError("V1_EXECUTION_RECIPE_FAMILY_INVALID")
    if compare(current) != compare(reference):
        raise ValueError("V1_EXECUTION_" + family.upper() + "_RECIPE_DIFFERS")
