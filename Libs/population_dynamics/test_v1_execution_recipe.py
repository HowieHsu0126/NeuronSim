"""Code reuse admits independent seeds and rejects changed scientific recipes."""
from copy import deepcopy

import pytest

from .v1_execution_recipe import require_same_recipe


def circuit():
    return dict(schema_version=1, builder_seed=100000000, glif_assignment_seed=101000000,
                expected_full_nodes=230924, canary_fraction=0.01, full_fraction=1.0,
                edge_chunk_size=1000000, source_root="/reference/sources",
                bmtk_wheel=dict(path="/reference/bmtk.whl", sha256="a" * 64),
                benchmark_circuit_candidate=dict(circuit_id="circuit_000"))


def visual():
    return dict(scope="development_source_runtime_qualification", unit_coverage="model_representatives",
                converter_python="/usr/bin/python3",
                stimulus=dict(duration_s=3.0, gray_screen_s=0.5, direction_deg=0,
                              contrast_percent=80, spatial_frequency_cycles_per_degree=.04,
                              temporal_frequency_hz=2.0),
                spike_generation=dict(trials=1, starting_noise_seed=130000000),
                background=dict(kind="source_constant_rate_sampler", rate_hz=1000.0,
                                rate_clock_step_ms=.25, starting_noise_seed=120000000))


def test_circuit_reuse_preserves_recipe_and_input_mappings():
    reference = circuit()
    current = deepcopy(reference)
    current.update(builder_seed=100000001, glif_assignment_seed=101000001,
                   benchmark_circuit_candidate=dict(circuit_id="circuit_001"), source_root="/other/sources")
    current["bmtk_wheel"]["path"] = "/other/bmtk.whl"
    saved = deepcopy(current)
    require_same_recipe(current, reference, family="circuit")
    assert current == saved and reference == circuit()
    for field, value in (("expected_full_nodes", 1000), ("full_fraction", .01), ("extra_option", True)):
        changed = deepcopy(current); changed[field] = value
        with pytest.raises(ValueError, match="CIRCUIT_RECIPE_DIFFERS"):
            require_same_recipe(changed, reference, family="circuit")
    changed = deepcopy(current); changed["bmtk_wheel"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="CIRCUIT_RECIPE_DIFFERS"):
        require_same_recipe(changed, reference, family="circuit")


def test_visual_reuse_preserves_condition_and_sampler_recipe():
    reference = visual(); current = deepcopy(reference)
    current["unit_coverage"] = "all_source_units"
    current["spike_generation"]["starting_noise_seed"] = 130001001
    current["background"]["starting_noise_seed"] = 120001001
    saved = deepcopy(current)
    require_same_recipe(current, reference, family="visual")
    assert current == saved
    for section, field, value in (("stimulus", "direction_deg", 45),
                                   ("stimulus", "duration_s", 1.0),
                                   ("background", "rate_clock_step_ms", 1.0),
                                   ("spike_generation", "trials", 2)):
        changed = deepcopy(current); changed[section][field] = value
        with pytest.raises(ValueError, match="VISUAL_RECIPE_DIFFERS"):
            require_same_recipe(changed, reference, family="visual")


@pytest.mark.parametrize("value", (True, -1, 2**32, 41.0, None))
@pytest.mark.parametrize("family,section,field", (("circuit", None, "builder_seed"),
    ("circuit", None, "glif_assignment_seed"), ("visual", "background", "starting_noise_seed"),
    ("visual", "spike_generation", "starting_noise_seed")))
def test_invalid_seed_cannot_disappear_from_recipe_comparison(value, family, section, field):
    reference = circuit() if family == "circuit" else visual()
    current = deepcopy(reference)
    (current if section is None else current[section])[field] = value
    with pytest.raises(ValueError, match="UINT32_SEED_REQUIRED"):
        require_same_recipe(current, reference, family=family)
