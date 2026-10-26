"""Pinned source integrity and independent native-parameter contracts."""
from copy import deepcopy
import builtins
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from . import mesocircuit_source as source_layer

SOURCE = Path(__file__).resolve().parents[1] / "external" / "mesocircuit_model"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "neural_population_mesocircuit.yaml"
DENSITY = np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902])


@pytest.fixture
def cfg():
    result = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    result["source_dir"] = str(SOURCE)
    return result


def write_config(tmp_path, cfg):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def copied_source(tmp_path):
    destination = tmp_path / "source"
    metadata = json.loads((SOURCE / "source.json").read_text())
    for name in ["source.json", *(entry["path"] for entry in metadata["files"])]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, target)
    return destination


def test_provenance_and_all_declared_files_are_verified():
    metadata = source_layer.verify_source(SOURCE)
    assert metadata["revision"] == source_layer.PINNED_REVISION
    assert metadata["scientific_reference_qualified"] is False
    assert source_layer.REQUIRED_FILES <= {entry["path"] for entry in metadata["files"]}


def test_metadata_and_parameter_derivation_do_not_import_nest(tmp_path, cfg, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "nest" or name.startswith("nest."):
            raise AssertionError("Configuration must not initialize NEST")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    validated = source_layer.load_config(write_config(tmp_path, cfg))
    net, sim = source_layer.derive_parameters(SOURCE, validated)
    np.testing.assert_array_equal(net["num_neurons"], [207, 58, 219, 55, 48, 11, 144, 29, 9])
    assert sum(net["num_neurons"]) == 780
    assert sim["rng_seed"] == 55001 and sim["rec_dev"] == ["spike_recorder"]
    assert net["th_radius"] == 0.5


@pytest.mark.parametrize("extent,scale", [(1.0, 0.01), (1.1, 0.01), (0.37, 0.031)])
def test_native_neuron_counts_follow_unrounded_density_area_scaling(cfg, extent, scale):
    cfg["network"].update(extent=extent, N_scaling=scale)
    net, _ = source_layer.derive_parameters(SOURCE, cfg)
    expected = np.rint(DENSITY * extent**2 * scale).astype(int)
    np.testing.assert_array_equal(net["num_neurons"], expected)
    np.testing.assert_array_equal(net["full_num_neurons"], np.rint(DENSITY * extent**2).astype(int))


def test_reference_candidate_can_exceed_unity_scaling_without_qualification(cfg):
    cfg["run_class"] = "reference_candidate"
    cfg["network"].update(N_scaling=1.3, K_scaling=1.1)
    net, _ = source_layer.derive_parameters(SOURCE, cfg)
    np.testing.assert_array_equal(net["num_neurons"], np.rint(DENSITY * 1.3).astype(int))
    assert net["N_scaling"] == 1.3 and net["K_scaling"] == 1.1
    assert source_layer.verify_source(SOURCE)["scientific_reference_qualified"] is False


def test_engineering_bound_uses_actual_derived_cortical_and_thalamic_counts(cfg):
    cfg["network"]["N_scaling"] = 0.1
    with pytest.raises(ValueError, match="2000_NEURONS"):
        source_layer.derive_parameters(SOURCE, cfg)


def test_native_dc_and_thalamic_parameters_are_not_reinterpreted(cfg):
    cfg["network"].update(dc_input=True, dc_amp=0.2, dc_start=60.0, dc_dur=40.0)
    net, sim = source_layer.derive_parameters(SOURCE, cfg)
    assert net["dc_amp"] == 0.2 and net["dc_start"] == 60.0 and net["dc_dur"] == 40.0
    np.testing.assert_array_equal(net["K_ext_PD2014"], [1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])
    assert net["th_pulse_start"] - net["th_delay_pulse_generator"] == 59.0
    assert sim["t_presim"] + sim["t_sim"] == 150.0


@pytest.mark.parametrize("section,key,value", [
    ("network", "extent", np.nan), ("network", "N_scaling", True),
    ("network", "K_scaling", 0.0), ("network", "extent", [1.0]),
    ("network", "connect_method", "fixedindegree"), ("network", "num_neurons", [1] * 9),
    ("network", "neuron_model", "parrot_neuron"), ("network", "beta_exc_inh", [True, 0.2]),
    ("network", "prop_speed_exc_inh", [0.3]), ("network", "beta_unscaled", [[0.2]]),
    ("network", "indegree_scaling", [[0.5, 1, 0.9]]),
    ("network", "th_rel_radius", 0.6), ("network", "th_pulse_start", 60.05),
    ("network", "th_pulse_start", 1.0), ("network", "th_interval", 0.0),
    ("network", "dc_input", 1), ("simulation", "rng_seed", True),
    ("simulation", "rng_seed", 1.5), ("simulation", "sim_resolution", np.inf),
    ("simulation", "t_presim", -1), ("simulation", "t_sim", 1000.1),
    ("simulation", "rec_dev", []), ("simulation", "rec_dev", None),
    ("simulation", "rec_dev", ["spike_recorder", "spike_recorder"]),
    ("simulation", "unknown", 1),
])
def test_invalid_override_semantics_are_rejected(tmp_path, cfg, section, key, value):
    cfg[section][key] = value
    with pytest.raises((ValueError, TypeError)):
        source_layer.load_config(write_config(tmp_path, cfg))


@pytest.mark.parametrize("enabled,kind", [(False, "pulses"), (True, "poisson")])
def test_unused_thalamic_mask_has_no_half_extent_bound(tmp_path, cfg, enabled, kind):
    cfg["network"].update(thalamic_input=enabled, thalamic_input_type=kind,
                          th_rel_radius=0.75, th_start=60.0, th_duration=20.0)
    validated = source_layer.load_config(write_config(tmp_path, cfg))
    assert validated["network"]["th_rel_radius"] == 0.75


def test_zero_background_rate_is_not_rejected_before_native_derivation(tmp_path, cfg):
    cfg["network"]["bg_rate"] = 0.0
    validated = source_layer.load_config(write_config(tmp_path, cfg))
    assert validated["network"]["bg_rate"] == 0.0
    # The pinned mean-input compensation divides by bg_rate. The adapter does
    # not replace that source equation or accept its nonfinite derived values.
    with np.errstate(divide="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="NONFINITE_NATIVE|INVALID_NATIVE_DERIVED"):
            source_layer.derive_parameters(SOURCE, validated)


def test_unknown_top_level_training_seed_is_not_native_rng(tmp_path, cfg):
    cfg["seed"] = 41
    with pytest.raises(ValueError):
        source_layer.load_config(write_config(tmp_path, cfg))


def test_enabled_native_control_must_fit_duration_and_grid(tmp_path, cfg):
    cfg["network"].update(thalamic_input_type="poisson", th_start=140.0, th_duration=20.0)
    with pytest.raises(ValueError, match="EXCEEDS_RUN"):
        source_layer.load_config(write_config(tmp_path, cfg))


def test_source_tamper_is_rejected_before_runpy(copied_source, cfg, monkeypatch):
    path = copied_source / "mesocircuit/parameterization/helpers_network.py"
    path.write_bytes(path.read_bytes() + b"\n# tampered\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("Unverified source executed")
    monkeypatch.setattr(source_layer.runpy, "run_path", forbidden)
    with pytest.raises(ValueError, match="DIGEST"):
        source_layer.derive_parameters(copied_source, cfg)


@pytest.mark.parametrize("change", ["revision", "missing_license", "duplicate_path", "unsafe_path"])
def test_provenance_cannot_omit_required_files_or_redirect_reads(copied_source, change):
    path = copied_source / "source.json"
    metadata = json.loads(path.read_text())
    if change == "revision":
        metadata["revision"] = "0" * 40
    elif change == "missing_license":
        metadata["files"] = [entry for entry in metadata["files"] if entry["path"] != "LICENSE.rst"]
    elif change == "duplicate_path":
        metadata["files"].append(deepcopy(metadata["files"][0]))
    else:
        metadata["files"][0]["path"] = "../outside"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        source_layer.verify_source(copied_source)



def test_complete_simulator_streams_preserve_declared_integer_seeds(tmp_path, cfg):
    validated = source_layer.load_config(write_config(tmp_path, cfg))
    assert validated == cfg
    assert validated["random_streams"] == {
        "spatial_layout": 55001, "connectivity": 55002,
        "initial_state": 55003, "background_drive": 55004,
    }
    assert all(type(seed) is int for seed in validated["random_streams"].values())
    assert validated["simulation"]["rng_seed"] == validated["random_streams"]["spatial_layout"]


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "bool", "float", "zero",
                                     "negative", "upper_bound", "not_mapping"])
def test_invalid_simulator_stream_blocks_are_rejected(tmp_path, cfg, change):
    streams = cfg["random_streams"]
    if change == "missing":
        del streams["background_drive"]
    elif change == "extra":
        streams["training"] = 41
    elif change == "duplicate":
        streams["background_drive"] = streams["connectivity"]
    elif change == "not_mapping":
        cfg["random_streams"] = None
    else:
        streams["initial_state"] = {"bool": True, "float": 55003.0, "zero": 0,
                                    "negative": -1, "upper_bound": 2**32}[change]
    with pytest.raises(ValueError, match="MESOCIRCUIT_.*STREAM"):
        source_layer.load_config(write_config(tmp_path, cfg))


@pytest.mark.parametrize("native_seed", [55002, 55001.0, True, None])
def test_native_constructor_seed_must_explicitly_match_layout(tmp_path, cfg, native_seed):
    if native_seed is None:
        del cfg["simulation"]["rng_seed"]
    else:
        cfg["simulation"]["rng_seed"] = native_seed
    with pytest.raises(ValueError, match="NATIVE_RNG_MUST_MATCH_SPATIAL_LAYOUT"):
        source_layer.load_config(write_config(tmp_path, cfg))


def test_legacy_single_rng_config_remains_unchanged(tmp_path, cfg):
    del cfg["random_streams"]
    cfg["simulation"]["rng_seed"] = 55
    legacy = deepcopy(cfg)
    validated = source_layer.load_config(write_config(tmp_path, cfg))
    assert validated == legacy
    assert "random_streams" not in validated
    assert validated["simulation"]["rng_seed"] == 55


def test_kernel_integral_variant_retains_native_parameters_and_raw_target(cfg):
    native, original_sim = source_layer.derive_parameters(SOURCE, cfg)
    variant_cfg = deepcopy(cfg)
    variant_cfg["connectivity_normalization"] = "kernel_integral"
    variant, variant_sim = source_layer.derive_parameters(SOURCE, variant_cfg)
    for key in native:
        if key not in {"p0_raw", "p0", "repeat_connect"}:
            np.testing.assert_equal(native[key], variant[key])
    np.testing.assert_equal(original_sim, variant_sim)
    mass = 1 - np.exp(-native["mask_radius"] / native["beta"]) * (1 + native["mask_radius"] / native["beta"])
    np.testing.assert_allclose(variant["p0_raw"], native["p0_raw"] / mass**2, rtol=1e-13)
    np.testing.assert_array_equal(variant["source_p0_raw"], native["p0_raw"])
    np.testing.assert_array_equal(variant["source_p0"], native["p0"])
    np.testing.assert_array_equal(variant["source_repeat_connect"], native["repeat_connect"])
    assert np.all((variant["p0"] >= 0) & (variant["p0"] <= 1))


@pytest.mark.parametrize("policy", ["rounded_indegree", True, None, ["kernel_integral"]])
def test_unknown_normalization_policy_rejected_by_config(tmp_path, cfg, policy):
    cfg["connectivity_normalization"] = policy
    with pytest.raises(ValueError, match="NORMALIZATION_POLICY"):
        source_layer.load_config(write_config(tmp_path, cfg))


def test_kernel_integral_rejects_nonexponential_profile_in_config(tmp_path, cfg):
    cfg["connectivity_normalization"] = "kernel_integral"
    cfg["network"]["connect_method"] = "distr_indegree_gauss"
    with pytest.raises(ValueError, match="REQUIRES_EXPONENTIAL"):
        source_layer.load_config(write_config(tmp_path, cfg))
