"""Pinned Mesocircuit parameters without importing or initializing NEST.

Native parameter derivation remains unchanged. Without random_streams, the
native single RNG is preserved. The optional four-stream block selects adapter
RNG behavior, not a source-identical numerical sequence. This module validates
that contract; the runner applies the streams. Neither mode qualifies science.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import runpy

import numpy as np
import yaml

from .mesocircuit_normalization import normalize_connectivity
from .mesocircuit_continuation import prepare_pulse_protocol

PINNED_REVISION = "f59ada6b3c3efb1fa18f96c638dd1741ebfb77e5"
PARAMETERS = "mesocircuit/parameterization/"
REQUIRED_FILES = frozenset({"LICENSE.rst", "mesocircuit/simulation/network.py",
    "mesocircuit/helpers/mpiops.py", PARAMETERS + "base_network_params.py",
    PARAMETERS + "base_simulation_params.py", PARAMETERS + "helpers_network.py"})
CONFIG_FIELDS = frozenset({"source_dir", "expected_nest_version", "run_class",
                          "local_num_threads", "network", "simulation"})
SIMULATOR_STREAMS = frozenset({"spatial_layout", "connectivity", "initial_state", "background_drive"})
GRID_RATIO_ATOL = 1e-9


def verify_source(source: Path) -> dict:
    """Validate the pinned intake manifest and every declared source file."""
    source = Path(source).resolve()
    metadata = json.loads((source / "source.json").read_text(encoding="utf-8"))
    if (metadata.get("revision") != PINNED_REVISION
            or metadata.get("repository") != "https://github.com/INM-6/mesocircuit-model"
            or metadata.get("license_file") != "LICENSE.rst"):
        raise ValueError("MESOCIRCUIT_SOURCE_PROVENANCE_MISMATCH")
    entries = metadata.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("MESOCIRCUIT_SOURCE_FILE_MANIFEST_REQUIRED")
    declared = set()
    for entry in entries:
        name = entry.get("path")
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or name in declared):
            raise ValueError("MESOCIRCUIT_SOURCE_PATH_INVALID")
        declared.add(name)
        path = (source / name).resolve()
        if not path.is_relative_to(source):
            raise ValueError("MESOCIRCUIT_SOURCE_PATH_ESCAPES_ROOT")
        payload = path.read_bytes()
        if (type(entry.get("bytes")) is not int or entry["bytes"] != len(payload)
                or hashlib.sha256(payload).hexdigest() != entry.get("sha256")):
            raise ValueError(f"MESOCIRCUIT_SOURCE_DIGEST_MISMATCH:{name}")
        blob = entry.get("git_blob_sha1")
        if blob is not None and hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest() != blob:
            raise ValueError(f"MESOCIRCUIT_GIT_BLOB_MISMATCH:{name}")
    if not REQUIRED_FILES.issubset(declared):
        raise ValueError("MESOCIRCUIT_REQUIRED_SOURCE_FILES_MISSING")
    return metadata


def _number(value, name, *, integer=False):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or (integer and value != int(value))):
        raise ValueError(f"FINITE_NUMERIC_PARAMETER_REQUIRED:{name}")
    return value


def _array(value, name, shape, *, integer=False):
    original = np.asarray(value, dtype=object)
    if any(isinstance(item, (bool, np.bool_)) for item in original.flat):
        raise ValueError(f"BOOLEAN_NUMERIC_ARRAY_REJECTED:{name}")
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all()
            or (integer and np.any(array != np.floor(array)))):
        raise ValueError(f"NATIVE_PARAMETER_ARRAY_INVALID:{name}")
    if integer and (np.any(array < np.iinfo(np.int64).min) or np.any(array >= 2**63)):
        raise ValueError(f"NATIVE_INTEGER_ARRAY_OUT_OF_RANGE:{name}")
    return array.astype(np.int64 if integer else np.float64)


def _merge_native(base, overrides, prefix=""):
    if not isinstance(overrides, dict) or set(overrides) - set(base):
        raise ValueError(f"UNKNOWN_NATIVE_PARAMETER:{prefix}")
    merged = deepcopy(base)
    for key, value in overrides.items():
        name, default = prefix + key, base[key]
        if key in {"populations", "neuron_model"}:
            if not np.array_equal(value, default):
                raise ValueError("NATIVE_POPULATION_AND_NEURON_MODEL_IDENTITY_REQUIRED")
        elif key == "beta_exc_inh" and value is not False:
            merged[key] = _array(value, name, (2,))
        elif key in {"indegree_scaling", "ext_indegree_scaling"}:
            width = 3 if key == "indegree_scaling" else 2
            shape = (len(value), width) if len(value) else (0,)
            merged[key] = _array(value, name, shape)
        elif isinstance(default, dict):
            merged[key] = _merge_native(default, value, name + ".")
        elif isinstance(default, np.ndarray):
            merged[key] = _array(value, name, default.shape, integer=default.dtype.kind in "iu")
        elif isinstance(default, bool):
            if type(value) is not bool:
                raise ValueError(f"BOOLEAN_PARAMETER_REQUIRED:{name}")
            merged[key] = value
        elif isinstance(default, str):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"STRING_PARAMETER_REQUIRED:{name}")
            merged[key] = value
        elif isinstance(default, list):
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"NATIVE_STRING_LIST_REQUIRED:{name}")
            merged[key] = deepcopy(value)
        else:
            merged[key] = _number(value, name, integer=isinstance(default, (int, np.integer)))
    return merged


def _header(cfg):
    if (not isinstance(cfg, dict) or not CONFIG_FIELDS.issubset(cfg)
            or set(cfg) - CONFIG_FIELDS - {"random_streams", "connectivity_normalization", "pulse_control", "kernel_update_interval", "export_initial_network"}):
        raise ValueError("MESOCIRCUIT_CONFIG_FIELDS_INVALID")
    if "export_initial_network" in cfg and type(cfg["export_initial_network"]) is not bool:
        raise ValueError("BOOLEAN_INITIAL_NETWORK_EXPORT_REQUIRED")
    update_policy = cfg.get("kernel_update_interval", "native")
    if type(update_policy) is not str or update_policy not in {"native", "resolution"}:
        raise ValueError("KERNEL_UPDATE_INTERVAL_POLICY_INVALID")
    policy = cfg.get("connectivity_normalization", "native_source")
    if type(policy) is not str or policy not in {"native_source", "kernel_integral"}:
        raise ValueError("MESOCIRCUIT_NORMALIZATION_POLICY_INVALID")
    if "random_streams" in cfg:
        streams = cfg["random_streams"]
        if not isinstance(streams, dict) or set(streams) != SIMULATOR_STREAMS:
            raise ValueError("MESOCIRCUIT_FOUR_SIMULATOR_STREAMS_REQUIRED")
        if any(type(seed) is not int or not 0 < seed < 2**32 for seed in streams.values()):
            raise ValueError("MESOCIRCUIT_SIMULATOR_STREAM_SEEDS_INVALID")
        if len(set(streams.values())) != len(SIMULATOR_STREAMS):
            raise ValueError("MESOCIRCUIT_SIMULATOR_STREAM_SEEDS_MUST_BE_DISTINCT")
        simulation = cfg["simulation"]
        if (not isinstance(simulation, dict) or type(simulation.get("rng_seed")) is not int
                or simulation["rng_seed"] != streams["spatial_layout"]):
            raise ValueError("MESOCIRCUIT_NATIVE_RNG_MUST_MATCH_SPATIAL_LAYOUT")
    if not isinstance(cfg["source_dir"], str) or not cfg["source_dir"].strip():
        raise ValueError("MESOCIRCUIT_SOURCE_DIRECTORY_REQUIRED")
    if cfg["expected_nest_version"] != "3.10.0":
        raise ValueError("MESOCIRCUIT_EXPECTED_NEST_VERSION_MUST_BE_3_10_0")
    if cfg["run_class"] not in {"engineering_diagnostic", "reference_candidate"}:
        raise ValueError("MESOCIRCUIT_RUN_CLASS_INVALID")
    threads = cfg["local_num_threads"]
    if type(threads) is not int or threads < 1:
        raise ValueError("POSITIVE_INTEGER_LOCAL_THREADS_REQUIRED")
    if cfg["run_class"] == "engineering_diagnostic" and threads != 1:
        raise ValueError("MESOCIRCUIT_ENGINEERING_REQUIRES_ONE_THREAD")


def _on_grid(value, dt):
    ratio = value / dt
    return math.isfinite(ratio) and math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=GRID_RATIO_ATOL)


def _validate_native(net, sim, cfg):
    choices = {"connect_method": {"fixedindegree_exp", "distr_indegree_exp", "distr_indegree_gauss"},
               "base_model": {"PD2014", "SvA2018"}, "V0_type": {"original", "optimized"},
               "delay_type": {"linear", "normal"}, "K_area_scale_method": {"PD2014", "beta"},
               "thalamic_input_type": {"pulses", "poisson"}}
    for key, allowed in choices.items():
        if net[key] not in allowed:
            raise ValueError(f"UNSUPPORTED_NATIVE_PARAMETER:{key}")
    if cfg.get("kernel_update_interval", "native") == "resolution" and net["delay_type"] != "linear":
        raise ValueError("RESOLUTION_UPDATE_REQUIRES_BOUNDED_LINEAR_DELAYS")
    if cfg.get("connectivity_normalization", "native_source") == "kernel_integral" and net["connect_method"] != "distr_indegree_exp":
        raise ValueError("KERNEL_INTEGRAL_REQUIRES_EXPONENTIAL_BERNOULLI")
    positive = ("extent", "N_scaling", "K_scaling", "PSP_exc_mean", "rel_weight_exc_to_inh",
                "beta_scaling", "beta_th", "mask_scaling", "delay_exc_mean", "delay_inh_mean",
                "delay_poisson", "th_interval", "th_delay_pulse_generator", "th_duration", "dc_dur",
                "num_neurons_th_1mm2")
    nonnegative = ("bg_rate", "weight_rel_std", "delay_rel_std", "th_start", "th_pulse_start", "th_rate", "dc_start",
                   "ext_indegree_scaling_global")
    for key in positive + nonnegative:
        if net[key] < 0 or (key in positive and net[key] == 0):
            raise ValueError(f"NATIVE_PARAMETER_DOMAIN_INVALID:{key}")
    if net["g"] >= 0 or net["th_rel_radius"] <= 0:
        raise ValueError("NATIVE_INHIBITION_OR_THALAMIC_RADIUS_INVALID")
    for key in ("beta_unscaled", "prop_speed_exc_inh", "num_neurons_1mm2_PD2014", "num_neurons_1mm2_SvA2018"):
        if np.any(net[key] <= 0):
            raise ValueError(f"NATIVE_POSITIVE_ARRAY_REQUIRED:{key}")
    if net["beta_exc_inh"] is not False and np.any(net["beta_exc_inh"] <= 0):
        raise ValueError("POSITIVE_BETA_PAIR_REQUIRED")
    for key in ("delay_offset_exc_inh", "K_ext_PD2014", "K_ext_SvA2018", "mean_rates_PD2014",
                "mean_rates_SvA2018", "indegrees_1mm2_SvA2018"):
        if np.any(net[key] < 0):
            raise ValueError(f"NATIVE_NONNEGATIVE_ARRAY_REQUIRED:{key}")
    for key in ("conn_probs_1mm2_PD2014", "conn_probs_th_1mm2"):
        if np.any((net[key] < 0) | (net[key] >= 1)):
            raise ValueError(f"NATIVE_CONNECTION_PROBABILITY_INVALID:{key}")
    for key in ("indegree_scaling", "ext_indegree_scaling"):
        rows = net[key]
        if len(rows):
            indices = rows[:, :-1]
            limits = np.array([9, 8] if key == "indegree_scaling" else [8])
            if (np.any(indices != np.floor(indices)) or np.any(indices < 0)
                    or np.any(indices >= limits) or np.any(rows[:, -1] < 0)):
                raise ValueError(f"NATIVE_POPULATION_SCALING_INVALID:{key}")
    neuron = net["neuron_params"]
    if (any(neuron[key] <= 0 for key in ("C_m", "tau_m", "tau_syn_default", "tau_syn_ex", "tau_syn_in", "t_ref"))
            or neuron["tau_m"] == neuron["tau_syn_default"]
            or any(np.any(np.asarray(value) < 0) for value in neuron["V0_std"].values())):
        raise ValueError("NATIVE_NEURON_PARAMETERS_INVALID")
    dt, total = sim["sim_resolution"], sim["t_presim"] + sim["t_sim"]
    if dt <= 0 or sim["t_presim"] < 0 or sim["t_sim"] <= 0 or sim["rec_V_int"] <= 0:
        raise ValueError("MESOCIRCUIT_SIMULATION_TIMES_INVALID")
    if not isinstance(sim["rec_dev"], list) or sim["rec_dev"] != list(dict.fromkeys(sim["rec_dev"])):
        raise ValueError("UNIQUE_NATIVE_RECORDERS_REQUIRED")
    if "spike_recorder" not in sim["rec_dev"] or set(sim["rec_dev"]) - {"spike_recorder", "voltmeter"}:
        raise ValueError("NATIVE_SPIKE_RECORDING_REQUIRED")
    if not 0 < sim["rng_seed"] < 2**32:
        raise ValueError("NATIVE_RNG_SEED_OUT_OF_RANGE")
    if Path(sim["fname_nodeids"]).name != sim["fname_nodeids"] or "\\" in sim["fname_nodeids"]:
        raise ValueError("NATIVE_NODE_ID_FILENAME_MUST_BE_LOCAL")
    grid = [sim["t_presim"], sim["t_sim"]]
    if "voltmeter" in sim["rec_dev"]:
        grid.append(sim["rec_V_int"])
    if net["thalamic_input"]:
        if net["thalamic_input_type"] == "pulses":
            # The native connection does not enable oversized periodic masks.
            if net["th_rel_radius"] > 0.5:
                raise ValueError("NATIVE_THALAMIC_MASK_EXCEEDS_PERIODIC_EXTENT")
            start, delay = net["th_pulse_start"], net["th_delay_pulse_generator"]
            if not delay >= dt or not delay < start < total:
                raise ValueError("NATIVE_THALAMIC_PULSE_WINDOW_INVALID")
            grid.extend((start, delay, start - delay, net["th_interval"]))
        else:
            if net["th_start"] + net["th_duration"] > total:
                raise ValueError("NATIVE_THALAMIC_POISSON_EXCEEDS_RUN")
            grid.extend((net["th_start"], net["th_duration"]))
    if net["dc_input"]:
        if net["dc_start"] + net["dc_dur"] > total:
            raise ValueError("NATIVE_DC_EXCEEDS_RUN")
        grid.extend((net["dc_start"], net["dc_dur"]))
    if not all(_on_grid(value, dt) for value in grid):
        raise ValueError("NATIVE_CONTROL_AND_SIMULATION_TIMES_MUST_ALIGN_WITH_GRID")
    if cfg["run_class"] == "engineering_diagnostic" and total > 1000:
        raise ValueError("MESOCIRCUIT_ENGINEERING_DURATION_EXCEEDS_1000_MS")


def _prepare(source, cfg):
    _header(cfg)
    verify_source(source)
    net = runpy.run_path(str(source / PARAMETERS / "base_network_params.py"))["net_dict"]
    sim = runpy.run_path(str(source / PARAMETERS / "base_simulation_params.py"))["sim_dict"]
    net = _merge_native(net, cfg["network"])
    sim = _merge_native(sim, cfg["simulation"])
    _validate_native(net, sim, cfg)
    if "pulse_control" in cfg and cfg["pulse_control"] is None:
        raise ValueError("PULSE_CONTROL_FIELDS_INVALID")
    prepare_pulse_protocol(net, sim, cfg.get("pulse_control"))
    return net, sim


def load_config(path: Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    _header(cfg)
    source = Path(cfg["source_dir"])
    cfg["source_dir"] = str(source if source.is_absolute() else (path.parents[2] / source).resolve())
    _prepare(Path(cfg["source_dir"]), cfg)
    return cfg


def derive_parameters(source: Path, cfg: dict) -> tuple[dict, dict]:
    source = Path(source).resolve()
    net, sim = _prepare(source, cfg)
    helper = runpy.run_path(str(source / PARAMETERS / "helpers_network.py"))
    net = helper["derive_dependent_parameters"](net)
    net = normalize_connectivity(net, cfg.get("connectivity_normalization", "native_source"))
    for key, value in net.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiu" and not np.isfinite(value).all():
            raise ValueError(f"NONFINITE_NATIVE_DERIVED_PARAMETER:{key}")
    if np.any(net["num_neurons"] < 1) or any(np.any(net[key] < 0) for key in ("indegrees", "ext_indegrees", "num_synapses")):
        raise ValueError("INVALID_NATIVE_DERIVED_COUNTS")
    if cfg["run_class"] == "engineering_diagnostic" and sum(map(int, net["num_neurons"])) > 2000:
        raise ValueError("MESOCIRCUIT_ENGINEERING_EXCEEDS_2000_NEURONS")
    return net, sim
