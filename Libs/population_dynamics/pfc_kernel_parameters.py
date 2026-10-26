"""Pinned PFC E-to-E kernel parameters and existing circuit identity payloads.

These checks constrain the executable source formula, not a scientifically
qualified parameter domain. The default payload intentionally retains historical
identities; numerical clocks remain part of the existing parameter identity.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.special import erf

from .provenance import canonical_json

DEFAULT_KERNEL_PARAMETERS = {"sigma_weight_profile": 20.0, "Jpos_excit2excit": 1.6}


def _normalization(sigma, j_pos):
    # Preserve the pinned source operation order and scipy.special.erf.
    tmp = math.sqrt(2. * math.pi) * sigma * erf(180. / math.sqrt(2.) / sigma) / 360.
    if not math.isfinite(tmp) or not 0. < tmp < 1.:
        raise ValueError("PFC_SOURCE_KERNEL_NORMALIZATION_INVALID")
    j_neg = (1. - j_pos * tmp) / (1. - tmp)
    if not math.isfinite(j_neg):
        raise ValueError("PFC_SOURCE_KERNEL_NORMALIZATION_INVALID")
    return j_neg


def resolve_kernel_parameters(config):
    parameters = {key: config.get(key, value) for key, value in DEFAULT_KERNEL_PARAMETERS.items()}
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in parameters.values()):
        raise ValueError("PFC_SOURCE_KERNEL_PARAMETERS_INVALID")
    sigma, j_pos = parameters["sigma_weight_profile"], parameters["Jpos_excit2excit"]
    if sigma <= 0. or j_pos < 0.:
        raise ValueError("PFC_SOURCE_KERNEL_PARAMETERS_INVALID")
    sigma, j_pos = float(sigma), float(j_pos)
    try:
        width_squared = sigma ** 2
    except OverflowError as error:
        raise ValueError("PFC_SOURCE_KERNEL_NORMALIZATION_INVALID") from error
    if not math.isfinite(width_squared) or width_squared <= 0.:
        raise ValueError("PFC_SOURCE_KERNEL_NORMALIZATION_INVALID")
    j_neg = _normalization(sigma, j_pos)
    # The profile is monotone in unsigned ring distance 0..180. These endpoints
    # bound every realized weight, including the opposite unit for even N_E.
    endpoints = [j_neg + (j_pos - j_neg) * math.exp(-.5 * distance ** 2 / sigma ** 2)
                 for distance in (0., 180.)]
    if any(not math.isfinite(value) or value < 0. for value in endpoints):
        raise ValueError("PFC_SOURCE_KERNEL_NONPHYSICAL_WEIGHTS")
    return dict(sigma_weight_profile=sigma, Jpos_excit2excit=j_pos)


def source_kernel(n_e, config=None):
    if type(n_e) is not int or n_e < 2 or n_e % 2:
        raise ValueError("PFC_SOURCE_KERNEL_EVEN_POPULATION_REQUIRED")
    parameters = resolve_kernel_parameters({} if config is None else config)
    sigma, j_pos = parameters["sigma_weight_profile"], parameters["Jpos_excit2excit"]
    j_neg = _normalization(sigma, j_pos)
    # Keep the old scalar loop and expression exactly: vectorization can change
    # the default floating-point kernel and invalidate source replay receipts.
    values = np.asarray([j_neg + (j_pos - j_neg) * math.exp(
        -.5 * (360. * min(j, n_e - j) / n_e) ** 2 / sigma ** 2)
        for j in range(n_e)], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.):
        raise ValueError("PFC_SOURCE_KERNEL_NONPHYSICAL_WEIGHTS")
    return values


def circuit_provenance(config, source_entry_sha256):
    parameters = resolve_kernel_parameters(config)
    intrinsic = {key: config[key] for key in (
        "excitatory_neurons", "inhibitory_neurons", "weight_scaling_factor", "dt_ms")}
    topology = dict(excitatory_kernel="source_circulant_gaussian",
                    inhibitory_self_connections=False, ei_and_ie="source_all_to_all",
                    neurons=[config["excitatory_neurons"], config["inhibitory_neurons"]],
                    source_entry_sha256=source_entry_sha256)
    if parameters != DEFAULT_KERNEL_PARAMETERS:
        # Omitted and explicit defaults preserve the established default IDs.
        # Nondefault weights alter both existing identities, not only a label.
        intrinsic.update(parameters)
        topology.update(parameters)
    parameter = dict(intrinsic, source_entry_sha256=source_entry_sha256)
    return dict(parameter_identity=hashlib.sha256(canonical_json(parameter).encode()).hexdigest(),
                connectivity_identity=hashlib.sha256(canonical_json(topology).encode()).hexdigest())


def require_kernel_coverage(receipt, config, implementation_sha256):
    """Admit only the exact tuple and helper used in a new qualification."""
    expected = resolve_kernel_parameters(config)
    declared = receipt.get("validated_kernel_parameters")
    if (not isinstance(declared, list) or len(declared) != 1
            or not isinstance(declared[0], dict)
            or set(declared[0]) != set(DEFAULT_KERNEL_PARAMETERS)
            or resolve_kernel_parameters(declared[0]) != expected):
        raise ValueError("PFC_SOURCE_EXECUTION_KERNEL_NOT_QUALIFIED")
    if receipt.get("kernel_parameterization_sha256") != implementation_sha256:
        raise ValueError("PFC_SOURCE_EXECUTION_KERNEL_IMPLEMENTATION_NOT_QUALIFIED")
    return expected
