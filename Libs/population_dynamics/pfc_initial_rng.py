"""Separate pinned EPFL initial voltages from Brian's external-noise RNG.

Only the source module's NumPy reference changes. Its two uniform voltage draws
use a private RandomState; other NumPy operations retain their native behavior.
This separation defines a new source experimental configuration, not exact
reproduction of historical runs sharing one global initial/background seed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import FunctionType, ModuleType

import numpy as np

from .pfc_factorized_source import SOURCE_ENTRY_SHA256


@dataclass(frozen=True)
class InitialVoltageDraw:
    """Full source draw, indexed by ascending zero-based population neuron ID."""

    population: str
    values_mV: np.ndarray
    units: str = "mV"
    unit_order: str = "ascending_zero_based_population_index"


class InitialVoltageTrace:
    """One source initialization: inhibitory draw followed by excitatory draw."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng_name = "numpy.random.RandomState(MT19937)"
        self._rng = np.random.RandomState(seed)
        self._draws = []

    @property
    def draws(self) -> tuple[InitialVoltageDraw, ...]:
        return tuple(self._draws)

    @property
    def complete(self) -> bool:
        return len(self._draws) == 2

    @property
    def initial_voltage_mV(self) -> np.ndarray:
        """Return all E neurons then all I neurons, matching the archive order."""
        if not self.complete:
            raise ValueError("PFC_INITIAL_VOLTAGE_TRACE_INCOMPLETE")
        return np.concatenate((self._draws[1].values_mV, self._draws[0].values_mV))

    def uniform(self, low, *, high, size):
        """Accept exactly the two voltage draw forms in the pinned source."""
        if self.complete:
            raise ValueError("PFC_INITIAL_VOLTAGE_DRAW_ORDER_UNSUPPORTED")
        if (np.ndim(low) != 0 or np.ndim(high) != 0
                or low != -60.0 or high != -50.0):
            raise ValueError("PFC_INITIAL_VOLTAGE_BOUNDS_UNSUPPORTED")
        if (isinstance(size, (bool, np.bool_))
                or not isinstance(size, (int, np.integer)) or size <= 0):
            raise ValueError("PFC_INITIAL_VOLTAGE_SHAPE_UNSUPPORTED")
        values = self._rng.uniform(low, high=high, size=size)
        recorded = values.copy()
        recorded.setflags(write=False)
        population = ("I", "E")[len(self._draws)]
        self._draws.append(InitialVoltageDraw(population, recorded))
        return values


class _SourceNumpy:
    """A module-local facade; random operations are limited to source voltage."""

    def __init__(self, trace: InitialVoltageTrace):
        self.random = trace

    def __getattr__(self, name):
        return getattr(np, name)


def install_initial_voltage_rng(module: ModuleType, seed: int) -> InitialVoltageTrace:
    """Install a fresh private voltage stream and return its initially empty trace.

    Reinstallation starts a new stream for this module and leaves earlier traces
    intact. Install once per source simulation and require a complete trace before
    archiving it. This coexists with the module-local uniform-projection adapter.
    It neither seeds Brian nor changes the process-global NumPy random namespace.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("PFC_INITIAL_VOLTAGE_SEED_INVALID")
    if not isinstance(module, ModuleType) or not isinstance(getattr(module, "__file__", None), str):
        raise ValueError("PFC_INITIAL_VOLTAGE_REQUIRES_PINNED_SOURCE")
    try:
        source_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("PFC_INITIAL_VOLTAGE_REQUIRES_PINNED_SOURCE") from error
    if source_hash != SOURCE_ENTRY_SHA256:
        raise ValueError("PFC_INITIAL_VOLTAGE_REQUIRES_PINNED_SOURCE")
    source_function = getattr(module, "simulate_wm", None)
    if (not isinstance(source_function, FunctionType)
            or source_function.__globals__ is not vars(module)):
        raise ValueError("PFC_INITIAL_VOLTAGE_SOURCE_FUNCTION_UNSUPPORTED")
    if module.numpy is not np and not isinstance(module.numpy, _SourceNumpy):
        raise ValueError("PFC_INITIAL_VOLTAGE_NUMPY_NAMESPACE_UNSUPPORTED")
    trace = InitialVoltageTrace(seed)
    module.numpy = _SourceNumpy(trace)
    return trace
