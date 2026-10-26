"""Evaluate the pinned EPFL dense projections without storing individual edges.

Only the three uniform projections are replaced. Source neuron equations,
RK2 updates, FFT E-to-E coupling, external PoissonInput, and stimulus operations
remain in the original module. This is a numerical execution comparator, not a
new neural model or a qualified large-network source.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

import brian2 as b2
import numpy as np

SOURCE_ENTRY_SHA256 = '416a914b9a358ae4b685d1a7119d14d777f31c70253f88e01b85d7ede993d720'
EXECUTION_NAME = 'factorized_uniform_projections'


class UniformProjection(b2.NetworkOperation):
    """A source-specific Brian operation with a restricted connect interface."""

    def __init__(self, source, target, *, summed_nmda=False):
        self.source_population = source
        self.target_population = target
        self.summed_nmda = summed_nmda
        self.connected = False
        self.exclude_self = False
        # Brian's native SummedVariableUpdater precedes state integration;
        # native zero-delay pre pathways follow thresholds, before reset.
        def apply_projection():
            self._apply_projection()
        super().__init__(apply_projection, clock=target.clock,
                         when='groups' if summed_nmda else 'synapses',
                         order=-1, name='pfc_uniform_projection*')

    def connect(self, *, condition=None, p=1.0):
        if self.connected or p != 1.0:
            raise ValueError('PFC_UNIFORM_PROJECTION_REQUIRES_ONE_COMPLETE_CONNECTION')
        same = self.source_population is self.target_population
        expected = 'i!=j' if same and not self.summed_nmda else None
        if condition != expected or (same and self.summed_nmda):
            raise ValueError('PFC_UNIFORM_PROJECTION_TOPOLOGY_UNSUPPORTED')
        self.exclude_self = same
        self.connected = True

    def before_run(self, run_namespace):
        if not self.connected:
            raise ValueError('PFC_UNIFORM_PROJECTION_NOT_CONNECTED')
        super().before_run(run_namespace)

    def _apply_projection(self):
        if self.summed_nmda:
            # Native numpy summed-variable accumulation visits presynaptic IDs
            # in order. cumsum retains that order, unlike pairwise np.sum.
            presynaptic = np.asarray(self.source_population.s_NMDA_[:])
            self.target_population.s_NMDA_total_[:] = np.cumsum(presynaptic, dtype=np.float64)[-1]
        else:
            # A single +count is algebraically equal but can round differently
            # from source np.add.at(..., 1.0) during a large synchronous burst.
            # Visit spikes in native order; each target sees the same sequence
            # of additions, including omission of the inhibitory self event.
            values = self.target_population.s_GABA_
            for index in self.source_population.spikes:
                if self.exclude_self:
                    values[:index] += 1.0
                    values[index + 1:] += 1.0
                else:
                    values[:] += 1.0


class UniformProjectionFactory:
    """Accept exactly the uniform Synapses calls made by the pinned source."""

    def __init__(self):
        self.projections = []

    def __call__(self, source, target, *, on_pre=None, delay=None, model=None, method=None):
        if (model == 's_NMDA_total_post = s_NMDA_pre : 1 (summed)'
                and on_pre is None and delay is None and method == 'rk2'):
            summed_nmda = True
        elif (on_pre == 's_GABA += 1.0' and model is None and method is None
              and (delay is None or float(delay / b2.ms) == 0.0)):
            summed_nmda = False
        else:
            raise ValueError('PFC_UNIFORM_PROJECTION_EQUATION_UNSUPPORTED')
        projection = UniformProjection(source, target, summed_nmda=summed_nmda)
        self.projections.append(projection)
        return projection


def install_uniform_projections(module: ModuleType) -> UniformProjectionFactory:
    """Patch only this freshly loaded pinned module, never global Brian state."""
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE_ENTRY_SHA256:
        raise ValueError('PFC_FACTORIZATION_REQUIRES_PINNED_SOURCE')
    if b2.__version__ != '2.5.4' or b2.prefs.codegen.target != 'numpy':
        raise ValueError('PFC_FACTORIZATION_REQUIRES_BRIAN_2_5_4_NUMPY')
    factory = UniformProjectionFactory()
    module.Synapses = factory
    return factory
