"""Recover source Poisson means from archived LFADS generator states.

This deterministic readout does not infer a state, advance a recurrent network,
sample spikes, or calibrate the model. All output neurons remain represented.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

SOURCE_REVISION = 'fd700f0cb2e104544c445d9fbf3991d8388ff18a'
L2_EPSILON = 1e-12
GENERATOR_WEIGHT = 'LFADS/gen/gen_2_fac/W:0'
OUTPUT_WEIGHT = 'LFADS/glm/fac_2_logrates_recording/W:0'
OUTPUT_BIAS = 'LFADS/glm/fac_2_logrates_recording/b:0'


@dataclass
class LFADSReadout:
    generator_weight: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray
    unit_ids: np.ndarray
    bin_seconds: float
    clip_value: float

    def __post_init__(self):
        g, w, b = self.generator_weight, self.output_weight, self.output_bias
        if (g.ndim != 2 or min(g.shape) < 1 or w.ndim != 2
                or w.shape[0] != g.shape[1] or b.shape != (1, w.shape[1])
                or self.unit_ids.shape != (w.shape[1],) or not len(self.unit_ids) or self.unit_ids.dtype.kind not in 'iu'
                or len(np.unique(self.unit_ids)) != len(self.unit_ids) or np.any(self.unit_ids < 0)
                or not np.isfinite(self.bin_seconds) or self.bin_seconds <= 0
                or not np.isfinite(self.clip_value) or not 0 < self.clip_value <= 80
                or any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (g, w, b))):
            raise ValueError('source float32 readout shapes, identities and clock required')
        # Source init_linear uses tf.nn.l2_normalize(dim=0). Its prose says
        # rows, but the executable operation normalizes full weight columns.
        squared_norm = np.sum(g * g, axis=0, dtype=np.float32, keepdims=True)
        self.normalized_generator_weight = g * (1 / np.sqrt(np.maximum(squared_norm, np.float32(L2_EPSILON))))

    @classmethod
    def from_checkpoint(cls, directory):
        directory = Path(directory)
        model = json.loads((directory / 'model.json').read_text())
        config = model['model_config']
        if (model['schema'] != 'neural_population_lfads_conditional_v2'
                or model['source_revision'] != SOURCE_REVISION):
            raise ValueError('pinned LFADS conditional checkpoint required')
        if config['factors_dim'] == len(model['unit_ids']):
            raise ValueError('identity output transform is outside this diagnostic contract')
        with np.load(directory / 'weights.npz', allow_pickle=False) as archive:
            values = [archive[name] for name in (GENERATOR_WEIGHT, OUTPUT_WEIGHT, OUTPUT_BIAS)]
        if values[0].shape != (config['gen_dim'], config['factors_dim']):
            raise ValueError('checkpoint generator dimensions differ from readout')
        return cls(*values, np.asarray(model['unit_ids'], dtype=np.int64),
                   model['bin_seconds'], config['_clip_value'])

    def factors(self, states):
        states = np.asarray(states)
        if (states.ndim < 2 or states.shape[-1] != self.generator_weight.shape[0]
                or states.dtype != np.float32 or not np.isfinite(states).all()):
            raise ValueError('finite source float32 generator states required')
        return states @ self.normalized_generator_weight

    def log_counts(self, factors, start, stop):
        return factors @ self.output_weight[:, start:stop] + self.output_bias[:, start:stop]

    def expected_field_counts(self, states, unit_field_index, field_count, *, chunk_units=4096):
        """Exponentiate each unit before summing into the exact source field map."""
        mapping = np.asarray(unit_field_index)
        if (mapping.shape != self.unit_ids.shape or mapping.dtype.kind not in 'iu'
                or type(field_count) is not int or field_count < 1
                or np.any(mapping < 0) or np.any(mapping >= field_count)
                or type(chunk_units) is not int or chunk_units < 1):
            raise ValueError('complete source-indexed field map and positive chunk required')
        factors = self.factors(states)
        shape = factors.shape[:-1]
        flat = factors.reshape(-1, factors.shape[-1])
        fields = np.zeros((len(flat), field_count), dtype=np.float64)
        minimum, maximum = np.inf, -np.inf
        for start in range(0, len(mapping), chunk_units):
            stop = min(start + chunk_units, len(mapping))
            logits = self.log_counts(flat, start, stop)
            if not np.isfinite(logits).all() or np.any(np.abs(logits) >= self.clip_value):
                raise FloatingPointError('LFADS_SOURCE_LOG_RATE_CLIP_ACTIVE_OR_NONFINITE')
            minimum = min(minimum, float(logits.min()))
            maximum = max(maximum, float(logits.max()))
            expected = np.exp(logits)
            projection = csr_matrix((np.ones(stop-start), (mapping[start:stop], np.arange(stop-start))),
                                    shape=(field_count, stop-start))
            fields += (projection @ expected.T).T
        return fields.reshape(*shape, field_count), dict(minimum_log_count=minimum, maximum_log_count=maximum)
