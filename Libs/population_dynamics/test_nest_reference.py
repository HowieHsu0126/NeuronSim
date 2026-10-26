"""Focused contracts for the official-source NEST adapter (no native simulation)."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Libs.population_dynamics.nest_reference import (
    NATIVE_POPULATIONS, SOURCE_FILES, fingerprint_connections, pulse_schedule,
    validate_config, verify_source,
)


def configuration():
    return dict(run_class='engineering_diagnostic', duration_ms=100., dt_ms=.1,
                presimulation_ms=0., n_scaling=.01, k_scaling=.01,
                local_num_threads=1, fingerprint_target_batch_size=2,
                random_streams=dict(connectivity=41, initial_state=42, background_drive=43),
                initial_state='optimized', poisson_input=True, background_rate_hz=8.,
                current_pulses=[])


class AdapterContracts(unittest.TestCase):
    def test_currents_preserve_baseline_and_native_population_names(self):
        pulses = [dict(population='L23E', onset_ms=1., duration_ms=3., amplitude_pa=7.),
                  dict(population='L23E', onset_ms=2., duration_ms=1., amplitude_pa=-2.)]
        baseline = np.arange(8, dtype=float) + 10.
        edges, values = pulse_schedule(pulses, baseline, 5.)
        np.testing.assert_array_equal(edges, [0., 1., 2., 3., 4., 5.])
        np.testing.assert_array_equal(values[:, 0], [10., 17., 15., 17., 10., 10.])
        np.testing.assert_array_equal(values[:, 1:], np.broadcast_to(baseline[1:], (6, 7)))
        pulses[0]['population'] = 'L2e'
        with self.assertRaisesRegex(ValueError, 'unknown current target'):
            pulse_schedule(pulses, baseline, 5.)

    def test_grid_misalignment_is_rejected(self):
        cfg = configuration()
        cfg['current_pulses'] = [dict(population='L4E', onset_ms=.15, duration_ms=1., amplitude_pa=1.)]
        with self.assertRaisesRegex(ValueError, 'align'):
            validate_config(cfg)

    def test_diagnostic_execution_bounds(self):
        for key, value in [('n_scaling', .02), ('k_scaling', .02),
                           ('duration_ms', 1001.), ('local_num_threads', 2)]:
            cfg = configuration()
            cfg[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'diagnostic requires'):
                validate_config(cfg)
        cfg = configuration()
        cfg.update(run_class='reference_candidate', n_scaling=1., k_scaling=1.)
        with self.assertRaisesRegex(ValueError, 'four threads'):
            validate_config(cfg)

    def test_named_streams_reject_ambiguous_seeds(self):
        for value in (True, 0, 1.5, 2**32):
            cfg = configuration()
            cfg['random_streams']['initial_state'] = value
            with self.subTest(seed=value), self.assertRaisesRegex(ValueError, 'seeds'):
                validate_config(cfg)

    def test_source_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            entries = []
            for name in SOURCE_FILES:
                content = name.encode()
                (source / name).write_bytes(content)
                entries.append(dict(path=name, sha256=hashlib.sha256(content).hexdigest()))
            (source / 'source.json').write_text(json.dumps(dict(git_tag='v3.8', files=entries)))
            verify_source(source)
            (source / 'network.py').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                verify_source(source)

    def test_connectivity_hash_is_chunk_independent_and_sensitive(self):
        class Connections:
            def __init__(self, records):
                self.records = records
            def get(self, names):
                return {name: [record[name] for record in self.records] for name in names}
        class Backend:
            def __init__(self):
                self.edges = [dict(source=9, target=2, weight=2., delay=.2),
                              dict(source=3, target=1, weight=-1., delay=.1),
                              dict(source=9, target=1, weight=3., delay=.3),
                              dict(source=1, target=3, weight=4., delay=.1)]
            @staticmethod
            def NodeCollection(ids):
                return ids
            def GetConnections(self, target, synapse_model):
                self.last_model = synapse_model
                return Connections([edge for edge in self.edges if edge['target'] in target])
        backend = Backend()
        ids = np.array([1, 2, 3])
        first = fingerprint_connections(backend, ids, 1)
        second = fingerprint_connections(backend, ids, 3)
        self.assertEqual(first['sha256'], second['sha256'])
        self.assertEqual(first['connection_count'], 4)
        self.assertEqual((first['delay_min_ms'], first['delay_max_ms']), (.1, .3))
        self.assertEqual(first['largest_batch_connections'], 2)
        backend.edges[0]['weight'] += .1
        self.assertNotEqual(first['sha256'], fingerprint_connections(backend, ids, 1)['sha256'])


if __name__ == '__main__':
    unittest.main()
