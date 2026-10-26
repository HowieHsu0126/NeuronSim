import unittest

import brian2 as b2
import numpy as np

from .pfc_factorized_source import UniformProjection, UniformProjectionFactory


class UniformProjectionTests(unittest.TestCase):
    def setUp(self):
        b2.start_scope()
        b2.defaultclock.dt = 0.1*b2.ms
        b2.prefs.codegen.target = 'numpy'

    def test_inhibitory_self_exclusion_and_broadcast(self):
        source = b2.NeuronGroup(3, 's_GABA:1', threshold='t == (i+1)*ms', reset='')
        source.s_GABA = [0.125, 1.25, 2.5]
        target = b2.NeuronGroup(7, 's_GABA:1')
        native = b2.NeuronGroup(7, 's_GABA:1')
        dense = b2.Synapses(source, native, on_pre='s_GABA += 1.0')
        dense.connect(p=1.0)
        ii = UniformProjection(source, source)
        ii.connect(condition='i!=j', p=1.0)
        ie = UniformProjection(source, target)
        ie.connect(p=1.0)
        observed = b2.StateMonitor(source, 's_GABA', record=True, when='end')
        sparse = b2.StateMonitor(target, 's_GABA', record=True, when='end')
        reference = b2.StateMonitor(native, 's_GABA', record=True, when='end')
        net = b2.Network(source, target, native, dense, ii, ie, observed, sparse, reference)
        net.run(4*b2.ms)
        np.testing.assert_array_equal(sparse.s_GABA, reference.s_GABA)
        t = np.asarray(observed.t/b2.ms)
        emitted = t[None,:] >= np.array([1,2,3])[:,None]
        expected = np.array([0.125,1.25,2.5])[:,None] + emitted.sum(0) - emitted
        np.testing.assert_array_equal(observed.s_GABA, expected)

    def test_nmda_uses_native_presynaptic_order_and_schedule(self):
        source = b2.NeuronGroup(37, 'ds_NMDA/dt=-s_NMDA/(2*ms):1', method='rk2')
        source.s_NMDA = np.random.RandomState(17).uniform(0, 1, 37)
        factored = b2.NeuronGroup(5, 's_NMDA_total:1')
        native = b2.NeuronGroup(5, 's_NMDA_total:1')
        dense = b2.Synapses(source, native,
                           model='s_NMDA_total_post = s_NMDA_pre : 1 (summed)', method='rk2')
        dense.connect(p=1.0)
        projection = UniformProjection(source, factored, summed_nmda=True)
        projection.connect(p=1.0)
        a = b2.StateMonitor(native, 's_NMDA_total', record=True, when='end')
        c = b2.StateMonitor(factored, 's_NMDA_total', record=True, when='end')
        b2.Network(source, native, factored, dense, projection, a, c).run(4*b2.ms)
        np.testing.assert_array_equal(a.s_NMDA_total, c.s_NMDA_total)
        self.assertEqual(float(c.s_NMDA_total[0,0]), float(np.cumsum(np.random.RandomState(17).uniform(0,1,37))[-1]))

    def test_nmda_accumulates_float32_weights_in_native_float64(self):
        source = b2.NeuronGroup(1024, 's_NMDA:1', dtype=np.float32)
        source.s_NMDA = 0.1
        native = b2.NeuronGroup(3, 's_NMDA_total:1', dtype=np.float32)
        factored = b2.NeuronGroup(3, 's_NMDA_total:1', dtype=np.float32)
        dense = b2.Synapses(source, native,
                           model='s_NMDA_total_post = s_NMDA_pre : 1 (summed)', method='rk2')
        dense.connect(p=1.0)
        projection = UniformProjection(source, factored, summed_nmda=True)
        projection.connect(p=1.0)
        b2.Network(source, native, factored, dense, projection).run(0.1*b2.ms)
        np.testing.assert_array_equal(native.s_NMDA_total[:], factored.s_NMDA_total[:])
        wrong = np.cumsum(np.full(1024, 0.1, dtype=np.float32))[-1]
        self.assertNotEqual(float(native.s_NMDA_total[0]), float(wrong))

    def test_large_synchronous_burst_keeps_source_addition_order(self):
        source = b2.SpikeGeneratorGroup(512, np.arange(512), np.zeros(512)*b2.ms)
        native = b2.NeuronGroup(3, 's_GABA:1')
        factored = b2.NeuronGroup(3, 's_GABA:1')
        native.s_GABA = factored.s_GABA = 1.990025
        dense = b2.Synapses(source, native, on_pre='s_GABA += 1.0')
        dense.connect(p=1.0)
        projection = UniformProjection(source, factored)
        projection.connect(p=1.0)
        b2.Network(source, native, factored, dense, projection).run(0.1*b2.ms)
        np.testing.assert_array_equal(native.s_GABA[:], factored.s_GABA[:])
        self.assertNotEqual(float(native.s_GABA[0]), 1.990025 + 512)

    def test_reject_changed_topology_or_equation(self):
        source = b2.NeuronGroup(3, 's_GABA:1', threshold='False', reset='')
        target = b2.NeuronGroup(3, 's_GABA:1')
        factory = UniformProjectionFactory()
        with self.assertRaisesRegex(ValueError, 'EQUATION_UNSUPPORTED'):
            factory(source, target, on_pre='s_GABA += 2.0')
        with self.assertRaisesRegex(ValueError, 'EQUATION_UNSUPPORTED'):
            factory(source, target, on_pre='s_GABA += 1.0', delay=1*b2.ms)
        projection = factory(source, source, on_pre='s_GABA += 1.0')
        with self.assertRaisesRegex(ValueError, 'TOPOLOGY_UNSUPPORTED'):
            projection.connect(p=1.0)
        projection.connect(condition='i!=j', p=1.0)
        with self.assertRaisesRegex(ValueError, 'ONE_COMPLETE_CONNECTION'):
            projection.connect(condition='i!=j', p=1.0)
        b2.Network(source, target, projection).run(0*b2.ms)


if __name__ == '__main__':
    unittest.main()
