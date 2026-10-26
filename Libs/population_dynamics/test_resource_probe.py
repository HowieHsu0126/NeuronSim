"""Bounded telemetry failure reporting without a live GPU or /proc scan."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from . import resource_probe


class ResourceProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/'resource.json'
        snapshot = dict(d_state_count=0, memory_bytes={'MemAvailable': 128*1024**3},
                        disk_bytes={'/': dict(total=200*1024**3, used=80*1024**3, free=120*1024**3)})
        self.enterContext(mock.patch.object(resource_probe, 'snapshot', return_value=snapshot))
        self.enterContext(mock.patch.dict(os.environ, {'SLURM_JOB_ID': 'fixture', 'CUDA_VISIBLE_DEVICES': '5'}))
        telemetry = 'procs memory swap io system cpu\nr b swpd free buff cache si so bi bo in cs us sy id wa st\n'
        telemetry += '0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 0 0\n'*4
        self.enterContext(mock.patch.object(resource_probe.subprocess, 'check_output', return_value=telemetry))
        self.process = mock.Mock(pid=12345, returncode=0)
        self.popen = self.enterContext(mock.patch.object(resource_probe.subprocess, 'Popen', return_value=self.process))

    def test_gpu_success_is_pending_until_query_completes(self):
        def start(*args, **kwargs):
            self.assertEqual(json.loads(self.path.read_text())['status'], 'PENDING_GPU_TELEMETRY')
            return self.process
        self.popen.side_effect = start
        def complete(**kwargs):
            self.assertEqual(json.loads(self.path.read_text())['status'], 'PENDING_GPU_TELEMETRY')
            return '5, fixture, 24000, 20000, 0\n', None
        self.process.communicate.side_effect = complete
        report = resource_probe.probe(self.path, 64)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['allocated_gpu_query']['exit_code'], 0)
        self.process.communicate.assert_called_once_with(timeout=10)

    def test_uninterruptible_query_returns_blocked_and_signals_only_once(self):
        self.process.communicate.side_effect = subprocess.TimeoutExpired('nvidia-smi', 10)
        self.process.poll.return_value = None
        report = resource_probe.probe(self.path, 64)
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(report['reasons'], ['GPU_TELEMETRY_UNRESPONSIVE'])
        self.assertEqual(report['allocated_gpu_query']['pid'], 12345)
        self.assertIsNone(report['allocated_gpu_query']['exit_code'])
        self.process.terminate.assert_called_once_with()
        self.process.wait.assert_not_called()
        self.process.kill.assert_not_called()
        self.assertEqual(json.loads(self.path.read_text()), report)

    def test_failed_query_preserves_failure_output(self):
        self.process.returncode = 1
        self.process.communicate.return_value = ('driver unavailable\n', None)
        report = resource_probe.probe(self.path)
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertIn('GPU_TELEMETRY_FAILED', report['reasons'])
        self.assertEqual(report['allocated_gpu_query']['output'], 'driver unavailable')

    def test_cpu_probe_does_not_start_gpu_query(self):
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            report = resource_probe.probe(self.path)
        self.assertEqual(report['status'], 'PASS')
        self.popen.assert_not_called()

    def test_blocked_host_does_not_start_an_unnecessary_driver_query(self):
        report = resource_probe.probe(self.path, 256)
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(report['reasons'], ['INSUFFICIENT_AVAILABLE_MEMORY'])
        self.popen.assert_not_called()

    def test_critical_root_storage_blocks_before_driver_query(self):
        snapshot = dict(d_state_count=0, memory_bytes={'MemAvailable': 128*1024**3},
                        disk_bytes={'/': dict(total=70*1024**3, used=55*1024**3, free=15*1024**3)})
        with mock.patch.object(resource_probe, 'snapshot', return_value=snapshot):
            report = resource_probe.probe(self.path, 64)
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(report['reasons'], ['ROOT_STORAGE_CRITICAL'])
        self.assertEqual(json.loads(self.path.read_text()), report)
        self.popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
