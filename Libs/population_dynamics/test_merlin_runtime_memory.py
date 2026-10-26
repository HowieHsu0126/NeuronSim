"""Test Merlin runtime memory admission without scientific inputs or third-party packages."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import ca3_merlin_remote as runtime
import v1_merlin_readout as readout


GIB = 1024 ** 3
NO_CGROUP_MOUNT = "31 20 0:26 / / rw,relatime - overlay overlay rw\n"
CGROUP_V2_MOUNT = "33 31 0:29 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
CGROUP_V1_MOUNT = "34 31 0:30 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"


class MerlinRuntimeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.files = {
            "/proc/meminfo": "MemTotal: 536870912 kB\nMemAvailable: 268435456 kB\n",
            "/proc/self/mountinfo": NO_CGROUP_MOUNT,
        }
        self.shape = dict(name="head", cpu=4, gpu=0, memory_mb=16384)
        self.stack.enter_context(mock.patch.dict(os.environ, {"ARNOLD_HEAD_MEM": "16384"}, clear=True))

        def read_text(path, *args, **kwargs):
            if str(path) not in self.files:
                raise FileNotFoundError(str(path))
            return self.files[str(path)]

        self.stack.enter_context(mock.patch.object(Path, "read_text", read_text))
        self.stack.enter_context(mock.patch.object(Path, "is_file", lambda path: str(path) in self.files))
        self.stack.enter_context(mock.patch.object(os, "sched_getaffinity", return_value={0, 1, 2, 3}))

    def controller(self, version, limit, usage):
        if version == "v2":
            self.files["/proc/self/mountinfo"] += CGROUP_V2_MOUNT
            names = ("memory.max", "memory.current")
        else:
            self.files["/proc/self/mountinfo"] += CGROUP_V1_MOUNT
            names = ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes")
        for name, value in zip(names, (limit, usage)):
            if value is not None:
                self.files["/sys/fs/cgroup/" + name] = str(value)

    def test_observed_no_cgroup_platform_reports_separate_scopes(self):
        result = readout.cpu_preflight(Path("/tmp/work"), self.shape)
        evidence = result["memory_evidence"]
        self.assertEqual(evidence["admission"]["status"], "PASS")
        self.assertEqual(evidence["admission"]["scope"], "merlin_role_allocation_and_host_available_memory")
        self.assertFalse(evidence["admission"]["container_available_memory_verified"])
        self.assertEqual(evidence["platform_allocation"]["allocated_memory_mib"], 16384)
        self.assertEqual(evidence["host_memory"]["scope"], "host")
        self.assertEqual(evidence["host_memory"]["memory_bytes"]["MemAvailable"], 256 * GIB)
        self.assertEqual(evidence["cgroup"]["status"], "cgroup_not_exposed")
        self.assertEqual(evidence["cgroup"]["mounts"], [])
        self.assertNotIn("effective_available", json.dumps(result))

    def test_worker_role_uses_actual_worker_allocation(self):
        os.environ.clear()
        os.environ["ARNOLD_WORKER_MEM"] = "242688"
        shape = dict(name="worker", cpu=15, gpu=1, memory_mb=242688)
        with mock.patch.object(runtime.subprocess, "check_output", return_value="ext4\n"), \
                mock.patch.object(runtime.shutil, "disk_usage", return_value=SimpleNamespace(free=500 * GIB)):
            result = runtime.resources({"resource_shape": shape}, Path("/tmp/work"))
        evidence = result["memory_evidence"]
        self.assertEqual(evidence["platform_allocation"]["environment_variable"], "ARNOLD_WORKER_MEM")
        self.assertEqual(evidence["admission"]["required_available_memory_gib"], 128)
        self.assertEqual(result["original_requested_memory_gib"], 160)
        self.assertNotIn("effective_available", json.dumps(result))

    def test_missing_platform_allocation_is_rejected(self):
        os.environ.clear()
        with self.assertRaisesRegex(ValueError, "MERLIN_PLATFORM_MEMORY_ALLOCATION_REQUIRED:ARNOLD_HEAD_MEM"):
            runtime.memory_evidence(self.shape, 8)

    def test_platform_allocation_below_request_is_rejected(self):
        os.environ["ARNOLD_HEAD_MEM"] = "8192"
        with self.assertRaisesRegex(ValueError, "MERLIN_PLATFORM_MEMORY_ALLOCATION_BELOW_REQUEST"):
            runtime.memory_evidence(self.shape, 8)

    def test_low_host_memory_is_rejected(self):
        self.files["/proc/meminfo"] = "MemTotal: 536870912 kB\nMemAvailable: 7340032 kB\n"
        with self.assertRaisesRegex(ValueError, "MERLIN_HOST_AVAILABLE_MEMORY_FLOOR_NOT_MET"):
            readout.cpu_preflight(Path("/tmp/work"), self.shape)

    def test_ca3_host_floor_remains_128_gib(self):
        os.environ["ARNOLD_WORKER_MEM"] = "242688"
        self.files["/proc/meminfo"] = "MemTotal: 536870912 kB\nMemAvailable: 133169152 kB\n"
        shape = dict(name="worker", cpu=15, gpu=1, memory_mb=242688)
        with self.assertRaisesRegex(ValueError, "MERLIN_HOST_AVAILABLE_MEMORY_FLOOR_NOT_MET"):
            runtime.resources({"resource_shape": shape}, Path("/tmp/work"))

    def test_finite_cgroup_headroom_below_floor_is_rejected(self):
        for version in ("v1", "v2"):
            with self.subTest(version=version):
                with mock.patch.dict(self.files):
                    self.controller(version, 16 * GIB, 9 * GIB)
                    with self.assertRaisesRegex(ValueError, "MERLIN_CGROUP_MEMORY_HEADROOM_FLOOR_NOT_MET:" + version):
                        runtime.memory_evidence(self.shape, 8)

    def test_finite_cgroup_headroom_at_floor_is_accepted(self):
        self.controller("v2", 16 * GIB, 8 * GIB)
        evidence = runtime.memory_evidence(self.shape, 8)
        controller = evidence["cgroup"]["controllers"][0]
        self.assertEqual(controller["scope"], "visible_cgroup")
        self.assertEqual(controller["limit_minus_usage_bytes"], 8 * GIB)
        self.assertEqual(evidence["cgroup"]["status"], "memory_controller_observed")

    def test_partial_memory_controller_is_rejected(self):
        for version, limit, usage in (("v2", "max", None), ("v2", None, 0),
                                      ("v1", 16 * GIB, None), ("v1", None, 0)):
            with self.subTest(version=version, limit=limit, usage=usage):
                with mock.patch.dict(self.files):
                    self.controller(version, limit, usage)
                    with self.assertRaisesRegex(ValueError, "MERLIN_CGROUP_MEMORY_EVIDENCE_INCOMPLETE:" + version):
                        runtime.memory_evidence(self.shape, 8)

    def test_unlimited_v2_does_not_hide_finite_v1_limit(self):
        self.controller("v2", "max", 0)
        self.controller("v1", 16 * GIB, 9 * GIB)
        with self.assertRaisesRegex(ValueError, "MERLIN_CGROUP_MEMORY_HEADROOM_FLOOR_NOT_MET:v1"):
            runtime.memory_evidence(self.shape, 8)

    def test_unknown_cgroup_mount_layout_is_rejected(self):
        self.files["/proc/self/mountinfo"] += "35 31 0:30 / /other-cgroup rw - cgroup cgroup rw,memory\n"
        with self.assertRaisesRegex(ValueError, "MERLIN_CGROUP_MEMORY_LAYOUT_UNSUPPORTED"):
            runtime.memory_evidence(self.shape, 8)

    def test_missing_mount_evidence_cannot_establish_unexposed_cgroup(self):
        self.files["/proc/self/mountinfo"] = ""
        with self.assertRaisesRegex(ValueError, "MERLIN_CGROUP_MOUNT_EVIDENCE_REQUIRED"):
            runtime.memory_evidence(self.shape, 8)

    def test_original_memory_requests_cannot_be_reduced(self):
        with self.assertRaisesRegex(ValueError, "V1_READOUT_ORIGINAL_16_GIB_MEMORY_REQUEST_REQUIRED"):
            readout.cpu_preflight(Path("/tmp/work"), dict(self.shape, memory_mb=8192))
        with self.assertRaisesRegex(ValueError, "CA3_ORIGINAL_160_GIB_MEMORY_REQUEST_REQUIRED"):
            runtime.resources({"resource_shape": dict(self.shape, memory_mb=131072)}, Path("/tmp/work"))


if __name__ == "__main__":
    unittest.main()
