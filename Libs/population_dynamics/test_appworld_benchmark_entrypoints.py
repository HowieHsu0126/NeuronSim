"""Check staged CLI and public CUGA imports on Merlin CPU.

These checks import real application source. They do not run model inference,
native tasks, or the full CUGA filesystem/process sandbox.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from population_dynamics import test_appworld_benchmark_merlin as fixture


_PUBLIC_IMPORT_CHECK = r"""
import importlib
import json
from pathlib import Path
import sys

public_lib = Path(sys.argv[1]).resolve(strict=True)
worker_modules = (
    "population_dynamics.appworld_benchmark_cuga_process",
    "population_dynamics.appworld_benchmark_cuga_channel",
    "population_dynamics.appworld_benchmark_cuga_runtime",
    "population_dynamics.appworld_benchmark_cuga_tools",
)
for name in worker_modules:
    importlib.import_module(name)

loaded = []
for name, module in tuple(sys.modules.items()):
    if name != "population_dynamics" and not name.startswith("population_dynamics."):
        continue
    origins = []
    filename = getattr(module, "__file__", None)
    if filename is not None:
        origins.append(Path(filename).resolve(strict=True))
    origins.extend(Path(path).resolve(strict=True)
                   for path in getattr(module, "__path__", ()))
    if not origins:
        raise AssertionError("Application module has no filesystem origin: " + name)
    if any(not origin.is_relative_to(public_lib) for origin in origins):
        raise AssertionError("Application import escaped the public library: " + name)
    loaded.append(name)
print(json.dumps({"requested": list(worker_modules), "loaded": sorted(loaded)}))
"""


class AppWorldEntrypointTests(fixture.TemporaryArtifacts):
    def _run_python(self, arguments: list[str], *, application_lib: Path):
        # Replace inherited PYTHONPATH. A fresh working directory prevents the
        # repository checkout from satisfying missing staged imports.
        environment = dict(
            os.environ,
            PYTHONPATH=str(application_lib),
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        result = subprocess.run(
            [sys.executable, *arguments], cwd=self.work, env=environment,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Python entry point failed: {arguments!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result

    def _assert_cli_help(self, module: str):
        result = self._run_python(
            ["-m", module, "--help"], application_lib=fixture.SCRIPT.parents[1],
        )
        self.assertIn("usage:", result.stdout.lower())
        self.assertIn("--help", result.stdout)

    def test_native_binding_cli_imports(self):
        self._assert_cli_help("population_dynamics.appworld_benchmark_native_binding")

    def test_development_pipeline_cli_imports(self):
        self._assert_cli_help("population_dynamics.appworld_benchmark_pipeline")

    def test_cuga_development_cli_imports(self):
        self._assert_cli_help("population_dynamics.appworld_benchmark_cuga_development")

    def test_public_cuga_worker_imports_use_only_copied_application_source(self):
        source_lib = fixture.SCRIPT.parents[1]
        public_lib = self.work / "public" / "Libs"
        for name in fixture.bootstrap.CUGA_PUBLIC_FILES:
            relative = Path(name).relative_to("Libs")
            source = source_lib / relative
            self.assertTrue(source.is_file(), f"Missing admitted public source: {name}")
            self.assertFalse(source.is_symlink(), f"Public source is a symlink: {name}")
            destination = public_lib / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        result = self._run_python(
            ["-c", _PUBLIC_IMPORT_CHECK, str(public_lib)], application_lib=public_lib,
        )
        report = json.loads(result.stdout)
        self.assertTrue(set(report["requested"]).issubset(report["loaded"]))
        self.assertIn("population_dynamics", report["loaded"])
