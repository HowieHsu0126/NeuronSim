"""Exercise launcher failure preservation before a native report exists."""
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest


def run_fixture(tmp_path, launcher):
    """Use the actual launcher's trap and its final success cleanup."""
    text = launcher.read_text()
    first = text.index("preserve_failure() {")
    last = text.index('mkdir -p "$scratch/Libs/population_dynamics"')
    program = tmp_path / "launcher_fixture.sh"
    program.write_text(
        '#!/bin/bash\nset -euo pipefail\nscratch=$1\ndestination=$2\nmode=$3\n'
        + text[first:last]
        + 'mkdir -p "$scratch/run"\n'
        + 'printf "partial native output\\n" > "$scratch/run/native.log"\n'
        + 'printf "{}\\n" > "$scratch/run/resource_latest.json"\n'
        + 'touch "$scratch/ready"\n'
        + 'if [[ "$mode" == fail ]]; then exit 9; fi\n'
        + 'if [[ "$mode" == cancel ]]; then while :; do sleep 0.05; done; fi\n'
        + text[text.rindex('rm -rf "$scratch"\ntrap - EXIT'):]
    )
    return program


@pytest.mark.parametrize("mode,exit_code", [("fail", 9), ("cancel", 143), ("success", 0)])
def test_pre_report_failure_is_preserved_and_success_cleans_up(tmp_path, mode, exit_code):
    launcher = Path(os.environ.get("NRNSIM_CA3_EPISODE_LAUNCHER",
                                  Path(__file__).resolve().parents[1] / "scripts/run_ca3_benchmark_episode.sbatch"))
    program = run_fixture(tmp_path, launcher)
    scratch, destination = tmp_path / "scratch", tmp_path / "destination"
    process = subprocess.Popen(["bash", str(program), str(scratch), str(destination), mode],
                               start_new_session=True)
    try:
        if mode == "cancel":
            deadline = time.monotonic() + 5
            while not (scratch / "ready").exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail("Launcher fixture did not reach its pre-report execution state")
                time.sleep(0.01)
            os.killpg(process.pid, signal.SIGTERM)
        assert process.wait(timeout=5) == exit_code
        if mode == "success":
            assert not scratch.exists()
            assert not destination.exists()
        else:
            assert scratch.is_dir()
            assert not (destination / "terminal_report.json").exists()
            failure = destination / "failure"
            assert (failure / "native.log").read_text() == "partial native output\n"
            assert (failure / "resource_latest.json").read_text() == "{}\n"
            assert (failure / "retained_scratch.txt").read_text().strip() == str(scratch)
            assert (failure / "exit_code.txt").read_text().strip() == str(exit_code)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
