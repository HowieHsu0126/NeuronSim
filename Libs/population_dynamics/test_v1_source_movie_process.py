"""Exercise source-process isolation and the actual export launcher on fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from .provenance import sha256_file
from .v1_benchmark import export_source_controls, inspect_source_movie, local_delivery


def movie_archive(root, amplitude):
    prepared = root / "prepared"
    (prepared / "lgnmodel").mkdir(parents=True)
    source = f"AMPLITUDE = {amplitude}\n" + '''
import numpy as np
from types import SimpleNamespace

def movie(data, t_max):
    return SimpleNamespace(data=data, t_range=np.arange(len(data))/1000.,
        row_range=np.arange(2), col_range=np.arange(2), frame_rate=1000.,
        labels=['time', 'x', 'y'], units=['second', 'degree', 'degree'])

class GratingMovie:
    def __init__(self, rows, cols):
        pass
    def create_movie(self, t_min=0, t_max=.004, gray_screen_dur=.001,
                     cpd=.04, temporal_f=1., theta=0., contrast=1.):
        ticks = np.arange(round(t_max*1000)+1)-round(gray_screen_dur*1000)
        data = np.zeros((len(ticks), 2, 2))
        data[ticks > 0] = (AMPLITUDE*contrast*(1+temporal_f*ticks[ticks > 0]))[:, None, None]
        return movie(data, t_max)

class FullFieldFlashMovie:
    def __init__(self, row_range, col_range, start, stop, frame_rate):
        pass
    def full(self, t_max):
        return movie(np.zeros((round(t_max*1000)+1, 2, 2)), t_max)
'''
    (prepared / "lgn_functions.py").write_text(source)
    for name in ("lgnmodel/__init__.py", "lgnmodel/poissongeneration.py"):
        (prepared / name).write_text("")
    names = ("lgn_functions.py", "lgnmodel/__init__.py", "lgnmodel/poissongeneration.py")
    (prepared / "prepared.json").write_text(json.dumps(dict(status="SOURCE_RUNTIME_PREPARED",
        files=[dict(path=name, prepared_sha256=sha256_file(prepared / name)) for name in names])))
    probe = root / "probe"
    probe.mkdir()
    config = dict(stimulus=dict(duration_s=.004, gray_screen_s=.001,
        spatial_frequency_cycles_per_degree=.04, temporal_frequency_hz=1.,
        direction_deg=0., contrast_percent=100.))
    (probe / "probe.json").write_text(json.dumps(dict(
        status="SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT", config=config)))
    with h5py.File(probe / "firing_rates_f_tot.h5", "w") as handle:
        handle["time"] = np.arange(5)/1000.
    (root / "delivery.json").write_text(json.dumps(dict(exit_code=0,
        files=[dict(path="probe/"+name, bytes=(probe/name).stat().st_size,
                    sha256=sha256_file(probe/name))
               for name in ("probe.json", "firing_rates_f_tot.h5")])))
    return root


def test_each_movie_uses_its_own_authenticated_runtime(tmp_path, monkeypatch):
    foreign = SimpleNamespace(__file__=str(tmp_path / "foreign/lgn_functions.py"))
    monkeypatch.setitem(sys.modules, "lgn_functions", foreign)
    for amplitude in (1, 7):
        archive = movie_archive(tmp_path / f"source_{amplitude}", amplitude)
        output = tmp_path / f"movie_{amplitude}"
        report = inspect_source_movie(archive, output)
        assert report["status"] == "SOURCE_MOVIE_CONTROL_WITNESS_PASS"
        assert report["source_rate_archive"] == str(archive.resolve())
        assert report["prepared_runtime_sha256"] == sha256_file(archive / "prepared/prepared.json")
        assert report["lgn_rates_recomputed"] is False
        with np.load(output / "movie_witness.npz") as witness:
            assert np.all(witness["selected_frames"][-1] == 4*amplitude)
        assert sys.modules["lgn_functions"] is foreign


def test_movie_child_failure_remains_a_failure(tmp_path):
    archive = movie_archive(tmp_path / "source", 1)
    with (archive / "prepared/lgn_functions.py").open("a") as stream:
        stream.write("\n# changed after qualification\n")
    output = tmp_path / "movie"
    with pytest.raises(subprocess.CalledProcessError):
        inspect_source_movie(archive, output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "FAILED_SOURCE_MOVIE_WITNESS"
    assert report["error"] == "SOURCE_PREPARED_MODULE_CHANGED:lgn_functions.py"


def test_control_child_failure_is_propagated_without_parent_imports(tmp_path, monkeypatch):
    foreign = SimpleNamespace(__file__=str(tmp_path / "foreign/lgn_functions.py"))
    monkeypatch.setitem(sys.modules, "lgn_functions", foreign)
    archive = movie_archive(tmp_path / "source", 1)
    movie = tmp_path / "movie_archive"
    inspect_source_movie(archive, movie / "movie")
    local_delivery(movie, "movie")
    output = tmp_path / "controls"
    with pytest.raises(subprocess.CalledProcessError):
        export_source_controls(archive, movie, tmp_path / "missing_spikes", output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "FAILED_SOURCE_VISUAL_COMMAND_ARCHIVE"
    assert report["rate_archive"] == str(archive.resolve())
    assert report["lgn_rates_recomputed"] is False
    assert report["spikes_recomputed"] is False
    assert sys.modules["lgn_functions"] is foreign


@pytest.mark.parametrize("mode, expected", [(None, None), ("resume", "--resume"),
                                           ("prepare_glif", "--prepare-glif-inputs")])
def test_export_launcher_preserves_default_and_resume_modes(tmp_path, mode, expected):
    package = tmp_path / "repo/Libs/population_dynamics"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "benchmark_production_block.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(sys.argv[1:]))\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    srun = binary / "srun"
    srun.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
    srun.chmod(0o755)
    output = tmp_path / "arguments.json"
    launcher = Path(__file__).resolve().parents[1] / "scripts/run_v1_production_block_export.sbatch"
    environment = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ["PATH"],
        SLURMD_NODENAME="cpu03", SLURM_JOB_ID="fixture", NRNSIM_BENCHMARK_PYTHON=sys.executable)
    environment.pop("SLURM_JOB_GPUS", None)
    command = ["bash", str(launcher), str(tmp_path / "repo"), "config", "qualification",
               "identities", "0", "0", "sources", str(output)]
    subprocess.run(command+([mode] if mode else []), env=environment, check=True)
    arguments = json.loads(output.read_text())
    flags = [value for value in arguments if value in ("--resume", "--prepare-glif-inputs")]
    assert flags == ([expected] if expected else [])
