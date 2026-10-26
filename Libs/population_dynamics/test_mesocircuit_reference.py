"""Protection of source output ownership and one-rank spatial semantics."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from .mesocircuit_reference import _fresh_native_directory, _SingleRankSpatialBinding


class Native:
    def __init__(self, ranks=1):
        self.ranks = ranks
        self.resolution = 0.1
    def NumProcesses(self):
        return self.ranks
    def GetLocalNodeCollection(self, nodes):
        return list(nodes)  # deliberately drops the collection's metadata


def test_spatial_binding_retains_identity_and_forwards_kernel_settings():
    native = Native()
    binding = _SingleRankSpatialBinding(native, SimpleNamespace(COMM_WORLD=SimpleNamespace(size=1)))
    collection = (1, 2, 3)
    assert binding.GetLocalNodeCollection(collection) is collection
    binding.resolution = 0.2
    assert native.resolution == 0.2
    assert native.GetLocalNodeCollection(collection) is not collection


@pytest.mark.parametrize("nest_ranks,mpi_ranks", [(2, 1), (1, 2), (2, 2)])
def test_spatial_binding_never_fakes_a_distributed_local_collection(nest_ranks, mpi_ranks):
    with pytest.raises(ValueError, match="ONE_RANK"):
        _SingleRankSpatialBinding(Native(nest_ranks), SimpleNamespace(COMM_WORLD=SimpleNamespace(size=mpi_ranks)))


def test_existing_output_is_rejected_without_exposing_it_to_native_wipe(tmp_path):
    output = tmp_path / "existing"
    (output / "raw_data").mkdir(parents=True)
    protected = output / "raw_data/important.txt"
    protected.write_text("keep")
    with pytest.raises(FileExistsError):
        with _fresh_native_directory(output):
            pytest.fail("existing output became native working directory")
    assert protected.read_text() == "keep"


def test_native_failure_restores_caller_working_directory(tmp_path):
    original = Path.cwd()
    with pytest.raises(RuntimeError):
        with _fresh_native_directory(tmp_path / "fresh"):
            assert Path.cwd() == tmp_path / "fresh"
            assert Path("raw_data").is_dir()
            raise RuntimeError("native failure")
    assert Path.cwd() == original
