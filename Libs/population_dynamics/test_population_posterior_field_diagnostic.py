"""Native archive preparation retains delivered byte and checksum bindings."""
import hashlib

import pytest

from .population_posterior_field_diagnostic import _native_archive_files, _stage


def receipt_row(payload):
    return {"path": "train/case/report.json", "source_bytes": len(payload),
            "destination_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_native_archive_preparation_stages_original_receipt_without_mutation(tmp_path):
    payload = b'{"case_id":"case"}\n'
    row = receipt_row(payload)
    receipt = {"files": [row]}
    source = tmp_path / "archive"
    path = source / row["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    files = _native_archive_files(receipt)
    consumed = {}
    staged = _stage(source, files, row["path"], tmp_path / "scratch" / row["path"], consumed)
    assert staged.read_bytes() == payload
    assert consumed == {row["path"]: row["sha256"]}
    assert files[row["path"]]["bytes"] == len(payload)
    assert files[row["path"]] is not row
    assert receipt == {"files": [receipt_row(payload)]}


@pytest.mark.parametrize("payload", [b"short", b"changed!"])
def test_native_archive_staging_rejects_size_and_checksum_corruption(tmp_path, payload):
    row = receipt_row(b"original")
    path = tmp_path / row["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    files = _native_archive_files({"files": [row]})
    consumed = {}
    with pytest.raises(ValueError, match="delivery input changed"):
        _stage(tmp_path, files, row["path"], tmp_path / "scratch" / "report.json", consumed)
    assert not consumed


@pytest.mark.parametrize("source_size,archive_size", [(8, 7), (-1, -1), (8, 8.0), (True, 1)])
def test_native_archive_preparation_rejects_invalid_copy_sizes(source_size, archive_size):
    row = receipt_row(b"original")
    row.update(source_bytes=source_size, destination_bytes=archive_size)
    with pytest.raises(ValueError, match="sizes"):
        _native_archive_files({"files": [row]})


def test_native_archive_preparation_rejects_duplicate_paths():
    row = receipt_row(b"original")
    with pytest.raises(ValueError, match="duplicate native"):
        _native_archive_files({"files": [row, dict(row)]})
