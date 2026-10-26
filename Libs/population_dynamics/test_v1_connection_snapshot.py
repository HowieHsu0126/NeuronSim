import hashlib
import h5py
import numpy as np
import pytest
from .v1_connection_snapshot import ConnectionSnapshot, update_digest, verify_snapshot


def batch():
    return dict(source=np.array([2,3]),target=np.array([1,1]),receptor_type=np.array([1,4]),
                weight=np.array([0.,-2.5]),requested_delay=np.array([.625,.75]),delay=np.array([.75,.75]))


def make_file(tmp_path):
    values=batch();digest=hashlib.sha256();update_digest(digest,"v1",values)
    path=tmp_path/"connections.h5"
    writer=ConnectionSnapshot(path,{"v1":{"edge_rows":2}},digest.hexdigest())
    writer.append("v1",values);writer.verify();writer.close()
    return path,digest.hexdigest()


def test_stored_source_digest_retains_zero_and_signed_weights(tmp_path):
    path,digest=make_file(tmp_path)
    assert verify_snapshot(path,digest,2)["rows"]==2
    with h5py.File(path,"r") as f:assert f["weight"][:].tolist()==[0.,-2.5]


def test_changed_weight_is_rejected_by_native_digest(tmp_path):
    path,digest=make_file(tmp_path)
    with h5py.File(path,"r+") as f:f["weight"][0]=1.
    with pytest.raises(ValueError,match="STORED_NATIVE_DIGEST_DIFFERS"):
        verify_snapshot(path,digest,2)


def test_missing_batch_range_is_rejected(tmp_path):
    path,digest=make_file(tmp_path)
    with h5py.File(path,"r+") as f:f["batch_start"][0]=1
    with pytest.raises(ValueError,match="STORED_BATCH_RANGE"):
        verify_snapshot(path,digest,2)


def test_float32_weight_is_not_silently_promoted(tmp_path):
    values=batch();values["weight"]=values["weight"].astype(np.float32)
    writer=ConnectionSnapshot(tmp_path/"partial.h5",{"v1":{"edge_rows":2}},"unused")
    try:
        with pytest.raises(ValueError,match="BATCH_DTYPE"):
            writer.append("v1",values)
    finally:writer.close()
