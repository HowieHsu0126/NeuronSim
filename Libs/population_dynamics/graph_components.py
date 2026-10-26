"""Bounded-memory CUDA weak-component validation for a streamed source graph."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


class CudaWeakComponents:
    """Sequentially consume edge batches; return each component's smallest index."""
    def __init__(self, library: Path, vertices: int):
        if type(vertices) is not int or not 0 < vertices < 2**31-256:
            raise ValueError("positive int32 vertex count required")
        self.vertices=vertices
        self.library=ctypes.CDLL(str(Path(library).resolve()))
        p,i,s=ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t
        self.library.components_create.argtypes=[i,p,p,s]
        self.library.components_add.argtypes=[p,p,p,i,p,s]
        self.library.components_labels.argtypes=[p,p,p,s]
        self.library.components_destroy.argtypes=[p]
        self.library.components_destroy.restype=None
        self.handle=ctypes.c_void_p()
        self._call("components_create",vertices,ctypes.byref(self.handle))

    def _call(self,name,*args):
        error=ctypes.create_string_buffer(4096)
        if getattr(self.library,name)(*args,error,len(error)):
            raise RuntimeError(name+": "+error.value.decode())

    def add(self,source,target):
        if not self.handle.value: raise RuntimeError("component accumulator is closed")
        source,target=np.asarray(source),np.asarray(target)
        if source.ndim!=1 or target.shape!=source.shape or len(source)>=2**31-256:
            raise ValueError("equal bounded one-dimensional edge arrays required")
        for values in (source,target):
            if values.dtype.kind not in "iu" or np.any(values<0) or np.any(values>=self.vertices):
                raise ValueError("edge endpoint outside declared vertices")
        source=np.ascontiguousarray(source,dtype=np.int32)
        target=np.ascontiguousarray(target,dtype=np.int32)
        self._call("components_add",self.handle,source.ctypes.data_as(ctypes.c_void_p),
                   target.ctypes.data_as(ctypes.c_void_p),len(source))

    def labels(self):
        if not self.handle.value: raise RuntimeError("component accumulator is closed")
        result=np.empty(self.vertices,dtype=np.int32)
        self._call("components_labels",self.handle,result.ctypes.data_as(ctypes.c_void_p))
        return result

    def close(self):
        if self.handle.value:
            self.library.components_destroy(self.handle);self.handle.value=None

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
