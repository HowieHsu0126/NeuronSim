"""CUDA construction of source-law Mesocircuit spatial projections.

Uses the fully derived source probabilities, repeated Bernoulli connections,
sign-truncated normal weights, and continuous linear delays. RNG realizations
are CUDA Philox realizations, not NEST-native snapshots. The exported raw delay
still requires the separately verified native clock conversion before import.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Iterator

import numpy as np

from .network_snapshot import load_initial_network
from .provenance import sha256_file

SOURCE_REVISION = "f59ada6b3c3efb1fa18f96c638dd1741ebfb77e5"
EDGE_DTYPE = np.dtype([("source", "<i8"), ("target", "<i8"), ("weight", "<f8"),
                       ("raw_delay", "<f8"), ("repeat", "<i4"), ("anchor_index", "<i4")])
SCHEMA = "mesocircuit_spatial_projection_graph_v2"


@dataclass(frozen=True)
class Projection:
    extent: float
    radius: float
    beta: float
    p0: float
    weight_mean: float
    weight_sd: float
    delay_offset: float
    speed: float
    repeats: int
    seed: int
    stream: int

    def validate(self) -> None:
        values = (self.extent,self.radius,self.beta,self.p0,self.weight_mean,
                  self.weight_sd,self.delay_offset,self.speed)
        if any(isinstance(x,(bool,np.bool_)) or not math.isfinite(x) for x in values):
            raise ValueError("finite numeric projection parameters required")
        if (self.extent <= 0 or not 0 < self.radius <= self.extent/2 or self.beta <= 0
                or not 0 <= self.p0 <= 1 or self.weight_sd < 0 or self.delay_offset <= 0
                or self.speed <= 0):
            raise ValueError("unsupported spatial projection domain")
        if self.weight_mean == 0 and self.weight_sd != 0:
            raise ValueError("zero source mean requires zero source weight deviation")
        if (type(self.repeats) is not int or not 0 <= self.repeats < 2**31
                or type(self.seed) is not int or not 0 < self.seed < 2**64
                or type(self.stream) is not int or not 0 <= self.stream < 72):
            raise ValueError("invalid repeats or Philox seed/subsequence identity")

    @classmethod
    def from_network(cls, net: dict, target: int, source: int, seed: int) -> Projection:
        if net["connect_method"] != "distr_indegree_exp" or net["delay_type"] != "linear":
            raise ValueError("CUDA source constructor supports exponential Bernoulli and linear delays")
        if net["num_pops"] != 9 or not 0 <= target < 8 or not 0 <= source < 9:
            raise ValueError("pinned source requires eight cortical targets and nine source populations")
        repeat = net["repeat_connect"][target][source]
        if isinstance(repeat,(bool,np.bool_)) or not math.isfinite(repeat) or repeat != int(repeat):
            raise ValueError("source repeat count must be an integer")
        mean = float(net["weight_matrix_mean"][target][source])
        result = cls(extent=float(net["extent"]),radius=float(net["mask_radius"][target][source]),
            beta=float(net["beta"][target][source]),p0=float(net["p0"][target][source]),
            weight_mean=mean,weight_sd=abs(mean*float(net["weight_rel_std"])),
            delay_offset=float(net["delay_offset_matrix"][target][source]),
            speed=float(net["prop_speed_matrix"][target][source]),
            repeats=int(net["repeat_connect"][target][source]),seed=seed,stream=target*9+source)
        result.validate()
        return result


def _population(ids, positions, extent):
    ids, positions = np.asarray(ids),np.asarray(positions)
    if (ids.ndim != 1 or not 0 < len(ids) < 2**31 or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids >= (2**64-144)//144)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("unique nonnegative unit IDs within Philox stream range required")
    if (positions.shape != (len(ids),2) or positions.dtype.kind not in "fiu"
            or not np.isfinite(positions).all() or np.any(np.ptp(positions,axis=0)>extent)):
        raise ValueError("finite source positions must fit within the declared layer extent")
    return np.ascontiguousarray(ids,dtype=np.int64),np.ascontiguousarray(positions,dtype=np.float64)


def cell_order(ids, positions, extent, radius):
    """Canonical O(N) storage; cell width is at least the circular mask radius."""
    ids,positions = _population(ids,positions,extent)
    limit = max(1,math.isqrt(len(ids)))
    cells = limit if extent/limit >= radius else max(1,int(math.floor(extent/radius)))
    # FreeLayer sets its tree origin from the actual position extrema. Ntree
    # canonicalizes even positions already inside the periodic square.
    origin = (positions.max(axis=0)+positions.min(axis=0))/2-extent/2
    canonical = origin+np.fmod(positions-origin,extent)
    canonical = np.where(canonical<origin,canonical+extent,canonical)
    xy = np.floor(((canonical-origin)/extent)*cells).astype(np.int64)
    np.clip(xy,0,cells-1,out=xy)
    cell_ids = xy[:,1]*cells+xy[:,0]
    order = np.lexsort((ids,cell_ids))
    offsets = np.concatenate(([0],np.cumsum(np.bincount(cell_ids,minlength=cells*cells))))
    return ids[order].copy(),canonical[order].copy(),offsets.astype(np.int32),cells,origin


class CudaProjection:
    """A single projection, with bounded output allocation and explicit lifetime."""
    def __init__(self, library: Path, source_ids, source_positions, target_ids, target_positions,
                 projection: Projection):
        projection.validate()
        sid,sx = _population(source_ids,source_positions,projection.extent)
        tid,tx = _population(target_ids,target_positions,projection.extent)
        _,left,right = np.intersect1d(sid,tid,return_indices=True)
        if not np.array_equal(sx[left],tx[right]):
            raise ValueError("one unit identity has conflicting spatial coordinates")
        sid,sx,offsets,cells,origin = cell_order(sid,sx,projection.extent,projection.radius)
        self.target_count,self.projection = len(tid),projection
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        pointer,integer,u64,size = ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64,ctypes.c_size_t
        self.library.spatial_create.argtypes = [pointer,pointer,integer,pointer,integer,pointer,pointer,
            integer,pointer,integer,u64,u64,pointer,pointer,size]
        self.library.spatial_counts.argtypes = [pointer,integer,integer,pointer,pointer,size]
        self.library.spatial_edges.argtypes = [pointer,integer,integer,pointer,pointer,pointer,size]
        self.library.spatial_destroy.argtypes = [pointer]
        self.library.spatial_destroy.restype = None
        self.handle = ctypes.c_void_p()
        values = np.asarray([projection.extent,projection.radius,projection.beta,projection.p0,
            projection.weight_mean,projection.weight_sd,projection.delay_offset,projection.speed,*origin],dtype=np.float64)
        self._call("spatial_create",self._pointer(sx),self._pointer(sid),len(sid),self._pointer(offsets),cells,
            self._pointer(tx),self._pointer(tid),len(tid),self._pointer(values),projection.repeats,
            projection.seed,projection.stream,ctypes.byref(self.handle))

    @staticmethod
    def _pointer(array):
        return array.ctypes.data_as(ctypes.c_void_p)

    def _call(self, name, *args):
        error = ctypes.create_string_buffer(4096)
        status = getattr(self.library,name)(*args,error,len(error))
        if status: raise RuntimeError(f"{name}: {error.value.decode('utf-8',errors='replace')}")

    def close(self):
        if self.handle.value:
            self.library.spatial_destroy(self.handle)
            self.handle.value = None

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, tb): self.close()

    def chunks(self, target_batch_size=256, maximum_edges=1_000_000) -> Iterator[np.ndarray]:
        if not self.handle.value: raise RuntimeError("projection is closed")
        if any(type(x) is not int or x < 1 for x in (target_batch_size,maximum_edges)):
            raise ValueError("positive integer target and edge batch capacities required")
        begin = 0
        while begin < self.target_count:
            count = min(target_batch_size,self.target_count-begin)
            counts = np.empty(count,dtype=np.uint64)
            self._call("spatial_counts",self.handle,begin,count,self._pointer(counts))
            # Prefix count bookkeeping is small; all pair sampling is on CUDA.
            total = sum(map(int,counts))
            while total > maximum_edges and count > 1:
                count = max(1,count//2)
                counts = counts[:count]
                total = sum(map(int,counts))
            if total > maximum_edges:
                raise MemoryError("one target exceeds the configured edge batch capacity")
            offsets = np.empty(count+1,dtype=np.uint64)
            offsets[0] = 0
            np.cumsum(counts,out=offsets[1:])
            edges = np.empty(total,dtype=EDGE_DTYPE)
            self._call("spatial_edges",self.handle,begin,count,self._pointer(offsets),self._pointer(edges))
            if not np.isfinite(edges["weight"]).all() or not np.isfinite(edges["raw_delay"]).all():
                raise ValueError("CUDA produced nonfinite connection attributes")
            yield edges
            begin += count


def native_delay_ticks(delays, resolution: float, tics_per_ms: float = 1000.) -> np.ndarray:
    """Pinned NEST 3.10 delay conversion, verified against native readback.

    ld_round performs a separately rounded binary64 addition of 0.5 followed by
    floor. It differs from ideal midpoint rounding immediately below a half-step.
    Delays are not first rounded to simulator tics.
    """
    if (not math.isfinite(resolution) or resolution<=0 or not math.isfinite(tics_per_ms)
            or tics_per_ms<=0):
        raise ValueError("positive finite native clock resolution required")
    tics_per_step=math.floor(resolution*tics_per_ms+.5)
    if tics_per_step<1 or tics_per_step/tics_per_ms != resolution:
        raise ValueError("resolution must equal the actual native clock step")
    values=np.asarray(delays,dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError("native delays must be finite and positive")
    scaled=values*(1./resolution)
    rounded=np.floor(scaled+.5)
    if not np.isfinite(rounded).all() or np.any(rounded<1) or np.any(rounded>=2**63):
        raise ValueError("native delay is outside positive int64 steps")
    return rounded.astype(np.int64)


def create_layout(library: Path, network: dict, config: dict) -> dict[str,np.ndarray]:
    """Uniform source geometry and untruncated voltage laws, sampled on CUDA.

    Cortical initial voltages are retained separately; TC parrot units do not
    acquire an invented membrane-potential state.
    """
    counts = np.asarray(network["num_neurons"])
    names = network["populations"]
    if (counts.shape != (9,) or counts.dtype.kind not in "iu" or np.any(counts < 1)
            or int(counts.sum()) >= 2**31 or len(names) != 9 or len(set(names)) != 9
            or names[-1] != "TC"):
        raise ValueError("nine valid source populations required")
    streams = config["random_streams"]
    seeds = [streams[key] for key in ("spatial_layout","initial_state")]
    if any(type(seed) is not int or not 0 < seed < 2**32 for seed in seeds) or seeds[0]==seeds[1]:
        raise ValueError("independent source layout/initial-state seeds required")
    parameters = network["neuron_params"]
    mode = network["V0_type"]
    if mode == "original":
        mean = [parameters["V0_mean"][mode]]*8
        deviation = [parameters["V0_std"][mode]]*8
    elif mode == "optimized":
        mean,deviation = parameters["V0_mean"][mode],parameters["V0_std"][mode]
    else: raise ValueError("unsupported source initial-voltage law")
    mean,deviation = np.asarray(mean,dtype=float),np.asarray(deviation,dtype=float)
    if (mean.shape != (8,) or deviation.shape != (8,) or not np.isfinite(mean).all()
            or not np.isfinite(deviation).all() or np.any(deviation<0)):
        raise ValueError("valid cortical normal initial-voltage parameters required")
    extent = float(network["extent"])
    if not math.isfinite(extent) or extent<=0: raise ValueError("positive finite extent required")
    ids = np.arange(1,int(counts.sum())+1,dtype=np.int64)
    means = np.repeat(np.r_[mean,0.],counts)
    deviations = np.repeat(np.r_[deviation,0.],counts)
    xy,vm = np.empty((len(ids),2)),np.empty(len(ids))
    backend = ctypes.CDLL(str(Path(library).resolve()))
    pointer = ctypes.c_void_p
    backend.spatial_layout.argtypes = [pointer,pointer,pointer,ctypes.c_int,ctypes.c_double,
        ctypes.c_uint64,ctypes.c_uint64,pointer,pointer,pointer,ctypes.c_size_t]
    error = ctypes.create_string_buffer(4096)
    status = backend.spatial_layout(CudaProjection._pointer(ids),CudaProjection._pointer(means),
        CudaProjection._pointer(deviations),len(ids),extent,*seeds,CudaProjection._pointer(xy),
        CudaProjection._pointer(vm),error,len(error))
    if status: raise RuntimeError("spatial_layout: "+error.value.decode())
    _population(ids,xy,extent)
    if np.any(xy < -extent/2) or np.any(xy >= extent/2): raise ValueError("sampled layout escaped source square")
    if not np.isfinite(vm).all(): raise ValueError("nonfinite sampled initial voltages")
    labels = np.repeat(np.asarray(names),counts)
    return {"unit_ids":ids,"populations":labels,"positions_mm":xy,
            "cortical_initial_vm_mv":vm[labels!="TC"]}


def build(snapshot: Path | None, library: Path, output: Path, seed: int,
          target_batch_size=256, maximum_edges=1_000_000, parameters: Path | None = None) -> dict:
    """Generate new recurrent/TC graph realizations on a verified source layout.

    External device edges are not generated here. Each shard retains source IDs,
    its Bernoulli-repeat index and unquantized physical delays. Original NEST
    topology is neither overwritten nor relabeled as CUDA-generated data.
    """
    output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    report = {"schema":SCHEMA,"status":"FAILED","scientific_reference_qualified":False,
        "snapshot":str(snapshot.resolve()) if snapshot else None,
        "parameters":str(parameters.resolve()) if parameters else None,"seed":seed,"rng":"cuRAND Philox4_32_10",
        "rng_scope":"independent target/projection topology and weight subsequences",
        "slurm_job_id":os.environ.get("SLURM_JOB_ID"),"source_revision":SOURCE_REVISION,
        "delay_representation":"unquantized source-law linear delays in ms",
        "external_device_edges_included":False,"projections":[],"parts":[],
        "periodic_geometry":"source Ntree canonical coordinates and masked-iterator anchors",
        "anchor_index":"bit0 subtracts extent in x; bit1 subtracts extent in y from the primary wrapped anchor"}
    try:
        report["library_sha256"] = sha256_file(library)
        if (snapshot is None) == (parameters is None):
            raise ValueError("choose exactly one native snapshot or resolved parameter input")
        if snapshot is not None:
            report["snapshot_manifest_sha256"] = sha256_file(snapshot/"manifest.json")
            metadata,arrays = load_initial_network(snapshot)
            if metadata["source_revision"] != SOURCE_REVISION: raise ValueError("unreviewed source revision")
            net = metadata["resolved_source_network"]
            report["layout_origin"] = "verified native source snapshot"
        else:
            payload = json.loads(parameters.read_text())
            if (payload.get("schema") != "mesocircuit_resolved_parameters_v1"
                    or payload.get("source_revision") != SOURCE_REVISION
                    or payload.get("scientific_reference_qualified") is not False):
                raise ValueError("unreviewed source parameter export")
            report["parameters_sha256"] = sha256_file(parameters)
            report["source_manifest_sha256"] = payload["source_manifest_sha256"]
            net = payload["network"]
            metadata = {"config":payload["config"]}
            arrays = create_layout(library,net,payload["config"])
            report["layout_origin"] = "CUDA sampled source uniform geometry and normal cortical voltage"
            report["resolved_simulation"] = payload["simulation"]
        names = net["populations"]
        ids,xy,labels = arrays["unit_ids"],arrays["positions_mm"],arrays["populations"]
        if set(labels) != set(names): raise ValueError("source population identity mismatch")
        if not np.array_equal([np.sum(labels==name) for name in names],net["num_neurons"]):
            raise ValueError("source population counts mismatch")
        report.update(cortical_neurons=int(np.sum(labels!="TC")),thalamic_neurons=int(np.sum(labels=="TC")),
            connectivity_normalization=metadata["config"].get("connectivity_normalization","native_source"),
            resolved_source_network=net,source_config=metadata["config"],
            target_batch_size=target_batch_size,maximum_edges=maximum_edges)
        np.savez(output/"layout.npz",**{k:arrays[k] for k in (
            "unit_ids","populations","positions_mm","cortical_initial_vm_mv") if k in arrays})
        report["layout_sha256"] = sha256_file(output/"layout.npz")
        report["implementation_sha256"] = sha256_file(Path(__file__))
        report["cuda_source_sha256"] = sha256_file(Path(__file__).parent/"cuda"/"spatial_connectivity.cu")
        report["resumed"] = False
        report["status"] = "GENERATING"
        (output/"manifest.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        total = 0
        for t,target in enumerate(names[:-1]):
            for s,source in enumerate(names):
                p = Projection.from_network(net,t,s,seed)
                sm,tm = labels==source,labels==target
                projection_start,projection_count = time.perf_counter(),0
                with CudaProjection(library,ids[sm],xy[sm],ids[tm],xy[tm],p) as generator:
                    for index,edges in enumerate(generator.chunks(target_batch_size,maximum_edges)):
                        name = f"projection_{t}_{s}_part_{index}.npy"
                        np.save(output/name,edges,allow_pickle=False)
                        report["parts"].append({"path":name,"source_population":source,"target_population":target,
                            "edges":len(edges),"sha256":sha256_file(output/name)})
                        projection_count += len(edges)
                total += projection_count
                report["projections"].append({"source_population":source,"target_population":target,
                    "parameters":asdict(p),"edges":projection_count,"seconds":time.perf_counter()-projection_start})
                report["edges"] = total
                (output/"manifest.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
                print(json.dumps({"projection":[t,s],"completed_projections":len(report["projections"]),
                                  "edges":total}),flush=True)
        report.update(status="GENERATED",edges=total)
    except Exception as exc:
        report.update(status="FAILED",error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    report["seconds"] = time.perf_counter()-started
    report["implementation_sha256"] = sha256_file(Path(__file__))
    report["cuda_source_sha256"] = sha256_file(Path(__file__).parent/"cuda"/"spatial_connectivity.cu")
    (output/"manifest.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--snapshot",type=Path)
    inputs.add_argument("--parameters",type=Path)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--target-batch-size",type=int,default=256)
    parser.add_argument("--maximum-edges",type=int,default=1_000_000)
    args = parser.parse_args()
    report = build(args.snapshot,args.library,args.output,args.seed,args.target_batch_size,args.maximum_edges,args.parameters)
    print(json.dumps({k:report[k] for k in ("status","seconds","scientific_reference_qualified")}))
    if report["status"] != "GENERATED":
        print(report.get("traceback",report.get("error")))
        raise SystemExit(1)


if __name__ == "__main__": main()
