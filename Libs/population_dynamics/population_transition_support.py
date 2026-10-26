"""Describe saved coordinates relative to the bounded transition drive.

Only the tanh drive is bounded. Initial and stochastic states are unbounded;
coordinate range alone does not establish an unreachable state.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def analyze(source,output):
    if not os.environ.get("SLURM_JOB_ID"):raise ValueError("Slurm allocation required")
    proof=json.loads((source/"analysis.json").read_text())
    if proof["status"]!="COMPLETE_PAIRED_CONDITIONAL_KL_DIAGNOSTIC" or proof["training_seeds"]!=[41,42,43]:
        raise ValueError("accepted paired coordinate diagnostic required")
    rows=[];consumed={}
    with tempfile.TemporaryDirectory(prefix="neuronsim-transition-support-",dir="/tmp") as temp:
        for seed in (41,42,43):
            for label in ("prior80","continued"):
                name=f"seed_{seed}_{label}_coordinates.npz";local=Path(temp)/name
                shutil.copyfile(source/name,local)
                actual=digest(local)
                if actual!=proof["artifacts"][name]:raise ValueError("saved coordinates changed")
                consumed[name]=actual
                with np.load(local,allow_pickle=False) as data:
                    target,prediction,bias=(data[k] for k in ("target_mean","conditional_mean","mean_squared_bias"))
                    if target.shape!=prediction.shape or target.shape!=bias.shape or not all(np.isfinite(x).all() for x in (target,prediction,bias)):
                        raise ValueError("finite aligned coordinates required")
                    for split in ("fit","selection"):
                        selected=data["splits"]==split
                        if int(selected.sum())!={"fit":56,"selection":16}[split]:raise ValueError("complete source split required")
                        for i,horizon in enumerate(data["horizons_ms"]):
                            t,p,b=(value[selected,:,i] for value in (target,prediction,bias))
                            outside=np.abs(t)>1
                            rows.append(dict(seed=seed,checkpoint=label,split=split,horizon_ms=float(horizon),
                                coordinate_count=t.size,target_min=float(t.min()),target_max=float(t.max()),
                                target_outside_tanh_drive_fraction=float(outside.mean()),
                                predicted_mean_outside_tanh_drive_fraction=float((np.abs(p)>1).mean()),
                                mean_squared_bias=float(b.mean()),
                                squared_bias_fraction_at_outside_targets=float(b[outside].sum()/b.sum()) if b.sum()>0 else None))
                local.unlink()
    result=dict(status="COMPLETE_SAVED_TRANSITION_COORDINATE_DESCRIPTION",training_seeds=[41,42,43],
        source=str(source),source_analysis_sha256=digest(source/"analysis.json"),consumed=consumed,rows=rows,
        interpretation="Only the tanh drive is bounded to [-1,1]; initial and stochastic states are unbounded. Coordinate range alone does not establish an unreachable state or a causal explanation of forecast error.",
        training_performed=False,encoding_performed=False,simulation_performed=False,scientific_claim_qualified=False,
        job_id=os.environ["SLURM_JOB_ID"],script_sha256=digest(Path(__file__)))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps([r for r in rows if r["split"]=="selection" and r["horizon_ms"]==500],indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();analyze(args.source,args.output)
