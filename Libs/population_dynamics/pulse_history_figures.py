"""Render the predeclared pulse-history contrasts and block-held-out errors."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from .provenance import sha256_file


def render(analysis: Path, style_helper: Path, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec=importlib.util.spec_from_file_location("figure_style",style_helper)
    style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
    style.apply_figure_style(sizes=(9,8,7))
    output.mkdir(parents=True,exist_ok=True)
    data=json.loads((analysis/"analysis.json").read_text())
    for name,digest in data["artifacts"].items():
        if sha256_file(analysis/name)!=digest:raise ValueError("analysis artifact changed")
    with (analysis/"paired_responses.csv").open() as f:rows=list(csv.DictReader(f))
    qc={}
    def save(fig,name,caption):
        for ax in fig.axes:
            limits=ax.get_ylim();lo,hi=sorted(limits)
            ax.set_yticks([tick for tick in ax.get_yticks() if lo <= tick <= hi]);ax.set_ylim(limits)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        text=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
              if t.get_visible() and t.get_text().strip()]
        overlap=[(a.get_text(),b.get_text()) for i,(a,aa) in enumerate(text) for b,bb in text[i+1:] if aa.overlaps(bb)]
        outside=[t.get_text() for t,b in text if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1)]
        qc[name]=dict(text_overlaps=overlap,text_outside_canvas=outside)
        for suffix in ("png","pdf","svg"):fig.savefig(output/(name+"."+suffix))
        (output/(name+".txt")).write_text(caption+"\n")
        plt.close(fig)
    times=sorted({float(r["focal_arrival_ms"]) for r in rows})
    fig,ax=plt.subplots(figsize=(6.6,3.5),layout="constrained")
    for i,(timepoint,history) in enumerate((t,h) for t in times for h in (0,1)):
        values=np.asarray([float(r["delta_cortical_spikes_per_neuron"]) for r in rows
                           if float(r["focal_arrival_ms"])==timepoint and int(r["history"])==history])
        ax.scatter(i+np.linspace(-.1,.1,len(values)),values,s=25,color="#666666",alpha=.75)
        ax.plot(i,values.mean(),"_",color="black",markersize=19,markeredgewidth=2)
    ax.axhline(0,color="#aaaaaa",linewidth=.7)
    ax.set_xticks(range(4),[f"{t:g} ms\n"+("3 prior pulses" if h else "No prior pulses") for t in times for h in (0,1)])
    ax.set_ylabel("Focal effect (spikes/neuron, 0–20 ms)")
    ax.set_title("Pulse response conditional on preceding stimulation")
    ax.margins(x=.12,y=.14)
    save(fig,"conditional_pulse_effect", "Each dot is one of six joint initial-state/background blocks on one100,000-neuron graph; horizontal marks are means. Focal effect is present minus omitted focal pulse. Both histories occur at each focal time; all48episodes are included. No claim of identical future noise, independent graph replication or biological generalization.")
    effects=data["effects"];names=["delta_cortical_spikes_per_neuron"]+[k for k in effects if k!="delta_cortical_spikes_per_neuron"]
    fig,ax=plt.subplots(figsize=(7.2,3.7),layout="constrained")
    for i,key in enumerate(names):
        v=np.asarray(effects[key]["block_values"]);mean=effects[key]["mean"];lo,hi=effects[key]["empirical_block_bootstrap_percentile_95"]
        ax.plot([i,i],[lo,hi],color="black",linewidth=1.1)
        ax.scatter(i+np.linspace(-.12,.12,len(v)),v,s=16,color="#777777",alpha=.7)
        ax.plot(i,mean,"_",markersize=14,color="black",markeredgewidth=2)
    ax.axhline(0,color="#888888",linewidth=.7)
    ax.set_xticks(range(len(names)),["All cortex" if key==names[0] else key.removeprefix("delta_") for key in names])
    ax.set_ylabel("History × focal-pulse effect (spikes/neuron)")
    ax.set_title("Interaction across independent initial-state and noise blocks")
    ax.margins(x=.06,y=.12)
    save(fig,"history_pulse_interaction", "Six block values per outcome; each averages the two focal-time interactions. An interaction is the pulse-present minus omission effect after three conditioning pulses, minus that effect without conditioning. Black marks show mean and empirical block-bootstrap percentile95%interval; exhaustive6^6resampling removes Monte Carlo error, not small-sample uncertainty. Population outcomes are descriptive; no formal multiple-testing claim.")
    cv=data["observable_state_probe"]["block_results"]
    fig,ax=plt.subplots(figsize=(4.2,3.7),layout="constrained")
    a=np.asarray([r["cell_mean_mse"] for r in cv]);b=np.asarray([r["state_mse"] for r in cv])
    for u,v in zip(a,b):ax.plot([0,1],[u,v],"o-",color="#777777",alpha=.7,markersize=4,linewidth=.7)
    ax.plot([0,1],[a.mean(),b.mean()],"_",color="black",markersize=19,markeredgewidth=2)
    ax.set_xticks([0,1],["History × time","History × time\n+ observed state"])
    ax.set_ylabel("Held-out block MSE (spikes/neuron)²")
    ax.set_title("Predictive contribution of pre-stimulus activity")
    ax.set_xlim(-.25,1.25);ax.margins(y=.13)
    save(fig,"state_prediction_errors", "Each line joins errors on one held-out initial-state/background block (four history×time rows); black marks average six outer-fold errors. Both models retain unpenalized four-cell means. State uses24pre-emission population-rate features and nested whole-block ridge selection. Training folds overlap, so these are descriptive errors without an independent improvement confidence interval.")
    qc["source_analysis_sha256"]=sha256_file(analysis/"analysis.json")
    qc["style_helper_sha256"]=sha256_file(style_helper)
    (output/"render_checks.json").write_text(json.dumps(qc,indent=2))
    print(json.dumps(qc))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",type=Path,required=True)
    parser.add_argument("--style-helper",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args();render(a.analysis,a.style_helper,a.output)


if __name__=="__main__":main()
