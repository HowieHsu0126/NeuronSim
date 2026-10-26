"""Render all development state-probe errors without inferential CV intervals."""
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

    spec = importlib.util.spec_from_file_location("figure_style", style_helper)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_figure_style(sizes=(9, 8, 7))
    report = json.loads((analysis / "analysis.json").read_text())
    with (analysis / "prediction_by_block.csv").open() as f:
        rows = list(csv.DictReader(f))
    labels = {"cells": "History × time means", "R_linear": "Rates · linear",
              "R_rbf": "Rates · kernel", "S_linear": "Spike summaries · linear",
              "S_rbf": "Spike summaries · kernel", "P_linear": "Privileged state · linear",
              "P_rbf": "Privileged state · kernel"}
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    for position, model in enumerate(report["models"]):
        block_rows = sorted([r for r in rows if r["model"] == model["model"]], key=lambda r: int(r["block"]))
        if len(block_rows) != report["independent_blocks"] or len({r["block"] for r in block_rows}) != len(block_rows):
            raise ValueError("complete six-block model denominator required")
        values = np.asarray([float(r["test_mse"]) for r in block_rows])
        if not np.isclose(values.mean(), model["cv_mse"], rtol=0, atol=1e-12):
            raise ValueError("block errors and reported average differ")
        ax.scatter(values, position + np.linspace(-.12, .12, len(values)), color="#777777", alpha=.7, s=20,
                   label="Held-out block" if position == 0 else None)
        ax.plot(values.mean(), position, "|", color="#0072B2", markersize=16, markeredgewidth=2,
                label="Mean over blocks" if position == 0 else None)
    ax.set_yticks(range(len(report["models"])), [labels[r["model"]] for r in report["models"]])
    ax.invert_yaxis()
    ax.set_xlabel("Held-out mean squared error (spikes/neuron)² · lower is better")
    ax.set_title("State summaries and prediction of the 0–20 ms pulse effect")
    ax.set_xlim(left=0)
    ax.margins(y=.09, x=.07)
    limits = ax.get_xlim()
    ax.set_xticks([t for t in ax.get_xticks() if limits[0] <= t <= limits[1]])
    ax.set_xlim(limits)
    ax.legend(loc="upper right")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = [t.get_text() for t in fig.findobj(matplotlib.text.Text)
               if t.get_visible() and t.get_text().strip()
               and not all(fig.bbox.contains(*point) for point in t.get_window_extent(renderer).get_points())]
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / ("state_prediction_errors." + suffix))
    plt.close(fig)
    caption = (
        "Each dot is one held-out initial-state/background block, with four nested history×time conditions; "
        "blue ticks are averages over all six blocks on the same100,000-cortical-neuron graph. "
        "Rates use24features; spike summaries use760; privileged summaries use864 and add voltage/current/refractory information. "
        "All predictors retain the same unpenalized four-cell means. Model selection and scaling use training blocks only. "
        "Outer training folds overlap, so no independent-improvement interval is shown. "
        "All six augmented models have higher mean held-out error than the cell means. "
        "This negative development result concerns a scalar response; it establishes neither state sufficiency nor an absence of state effects. "
        "The full1000/1500ms results and training errors are retained in model_summary.csv and prediction_by_block.csv."
    )
    (output / "state_prediction_errors.txt").write_text(caption + "\n")
    receipt = dict(source_analysis_sha256=sha256_file(analysis / "analysis.json"),
                   source_block_errors_sha256=sha256_file(analysis / "prediction_by_block.csv"),
                   models=len(report["models"]), blocks=report["independent_blocks"],
                   confidence_intervals=False, text_outside_canvas=outside,
                   style_helper_sha256=sha256_file(style_helper))
    (output / "render_checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--style-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    render(a.analysis, a.style_helper, a.output)


if __name__ == "__main__":
    main()
