"""Compare delivered EAG and population ensembles without new model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .latent_spike_diffusion_comparison import compare as compare_joint_counts


def compare(eag_root, population_root, output):
    return compare_joint_counts(eag_root, population_root, output, baseline_kind="eag")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("eag-root", "population-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.eag_root, args.population_root, args.output)))


if __name__ == "__main__":
    main()
