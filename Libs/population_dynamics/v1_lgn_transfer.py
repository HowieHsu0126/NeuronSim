"""Preserve SymPy transfer arithmetic with direct numeric Heaviside dispatch."""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import importlib
import math
from pathlib import Path

import numpy as np
import sympy

from .v1_lgn_runtime import load_prepared_runtime, probe_runtime


def numeric_heaviside(value, at_zero=0.5):
    """Return the source SymPy constants for finite numeric signs.

    The generated expression still performs its original SymPy multiplication
    and addition. Zero, nonfinite values and symbolic arguments use SymPy's
    unchanged implementation, including the declared value at the boundary.
    """
    if isinstance(value, (float, np.floating)) and math.isfinite(value):
        if value > 0:
            return sympy.S.One
        if value < 0:
            return sympy.S.Zero
    elif isinstance(value, sympy.Float):
        if value.is_positive:
            return sympy.S.One
        if value.is_negative:
            return sympy.S.Zero
    return sympy.Heaviside(value, at_zero)


@contextmanager
def numeric_transfer_support():
    """Accelerate closures built in this invocation; retain source expressions."""
    transfer = importlib.import_module("lgnmodel.transferfunction")
    original = transfer.lambdify

    def numeric_lambdify(*args, **kwargs):
        closure = original(*args, **kwargs)
        if closure.__globals__.get("Heaviside") is sympy.Heaviside:
            closure.__globals__["Heaviside"] = numeric_heaviside
        return closure

    transfer.lambdify = numeric_lambdify
    try:
        yield
    finally:
        transfer.lambdify = original


def main() -> None:
    """Run the existing rate probe with unchanged source expression arithmetic."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tables-archive", type=Path)
    source.add_argument("--input-construction", type=Path)
    parser.add_argument("--reference-rate-archive", type=Path)
    args = parser.parse_args()
    load_prepared_runtime(args.runtime)
    with numeric_transfer_support():
        probe_runtime(args.runtime, args.tables_archive, args.config, args.output,
                      args.reference_rate_archive, input_construction_archive=args.input_construction)


if __name__ == "__main__":
    main()
