"""Compare a recovered CA3 workbook with the pinned native count/probability literals.

This is source reconciliation, not generator replay or scale qualification.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import struct

from .provenance import sha256_file

CORE_TYPES = 8
NATIVE_CONNECTIONS = 51
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def source_name(value: str) -> str:
    """Apply the pinned utilities.cellTypeNameCARL value transformation."""
    return value.replace(" ", "_").replace("+", "").replace("-", "_")


def native_parameters(text: str) -> tuple[dict[str, int], dict[tuple[str, str], str]]:
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    groups = re.findall(
        r'int\s+(\w+)\s*=\s*sim\.createGroup\(\s*"([^"]+)"\s*,\s*(\d+)\s*,', text)
    if len(groups) != CORE_TYPES or text.count("sim.createGroup(") != len(groups):
        raise ValueError("NATIVE_CORE_GROUP_SCHEMA")
    counts = {}
    for symbol, label, count in groups:
        if symbol != label or symbol in counts or int(count) <= 0:
            raise ValueError("NATIVE_CORE_GROUP_IDENTITY")
        counts[symbol] = int(count)
    connections = re.findall(
        r'sim\.connect\(\s*(\w+)\s*,\s*(\w+)\s*,\s*"random"\s*,'
        r'\s*RangeWeight\([^)]*\)\s*,\s*(' + NUMBER + r')f\s*,', text)
    if len(connections) != NATIVE_CONNECTIONS or text.count("sim.connect(") != len(connections):
        raise ValueError("NATIVE_CONNECTION_SCHEMA")
    probabilities = {}
    for pre, post, value in connections:
        if pre not in counts or post not in counts or (pre, post) in probabilities:
            raise ValueError("NATIVE_CONNECTION_IDENTITY")
        number = Decimal(value)
        if not number.is_finite() or not 0 < number <= 1:
            raise ValueError("NATIVE_CONNECTION_PROBABILITY")
        probabilities[pre, post] = value
    return counts, probabilities


def _number(cell: dict, *, count: bool = False) -> Decimal:
    if cell.get("type") != "n" or cell.get("stored_value") is None:
        raise ValueError("WORKBOOK_MISSING_NUMERIC_VALUE:" + cell["cell"])
    value = Decimal(cell["stored_value"])
    if not value.is_finite() or (count and (value <= 0 or value != int(value))):
        raise ValueError("WORKBOOK_INVALID_COUNT:" + cell["cell"])
    if not count and not 0 <= value <= 1:
        raise ValueError("WORKBOOK_INVALID_PROBABILITY:" + cell["cell"])
    return value


def compare(tables: dict, header: str, generator: str) -> dict:
    counts, native = native_parameters(header)
    sheets = {sheet["name"]: sheet for sheet in tables["sheets"]}
    if len(sheets) != len(tables["sheets"]):
        raise ValueError("DUPLICATE_WORKBOOK_SHEET")
    cells = {}
    for name in ("pop_size_neuron_type", "syn_conn_prob"):
        rows = sheets[name]["cells"]
        cells[name] = {row["cell"]: row for row in rows}
        if len(cells[name]) != len(rows):
            raise ValueError("DUPLICATE_WORKBOOK_CELL")
    pop = cells["pop_size_neuron_type"]
    prob = cells["syn_conn_prob"]
    if [pop[c]["stored_value"] for c in ("A1", "B1")] != [
            "Neuron Type (Internal)", "Population Size (Internal)"]:
        raise ValueError("WORKBOOK_POPULATION_SCHEMA")
    if prob["A1"]["stored_value"] != "Pre-Post":
        raise ValueError("WORKBOOK_PROBABILITY_SCHEMA")
    populations = [source_name(pop[f"A{r}"]["stored_value"]) for r in range(2, 10)]
    pre_names = [source_name(prob[f"A{r}"]["stored_value"]) for r in range(2, 10)]
    columns = {source_name(prob[f"{c}1"]["stored_value"]): c for c in "BCDEFGHI"}
    expected = set(counts)
    if (len(set(populations)) != CORE_TYPES or set(populations) != expected
            or len(set(pre_names)) != CORE_TYPES or set(pre_names) != expected
            or len(columns) != CORE_TYPES or set(columns) != expected):
        raise ValueError("WORKBOOK_CORE_TYPE_IDENTITIES")
    population_rows = []
    for row, name in enumerate(populations, 2):
        cell = pop[f"B{row}"]
        value = int(_number(cell, count=True))
        population_rows.append(dict(
            type=name, workbook_cell=cell, workbook_count=value,
            native_count=counts[name], equal=value == counts[name]))
    probability_rows = []
    # The native generator reindexes columns by type name before emitting pairs.
    # Compare all 64 directed pairs; a source zero must not hide a missing edge.
    for row, pre in enumerate(pre_names, 2):
        for post in pre_names:
            cell = prob[f"{columns[post]}{row}"]
            value = _number(cell)
            literal = native.get((pre, post), "0")
            native_value = Decimal(literal)
            workbook_bits = struct.pack("!f", float(value)).hex()
            native_bits = struct.pack("!f", float(native_value)).hex()
            probability_rows.append(dict(
                pre=pre, post=post, workbook_cell=cell, native_literal=literal,
                native_connection_present=(pre, post) in native,
                decimal_equal=value == native_value,
                absolute_decimal_difference=str(abs(value - native_value)),
                float32_equal=workbook_bits == native_bits,
                workbook_float32_hex=workbook_bits, native_float32_hex=native_bits))
    required_sheets = sorted(set(re.findall(r"sheet_name\s*=\s*['\"]([^'\"]+)['\"]", generator)))
    missing_sheets = [name for name in required_sheets if name not in sheets]
    summary = dict(
        population_count_comparisons=len(population_rows),
        population_counts_equal=sum(row["equal"] for row in population_rows),
        candidate_core_neurons=sum(row["workbook_count"] for row in population_rows),
        native_core_neurons=sum(counts.values()),
        directed_probability_comparisons=len(probability_rows),
        native_active_connections=len(native),
        candidate_active_connections=sum(_number(row["workbook_cell"]) > 0 for row in probability_rows),
        probabilities_decimal_equal=sum(row["decimal_equal"] for row in probability_rows),
        probabilities_float32_equal=sum(row["float32_equal"] for row in probability_rows),
        missing_generator_sheets=missing_sheets,
        population_order_matches_native=list(counts) == populations,
        probability_row_order_matches_native=list(counts) == pre_names)
    matches = (summary["population_counts_equal"] == CORE_TYPES
               and summary["probabilities_float32_equal"] == CORE_TYPES**2)
    return dict(
        status="SOURCE_COUNT_PROBABILITY_VALUES_MATCH" if matches else "SOURCE_PARAMETERS_DIFFER",
        summary=summary, populations=population_rows, probabilities=probability_rows,
        required_generator_sheets=required_sheets,
        formulas_evaluated=False, macros_executed=False,
        generator_replay_verified=False, workbook_equivalence_qualified=False,
        source_scale_qualified=False,
        scope="Eight core counts and all directed connection probabilities only; "
              "intrinsic, synaptic and external-input parameters are not certified by this comparison.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.tables.read_text()), args.header.read_text(),
                     args.generator.read_text())
    result["inputs"] = {key: dict(path=str(path), sha256=sha256_file(path))
                        for key, path in vars(args).items() if key != "output"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], **result["summary"])))


if __name__ == "__main__":
    main()
