"""Source comparison fixtures preserve type identities, absent edges and mismatches."""
from copy import deepcopy
import unittest

from .ca3_parameter_comparison import compare, CORE_TYPES, NATIVE_CONNECTIONS


def fixture():
    names = [f"CA3 Type {i}" for i in range(CORE_TYPES)]
    symbols = [name.replace(" ", "_") for name in names]
    groups = [f'int {name} = sim.createGroup("{name}", {i+1}, INHIBITORY_NEURON);'
              for i, name in enumerate(symbols)]
    pairs = [(pre, post) for pre in symbols for post in symbols][:NATIVE_CONNECTIONS]
    connections = [f'sim.connect({pre}, {post}, "random", RangeWeight(0,1,2), 0.125f, RangeDelay(1));'
                   for pre, post in pairs]
    pop = [{"cell": "A1", "type": "s", "stored_value": "Neuron Type (Internal)"},
           {"cell": "B1", "type": "s", "stored_value": "Population Size (Internal)"}]
    prob = [{"cell": "A1", "type": "s", "stored_value": "Pre-Post"}]
    # Deliberately reverse columns, as in real source tables whose orders differ.
    for column, name in zip("BCDEFGHI", reversed(names)):
        prob.append(dict(cell=column+"1", type="s", stored_value=name))
    for row, name in enumerate(names, 2):
        pop.extend([dict(cell=f"A{row}", type="s", stored_value=name),
                    dict(cell=f"B{row}", type="n", stored_value=str(row-1))])
        prob.append(dict(cell=f"A{row}", type="s", stored_value=name))
        for column, post in zip("BCDEFGHI", reversed(symbols)):
            prob.append(dict(cell=f"{column}{row}", type="n",
                             stored_value="0.125" if (symbols[row-2], post) in pairs else "0"))
    tables = dict(sheets=[dict(name="pop_size_neuron_type", cells=pop),
                         dict(name="syn_conn_prob", cells=prob)])
    return tables, "\n".join(groups+connections), "read_excel(path, sheet_name='external')"


class ParameterComparisonTests(unittest.TestCase):
    def test_named_columns_comments_and_missing_external_sheet(self):
        tables, header, generator = fixture()
        header += '\n// sim.connect(fake, fake, "random", RangeWeight(0,1,2), 1f, RangeDelay(1));'
        result = compare(tables, header, generator)
        self.assertEqual(result["summary"]["population_counts_equal"], 8)
        self.assertEqual(result["summary"]["probabilities_float32_equal"], 64)
        self.assertEqual(result["summary"]["missing_generator_sheets"], ["external"])
        self.assertFalse(result["workbook_equivalence_qualified"])

    def test_count_difference_and_added_edge_are_retained(self):
        tables, header, generator = fixture()
        tables["sheets"][0]["cells"][3]["stored_value"] = "99"
        cell = next(c for c in tables["sheets"][1]["cells"] if c["cell"] == "B9")
        cell["stored_value"] = "0.25"
        result = compare(tables, header, generator)
        self.assertEqual(result["status"], "SOURCE_PARAMETERS_DIFFER")
        self.assertEqual(result["summary"]["population_counts_equal"], 7)
        self.assertEqual(result["summary"]["probabilities_float32_equal"], 63)
        self.assertEqual(len(result["probabilities"]), 64)

    def test_formula_cache_is_reported_without_evaluation(self):
        tables, header, generator = fixture()
        cell = tables["sheets"][0]["cells"][3]
        cell["formula"] = {"text": "UNSUPPORTED_FUNCTION()", "attributes": {}}
        result = compare(tables, header, generator)
        self.assertIn("formula", result["populations"][0]["workbook_cell"])
        self.assertFalse(result["formulas_evaluated"])

    def test_invalid_identity_missing_value_and_probability_rejected(self):
        original, header, generator = fixture()
        for change in ("identity", "missing", "range", "fractional_count"):
            tables = deepcopy(original)
            if change == "identity":
                tables["sheets"][0]["cells"][2]["stored_value"] = "another type"
            elif change == "fractional_count":
                tables["sheets"][0]["cells"][3]["stored_value"] = "1.5"
            else:
                cell = next(c for c in tables["sheets"][1]["cells"] if c["cell"] == "B2")
                cell["stored_value"] = None if change == "missing" else "1.01"
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare(tables, header, generator)


if __name__ == "__main__":
    unittest.main()
