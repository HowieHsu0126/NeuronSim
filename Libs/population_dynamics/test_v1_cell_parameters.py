"""Specimen/model identities and GLIF family selection use source metadata."""
from copy import deepcopy

import pytest

from .v1_cell_parameters import GLIF3_TEMPLATE, select_models


def specimen(identity, model_id, file_id):
    return {"id": identity, "neuronal_models": [{"id": model_id,
        "neuronal_model_template": {"id": GLIF3_TEMPLATE},
        "well_known_files": [{"id": file_id, "attachable_id": model_id,
            "attachable_type": "NeuronalModel", "well_known_file_type": {"name": "NeuronalModelParameters"},
            "download_link": "/api/v2/well_known_file_download/" + str(file_id)}]}]}


def test_model_ids_are_not_inferred_from_specimen_or_source_type_ids():
    records = [specimen(12, 92, 302), specimen(11, 91, 301)]
    # Another available model family must not replace the source-defined GLIF3.
    other = deepcopy(records[0]["neuronal_models"][0])
    other.update(id=12, neuronal_model_template={"id": 395310469})
    records[0]["neuronal_models"].insert(0, other)
    rows = select_models(records, [11, 12])
    assert [(row["specimen_id"], row["neuronal_model_id"], row["well_known_file_id"])
            for row in rows] == [(11, 91, 301), (12, 92, 302)]


@pytest.mark.parametrize("change", ["missing_specimen", "duplicate_specimen", "missing_glif3", "ambiguous_glif3", "wrong_attachment", "wrong_download"])
def test_missing_or_ambiguous_source_bindings_are_rejected(change):
    records = [specimen(11, 91, 301), specimen(12, 92, 302)]
    if change == "missing_specimen":
        records.pop()
    elif change == "duplicate_specimen":
        records.append(deepcopy(records[0]))
    elif change == "missing_glif3":
        records[0]["neuronal_models"].clear()
    elif change == "ambiguous_glif3":
        records[0]["neuronal_models"].append(deepcopy(records[0]["neuronal_models"][0]))
    elif change == "wrong_attachment":
        records[0]["neuronal_models"][0]["well_known_files"][0]["attachable_id"] = 11
    else:
        records[0]["neuronal_models"][0]["well_known_files"][0]["download_link"] = '/api/v2/well_known_file_download/999'
    with pytest.raises(ValueError, match="V1_"):
        select_models(records, [11, 12])
