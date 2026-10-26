"""Validate full population row identities and rejection of rate differences."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from . import v1_lgn_rate_cost as cost


@pytest.mark.parametrize("changed", [False, True])
def test_full_reference_uses_ids_and_rejects_one_changed_value(tmp_path, monkeypatch, changed):
    units = 17400
    ids = np.arange(units)
    clock = np.arange(3) / 1000.
    rates = ids[:, None] + np.arange(3)[None, :]
    cells = pd.DataFrame(dict(model_id=[f"model_{index % 14}" for index in ids],
                              x=ids + .1, y=ids + .2), index=ids)
    binding = dict(content="same construction")
    reference = tmp_path / "reference"
    (reference / "probe").mkdir(parents=True)
    previous = dict(status="SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT",
                    input_construction_binding=binding, source_unit_ids=ids[::-1].tolist(),
                    config=dict(stimulus=dict(contrast_percent=80, duration_s=.002,
                        gray_screen_s=0., spatial_frequency_cycles_per_degree=.04,
                        temporal_frequency_hz=2., direction_deg=0)))
    (reference / "probe/probe.json").write_text(json.dumps(previous))
    with h5py.File(reference / "probe/firing_rates_f_tot.h5", "w") as saved:
        saved["time"] = clock
        saved["firing_rates_Hz"] = rates[::-1]
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "prepared.json").write_text("{}")

    def calculate(nodes, kind, prefix, *parameters):
        assert [row[0] for row in nodes.nodes()] == ids.tolist()
        actual = rates.copy()
        if changed:
            actual[500, 1] += 1
        with h5py.File(prefix + "_f_tot.h5", "w") as generated:
            generated["time"] = clock
            generated["firing_rates_Hz"] = actual

    monkeypatch.setattr(cost, "bound_json", lambda *args: (previous, {}))
    monkeypatch.setattr(cost, "read_input_construction_cells", lambda *args: (cells, None, binding))
    monkeypatch.setattr(cost, "delivered_file", lambda archive, name: archive / name)
    monkeypatch.setattr(cost, "load_prepared_runtime", lambda *args: (SimpleNamespace(calculate_firing_rate=calculate), None))
    output = tmp_path / "output"
    if changed:
        with pytest.raises(ValueError, match="REFERENCE_VALUES_DIFFER"):
            cost.profile_rates(runtime, reference, reference, output, all_units=True)
        report = json.loads((output / "cost.json").read_text())
        assert report["status"] == "FAILED_LGN_RATE_COST_DIAGNOSTIC"
        assert report["exact"]["rates"] is False
    else:
        report = cost.profile_rates(runtime, reference, reference, output, all_units=True)
        assert all(report["exact"].values())
        assert report["rate_shape"] == [units, 3]
        assert report["profiled_execution"] is False
        assert not (output / "source_calls.prof").exists()
    assert report["production_episodes"] == report["coupled_neural_episodes"] == 0
