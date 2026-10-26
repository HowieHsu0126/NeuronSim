"""Complete source-bound count cohorts shared by forecasting backends.

The cohort keeps the existing 56/16 development split, complete episodes and
source observation identities. Importing this module loads no model framework.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .native_count_windows import NativeEpisodeCounts
from .provenance import sha256_file


def partition(design, config):
    if design["acquired_split"] != "train":
        raise ValueError("development training consumes only the training cohort")
    fitting, validation = set(config["fit_blocks"]), set(config["validation_blocks"])
    all_blocks = {c["block"] for c in design["cases"]}
    if not fitting or not validation or fitting & validation or fitting | validation != all_blocks:
        raise ValueError("complete disjoint fitting/validation block partition required")
    if config["training_seeds"] != [41, 42, 43]:
        raise ValueError("training seeds must be exactly41/42/43")
    rows = sorted(design["cases"], key=lambda c: c["case_id"])
    if any(c["split"] != "train" or c["family"] != "trajectory" for c in rows):
        raise ValueError("homogeneous complete training trajectories required")
    return ([c for c in rows if c["block"] in fitting],
            [c for c in rows if c["block"] in validation])


def load_dataset_index(fields: Path) -> dict[str, dict]:
    """Read the existing hash-bound episode table, rejecting incomplete identities."""
    fields = Path(fields)
    dataset = json.loads((fields / "dataset.json").read_text())
    if (dataset["status"] != "BUILT"
            or sha256_file(fields / "episodes.csv") != dataset["episode_table_sha256"]):
        raise ValueError("complete field dataset and unchanged episode table required")
    with (fields / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (not rows or len(rows) != dataset["completed_episodes"]
            or len({r["case_id"] for r in rows}) != len(rows)
            or any(not r["case_id"] or Path(r["case_id"]).name != r["case_id"]
                   or r["split"] != dataset["split"]
                   or any(len(r[key]) != 64 for key in ("raw_report_sha256", "field_summary_sha256"))
                   for r in rows)):
        raise ValueError("complete unique case identities and report/field digests required")
    return {row["case_id"]: row for row in rows}


class TrajectoryCohort:
    """Verified complete Philox episodes; no neuron selection or time cropping."""
    def __init__(self, acquisition, fields, config):
        self.acquisition, self.fields, self.config = acquisition, fields, config
        design = json.loads((acquisition / 'design.json').read_text())
        dataset = json.loads((fields / 'dataset.json').read_text())
        ledger = json.loads((acquisition / 'acquisition.json').read_text())
        self.fit, self.validation = partition(design, config)
        cases = self.fit + self.validation
        if len(self.fit) != 56 or len(self.validation) != 16:
            raise ValueError('complete 56-fit/16-selection Philox episode partition required')
        self.records = load_dataset_index(fields)
        if (len(self.fit) != 56 or len(self.validation) != 16 or len(self.records) != 72
                or set(self.records) != {case['case_id'] for case in cases}
                or any({c['template_id'] for c in cases if c['block'] == b} != {f'familiar_{i}' for i in range(4)} for b in range(18))):
            raise ValueError('complete 56-fit/16-selection Philox episode partition required')
        if (dataset['split'] != 'train' or dataset['design_sha256'] != sha256_file(acquisition / 'design.json')
                or dataset['cortical_units'] != [100000] or dataset['settings']['bin_ms'] != 5.
                or dataset['control_delivery_violations'] != 0 or dataset['refractory_violations'] != 0
                or ledger['status'] != 'ACQUIRED' or ledger['completed_episodes'] != 72
                or len(ledger['runs']) != 72 or {r['case_id'] for r in ledger['runs']} != set(self.records)
                or any(row['exit_code'] != 0 for row in ledger['runs'])):
            raise ValueError('complete source-bound training acquisition required')
        self.source_identity = dict(native_library_sha256=design['native_library_sha256'],
            graph_manifest_sha256=design['graph_manifest_sha256'], native_poisson_rng_family=config['expected_poisson_rng_family'])
        for case in cases:
            self._verify_metadata(case)
        first = NativeEpisodeCounts(acquisition / cases[0]['case_id'], fields / cases[0]['case_id'])
        self.unit_ids, self.unit_field_index, self.field_ids = first.unit_ids.copy(), first.unit_field_index.copy(), first.field_ids.copy()
        self.history_bins, self.horizon_bins, self.bin_seconds = config['history_bins'], config['horizon_bins'], config['bin_ms'] / 1000
        self._verify_episode(first)
        self.binding = dict(acquisition_design_sha256=sha256_file(acquisition / 'design.json'),
            acquisition_ledger_sha256=sha256_file(acquisition / 'acquisition.json'),
            dataset_sha256=sha256_file(fields / 'dataset.json'), episode_table_sha256=dataset['episode_table_sha256'],
            native_source_identity=self.source_identity,
            episode_bindings={case_id:{key:row[key] for key in ('raw_report_sha256','field_summary_sha256')} for case_id,row in self.records.items()})

    def _verify_metadata(self, case):
        row = self.records[case['case_id']]
        raw, field = self.acquisition / case['case_id'], self.fields / case['case_id']
        if (row['split'] != 'train' or int(row['block']) != case['block'] or row['template_id'] != case['template_id']
                or int(row['cortical_units']) != 100000 or sha256_file(raw / 'report.json') != row['raw_report_sha256']
                or sha256_file(field / 'summary.json') != row['field_summary_sha256']):
            raise ValueError('bound native episode/field identity changed')
        report = json.loads((raw / 'report.json').read_text())
        if (report['status'] != 'EXECUTED' or report['config'] != case['settings']
                or report['duration_ms'] != 6000. or report['dt_ms'] != .1
                or any(report.get(key) != value for key,value in self.source_identity.items())):
            raise ValueError('native episode source, full duration or actual stimulus program changed')

    def _verify_episode(self, data):
        if (len(data.unit_ids) != 100000 or data.unit_field_index.shape != data.unit_ids.shape
                or data.unit_field_index.dtype.kind not in 'iu' or data.field_ids.ndim != 1
                or np.any(data.unit_field_index < 0) or np.any(data.unit_field_index >= len(data.field_ids))
                or len(set(map(str,data.field_ids))) != len(data.field_ids)
                or data.bin_ms != 5. or data.dt_ms != .1
                or data.duration_ticks * data.dt_ms != 6000. or data.duration_ticks // data.bin_ticks != 1200
                or not np.array_equal(data.unit_ids, self.unit_ids)
                or not np.array_equal(data.unit_field_index, self.unit_field_index)
                or not np.array_equal(data.field_ids, self.field_ids)):
            raise ValueError('complete 1200-bin/100000-unit episode and fixed observation map required')

    def window(self, case):
        self._verify_metadata(case)
        data = NativeEpisodeCounts(self.acquisition / case['case_id'], self.fields / case['case_id'])
        self._verify_episode(data)
        counts, stimulus = data.window(0, data.duration_ticks)
        if counts.shape != (1200,100000) or stimulus.shape != (1200,1):
            raise ValueError('complete identified episode counts and actual TC inputs required')
        return counts, stimulus
