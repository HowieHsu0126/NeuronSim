"""Portable metadata fixtures for scoped frozen-source parameter binding."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

from population_dynamics import pfc_refinement_source_binding as binding
from population_dynamics.pfc_refinement_input_law import analyze
from population_dynamics.provenance import canonical_json


def json_bytes(value):
    return canonical_json(value).encode("utf-8")


def file_entry(name, data):
    return dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def exact(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


class Fixture:
    def __init__(self, root):
        self.root = root
        self.prepared = root / "prepared"
        self.prepared.mkdir()
        self.index_path = self.prepared / "source_index.json"
        self.analysis_path = root / "analysis.json"
        self.output = root / "binding.json"
        self.code = {
            "pfc_clock_coupling": b"# approved synthetic fixture: q=p/4; r=q/(1-q)\n",
            "pfc_native_drive": b"# approved synthetic fixture: N=1000; rate=1.4\n",
            "pfc_coupled_drive": b"# approved synthetic fixture: two clock levels\n",
        }
        self.hashes = {name: hashlib.sha256(data).hexdigest() for name, data in self.code.items()}
        self.index = dict(required_original_blocks=32, source_blocks={})
        parent_config = dict(excitatory_neurons=80000, inhibitory_neurons=20000,
                             dt_ms=0.05, duration_ms=5000.0)
        self.analysis_path.write_bytes(json_bytes(analyze(parent_config)))
        for seed in range(95001, 95033):
            case = dict(case_id=f"pfc_{seed}_strong_cue", source_rng_seed=seed,
                        stimulus_strength_nA=0.5, distractor_strength_nA=0.0)
            driver = dict(binomial_n=1000, binomial_probability=7e-5, external_rate_Hz=1.4,
                          n_e=80000, n_i=20000, total_ticks=100000, ticks_generated=100000,
                          dt_seconds=5e-5, random_stream_policy="source_global",
                          seeds={"source_rng_seed": seed})
            sources = {}
            native_archive = f"Output/native/{seed}"
            native_dir = "source_run"
            native_report = self.report(case, 0.05)
            native_report.update(status="PFC_SOURCE_EXECUTION_COMPLETE", driver=driver,
                                 code={"drive_sha256": self.hashes["pfc_native_drive"]})
            self.write_metadata(seed, "native", self.config(0.05), native_report)
            native_raw = file_entry(native_dir + "/raw.npz", b"unread native payload")
            native_delivery = dict(status="ARCHIVED_PFC_SOURCE_EXECUTION",
                                   source_execution_complete=True, exit_code=0, files=[
                self.snapshot_entry(seed, "native", "config.yaml", native_dir + "/config.yaml"),
                self.snapshot_entry(seed, "native", "report.json", native_dir + "/report.json"),
                native_raw,
            ])
            native_digest = self.write_delivery(seed, ["native"], native_delivery)
            sources["native"] = self.source(seed, "native", native_archive, native_raw, native_digest, None)
            native_report_sha = self.snapshot_entry(seed, "native", "report.json", "")["sha256"]
            for repeat, base in (("A", 95411), ("B", 99411)):
                labels = ["half_" + repeat, "quarter_" + repeat]
                archive = f"Output/coupled/{seed}_{repeat}"
                seeds = [base + 2 * (seed - 95001), base + 2 * (seed - 95001) + 1]
                entries = []
                raws = {}
                for label in labels:
                    dt = binding.SOURCE_CLOCKS[label]
                    factor = 2 if label.startswith("half") else 4
                    ticks = 100000 * factor
                    report = self.report(case, dt)
                    report.update(status="COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE",
                                  coarse_dt_ms=0.05, coupled_core_units=100000, refinement_seeds=seeds,
                                  code=self.hashes,
                                  coarse_reference=dict(archive=native_archive, delivery_sha256=native_digest,
                                                        report_sha256=native_report_sha),
                                  drive=dict(all_refinement_levels_generated=True, refinement_factor=factor,
                                             total_ticks=ticks, ticks_generated=ticks, n_e=80000, n_i=20000,
                                             dt_seconds=5e-5 / factor,
                                             coupling=dict(coarse_source=driver, refinement_seeds=seeds)))
                    self.write_metadata(seed, label, self.config(dt), report)
                    case_dir = f"source_cases/dt_{str(dt).replace('.', 'p')}/{case['case_id']}"
                    raw = file_entry(case_dir + "/raw.npz", b"unread coupled payload")
                    raws[label] = raw
                    entries.extend([
                        self.snapshot_entry(seed, label, "config.yaml", case_dir + "/config.yaml"),
                        self.snapshot_entry(seed, label, "report.json", case_dir + "/report.json"), raw,
                    ])
                for name, data in self.code.items():
                    entries.append(file_entry(f"Libs/population_dynamics/{name}.py", data))
                status = "COMPLETE_COUPLED_CLOCK_CALIBRATION_BLOCK" if repeat == "A" else "COMPLETE_COUPLED_REFINEMENT_REPEAT_BLOCK"
                digest = self.write_delivery(seed, labels, dict(status=status, verified=True, files=entries))
                for label in labels:
                    sources[label] = self.source(seed, label, archive, raws[label], digest, seeds)
                if seed == 95001 and repeat == "A":
                    for name, data in self.code.items():
                        path = root / archive / "Libs/population_dynamics" / (name + ".py")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
            self.index["source_blocks"][str(seed)] = dict(sources=sources)
        self.reseal()

    @staticmethod
    def config(dt):
        return dict(excitatory_neurons=80000, inhibitory_neurons=20000,
                    excitatory_preference_bins=16, dt_ms=dt, duration_ms=5000.0)

    def report(self, case, dt):
        ticks = round(5000 / dt)
        return dict(resolved_config=self.config(dt), selected_case=case, source_execution_complete=True,
                    completed_duration_ticks=ticks,
                    inputs_used=dict(ticks_consumed=ticks, E=dict(values=80000 * ticks),
                                     I=dict(values=20000 * ticks)))

    def metadata(self, seed, label):
        return self.prepared / f"source_metadata/{seed}/{label}"

    def write_metadata(self, seed, label, config, report):
        path = self.metadata(seed, label)
        path.mkdir(parents=True)
        (path / "config.yaml").write_text(yaml.safe_dump(config))
        (path / "report.json").write_bytes(json_bytes(report))

    def snapshot_entry(self, seed, label, name, original_name):
        return file_entry(original_name, (self.metadata(seed, label) / name).read_bytes())

    def write_delivery(self, seed, labels, delivery):
        data = json_bytes(delivery)
        for label in labels:
            (self.metadata(seed, label) / "delivery.json").write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def source(seed, label, archive, raw, digest, seeds):
        return dict(archive=archive, delivery_sha256=digest, metadata=f"source_metadata/{seed}/{label}",
                    raw=raw, dt_ms=binding.SOURCE_CLOCKS[label], refinement_seeds=seeds,
                    source_parent_dt_ms=None if label == "native" else 0.05)

    def change_report(self, seed, label, change):
        path = self.metadata(seed, label) / "report.json"
        report = json.loads(path.read_bytes())
        change(report)
        path.write_bytes(json_bytes(report))

    def reseal(self):
        """Refresh existing fixture manifests after a declared semantic change."""
        for seed_text, block in self.index["source_blocks"].items():
            seed = int(seed_text)
            for labels in (["native"], ["half_A", "quarter_A"], ["half_B", "quarter_B"]):
                original = json.loads((self.metadata(seed, labels[0]) / "delivery.json").read_bytes())
                by_name = {row["path"]: row for row in original["files"]}
                for label in labels:
                    case_dir = Path(block["sources"][label]["raw"]["path"]).parent
                    for name in ("config.yaml", "report.json"):
                        original_name = (case_dir / name).as_posix()
                        by_name[original_name] = self.snapshot_entry(seed, label, name, original_name)
                original["files"] = list(by_name.values())
                digest = self.write_delivery(seed, labels, original)
                for label in labels:
                    block["sources"][label]["delivery_sha256"] = digest
        self.seal_master()

    def seal_master(self):
        self.index_path.write_bytes(json_bytes(self.index))
        rows = [file_entry("source_index.json", self.index_path.read_bytes())]
        for seed in range(95001, 95033):
            for label in binding.SOURCE_CLOCKS:
                for name in ("config.yaml", "delivery.json", "report.json"):
                    relative = f"source_metadata/{seed}/{label}/{name}"
                    rows.append(file_entry(relative, (self.prepared / relative).read_bytes()))
        (self.prepared / "delivery.json").write_bytes(json_bytes(
            dict(status="COMPLETE_PFC_HISTORY_PROFILE_CENSUS_PREPARE", verified=True, files=rows)))

    def execute(self):
        return binding.run(self.analysis_path, self.index_path, self.root, self.output)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path)
    # The production approval remains fixed. Portable unit fixtures substitute
    # one explicitly approved synthetic hash group without a runtime override.
    monkeypatch.setattr(binding, "SUPPORTED_CODE_HASHES", fixture.hashes)
    return fixture


def test_complete_family_counts_shared_trees_once_and_preserves_scope(cohort):
    parent_bytes = cohort.analysis_path.read_bytes()
    report = cohort.execute()
    parent = json.loads(parent_bytes)
    bound = exact(parent["deterministic_parent_envelope"]["full_history_tv_bound"])
    assert report["parameter_arithmetic_binding_verified"] is True
    assert (report["indexed_metadata_sources"], report["authenticated_source_metadata_files"],
            report["coupled_source_labels_verified"], report["shared_two_level_trees"],
            report["unique_coupled_source_archives"]) == (160, 480, 128, 64, 64)
    assert exact(report["one_tree_tv_upper"]) == bound
    assert exact(report["per_block_two_A_B_tree_tv_upper"]) == 2 * bound
    assert exact(report["whole_indexed_family_tv_upper"]) == 64 * bound
    assert len(report["source_bindings"]) == 160 and len(report["trees"]) == 64
    assert len(report["representative_frozen_code_bindings"]) == 1
    assert len(report["representative_frozen_code_bindings"][0]["files"]) == 3
    assert report["provenance"]["explicit_paths_read"] < 1000
    assert all(not row["path"].endswith(".npz") for row in report["provenance"]["consumed_artifacts"])
    assert cohort.analysis_path.read_bytes() == parent_bytes
    for key in ("full_source_runtime_qualified", "finite_sampler_law_certified", "finite_prng_law_certified",
                "source_history_confidence_certified", "confirmation_0p01_128_certified",
                "coarse_fine_neuronal_equality_certified", "scientific_fidelity_qualified",
                "production_qualified", "statistical_independence_established", "seeds_used_to_infer_independence",
                "history_reconstructions_counted", "benchmark_development_episodes_counted"):
        assert report[key] is False
    for key in ("new_simulations", "source_simulations", "new_trials", "new_episodes", "production_episodes"):
        assert report[key] == 0


def test_tampered_snapshot_rejected_before_output(cohort):
    path = cohort.metadata(95001, "half_A") / "report.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="DIGEST_MISMATCH"):
        cohort.execute()
    assert not cohort.output.exists()


def test_index_digest_cannot_replace_original_cached_delivery(cohort):
    cohort.index["source_blocks"]["95001"]["sources"]["half_A"]["delivery_sha256"] = "0" * 64
    cohort.index_path.write_bytes(json_bytes(cohort.index))
    master_path = cohort.prepared / "delivery.json"
    master = json.loads(master_path.read_bytes())
    master["files"][0] = file_entry("source_index.json", cohort.index_path.read_bytes())
    master_path.write_bytes(json_bytes(master))
    with pytest.raises(ValueError, match="ORIGINAL_DELIVERY_DIGEST_CHANGED"):
        cohort.execute()


@pytest.mark.parametrize(("change", "reason"), [
    (lambda r: r["drive"]["coupling"]["coarse_source"].update(binomial_probability=7.1e-5),
     "COARSE_BINARY64_PROBABILITY_CHANGED"),
    (lambda r: r.update(completed_duration_ticks=199999), "SOURCE_CONSUMED_TICKS_CHANGED"),
    (lambda r: r["drive"].update(all_refinement_levels_generated=False), "COMPLETE_TWO_LEVEL_TREE_REQUIRED"),
    (lambda r: r["code"].update(pfc_clock_coupling="a" * 64), "UNSUPPORTED_SOURCE_CODE"),
    (lambda r: r["coarse_reference"].update(report_sha256="b" * 64), "ORIGINAL_COARSE_REFERENCE_CHANGED"),
])
def test_authenticated_semantic_failures_remain_rejected(cohort, change, reason):
    cohort.change_report(95001, "half_A", change)
    cohort.reseal()
    with pytest.raises(ValueError, match=reason):
        cohort.execute()


@pytest.mark.parametrize(("key", "value", "reason"), [
    ("excitatory_neurons", 79998, "SOURCE_GEOMETRY_CHANGED"),
    ("dt_ms", 0.05, "SOURCE_CLOCK_OR_DURATION_CHANGED"),
    ("duration_ms", 4999.0, "SOURCE_CLOCK_OR_DURATION_CHANGED"),
])
def test_geometry_and_clock_checked_after_manifest_authentication(cohort, key, value, reason):
    path = cohort.metadata(95001, "half_A") / "config.yaml"
    config = yaml.safe_load(path.read_bytes())
    config[key] = value
    path.write_text(yaml.safe_dump(config))
    cohort.change_report(95001, "half_A", lambda report: report.update(resolved_config=config))
    cohort.reseal()
    with pytest.raises(ValueError, match=reason):
        cohort.execute()


def test_incompatible_half_quarter_streams_cannot_be_counted_as_one_tree(cohort):
    seeds = [99001, 99002]
    source = cohort.index["source_blocks"]["95001"]["sources"]["quarter_A"]
    source["refinement_seeds"] = seeds
    def change(report):
        report["refinement_seeds"] = seeds
        report["drive"]["coupling"]["refinement_seeds"] = seeds
    cohort.change_report(95001, "quarter_A", change)
    cohort.reseal()
    with pytest.raises(ValueError, match="INCOMPATIBLE_HALF_QUARTER_TREE"):
        cohort.execute()


def test_declared_supported_code_bytes_are_authenticated(cohort):
    path = cohort.root / "Output/coupled/95001_A/Libs/population_dynamics/pfc_clock_coupling.py"
    path.write_bytes(b"# changed source arithmetic\n")
    with pytest.raises(ValueError, match="METADATA_OR_CODE_DIGEST_MISMATCH"):
        cohort.execute()


def test_display_numbers_do_not_supply_the_parent_proof(cohort):
    parent = json.loads(cohort.analysis_path.read_bytes())
    parent["levels"][0]["parent_probability"]["display_float"] = 0.99
    parent["deterministic_parent_envelope"]["full_history_tv_bound"]["display_float"] = 0.0
    cohort.analysis_path.write_bytes(json_bytes(parent))
    report = cohort.execute()
    assert report["proof_uses_display_floats"] is False
    assert exact(report["one_tree_tv_upper"]) > 0


def test_changed_exact_parent_probability_rejected(cohort):
    parent = json.loads(cohort.analysis_path.read_bytes())
    parent["levels"][0]["parent_probability"]["numerator"] = "1"
    cohort.analysis_path.write_bytes(json_bytes(parent))
    with pytest.raises(ValueError, match="PARENT_EXACT_PARAMETER_CHANGED"):
        cohort.execute()


def test_cli_dispatch_and_fresh_output_contract(cohort, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pfc_refinement_source_binding", "--analysis", str(cohort.analysis_path),
                                    "--source-index", str(cohort.index_path), "--project-root", str(cohort.root),
                                    "--output", str(cohort.output)])
    assert binding.main() == 0
    assert json.loads(capsys.readouterr().out)["shared_two_level_trees"] == 64
    before = cohort.output.read_bytes()
    with pytest.raises(FileExistsError, match="FRESH_OUTPUT_REQUIRED"):
        binding.main()
    assert cohort.output.read_bytes() == before


@pytest.mark.parametrize("direct_config", ["not_listed", "different_original_bytes"])
def test_prepared_config_can_bind_authenticated_report_without_direct_byte_identity(cohort, direct_config):
    labels = ["half_A", "quarter_A"]
    original_path = cohort.metadata(95001, labels[0]) / "delivery.json"
    original = json.loads(original_path.read_bytes())
    if direct_config == "not_listed":
        original["files"] = [row for row in original["files"]
                             if not row["path"].endswith("config.yaml")]
    else:
        for row in original["files"]:
            if row["path"].endswith("config.yaml"):
                row.update(bytes=7, sha256=hashlib.sha256(b"source\n").hexdigest())
    digest = cohort.write_delivery(95001, labels, original)
    for label in labels:
        cohort.index["source_blocks"]["95001"]["sources"][label]["delivery_sha256"] = digest
    cohort.seal_master()
    report = cohort.execute()
    rows = [row for row in report["source_bindings"]
            if row["source_block"] == "95001" and row["source_label"] in labels]
    assert len(rows) == 2
    assert all(row["original_direct_config_binding_verified"] is False for row in rows)
    assert all(row["prepared_config_binding"] ==
               "master_digest_and_authenticated_original_report_resolved_config" for row in rows)
    assert all((row["original_config_manifest_entry"] is None) == (direct_config == "not_listed")
               for row in rows)
    assert report["parameter_arithmetic_binding_verified"] is True
    assert report["full_source_runtime_qualified"] is False


def test_prepared_config_must_equal_authenticated_original_resolved_config(cohort):
    path = cohort.metadata(95001, "half_A") / "config.yaml"
    config = yaml.safe_load(path.read_bytes())
    config["duration_ms"] = 4999.0
    path.write_text(yaml.safe_dump(config))
    cohort.reseal()
    with pytest.raises(ValueError, match="SOURCE_CONFIG_REPORT_MISMATCH"):
        cohort.execute()
