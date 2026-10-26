"""Contract tests use synthetic fixtures only; they do not qualify real recordings."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import requests
import yaml

from .cohort import build_cohort, cohort_session, freeze_cohort, resolve_source
from .config import load_config
from .download import _remote_identity, download_nwb, verify_local_nwb
from .qualification import NwbSpikeTimes, SessionData, qualify_session, read_nwb_handle, reconstruct_trials, write_selected_recording

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/neural_population_dynamics.yaml"


def config():
    result = load_config(CONFIG_PATH)
    result["cohort"]["expected_metadata_session_count"] = 5
    result["cohort"]["expected_protocol_session_count"] = 4
    result["cohort"]["split_counts_by_donor"] = {"train":1,"validation":0,"test":1}
    return result


def metadata():
    return [
        {"id": 847657808, "specimen_id": 827809884, "donor_id": 827809879, "session_type": "functional_connectivity", "has_nwb": "True"},
        {"id": 847657809, "specimen_id": 827809885, "donor_id": 827809879, "session_type": "functional_connectivity", "has_nwb": "True"},
        {"id": 847657810, "specimen_id": 100, "donor_id": 90, "session_type": "functional_connectivity", "has_nwb": "True"},
        {"id": 847657811, "specimen_id": 101, "donor_id": 91, "session_type": "brain_observatory_1.1", "has_nwb": "True"},
        {"id": 847657812, "specimen_id": 102, "donor_id": 92, "session_type": "functional_connectivity", "has_nwb": "False"},
    ]


def session_fixture():
    presentations = pd.DataFrame({
        "stimulus_name": ["natural_movie_fixture"] * 6 + ["drifting_gratings"] * 2,
        "stimulus_block": [0]*6+[1,1], "frame": [0,1,2,0,1,2,np.nan,np.nan],
        "start_time": [1,1.1,1.2,1.3,1.4,1.5,2,3],
        "stop_time": [1.1,1.2,1.3,1.4,1.5,1.6,2.5,3.5],
        "orientation": [np.nan]*6+[90,90],
    }, index=pd.Index(range(8), name="stimulus_presentation_id"))
    units = pd.DataFrame({"quality": ["good", "good"], "presence_ratio": [0.99,0.8],
                          "amplitude_cutoff": [0.01,0.01], "isi_violations": [0.02,0.02]}, index=[10,11])
    return SessionData(847657808, "827809879", units,
                       {10: np.array([0.2,0.9,1.2,2.4]), 11: np.array([0.4,1.8])}, presentations,
                       pd.DataFrame({"start_time": [1.15], "stop_time": [1.17], "tags": [["probeA"]]}),
                       pd.DataFrame({"start_time": [4.0], "stop_time": [4.01], "power": [1.0]}))


class ConfigAndCohortTests(unittest.TestCase):
    def test_current_seeds_are_exact(self):
        self.assertEqual(config()["training_seeds"], [41,42,43])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.yaml"
            for value in ([], {**config(), "training_seeds": [40,41,42]},
                          {**config(), "eligibility": {}}):
                path.write_text(yaml.safe_dump(value))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_all_matching_sessions_selected_without_activity_screening(self):
        cohort = build_cohort(metadata(), config())
        self.assertEqual(cohort["metadata_session_count"], 5)
        self.assertEqual(cohort["included_session_count"], 3)
        self.assertFalse(cohort["dataset_ready"])
        self.assertEqual(cohort["sessions"][4]["exclusion_reasons"], ["session_nwb_missing_or_unverified"])

    def test_same_animal_and_exploration_anchor_stay_train(self):
        rows = build_cohort(metadata(), config())["sessions"]
        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(rows[0]["split"], rows[1]["split"])
        self.assertEqual(rows[0]["source"]["file_id"], 1026123943)
        self.assertIsNone(rows[1]["source"]["file_id"])

    def test_repeated_freeze_order_invariant_and_changed_policy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"cohort.json"
            first = freeze_cohort(path, metadata(), config())
            self.assertEqual(first, freeze_cohort(path, list(reversed(metadata())), config()))
            changed = config(); changed["cohort"]["split_salt"] += "changed"
            with self.assertRaises(FileExistsError):
                freeze_cohort(path, metadata(), changed)

    def test_duplicate_sessions_rejected_and_missing_animals_accounted(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE_SESSION_ID"):
            build_cohort(metadata()+metadata()[:1], config())
        rows = metadata(); rows[2].pop("donor_id")
        with self.assertRaisesRegex(ValueError, "DONOR_QUOTA_COUNT_MISMATCH"):
            build_cohort(rows, config())

    def test_incomplete_metadata_or_missing_anchor_donor_cannot_freeze(self):
        with self.assertRaisesRegex(ValueError, "SOURCE_METADATA_COHORT_INCOMPLETE"):
            build_cohort(metadata()[:-1], config())
        rows=metadata();rows[0].pop("donor_id")
        with self.assertRaisesRegex(ValueError, "EXPLORATION_ANIMAL_IDENTITY_MISSING"):
            build_cohort(rows, config())

    def test_consumed_cohort_rejects_cross_animal_split_tamper(self):
        cohort = build_cohort(metadata(), config())
        cohort["sessions"][1]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "ANIMAL_CROSSES_SPLITS"):
            cohort_session(cohort, 847657808)

    def test_qualify_cli_source_inputs_are_mutually_exclusive(self):
        import contextlib,io
        from .__main__ import parser
        base=["--config",str(CONFIG_PATH),"qualify","--session-id","847657808"]
        parsed=parser().parse_args(base+["--nwb","source.nwb"])
        self.assertEqual(parsed.local_nwb,Path("source.nwb"))
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            parser().parse_args(base+["--nwb","source.nwb","--source-url","https://example.invalid"])

    def test_metadata_cli_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/"metadata.json"; source.write_text(json.dumps(metadata()))
            cfg=root/"config.yaml";cfg.write_text(yaml.safe_dump(config()))
            script = '''import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0]=='torch': raise AssertionError('Torch import forbidden')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from Libs.population_dynamics.__main__ import main
code=main(['--config',sys.argv[1],'cohort','--metadata-json',sys.argv[2],'--output-dir',sys.argv[3]])
assert 'torch' not in sys.modules
raise SystemExit(code)
'''
            result = subprocess.run([sys.executable,"-c",script,str(cfg),str(source),str(root/"out")],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)["included_session_count"],3)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.config = config()
        self.config["movie_frames_per_repeat"]["natural_movie_fixture"] = 3
        self.row = cohort_session(build_cohort(metadata(), self.config),847657808)

    def test_movie_repeats_not_frames_and_invalid_overlap_not_shortened(self):
        data = session_fixture()
        trials,members = reconstruct_trials(data.presentations,data.invalid_times,data.session_id,827809879,"train",self.config["movie_frames_per_repeat"])
        self.assertEqual(len(trials),4)
        self.assertEqual(len(members),8)
        self.assertEqual(list(trials["presentation_count"]),[3,3,1,1])
        self.assertEqual(trials.iloc[0]["start_time"],1.0)
        self.assertEqual(trials.iloc[0]["stop_time"],1.3)
        self.assertFalse(trials.iloc[0]["structurally_eligible"])
        self.assertTrue(trials.iloc[1]["structurally_eligible"])
        self.assertEqual(set(members["split"]),{"train"})
        replay,_ = reconstruct_trials(data.presentations,data.invalid_times,data.session_id,827809879,"train",self.config["movie_frames_per_repeat"])
        self.assertEqual(list(replay["trial_id"]),list(trials["trial_id"]))

    def test_missing_movie_frame_remains_an_incomplete_trial(self):
        data = session_fixture(); data.presentations = data.presentations.drop(index=1)
        trials,_ = reconstruct_trials(data.presentations,data.invalid_times,data.session_id,827809879,"train",self.config["movie_frames_per_repeat"])
        self.assertEqual(len(trials),4)
        self.assertIn("incomplete_or_nonsequential",trials.iloc[0]["exclusion_reasons"])
        self.assertEqual(trials.iloc[0]["stop_time"],1.3)

    def test_source_stimulus_parameters_are_not_collapsed(self):
        data=session_fixture()
        data.presentations.loc[6,"direction"]=0
        data.presentations.loc[7,"direction"]=90
        trials,_=reconstruct_trials(data.presentations,data.invalid_times,data.session_id,827809879,"train",self.config["movie_frames_per_repeat"])
        self.assertNotEqual(trials.iloc[2]["condition_id"],trials.iloc[3]["condition_id"])

    def test_unknown_movie_length_and_frame_gap_are_explicit(self):
        data = session_fixture(); data.presentations.loc[1,"start_time"] += 0.001
        trials,_ = reconstruct_trials(data.presentations,data.invalid_times,data.session_id,827809879,"train",{})
        self.assertIn("movie_repeat_length_not_declared",trials.iloc[0]["exclusion_reasons"])
        self.assertIn("movie_frame_time_gap_or_overlap",trials.iloc[0]["exclusion_reasons"])

    def test_partial_preserves_raw_units_and_opto_separation(self):
        result = qualify_session(session_fixture(),self.row,self.config)
        summary = result["summary"]
        self.assertEqual(summary["status"],"PARTIAL")
        self.assertEqual(summary["raw_unit_count"],2)
        self.assertEqual(summary["qc_eligible_unit_count"],1)
        self.assertEqual(summary["raw_spike_count"],6)
        self.assertEqual(summary["optotagging_presentation_count"],1)
        self.assertFalse(summary["dataset_ready"])
        self.assertFalse(summary["model_ready"])
        self.assertFalse(summary["recording_coverage_source_verified"])
        self.assertIn("presence_ratio_threshold",result["units"].iloc[1]["exclusion_reasons"])

    def test_lazy_spike_indices_cover_exact_raw_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory)/"spikes.h5", "w") as handle:
                dataset=handle.create_dataset("times",data=np.array([0.1,0.2,0.3,0.4,0.5]))
                lazy=NwbSpikeTimes(dataset,np.array([10,11]),np.array([2,5]))
                self.assertEqual(list(lazy),[10,11])
                np.testing.assert_array_equal(lazy[11],np.array([0.3,0.4,0.5]))
                for ends in (np.array([2,4]),np.array([4,2,5],dtype=np.uint64)):
                    with self.assertRaises(ValueError):
                        NwbSpikeTimes(dataset,np.arange(len(ends)),ends)

    def test_bad_spike_shape_units_sort_and_identity_fail(self):
        changes = [lambda data: setattr(data,"spike_time_unit","milliseconds"),
                   lambda data: data.spike_times.__setitem__(10,np.array([[0.1,0.2]])),
                   lambda data: data.spike_times.__setitem__(10,np.array([0.2,0.1])),
                   lambda data: data.spike_times.__setitem__(10,np.array([np.nan])),
                   lambda data: data.spike_times.__setitem__(12,np.array([0.1])),
                   lambda data: setattr(data,"session_id",123),
                   lambda data: setattr(data,"subject_id","827809884")]
        for change in changes:
            data = session_fixture();change(data)
            with self.subTest(change=change),self.assertRaises(ValueError):
                qualify_session(data,self.row,self.config)

    def test_duplicate_presentations_and_invalid_intervals_fail(self):
        data = session_fixture();data.presentations.index = [0]*8
        with self.assertRaisesRegex(ValueError,"DUPLICATE_PRESENTATION"):
            qualify_session(data,self.row,self.config)
        data = session_fixture();data.invalid_times.loc[0,"stop_time"] = 0
        with self.assertRaisesRegex(ValueError,"INVALID_TIME_INTERVAL"):
            qualify_session(data,self.row,self.config)

    def test_signed_relative_clock_is_preserved_without_zero_clipping(self):
        data=session_fixture()
        data.spike_times[10]=np.array([-0.2,0.1])
        data.presentations[["start_time","stop_time"]] -= 2
        data.invalid_times[["start_time","stop_time"]] -= 2
        result=qualify_session(data,self.row,self.config)
        self.assertEqual(result["trials"].iloc[0]["start_time"],-1.0)
        self.assertEqual(result["units"].iloc[0]["first_observed_spike_seconds"],-0.2)

    def test_unavailable_history_is_not_clipped(self):
        data = session_fixture();data.recording_intervals=np.array([[0,10]])
        self.config["eligibility"] = {"minimum_eligible_units":1,"minimum_valid_repeats_per_condition":1,
                                      "required_history_seconds":2,"required_horizon_seconds":1}
        result = qualify_session(data,self.row,self.config)
        self.assertEqual(result["trials"].iloc[0]["requested_window_coverage"],"INSUFFICIENT_RECORDING_COVERAGE")
        self.assertEqual(result["trials"].iloc[0]["start_time"],1)
        self.assertEqual(result["summary"]["status"],"PARTIAL")


class Hdf5ReaderTests(unittest.TestCase):
    def _write_fixture(self, path):
        with h5py.File(path, "w") as handle:
            handle.attrs["nwb_version"]="2.6.0"
            handle.create_dataset("identifier",data="847657808")
            handle.create_dataset("general/subject/subject_id",data="827809879")
            handle.create_dataset("units/id",data=[10,11])
            handle.create_dataset("units/quality",data=["good","good"])
            handle.create_dataset("units/peak_channel_id",data=[100,999])
            channels=handle.create_group("general/extracellular_ephys/electrodes")
            channels.create_dataset("id",data=[100])
            channels.create_dataset("location",data=["VISp"])
            channels.create_dataset("probe_id",data=[1])
            channels.create_dataset("probe_vertical_position",data=[100.0])
            channels.create_dataset("probe_horizontal_position",data=[20.0])
            handle.create_dataset("units/presence_ratio",data=[0.99,0.2])
            handle.create_dataset("units/amplitude_cutoff",data=[0.01,0.01])
            handle.create_dataset("units/isi_violations",data=[0.01,0.01])
            handle.create_dataset("units/spike_times",data=[0.1,0.2,-999.0])
            handle["units/spike_times"].attrs["description"]="times (s) of detected spiking events"
            for path,unit in (("processing/running/running_speed","cm/s"),
                              ("processing/filtered_gaze_mapping/pupil_area","Pixels ^ 2")):
                signal=handle.create_dataset(path+"/data",data=[float("nan"),1.0]);signal.attrs["unit"]=unit
                handle.create_dataset(path+"/timestamps",data=[0.1,0.2])
            handle.create_dataset("processing/running/running_speed_end_times/timestamps",data=[0.15,0.25])
            handle.create_dataset("processing/running/running_speed_end_times/data",data=[float("nan")]*2)
            handle.create_dataset("units/spike_times_index",data=[2,3])
            fixture=session_fixture().presentations.iloc[:6]
            group=handle.create_group("intervals/natural_movie_fixture_presentations")
            group.attrs["colnames"]=["start_time","stop_time","frame","stimulus_block"]
            group.create_dataset("id",data=np.arange(6))
            for name in group.attrs["colnames"]:group.create_dataset(name,data=fixture[name].to_numpy())
            invalid=handle.create_group("intervals/invalid_times")
            invalid.attrs["colnames"]=["start_time","stop_time","tags"]
            invalid.create_dataset("id",data=[0])
            invalid.create_dataset("start_time",data=[1.15])
            invalid.create_dataset("stop_time",data=[1.17])
            invalid.create_dataset("tags",data=["EcephysSession","847657808","all_probes"])
            invalid.create_dataset("tags_index",data=[3])

    def test_handle_reads_ragged_tags_and_only_qc_spike_values(self):
        cfg=config();cfg["movie_frames_per_repeat"]["natural_movie_fixture"]=3
        row=cohort_session(build_cohort(metadata(),cfg),847657808)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fixture.nwb";self._write_fixture(path)
            with h5py.File(path,"r") as handle:
                data=read_nwb_handle(handle,847657808,source_etag="v1",etag_reader=lambda:"v1")
                self.assertEqual(data.invalid_times.iloc[0]["tags"],["EcephysSession","847657808","all_probes"])
                result=qualify_session(data,row,cfg)
                self.assertEqual(result["summary"]["raw_spike_count"],3)
                self.assertFalse(result["summary"]["source_full_asset_downloaded"])
                self.assertTrue(result["summary"]["required_columns_complete"])
                self.assertTrue(result["summary"]["source_version_rechecked"])
                self.assertEqual(set(result["spike_slice_evidence"]),{"10"})
                self.assertTrue(np.isnan(data.covariates["running_speed"].iloc[0]["value"]))
                self.assertEqual(data.covariates["running_speed"].iloc[0]["stop_time"],0.15)
                self.assertEqual(data.covariates["filtered_pupil_area"].attrs["unit"],"Pixels ^ 2")
                self.assertFalse(data.covariates["filtered_pupil_area"].attrs["causal_use_qualified"])
                self.assertNotIn("/processing/running/running_speed_end_times/data",result["column_evidence"])
                self.assertEqual(result["units"].iloc[1]["spike_time_validation"],"NOT_READ_QC_EXCLUDED")
                self.assertIn("/intervals/invalid_times/tags_index",result["column_evidence"])

    def test_retained_selected_recording_replays_spikes_state_and_left_join(self):
        cfg=config();cfg["movie_frames_per_repeat"]["natural_movie_fixture"]=3
        row=cohort_session(build_cohort(metadata(),cfg),847657808)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fixture.nwb";self._write_fixture(path)
            recording=Path(directory)/"selected_recording.h5"
            with h5py.File(path,"r") as handle:
                data=read_nwb_handle(handle,847657808,source_etag="v1",etag_reader=lambda:"v1",retain_spikes=True)
                result=qualify_session(data,row,cfg)
                self.assertEqual(len(result["units"]),2)
                self.assertEqual(result["units"].iloc[0]["recording_channel_location"],"VISp")
                self.assertTrue(pd.isna(result["units"].iloc[1]["recording_channel_location"]))
                receipt=write_selected_recording(recording,data,result,{"file_id":123,"etag":"v1"})
            self.assertTrue(receipt["recorded_columns_replayed"])
            with h5py.File(recording,"r") as saved:
                np.testing.assert_array_equal(saved["unit_ids"][:],[10])
                np.testing.assert_array_equal(saved["spike_times_s"][:],[0.1,0.2])
                np.testing.assert_array_equal(saved["spike_offsets"][:],[0,2])
                self.assertTrue(np.isnan(saved["running_speed/value"][0]))
                self.assertEqual(saved["electrodes/location"].asstr()[0],"VISp")
                self.assertNotIn("nwb_version",saved.attrs)
                self.assertFalse(saved.attrs["dataset_ready"])

    def test_local_qualify_cli_persists_partial_recording_and_unique_conditions(self):
        import contextlib,io
        from .__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);nwb=root/"fixture.nwb";self._write_fixture(nwb)
            cfg=config();cfg["movie_frames_per_repeat"]["natural_movie_fixture"]=3
            cfg["cohort"]["known_sources"][847657808]["expected_size_bytes"]=nwb.stat().st_size
            cfg_path=root/"config.yaml";cfg_path.write_text(yaml.safe_dump(cfg))
            cohort=root/"cohort.json";freeze_cohort(cohort,metadata(),cfg)
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                code=main(["--config",str(cfg_path),"qualify","--session-id","847657808",
                           "--cohort",str(cohort),"--nwb",str(nwb),"--output-dir",str(root/"output")])
            self.assertEqual(code,2)
            result=root/"output/session_847657808/source_qualification"
            manifest=json.loads((result/"manifest.json").read_text())
            self.assertEqual(manifest["selected_recording"]["path"],"selected_recording.h5")
            self.assertTrue((result/manifest["selected_recording"]["path"]).exists())
            conditions=pd.read_csv(result/"stimulus_conditions.csv")
            self.assertEqual(len(conditions),1)
            parameters=json.loads(conditions.iloc[0]["parameters_json"])
            self.assertNotIn("source_presentation_id",parameters)
            self.assertNotIn("frame",parameters)
            self.assertFalse(json.loads((result/"summary.json").read_text())["dataset_ready"])

    def test_changed_etag_or_wrong_session_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fixture.nwb";self._write_fixture(path)
            with h5py.File(path,"r") as handle:
                with self.assertRaisesRegex(ValueError,"ETAG"):
                    read_nwb_handle(handle,847657808,source_etag="v1",etag_reader=lambda:"v2")
                with self.assertRaisesRegex(ValueError,"SESSION_IDENTITY"):
                    read_nwb_handle(handle,123)

    def test_wrong_timestamp_unit_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fixture.nwb";self._write_fixture(path)
            with h5py.File(path,"r+") as handle:
                handle["processing/running/running_speed/timestamps"].attrs["unit"]="milliseconds"
            with h5py.File(path,"r") as handle:
                with self.assertRaisesRegex(ValueError,"TIME_UNIT_MUST_BE_SECONDS"):
                    read_nwb_handle(handle,847657808)

    def test_malformed_tag_vector_index_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fixture.nwb";self._write_fixture(path)
            with h5py.File(path,"r+") as handle:handle["intervals/invalid_times/tags_index"][0]=2
            with h5py.File(path,"r") as handle:
                with self.assertRaisesRegex(ValueError,"VECTOR_INDEX_INVALID"):
                    read_nwb_handle(handle,847657808)


    def test_diagnostic_selection_retains_denominators_without_reading_other_eligible_units(self):
        cfg=config();cfg["movie_frames_per_repeat"]["natural_movie_fixture"]=3
        row=cohort_session(build_cohort(metadata(),cfg),847657808)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fixture.nwb";self._write_fixture(path)
            with h5py.File(path,"r+") as handle:
                handle["units/presence_ratio"][1]=.99
                handle["units/spike_times"][2]=np.nan
            with h5py.File(path,"r") as handle:
                data=read_nwb_handle(handle,847657808,source_etag="v1",etag_reader=lambda:"v1",retain_spikes=True)
                events=[]
                result=qualify_session(data,row,cfg,diagnostic_unit_ids=[10],progress=events.append)
                self.assertEqual(set(data.spike_times.consumed_units),{"10"})
                self.assertEqual(result["summary"]["raw_unit_count"],2)
                self.assertEqual(result["summary"]["qc_eligible_unit_count"],2)
                self.assertEqual(result["summary"]["raw_spike_count"],3)
                self.assertEqual(result["summary"]["validated_spike_count"],2)
                self.assertEqual(result["summary"]["diagnostic_omitted_eligible_unit_count"],1)
                self.assertFalse(result["summary"]["required_columns_complete"])
                self.assertTrue(result["summary"]["source_metadata_columns_complete"])
                self.assertEqual(result["summary"]["status"],"PARTIAL")
                self.assertEqual(result["units"].iloc[1]["spike_time_validation"],"NOT_READ_DIAGNOSTIC_LIMIT")
                self.assertTrue(result["units"].iloc[1]["qc_eligible"])
                self.assertEqual(events[-1]["eligible_units_validated"],1)
                saved=Path(directory)/"selected.h5"
                write_selected_recording(saved,data,result,{"file_id":123,"etag":"v1"})
            with h5py.File(saved,"r") as handle:
                np.testing.assert_array_equal(handle["unit_ids"][:],[10])
                self.assertEqual(handle.attrs["qualification_scope"],"training_anchor_engineering_sample")
                self.assertFalse(handle.attrs["required_columns_complete"])
                self.assertEqual(handle.attrs["qc_eligible_unit_count"],2)
            with h5py.File(path,"r") as handle:
                data=read_nwb_handle(handle,847657808,source_etag="v1",etag_reader=lambda:"v1")
                with self.assertRaisesRegex(ValueError,"SPIKE_TIMES_INVALID:11"):
                    qualify_session(data,row,cfg)

    def test_invalid_diagnostic_selection_rejects_before_any_spike_read(self):
        cfg=config();cfg["movie_frames_per_repeat"]["natural_movie_fixture"]=3
        anchor=cohort_session(build_cohort(metadata(),cfg),847657808)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fixture.nwb";self._write_fixture(path)
            for ids,row in (([],anchor),([10,10],anchor),([True],anchor),([12],anchor),
                            ([10,11],anchor),([10],{**anchor,"split":"test"})):
                with self.subTest(ids=ids,split=row["split"]),h5py.File(path,"r") as handle:
                    data=read_nwb_handle(handle,847657808,source_etag="v1",etag_reader=lambda:"v1")
                    with self.assertRaisesRegex(ValueError,"DIAGNOSTIC"):
                        qualify_session(data,row,cfg,diagnostic_unit_ids=ids)
                    self.assertEqual(data.spike_times.consumed_units,{})

    def test_behavioral_nonzero_or_nonfinite_offset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fixture.nwb";self._write_fixture(path)
            for offset in (1.0,float("nan"),float("inf")):
                with h5py.File(path,"r+") as handle:
                    handle["processing/running/running_speed/data"].attrs["offset"]=offset
                with self.subTest(offset=offset),h5py.File(path,"r") as handle:
                    with self.assertRaisesRegex(ValueError,"STATE_SIGNAL_OFFSET_UNSUPPORTED"):
                        read_nwb_handle(handle,847657808)

    def test_diagnostic_cli_records_selection_and_cannot_claim_session_qualification(self):
        import contextlib,io
        from .__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);nwb=root/"fixture.nwb";self._write_fixture(nwb)
            cfg=config();cfg["movie_frames_per_repeat"]["natural_movie_fixture"]=3
            cfg["cohort"]["known_sources"][847657808]["expected_size_bytes"]=nwb.stat().st_size
            cfg_path=root/"config.yaml";cfg_path.write_text(yaml.safe_dump(cfg))
            cohort=root/"cohort.json";freeze_cohort(cohort,metadata(),cfg)
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                code=main(["--config",str(cfg_path),"qualify","--session-id","847657808",
                           "--cohort",str(cohort),"--nwb",str(nwb),"--output-dir",str(root/"output"),
                           "--diagnostic-unit-ids","10"])
            self.assertEqual(code,2)
            result=root/"output/session_847657808/source_qualification"
            manifest=json.loads((result/"manifest.json").read_text())
            self.assertEqual(manifest["diagnostic_unit_ids"],[10])
            self.assertEqual(manifest["qualification_scope"],"training_anchor_engineering_sample")
            self.assertEqual(manifest["status"],"PARTIAL")
            self.assertFalse(json.loads((result/"summary.json").read_text())["dataset_ready"])


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        path=self.root/"fixture.nwb"
        with h5py.File(path,"w") as handle:
            handle.attrs["nwb_version"]="2.6.0"
            handle.create_dataset("payload",data=np.arange(100))
        self.content=path.read_bytes()
        self.source={"file_id":123,"url":"https://example.invalid/nwb/123","expected_size_bytes":len(self.content)}
        self.network=config()["network"]

    def tearDown(self):
        self.temp.cleanup()

    def _session(self,status,headers,blocks,error=None):
        class Response:
            status_code=status
            def __init__(self):self.headers=headers
            def __enter__(self):return self
            def __exit__(self,*args):return None
            def raise_for_status(self):return None
            def iter_content(self,chunk_size):
                yield from blocks
                if error:raise error
        class Session:
            def __enter__(self):return self
            def __exit__(self,*args):return None
            def get(self,*args,**kwargs):return Response()
        return Session

    def test_sparse_remote_access_has_bounded_ranges_and_closes_handle(self):
        import io
        from unittest.mock import MagicMock
        from .download import open_remote_nwb
        metadata_fs=MagicMock()
        metadata_fs.info.return_value={"size":len(self.content),"ETag":'"v1"'}
        data_fs=MagicMock()
        stream=io.BytesIO(self.content)
        data_fs.open.return_value=stream
        events=[]
        with patch("fsspec.filesystem",side_effect=[metadata_fs,data_fs]) as factory:
            with open_remote_nwb(self.source["url"],len(self.content),"v1",self.network,progress=events.append) as (handle,identity,etag):
                self.assertEqual(handle.attrs["nwb_version"],"2.6.0")
                self.assertEqual(etag(),"v1")
                self.assertEqual(events[0]["event"],"REMOTE_IDENTITY_VERIFIED")
            self.assertFalse(handle.id.valid)
            self.assertTrue(stream.closed)
            options=data_fs.open.call_args.kwargs
            self.assertEqual(options["cache_type"],"blockcache")
            self.assertEqual(options["block_size"],self.network["range_block_size_bytes"])
            timeout=factory.call_args_list[1].kwargs["client_kwargs"]["timeout"]
            self.assertEqual(timeout.total,self.network["range_total_timeout_seconds"])
            self.assertGreater(timeout.total,0)

    def test_invalid_range_request_budget_fails_before_data_open(self):
        from unittest.mock import MagicMock
        from .download import open_remote_nwb
        metadata_fs=MagicMock()
        metadata_fs.info.return_value={"size":len(self.content),"ETag":'"v1"'}
        for key,value in (("range_total_timeout_seconds",float("inf")),("range_total_timeout_seconds",0),
                          ("range_block_size_bytes",True),("range_cache_blocks",-1)):
            network={**self.network,key:value}
            with self.subTest(key=key,value=value),patch("fsspec.filesystem",return_value=metadata_fs):
                with self.assertRaisesRegex(ValueError,"RANGE"):
                    with open_remote_nwb(self.source["url"],len(self.content),"v1",network):
                        self.fail("invalid request budget reached a source handle")
            metadata_fs.open.assert_not_called()

    def test_partial_bytes_never_qualify_and_resume_preserves_all_bytes(self):
        cache=self.root/"cache";half=len(self.content)//2
        with patch("Libs.population_dynamics.download.requests.Session",self._session(200,{"Content-Length":str(len(self.content))},[self.content[:half]],requests.ConnectionError("interrupted"))):
            with self.assertRaises(requests.ConnectionError):download_nwb(self.source,cache,self.network)
        self.assertFalse((cache/"allen_ecephys_123.nwb").exists())
        partial=cache/"allen_ecephys_123.nwb.part"
        with self.assertRaises(ValueError):verify_local_nwb(partial,len(self.content))
        with patch("Libs.population_dynamics.download.requests.Session",self._session(206,{"Content-Range":f"bytes {half}-{len(self.content)-1}/{len(self.content)}"},[self.content[half:]])):
            path,record=download_nwb(self.source,cache,self.network)
        self.assertEqual(path.read_bytes(),self.content)
        self.assertEqual(record["sha256"],hashlib.sha256(self.content).hexdigest())
        self.assertFalse(partial.exists())

    def test_resume_requires_correct_range_and_does_not_append_full_response(self):
        cache=self.root/"cache";cache.mkdir();partial=cache/"allen_ecephys_123.nwb.part"
        partial.write_bytes(self.content[:100])
        with patch("Libs.population_dynamics.download.requests.Session",self._session(200,{"Content-Length":str(len(self.content))},[self.content])):
            with self.assertRaisesRegex(ValueError,"RESUME_NOT_HONORED"):
                download_nwb(self.source,cache,self.network)
        self.assertEqual(partial.read_bytes(),self.content[:100])

    def test_remote_source_identity_requires_exact_size_and_etag(self):
        from unittest.mock import MagicMock
        session=MagicMock()
        session.info.return_value={"size":1234,"ETag":'"etag-v1"'}
        result=_remote_identity(session,"https://example.invalid/data",1234,"etag-v1",self.network)
        session.info.assert_called_once_with("https://example.invalid/data",timeout=self.network["metadata_total_timeout_seconds"])
        self.assertEqual(result["etag"],"etag-v1")
        self.assertFalse(result["source_full_asset_downloaded"])
        for size,etag in ((1235,"etag-v1"),(1234,"etag-v2")):
            with self.assertRaises(ValueError):
                _remote_identity(session,"https://example.invalid/data",size,etag,self.network)

    def test_remote_metadata_total_timeout_propagates(self):
        from unittest.mock import MagicMock
        filesystem=MagicMock()
        filesystem.info.side_effect=TimeoutError("metadata operation expired")
        with self.assertRaisesRegex(TimeoutError,"metadata operation expired"):
            _remote_identity(filesystem,"https://example.invalid/data",1234,"etag-v1",self.network)
        filesystem.info.assert_called_once_with("https://example.invalid/data",timeout=120)

    def test_invalid_metadata_deadline_rejects_before_network(self):
        from unittest.mock import MagicMock
        for value in (None,0,-1,float("inf"),float("nan"),True,"120"):
            filesystem=MagicMock()
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"METADATA_TOTAL_TIMEOUT"):
                _remote_identity(filesystem,"https://example.invalid/data",1234,"etag-v1",{**self.network,"metadata_total_timeout_seconds":value})
            filesystem.info.assert_not_called()

    def test_unknown_or_wrong_size_and_bad_header_reject(self):
        path=self.root/"fixture.nwb"
        for size in (None,len(self.content)+1):
            with self.assertRaises(ValueError):verify_local_nwb(path,size)
        path.write_bytes(b"x"*len(self.content))
        with self.assertRaisesRegex(ValueError,"SIGNATURE_MISMATCH"):
            verify_local_nwb(path,len(self.content))

    def test_source_change_cannot_reuse_partial(self):
        cache=self.root/"cache";half=len(self.content)//2
        with patch("Libs.population_dynamics.download.requests.Session",self._session(200,{"Content-Length":str(len(self.content))},[self.content[:half]],requests.ConnectionError("interrupted"))):
            with self.assertRaises(requests.ConnectionError):download_nwb(self.source,cache,self.network)
        changed={**self.source,"url":"https://example.invalid/different"}
        with self.assertRaises(FileExistsError):download_nwb(changed,cache,self.network)


if __name__ == "__main__":
    unittest.main()
