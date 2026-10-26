"""Recompute a suppressed network reference for observed multispike histories.

Only the focal unit's output spikes after its retained first spike are removed.
The complete network and native draws continue; deleting an archived event is
not sufficient. This executable validates a previously saved full-source witness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from . import pfc_deterministic_replay as source
from . import pfc_native_drive as native
from . import pfc_first_passage_reference as reference
from . import pfc_first_passage_witness as witness
from .pfc_first_passage_witness import FocalOverrideDrive, create_bound_drive, bound_prefix_state
from .pfc_first_passage_probability import first_passage_ticks
from .pfc_coupled_drive import create_coupled_drive
from .provenance import sha256_file


def require(condition, message):
    if not condition:
        raise ValueError("PFC_FIRST_PASSAGE_SUPPRESSION_"+message)


def replay_reference(config, case, initial, original_tape, selection, *, counts_E=None, counts_I=None, drive=None):
    """Return full altered outputs, first-spike initial state, and recurrent forcing.

    One replay per focal window is necessary when its actual second spike may
    have changed the later network. Other focal windows cannot be suppressed
    together because they define different conditional reference networks.
    """
    reference._check_tape(config, original_tape)
    u, tau, stop = selection.unit_index, selection.first_spike_tick, selection.window_stop_tick
    require(0 <= u < original_tape.n_e and selection.window_start_tick <= tau < stop <= original_tape.total_ticks,
            "FOCAL_WINDOW_IDENTITY")
    old = original_tape.events[original_tape.events[:, 0] < stop]
    old_focal = old[(old[:, 1] == u) & (old[:, 0] >= selection.window_start_tick), 0]
    require(len(old_focal) > 0 and old_focal[0] == tau, "OBSERVED_FIRST_SPIKE")
    require((drive is not None and counts_E is None and counts_I is None)
            or (drive is None and isinstance(counts_E,np.ndarray) and isinstance(counts_I,np.ndarray)),
            "ONE_COMPLETE_INPUT_PROVIDER_REQUIRED")
    prefix = dict(config, duration_ms=stop*config["dt_ms"])
    args = dict(drive=drive) if drive is not None else dict(counts_E=counts_E[:stop], counts_I=counts_I[:stop])
    actual = source.simulate(prefix, case, initial, **args,
        focal_spike_suppression=source.FocalSpikeSuppression(u, tau, stop))
    tape = reference.validate_tape(prefix, actual["raw"])
    focal = tape.events[(tape.events[:, 1] == u) & (tape.events[:, 0] >= selection.window_start_tick), 0]
    require(np.array_equal(focal, [tau]), "FOCAL_REFERENCE_NOT_SUPPRESSED")
    first_hit = int(old_focal[1]) if len(old_focal) > 1 else stop
    require(np.array_equal(old[old[:, 0] < first_hit], tape.events[tape.events[:, 0] < first_hit]),
            "PREHIT_NETWORK_CHANGED")
    require(np.array_equal(old[(old[:, 0] == first_hit) & (old[:, 1] != u)],
                           tape.events[(tape.events[:, 0] == first_hit) & (tape.events[:, 1] != u)]),
            "HIT_TICK_NONFOCAL_EVENTS_CHANGED")
    removed = actual["suppressed_reference"]["threshold_candidate_ticks"]
    require((len(old_focal) == 1 and len(removed) == 0)
            or (len(old_focal) > 1 and len(removed) > 0 and removed[0] == first_hit),
            "FIRST_SUPPRESSED_CROSSING")
    diagnostics = dict(observed_focal_window_ticks=old_focal.tolist(), retained_focal_window_ticks=focal.tolist(),
        first_suppressed_crossing_tick=None if len(old_focal) == 1 else first_hit,
        prehit_network_events_equal=True, hit_tick_nonfocal_events_equal=True,
        original_window_multispike=len(old_focal) > 1, network_feedback_recomputed=True,
        posthit_network_equality_required=False, source_law_trajectory=False)
    return actual, diagnostics


def replay_history_window(config, case, initial, original_tape, row, *, drive=None, counts_E=None, counts_I=None):
    """Adapter for the existing history runner's per-window replay interface."""
    require(row["category"] == "suppressed_network_reference_required" and row["observed_count"] >= 2,
            "MULTISPIKE_HISTORY_REQUIRED")
    selection = reference.FirstSpikeWindow(row["unit_index"],row["first_spike_tick"],
        row["window_start_tick"],row["window_stop_tick"],row["refractory_ticks"])
    return replay_reference(config,case,initial,original_tape,selection,
        drive=drive,counts_E=counts_E,counts_I=counts_I)


def verify_feedback_reconstruction(config, case, actual, unsuppressed_terminal, selection,
                                   *, drive=None, counts_E=None, counts_I=None):
    """Rebuild all synaptic states from altered events and test recurrent effect."""
    result = actual["suppressed_reference"]
    forcing = result["forcing"]
    captured = 0

    def compare(stage, tick, state):
        nonlocal captured
        if stage != "pre_rk2" or not selection.first_spike_tick < tick < selection.window_stop_tick:
            return
        require(int(forcing["tick"][captured]) == tick
                and forcing["time_seconds"][captured] == tick * (config["dt_ms"] * .001),
                "REBUILT_FORCING_CLOCK")
        for key in ("s_GABA", "s_NMDA_total", "I_stim", "s_NMDA", "x"):
            require(forcing[key][captured] == state["E__"+key][selection.unit_index],
                    "REBUILT_FORCING_CHANGED:"+key)
        captured += 1

    rebuilt = reference.reconstruct(config, case, reference.validate_tape(config, actual["raw"]),
        stop_tick=selection.window_stop_tick, selection=selection, drive=drive,
        counts_E=counts_E, counts_I=counts_I, on_state_view=compare,
        capture_ticks=range(selection.first_spike_tick+1, selection.window_stop_tick))
    require(captured == len(forcing["tick"]), "REBUILT_FORCING_INCOMPLETE")
    require(rebuilt["conditional_initial"] == result["conditional_initial"], "REBUILT_CONDITIONAL_INITIAL")
    fields = []
    for key, value in rebuilt["terminal_state"].items():
        require(np.array_equal(value, actual["terminal_state"][key]), "REBUILT_TERMINAL_CHANGED:"+key)
        fields.append(key)
    key = "I__s_NMDA_total"
    before = unsuppressed_terminal[key]
    after = actual["terminal_state"][key]
    require(before.shape == after.shape and np.isfinite(before).all() and np.isfinite(after).all()
            and not np.array_equal(before, after), "NO_OBSERVABLE_RECURRENT_FEEDBACK_EFFECT")
    return dict(all_terminal_synaptic_fields_equal=fields, forcing_fields_equal=list(forcing),
                forcing_ticks=captured, nonfocal_recurrent_field=key,
                nonfocal_recurrent_values_changed=int(np.count_nonzero(before != after)),
                maximum_absolute_recurrent_change=float(np.max(np.abs(before-after))),
                membrane_reintegrated=False, source_law_trajectory=False)


def validate_fine_histories(settings, inputs, output, *, case_names=None):
    """Validate the two declared saved histories; do not search or replace them."""
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_PFC_FINE_CLOCK_SUPPRESSION", cases=[],
                  new_independent_source_blocks=0, new_full_network_episodes=0,
                  scientific_fidelity_qualified=False, production_episodes=0)
    try:
        require([row["dt_ms"] for row in settings["cases"]] == [.025, .0125], "BOTH_FINE_CLOCKS_REQUIRED")
        registered = [row["name"] for row in settings["cases"]]
        selected = registered if case_names is None else list(case_names)
        require(selected and len(set(selected)) == len(selected)
                and all(name in registered for name in selected), "REGISTERED_FINE_CASE_SELECTION")
        report.update(registered_case_names=registered, selected_case_names=selected,
                      unexecuted_case_names=[name for name in registered if name not in selected])
        for row in settings["cases"]:
            if row["name"] not in selected:
                continue
            root = inputs/row["name"]
            frame = json.loads((root/"frame_row.json").read_text())
            require(frame["sample_index"] == row["sample_index"]
                    and frame["category"] == "conditional_integration_required"
                    and frame["observed_count"] == 1 and frame["proposal_hits"] > 0,
                    "REGISTERED_SAVED_SINGLE_HISTORY")
            with np.load(root/"first_hit_witnesses.npz", allow_pickle=False) as data:
                require(len(data["counts"]) > 0, "SAVED_FIRST_HIT_REQUIRED")
                hit = int(data["first_hit_tick"][0]); sample = int(data["sample_index"][0])
            source_config = yaml.safe_load((root/"source/config.yaml").read_text())
            require(source_config["dt_ms"] == row["dt_ms"], "DECLARED_FINE_CLOCK")
            case_settings = dict(case_id=row["case_id"], source_rng_seed=row["source_rng_seed"],
                expected_units=settings["expected_units"], dt_ms=row["dt_ms"],
                source_duration_ms=source_config["duration_ms"], witness_row=0,
                expected_sample_index=sample, expected_hit_tick=hit,
                input_stream="coupled_binomial_clock_refinement_stream",
                verify_feedback_reconstruction=True,
                witness_source=dict(job_id="same_execution_saved_history"),
                **{key:frame[key] for key in ("unit_index", "first_spike_tick", "window_start_tick", "window_stop_tick")})
            destination = output/row["name"]; destination.mkdir()
            saved = witness.validate(case_settings, root/"source", root/"reference",
                                     root/"first_hit_witnesses.npz", destination/"witness")
            accepted = validate(case_settings, root/"source", root/"reference",
                root/"first_hit_witnesses.npz", destination/"witness", destination/"suppression")
            report["cases"].append(dict(name=row["name"], dt_ms=row["dt_ms"],
                source_rng_seed=row["source_rng_seed"], saved_history_sample_index=row["sample_index"],
                saved_first_hit_tick=hit, witness_status=saved["status"], suppression_status=accepted["status"],
                complete_prefix_rng_equal=accepted["native_rng_equal"],
                feedback_reconstruction=accepted["feedback_reconstruction"]))
        report["status"] = ("COMPLETE_PFC_FINE_CLOCK_SUPPRESSION" if len(selected) == len(registered)
                            else "PARTIAL_PFC_FINE_CLOCK_SUPPRESSION")
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc)); raise
    finally:
        (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def validate(settings, source_dir, reference_dir, witnesses_path, witness_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_PFC_FIRST_PASSAGE_SUPPRESSION", new_full_network_episodes=0,
        new_independent_source_blocks=0, production_episodes=0, confirmation_execution_released=False,
        scientific_fidelity_qualified=False, population_history_uncertainty_established=False)
    started = time.perf_counter()
    try:
        config = yaml.safe_load((source_dir/"config.yaml").read_text())
        sr = json.loads((source_dir/"report.json").read_text()); case = sr["selected_case"]
        require(config == sr["resolved_config"] and case["source_rng_seed"] == settings["source_rng_seed"]
                and case["case_id"] == settings["case_id"], "ORIGINAL_SOURCE_IDENTITY")
        wr = json.loads((witness_dir/"report.json").read_text())
        require(wr["status"] == "COMPLETE_PFC_FIRST_PASSAGE_WITNESS" and wr["prefix_rng_equal"], "ACCEPTED_WITNESS")
        expected = json.loads((reference_dir/"conditional_initial.json").read_text())
        u, tau, stop = (expected[k] for k in ("unit_index", "first_spike_tick", "window_stop_tick"))
        require((u,tau,stop) == tuple(settings[k] for k in ("unit_index","first_spike_tick","window_stop_tick"))
                and config["excitatory_neurons"]+config["inhibitory_neurons"] == settings["expected_units"]
                and config["dt_ms"] == settings["dt_ms"], "REFERENCE_IDENTITY_AND_SCALE")
        with np.load(witnesses_path, allow_pickle=False) as data:
            counts = data["counts"][settings["witness_row"]].copy()
            hit = int(data["first_hit_tick"][settings["witness_row"]])
        require(hit == settings["expected_hit_tick"], "SAVED_WITNESS_HIT")
        prefix = dict(config, duration_ms=stop*config["dt_ms"])
        with np.load(witness_dir/"source_raw.npz", allow_pickle=False) as data:
            original = reference.validate_tape(prefix, {k:data[k] for k in data.files})
        initial, native_drive, kind = create_bound_drive(config, case, sr, settings)
        require(np.array_equal(initial,np.load(source_dir/"initial_voltage.npy",allow_pickle=False)), "SOURCE_INITIAL_STATE")
        override = FocalOverrideDrive(native_drive,unit_index=u,start_tick=tau,stop_tick=stop-1,counts=counts)
        selection = reference.FirstSpikeWindow(u,tau,settings["window_start_tick"],stop,
                                               round(.002/(config["dt_ms"]*.001)))
        actual, checks = replay_reference(prefix,case,initial,original,selection,drive=override)
        # Preserve the new full-network counterfactual before numerical acceptance.
        np.savez_compressed(output/"source_raw.npz",**actual["raw"])
        np.savez_compressed(output/"source_terminal.npz",**actual["terminal_state"])
        result = actual["suppressed_reference"]
        (output/"conditional_initial.json").write_text(json.dumps(result["conditional_initial"],indent=2)+"\n")
        np.savez_compressed(output/"reference_window.npz",**result["forcing"])
        np.savez_compressed(output/"suppressed_threshold_candidates.npz",ticks=result["threshold_candidate_ticks"])
        state = bound_prefix_state(native_drive, kind)
        np.savez_compressed(output/"native_prefix_rng.npz",**state)
        report.update(event_boundary=checks,source_cost=actual["cost"],source_units=len(initial),
            replacement=override.evidence(),replayed_prefix_ticks=stop,
            suppressed_threshold_candidates=len(result["threshold_candidate_ticks"]))
        require(checks["original_window_multispike"] and checks["first_suppressed_crossing_tick"] == hit,
                "ACTUAL_MULTISPIKE_REQUIRED")
        require(result["conditional_initial"] == expected, "CONDITIONAL_INITIAL_CHANGED")
        comparisons={}
        with np.load(reference_dir/"reference_window.npz",allow_pickle=False) as data:
            require(set(data.files) == set(result["forcing"]), "FORCING_FIELDS")
            for key in data.files:
                equal = np.array_equal(data[key], result["forcing"][key]);comparisons[key]=dict(equal=bool(equal),values=int(data[key].size))
                require(equal,"FULL_SUPPRESSED_FORCING_CHANGED:"+key)
        predicted = first_passage_ticks(config,result["conditional_initial"],result["forcing"],counts[None,:])
        require(int(predicted[0]) == hit,"FULL_REFERENCE_KERNEL_FIRST_HIT")
        with np.load(witness_dir/"native_prefix_rng.npz",allow_pickle=False) as data:
            require(set(data.files)==set(state) and all(np.array_equal(data[k],state[k]) for k in data.files),
                    "COMPLETE_NATIVE_RNG_CHANGED")
        if settings.get("verify_feedback_reconstruction", False):
            _, check_drive, check_kind = create_bound_drive(config, case, sr, settings)
            require(check_kind == kind, "RECONSTRUCTED_INPUT_KIND")
            check_override = FocalOverrideDrive(check_drive, unit_index=u, start_tick=tau,
                                               stop_tick=stop-1, counts=counts)
            with np.load(witness_dir/"source_terminal.npz", allow_pickle=False) as data:
                unsuppressed = {key: data[key] for key in data.files}
            report["feedback_reconstruction"] = verify_feedback_reconstruction(
                config, case, actual, unsuppressed, selection, drive=check_override)
            rebuilt_state = bound_prefix_state(check_drive, kind)
            require(set(rebuilt_state) == set(state)
                    and all(np.array_equal(value, rebuilt_state[key]) for key, value in state.items()),
                    "RECONSTRUCTED_PREFIX_RNG_CHANGED")
        report.update(status="COMPLETE_PFC_FIRST_PASSAGE_SUPPRESSION",input_stream=kind,original_witness_job=settings["witness_source"]["job_id"],
            conditional_initial_equal=True,full_forcing_comparisons=comparisons,reference_kernel_first_hit=int(predicted[0]),
            native_rng_equal=True,actual_suppressed_prefix_replays=1,
            interpretation="Observed multispike witness now has a recomputed full-network suppressed reference; no new population probability or confirmation qualification.")
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        report["wall_seconds"] = time.perf_counter()-started
        (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report



def continue_feedback(settings, archive, output, *, case_name="half_A"):
    """Advance a saved paired terminal state beyond its original window.

    The original suppression result and probability window retain their status.
    This diagnostic tests the delayed recurrent effect. It does not generate an
    independent episode or qualify the first-passage probability estimator.
    """
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_PFC_FEEDBACK_CONTINUATION", case_name=case_name,
                  original_probability_gate_passed=False, production_episodes=0,
                  new_full_network_episodes=0, new_independent_source_blocks=0,
                  scientific_fidelity_qualified=False, source_law_trajectory=False,
                  input_bindings=[])
    started = time.perf_counter()
    try:
        rows = [row for row in settings["cases"] if row["name"] == case_name]
        require(len(rows) == 1, "REGISTERED_FEEDBACK_CASE")
        row = rows[0]
        delivery = json.loads((archive/"delivery.json").read_text())
        files = {item["path"]: item for item in delivery["files"]}

        def read_bound(name):
            path = archive/name
            item = files[name]
            require(path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"],
                    "SAVED_FEEDBACK_FILE_CHANGED:"+name)
            report["input_bindings"].append(dict(source=str(path), **item))
            return path

        parent_name = "report.json" if "report.json" in files else "execution.json"
        parent = json.loads(read_bound(parent_name).read_text())
        report["original_execution_status"] = parent["status"]
        # The original generic runner records source paths. The declared-case
        # runner binds the same source through the frozen settings and delivery.
        if "inputs" in parent:
            parent_inputs = {item["path"]: item for item in parent["inputs"]}
            source_inputs = {name: parent_inputs["source_inputs/"+case_name+"/source/"+name]
                             for name in ("config.yaml", "report.json", "initial_voltage.npy")}
        else:
            require(parent.get("selected_case") == case_name and parent.get("exit_code") == 1,
                    "REGISTERED_CASE_EXECUTION")
            origin = Path(row["source"]["archive"])
            receipt_path = origin/"delivery.json"
            require(sha256_file(receipt_path) == row["source"]["delivery_sha256"],
                    "FEEDBACK_SOURCE_DELIVERY_CHANGED")
            receipt = json.loads(receipt_path.read_text())
            entries = {item["path"]: item for item in receipt["files"]}
            source_inputs = {}
            for name in ("config.yaml", "report.json", "initial_voltage.npy"):
                relative = str(Path(row["source"]["case_path"])/name)
                source_inputs[name] = dict(source=str(origin/relative), **entries[relative])
        def original_source(name):
            item = source_inputs[name]
            path = Path(item["source"])
            require(sha256_file(path) == item["sha256"], "FEEDBACK_SOURCE_CHANGED:"+name)
            report["input_bindings"].append(dict(path=str(path),sha256=item["sha256"]))
            return path

        config = yaml.safe_load(original_source("config.yaml").read_text())
        origin = json.loads(original_source("report.json").read_text())
        case = origin["selected_case"]
        require(config == origin["resolved_config"] and config["dt_ms"] == row["dt_ms"]
                and case["case_id"] == row["case_id"] and case["source_rng_seed"] == row["source_rng_seed"],
                "FEEDBACK_SOURCE_IDENTITY")
        initial = np.load(original_source("initial_voltage.npy"),allow_pickle=False)
        require(len(initial) == settings["expected_units"], "FEEDBACK_COMPLETE_UNIT_COUNT")
        base = "validation/"+case_name+"/"
        witness_report = json.loads(read_bound(base+"witness/report.json").read_text())
        suppressed_report = json.loads(read_bound(base+"suppression/report.json").read_text())
        require(witness_report["status"] == "COMPLETE_PFC_FIRST_PASSAGE_WITNESS"
                and witness_report["prefix_rng_equal"], "ACCEPTED_FEEDBACK_WITNESS")
        require(suppressed_report.get("error",{}).get("message") ==
                "PFC_FIRST_PASSAGE_SUPPRESSION_NO_OBSERVABLE_RECURRENT_FEEDBACK_EFFECT",
                "REGISTERED_FEEDBACK_FAILURE")
        start = witness_report["prefix_ticks"]
        u = witness_report["unit_index"]
        hit = suppressed_report["event_boundary"]["first_suppressed_crossing_tick"]
        require(suppressed_report["replayed_prefix_ticks"] == start
                and hit in (start-2, start-1), "END_WINDOW_FEEDBACK_CASE")
        # Recurrent projection precedes synaptic RK2; output x increments follow
        # RK2. An output at hit can first affect the nonfocal projection at hit+2.
        first_projection = hit+2
        report.update(unit_count=len(initial),unit_index=u,dt_ms=config["dt_ms"],
                      start_tick=start,stop_tick_exclusive=start+2,
                      first_hit_tick=hit,first_nonfocal_projection_tick=first_projection,
                      original_probability_window=dict(start_tick=witness_report["window_start_tick"],
                          stop_tick_exclusive=start),original_suppression_status=suppressed_report["status"])

        def arrays(name):
            with np.load(read_bound(name),allow_pickle=False) as data:
                return {key:data[key].copy() for key in data.files}

        branches = {}
        for name in ("witness","suppression"):
            branch_dir = output/name
            branch_dir.mkdir()
            branches[name] = dict(terminal=arrays(base+name+"/source_terminal.npz"),
                raw=arrays(base+name+"/source_raw.npz"),rng=arrays(base+name+"/native_prefix_rng.npz"))
        left,right = branches["witness"]["rng"],branches["suppression"]["rng"]
        require(left.keys() == right.keys() and all(np.array_equal(left[k],right[k]) for k in left),
                "PAIRED_PREFIX_RNG_CHANGED")
        inputs = witness_report["inputs_used"]
        report["transitions"] = []
        for name,branch in branches.items():
            generated,drive = create_coupled_drive(config,case,
                witness_report["replacement"]["original_native_metadata"]["coarse_dt_seconds"]*1000,
                row["source"]["refinement_seeds"])
            require(np.array_equal(generated,initial), "FEEDBACK_INITIAL_DRAW_IDENTITY")
            previous_inputs = dict(inputs, input_mode="native_numpy_stream")
            for tick in range(start,start+2):
                actual = source.simulate(config,case,initial,drive=drive,
                    terminal_state=branch["terminal"],start_tick=tick,stop_tick=tick+1,
                    prefix_raw=branch["raw"],prefix_inputs_used=previous_inputs,
                    drive_state=branch["rng"] if tick == start else None)
                branch["terminal"],branch["raw"] = actual["terminal_state"],actual["raw"]
                previous_inputs = actual["inputs_used"]
                destination=output/name/("tick_"+str(tick+1))
                destination.mkdir()
                np.savez_compressed(destination/"source_terminal.npz",**branch["terminal"])
                np.savez_compressed(destination/"source_raw.npz",**branch["raw"])
                branch["rng"] = source.snapshot_drive_state(drive)
                np.savez_compressed(destination/"native_prefix_rng.npz",**branch["rng"])
                (destination/"transition.json").write_text(json.dumps(dict(
                    start_tick=tick,stop_tick_exclusive=tick+1,cost=actual["cost"],
                    inputs_used=previous_inputs),indent=2)+"\n")
            branch["inputs"] = previous_inputs
        require(branches["witness"]["inputs"] == branches["suppression"]["inputs"],
                "PAIRED_CONTINUATION_INPUT_TOTALS")
        for tick in range(start,start+2):
            states={}
            rng={}
            for name in branches:
                destination=output/name/("tick_"+str(tick+1))
                with np.load(destination/"source_terminal.npz",allow_pickle=False) as data:
                    states[name]={key:data[key] for key in data.files}
                with np.load(destination/"native_prefix_rng.npz",allow_pickle=False) as data:
                    rng[name]={key:data[key] for key in data.files}
            require(rng["witness"].keys() == rng["suppression"].keys()
                    and all(np.array_equal(rng["witness"][k],rng["suppression"][k]) for k in rng["witness"]),
                    "PAIRED_CONTINUATION_RNG")
            a,b=states["witness"]["I__s_NMDA_total"],states["suppression"]["I__s_NMDA_total"]
            changed=int(np.count_nonzero(a!=b))
            require(np.isfinite(a).all() and np.isfinite(b).all(), "FEEDBACK_FINITE")
            require(changed == (0 if tick < first_projection else config["inhibitory_neurons"]),
                    "FEEDBACK_SOURCE_DELAY_CHANGED")
            report["transitions"].append(dict(tick=tick,nonfocal_nmda_values_changed=changed,
                maximum_absolute_nmda_change=float(np.max(np.abs(a-b))),paired_native_rng_equal=True))
        report.update(status="COMPLETE_PFC_DELAYED_FEEDBACK_DIAGNOSTIC",
                      original_prefix_simulations=0,additional_ticks_per_branch=2,
                      original_probability_window_changed=False,
                      interpretation="Delayed nonfocal feedback in the saved counterfactual pair; original window gate remains failed.")
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__,message=str(exc))
        raise
    finally:
        report["wall_seconds"] = time.perf_counter()-started
        (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("config","source","reference","witnesses","witness-source","inputs","output"):
        parser.add_argument("--"+key,type=Path,required=key in ("config","output"))
    parser.add_argument("--case-name", action="append", choices=("half_A", "quarter_A"))
    parser.add_argument("--continue-feedback-from",type=Path)
    args=parser.parse_args();settings=yaml.safe_load(args.config.read_text())
    if args.continue_feedback_from is not None:
        require(settings["schema"] == "pfc_fine_clock_suppression_v1", "FEEDBACK_FINE_CONFIG")
        require(args.case_name is not None and len(args.case_name) == 1, "ONE_FEEDBACK_CASE")
        result=continue_feedback(settings,args.continue_feedback_from,args.output,case_name=args.case_name[0])
        print(json.dumps(dict(status=result["status"])))
        return
    if settings["schema"] == "pfc_fine_clock_suppression_v1":
        require(args.inputs is not None, "FINE_HISTORY_INPUTS_REQUIRED")
        result=validate_fine_histories(settings,args.inputs,args.output,case_names=args.case_name)
    else:
        require(args.case_name is None, "CASE_SELECTION_REQUIRES_FINE_HISTORY")
        require(settings["schema"]=="pfc_first_passage_suppression_v1","CONFIG_SCHEMA")
        require(all(value is not None for value in
                    (args.source,args.reference,args.witnesses,args.witness_source)), "VALIDATION_INPUTS_REQUIRED")
        result=validate(settings,args.source,args.reference,args.witnesses,args.witness_source,args.output)
    print(json.dumps(dict(status=result["status"])))


if __name__=="__main__":main()
