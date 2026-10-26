"""Keep the full candidate denominator and reject changed saved controls."""
from __future__ import annotations

from copy import deepcopy
import unittest

from .native_pulse_programs import native_candidate_protocols

SEEDS = (41, 42, 43)
NAMES = ("omit", "first_future_pulse", "original_future_pulses")
INITIAL = [10.0, 999.5, 1000.0, 1000.1, 1100.0, 1200.0, 1510.0]
COMMITTED = [10.0, 999.5, 1000.0]
FUTURES = ([], [1000.1], [1000.1, 1100.0, 1200.0])
GRAPH_DIGEST = "a" * 64


def inputs():
    case = dict(case_id="train_block_014_familiar_0", split="train", block=14,
                template_id="familiar_0", settings=dict(pulse_control=dict(initial_emission_ms=list(INITIAL))))
    graph = dict(graph_manifest_sha256=GRAPH_DIGEST,
                 resolved_source_network=dict(thalamic_input=True, thalamic_input_type="pulses",
                                              th_pulse_start=3.0, th_interval=7.0,
                                              th_delay_pulse_generator=1.0))
    identity = dict(gpu_source_revision="74fc021cad2b4bd5c011c390b3f95a302b663483",
                    native_library_sha256="b" * 64, native_poisson_rng_family="per_edge_philox4_32_10",
                    graph_manifest_sha256=GRAPH_DIGEST, dt_ms=0.1)
    rows = []
    for name, future in zip(NAMES, FUTURES):
        effective = COMMITTED + future
        record = dict(schema="mesocircuit_future_pulse_protocol_v1", time_units="ms",
                      time_origin="native biological time zero",
                      control_semantics="generator emissions; fixed native spatial mask, weight and delay",
                      resolution_ms=0.1, generator_delay_ms=1.0, decision_time_ms=1000.0,
                      initial_schedule_overridden=True, initial_emission_ms=list(INITIAL),
                      requested_future_emission_ms=list(future), committed_emission_ms=list(COMMITTED),
                      inflight_tc_arrival_ms=[1000.5, 1001.0], effective_emission_ms=effective,
                      effective_tc_arrival_ms=[value + 1.0 for value in effective],
                      observed_history_interval="[0, decision_time_ms)",
                      full_microscopic_state_qualified=False, paired_stochastic_future_noise_verified=False,
                      inverse_intervention_qualified=False)
        rows.append(dict(candidate=name, protocol=record, objective=-1.0))
    base = dict(status="DEVELOPMENT_NATIVE_PULSE_SEARCH_EXECUTED",
                configuration=dict(case_id=case["case_id"], training_seeds=list(SEEDS),
                                   decision_time_ms=1000.0, horizon_ms=500.0,
                                   native_intervention_execution=False, repeat_training=False,
                                   scientific_success_tolerance=None),
                source_unit_count=100000, horizon_bins=100, training_performed=False,
                mechanistic_simulation_performed=False, inverse_intervention_qualified=False,
                scientific_claim_qualified=False, success_tolerance_qualified=False,
                native_execution_performed=False, native_future_noise_pairing_verified=False,
                all_candidate_outcomes_retained=True, native_binding=dict(native_source_identity=identity,
                episode_report_sha256="c" * 64), graph_manifest_sha256=GRAPH_DIGEST,
                field_names=[f"field_{index}" for index in range(128)], candidates=rows)
    reports = {}
    for index, seed in enumerate(SEEDS):
        reports[seed] = deepcopy(base)
        reports[seed].update(training_seed=seed, selected_candidate=NAMES[index])
    return reports, case, graph


class NativePulseProgramTests(unittest.TestCase):
    def test_all_candidates_survive_different_seed_choices_and_score_signs(self):
        reports, case, graph = inputs()
        reports[41]["candidates"][0]["objective"] = -100.0
        reports[42]["candidates"][1]["objective"] = 100.0
        programs, selected = native_candidate_protocols(reports, case, graph)
        self.assertEqual(tuple(programs), NAMES)
        self.assertEqual(selected, dict(zip(SEEDS, NAMES)))
        self.assertEqual([programs[name]["future_emission_ms"] for name in NAMES], list(FUTURES))
        for name in NAMES:
            self.assertEqual(programs[name]["initial_emission_ms"], INITIAL)
            self.assertEqual(programs[name]["decision_time_ms"], 1000.0)
        # The source pulse after the scored horizon is not offered as a future
        # action. Committed emissions remain in every original initial program.
        self.assertNotIn(1510.0, programs["original_future_pulses"]["future_emission_ms"])
        programs["omit"]["initial_emission_ms"].clear()
        self.assertEqual(case["settings"]["pulse_control"]["initial_emission_ms"], INITIAL)
        self.assertEqual(programs["first_future_pulse"]["initial_emission_ms"], INITIAL)

    def test_seed_denominator_and_selected_candidate_identity_are_required(self):
        for alteration in ("missing", "extra", "string_seed", "wrong_saved_seed", "unknown_choice"):
            with self.subTest(alteration=alteration):
                reports, case, graph = inputs()
                if alteration == "missing":
                    del reports[43]
                elif alteration == "extra":
                    reports[44] = deepcopy(reports[43])
                elif alteration == "string_seed":
                    reports["41"] = reports.pop(41)
                elif alteration == "wrong_saved_seed":
                    reports[42]["training_seed"] = 41
                else:
                    reports[43]["selected_candidate"] = "favorable_subset"
                with self.assertRaises(ValueError):
                    native_candidate_protocols(reports, case, graph)

    def test_saved_future_committed_and_inflight_controls_must_match_source(self):
        for field in ("initial_emission_ms", "requested_future_emission_ms", "committed_emission_ms",
                      "inflight_tc_arrival_ms", "effective_emission_ms", "effective_tc_arrival_ms"):
            with self.subTest(field=field):
                reports, case, graph = inputs()
                reports[42]["candidates"][1]["protocol"][field] = []
                with self.assertRaises(ValueError):
                    native_candidate_protocols(reports, case, graph)

    def test_changed_case_graph_clock_episode_or_field_order_is_rejected(self):
        for alteration in ("case", "graph", "clock", "episode", "fields", "boundary", "horizon"):
            with self.subTest(alteration=alteration):
                reports, case, graph = inputs()
                if alteration == "case":
                    case["block"] = 15
                elif alteration == "graph":
                    graph["graph_manifest_sha256"] = "d" * 64
                elif alteration == "clock":
                    reports[43]["native_binding"]["native_source_identity"]["dt_ms"] = 0.2
                elif alteration == "episode":
                    reports[43]["native_binding"]["episode_report_sha256"] = "d" * 64
                elif alteration == "fields":
                    reports[43]["field_names"] = reports[43]["field_names"][::-1]
                elif alteration == "boundary":
                    reports[41]["configuration"]["decision_time_ms"] = 1100.0
                else:
                    reports[41]["configuration"]["horizon_ms"] = 600.0
                with self.assertRaises(ValueError):
                    native_candidate_protocols(reports, case, graph)

    def test_missing_candidate_and_unsupported_claim_declarations_are_rejected(self):
        alterations = ("missing_candidate", "duplicate_candidate", "qualification", "simulated",
                       "training", "outcomes", "missing_false_flag", "prepared")
        for alteration in alterations:
            with self.subTest(alteration=alteration):
                reports, case, graph = inputs()
                if alteration == "missing_candidate":
                    reports[41]["candidates"].pop()
                elif alteration == "duplicate_candidate":
                    reports[41]["candidates"][2] = deepcopy(reports[41]["candidates"][1])
                elif alteration == "qualification":
                    reports[41]["scientific_claim_qualified"] = True
                elif alteration == "simulated":
                    reports[41]["mechanistic_simulation_performed"] = True
                elif alteration == "training":
                    reports[41]["training_performed"] = True
                elif alteration == "outcomes":
                    reports[41]["all_candidate_outcomes_retained"] = False
                elif alteration == "missing_false_flag":
                    del reports[41]["inverse_intervention_qualified"]
                else:
                    reports[41]["status"] = "PREPARED_NATIVE_PULSE_SEARCH_DEVELOPMENT_INPUTS"
                with self.assertRaises(ValueError):
                    native_candidate_protocols(reports, case, graph)


if __name__ == "__main__":
    unittest.main()
