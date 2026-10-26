"""Contract checks; real source/PointNet/NEST execution remains the decisive probe."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from .provenance import sha256_file
from . import v1_pointnet_native as native
from .v1_nest_time_semantics import encode_precise_times
from .v1_pointnet_native import (bound_files, canonical_connections, collect_spikes, native_vector,
                                parameter_join, require_input_reference, write_diagnostic_spikes)


def test_delivery_rejects_changed_bytes_and_failed_status(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"value": 1}')
    delivery = dict(exit_code=0, files=[dict(path=path.name, bytes=path.stat().st_size, sha256=sha256_file(path))])
    (tmp_path / "delivery.json").write_text(json.dumps(delivery))
    bound_files(tmp_path, (path.name,))
    path.write_text('{"value": 2}')
    with pytest.raises(ValueError, match="DELIVERED_FILE_CHANGED"):
        bound_files(tmp_path, (path.name,))
    delivery["exit_code"] = 1
    (tmp_path / "delivery.json").write_text(json.dumps(delivery))
    with pytest.raises(ValueError, match="PASSED_DELIVERY"):
        bound_files(tmp_path, (path.name,))


def test_input_reference_rejects_different_recurrent_realization():
    with pytest.raises(ValueError, match="DIFFERENT_RECURRENT"):
        require_input_reference(dict(delivery_sha256="a"), dict(reference=dict(delivery_sha256="b")))


def test_realized_parameter_join_requires_specimen_and_source_template():
    row = dict(model_type="point_process", model_template="nest:glif_lif_asc_psc", dynamics_params="7_glif.json")
    reference = [dict(specimen_id=7, converted_template="nest:glif_psc")]
    assert set(parameter_join({7: row}, reference)) == {7}
    with pytest.raises(ValueError, match="REFERENCE_MISSING"):
        parameter_join({8: row}, reference)
    with pytest.raises(ValueError, match="TEMPLATE_DIFFERS"):
        parameter_join({7: dict(row, model_template="nest:glif_psc")}, reference)
    with pytest.raises(ValueError, match="UNIQUE_REFERENCE"):
        parameter_join({7: row}, reference + reference)


def test_native_batch_rejects_broadcastable_wrong_shape_and_nonfinite():
    assert np.array_equal(native_vector(0., 3, "weight"), [0., 0., 0.])
    with pytest.raises(ValueError, match="FIELD_INVALID"):
        native_vector(np.ones((3, 1)), 3, "weight")
    with pytest.raises(ValueError, match="FIELD_INVALID"):
        native_vector([1., np.nan], 2, "weight")
    with pytest.raises(ValueError, match="FIELD_INVALID"):
        native_vector([1.5], 1, "source", integer=True)


def test_connection_multiset_preserves_parallel_and_zero_weight_rows():
    values = dict(source=np.array([2, 1, 2]), target=np.array([1, 2, 1]),
                  receptor_type=np.array([1, 4, 1]), weight=np.array([0., -2., 3.]), delay=np.array([1., 1.75, 1.]))
    canonical = canonical_connections(values)
    reversed_values = {key: value[::-1] for key, value in values.items()}
    assert all(np.array_equal(value, canonical_connections(reversed_values)[key]) for key, value in canonical.items())
    assert len(canonical["source"]) == 3
    assert np.count_nonzero(canonical["weight"] == 0) == 1


@pytest.mark.parametrize('with_other_nodes', [False, True])
def test_native_complementary_count_matches_target_query_and_detects_extra_edges(tmp_path, monkeypatch, with_other_nodes):
    import nest

    nest.ResetKernel()
    try:
        cortex = nest.Create('iaf_psc_alpha', 4)
        ids = np.asarray(cortex.tolist(), dtype=np.int64)
        sources = np.array([ids[0], ids[0], ids[1]], dtype=np.int64)
        targets = np.array([ids[1], ids[1], ids[2]], dtype=np.int64)
        nest.Connect(sources, targets, 'one_to_one', dict(weight=np.array([0., 2., -1.])))
        if with_other_nodes:
            generators = nest.Create('spike_generator', 2)
            nest.Connect(generators[:1], cortex)
            # One generator is silent and unconnected. Other noncortical
            # neurons can receive connections and must remain in the query.
            other_neuron = nest.Create('iaf_psc_alpha')
            nest.Connect(cortex[:1], other_neuron)
            recorder = nest.Create('spike_recorder')
            nest.Connect(cortex, recorder)
            with pytest.raises(Exception) as rejected:
                nest.Connect(cortex[:1], generators[:1])
            assert type(rejected.value).__name__ == 'IllegalConnection'
        expected = 7 if with_other_nodes else 3
        count, _ = native.cortical_connection_count(nest, ids)
        assert count == len(nest.GetConnections(target=cortex)) == expected
        # Exercise the nonmaterializing production branch on this small graph.
        monkeypatch.setattr(native, 'MAX_MATERIALIZED_CONNECTION_ROWS', 0)
        evidence = native.ConnectionEvidence(nest, None,
            dict(v1=dict(edge_rows=expected, physical_synapses=expected)), tmp_path)
        evidence.rows['v1'] = expected
        assert evidence.verify(ids)['native_connections_to_cortex'] == expected
        if with_other_nodes:
            nest.Connect(cortex[:1], recorder)
            assert evidence.verify(ids)['native_connections_to_cortex'] == expected
        nest.Connect(cortex[:1], cortex[2:3])  # Deliberately bypass capture.
        count, _ = native.cortical_connection_count(nest, ids)
        assert count == len(nest.GetConnections(target=cortex)) == expected + 1
        with pytest.raises(ValueError, match='CORTICAL_TARGET_CONNECTION_COUNT_DIFFERS'):
            evidence.verify(ids)
        with pytest.raises(ValueError, match='COUNT_IDENTITIES_MISSING'):
            native.cortical_connection_count(nest, np.append(ids, 999999))
        with pytest.raises(ValueError, match='COUNT_IDENTITIES_INVALID'):
            native.cortical_connection_count(nest, np.append(ids, ids[0]))
    finally:
        nest.ResetKernel()


def test_native_complementary_count_rejects_mpi_scope_before_query():
    kernel = SimpleNamespace(NumProcesses=lambda: 2)
    with pytest.raises(ValueError, match='REQUIRES_SERIAL_KERNEL'):
        native.cortical_connection_count(kernel, np.array([1]))


@pytest.mark.parametrize('models, queried', [
    (['spike_generator', 'spike_recorder'], [4]),
    (['spike_generator', 'spike_generator'], []),
    (['copied_generator', 'iaf_psc_alpha'], [3, 4]),
])
def test_native_complementary_count_skips_only_pinned_generator_targets(models, queried):
    query = Mock(return_value=[None] * 2)
    kernel = SimpleNamespace(NumProcesses=lambda: 1,
        GetNodes=lambda: np.array([1, 2, 3, 4]),
        NodeCollection=lambda ids: ids, GetStatus=lambda nodes, key: models,
        GetKernelStatus=lambda key: 5, GetConnections=query)
    count, report = native.cortical_connection_count(kernel, np.array([1, 2]))
    assert count == (3 if queried else 5)
    assert report['noncortical_target_nodes'] == 2
    if queried:
        query.assert_called_once_with(target=queried)
    else:
        query.assert_not_called()


def test_native_complementary_count_rejects_incomplete_model_inventory():
    kernel = SimpleNamespace(NumProcesses=lambda: 1,
        GetNodes=lambda: np.array([1, 2]), NodeCollection=lambda ids: ids,
        GetStatus=lambda nodes, key: [], GetKernelStatus=lambda key: 1)
    with pytest.raises(ValueError, match='NONCORTICAL_MODEL_COVERAGE_DIFFERS'):
        native.cortical_connection_count(kernel, np.array([1]))


def test_native_fine_clock_keeps_requested_tuples_and_matches_source_clock_digest(tmp_path):
    import nest

    def capture(resolution, reference_digest=None, changed_weight=False):
        nest.ResetKernel()
        nest.SetKernelStatus(dict(resolution=resolution, local_num_threads=1))
        cortex = nest.Create('glif_psc', 2)
        ids = np.asarray(cortex.tolist(), dtype=np.int64)
        source, target = ids[[0, 0, 1]], ids[[1, 1, 0]]
        syn = dict(synapse_model='static_synapse', receptor_type=np.array([1, 1, 1]),
                   weight=np.array([2. if changed_weight else 1., 0., -2.]), delay=np.array([1.35, 1.75, 2.]))
        saved = {key: value.copy() for key, value in syn.items() if isinstance(value, np.ndarray)}
        calls = []

        def connect(actual_source, actual_target, conn_spec, syn_spec):
            calls.append((actual_source, actual_target, syn_spec))
            return nest.Connect(actual_source, actual_target, conn_spec, syn_spec)

        pool = SimpleNamespace(_gid2pop_id={int(identity): 0 for identity in ids},
                               get_node_ids=lambda values: (values, np.full(len(values), 'v1')))
        graph = SimpleNamespace(gid_map=pool, _nest_connect=connect)
        output = tmp_path/('fine_changed' if changed_weight else str(resolution))
        (output/'native').mkdir(parents=True)
        evidence = native.ConnectionEvidence(nest, graph, dict(v1=dict(edge_rows=3, physical_synapses=3)), output,
            resolution_ms=resolution, reference_source_batch_sha256=reference_digest)
        with evidence.capture():
            graph._nest_connect(source, target, syn_spec=syn)
        assert graph._nest_connect is connect
        assert calls[0][0] is source and calls[0][1] is target and calls[0][2] is syn
        for key, value in saved.items():
            np.testing.assert_array_equal(syn[key], value)
        return evidence.verify(ids)

    try:
        reference = capture(.25)
        fine = capture(.125, reference['source_batch_sha256'])
        assert fine['reference_clock_source_batch_exact'] is True
        assert fine['reference_clock_source_batch_sha256'] == reference['source_batch_sha256']
        assert fine['source_batch_sha256'] != reference['source_batch_sha256']
        assert fine['exact_native_tuples_verified'] == 3
        assert fine['zero_weight_rows_by_population'] == {'v1': 1}
        with pytest.raises(ValueError, match='REFERENCE_CLOCK_SOURCE_BATCH_DIFFERS'):
            capture(.125, reference['source_batch_sha256'], changed_weight=True)
    finally:
        nest.ResetKernel()


def test_diagnostic_files_keep_populations_separate_and_all_units(tmp_path):
    for population, ids in (("lgn", np.array([0, 2, 9])), ("bkg", np.array([0]))):
        path = tmp_path / (population + ".h5")
        write_diagnostic_spikes(path, population, ids)
        with h5py.File(path) as handle:
            assert list(handle["spikes"]) == [population]
            group = handle["spikes"][population]
            assert np.array_equal(group["node_ids"][:], np.repeat(ids, 2))
            assert np.array_equal(group["timestamps"][:], np.tile([.125, .5], len(ids)))
            assert group["timestamps"].attrs["units"] == "ms"
    with pytest.raises(ValueError, match="SOURCE_IDENTITIES_INVALID"):
        write_diagnostic_spikes(tmp_path / "duplicate.h5", "lgn", np.array([1, 1]))


def test_cortical_output_preserves_silent_units_and_rejects_unknown_ids(tmp_path):
    path = tmp_path / "spikes.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("spikes/v1")
        group.create_dataset("node_ids", data=[9, 2, 9])
        group.create_dataset("timestamps", data=[1., .5, 2.]).attrs["units"] = "ms"
    events, counts = collect_spikes(path, np.array([2, 7, 9]), np.array([11, 12, 13]))
    assert np.array_equal(counts, [1, 0, 2])
    assert np.array_equal(events["native_id"], [11, 13, 13])
    with pytest.raises(ValueError):
        collect_spikes(path, np.array([2, 7]), np.array([11, 12]))



def _input_case():
    data = dict(lgn=dict(sorted_ids=np.array([2, 7, 9], dtype=np.uint64)),
                bkg=dict(sorted_ids=np.array([0], dtype=np.int64)))
    trains = [np.array([.5, .125]), np.empty(0), np.array([.052912, .25, .9]), np.array([.75])]
    plan = dict(population=np.array([0, 0, 0, 1]), source_id=np.array([9, 2, 7, 0]),
                source_counts=np.array([2, 0, 3, 1]), trains=trains)
    return data, plan


def test_external_plan_joins_source_ids_and_retains_silent_units():
    data, plan = _input_case()
    trains = native.input_trains_by_population(plan, data)
    assert [len(row) for row in trains['lgn']] == [0, 3, 2]
    assert np.array_equal(trains['lgn'][2], [.125, .5])
    assert np.array_equal(trains['bkg'][0], [.75])
    assert np.array_equal(plan['trains'][0], [.5, .125])


def test_external_plan_rejects_same_count_with_another_source_identity():
    data, plan = _input_case(); data['lgn']['sorted_ids'] = np.array([2, 7, 10])
    with pytest.raises(ValueError, match='EXTERNAL_INPUT_SOURCE_IDS_DIFFER'):
        native.input_trains_by_population(plan, data)


def _point_config(tmp_path):
    source = tmp_path/'source';source.mkdir()
    value = dict(run=dict(duration=3000., dt=.25), target_simulator='NEST', manifest={},
        inputs={pop: dict(node_set=pop, input_type='spikes', module='csv', input_file=pop+'.csv') for pop in ['lgn','bkg']})
    path = source/'source_asset.txt';path.write_text(json.dumps(value))
    (source/'source.json').write_text(json.dumps(dict(artifact=path.name, bytes=path.stat().st_size, sha256=sha256_file(path))))
    bodies = {'weight_semantics/legacy_'+name: json.dumps(dict(receptor_type=i+1)).encode()
              for i,name in enumerate(['e2e','e2i','i2e','i2i'])}
    return source, bodies


@pytest.mark.parametrize('external', [False, True])
def test_point_config_routes_two_population_files_and_preserves_default(tmp_path, external):
    source, bodies = _point_config(tmp_path);data,_ = _input_case()
    files = None
    if external:
        files = {name: tmp_path/(name+'.h5') for name in ('lgn','bkg')}
        for name,path in files.items():
            write_diagnostic_spikes(path, name, data[name]['sorted_ids'])
    output = tmp_path/'output';output.mkdir()
    config_path,binding = native.stage_point_config(source, output, data, bodies, spike_files=files)
    config = json.loads(config_path.read_text())
    for name,value in config['inputs'].items():
        path = Path(value['input_file'])
        assert value['node_set'] == name and value['module'] == 'sonata'
        assert path.name == name+('_spikes.h5' if external else '_diagnostic_spikes.h5')
        with h5py.File(path) as handle:
            assert list(handle['spikes']) == [name]
        if external:
            assert sha256_file(path) == sha256_file(files[name])
    assert config['run']['duration'] == 3000. and config['run']['dt'] == .25
    assert binding['source_clock'] == dict(duration_ms=3000., resolution_ms=.25)
    assert binding['diagnostic_replacement'] is (not external)
    assert ('pulse_times_ms' in binding) is (not external)


def test_fine_execution_config_keeps_author_clock_and_input_times(tmp_path):
    source, bodies = _point_config(tmp_path)
    data, _ = _input_case()
    original = (source/'source_asset.txt').read_bytes()
    output = tmp_path/'output'; output.mkdir()
    config_path, binding = native.stage_point_config(source, output, data, bodies, resolution_ms=.125)
    assert json.loads(config_path.read_text())['run'] == dict(duration=3000., dt=.125, n_thread=1, precise_times=True)
    assert binding['source_clock'] == dict(duration_ms=3000., resolution_ms=.25)
    assert (output/'source_point_config.json').read_bytes() == original
    assert binding['pulse_times_ms'] == [.125, .5]
    with h5py.File(output/'inputs/lgn_diagnostic_spikes.h5') as handle:
        assert handle['spikes/lgn/timestamps'][:].tolist() == [.125, .5] * 3


def _clock_protocol():
    return dict(native.CLOCK_PROTOCOL_FIXED, reference_canary_archive='canary', reference_full_archive='full')


@pytest.mark.parametrize('key,value', [
    ('execution_resolution_ms', .0625), ('reference_resolution_ms', .125), ('duration_ms', 100.),
    ('maximum_new_full_conditions', 2), ('independent_replay_required', False),
    ('allow_further_resolution_sweep', True), ('production_eligible', True), ('scientific_qualification', True),
    ('schema_version', True), ('training_seed', 42)])
def test_clock_protocol_rejects_unapproved_conditions(tmp_path, key, value):
    protocol = _clock_protocol(); protocol[key] = value
    path = tmp_path/'clock.yaml'; path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='NOT_AUTHORIZED'):
        native.load_clock_protocol(path)


def test_clock_protocol_load_and_native_clock_readback_are_explicit(tmp_path):
    path = tmp_path/'clock.yaml'; path.write_text(json.dumps(_clock_protocol()))
    assert native.load_clock_protocol(path) == _clock_protocol()
    values = dict(resolution=.125, tics_per_ms=1000.)
    kernel = SimpleNamespace(GetKernelStatus=lambda key: values[key])
    assert native.execution_clock(kernel, .125) == dict(duration_ms=3000., resolution_ms=.125,
                                                       tics_per_ms=1000., tics_per_step=125)
    with pytest.raises(ValueError, match='EXECUTION_CLOCK_DIFFERS'):
        native.execution_clock(kernel, .25)
    values['tics_per_ms'] = 2000.
    with pytest.raises(ValueError, match='EXECUTION_CLOCK_DIFFERS'):
        native.execution_clock(kernel, .125)


def test_refractory_metadata_keeps_physical_parameters_separate_from_derived_counts():
    references = {7: dict(converted_parameters=dict(t_ref=1.55))}
    result = native.refractory_clock_semantics(references, dict(tics_per_ms=1000., resolution_ms=.125))
    assert result['private_native_state_instrumented'] is False
    assert result['models']['7'] == dict(t_ref_ms=1.55, reference_steps=7, execution_steps=13,
                                        reference_duration_ms=1.75, execution_duration_ms=1.625)
    assert references[7]['converted_parameters'] == dict(t_ref=1.55)


class _NativeFixturePool:
    def __init__(self, plan, native_ids):
        self.forward = {(('lgn','bkg')[int(pop)], int(source_id)): int(nid)
                        for pop,source_id,nid in zip(plan['population'], plan['source_id'], native_ids)}
        self.reverse = {nid: key for key,nid in self.forward.items()}

    def get_nestids(self, name, source_ids):
        return np.array([self.forward[name,int(identity)] for identity in source_ids], dtype=np.int64)

    def get_node_ids(self, native_ids):
        values = [self.reverse[int(identity)] for identity in native_ids]
        return np.array([item[1] for item in values]), np.array([item[0] for item in values])


@pytest.mark.parametrize('resolution_ms', [.25, .125])
@pytest.mark.parametrize('kind', ['diagnostic', 'variable', 'silent', 'near_grid'])
def test_real_native_generator_readback_preserves_requested_schema(tmp_path, kind, resolution_ms):
    import nest

    data,plan = _input_case()
    external = kind != 'diagnostic'
    if kind == 'silent':
        plan['trains'] = [np.empty(0) for _ in plan['trains']]
        plan['source_counts'] = np.zeros(4, dtype=np.int64)
    if kind == 'near_grid':
        plan['trains'] = [np.array([1.1, 1.1]), np.empty(0),
                          np.array([np.nextafter(resolution_ms, 0), resolution_ms,
                                    np.nextafter(resolution_ms, np.inf), resolution_ms]), np.array([.75])]
        plan['source_counts'] = np.array([2, 0, 4, 1])
    nest.ResetKernel()
    nest.SetKernelStatus(dict(resolution=resolution_ms, local_num_threads=1))
    try:
        generators = nest.Create('spike_generator', 4, params=dict(precise_times=True))
        native_ids = np.asarray(generators.tolist(), dtype=np.int64)
        for identity,train in zip(native_ids, plan['trains']):
            values = np.sort(train) if external else np.asarray(native.PULSES_MS)
            nest.SetStatus(nest.NodeCollection([int(identity)]), dict(spike_times=values))
        pool = _NativeFixturePool(plan, native_ids)
        sim = SimpleNamespace(net=SimpleNamespace(_virtual_gids=pool))
        evidence = SimpleNamespace(verify=lambda _: dict(native_connections_to_cortex=0, scope='four-generator unit fixture'))
        (tmp_path/'native').mkdir(); report = {}
        native.InputReadback(nest, data, evidence, np.empty(0, dtype=np.int64), tmp_path, report,
                             input_plan=plan if external else None, resolution_ms=resolution_ms).initialize(sim)
        assert report['execution_clock']['resolution_ms'] == resolution_ms
        assert report['execution_clock']['tics_per_step'] == int(1000 * resolution_ms)
        with np.load(tmp_path/'native/input_events.npz', allow_pickle=False) as result:
            assert result['source_id'].tolist() == [2,7,9,0]
            assert result['native_id'].tolist() == [int(native_ids[i]) for i in [1,2,0,3]]
            assert result['precise_times'].all()
            if external:
                assert report['input_event_schema'] == native.VISUAL_SCHEMA
                assert result['source_id'].dtype == np.dtype('int64')
                counts = [0,0,0,0] if kind == 'silent' else [0,4,2,1] if kind == 'near_grid' else [0,3,2,1]
                assert result['source_event_counts'].tolist() == counts
                assert result['source_event_unit'].tolist() == np.repeat(np.arange(4),counts).tolist()
                requested = result['requested_times_ms']
                encoded = encode_precise_times(requested, report['input_readback']['tics_per_ms'], resolution_ms)
                assert requested.ndim == 1
                assert np.array_equal(result['native_times_ms'], encoded['decoded_times_ms'])
                assert report['input_readback']['silent_generators'] == (4 if kind == 'silent' else 1)
            else:
                assert report['input_event_schema'] == native.DIAGNOSTIC_SCHEMA
                assert result['requested_times_ms'].shape == (4,2)
                assert np.array_equal(result['native_times_ms'], np.tile(native.PULSES_MS,(4,1)))
                assert 'source_event_unit' not in result
                assert report['input_readback']['all_generator_pulses_exact']
    finally:
        nest.ResetKernel()


def _comparison_fixture(path, pid):
    (path/'native').mkdir(parents=True)
    np.savez(path/'native/input_events.npz', source_id=np.array([2]), source_event_counts=np.array([0]))
    (path/'native/parameters.json').write_text('{}')
    report = dict(status=native.VISUAL_PASS_STATUS, exit_code=0, hostname='unit-fixture', process_id=pid,
        bindings=dict(spike_inputs=dict(input_identity_sha256='events-a')), original_config_sha256='config',
        implementation_sha256='code', mode='canary', source_denominators={}, nest_version='3.10.0',
        kernel_rng_seed=1, local_num_threads=1, stimulus_kind='unit-fixture', input_readback={},connections={},
        cortical_units=0, realized_glif_types=0, array_artifacts=['native/input_events.npz'],simulated_ms=0.,
        wheel=dict(sha256='wheel'), input_mode=native.VISUAL_MODE,input_event_schema=native.VISUAL_SCHEMA)
    (path/'report.json').write_text(json.dumps(report))
    return report


def _clock_reference_fixture(path, *, bad_comparison=False, wrong_job=False):
    for name, pid in [('primary', 1), ('replay', 2)]:
        report = _comparison_fixture(path/name, pid)
        report.update(status=native.PASS_STATUS, input_mode=native.DIAGNOSTIC_MODE,
            input_event_schema=native.DIAGNOSTIC_SCHEMA, simulated_ms=3000., native_nest_simulation=True,
            native_parameters_and_initial_state_exact=True, nest_version=native.NEST_VERSION,
            runtime=dict(slurm=dict(SLURM_JOB_ID='592538')),
            input_readback=dict(tics_per_ms=1000.), connections=dict(source_batch_sha256='a'*64),
            bindings=dict(point_config=dict(source_clock=dict(duration_ms=3000., resolution_ms=.25))))
        (path/name/'report.json').write_text(json.dumps(report))
    comparison = native.compare(path/'primary', path/'replay', path/'comparison')
    if bad_comparison:
        comparison['left_report_sha256'] = 'b'*64
        (path/'comparison/report.json').write_text(json.dumps(comparison))
    files = [dict(path=name, bytes=(path/name).stat().st_size, sha256=sha256_file(path/name))
             for name in ('primary/report.json', 'replay/report.json', 'comparison/report.json')]
    (path/'delivery.json').write_text(json.dumps(dict(status='PASS_POINTNET_NATIVE_DIAGNOSTIC', exit_code=0,
        job_id='other' if wrong_job else '592538', mode='canary', files=files)))
    current = json.loads((path/'primary/report.json').read_text())
    return dict(_clock_protocol(), reference_canary_archive=str(path)), current


def test_temporal_reference_consumes_bound_replay_and_source_identity(tmp_path):
    protocol, current = _clock_reference_fixture(tmp_path/'reference')
    reference, binding = native.bind_clock_reference(protocol, current)
    assert reference == current
    assert binding['reference_primary_report_sha256'] == sha256_file(tmp_path/'reference/primary/report.json')
    assert binding['reference_delivery_sha256'] == sha256_file(tmp_path/'reference/delivery.json')
    assert binding['reference_clock_source_batch_sha256'] == 'a'*64
    assert binding['unchanged_upstream_bindings'] is True


@pytest.mark.parametrize('changed_binding', ['recurrent', 'spike_inputs', 'parameters', 'sources'])
def test_temporal_reference_rejects_changed_upstream_identity(tmp_path, changed_binding):
    protocol, current = _clock_reference_fixture(tmp_path/'reference')
    current['bindings'][changed_binding] = dict(sha256='different-source-with-same-size')
    with pytest.raises(ValueError, match='TEMPORAL_SOURCE_IDENTITY_DIFFERS'):
        native.bind_clock_reference(protocol, current)


@pytest.mark.parametrize('defect', ['comparison', 'job', 'changed_report'])
def test_temporal_reference_rejects_wrong_receipt_or_archive(tmp_path, defect):
    protocol, current = _clock_reference_fixture(tmp_path/'reference',
        bad_comparison=defect == 'comparison', wrong_job=defect == 'job')
    if defect == 'changed_report':
        with (tmp_path/'reference/primary/report.json').open('a') as handle:
            handle.write(' ')
    with pytest.raises(ValueError, match='REFERENCE_REPLAY_REQUIRED|SOURCE_CLOCK_REFERENCE_REQUIRED|DELIVERED_FILE_CHANGED'):
        native.bind_clock_reference(protocol, current)


@pytest.mark.parametrize('difference', ['bundle', 'mode', 'clock'])
def test_native_comparison_rejects_different_input_context_even_with_equal_arrays(tmp_path, difference):
    left,right = tmp_path/'left',tmp_path/'right'
    _comparison_fixture(left, 1); report = _comparison_fixture(right, 2)
    if difference == 'bundle':
        report['bindings']['spike_inputs']['input_identity_sha256'] = 'events-b'
    elif difference == 'mode':
        report.update(input_mode=native.DIAGNOSTIC_MODE, input_event_schema=native.DIAGNOSTIC_SCHEMA, status=native.PASS_STATUS)
    else:
        report['execution_clock'] = dict(duration_ms=3000., resolution_ms=.125, tics_per_ms=1000., tics_per_step=125)
    (right/'report.json').write_text(json.dumps(report))
    result = native.compare(left, right, tmp_path/'comparison')
    assert result['exit_code'] == 1
    assert all(row['equal'] for row in result['files'])
    assert {'bundle': 'bindings', 'mode': 'input_mode', 'clock': 'execution_clock'}[difference] in result['binding_or_runtime_differences']


@pytest.mark.parametrize('bad_clock', [False, True])
def test_temporal_status_requires_same_fine_clock_replay(tmp_path, bad_clock):
    left, right = tmp_path/'left', tmp_path/'right'
    for path, pid in [(left, 1), (right, 2)]:
        report = _comparison_fixture(path, pid)
        report.update(status=native.TEMPORAL_PASS_STATUS,
            clock_protocol=dict(sha256='protocol', resolved=_clock_protocol()),
            execution_clock=dict(duration_ms=3000., resolution_ms=.125, tics_per_ms=1000., tics_per_step=125))
        if bad_clock and path == right:
            report['execution_clock'].update(resolution_ms=.25, tics_per_step=250)
        (path/'report.json').write_text(json.dumps(report))
    if bad_clock:
        with pytest.raises(ValueError, match='TEMPORAL_REPLAY_CLOCK_DIFFERS'):
            native.compare(left, right, tmp_path/'comparison')
    else:
        assert native.compare(left, right, tmp_path/'comparison')['exit_code'] == 0


@pytest.mark.parametrize('external', [False, True])
@pytest.mark.parametrize('fine_clock', [False, True])
def test_probe_cli_preserves_default_and_passes_optional_input_archive(monkeypatch, external, fine_clock):
    call = Mock(return_value=dict(status=native.PASS_STATUS, exit_code=0));monkeypatch.setattr(native,'probe',call)
    args = ['probe']
    for name in ['config','recurrent','inputs','parameters','source-config','output']:
        args += ['--'+name, name]
    if external:args += ['--spike-inputs','visual_archive']
    if fine_clock:args += ['--clock-protocol', 'clock.yaml']
    assert native.main(args) == 0
    assert call.call_args.kwargs['spike_inputs_archive'] == (Path('visual_archive') if external else None)
    assert call.call_args.kwargs['clock_protocol'] == (Path('clock.yaml') if fine_clock else None)


def test_benchmark_clock_canary_reuses_only_full_reference_location():
    first = dict(_clock_protocol(), scope=native.BENCHMARK_CLOCK_SCOPE)
    second = dict(first, reference_full_archive="another-complete-condition")
    native.validate_clock_protocol(first)
    native.require_clock_canary_protocol(second, first)
    for key, value in (("reference_canary_archive", "other-canary"),
                       ("execution_resolution_ms", .25),
                       ("independent_replay_required", False),
                       ("maximum_new_full_conditions", 2),
                       ("scientific_qualification", True)):
        with pytest.raises(ValueError, match="NOT_AUTHORIZED|CANARY_PROTOCOL_DIFFERS"):
            native.require_clock_canary_protocol(dict(second, **{key: value}), first)


def test_legacy_clock_canary_keeps_single_reference_and_scope():
    first = _clock_protocol()
    with pytest.raises(ValueError, match="CANARY_PROTOCOL_DIFFERS"):
        native.require_clock_canary_protocol(dict(first, reference_full_archive="other"), first)
    with pytest.raises(ValueError, match="CANARY_PROTOCOL_DIFFERS"):
        native.require_clock_canary_protocol(dict(first, scope=native.BENCHMARK_CLOCK_SCOPE), first)
