import copy
import json
import shutil
from dataclasses import replace
import h5py
import numpy as np
import pytest

from .circuit_data import SCHEMA, CircuitArchiveWriter, CircuitDataset, CircuitEpisode
from .v1_source_dataset import exact_ticks, metadata_units, prepare_spikes, input_case_metadata, verify_export
from . import v1_source_dataset as source_dataset
from .provenance import sha256_file
from .v1_lgn_spikes import input_plan_identity
from .v1_nest_time_semantics import encode_precise_times
from .v1_visual_controls import CONTROL_FEATURES, make_commands


def metadata(tmp_path):
    path=tmp_path/'nodes.h5'
    with h5py.File(path,'w') as f:
        g=f.create_group('nodes/v1')
        g['node_id']=[2,0,1];g['node_type_id']=[101,101,202]
        g['node_group_id']=[0,0,0];g['node_group_index']=[1,2,0]
        data=g.create_group('0')
        for i,axis in enumerate(('x','y','z')):data[axis]=np.array([20.,30.,10.])+i
    types=tmp_path/'types.csv'
    types.write_text('node_type_id pop_name population location model_type\n101 E v1 Cortex point_process\n202 I v1 Cortex point_process\n')
    native={'source_id':np.arange(3),'node_type_id':np.array([101,202,101]),'native_id':np.array([99,5,20])}
    return path,types,native


def units(tmp_path):
    paths=metadata(tmp_path)
    result,_=metadata_units(*paths,'fixture-source',{'connectivity_identity':'graph','parameter_identity':'params'},'modeled point positions')
    return result


def test_source_and_node_group_permutations_preserve_coordinates_and_types(tmp_path):
    result=units(tmp_path)
    np.testing.assert_array_equal(result.unit_ids,[0,1,2])
    np.testing.assert_array_equal(result.coordinates,[[10.,11.,12.],[20.,21.,22.],[30.,31.,32.]])
    np.testing.assert_array_equal(result.node_type_ids,[101,202,101])
    assert result.populations.tolist()==['E','I','E']
    assert result.coordinate_units=='um'


@pytest.mark.parametrize('change',['group_index','type_identity','coordinate_units'])
def test_inconsistent_source_metadata_is_rejected(tmp_path,change):
    nodes,types,native=metadata(tmp_path)
    if change=='type_identity':native['node_type_id'][0]=202
    else:
        with h5py.File(nodes,'r+') as f:
            if change=='group_index':f['nodes/v1/node_group_index'][:]=[0,0,1]
            else:f['nodes/v1/0/x'].attrs['units']='mm'
    with pytest.raises(ValueError):
        metadata_units(nodes,types,native,'fixture-source',{'connectivity_identity':'graph','parameter_identity':'params'},'model positions')


@pytest.mark.parametrize('times',[[.1],[-.25],[3.25],[np.nan],[np.inf]])
def test_noninteger_or_out_of_window_native_time_is_not_rounded(times):
    with pytest.raises(ValueError):exact_ticks(np.asarray(times),.25,12)


def test_endpoint_duplicates_silent_units_and_no_controls_survive_shared_archive(tmp_path):
    unit=units(tmp_path)
    activity={'spike_time_ms':np.array([3.,2.75,.25,2.75]),'spike_source_id':np.array([0,1,0,1]),
              'spike_counts':np.array([2,2,0])}
    ticks,indices=prepare_spikes(activity,unit,12,.25)
    assert ticks.tolist()==[1,11,11,12] and indices.tolist()==[0,1,1,0]
    record={'complete_unit_recording':True,'simulation_completed':True,'recorded_until_tick':12,
            'expected_duration_ticks':12,'core_unit_count':3,'source_revision':'fixture-source',
            'circuit_provenance':unit.circuit_provenance}
    episode=CircuitEpisode('source_episode',unit.circuit_id,'source_block','development',12,.25,ticks,indices,
                           np.empty(0,dtype=np.int64),np.empty((0,0)),(),source_record=record,
                           privileged={'final_endpoint_unit_indices':indices[ticks==12]})
    root=tmp_path/'dataset';binding={'schema':SCHEMA,'scope':'source_reproduction','scenario_id':'v1_visual',
                                  'source_revision':'fixture-source','sample_unit':'complete_mechanistic_episode'}
    with CircuitArchiveWriter(root,binding) as writer:writer.register_circuit(unit);writer.add(episode)
    dataset=CircuitDataset(root,splits=('development',));batch=dataset.window('source_episode',boundary_tick=6,history_ticks=6,horizon_ticks=6,bin_ticks=1)
    counts=np.concatenate((batch['history'],batch['targets']))
    assert counts.sum()==3 and counts[11,1]==2 and not counts[:,2].any()
    assert batch['known_controls'].shape==(12,0) and 'privileged' not in batch
    row=dataset.catalog['shards'][0]
    with h5py.File(root/row['path'],'r') as f:
        assert f['source_episode/spike_ticks'][-1]==12
        np.testing.assert_array_equal(f['source_episode/privileged/final_endpoint_unit_indices'][:],[0])


def test_count_mismatch_cannot_hide_a_missing_event(tmp_path):
    unit=units(tmp_path)
    bad={'spike_time_ms':np.array([.25]),'spike_source_id':np.array([0]),'spike_counts':np.array([2,0,0])}
    with pytest.raises(ValueError,match='COUNT_READBACK'):
        prepare_spikes(bad,unit,12,.25)


def test_generated_input_case_keeps_noise_identity_and_stimulus_provenance():
    native = dict(kernel_rng_seed=143)
    old_case, old_block, old_seeds, old_record = input_case_metadata(native, 'circuit-A')
    assert (old_case, old_block) == ('v1_original_input_reproduction', 'v1_original_input_block')
    assert old_seeds == {'native_kernel_rng_seed': 143} and old_record == {}
    native['generated_lgn'] = dict(spike_report_sha256='a'*64, input_identity_sha256='e'*64, trial=0, original_plan_sha256='b'*64,
        input_origin_sha256='c'*64, config=dict(spike_generation=dict(starting_noise_seed=12345, trials=1),
        stimulus=dict(contrast_percent=80.0)))
    first_case, first_block, seeds, record = input_case_metadata(native, 'circuit-A')
    assert first_case != old_case and first_block != old_block
    assert seeds == {'native_kernel_rng_seed': 143, 'lgn_starting_noise_seed': 12345}
    assert record['config']['stimulus']['contrast_percent'] == 80.0
    assert record['admissible_control_mapping_qualified'] is False
    native['generated_lgn']['spike_report_sha256'] = 'd'*64
    second_case, second_block, _, _ = input_case_metadata(native, 'circuit-A')
    # Export timestamps/provenance do not create another scientific episode.
    assert first_case == second_case and first_block == second_block
    native['generated_lgn']['input_identity_sha256'] = 'f'*64
    third_case, third_block, _, _ = input_case_metadata(native, 'circuit-A')
    assert third_case != first_case and third_block == first_block
    fourth_case, _, _, _ = input_case_metadata(native, 'circuit-B')
    assert fourth_case != third_case


def test_declared_visual_commands_pass_v1_archive_verifier_at_source_onset(tmp_path):
    unit = units(tmp_path)
    commands = make_commands(dict(duration_s=3.0, gray_screen_s=0.5,
        spatial_frequency_cycles_per_degree=0.04, temporal_frequency_hz=2.0,
        direction_deg=270.0, contrast_percent=80.0))
    record = dict(complete_unit_recording=True, simulation_completed=True,
        recorded_until_tick=12000, expected_duration_ticks=12000, core_unit_count=3,
        source_revision='fixture-source', circuit_provenance=unit.circuit_provenance)
    episode = CircuitEpisode('visual_fixture', unit.circuit_id, 'source_block', 'development',
        12000, 0.25, np.array([2003, 2004, 12000]), np.array([0, 1, 2]),
        commands['control_ticks'], commands['control_values'], CONTROL_FEATURES,
        source_record=record, privileged={'incoming_source_ids': np.array([5, 6])})
    root = tmp_path/'visual_dataset'
    binding = dict(schema=SCHEMA, scope='source_reproduction', scenario_id='v1_visual',
                   source_revision='fixture-source', sample_unit='complete_mechanistic_episode')
    with CircuitArchiveWriter(root, binding) as writer:
        writer.register_circuit(unit)
        writer.add(episode)
    verified = verify_export(CircuitDataset(root, splits=('development',)), episode, unit)
    assert verified['admissible_control_features'] == 6
    assert verified['endpoint_events_retained'] == 1
    assert any(window['start_tick'] == 2000 and window['events'] == 2 for window in verified['windows'])
    assert verified['window_privileged_fields_exposed'] is False


# Synthetic three-unit fixtures exercise the archive contract without simulation.


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def _file_identity(path):
    return dict(bytes=path.stat().st_size, sha256=sha256_file(path))


def _seal_fixture(path, status):
    files = [dict(path=str(p.relative_to(path)), **_file_identity(p))
             for p in sorted(path.rglob('*')) if p.is_file() and p.name != 'delivery.json']
    _json(path/'delivery.json', dict(status=status, exit_code=0, job_id='synthetic', files=files))


def _pointnet_input(packed):
    population = np.array([0, 0, 1], dtype=np.int64)
    ids = np.array([0, 1, 0], dtype=np.int64)
    counts = np.array([2, 0, 1] if packed else [2, 2, 2], dtype=np.int64)
    event_units = np.repeat(np.arange(3), counts)
    times = np.array([.1, .25, 1.1] if packed else [.1, .25, .2, .5, .3, .75])
    encoded = encode_precise_times(times, 1000.)
    incoming = dict(population=np.array(['lgn', 'lgn', 'bkg']), source_id=ids.astype(np.uint64),
        native_id=np.array([10, 11, 12]), inverse_source_id=ids.astype(np.uint32),
        source_event_counts=counts, precise_times=np.ones(3, dtype=bool))
    values = dict(requested_times_ms=times, native_times_ms=encoded['decoded_times_ms'],
                  encoded_steps=encoded['steps'], encoded_offsets_ms=encoded['offsets_ms'])
    incoming.update(values if packed else {k:v.reshape(3, 2) for k,v in values.items()})
    if packed:
        incoming['source_event_unit'] = event_units
    identity = input_plan_identity(dict(population=population, source_id=ids, source_counts=counts,
        source_event_unit=event_units, source_event_time_ms=times))
    mode = 'constructed_visual_inputs' if packed else 'diagnostic_pulses'
    native = dict(input_mode=mode, input_event_schema='packed_source_events_v1' if packed else 'diagnostic_pulse_matrix_v1',
        input_readback=dict(generators=3, events=len(times), tics_per_ms=1000.,
                            all_population_and_source_ids_exact=True), bindings={})
    native['input_readback']['all_generator_events_exact' if packed else 'all_generator_pulses_exact'] = True
    if packed:
        native['bindings']['spike_inputs'] = dict(input_identity_sha256=identity,
            config=dict(spike_generation=dict(starting_noise_seed=123), background=dict(starting_noise_seed=456)))
    return native, incoming, identity


def _pointnet_fixture(tmp_path, packed=False):
    native_root, authority_root = tmp_path/'native', tmp_path/'authority'
    native_root.mkdir(); authority_root.mkdir()
    authority = dict(status='CONSTRUCTED_V1_LENGTH_UNIT_SUPPORTED', exit_code=0, coordinate_units='um',
        glif_coordinate_preservation_required=True, hdf5_unit_attribute_declaration_claimed=False,
        coordinate_semantics='Synthetic sampled model positions', unit_evidence_basis='Synthetic fixture only',
        inner_radial_range=[1., 400.], outer_radial_range=[400., 845.], cortical_locations=['VisL4'], source_bindings={})
    for role in source_dataset.COORDINATE_SOURCES:
        folder=authority_root/'sources'/role; folder.mkdir(parents=True)
        asset=folder/'source_asset.txt'
        asset.write_text(json.dumps(dict(inner_radial_range=[1.,400.], outer_radial_range=[400.,845.],
                                        locations=['VisL4'])) if role=='original_node_parameters' else 'synthetic source '+role)
        info=_file_identity(asset)
        _json(folder/'source.json',dict(artifact=asset.name, **info))
        authority['source_bindings'][role]=dict(info, receipt_sha256=sha256_file(folder/'source.json'))
    for leaf in ('model_page.html','model_page.txt','model_readme.txt'):
        (authority_root/leaf).write_text('Synthetic authority fixture; no empirical evidence.')
    authority['official_page']=dict(url=source_dataset.OFFICIAL_MODEL_URL, path='model_page.html',
                                     **_file_identity(authority_root/'model_page.html'))
    authority['official_text_sha256']=sha256_file(authority_root/'model_page.txt')
    authority['model_readme_sha256']=sha256_file(authority_root/'model_readme.txt')
    _json(authority_root/'authority.json',authority)
    _seal_fixture(authority_root,authority['status'])
    nodes,types,state=metadata(tmp_path)
    types.write_text(types.read_text().replace('Cortex','VisL4'))
    state.update(inverse_source_id=np.arange(3,dtype=np.uint32), population=np.array(['v1']*3),
                 initial_vm_mV=np.array([-70.,-71.,-72.]), terminal_vm_mV=np.array([-60.,-61.,-62.]),
                 spike_count=np.array([2,2,0]))
    spikes=dict(source_id=np.array([0,1,0,1]), native_id=np.array([99,5,99,5]),
                time_ms=np.array([3000.,2999.75,.25,2999.75]))
    native,incoming,input_identity=_pointnet_input(packed)
    native.update(status=source_dataset.POINTNET_STATUSES[native['input_mode']], exit_code=0,
        mode='canary', array_artifacts=list(source_dataset.POINTNET_ARRAYS), kernel_rng_seed=143,
        simulated_ms=3000., cortical_units=3, cortical_spike_events=4, source_denominators={'cortical_units':3},
        connections={'native_rows':0,'physical_synapses':0}, implementation_sha256='a'*64,
        nest_version='synthetic', local_num_threads=1, native_parameters_and_initial_state_exact=True,
        observation_scope='synthetic development fixture', wheel={'sha256':'b'*64})
    native['bindings'].update(sources=authority['source_bindings'],
        point_config={'source_clock':dict(duration_ms=3000.,resolution_ms=.25)},
        recurrent={'files':{'primary/report.json':{'bytes':32,'sha256':'c'*64}}},inputs={'files':{}})
    network=native_root/'primary/network';network.mkdir(parents=True)
    shutil.copy2(nodes,network/'v1_nodes.h5');shutil.copy2(types,network/'v1_node_types.csv')
    bio=native_root/'recurrent/primary/biophysical/v1_nodes.h5';bio.parent.mkdir(parents=True);shutil.copy2(nodes,bio)
    network_identities={}
    for population in ('v1','lgn','bkg'):
        for suffix in ('nodes.h5','node_types.csv','v1_edges.h5','v1_edge_types.csv'):
            name=population+'_'+suffix
            info=_file_identity(network/name) if (network/name).exists() else dict(bytes=0,sha256='d'*64)
            native['bindings']['recurrent' if population=='v1' else 'inputs']['files']['primary/glif/'+name]=info
            network_identities[name]=info
    native['bindings']['recurrent']['files']['primary/biophysical/v1_nodes.h5']=_file_identity(bio)
    parameters={'101':dict(all_units_exact=True,units=2,model='glif_psc',parameters={'V_m':-70.}),
                '202':dict(all_units_exact=True,units=1,model='glif_psc',parameters={'V_m':-71.})}
    for process in ('primary','replay'):
        folder=native_root/process/'native';folder.mkdir(parents=True)
        for leaf,value in [('cortical_state.npz',state),('cortical_spikes.npz',spikes),('input_events.npz',incoming)]:
            np.savez(folder/leaf,**value)
        _json(folder/'parameters.json',parameters);_json(native_root/process/'report.json',native)
    comparison=dict(status='POINTNET_NATIVE_REPLAY_EQUAL',exit_code=0,all_native_outputs_equal=True,
        independent_processes=True,binding_or_runtime_differences=[],
        left_report_sha256=sha256_file(native_root/'primary/report.json'),
        right_report_sha256=sha256_file(native_root/'replay/report.json'),
        files=[dict(path=key,equal=True) for key in [*source_dataset.POINTNET_ARRAYS,'native/parameters.json']])
    _json(native_root/'comparison/report.json',comparison)
    _seal_fixture(native_root,source_dataset.POINTNET_DELIVERIES[native['input_mode']])
    return native_root,authority_root,native,network_identities,parameters,incoming,input_identity


@pytest.mark.parametrize('packed',[False,True])
def test_pointnet_full_export_retains_source_evidence_without_controls(tmp_path,packed):
    native,authority,report,_,_,incoming,identity=_pointnet_fixture(tmp_path,packed)
    output=tmp_path/'export'
    exported=source_dataset.export_pointnet(native,authority,output)
    assert exported['status']=='EXPORTED_VERIFIED_POINTNET_SOURCE_OBSERVATION'
    assert exported['source_units']==3 and exported['complete_source_cases']==1
    assert exported['replay_processes_verified']==2 and exported['replays_counted_as_episodes']==0
    assert exported['production_episodes']==0 and exported['scientific_qualification'] is False
    assert exported['at_least_100000_cortical_units'] is False
    assert exported['verification']['endpoint_events_retained']==1
    assert exported['verification']['silent_units_retained']==1
    dataset=CircuitDataset(output/'dataset',splits=('development',))
    batch=dataset.window(exported['episode_id'],boundary_tick=4,history_ticks=4,horizon_ticks=4,bin_ticks=1)
    assert batch['known_controls'].shape==(8,0) and batch['control_features']==[]
    assert 'privileged' not in batch and not any(k.startswith('incoming_') for k in batch)
    shard=dataset.shards[dataset.episodes[exported['episode_id']]['shard']]
    with h5py.File(output/'dataset'/shard['path'],'r') as f:
        private=f[exported['episode_id']+'/privileged']
        np.testing.assert_array_equal(private['incoming_population_index'][:],[0,0,1])
        np.testing.assert_array_equal(private['incoming_raw_event_time_ms'][:],incoming['requested_times_ms'].reshape(-1))
        np.testing.assert_array_equal(private['incoming_native_event_time_ms'][:],incoming['native_times_ms'].reshape(-1))
        np.testing.assert_array_equal(private['incoming_source_event_counts'][:],[2,0,1] if packed else [2,2,2])
        np.testing.assert_array_equal(private['initial_vm_mV'][:],[-70.,-71.,-72.])
    actual,_=source_dataset.pointnet_input_evidence(report,incoming)
    assert actual==identity


def test_constructed_locations_are_explicit_and_coordinate_equality_is_exact(tmp_path):
    nodes,types,native=metadata(tmp_path)
    types.write_text(types.read_text().replace('Cortex','VisL4'))
    with pytest.raises(ValueError,match='NONCORTICAL'):
        metadata_units(nodes,types,native,'fixture',{'connectivity_identity':'g','parameter_identity':'p'},'positions')
    unit,_=metadata_units(nodes,types,native,'fixture',{'connectivity_identity':'g','parameter_identity':'p'},
                          'positions',circuit_id='constructed',allowed_locations=('VisL4',))
    assert unit.circuit_id=='constructed'
    assert source_dataset.verify_coordinate_preservation(unit,nodes)['coordinate_values_exact']==9
    with h5py.File(nodes,'r+') as f:
        value=f['nodes/v1/0/x'][0];f['nodes/v1/0/x'][0]=np.nextafter(value,np.inf)
    with pytest.raises(ValueError,match='COORDINATES_DIFFER'):
        source_dataset.verify_coordinate_preservation(unit,nodes)


@pytest.mark.parametrize('failure',['source_binding','failed_authority'])
def test_constructed_authority_rejects_wrong_source_or_failure(tmp_path,failure):
    _,authority,native,*_=_pointnet_fixture(tmp_path)
    if failure=='source_binding':
        native=copy.deepcopy(native);native['bindings']['sources']['original_node_functions']['sha256']='e'*64
    else:
        body=json.loads((authority/'authority.json').read_text());body['status']='FAILED'
        _json(authority/'authority.json',body);_seal_fixture(authority,'FAILED')
    with pytest.raises(ValueError,match='COORDINATE'):
        source_dataset.constructed_coordinate_authority(authority,native)


@pytest.mark.parametrize('failure',['parameters_omitted','corrupt_replay'])
def test_pointnet_comparison_requires_parameters_and_rechecks_arrays(tmp_path,failure):
    archive,_,native,*_=_pointnet_fixture(tmp_path)
    if failure=='parameters_omitted':
        path=archive/'comparison/report.json';comparison=json.loads(path.read_text())
        comparison['files']=[row for row in comparison['files'] if row['path']!='native/parameters.json']
        _json(path,comparison)
    else:
        path=archive/'replay/native/cortical_spikes.npz'
        with np.load(path,allow_pickle=False) as f:values={k:f[k] for k in f.files}
        values['time_ms'][0]=2999.5;np.savez(path,**values)
    _seal_fixture(archive,source_dataset.POINTNET_DELIVERIES[native['input_mode']])
    with pytest.raises(ValueError,match='COMPARISON_MISSING|PAIRED_ARRAYS_DIFFER'):
        source_dataset.read_pointnet_outputs(archive)


@pytest.mark.parametrize('packed',[False,True])
def test_pointnet_input_time_and_integer_identity_fail_closed(packed):
    native,incoming,_=_pointnet_input(packed)
    wrong=copy.deepcopy(incoming);wrong['native_times_ms'].flat[0]=np.nextafter(wrong['native_times_ms'].flat[0],np.inf)
    with pytest.raises(ValueError,match='PRECISE_TIME_READBACK'):
        source_dataset.pointnet_input_evidence(native,wrong)
    wrong=copy.deepcopy(incoming);wrong['source_id'][0]=2**63
    with pytest.raises(ValueError,match='INTEGER_INVENTORY'):
        source_dataset.pointnet_input_evidence(native,wrong)


def test_pointnet_identity_tracks_physical_content_not_execution_receipts(tmp_path):
    _,_,native,network,parameters,_,input_identity=_pointnet_fixture(tmp_path)
    original=source_dataset.pointnet_circuit_identity(native,network,parameters)
    updated=copy.deepcopy(native)
    updated.update(job_id='another-job',started_utc='later',delivery_sha256='f'*64)
    updated['bindings']['recurrent']['files']['primary/report.json']['sha256']='e'*64
    for value in updated['bindings']['sources'].values():value['receipt_sha256']='f'*64
    assert source_dataset.pointnet_circuit_identity(updated,network,parameters)==original
    circuit=original[0];vm=np.array([-70.,-71.,-72.]);clock={'duration_ms':3000.,'resolution_ms':.25}
    case=source_dataset.pointnet_case_metadata(native,circuit,vm,input_identity,clock)
    assert source_dataset.pointnet_case_metadata(updated,circuit,vm,input_identity,clock)==case
    changed=copy.deepcopy(network);changed['v1_v1_edges.h5']['sha256']='e'*64
    assert source_dataset.pointnet_circuit_identity(native,changed,parameters)[0]!=circuit
    changed_case=source_dataset.pointnet_case_metadata(native,circuit,vm,'e'*64,clock)
    assert changed_case[0]!=case[0] and changed_case[1]==case[1]
    changed_vm=vm.copy();changed_vm[0]=np.nextafter(vm[0],np.inf)
    changed_case=source_dataset.pointnet_case_metadata(native,circuit,changed_vm,input_identity,clock)
    assert changed_case[0]!=case[0] and changed_case[1]!=case[1]


@pytest.mark.parametrize('pointnet',[False,True])
def test_source_cli_preserves_both_routes(monkeypatch,tmp_path,pointnet):
    calls=[]
    def fake(*args):
        calls.append(args)
        return {'status':'EXPORTED_VERIFIED_POINTNET_SOURCE_OBSERVATION' if pointnet else 'EXPORTED_VERIFIED_V1_SOURCE_OBSERVATION'}
    monkeypatch.setattr(source_dataset,'export_pointnet' if pointnet else 'export',fake)
    args=['--output',str(tmp_path/'export')]
    args+=['--pointnet-native-archive','native','--coordinate-authority','authority'] if pointnet else [
        '--native-archive','native','--repeatability-archive','replay','--input-replay-archive','inputs','--asset-root','assets']
    assert source_dataset.main(args)==0 and len(calls)==1


def test_pointnet_observation_verifies_activity_between_silent_boundaries(tmp_path):
    archive,authority,native,*_=_pointnet_fixture(tmp_path)
    for process in ('primary','replay'):
        path=archive/process/'native/cortical_spikes.npz'
        with np.load(path,allow_pickle=False) as f:
            values={key:f[key] for key in f.files}
        values['time_ms']=np.array([1500.,1500.,1500.25,1500.25])
        np.savez(path,**values)
    _seal_fixture(archive,source_dataset.POINTNET_DELIVERIES[native['input_mode']])
    report=source_dataset.export_pointnet(archive,authority,tmp_path/'export')
    windows=report['verification']['windows']
    assert next(row for row in windows if row['start_tick']==0)['events']==0
    assert next(row for row in windows if row['stop_tick']==12000)['events']==0
    assert any(row['events']==4 and row['all_units']==3 for row in windows)



def test_pointnet_source_commands_preserve_episode_and_privileged_exclusion(tmp_path,monkeypatch):
    archive,authority,native,*_=_pointnet_fixture(tmp_path,packed=True)
    plain=source_dataset.export_pointnet(archive,authority,tmp_path/'plain')
    stimulus=dict(duration_s=3.,gray_screen_s=.5,spatial_frequency_cycles_per_degree=.04,
                  temporal_frequency_hz=2.,direction_deg=315.,contrast_percent=80.)
    commands=make_commands(stimulus)
    def load_commands(path,bound_native,*,native_family):
        assert native_family=='pointnet' and bound_native['input_mode']=='constructed_visual_inputs'
        assert path==tmp_path/'commands'
        return dict(control_ticks=commands['control_ticks'],control_values=commands['control_values'],
            control_features=CONTROL_FEATURES,source_control_record=dict(source_family='constructed_lgn_background',
                source_command_representation_verified=True,scientific_qualification=False,
                source_fidelity_qualified=False,visual_action_mapping_qualified=False,production_domain_qualified=False))
    monkeypatch.setattr(source_dataset,'load_visual_controls',load_commands)
    controlled=source_dataset.export_pointnet(archive,authority,tmp_path/'controlled',tmp_path/'commands')
    assert controlled['episode_id']==plain['episode_id'] and controlled['circuit_id']==plain['circuit_id']
    assert controlled['complete_source_cases']==1 and controlled['production_episodes']==0
    assert controlled['exported_declared_visual_controls'] is True
    assert controlled['source_command_representation_verified'] is True
    assert controlled['scientific_qualification'] is False
    assert controlled['admissible_visual_control_mapping_qualified'] is False
    dataset=CircuitDataset(tmp_path/'controlled/dataset',splits=('development',))
    batch=dataset.window(controlled['episode_id'],boundary_tick=2004,history_ticks=4,horizon_ticks=4,bin_ticks=1)
    expected=np.vstack((np.repeat(commands['control_values'][500:501],4,axis=0),
                        np.repeat(commands['control_values'][501:502],4,axis=0)))
    np.testing.assert_array_equal(batch['known_controls'],expected)
    assert 'privileged' not in batch and not any(key.startswith('incoming_') for key in batch)
    assert controlled['verification']['window_privileged_fields_exposed'] is False


def test_pointnet_diagnostic_case_rejects_visual_controls(tmp_path):
    archive,authority,*_=_pointnet_fixture(tmp_path)
    with pytest.raises(ValueError,match='VISUAL_CONTROLS_REQUIRE_CONSTRUCTED_INPUTS'):
        source_dataset.export_pointnet(archive,authority,tmp_path/'export',tmp_path/'commands')
    report=json.loads((tmp_path/'export/report.json').read_text())
    assert report['status']=='FAILED_POINTNET_SOURCE_OBSERVATION_EXPORT'
    assert report['production_episodes']==0 and report['scientific_qualification'] is False


def test_pointnet_cli_passes_optional_visual_control_archive(monkeypatch,tmp_path):
    calls=[]
    def export(*args):
        calls.append(args);return {'status':'EXPORTED_VERIFIED_POINTNET_SOURCE_OBSERVATION'}
    monkeypatch.setattr(source_dataset,'export_pointnet',export)
    result=source_dataset.main(['--pointnet-native-archive','native','--coordinate-authority','authority',
        '--visual-controls-archive',str(tmp_path/'commands'),'--output',str(tmp_path/'export')])
    assert result==0 and calls[0][-1]==tmp_path/'commands'


def test_source_tuning_uses_source_id_and_group_index_order(tmp_path):
    nodes, types, native = metadata(tmp_path)
    with h5py.File(nodes, "r+") as source:
        source["nodes/v1/0/tuning_angle"] = np.array([90., 135., 45.])
    unit, _ = metadata_units(nodes, types, native, "fixture-source",
        dict(connectivity_identity="graph", parameter_identity="params"), "source geometry")
    np.testing.assert_array_equal(unit.source_tuning_angle_deg, [45., 90., 135.])
    with h5py.File(nodes, "r+") as source:
        del source["nodes/v1/0/tuning_angle"]
    legacy, _ = metadata_units(nodes, types, native, "fixture-source",
        dict(connectivity_identity="graph", parameter_identity="params"), "source geometry")
    assert legacy.source_tuning_angle_deg is None


def test_export_validator_requires_exact_source_tuning_metadata(tmp_path):
    unit = replace(units(tmp_path), source_tuning_angle_deg=np.array([45., 90., 135.]))
    ep = CircuitEpisode('source_episode', unit.circuit_id, 'source_block', 'development', 12, .25,
        np.array([1, 11, 12]), np.array([0, 1, 0]), np.empty(0, dtype=np.int64), np.empty((0, 0)), (),
        source_record=dict(complete_unit_recording=True, simulation_completed=True, recorded_until_tick=12,
            expected_duration_ticks=12, core_unit_count=3, source_revision=unit.source_revision,
            circuit_provenance=unit.circuit_provenance))
    root = tmp_path / 'dataset'
    with CircuitArchiveWriter(root, dict(schema=SCHEMA, scope='source_reproduction', scenario_id='v1_visual',
            source_revision=unit.source_revision, sample_unit='complete_mechanistic_episode')) as writer:
        writer.register_circuit(unit); writer.add(ep)
    data = CircuitDataset(root, splits=('development',))
    assert verify_export(data, ep, unit)['static_unit_metadata_exact'] == 3
    with pytest.raises(ValueError, match='source_tuning_angle_deg'):
        verify_export(data, ep, replace(unit, source_tuning_angle_deg=np.array([45., 91., 135.])))
    with pytest.raises(ValueError, match='source_tuning_angle_deg'):
        verify_export(data, ep, replace(unit, source_tuning_angle_deg=None))
