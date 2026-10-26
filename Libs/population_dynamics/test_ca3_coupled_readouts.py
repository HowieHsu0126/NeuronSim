"""Sparse coupled-format fixtures verify software, never neural dynamics."""
from copy import deepcopy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

from . import ca3_control_trace as control
from . import ca3_native_observation as observer
from . import ca3_native_observation_replay as observation_replay
from . import ca3_readout_replay as readout_replay
from . import ca3_retrieval_readouts as readouts
from .ca3_coupled_extension import extend_source
from .ca3_reference import prepare_retrieval_program
from .carlsim_records import NETWORK_HEADER, GROUP_HEADER, NETWORK_SIGNATURE, validate_retrieval
from .provenance import sha256_file
from .test_ca3_coupled_extension import source_fixture
from .test_ca3_retrieval_readouts import _events, PYR_EVENTS, runtime
from .test_ca3_native_observation_replay import add_memory_overlay, record, refresh_bindings


def write_json(path, value):
    path.write_text(json.dumps(value))


def fixture(root, *, short=False, observed=False, conductance_memory=False):
    (root/'native_run/results').mkdir(parents=True)
    (root/'bin').mkdir()
    (root/'bin/ca3_retrieval').write_bytes(b'fixture binary; no simulator execution')
    header, main, pattern = source_fixture('retrieval')
    derived_header, derived_main, derived_pattern, extension = extend_source(
        header, main, pattern, multiplier=2, role='retrieval')
    (root/'main.cpp').write_bytes(derived_main.encode())
    (root/'config.h').write_bytes(derived_header.encode())
    contract = prepare_retrieval_program(root/'main.cpp',root/'config.h',local_input=True,population_multiplier=2)
    (root/'trainPat.csv').write_bytes(derived_pattern.encode())
    (root/'native_run/trainPat.csv').write_bytes(derived_pattern.encode())
    extension.update(source_pattern_sha256=hashlib.sha256(pattern.encode()).hexdigest(),
                     derived_pattern_sha256=sha256_file(root/'trainPat.csv'))
    assert sha256_file(root/'main.cpp') == readouts.COUPLED_OVERLAY_SHA256
    assert sha256_file(root/'config.h') == readouts.COUPLED_CONFIG_SHA256
    assert sha256_file(root/'trainPat.csv') == readouts.COUPLED_PATTERN_SHA256
    identity=readouts.source_identity(coupled_extension=True)
    layout={};cursor=0
    for name in sorted(identity['populations']):
        n=identity['populations'][name];layout[name]=(cursor,cursor+n);cursor+=n
    order=list(reversed(sorted(layout)));groups=[];all_events={}
    payload=NETWORK_HEADER.pack(NETWORK_SIGNATURE,.3,45.85,1.,cursor,len(order))
    for group_id,name in enumerate(order):
        start,stop=layout[name];n=stop-start
        payload+=GROUP_HEADER.pack(start,stop-1,n,1,1,name.encode())
        groups.append(observer.Group(group_id,start,stop,0,group_id,start,stop,1,name))
        rows=([(0,0),(1,n-1)] if short else
              [(t,2*u) for t,u in PYR_EVENTS] if name==readouts.PYRAMIDAL else [(0,0),(1000,n-1)])
        if observed: rows=list(reversed(rows))
        all_events[name]=rows
        _events(root/'native_run/results'/('spk_'+name+'.dat'),n,rows)
    (root/'native_run'/identity['input_network']).write_bytes(payload)
    proofs=[]
    for path,preserved,source,derived in [
        (readouts.MAIN_PATH,False,readouts.AUTHOR_MAIN_SHA256,sha256_file(root/'main.cpp')),
        ('generateCONFIGStateSTP.h',False,readouts.CONFIG_SHA256,sha256_file(root/'config.h')),
        ('generateSETUPStateSTP.h',True,readouts.SETUP_SHA256,readouts.SETUP_SHA256)]:
        proofs.append(dict(path=path,source_bytes_preserved=preserved,source_path_binding_only=False,
            source_sha256=source,staged_sha256=derived,coupled_extension_transformation_verified=True))
    build=dict(status='COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED',
        scope='coupled_extension_retrieval',expected_duration_ms=1001,
        framework_revision=readouts.SOURCE_REVISION,author_revision=readouts.AUTHOR_REVISION,
        core_neurons=168106,populations=identity['populations'],integration='RUNGE_KUTTA4_5',delay_ms=1,
        source_rng={'source_seed':10},pattern_sha256=extension['derived_pattern_sha256'],pattern_integer_count=2200,
        original_author_main_sha256=readouts.AUTHOR_MAIN_SHA256,overlay_source_sha256=sha256_file(root/'main.cpp'),
        source_file_byte_checks=proofs,source_code_semantics_preserved_before_stop=False,
        retrieval_program_changed_only_in_input_path=False,retrieval_input_contract=contract,
        coupled_extension=extension,coupled_extension_config=dict(path=str(root/'coupled_extension.yaml'),
            sha256='8'*64,resolved={'population_multiplier':2}),
        binary_sha256=sha256_file(root/'bin/ca3_retrieval'))
    paths=['build_report.json','trainPat.csv','bin/ca3_retrieval']
    if short:
        for path in (*observation_replay.REBUILT_ARTIFACTS[1:],observation_replay.SHARED_RUNTIME):
            p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(path.encode());paths.append(path)
    if observed:
        originals={name:(observation_replay.PINNED_SOURCE_ROOT/name).read_bytes() for name in observer.PINNED_FILES}
        overlays=observer.overlay_bytes(originals)
        build['observer_implementation_sha256']=sha256_file(Path(observer.__file__))
        build['native_observation_overlay']=dict(framework_revision=observer.REVISION,
            enabled_by=observer.ENVIRONMENT_VARIABLE,original_sha256=observer.PINNED_FILES.copy(),
            overlay_sha256={name:hashlib.sha256(data).hexdigest() for name,data in overlays.items()},
            current_units=observer.CURRENT_UNIT.copy(),counter_source='native nSpikeCnt',count_interval_ms=1,
            accumulated_count_dtype='int64',current_source='runtime extCurrent; GPU device-to-host or CPU direct',
            native_build_verified=True,native_execution_verified=False)
        p=root/observation_replay.OBSERVER_ADAPTER_PATH;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(Path(observer.__file__).read_bytes());paths.append(observation_replay.OBSERVER_ADAPTER_PATH)
        for name,data in overlays.items():
            relative='observer_source/'+name;p=root/relative;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(data);paths.append(relative)
    if conductance_memory:
        add_memory_overlay(root,build,paths,observed=observed)
    write_json(root/'build_report.json',build)
    write_json(root/'manifest.json',dict(files=[dict(path=p,sha256=sha256_file(root/p)) for p in paths]))
    native=deepcopy(build)
    native.update(status='COUPLED_EXTENSION_RETRIEVAL_EXIT_ZERO',exit_code=0,native_run_executed=True,
        retrieval_executed=True,normalized_input_qualified=True,
        normalized_input_sha256=sha256_file(root/'native_run'/identity['input_network']),
        runtime_manifest_sha256=sha256_file(root/'manifest.json'),native_pid=2 if observed else 1,
        slurm_job_id='fixture-observed' if observed else 'fixture-original',
        started_utc='2026-09-29T01:01:00Z' if observed else '2026-09-29T01:00:00Z',
        finished_utc='2026-09-29T01:01:01Z' if observed else '2026-09-29T01:00:01Z')
    for key in observation_replay.INPUT_FIELDS:
        if key!='normalized_input_sha256':native[key]='a'*64
    if observed:
        with (root/'native_observation.bin').open('wb') as stream:
            stream.write(observer.HEADER.pack(observer.MAGIC,len(groups),168106))
            for g in groups:
                name=g.name.encode();stream.write(observer.GROUP.pack(g.group_id,g.start,g.stop,0,g.group_id,g.start,g.stop,1,len(name))+name)
            for g in groups:stream.write(record(b'R',0,0,g.group_id,np.ones(g.units)))
            for t in range(2):
                for g in groups:stream.write(record(b'A',t,t+1,g.group_id,np.ones(g.units)))
                for g in groups:
                    counts=np.zeros(g.units,dtype='<i4')
                    for time,unit in all_events[g.name]:
                        if time==t:counts[unit]+=1
                    stream.write(record(b'C',t,t+1,g.group_id,counts))
            stream.write(observer.RECORD.pack(b'E',2,2,-1,0))
        native.update(native_observation_stream_validated=True,native_observation_sha256=sha256_file(root/'native_observation.bin'),
            native_observation_bytes=(root/'native_observation.bin').stat().st_size,
            observed_native_duration_ms=2,host_request_runtime_current_equal=True)
        native['native_observation_overlay']['native_execution_verified']=True
    write_json(root/'native_report.json',native)
    validate_retrieval(root,root/'observations',coupled_extension=True)
    if short:
        # Two-ms observer fixture retains all168106 identities; it is not a
        # complete native retrieval or a source fidelity qualification.
        for filename in ['build_report.json','native_report.json']:
            p=root/filename;value=json.loads(p.read_text());value['expected_duration_ms']=2;write_json(p,value)
        p=root/'observations/population_counts.npz'
        with np.load(p,allow_pickle=False) as saved:values={k:saved[k] for k in saved.files}
        values['population_counts_per_ms']=values['population_counts_per_ms'][:2]
        np.savez_compressed(p,**values)
        p=root/'observations/recording.json';value=json.loads(p.read_text());value['duration_ms']=2
        value['observation_sha256']=sha256_file(root/'observations/population_counts.npz');write_json(p,value)
        refresh_bindings(root)
    return root,layout[readouts.PYRAMIDAL][0]



def add_realization(root):
    """Add real source/config binding to an explicitly synthetic event fixture."""
    from .ca3_realization import SourceRealization, contract, program
    from .config import load_config

    spec = SourceRealization('coupled-development', 'noise-01', 'retrieval', 'development', 10, 1001, 2001)
    campaign = Path(os.environ.get('NEURONSIM_CA3_CAMPAIGN_CONFIG', str(
        Path(__file__).resolve().parents[1]/'configs/neural_population_dynamics.yaml')))
    config = root/'source_realization.yaml'
    write_json(config, {**load_config(campaign), 'ca3_source_realization': spec.record()})
    original = (root/'main.cpp').read_bytes()
    effective = program(original, spec)
    (root/'main.cpp').write_bytes(effective)
    (root/'realization_source.cpp').write_bytes(effective)
    for filename in ('build_report.json','native_report.json'):
        path=root/filename;value=json.loads(path.read_text())
        if filename=='build_report.json':value['status']='CA3_REALIZATION_BUILD_PASSED_NATIVE_QUALIFICATION_PENDING'
        value['source_rng']='explicit_post_setup_noise_block_on_source_reset'
        value['source_realization']=contract(spec)
        value['source_realization_config']=dict(path=str(config),sha256=sha256_file(config),resolved=spec.record())
        value['overlay_source_sha256']=sha256_file(root/'realization_source.cpp')
        row=next(r for r in value['source_file_byte_checks'] if r['path']==readouts.MAIN_PATH)
        row.update(pre_realization_staged_sha256=row['staged_sha256'],staged_sha256=value['overlay_source_sha256'],
                   source_realization_transformation_verified=True)
        if filename=='native_report.json':
            value['realization_circuit_binding']=dict(circuit_id=spec.circuit_id,circuit_seed=10,
                normalized_input_sha256=value['normalized_input_sha256'],formation_report_sha256=value['formation_report_sha256'],
                block_id=spec.block_id,new_circuit_claimed=False,microscopic_state_restarted_from_checkpoint=False)
        write_json(path,value)
    manifest=json.loads((root/'manifest.json').read_text())
    for name in ('realization_source.cpp','source_realization.yaml'):
        manifest['files'].append(dict(path=name,sha256=sha256_file(root/name)))
    write_json(root/'manifest.json',manifest)
    native=json.loads((root/'native_report.json').read_text())
    path=root/'observations/recording.json';recording=json.loads(path.read_text())
    recording.update(source_realization=native['source_realization'],realization_circuit_binding=native['realization_circuit_binding'])
    write_json(path,recording);refresh_bindings(root)
    return config


def test_explicit_realization_source_readout_and_independent_replay(tmp_path):
    root,_=fixture(tmp_path/'runtime');config=add_realization(root)
    result=readouts.analyze(root,tmp_path/'readouts',coupled_extension=True,source_realization_config=config)
    checked=readout_replay.replay(root,tmp_path/'readouts',tmp_path/'checked.json',
                                coupled_extension=True,source_realization_config=config)
    assert result['source_realization']==checked['source_realization']
    assert not result['scientific_fidelity_qualified']
    assert checked['array_values_compared'] > 900000
    with pytest.raises(ValueError,match='EXPLICIT_CONFIGURATION_REQUIRED'):
        readouts.analyze(root,tmp_path/'without-config',coupled_extension=True)
    with pytest.raises(ValueError,match='EXPLICIT_CONFIGURATION_REQUIRED'):
        readout_replay.replay(root,tmp_path/'readouts',tmp_path/'without-config.json',coupled_extension=True)
    wrong=tmp_path/'wrong.yaml';value=json.loads(config.read_text());value['ca3_source_realization']['engine_seed']=2002
    write_json(wrong,value)
    with pytest.raises(ValueError,match='BUILD_AND_REQUEST_DIFFER'):
        readouts.analyze(root,tmp_path/'wrong',coupled_extension=True,source_realization_config=wrong)


def test_source_bytes_full_identities_and_independent_readout_replay(tmp_path):
    root,start=fixture(tmp_path/'runtime');output=tmp_path/'readouts'
    result=readouts.analyze(root,output,coupled_extension=True)
    assert result['full_record_recounted_units']==168106
    assert result['window_counted_pyramidal_units']==150754
    assert (result['target_never_boosted_units'],result['other_never_boosted_units'],result['eligible_for_boost_units'])==(274,548,276)
    with np.load(output/'pyramidal_window_counts.npz',allow_pickle=False) as saved:
        assert saved['window_counts'].shape==(3,2,150754)
        assert saved['window_counts'][0,1,276]==3
        np.testing.assert_array_equal(saved['pyramidal_global_unit_ids'],np.arange(start,start+150754))
    replay=readout_replay.replay(root,output,tmp_path/'replay.json',coupled_extension=True)
    assert replay['window_count_values_compared']==904524 and replay['scalar_values_compared']==141
    assert not result['scientific_fidelity_qualified'] and not replay['scientific_claim_qualified']


def test_coupled_readout_requires_explicit_opt_in(tmp_path):
    root,_=fixture(tmp_path/'runtime')
    with pytest.raises(ValueError):readouts.analyze(root,tmp_path/'rejected')
    readouts.analyze(root,tmp_path/'readouts',coupled_extension=True)
    with pytest.raises(ValueError):readout_replay.replay(root,tmp_path/'readouts',tmp_path/'replay.json')
    assert not (tmp_path/'rejected').exists()


@pytest.mark.parametrize('mutation',['source_hash','path_only','fresh_checkpoint','config','pattern_origin','decoder_identity','changed_event'])
def test_coupled_readout_rejects_mixed_or_mutated_provenance(tmp_path,mutation):
    root,_=fixture(tmp_path/'runtime');p=root/'build_report.json';b=json.loads(p.read_text())
    if mutation=='source_hash':b['source_file_byte_checks'][0]['staged_sha256']='a'*64
    elif mutation=='path_only':b['retrieval_input_contract']['source_path_binding_only']=True
    elif mutation=='fresh_checkpoint':b['retrieval_input_contract']['fresh_coupled_checkpoint_required']=False
    elif mutation=='config':b['coupled_extension_config']['sha256']='9'*64
    elif mutation=='pattern_origin':b['coupled_extension']['source_pattern_sha256']='9'*64
    elif mutation=='decoder_identity':
        p=root/'observations/recording.json';b=json.loads(p.read_text());b['coupled_extension']['graph_or_output_cloning']=True
    elif mutation=='changed_event':
        p=root/'native_run/results/spk_CA3_Pyramidal.dat';p.write_bytes(p.read_bytes()+struct.pack('<ii',520,276))
    if mutation!='changed_event':write_json(p,b)
    with pytest.raises(ValueError):readouts.analyze(root,tmp_path/'rejected',coupled_extension=True)
    assert not (tmp_path/'rejected').exists()


def test_declared_commands_double_membership_without_changing_time_or_assignment_semantics():
    _,_,pattern=source_fixture('retrieval');old=np.asarray([int(x) for x in pattern.split()])
    derived=(2*old[:,None]+np.arange(2)).ravel()
    original=control.source_commands(old,100)
    commands=control.source_commands(derived,200,coupled_extension=True)
    np.testing.assert_array_equal(commands['time_ms'],np.repeat(original['time_ms'],2))
    np.testing.assert_array_equal(commands['unit_ids'],(2*original['unit_ids'][:,None]+np.arange(2)).ravel())
    assert np.all(commands['set_current']==10000)
    with pytest.raises(ValueError):control.source_commands(derived,200)
    with pytest.raises(ValueError):control.source_commands(old,200,coupled_extension=True)


@pytest.mark.parametrize('conductance_memory,explicit_realization',[(False,False),(True,False),(True,True)])
def test_coupled_observer_replay_retains_all_units_and_rejects_mixed_identity(tmp_path,monkeypatch,conductance_memory,explicit_realization):
    left,_=fixture(tmp_path/'original',short=True,conductance_memory=conductance_memory)
    right,_=fixture(tmp_path/'observed',short=True,observed=True,conductance_memory=conductance_memory)
    options={}
    if explicit_realization:
        config=add_realization(left);add_realization(right)
        options['source_realization_config']=config
    monkeypatch.setattr(readouts,'DURATION_MS',2)
    monkeypatch.setattr(observation_replay,'DURATION_MS',2)
    result=observation_replay.compare(left,right,tmp_path/'pass.json',coupled_extension=True,**options)
    assert result['status']=='PASS_COUPLED_EXTENSION_CA3_OBSERVATION_REPLAY'
    assert result['units']==168106 and result['unit_millisecond_counts_compared']==336212
    provenance=result['coupled_configuration_provenance']
    assert provenance['original']['path'] != provenance['observed']['path']
    assert result['coupled_extension_config'] == provenance['observed']
    assert result['original_recording_sha256'] == sha256_file(left/'observations/recording.json')
    assert result['event_multisets_equal'] and not result['event_order_equal']
    assert not result['scientific_fidelity_qualified']
    assert result['source_overlay_verification']['observed']['conductance_memory_enabled']==conductance_memory
    with pytest.raises(ValueError):observation_replay.compare(left,right,tmp_path/'default-rejected.json')
    for filename in ['build_report.json','native_report.json']:
        p=right/filename;b=json.loads(p.read_text());b['coupled_extension_config']['sha256']='9'*64;write_json(p,b)
    refresh_bindings(right)
    with pytest.raises(ValueError,match='COUPLED_IDENTITY_DIFFERS'):
        observation_replay.compare(left,right,tmp_path/'mixed-rejected.json',coupled_extension=True,**options)


def test_readout_and_replay_cli_dispatch(tmp_path,monkeypatch):
    root,_=fixture(tmp_path/'runtime');output=tmp_path/'readouts'
    monkeypatch.setattr(sys,'argv',['readouts','--runtime',str(root),'--output',str(output),'--coupled-extension'])
    readouts.main()
    monkeypatch.setattr(sys,'argv',['replay','--runtime',str(root),'--readouts',str(output),
                                  '--output',str(tmp_path/'replay.json'),'--coupled-extension'])
    readout_replay.main()
    assert json.loads((tmp_path/'replay.json').read_text())['status']=='PASS_COUPLED_EXTENSION_RAW_EVENT_READOUT_REPLAY'


def test_reference_independent_readout_replay_still_works(runtime,tmp_path):
    root,_=runtime;output=tmp_path/'reference-readouts'
    readouts.analyze(root,output)
    result=readout_replay.replay(root,output,tmp_path/'reference-replay.json')
    assert result['status']=='PASS_INDEPENDENT_RAW_EVENT_READOUT_REPLAY'
    assert result['window_count_values_compared']==452262


@pytest.mark.parametrize('cue_omission,native_reference,realization', [(False,False,False),(True,False,False),(False,True,False),(False,False,True)])
def test_launcher_forwards_coupled_retrieval_input_binding(tmp_path, cue_omission, native_reference,realization):
    script=Path(__file__).resolve().parents[1]/'scripts/run_population_ca3_reference.sh'
    project=tmp_path/'project';package=project/'Libs/population_dynamics';package.mkdir(parents=True)
    for name in ['__init__','provenance','config','circuit_sources','carlsim_records','ca3_native_observation',
                 'ca3_parameter_comparison','ca3_coupled_extension','ca3_reference','ca3_toolchain',
                 'ca3_conductance_memory','ca3_cue_omission','ca3_realization']:
        (package/(name+'.py')).write_text('# launcher fixture; not executed\n')
    cfg=tmp_path/'campaign.yaml';cfg.write_text('ca3_coupled_extension:\n  population_multiplier: 2\n')
    captured=tmp_path/'argv.json';fake=tmp_path/'python-fixture'
    fake.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
        'if "-c" in sys.argv: print("/nonexistent-fixture-include")\n'
        'if "Libs.population_dynamics.ca3_reference" in sys.argv:\n'
        ' open(os.environ["CAPTURED_ARGV"],"w").write(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    user='neuronsim-launcher-fixture-'+tmp_path.name
    scratch=Path('/tmp')/user
    env=dict(os.environ,USER=user,SLURM_JOB_ID='fixture',SLURM_SUBMIT_DIR=str(project),
        CA3_PYTHON=str(fake),CA3_SCOPE='retrieval',CA3_COUPLED_EXTENSION_CONFIG=str(cfg),CAPTURED_ARGV=str(captured),
        CA3_CUE_OMISSION='1' if cue_omission else '0',
        CA3_NATIVE_OBSERVATION='1' if cue_omission or native_reference else '0',
        CA3_NATIVE_REFERENCE='1' if native_reference else '0',CA3_CONDUCTANCE_MEMORY='1' if native_reference or realization else '0',
        CA3_SOURCE_REALIZATION_CONFIG=str(cfg) if realization else '')
    if native_reference:env.pop('CA3_COUPLED_EXTENSION_CONFIG')
    try:
        run=subprocess.run(['bash',str(script),'','--build-only','89'],env=env,capture_output=True,text=True)
        assert run.returncode==0,run.stderr
        arguments=json.loads(captured.read_text())
        assert '--local-retrieval-input' in arguments
        assert ('--coupled-extension-config' in arguments) is (not native_reference)
        assert arguments[arguments.index('--scope')+1]=='retrieval'
        assert ('--cue-omission' in arguments) is cue_omission
        assert ('--native-observation' in arguments) is (cue_omission or native_reference)
        assert ('--conductance-memory' in arguments) is (native_reference or realization)
        assert ('--source-realization-config' in arguments) is realization
        if realization:
            staged=Path(arguments[arguments.index('--source-realization-config')+1])
            assert staged.read_bytes()==cfg.read_bytes()
    finally:
        if scratch.exists():shutil.rmtree(scratch)


@pytest.mark.parametrize('explicit_realization',[False,True])
def test_complete_coupled_control_delivery_streams_all_units(tmp_path,monkeypatch,explicit_realization):
    # Exercise all1001*168106 current values without allocating a multi-GB
    # trace file. Binary reader/group binding is separately tested above.
    root,_=fixture(tmp_path/'runtime')
    options={}
    if explicit_realization:options['source_realization_config']=add_realization(root)
    trace=root/'native_observation.bin';trace.write_bytes(b'fixture reader supplied by test')
    path=root/'native_report.json';native=json.loads(path.read_text())
    native.update(native_observation_sha256=sha256_file(trace),native_observation_bytes=trace.stat().st_size)
    write_json(path,native)
    proof=dict(status='PASS_COUPLED_EXTENSION_CA3_OBSERVATION_REPLAY',units=168106,duration_ms=1001,
        actual_current_trace_verified=True,native_internal_spike_counter_verified=True,event_multisets_equal=True,
        terminal_native_clock_independently_verified=True,observation_source_execution_equivalence_qualified=True,
        helper_implementation_sha256={'ca3_native_observation':sha256_file(Path(observer.__file__))},
        observed_report_sha256=sha256_file(path),native_observation_sha256=sha256_file(trace),
        current_units=observer.CURRENT_UNIT.copy(),coupled_extension=native['coupled_extension'],
        coupled_extension_config=native['coupled_extension_config'])
    if explicit_realization:
        proof.update(source_realization=native['source_realization'],realization_circuit_binding=native['realization_circuit_binding'])
    proof_path=tmp_path/'reconciliation.json';write_json(proof_path,proof)
    groups=[];offset=0
    for i,(name,n) in enumerate(native['populations'].items()):
        groups.append(observer.Group(i,offset,offset+n,0,i,offset,offset+n,1,name));offset+=n
    pyr=next(g for g in groups if g.name==readouts.PYRAMIDAL)
    pattern=np.asarray([int(x) for x in (root/'trainPat.csv').read_text().split()])
    commands=control.source_commands(pattern,pyr.start,coupled_extension=True)
    def stream(path,*,expected_stop_ms):
        assert path==trace and expected_stop_ms==1001
        counts=np.zeros(168106,dtype=np.int64)
        for t in range(expected_stop_ms):
            current=np.ones(168106,dtype='<f4')
            current[commands['unit_ids'][commands['time_ms']==t]]=10000
            yield observer.Interval(t,t+1,counts,current,current)
    monkeypatch.setattr(observer,'read_header',lambda path:(168106,groups))
    monkeypatch.setattr(observer,'iter_intervals',stream)
    result=control.analyze(root,proof_path,root/'main.cpp',root/'config.h',tmp_path/'control',coupled_extension=True,**options)
    assert result['status']=='VERIFIED_COUPLED_EXTENSION_CA3_CONTROL_DELIVERY'
    assert result['actual_current_values_checked']==168274106
    assert result['ordered_assignments_verified']==len(commands['unit_ids'])
    assert result['inhibitory_held_unit_ms_verified']==60*(168106-150754)
    assert result['current_units']['model_specific_physical_unit']=='pA'
    assert result['actual_currents_role']=='source_provenance_not_future_model_input'
    assert not result['scientific_fidelity_qualified']
    with pytest.raises(ValueError):
        control.analyze(root,proof_path,root/'main.cpp',root/'config.h',tmp_path/'default-rejected')


def test_explicit_realization_exact_process_replay_binding(tmp_path):
    from . import carlsim_retrieval_replay as process_replay

    left,_=fixture(tmp_path/'original');config=add_realization(left)
    right=tmp_path/'repeat';shutil.copytree(left,right)
    path=right/'native_report.json';native=json.loads(path.read_text())
    native.update(native_pid=2,slurm_job_id='fixture-realization-repeat',
                  started_utc='2026-09-29T01:00:02Z',finished_utc='2026-09-29T01:00:03Z')
    write_json(path,native);refresh_bindings(right)
    result=process_replay.compare(left,right,tmp_path/'repeated.json',coupled_extension=True,
                                  source_realization_config=config)
    assert result['status']=='PASS_COUPLED_EXTENSION_RETRIEVAL_REPLAY_AND_COUNTS'
    assert result['source_realization']==native['source_realization']
    assert result['time_or_id_tolerance']==0
    assert not result['scientific_fidelity_qualified']
    with pytest.raises(ValueError,match='EXPLICIT_CONFIGURATION_REQUIRED'):
        process_replay.compare(left,right,tmp_path/'missing-config.json',coupled_extension=True)
