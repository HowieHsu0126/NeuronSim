"""Complete synthetic clocks and real-format binding fixtures; no native outcomes."""
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from . import ca3_cue_contrast as contrast
from . import ca3_native_observation as observer
from . import ca3_host_current_replay as host
from .ca3_native_observation_replay import BoundRun
from .carlsim_records import SPIKE_HEADER, SPIKE_SIGNATURE
from .provenance import sha256_file
from .test_ca3_host_current_replay import engine
from .test_ca3_coupled_readouts import fixture as coupled_fixture


@pytest.fixture(scope='module')
def numerics(engine):
    groups, offset = [], 0
    for i, name in enumerate(host.SOURCE_ORDER):
        size = 1650 if name == contrast.PYRAMIDAL else 2
        groups.append(observer.Group(i, offset, offset+size, 0, i, offset, offset+size, 1, name))
        offset += size
    empty = dict(time_ms=np.empty(0, dtype=np.int32), unit_ids=np.empty(0, dtype=np.int64))
    raw = subprocess.check_output([str(engine)], input=host.engine_input(groups, empty))
    currents = np.frombuffer(raw, dtype='<f4').reshape(1001, offset).copy()
    # Deliberately unordered input, duplicated pre-cue events, and both signs of effect.
    source = np.array([[1000,1100],[519,276],[0,1650],[539,551],[520,550],[540,826],[519,276]], dtype=np.int64)
    omission = np.array([[519,276],[1000,1651],[520,550],[0,1650],[519,276]], dtype=np.int64)
    return tuple(groups), currents, source, omission


def write_trace(path, groups, currents, events, *, stop_ms=1001):
    units = currents.shape[1]
    counts = np.zeros((1001, units), dtype='<i4')
    np.add.at(counts, (events[:,0], events[:,1]), 1)
    with path.open('wb') as stream:
        stream.write(observer.HEADER.pack(observer.MAGIC, len(groups), units))
        for g in groups:
            name = g.name.encode()
            stream.write(observer.GROUP.pack(g.group_id,g.start,g.stop,g.net_id,g.local_group,
                g.local_start,g.local_stop,g.gpu,len(name))+name)
        for t in range(stop_ms):
            for kind in (b'R', b'A', b'C'):
                for g in groups:
                    values = counts[t,g.start:g.stop] if kind == b'C' else currents[t,g.start:g.stop]
                    stream.write(observer.RECORD.pack(kind,t,t if kind==b'R' else t+1,g.group_id,g.units))
                    stream.write(values.tobytes())
        stream.write(observer.RECORD.pack(b'E',stop_ms,stop_ms,-1,0))


def make_run(root, groups, events, currents=None, *, trace_events=None, stop_ms=1001):
    (root/'native_run/results').mkdir(parents=True)
    units = sum(g.units for g in groups)
    totals = np.zeros(units, dtype=np.int64)
    populations = np.zeros((1001,len(groups)), dtype=np.int64)
    rows, hashes = [], {}
    for g in groups:
        selected = events[(events[:,1]>=g.start)&(events[:,1]<g.stop)].copy()
        relative = 'native_run/results/spk_'+g.name+'.dat'
        with (root/relative).open('wb') as stream:
            stream.write(SPIKE_HEADER.pack(SPIKE_SIGNATURE,.2,g.units,1,1))
            if len(selected):
                local = selected.copy(); local[:,1] -= g.start
                stream.write(local.astype('<i4').tobytes())
                for t, unit in selected:
                    totals[unit] += 1; populations[t,g.group_id] += 1
        rows.append(dict(group=g.name,relative_path=relative,sha256=sha256_file(root/relative),
            records=len(selected),first_event_ms=int(selected[:,0].min()) if len(selected) else None,
            last_event_ms=int(selected[:,0].max()) if len(selected) else None))
    if currents is not None:
        write_trace(root/'native_observation.bin',groups,currents,
                    events if trace_events is None else trace_events,stop_ms=stop_ms)
    saved = dict(unit_ids=np.arange(units),unit_counts=totals,population_counts_per_ms=populations,
                 group_names=np.array([g.name for g in groups]))
    return BoundRun(root,{},dict(core_neurons=units),dict(files=rows,records=len(events)),
                    groups,saved,hashes,{}, {},dict(scope='synthetic_numeric_fixture'))


def runs(tmp_path, numerics, *, changed_currents=False, counter_time_shift=False, stop_ms=1001):
    groups, currents, source, omission = numerics
    left = make_run(tmp_path/'source', groups, source)
    effective = currents.copy()
    if changed_currents:
        effective[-1,-1] = np.nextafter(effective[-1,-1],np.float32(np.inf))
    recorded_events = omission.copy()
    if counter_time_shift:
        recorded_events[2,0] += 1  # identical per-unit total, different native millisecond
    right = make_run(tmp_path/'omission',groups,recorded_events,effective,
                     trace_events=omission,stop_ms=stop_ms)
    return left, right


def test_complete_numeric_cli_keeps_silent_units_duplicates_and_final_ms(tmp_path, monkeypatch, numerics, engine):
    left, right = runs(tmp_path,numerics)
    # Only the unavailable scientific archive binding is supplied by this fixture.
    # Native-format events, counters, every current, projection and serialization run.
    monkeypatch.setattr(contrast,'bind_pair',lambda *args:(left,right,engine,{'fixture':True}))
    output = tmp_path/'result'
    monkeypatch.setattr('sys.argv',['contrast','--source',str(left.root),'--omission',str(right.root),
        '--source-analysis','fixture','--source-rng','fixture','--output',str(output)])
    contrast.main()
    report = json.loads((output/'report.json').read_text())
    assert report['status']=='MEASURED_CA3_CUE_OMISSION_CONTRAST'
    assert (report['source_events'],report['omission_events'],report['compared_pre_cue_events'])==(7,5,3)
    assert report['omission_current_checks']['current_values_compared']==1664*1001
    assert report['omission_counter_checks']['unit_millisecond_counts_compared']==1664*1001
    assert report['phase_rows']==98
    assert report['scientific_fidelity_qualified'] is False
    assert report['inferential_statistics_performed'] is False
    with np.load(output/'counts.npz',allow_pickle=False) as values:
        assert len(values['unit_ids'])==1664
        assert int(values['source_unit_totals'].sum())==7
        assert int(values['omission_unit_totals'].sum())==5
        assert list(values['field_unit_counts'][-6:])==[276,274]*3
        assert len(values['time_ms'])==1001
    with (output/'phase_counts.csv').open() as stream:rows=list(csv.DictReader(stream))
    terminal=next(r for r in rows if r['phase']=='background_after_cue_3' and r['field']=='CA3_Axo_Axonic')
    assert (terminal['start_ms'],terminal['stop_ms'],terminal['source_minus_omission_count'])==('1000','1001','-1')
    assert float(terminal['source_minus_omission_hz'])==-500.0
    second=next(r for r in rows if r['phase']=='cue_1' and r['field']=='assembly_2_source_eligible')
    assert int(second['source_minus_omission_count'])==1
    assert float(second['source_minus_omission_hz'])==pytest.approx(1000/(276*20))
    full=next(r for r in report['whole_schedule_fields'] if r['field']=='CA3_Axo_Axonic')
    assert full['source_minus_omission_hz']==pytest.approx(-1000/(2*1001))


@pytest.mark.parametrize('failure,expected',[('current','RNG_REPLAY_DIFFERENT'),
    ('counter','PER_UNIT_MILLISECOND_COUNTER_MISMATCH'),('last_ms','DURATION_OR_FOOTER_INVALID')])
def test_incomplete_or_different_traces_cannot_leave_success(tmp_path,monkeypatch,numerics,engine,failure,expected):
    left,right=runs(tmp_path,numerics,changed_currents=failure=='current',counter_time_shift=failure=='counter',
                    stop_ms=1000 if failure=='last_ms' else 1001)
    monkeypatch.setattr(contrast,'bind_pair',lambda *args:(left,right,engine,{'fixture':True}))
    with pytest.raises(ValueError,match=expected):
        contrast.analyze(left.root,right.root,'fixture','fixture',tmp_path/'result')
    report=json.loads((tmp_path/'result/report.json').read_text())
    assert report['status']=='FAILED' and report['scientific_fidelity_qualified'] is False
    if failure=='counter':
        assert report['omission_counter_checks']['first_counter_mismatch']['start_ms']==520
    if failure=='current':
        assert report['omission_current_checks']['mismatched_values']==1
        assert report['omission_current_checks']['first_mismatches'][0]['time_ms']==1000


def test_prefix_comparison_is_order_invariant_but_preserves_multiplicity(numerics):
    _,_,source,omission=numerics
    assert contrast.compare_prefix(source,omission)==3
    with pytest.raises(ValueError,match='PRE_CUE_EVENT_MULTISET'):
        contrast.compare_prefix(source,omission[:-1])
    assert contrast.compare_prefix(np.array([[520,1]]),np.empty((0,2),dtype=int))==0
    with pytest.raises(ValueError,match='PRE_CUE_EVENT_MULTISET'):
        contrast.compare_prefix(np.array([[519,1]]),np.empty((0,2),dtype=int))


def test_field_bounds_and_fixed_source_subsets(numerics):
    groups,_,source,_=numerics
    names,sizes,counts=contrast.field_counts(source,groups)
    assert counts[519,names.index('assembly_1_source_never_boosted')]==2
    assert counts[519,names.index('assembly_1_source_eligible')]==0
    assert sizes[names.index('assembly_1_source_never_boosted')]==274
    for bad in (np.array([[1001,0]]),np.array([[0,1664]]),np.array([[0.5,1]])):
        with pytest.raises(ValueError,match='NATIVE_EVENT_BOUNDS'):
            contrast.field_counts(bad,groups)


def metadata_fixture(root):
    """The reused two-ms fixture tests binding only, never full-clock acceptance."""
    coupled_fixture(root,observed=True,conductance_memory=True)
    build=json.loads((root/'build_report.json').read_text())
    native=json.loads((root/'native_report.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    names=[row['path'] for row in manifest['files']]
    for name in contrast.LIBRARIES:
        path=root/name;path.parent.mkdir(exist_ok=True)
        path.write_bytes(('fixture library '+name).encode())
        if name not in names:names.append(name)
    manifest.update(scope='coupled_extension_retrieval_runtime',native_observation_enabled=True)

    def refresh():
        (root/'build_report.json').write_text(json.dumps(build))
        manifest['files']=[dict(path=n,bytes=(root/n).stat().st_size,sha256=sha256_file(root/n)) for n in names]
        (root/'manifest.json').write_text(json.dumps(manifest))
        native['runtime_manifest_sha256']=sha256_file(root/'manifest.json')
        (root/'native_report.json').write_text(json.dumps(native))
        recording=json.loads((root/'observations/recording.json').read_text())
        recording['native_report_sha256']=sha256_file(root/'native_report.json')
        (root/'observations/recording.json').write_text(json.dumps(recording))
        all_names=set(names)|{'manifest.json','native_report.json','native_observation.bin',
                             'observations/recording.json','observations/population_counts.npz'}
        all_names.update(row['relative_path'] for row in recording['files'])
        rows=[dict(path=n,bytes=(root/n).stat().st_size,sha256=sha256_file(root/n)) for n in sorted(all_names)]
        (root/'delivery.json').write_text(json.dumps(dict(status='COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED',
            exit_code=0,copied_files_verified=True,files=rows)))
    refresh()
    return build,native,manifest,refresh


def test_binder_rejects_short_recording_and_extra_configuration_change(tmp_path):
    root=tmp_path/'metadata';build,native,_,refresh=metadata_fixture(root)
    with pytest.raises(ValueError,match='COMPLETED_NATIVE_OBSERVATION'):
        contrast.bind_run(root,omission=False)
    # A self-reported full clock does not make these two-ms data usable by analyze.
    native['observed_native_duration_ms']=1001;refresh()
    bound=contrast.bind_run(root,omission=False)
    assert bound.native['core_neurons']==168106
    for record in (build,native):
        row=next(row for row in record['source_file_byte_checks'] if row['path']=='generateCONFIGStateSTP.h')
        row['staged_sha256']='0'*64
    refresh()
    with pytest.raises(ValueError,match='BOUND_SOURCE_PROGRAM_REQUIRED'):
        contrast.bind_run(root,omission=False)


@pytest.mark.parametrize('change,expected',[('ids','UNIT_IDENTITIES'),('weights','INITIALIZATION_OR_BACKEND'),
                                         ('library','BACKEND_LIBRARY_BYTES')])
def test_pair_rejects_identity_or_backend_change_before_using_proofs(tmp_path,monkeypatch,numerics,change,expected):
    left,right=runs(tmp_path,numerics)
    for run in (left,right):
        run.native.update({name:'a'*64 for name in contrast.INPUT_FIELDS})
        run.native.update(coupled_extension={},coupled_extension_config=dict(sha256='a'*64,resolved={'population_multiplier':2},path=str(run.root)),
            native_observation_overlay={},observer_implementation_sha256='reader',
            conductance_memory_overlay={},conductance_memory_implementation_sha256='memory')
        run.hashes.update({name:'b'*64 for name in contrast.LIBRARIES})
        run.hashes.update({'manifest.json':'c'*64,'build_report.json':'d'*64})
    paired=dict(backend_library_bytes_equal=True,manifest_sha256='c'*64,source_build_sha256='d'*64,
                libraries=[dict(path=name,sha256='b'*64) for name in contrast.LIBRARIES])
    (right.root/'manifest.json').write_text(json.dumps(dict(paired_backend=paired)))
    if change=='ids':right.groups=right.groups[::-1]
    elif change=='weights':right.native['normalized_input_sha256']='e'*64
    else:right.hashes[contrast.LIBRARIES[0]]='e'*64
    monkeypatch.setattr(contrast,'bind_run',lambda root,omission:right if omission else left)
    with pytest.raises(ValueError,match=expected):
        contrast.bind_pair(left.root,right.root,tmp_path/'unused-analysis',tmp_path/'unused-rng')
