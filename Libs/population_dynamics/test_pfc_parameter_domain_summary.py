"""Complete-manifest, signed contrast and field covariance contracts."""
import copy
from pathlib import Path
import json

import numpy as np
import pytest

from . import pfc_parameter_domain_summary as summary
from .test_pfc_parameter_domain_quality import plan


def manifest(tmp_path,value):
    p=tmp_path/'observations.json';p.write_text(json.dumps(value));return p


def test_complete3000_manifest_requires_each_identity_and_unique_directory(tmp_path):
    prepared=plan()
    value={r['identity']['episode_id']:dict(path=str(tmp_path/f'episode_{i}'),
        development_report_sha256='a'*64,observation_report_sha256='b'*64) for i,r in enumerate(prepared['rows'])}
    assert len(summary.observation_mapping(manifest(tmp_path,value),prepared))==3000
    subset=dict(value);subset.pop(next(iter(subset)))
    with pytest.raises(ValueError,match='COMPLETE3000_MANIFEST_REQUIRED'):
        summary.observation_mapping(manifest(tmp_path,subset),prepared)
    duplicate=dict(value);keys=list(duplicate);duplicate[keys[1]]=duplicate[keys[0]]
    with pytest.raises(ValueError,match='DUPLICATE_DIRECTORY'):
        summary.observation_mapping(manifest(tmp_path,duplicate),prepared)


def test_changed_report_hash_rejects_before_cell_reader(tmp_path,monkeypatch):
    (tmp_path/'development5ms').mkdir();(tmp_path/'observation1ms').mkdir()
    (tmp_path/'development5ms/report.json').write_text('{}');(tmp_path/'observation1ms/report.json').write_text('{}')
    monkeypatch.setattr(summary.quality,'load_cell',lambda *a:pytest.fail('changed hash entered source reader'))
    with pytest.raises(ValueError,match='BOUND_REPORT_CHANGED'):
        summary.load_observation(dict(path=str(tmp_path),development_report_sha256='a'*64,observation_report_sha256='b'*64),{})


def test_paired_negative_effects_undefined_angles_and_signed_covariance_stay_visible():
    loaded={}
    for k in range(10):
        cells=[]
        for b in range(3):
            value=float(k) # cue-minus-omission=-5; distractor interaction=0.
            phase=dict(circular_cue_error_deg='' if k==3 and b==1 else '-10',
                **{field:value for field in summary.quality.RATE_FIELDS})
            field_counts=np.tile(np.arange(5000,dtype=np.int64)[:,None]%(b+2),(1,17))
            field_counts[:,1]=3-field_counts[:,0] # Signed negative temporal covariance.
            report={'identity':{'split':'calibration'}}
            old={'phase_edges_ticks':np.array([0,400000]),'phase_unit_counts':np.array([[k+b,1]],dtype=np.int64)}
            arrays={'population_field_counts':field_counts,'unit_counts':np.array([k+b,1],dtype=np.int64),'vector_x':np.zeros(5000),'vector_y':np.zeros(5000)}
            cells.append((report,[phase],old,{},arrays))
        loaded[k]=cells
    phases,paired,contrasts,covariances,units=summary.summarize_circuit(86,loaded)
    negative=[r for r in contrasts if r['contrast']=='cue_minus_omission']
    assert negative and all(r['mean']==-5. for r in negative)
    undefined=[r for r in phases if r['condition_id']=='cue_antipodal' and r['metric']=='absolute_circular_cue_error_deg'][0]
    assert undefined['total_blocks']==3 and undefined['undefined_blocks']==1 and undefined['mean'] is None
    assert any(r['temporal_covariance_counts2']<0 for r in covariances)
    assert len(covariances)==10*17*17
    assert len(units)==60


def test_full3000_publish_and_last_cell_failure_leave_no_subset_output(tmp_path,monkeypatch):
    prepared=plan()
    for row in prepared['rows']:
        row['condition']={'condition_id':summary.quality.CONDITIONS[row['identity']['condition_index']]}
        row['config_path']='recipe.yaml';row['config_sha256']='fixture_hash'
    prep=tmp_path/'prepared';prep.mkdir();(prep/'execution.json').write_text(json.dumps(prepared));(prep/'recipe.yaml').write_text('fixture')
    mapping={r['identity']['episode_id']:dict(path=str(tmp_path/f'episode_{i}'),
        development_report_sha256='a'*64,observation_report_sha256='b'*64) for i,r in enumerate(prepared['rows'])}
    path=manifest(tmp_path,mapping)
    monkeypatch.setattr(summary, 'cpu_execution_provenance', lambda: dict(
        backend='slurm', run_id='123456', trial_id=None, hostname='fixture-host',
        slurm_job_id='123456', slurm_node='cpu04', allocated_cpus=1, allocated_gpus=0))
    monkeypatch.setattr(summary,'sha256_file',lambda p:'fixture_hash')
    called=[]
    def read(entry,row):
        called.append(row['identity']['episode_id']);b=row['identity']['block_index']
        report=dict(identity=row['identity'],initial_voltage={'file_sha256':str(b)},
            external_input={'final_rng_sha256':str(b),'driver':{}},source_archive='source',
            source_delivery_sha256='source_hash',source_cost={'elapsed_seconds':1.})
        return report,[],{}, {"observation_map":{"frozen":True}},{}
    monkeypatch.setattr(summary,'load_observation',read)
    def summarize(circuit,loaded):
        return ([{'circuit':circuit}],[{'circuit':circuit}],[{'circuit':circuit}],[{'circuit':circuit}],{})
    monkeypatch.setattr(summary,'summarize_circuit',summarize)
    monkeypatch.setattr(summary, 'summarize_phase_fluctuations', lambda *args:
        ([{'phase': 0}], [{'phase': 0}]))
    output=tmp_path/'complete'
    report=summary.aggregate(prep,path,output)
    assert len(called)==3000 and report['complete_development_episodes']==3000
    assert len(report['realized_paired_blocks'])==300
    assert report['production_episodes']==0 and not report['source_qualification_granted']
    assert not report['uncertainty_coverage_qualified'] and (output/'report.json').exists()
    def missing_last(entry,row):
        if row['task_index']==2999:raise ValueError('missing final acquired cell')
        return read(entry,row)
    monkeypatch.setattr(summary,'load_observation',missing_last)
    incomplete=tmp_path/'incomplete'
    with pytest.raises(ValueError,match='missing final acquired cell'):
        summary.aggregate(prep,path,incomplete)
    assert not incomplete.exists()


def fluctuation_fixture():
    sizes = np.ones(17, dtype=np.int64)
    counts = np.zeros((3, 8, 17), dtype=np.int64)
    counts[:, ::2, 0] = 1
    counts[:, 1::2, 1] = 1
    counts[1, :, 16] = np.arange(8) % 2
    counts[2, :, 16] = 1
    edges = np.array([0, 320, 640], dtype=np.int64)
    totals = np.stack([counts[:, :4].sum(axis=1), counts[:, 4:].sum(axis=1)], axis=1)
    return counts, totals, edges, sizes


def test_phase1ms_binary_diagonal_equals_full_unit_matrix_and_keeps_negative_covariance():
    counts, totals, edges, sizes = fluctuation_fixture()
    covariance, population = summary.phase_fluctuations(counts, totals, edges, sizes)
    assert len(covariance) == 2 * 17 * 17 and len(population) == 2 * 3 * 3
    assert any(row['temporal_covariance_counts2'] < 0 for row in covariance)
    for row in population:
        block = summary.BLOCKS.index(row['block_index'])
        lo, hi = ((0, 16) if row['population'] == 'excitatory' else
            (16, 17) if row['population'] == 'inhibitory' else (0, 17))
        first = row['start_tick'] // 80
        stop = row['stop_tick'] // 80
        x = counts[block, first:stop, lo:hi]
        unit_diagonal = x.var(axis=0, ddof=1).sum() / (hi - lo)**2 / .001**2
        variance = x.mean(axis=1).var(ddof=1) / .001**2
        assert row['temporal_diagonal_contribution_rate_Hz2'] == pytest.approx(unit_diagonal)
        assert row['temporal_population_mean_variance_rate_Hz2'] == pytest.approx(variance)
        assert row['temporal_ordered_offdiagonal_contribution_rate_Hz2'] == pytest.approx(variance - unit_diagonal)
        assert row['between_block_matched_time_diagonal_contribution_rate_Hz2'] is None
    assert any(row['temporal_ordered_offdiagonal_contribution_rate_Hz2'] < 0 for row in population)


def test_phase1ms_matched_time_covariance_does_not_count_task_waveform_as_history_noise():
    counts, totals, edges, sizes = fluctuation_fixture()
    covariance, _ = summary.phase_fluctuations(counts, totals, edges, sizes)
    e = [row for row in covariance if row['leading_field'] == row['trailing_field'] == 0]
    assert all(row['temporal_covariance_rate_Hz2'] > 0 for row in e)
    assert all(row['between_block_matched_time_covariance_rate_Hz2'] == 0 for row in e)
    assert all(row['blocks'] == 3 and row['time_bins'] == 4 for row in e)


def test_phase1ms_rejects_mixed_control_boundary_and_false_binary_counts():
    counts, totals, edges, sizes = fluctuation_fixture()
    with pytest.raises(ValueError, match='BOUNDARY_ALIGNED1MS'):
        summary.phase_fluctuations(counts, totals, np.array([0, 321, 640]), sizes)
    invalid = counts.copy(); invalid[0, 0, 0] = 2
    with pytest.raises(ValueError, match='BINARY1MS_COUNT_BOUND'):
        summary.phase_fluctuations(invalid, totals, edges, sizes)
    invalid = totals.copy(); invalid[0, 0, 0] += 1
    with pytest.raises(AssertionError):
        summary.phase_fluctuations(counts, invalid, edges, sizes)
