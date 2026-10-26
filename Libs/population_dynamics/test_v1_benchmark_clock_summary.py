import numpy as np
import pytest

from .v1_benchmark_clock_summary import summarize_block


def test_identical_clocks_preserve_silence_and_zero_effects():
    counts=np.zeros((10,3),dtype=np.int64);counts[0,1]=5;counts[:8,2]=2
    arrays, rows=summarize_block(counts,counts.copy(),0)
    for row in rows:
        assert row['units']==3 and row['paired_mean_difference']==0
    assert rows[0]['paired_defined_units']==2
    assert np.isnan(arrays['reference_direction_selectivity'][0])
    assert arrays['reference_direction_selectivity'][1]==1


def test_changed_defined_set_and_negative_unsigned_contrast_are_retained():
    left=np.zeros((10,2),dtype=np.uint64);right=left.copy()
    left[0,0]=7;right[0,1]=9;left[9,0]=3;right[9,1]=5
    arrays, rows=summarize_block(left,right,0)
    assert rows[0]['paired_defined_units']==0 and rows[0]['paired_mean_difference'] is None
    assert arrays['reference_low_minus_high_rate_hz'].tolist()==[-1.6,0.]
    assert arrays['fine_low_minus_high_rate_hz'].tolist()==[0.,-1.6]
    assert rows[2]['paired_defined_units']==2


def test_incomplete_or_noninteger_condition_counts_fail():
    for counts in (np.zeros((9,2),dtype=int),np.zeros((10,2),dtype=float),-np.ones((10,2),dtype=int)):
        with pytest.raises(ValueError,match='TEN_CONDITION_COUNTS_REQUIRED'):
            summarize_block(counts,counts.copy(),0)


def test_summary_cli_consumes_complete_blocks_and_rejects_rebound_case(tmp_path):
    """Synthetic readout fixtures test joins and output arithmetic, not simulation."""
    import csv
    import json
    import os
    import subprocess
    import sys
    from .provenance import sha256_file

    units=230924
    cases, jobs=[], []
    first_delivery=None
    for circuit, block in ((0,0),(1,1),(1,9),(2,1)):
        for condition in range(10):
            key=f'fixture_c{circuit}_b{block}_k{condition}'
            block_id=f'fixture_c{circuit}_b{block}'
            identity=dict(episode_id=key,circuit_id=f'fixture_c{circuit}',block_id=block_id,
                          block_index=block,condition_index=condition)
            directory=tmp_path/key;analysis=directory/'analysis';analysis.mkdir(parents=True)
            reference=np.zeros(units,dtype=np.int64);fine=reference.copy()
            reference[0]=condition+1 if condition<8 else (0 if condition==8 else 2)
            fine[0]=2*reference[0] if condition<8 else (0 if condition==8 else 3)
            fine[1]=1
            command_reference=reference.copy();command_reference[0]=max(0,command_reference[0]-1)
            command_fine=fine.copy();command_fine[0]=max(0,command_fine[0]-1)
            np.savez_compressed(analysis/'per_unit.npz',source_id=np.arange(units,dtype=np.int64),
                reference_source_response_counts=reference,fine_source_response_counts=fine,
                reference_command_response_counts=command_reference,fine_command_response_counts=command_fine)
            job=str(100+len(cases));reader=str(200+len(cases));reference_path='/fixture/reference/'+key;fine_path='/fixture/fine/'+key
            report=dict(status='V1_CLOCK_COMPARISON_COMPLETE_DEVELOPMENT',exit_code=0,mode='full',
                source_bindings_and_requested_inputs_exact=True,physical_parameters_and_initial_states_exact=True,
                reference_resolution_ms=.25,fine_resolution_ms=.125,
                inputs=dict(reference=dict(consumed={'delivery.json':dict(sha256='fixture-'+key)}),fine=dict(job_id=job)),
                all_units=dict(units=units,reference_spikes=int(reference.sum()),fine_spikes=int(fine.sum()),
                    spike_count_difference=int(fine.sum()-reference.sum()),
                    changed_spike_count_units=int(np.count_nonzero(fine!=reference)),
                    reference_silent_units=int(np.count_nonzero(reference==0)),fine_silent_units=int(np.count_nonzero(fine==0))),
                artifacts={'per_unit.npz':dict(bytes=(analysis/'per_unit.npz').stat().st_size,sha256=sha256_file(analysis/'per_unit.npz'))})
            (analysis/'report.json').write_text(json.dumps(report))
            delivery=dict(exit_code=0,job_id=reader,reference_archive=reference_path,fine_archive=fine_path,
                          report_sha256=sha256_file(analysis/'report.json'))
            (directory/'delivery.json').write_text(json.dumps(delivery))
            if first_delivery is None:first_delivery=directory/'delivery.json'
            cases.append(dict(identity=identity,reference=reference_path,delivery_sha256='fixture-'+key))
            jobs.append(dict(identity=identity,comparison_job=reader,native_job=job,
                             comparison_output=str(directory),native_archive=fine_path))
    plan=tmp_path/'cases.json';plan.write_text(json.dumps(dict(full_units=units,cases=cases)))
    submissions=tmp_path/'submissions.json';submissions.write_text(json.dumps(dict(cases=jobs)))
    command=[sys.executable,'-B','-m','population_dynamics.v1_benchmark_clock_summary',
             '--cases',str(plan),'--submissions',str(submissions)]
    result=subprocess.run(command+['--output',str(tmp_path/'summary')],capture_output=True,text=True,env=os.environ.copy())
    assert result.returncode==0,result.stderr
    report=json.loads((tmp_path/'summary/report.json').read_text())
    assert (report['complete_pairs'],report['complete_blocks'],report['circuit_identities'])==(40,4,3)
    assert report['production_episodes']==0 and report['scientific_qualification'] is False
    with (tmp_path/'summary/task_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==40
    row=next(row for row in rows if row['block_id']=='fixture_c1_b1' and row['window']=='source_response'
             and row['metric']=='low_minus_high_rate_hz')
    assert float(row['paired_mean_difference'])==pytest.approx(-1/(2.5*units))
    row=next(row for row in rows if row['block_id']=='fixture_c0_b0' and row['window']=='source_response'
             and row['metric']=='direction_selectivity')
    assert (int(row['reference_defined_units']),int(row['fine_defined_units']),int(row['paired_defined_units']))==(1,2,1)
    changed=json.loads(first_delivery.read_text());changed['reference_archive']='/fixture/wrong_reference';first_delivery.write_text(json.dumps(changed))
    rejected=subprocess.run(command+['--output',str(tmp_path/'rejected')],capture_output=True,text=True,env=os.environ.copy())
    assert rejected.returncode!=0 and 'BOUND_CASE_COMPARISON_REQUIRED' in rejected.stderr
    rejected_report=json.loads((tmp_path/'rejected/report.json').read_text())
    assert rejected_report['exit_code']==1 and 'error' in rejected_report
