"""Tiny indexed HDF5 source and fake HTTP responses; no real network access."""
import io
import json
import hashlib
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import requests

from .v1_sonata_witness import RangeFile, query_indexed


class Body(io.BytesIO):
    def __init__(self, value):
        super().__init__(value)
        self.bytes_read = 0

    def read(self, size=-1):
        value = super().read(size)
        self.bytes_read += len(value)
        return value


class Response:
    def __init__(self, body, start, stop, total, mode):
        self.status_code = 410 if mode == 'gone' else 200 if mode == 'ignored_range' else 206
        self.headers = {'Content-Range': f'bytes {start}-{stop}/{total}',
                        'Content-Length': str(stop-start+1), 'ETag': 'changed' if mode == 'changed_etag' else 'source-etag'}
        self.raw = Body(body)
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError('410 Gone')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed = True
        self.raw.close()


class Session:
    def __init__(self, payload, mode='exact'):
        self.payload, self.mode = payload, mode
        self.calls, self.responses = [], []
        self.closed = False
        self.trust_env = True
        self.proxies = {}

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        start, stop = map(int, kwargs['headers']['Range'].removeprefix('bytes=').split('-'))
        body = self.payload if self.mode == 'ignored_range' else self.payload[start:stop+1]
        response = Response(body, start, stop, len(self.payload), self.mode)
        self.responses.append(response)
        return response

    def close(self):
        self.closed = True


def remote(payload, **options):
    session = Session(payload, options.pop('mode', 'exact'))
    reader = RangeFile({'resolved_url': 'https://synthetic.example/source.h5',
                        'expected_bytes': len(payload), 'etag': 'source-etag'},
                       'http://127.0.0.1:18080', session=session, **options)
    return reader, session


class SonataWitnessTests(unittest.TestCase):
    def test_real_hdf5_index_retains_all_duplicate_rows_and_group_properties(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-sonata-index-') as temporary:
            path = Path(temporary)/'fixture.h5'
            with h5py.File(path, 'w') as out:
                group = out.create_group('edges/v1_to_v1')
                values = {'source_node_id': [5, 7, 5, 5, 8, 5, 9],
                          'target_node_id': [1, 1, 1, 3, 2, 1, 0],
                          'edge_type_id': [10, 11, 12, 13, 14, 15, 16],
                          'edge_group_id': [0, 0, 0, 0, 1, 0, 1],
                          'edge_group_index': [0, 1, 2, 3, 0, 4, 1]}
                for name, vector in values.items():
                    group.create_dataset(name, data=np.asarray(vector, dtype=np.uint64))
                properties = group.create_group('0')
                properties.create_dataset('nsyns', data=np.array([3, 4, 7, 9, 11], dtype=np.uint32))
                properties.create_dataset('syn_weight', data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
                properties.create_dataset('weight_sigma', data=np.full(5, 30.0))
                properties.create_dataset('delay', data=np.full(5, 1.8))
                properties = group.create_group('1')
                properties.create_dataset('nsyns', data=np.array([2, 5], dtype=np.uint32))
                properties.attrs['delay'] = 1.2
                index = group.create_group('indices/target_to_source')
                index.create_dataset('node_id_to_range', data=np.array([[0, 1], [1, 3], [3, 4], [4, 5], [5, 5]], dtype=np.uint64))
                index.create_dataset('range_to_edge_id', data=np.array([[6, 7], [0, 3], [5, 6], [4, 5], [3, 4]], dtype=np.uint64))
            reader, session = remote(path.read_bytes())
            report = {'include_target_context': True}
            selectors = [dict(name='duplicates', source_id=5, target_id=1),
                         dict(name='other_group', source_id=8, target_id=2),
                         dict(name='empty_target', source_id=123, target_id=4)]
            with reader:
                with io.BufferedReader(reader, buffer_size=65536) as buffered:
                    with h5py.File(buffered, 'r') as handle:
                        query_indexed(handle, selectors, report)
        self.assertEqual(report['source_edge_denominator'], 7)
        self.assertEqual(report['completed_selectors'], 3)
        self.assertEqual([row['global_row_id'] for row in report['cases'][0]['matching_rows']], [0, 2, 5])
        self.assertEqual([row['properties']['nsyns']['value'] for row in report['cases'][0]['matching_rows']], [3, 7, 11])
        self.assertEqual(report['cases'][0]['scanned_source_rows'], 4)
        context = report['cases'][0]['target_context']
        self.assertTrue(context['complete'])
        self.assertEqual(context['indexed_incoming_rows'], 4)
        self.assertEqual(context['edge_type_id_counts'], {'10': 1, '11': 1, '12': 1, '15': 1})
        self.assertEqual([row['global_row_id'] for row in context['incoming_rows']], [0, 1, 2, 5])
        self.assertEqual(context['incoming_rows'][1]['source_node_id'], 7)
        self.assertEqual(report['cases'][2]['target_context']['indexed_incoming_rows'], 0)
        other = report['cases'][1]['matching_rows'][0]
        self.assertEqual(other['properties']['syn_weight']['status'], 'MISSING')
        self.assertEqual(other['properties']['delay']['value'], 1.2)
        self.assertTrue(report['cases'][2]['matches_complete'])
        self.assertFalse(report['cases'][2]['source_pair_present'])
        self.assertTrue(session.closed)
        self.assertTrue(all(response.closed for response in session.responses))
        self.assertFalse(session.trust_env)
        for _, call in session.calls:
            self.assertTrue(call['verify'])
            self.assertEqual(call['headers']['If-Match'], 'source-etag')
            self.assertTrue(call['stream'])
            self.assertFalse(call['allow_redirects'])
            self.assertEqual(call['proxies']['https'], 'http://127.0.0.1:18080')

    def test_context_checks_unmatched_rows_target_ids(self):
        with h5py.File(io.BytesIO(), 'w') as handle:
            group = handle.create_group('edges/v1_to_v1')
            for key, values in dict(source_node_id=[5, 6], target_node_id=[0, 1],
                    edge_type_id=[10, 11], edge_group_id=[0, 0], edge_group_index=[0, 1]).items():
                group[key] = np.asarray(values, dtype=np.uint64)
            group.create_group('0')['nsyns'] = np.asarray([3, 4], dtype=np.uint32)
            index = group.create_group('indices/target_to_source')
            index['node_id_to_range'] = np.asarray([[0, 1]], dtype=np.uint64)
            index['range_to_edge_id'] = np.asarray([[0, 2]], dtype=np.uint64)
            selector = [dict(name='one_pair', source_id=5, target_id=0)]
            query_indexed(handle, selector, {})
            with self.assertRaisesRegex(ValueError, 'CONTEXT_TARGET_OR_IDENTIFIER_INVALID'):
                query_indexed(handle, selector, {'include_target_context': True})

    def test_pair_only_resume_cannot_satisfy_target_context_request(self):
        from .v1_sonata_witness import _resume_cases
        with tempfile.TemporaryDirectory(prefix='sonata-context-scope-') as temp:
            path = Path(temp) / 'prior.json'
            path.write_text(json.dumps({'completed_selectors': 3, 'include_target_context': False}))
            with self.assertRaisesRegex(ValueError, 'RESUME_TARGET_CONTEXT_SCOPE_MISMATCH'):
                _resume_cases(path, {'include_target_context': True}, [])

    def test_historical_index_spelling_and_ambiguous_index_rejection(self):
        for names in [('indices',), ('indicies',), ('indices', 'indicies')]:
            with self.subTest(names=names), h5py.File(io.BytesIO(), 'w') as handle:
                group=handle.create_group('edges/v1_to_v1')
                for name,values in {'source_node_id':[5,5], 'target_node_id':[0,0],
                        'edge_type_id':[10,11], 'edge_group_id':[0,0],
                        'edge_group_index':[0,1]}.items():
                    group.create_dataset(name,data=np.asarray(values,dtype=np.uint64))
                group.create_group('0').create_dataset('nsyns',data=np.array([3,7],dtype=np.uint32))
                for name in names:
                    index=group.create_group(name+'/target_to_source')
                    index.create_dataset('node_id_to_range',data=np.array([[0,1]],dtype=np.uint64))
                    index.create_dataset('range_to_edge_id',data=np.array([[0,2]],dtype=np.uint64))
                report={}
                if len(names)==2:
                    with self.assertRaisesRegex(ValueError,'AMBIGUOUS'):
                        query_indexed(handle,[dict(name='source',source_id=5,target_id=0)],report)
                else:
                    query_indexed(handle,[dict(name='source',source_id=5,target_id=0)],report)
                    self.assertEqual(report['target_index_path'],names[0]+'/target_to_source')
                    self.assertEqual([r['global_row_id'] for r in report['cases'][0]['matching_rows']],[0,1])

    def test_changed_etag_or_ignored_range_fails_before_reading_any_body(self):
        for mode in ('changed_etag', 'ignored_range'):
            with self.subTest(mode=mode):
                reader, session = remote(b'0123456789abcdef', mode=mode)
                with reader:
                    with self.assertRaises(ValueError):
                        reader.read(4)
                    self.assertEqual(reader.body_bytes, 0)
                    self.assertIsInstance(reader.failed, str)
                    self.assertEqual(session.responses[0].raw.bytes_read, 0)
                    self.assertTrue(session.responses[0].closed)
                    with self.assertRaises(OSError):
                        reader.read(4)
                    self.assertEqual(len(session.calls), 1)
                self.assertTrue(session.closed)

    def test_initial_http_failure_exits_cleanly_without_constructing_hdf5(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-sonata-expired-') as temporary:
            root=Path(temporary)
            comparison=root/'comparison.json';comparison.write_text(json.dumps({'comparison':{'tuples':7}}))
            witness=root/'witness.json';witness.write_text(json.dumps({
                'status':'SOURCE_WEIGHT_ARITHMETIC_DIAGNOSTIC_COMPLETE','exit_code':0,
                'comparison_report_sha256':hashlib.sha256(comparison.read_bytes()).hexdigest(),
                'cases':[dict(name=str(i),source_id=i,target_id=i+1,raw_group=0,raw_offset=i) for i in range(3)]}))
            source=root/'source.json';source.write_text(json.dumps({'resolved_url':'https://synthetic.example/source.h5',
                'expected_bytes':131072,'etag':'source-etag'}))
            script="""
from pathlib import Path
import sys
from unittest.mock import patch
from population_dynamics import v1_sonata_witness as witness
from population_dynamics.test_v1_sonata_witness import Session
root=Path(sys.argv[1]);session=Session(b'x'*131072,mode='gone')
with patch.object(witness.requests,'Session',return_value=session), patch.object(witness.h5py,'File') as hdf5:
 try:
  witness.run_query(root/'source.json',root/'witness.json',root/'comparison.json',
                    'http://127.0.0.1:18080',root/'output',max_bytes=65536,max_seconds=10)
 except Exception:
  hdf5.assert_not_called()
  assert session.closed and len(session.responses)==1 and session.responses[0].closed
  assert session.responses[0].raw.bytes_read==0
  raise
"""
            process=subprocess.run([sys.executable,'-B','-c',script,str(root)],capture_output=True,text=True,timeout=20)
            self.assertEqual(process.returncode,1,process.stderr)
            report=json.loads((root/'output/report.json').read_text())
            self.assertEqual(report['error_type'],'HTTPError')
            self.assertIn('410',report['error'])
            self.assertEqual(report['http']['requests'],1)
            self.assertEqual(report['http']['response_body_bytes'],0)
            self.assertEqual(report['completed_selectors'],0)
            self.assertFalse(report['scientific_qualification'])

    def test_published_entry_redirect_preserves_range_identity_and_counts_its_body(self):
        class Adapter(requests.adapters.BaseAdapter):
            def __init__(self, redirect_size=2, destination='https://synthetic.example/source.h5'):
                self.calls=[];self.redirect_size=redirect_size;self.destination=destination;self.closed=False
            def send(self, request, **kwargs):
                self.calls.append((request, kwargs))
                response=requests.Response();response.url=request.url;response.request=request
                if '/published?' in request.url:
                    response.status_code=302
                    response.headers.update({'Location':self.destination,'Content-Length':str(self.redirect_size)})
                    response.raw=Body(b'go')
                else:
                    start,stop=map(int,request.headers['Range'][6:].split('-'))
                    response.status_code=206
                    response.headers.update({'Content-Range':f'bytes {start}-{stop}/16',
                                      'Content-Length':str(stop-start+1),'ETag':'source-etag'})
                    response.raw=Body(b'0123456789abcdef'[start:stop+1])
                return response
            def close(self):
                self.closed=True
        source={'resolved_url':'https://synthetic.example/expired',
                'published_url':'https://synthetic.example/published?dl=0',
                'expected_bytes':16,'etag':'source-etag'}
        session=requests.Session();adapter=Adapter();session.mount('https://',adapter)
        with RangeFile(source,'http://127.0.0.1:18080',max_bytes=12,session=session) as reader:
            self.assertEqual(reader.read(4),b'0123')
            self.assertEqual(reader.read(4),b'4567')
            self.assertEqual(reader.body_bytes,12)
            self.assertEqual(reader.redirect_body_bytes,4)
            self.assertEqual(reader.redirects,2)
            self.assertEqual(len(adapter.calls),4)
            self.assertNotEqual(adapter.calls[0][0].url,adapter.calls[2][0].url)
            for request,options in adapter.calls:
                self.assertTrue(options['verify'])
                self.assertEqual(request.headers['If-Match'],'source-etag')
        self.assertTrue(adapter.closed)
        for size,destination,budget in [(5000,'https://synthetic.example/source.h5',16),
                                        (2,'http://synthetic.example/source.h5',16),
                                        (2,'https://synthetic.example/source.h5',4)]:
            with self.subTest(size=size,destination=destination,budget=budget):
                session=requests.Session();adapter=Adapter(size,destination);session.mount('https://',adapter)
                with RangeFile(source,'http://127.0.0.1:18080',max_bytes=budget,session=session) as reader:
                    with self.assertRaisesRegex(ValueError,'REDIRECT'):
                        reader.read(4)
                    self.assertLessEqual(reader.body_bytes,budget)
                    self.assertLessEqual(len(adapter.calls),2)

    def test_resume_queries_only_unfinished_pair_and_rejects_changed_source(self):
        from unittest.mock import patch
        from . import v1_sonata_witness as module
        with tempfile.TemporaryDirectory(prefix='synthetic-sonata-resume-') as temporary:
            root=Path(temporary);path=root/'source.h5'
            with h5py.File(path,'w') as handle:
                group=handle.create_group('edges/v1_to_v1')
                for name,values in {'source_node_id':[5,5,5], 'target_node_id':[0,1,2],
                        'edge_type_id':[10,11,12], 'edge_group_id':[0,0,0],
                        'edge_group_index':[0,1,2]}.items():
                    group.create_dataset(name,data=np.asarray(values,dtype=np.uint64))
                group.create_group('0').create_dataset('nsyns',data=np.array([3,6,7],dtype=np.uint32))
                index=group.create_group('indicies/target_to_source')
                for name in ['node_id_to_range','range_to_edge_id']:
                    index.create_dataset(name,data=np.array([[0,1],[1,2],[2,3]],dtype=np.uint64))
                handle.create_dataset('padding',data=np.zeros(2*1024*1024,dtype=np.uint8))
            payload=path.read_bytes();source=root/'download.json'
            source.write_text(json.dumps(dict(resolved_url='https://synthetic.example/source.h5',
                expected_bytes=len(payload),etag='source-etag')))
            comparison=root/'comparison.json';comparison.write_text(json.dumps({'comparison':{'tuples':3}}))
            comparison_hash=hashlib.sha256(comparison.read_bytes()).hexdigest()
            selectors=[dict(name=str(i),source_id=5,target_id=i,raw_group=0,raw_offset=i) for i in range(3)]
            witness=root/'witness.json';witness.write_text(json.dumps(dict(
                status='SOURCE_WEIGHT_ARITHMETIC_DIAGNOSTIC_COMPLETE',exit_code=0,
                comparison_report_sha256=comparison_hash,cases=selectors)))
            prior=dict(comparison_report_sha256=comparison_hash,
                witness_report_sha256=hashlib.sha256(witness.read_bytes()).hexdigest(),
                source_expected_bytes=len(payload),source_etag='source-etag',population=module.POPULATION,
                selectors=selectors,source_edge_denominator=3,completed_selectors=2,
                cases=[dict(case,status='COMPLETE',matches_complete=True,matching_row_count=1,
                    source_pair_present=True,matching_rows=[dict(global_row_id=i)])
                    for i,case in enumerate(selectors[:2])])
            resume=root/'resume.json';resume.write_text(json.dumps(prior));session=Session(payload)
            with patch.object(module.requests,'Session',return_value=session), \
                    patch.object(module,'query_indexed',wraps=module.query_indexed) as query:
                report=module.run_query(source,witness,comparison,'http://127.0.0.1:18080',
                    root/'output',max_bytes=1024*1024,max_seconds=10,resume_report=resume)
                self.assertEqual(query.call_args.args[1],[selectors[2]])
            self.assertEqual(report['completed_selectors'],3)
            self.assertEqual(report['reused_complete_selectors'],['0','1'])
            self.assertEqual([c['name'] for c in report['cases']],['0','1','2'])
            self.assertEqual(report['cases'][2]['matching_rows'][0]['properties']['nsyns']['value'],7)
            for key,value in [('source_etag','different'),('witness_report_sha256','different'),
                              ('completed_selectors',1)]:
                with self.subTest(key=key):
                    changed=dict(prior);changed[key]=value;resume.write_text(json.dumps(changed))
                    with patch.object(module.requests,'Session') as session_factory:
                        with self.assertRaisesRegex(ValueError,'RESUME'):
                            module.run_query(source,witness,comparison,'http://127.0.0.1:18080',
                                root/('rejected_'+key),max_bytes=1024*1024,max_seconds=10,resume_report=resume)
                        session_factory.assert_not_called()

    def test_body_budget_rejects_oversized_request_and_cumulative_refetch(self):
        reader, session = remote(b'0123456789abcdef', max_bytes=8)
        with reader:
            with self.assertRaisesRegex(ValueError, 'BUDGET_EXCEEDED_BEFORE_FETCH'):
                reader.read(9)
            self.assertEqual(session.calls, [])
        reader, session = remote(b'0123456789abcdef', max_bytes=8)
        with reader:
            self.assertEqual(reader.read(4), b'0123')
            reader.seek(0)
            with self.assertRaisesRegex(ValueError, 'BUDGET_EXCEEDED_BEFORE_FETCH'):
                reader.read(5)
            self.assertEqual(reader.body_bytes, 4)
            self.assertEqual(len(session.calls), 1)
        self.assertTrue(session.closed)


if __name__ == '__main__':
    unittest.main()
