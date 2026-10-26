from pathlib import Path
import json
import struct
import tempfile
import unittest
from .test_carlsim_records import network,events
from .carlsim_records import validate_prefix,_sha
from .carlsim_prefix_replay import compare


def fixture(root,*,original=None,time_shift=0,wall_time=2.,tail=b'unchanged-unparsed-synapse-bytes',author='author'):
    run=root/'native_run';(run/'results').mkdir(parents=True)
    checkpoint=run/'network5p.dat';network(checkpoint)
    payload=bytearray(checkpoint.read_bytes());payload[12:16]=struct.pack('<f',wall_time)
    checkpoint.write_bytes(payload+tail)
    events(run/'results/spk_I.dat',2,[(0,0),(12,1)])
    events(run/'results/spk_E.dat',3,[(20+time_shift,2),(20+time_shift,2),(2,0),(4449,1)])
    report=dict(status='NATIVE_PREFIX_EXIT_ZERO',exit_code=0,framework_revision='c4955998b950b9c561b3373a71b611aedc705be9',author_revision=author,reused_binary_sha256='a'*64,scope='native_source_prefix',native_run_executed=True,checkpoint_sha256=_sha(checkpoint),expected_prefix_ms=4450,core_neurons=5,populations={'I':2,'E':3})
    if original:report['independent_repeat_of_report_sha256']=_sha(original/'native_report.json')
    (root/'native_report.json').write_text(json.dumps(report))
    validate_prefix(root,root/'observations')


class PrefixReplay(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_wall_time_is_the_only_ignored_byte_range(self):
        a,b=self.root/'a',self.root/'b';fixture(a);fixture(b,original=a,wall_time=9.)
        result=compare(a,b,self.root/'replay.json')
        self.assertEqual(result['status'],'PASS_NATIVE_PREFIX_REPLAY')
        self.assertEqual(result['independently_reconstructed_counts'][0]['events'],6)
        self.assertEqual(result['independently_reconstructed_counts'][0]['unit_count_values'],5)
        self.assertEqual(result['checks'][0]['ignored_byte_range'],[12,16])
        self.assertFalse(result['full_state_restart_available'])
        self.assertFalse(result['native_internal_spike_counter_verified'])
    def test_changed_events_preserve_a_failed_replay_receipt(self):
        a,b=self.root/'a',self.root/'b';fixture(a);fixture(b,original=a,time_shift=1)
        with self.assertRaisesRegex(ValueError,'repeat differs'):compare(a,b,self.root/'replay.json')
        result=json.loads((self.root/'replay.json').read_text())
        self.assertEqual(result['status'],'MISMATCH_NATIVE_PREFIX_REPLAY')
        self.assertFalse(next(c for c in result['checks'] if c['name']=='E')['equal'])
    def test_payload_and_source_identity_are_not_ignored(self):
        a,b,c=self.root/'a',self.root/'b',self.root/'c';fixture(a)
        fixture(b,original=a,tail=b'different-unparsed-synapse-bytes')
        with self.assertRaisesRegex(ValueError,'repeat differs'):compare(a,b,self.root/'payload.json')
        fixture(c,original=a,author='changed')
        with self.assertRaisesRegex(ValueError,'source changed: author_revision'):compare(a,c,self.root/'source.json')


if __name__=='__main__':unittest.main()
