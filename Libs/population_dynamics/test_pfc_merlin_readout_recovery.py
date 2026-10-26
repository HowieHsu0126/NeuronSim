"""Merlin readout recovery retains verified output and rejects stable corruption."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import pfc_downstream_runtime as runtime


class ReadoutRecoveryTests(unittest.TestCase):
    def test_download_refreshes_stale_progress_metadata(self):
        data = b'verified repackaged original science'
        receipt = dict(index=1000, run_id='original', checkpoint_path='hdfs://result',
            persisted=True, exit_code=0, result_bytes=len(data), result_sha256=hashlib.sha256(data).hexdigest())
        with tempfile.TemporaryDirectory() as tmp, patch.object(runtime, 'hdfs_json', return_value=receipt), \
                patch.object(runtime.subprocess, 'run') as action:
            bundle = Path(tmp)/'result.tar.gz'
            def download(*args, **kwargs):
                bundle.write_bytes(data)
                return subprocess.CompletedProcess(args[0], 0)
            action.side_effect = download
            actual = runtime.download_success(dict(receipt, result_bytes=1, result_sha256='old'), 1000, bundle)
            self.assertEqual(actual['result_sha256'], receipt['result_sha256'])

    def test_stable_bundle_corruption_is_rejected(self):
        receipt = dict(index=1, run_id='original', checkpoint_path='hdfs://result',
            persisted=True, exit_code=0, result_bytes=3, result_sha256=hashlib.sha256(b'new').hexdigest())
        with tempfile.TemporaryDirectory() as tmp, patch.object(runtime, 'hdfs_json', return_value=receipt), \
                patch.object(runtime.subprocess, 'run') as action:
            bundle = Path(tmp)/'result.tar.gz'
            def download(*args, **kwargs):
                bundle.write_bytes(b'bad')
                return subprocess.CompletedProcess(args[0], 0)
            action.side_effect = download
            with self.assertRaisesRegex(ValueError, 'BUNDLE_IDENTITY_DIFFERS'):
                runtime.download_success(receipt, 1, bundle)

    def test_persisted_readouts_restore_without_source_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); domain=root/'domain'; work=root/'work'; durable=root/'durable'
            work.mkdir();durable.mkdir()
            observation=domain/'observations/task_0001';source=domain/'source_results/task_0001/run_merlin'
            payloads={}
            for relative in ('development5ms/report.json','development5ms/phase_responses.csv',
                    'development5ms/temporal_responses.csv','development5ms/unit_population_statistics.npz',
                    'observation1ms/report.json','observation1ms/observation.npz'):
                path=observation/relative;path.parent.mkdir(parents=True,exist_ok=True)
                data=json.dumps(dict(source_archive=str(source))).encode() if relative=='development5ms/report.json' else relative.encode()
                path.write_bytes(data);payloads[path]=data
            for relative in ('delivery.json','source_run/report.json','qualification/delivery.json',
                    'qualification/comparison/qualification.json'):
                path=source/relative;path.parent.mkdir(parents=True,exist_ok=True);data=relative.encode();path.write_bytes(data);payloads[path]=data
            published={}
            def hdfs(command, **kwargs):
                if command[2]=='-put':
                    target=durable/str(len(published));shutil.copyfile(command[3],target);published[command[4]]=target
                elif command[2]=='-get':shutil.copyfile(published[command[3]],command[4])
                return subprocess.CompletedProcess(command,0)
            with patch.object(runtime,'DOMAIN',domain),patch.object(runtime,'HERE',work), \
                    patch.dict(runtime.os.environ,dict(RH2_CHECKPOINT_HDFS='hdfs://checkpoint',
                        RH2_JOB_RUN_ID='run',ARNOLD_TRIAL_ID='trial')),patch.object(runtime.subprocess,'run',side_effect=hdfs):
                snapshot=runtime.persist_readouts([1],{},lambda index:dict(path=str(observation)))
                self.assertEqual(snapshot['files'],10)
                self.assertEqual(snapshot['source_simulations'],0)
                self.assertIn(snapshot['bundle_uri'].rsplit('/',1)[0]+'/receipt.json',published)
                for path in payloads:path.unlink()
                self.assertEqual(runtime.restore_snapshot(snapshot),[1])
                for path,data in payloads.items():self.assertEqual(path.read_bytes(),data)
                self.assertTrue(all(command.args[0][0]=='hdfs' for command in runtime.subprocess.run.call_args_list))

    def test_snapshot_rejects_hash_or_member_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle=Path(tmp)/'bundle.tar.gz';bundle.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'BUNDLE_HASH_DIFFERS'):
                runtime.unpack_readouts(bundle,hashlib.sha256(b'good').hexdigest())


if __name__=='__main__':unittest.main()
