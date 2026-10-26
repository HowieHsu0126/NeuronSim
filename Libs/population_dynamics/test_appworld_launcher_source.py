"""Exercise normalized batch routing and the real AppWorld staging digest gate."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
BINDING_RELATIVE = "Output/neural_population_dynamics/agentic_extension/phase_action_decoder/bounded_submission_binding_01a128fd"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LauncherSourceTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(os.environ.get("APPWORLD_LAUNCHER_TEST_REPO", REPOSITORY))
        self.wrapper = Path(os.environ.get("APPWORLD_LAUNCHER_TEST_SOURCE",
            self.repo / "Libs/scripts/run_appworld_local_collaboration.sbatch"))
        self.binding = self.repo / BINDING_RELATIVE
        self.temp = tempfile.TemporaryDirectory(prefix="appworld-launcher-test-", dir="/tmp/neuronsim")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = self.root / "repo"
        self.fixture.mkdir()
        self.archive = self.root / "archive"
        self.work = Path("/tmp/neuronsim") / ("neuronsim-appworld-local-" + str(900000000 + os.getpid()))
        self.addCleanup(lambda: shutil.rmtree(self.work, ignore_errors=True))
        self.normalized = self.root / "slurm_script"
        text = self.wrapper.read_text()
        self.normalized.write_text("\n".join(line for line in text.splitlines() if line.strip()) + "\n")
        self.validated = self.root / "validated_launcher.sbatch"
        shutil.copy2(self.binding / "inputs/launcher.sbatch", self.validated)
        self.interpreter = self.root / "fixture_python"
        adapter = r"""#!{python}
import os,pathlib,sys
fixture=pathlib.Path(os.environ['FIXTURE_REPO'])
code=sys.stdin.read()
code=code.replace("repo = Path('/workspace/NeuronSim')", "repo = Path("+repr(str(fixture))+")")
import socket
socket.gethostname=lambda: os.environ['FIXTURE_HOSTNAME']
pathlib.Path(os.environ['FIXTURE_HELPER_LOG']).write_text('helper entered\n')
code=code.replace("if not Path('/data/neuronsim/neuronsim_models/qwen3_coder_30b_a3b_b2cff646').is_dir():", "if False:")
code=code.replace("archive = Path(archive_text)", "archive = Path(os.environ['FIXTURE_ARCHIVE'])")
code=code.replace("work, archive = Path(work_text), Path(archive_text)", "work, archive = Path(work_text), Path(os.environ['FIXTURE_ARCHIVE'])")
start=code.index('def health():')
stop=code.index('def archive_node(',start)
code=code[:start]+"def health():\n    write_json(work / 'resource.json', {{'fixture': True}})\n\n"+code[stop:]
sys.argv=[sys.argv[0]]+sys.argv[2:]
exec(compile(code,'actual_launcher_helper','exec'))
""".format(python=sys.executable)
        self.interpreter.write_text(adapter)
        self.interpreter.chmod(0o700)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        blocker = self.bin / "srun"
        blocker.write_text("#!/usr/bin/env bash\nprintf '%s\n' \"$*\" >> \"$FIXTURE_SRUN_LOG\"\nexit 77\n")
        blocker.chmod(0o700)
        proof=json.loads((self.binding / "prefix_validation_report.json").read_text())
        for row in proof['inputs']:
            target=self.fixture / row['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.binding / 'inputs' / row['path'], target)
        actual_sources=("Libs/__init__.py", "Libs/population_dynamics/__init__.py",
            "Libs/population_dynamics/provenance.py", "Libs/population_dynamics/appworld_reference.py")
        for name in actual_sources:
            target=self.fixture / name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(self.repo / name,target)
        docs=self.fixture / "Output/neural_population_dynamics/agentic_extension/appworld_two_agent_development/api_docs.json"
        docs.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(self.fixture / 'api_docs.json',docs)
        dependencies=yaml.safe_load((self.fixture / 'phase_action_decoder_protocol.yaml').read_text())['decoder']
        wheels=self.fixture / dependencies['dependency_archive']
        wheels.mkdir(parents=True,exist_ok=True)
        shutil.copy2(self.repo / dependencies['dependency_archive'] / 'delivery.json', wheels / 'delivery.json')
        for name in dependencies['wheels']:
            shutil.copy2(self.binding / 'inputs/decoder_wheels' / name,wheels / name)
        original=self.repo / 'Output/neural_population_dynamics/agentic_extension/appworld_local_collaboration/run_607907/gpu_host'
        shutil.copy2(original / 'readiness_report.json',self.fixture / 'readiness_report.json')
        self.env=dict(os.environ, APPWORLD_GPU_PYTHON=str(self.interpreter),
            APPWORLD_MODEL_METADATA=str(self.fixture / 'model_metadata.json'),
            APPWORLD_READINESS_REPORT=str(self.fixture / 'readiness_report.json'),
            APPWORLD_LAUNCHER_SOURCE=str(self.validated),
            APPWORLD_RECORDED_PREFIX_PROTOCOL=str(self.fixture / 'prefix/protocol.yaml'),
            APPWORLD_PREFIX_VALIDATION_REPORT=str(self.binding / 'prefix_validation_report.json'),
            APPWORLD_PHASE_ACTION_DECODER_PROTOCOL=str(self.fixture / 'phase_action_decoder_protocol.yaml'),
            APPWORLD_ACTION_DECODER_BOUNDS=str(self.fixture / 'action_decoder_bounds.yaml'),
            SLURM_HET_JOB_ID=self.work.name.rsplit('-',1)[1],SLURM_HET_SIZE='2',
            FIXTURE_REPO=str(self.fixture),FIXTURE_ARCHIVE=str(self.archive),
            FIXTURE_HOSTNAME="gpu03",FIXTURE_HELPER_LOG=str(self.root / "helper.log"),
            FIXTURE_SRUN_LOG=str(self.root / 'srun.log'),PATH=str(self.bin)+os.pathsep+os.environ['PATH'],
            PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.env.pop("APPWORLD_GPU_NODE", None)
        prefix=yaml.safe_load((self.fixture / 'prefix/protocol.yaml').read_text())
        for row in prefix['inputs'].values():
            shutil.copy2(original / 'prefix' / row['path'],self.fixture / 'prefix' / row['path'])

    def run_entry(self):
        return subprocess.run(['bash',str(self.normalized),'1'],env=self.env,
                              text=True,capture_output=True,check=False)

    def test_normalized_entry_stages_exact_validated_source_and_records_wrapper(self):
        subprocess.run(['bash','-n',str(self.wrapper)],check=True,capture_output=True)
        result=self.run_entry()
        self.assertEqual(result.returncode,1,result.stderr)
        self.assertNotIn('APPWORLD_DECODER_VALIDATED_CODE_CHANGED',result.stderr)
        snapshot=self.archive / 'inputs'
        receipt=json.loads((snapshot / 'copy_out.json').read_text())
        self.assertTrue(receipt['verified'])
        self.assertEqual(sha256(snapshot / 'launcher.sbatch'),sha256(self.validated))
        self.assertEqual(sha256(snapshot / 'execution_launcher.sbatch'),sha256(self.normalized))
        binding=json.loads((snapshot / 'launcher_source_binding.json').read_text())
        self.assertEqual(binding['source_kind'],'explicit_validated_source')
        self.assertEqual(binding['validated_source_sha256'],sha256(self.validated))
        self.assertEqual(binding['execution_wrapper_sha256'],sha256(self.normalized))
        log=(self.root / 'srun.log').read_text()
        self.assertIn('__stage_cpu',log)
        self.assertNotIn('__host',log)
        self.assertFalse((self.archive / 'gpu_host/host').exists())
        observed={'validated_source_sha256':sha256(self.validated),
            'execution_wrapper_sha256':sha256(self.normalized),'frozen_inputs':len(receipt['files']),
            'blocked_srun_exit':77,'model_generations':0,'native_executions':0}
        if os.environ.get('APPWORLD_LAUNCHER_TEST_OBSERVATION'):
            Path(os.environ['APPWORLD_LAUNCHER_TEST_OBSERVATION']).write_text(json.dumps(
                {'observed':observed,'source_binding':binding,'staged_inputs_receipt':receipt},indent=2)+'\n')
        print(json.dumps(observed))

    def test_missing_explicit_decoder_source_is_rejected_before_staging(self):
        del self.env['APPWORLD_LAUNCHER_SOURCE']
        result=self.run_entry()
        self.assertEqual(result.returncode,1)
        self.assertIn('APPWORLD_LAUNCHER_SOURCE must name the exact validated launcher',result.stderr)
        self.assertFalse(self.archive.exists())
        self.assertFalse((self.root / 'srun.log').exists())

    def test_changed_explicit_source_preserves_existing_digest_rejection(self):
        self.validated.write_text(self.validated.read_text()+'# changed source\n')
        result=self.run_entry()
        self.assertEqual(result.returncode,1)
        self.assertIn('APPWORLD_DECODER_VALIDATED_CODE_CHANGED:launcher.sbatch',result.stderr)
        self.assertFalse((self.archive / 'inputs').exists())
        self.assertTrue((self.archive / 'gpu_host/copy_out.json').exists())
        self.assertFalse((self.root / 'srun.log').exists())


    def test_cpu03_gpu_host_stages_with_the_existing_cpu_controller_route(self):
        self.env['APPWORLD_GPU_NODE'] = 'cpu03'
        self.env['FIXTURE_HOSTNAME'] = 'cpu03'
        result = self.run_entry()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertNotIn('APPWORLD_COMPONENT_NODE_MISMATCH', result.stderr)
        receipt = json.loads((self.archive / 'inputs/copy_out.json').read_text())
        self.assertTrue(receipt['verified'])
        log = (self.root / 'srun.log').read_text()
        self.assertIn('__stage_cpu', log)
        self.assertNotIn('__host', log)

    def test_wrong_actual_gpu_node_is_rejected_before_staging(self):
        for selected, actual in ((None, 'cpu03'), ('cpu03', 'gpu03')):
            with self.subTest(selected=selected, actual=actual):
                if selected is None:
                    self.env.pop('APPWORLD_GPU_NODE', None)
                else:
                    self.env['APPWORLD_GPU_NODE'] = selected
                self.env['FIXTURE_HOSTNAME'] = actual
                result = self.run_entry()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('APPWORLD_COMPONENT_NODE_MISMATCH', result.stderr)
                self.assertTrue((self.root / 'helper.log').exists())
                self.assertFalse(self.work.exists())
                self.assertFalse(self.archive.exists())
                self.assertFalse((self.root / 'srun.log').exists())

    def test_cpu04_gpu_host_is_rejected_before_staging_or_model_execution(self):
        self.env['APPWORLD_GPU_NODE'] = 'cpu04'
        for arguments in (['1'], ['__host', str(self.work)], ['__native', str(self.work)],
                          ['__stage_cpu', str(self.work), str(self.archive)]):
            with self.subTest(arguments=arguments):
                result = subprocess.run(['bash', str(self.normalized), *arguments],
                    env=self.env, text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('APPWORLD_GPU_NODE_COLLIDES_WITH_CPU_CONTROLLER', result.stderr)
                self.assertFalse((self.root / 'helper.log').exists())
                self.assertFalse(self.work.exists())
                self.assertFalse(self.archive.exists())
                self.assertFalse((self.root / 'srun.log').exists())

    def test_unsupported_gpu_host_is_rejected_before_staging(self):
        self.env['APPWORLD_GPU_NODE'] = 'cpu05'
        result = self.run_entry()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('APPWORLD_GPU_NODE_UNSUPPORTED', result.stderr)
        self.assertFalse((self.root / 'helper.log').exists())
        self.assertFalse(self.work.exists())
        self.assertFalse(self.archive.exists())
        self.assertFalse((self.root / 'srun.log').exists())


class LauncherInterruptionTests(unittest.TestCase):
    """Run the actual shell modes with disposable, non-scientific processes."""

    def setUp(self):
        source = Path(os.environ.get("APPWORLD_LAUNCHER_TEST_SOURCE",
            REPOSITORY / "Libs/scripts/run_appworld_local_collaboration.sbatch"))
        self.temp = tempfile.TemporaryDirectory(prefix="appworld-interruption-test-", dir="/tmp/neuronsim")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = self.root / "repo"
        self.fixture.mkdir()
        self.archive = self.root / "archive"
        self.job_id = str(910000000 + os.getpid())
        self.work = Path("/tmp/neuronsim") / ("neuronsim-appworld-local-" + self.job_id)
        self.addCleanup(lambda: shutil.rmtree(self.work, ignore_errors=True))
        self.python = self.root / "fixture_python"
        adapter = r"""#!{python}
import json,os,pathlib,signal,sys,time
work=pathlib.Path(os.environ['FIXTURE_WORK'])
if sys.argv[1] != '-':
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        raise RuntimeError("FIXTURE_CHILD_SIGINT_IGNORED")
    target=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
    target.mkdir(exist_ok=True)
    sys.stderr.write('fixture process entered before terminal report\n')
    sys.stderr.flush()
    pathlib.Path(os.environ['FIXTURE_READY']).write_text(str(os.getpid())+'\n')
    mode=os.environ.get('FIXTURE_PROCESS','fail')
    if mode == 'wait':
        time.sleep(30)
    if mode == 'stubborn':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pathlib.Path(os.environ['FIXTURE_READY']).write_text(str(os.getpid())+'\n')
        time.sleep(30)
    if mode == 'success':
        if os.environ.get('FIXTURE_NATIVE_STREAM'):
            sys.stdout.write(sys.stdin.readline())
        (target/'report.json').write_text(json.dumps({{'fixture': True}})+'\n')
        raise SystemExit(2)
    raise SystemExit(17)
code=sys.stdin.read()
code=code.replace("if socket.gethostname().split('.')[0] != expected_node:", "if False:")
code=code.replace("work, archive = Path(work_text), Path(archive_text)",
                  "work, archive = Path(work_text), Path(os.environ['FIXTURE_ARCHIVE'])")
sys.argv=[sys.argv[0]]+sys.argv[2:]
if sys.argv[1] == 'stage_gpu':
    work.mkdir()
    (work/'resource.json').write_text(json.dumps({{'fixture': True}})+'\n')
    archive=pathlib.Path(os.environ['FIXTURE_ARCHIVE'])
    (archive/'inputs').mkdir(parents=True)
    launcher=pathlib.Path(os.environ['FIXTURE_LAUNCHER'])
    (work/'launcher.sbatch').write_bytes(launcher.read_bytes())
    (archive/'inputs/launcher.sbatch').write_bytes(launcher.read_bytes())
    if os.environ.get('FIXTURE_STAGE_WAIT'):
        pathlib.Path(os.environ['FIXTURE_READY']).write_text(str(os.getpid())+'\n')
        time.sleep(30)
    raise SystemExit(int(os.environ.get('FIXTURE_STAGE_EXIT','0')))
exec(compile(code,'actual_launcher_helper','exec'))
""".format(python=sys.executable)
        self.python.write_text(adapter)
        self.python.chmod(0o700)
        self.launcher = self.root / "launcher.sbatch"
        launcher = source.read_text().replace(
            "REPO=/workspace/NeuronSim", "REPO=" + str(self.fixture)).replace(
            "CPU_PYTHON=/tmp/neuronsim/neuronsim-appworld-runtime-575402/venv/bin/python",
            "CPU_PYTHON=" + str(self.python))
        self.launcher.write_text(launcher)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        srun = self.bin / "srun"
        srun.write_text(r"""#!{python}
import os,pathlib,signal,sys,time
if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
    raise RuntimeError("FIXTURE_SRUN_SIGINT_IGNORED")
args=sys.argv[1:]
mode=next(arg for arg in args if arg.startswith('__'))
with open(os.environ['FIXTURE_SRUN_LOG'],'a') as stream:
    stream.write(mode+'\n')
if mode == '__host':
    work=pathlib.Path(os.environ['FIXTURE_WORK'])
    (work/'host').mkdir()
    pathlib.Path(os.environ['FIXTURE_READY']).write_text(str(os.getpid())+'\n')
    selected=os.environ.get('FIXTURE_PROCESS','fail')
    if selected == 'wait':
        time.sleep(30)
    if selected == 'success':
        (work/'host/report.json').write_text('{{"fixture":true}}\n')
        raise SystemExit(2)
    raise SystemExit(17)
raise SystemExit(0)
""".format(python=sys.executable))
        srun.chmod(0o700)
        self.env = dict(os.environ,
            APPWORLD_GPU_PYTHON=str(self.python), APPWORLD_MODEL_METADATA=str(self.root / "metadata.json"),
            APPWORLD_READINESS_REPORT=str(self.root / "readiness.json"),
            APPWORLD_LAUNCHER_SOURCE=str(self.launcher), APPWORLD_HET_LEADER_JOB_ID=self.job_id,
            SLURM_HET_JOB_ID=self.job_id, SLURM_HET_SIZE="2", APPWORLD_SCRATCH_ROOT="/tmp/neuronsim",
            FIXTURE_ARCHIVE=str(self.archive), FIXTURE_WORK=str(self.work),
            FIXTURE_LAUNCHER=str(self.launcher), FIXTURE_READY=str(self.root / "ready"),
            FIXTURE_SRUN_LOG=str(self.root / "srun.log"),
            PATH=str(self.bin)+os.pathsep+os.environ["PATH"], PYTHONDONTWRITEBYTECODE="1")
        # Avoid inherited development protocols. These tests do not stage research inputs.
        for name in ("APPWORLD_PHASE_ACTION_DECODER_PROTOCOL", "APPWORLD_RECORDED_PREFIX_PROTOCOL",
                     "APPWORLD_OBSERVATION_HISTORY_REPAIR", "APPWORLD_VERIFICATION_REPLY_PROTOCOL",
                     "APPWORLD_ACTION_DECODER_BOUNDS"):
            self.env.pop(name, None)

    def prepare_component(self):
        self.work.mkdir()
        (self.work / "resource.json").write_text('{"fixture": true}\n')
        # Large log capture must be bounded. Unlisted files must remain only in scratch.
        (self.work / "host").mkdir()
        (self.work / "host/controller_stderr.log").write_bytes(b'x' * (1024 * 1024 + 11))
        (self.work / "unlisted_large_data.bin").write_bytes(b'untouched')

    def run_entry(self, mode=None):
        args = ["bash", str(self.launcher)]
        args += [mode, str(self.work)] if mode else ["1"]
        return subprocess.run(args, env=self.env, text=True, capture_output=True, check=False, timeout=10)

    def interrupt_entry(self, mode, received_signal):
        self.env.setdefault("FIXTURE_PROCESS", "wait")
        args = ["bash", str(self.launcher)]
        args += [mode, str(self.work)] if mode else ["1"]
        process = subprocess.Popen(args, env=self.env, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            while not (self.root / "ready").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    self.fail("fixture did not reach the pre-report wait")
                time.sleep(0.01)
            child_pid = int((self.root / "ready").read_text())
            process.send_signal(received_signal)
            stdout, stderr = process.communicate(timeout=5)
            with self.assertRaises(ProcessLookupError, msg="task-owned fixture child remains alive"):
                os.kill(child_pid, 0)
            return process.returncode, stdout, stderr
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()

    def inspect_failure(self, component, code, received_signal=None):
        destination = self.archive / "failure" / component
        receipt = json.loads((destination / "interruption.json").read_text())
        self.assertEqual(receipt["shell_exit_code"], code)
        self.assertEqual(receipt["signal"], received_signal)
        self.assertEqual(receipt["scratch_path"], str(self.work))
        self.assertTrue(receipt["scratch_exists_at_capture"])
        self.assertFalse(receipt["scratch_removed"])
        self.assertFalse(receipt["copy_errors"])
        self.assertTrue(self.work.is_dir())
        self.assertIn("host/generation_attempts.json", receipt["missing_files"])
        self.assertFalse((destination / "unlisted_large_data.bin").exists())
        if (self.work / "host/controller_stderr.log").exists():
            log = next(row for row in receipt["files"] if row["source"] == "host/controller_stderr.log")
            self.assertTrue(log["truncated"])
            self.assertEqual(log["capture"], "tail")
            self.assertEqual(log["captured_bytes"], 1024 * 1024)
            self.assertEqual((destination / log["path"]).stat().st_size, 1024 * 1024)
            self.assertFalse((destination / "host/controller_stderr.log").exists())
        self.assertFalse((self.archive / "gpu_host/copy_out.json").exists())
        evidence = os.environ.get("APPWORLD_INTERRUPTION_TEST_EVIDENCE")
        if evidence:
            target = Path(evidence) / (self._testMethodName + "_" + component + "_" + str(code))
            shutil.copytree(destination, target)
        return receipt

    def test_host_and_native_pre_report_failure_preserves_actual_code_and_logs(self):
        for mode, component, log in (("__host", "gpu_host", "host_stderr.log"),
                                     ("__native", "cpu_controller", "native_stderr.log")):
            with self.subTest(component=component):
                if self.work.exists():
                    shutil.rmtree(self.work)
                self.prepare_component()
                result = self.run_entry(mode)
                self.assertEqual(result.returncode, 17, result.stderr)
                receipt = self.inspect_failure(component, 17)
                destination = self.archive / "failure" / component
                process = json.loads((destination / "process_exit.json").read_text())
                self.assertEqual(process["python_cli_exit_code"], 17)
                self.assertIn("fixture process entered", (destination / log).read_text())
                self.assertIn("host/report.json", receipt["missing_files"])
                self.assertFalse((self.work / "controller/report.json").exists())

    def test_host_and_native_sigterm_preserves_missing_receipts(self):
        for mode, component in (("__host", "gpu_host"), ("__native", "cpu_controller")):
            with self.subTest(component=component):
                if self.work.exists():
                    shutil.rmtree(self.work)
                (self.root / "ready").unlink(missing_ok=True)
                self.prepare_component()
                code, stdout, stderr = self.interrupt_entry(mode, signal.SIGTERM)
                self.assertEqual(code, 143, stderr)
                receipt = self.inspect_failure(component, 143, "TERM")
                self.assertIn("process_exit.json", receipt["missing_files"])
                if mode == "__native":
                    self.assertEqual(stdout, "")

    def test_host_and_native_sigint_preserves_exit130(self):
        for mode, component in (("__host", "gpu_host"), ("__native", "cpu_controller")):
            with self.subTest(component=component):
                if self.work.exists():
                    shutil.rmtree(self.work)
                (self.root / "ready").unlink(missing_ok=True)
                self.prepare_component()
                code, _, stderr = self.interrupt_entry(mode, signal.SIGINT)
                self.assertEqual(code, 130, stderr)
                self.inspect_failure(component, 130, "INT")

    def test_signal_handler_reaps_only_owned_child_when_term_is_ignored(self):
        self.prepare_component()
        self.env["FIXTURE_PROCESS"] = "stubborn"
        code, _, stderr = self.interrupt_entry("__host", signal.SIGTERM)
        self.assertEqual(code, 143, stderr)
        self.inspect_failure("gpu_host", 143, "TERM")

    def test_parent_pre_report_failure_skips_recursive_archive(self):
        result = self.run_entry()
        self.assertEqual(result.returncode, 17, result.stderr)
        receipt = self.inspect_failure("launcher", 17)
        self.assertEqual(receipt["phase"], "host_step")
        self.assertEqual((self.root / "srun.log").read_text().splitlines(), ["__stage_cpu", "__host"])
        self.assertIn("launcher_result.json", receipt["missing_files"])

    def test_parent_sigterm_preserves_exit143_without_archive_step(self):
        code, _, stderr = self.interrupt_entry(None, signal.SIGTERM)
        self.assertEqual(code, 143, stderr)
        self.inspect_failure("launcher", 143, "TERM")
        self.assertEqual((self.root / "srun.log").read_text().splitlines(), ["__stage_cpu", "__host"])

    def test_parent_sigterm_during_staging_preserves_exit143_and_child_termination(self):
        self.env["FIXTURE_STAGE_WAIT"] = "1"
        code, _, stderr = self.interrupt_entry(None, signal.SIGTERM)
        self.assertEqual(code, 143, stderr)
        receipt = self.inspect_failure("launcher", 143, "TERM")
        self.assertEqual(receipt["phase"], "stage_gpu")
        self.assertFalse((self.root / "srun.log").exists())

    def test_parent_success_keeps_existing_verified_archive_and_cleanup(self):
        self.env["FIXTURE_PROCESS"] = "success"
        result = self.run_entry()
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads((self.archive / "gpu_host/copy_out.json").read_text())
        self.assertTrue(receipt["verified"])
        self.assertTrue(receipt["scratch_removed_after_copy"])
        self.assertEqual(receipt["gpu_host_step_exit_code"], 2)
        self.assertFalse(self.work.exists())
        self.assertFalse((self.archive / "failure").exists())
        self.assertEqual((self.root / "srun.log").read_text().splitlines(),
                         ["__stage_cpu", "__host", "__archive_cpu"])

    def test_component_success_records_code2_without_failure_snapshot(self):
        self.prepare_component()
        self.env["FIXTURE_PROCESS"] = "success"
        self.env["FIXTURE_NATIVE_STREAM"] = "1"
        result = subprocess.run(["bash", str(self.launcher), "__native", str(self.work)],
            env=self.env, input='{"fixture_stream":true}\n', text=True,
            capture_output=True, check=False, timeout=10)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, '{"fixture_stream":true}\n')
        self.assertEqual(json.loads((self.work / "process_exit.json").read_text())["python_cli_exit_code"], 2)
        self.assertFalse((self.archive / "failure").exists())


if __name__ == '__main__':
    unittest.main()
