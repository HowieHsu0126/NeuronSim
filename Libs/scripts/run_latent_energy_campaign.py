"""Prepare and execute a source-bound EAG seed on an allocated GPU.

Training, evaluation and replay use the existing native cohort and shared phase
runner. Each resource attempt is preserved; completed seed links are published
only after full independent replay. A bounded training slice can be resumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

import yaml


CONFIG = "Libs/configs/neural_population_latent_energy.yaml"
ALLOWED_NODES = {"cpu03", "cpu04", "cpu05", "cpu06", "gpu03"}


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_json(helper, root, delivery, name):
    rows = helper.manifest_rows(delivery)
    if name not in rows or helper.digest(root / name) != rows[name]["sha256"]:
        raise ValueError("bound evidence missing or changed: " + name)
    return json.loads((root / name).read_text())


def prepare(repo, destination, cpu_validation, gpu_validation, full_unit_validation, campaign):
    helper = load_script(repo / "Libs/scripts/run_population_trajectory_campaign.py", "eag_preparation_helper")
    cpu, gpu, full = [json.loads((path / "delivery.json").read_text())
                      for path in (cpu_validation, gpu_validation, full_unit_validation)]
    if (any(row.get("status") != "PASS" or row.get("exit_code") != 0 for row in (cpu, gpu, full))
            or cpu.get("validation_scope") != "--tests-only"
            or gpu.get("validation_scope") != "--microbatch-only"
            or full.get("source_units") != 100000 or full.get("generated_unit_count_values") != 160000000
            or full.get("independently_replayed_unit_count_values") != 160000000):
        raise ValueError("complete EAG tests, allocated-GPU batch measurement and full-unit replay required")
    tests = verified_json(helper, cpu_validation, cpu, "phases/focused_tests/receipt.json")
    measured = verified_json(helper, gpu_validation, gpu, "microbatch.json")
    config = yaml.safe_load((repo / CONFIG).read_text())
    if (tests.get("status") != "PASS" or tests.get("exit_code") != 0
            or measured.get("status") != "PASS" or measured.get("optimizer_steps") != 2
            or measured.get("source_units") != 100000 or measured.get("prefix_bins") != 1000
            or measured.get("future_bins") != 100 or measured.get("logical_batch_size") != 256
            or measured.get("microbatch_size") != config["energy"]["microbatch_size"]
            or measured["updates"][0]["changed_parameter_tensors"] != 0
            or measured["updates"][1]["changed_parameter_tensors"] < 1):
        raise ValueError("validated source batch and actual warmup parameter update required")
    names = helper.software_closure(repo, ["latent_energy_training", "latent_energy_evaluation", "resource_probe"])
    source = repo / "Libs/external/LDNS/source.json"
    names += [str(source.relative_to(repo)), *["Libs/external/LDNS/" + name for name in json.loads(source.read_text())["files"]]]
    names += [CONFIG]
    cpu_rows, gpu_rows, full_rows = [helper.manifest_rows(proof) for proof in (cpu, gpu, full)]
    for name in names:
        current = helper.digest(repo / name)
        if any(name not in rows or rows[name]["sha256"] != current for rows in (cpu_rows, gpu_rows)):
            raise ValueError("EAG prepared source differs from tested/measured source: " + name)
    adapter = "Libs/population_dynamics/latent_energy.py"
    if full_rows[adapter]["sha256"] != helper.digest(repo / adapter):
        raise ValueError("full-unit EAG count model changed since replay validation")
    # The optional shared-mask training API was added after the full-unit probe.
    # Its unchanged default and replay semantics are covered by the bound tests.
    names += ["Libs/scripts/run_population_trajectory_campaign.py", "Libs/scripts/run_latent_energy_campaign.py"]
    scale_root = repo / "Output/neural_population_dynamics/model/trajectory_development/run_606139/seed_41"
    scale_proof = json.loads((scale_root / "delivery.json").read_text())
    if scale_proof.get("status") != "PASS":
        raise ValueError("accepted population fitting scale required")
    scale_row = helper.manifest_rows(scale_proof)["forecasts/fitting_field_scale.npz"]
    if destination.exists():
        raise FileExistsError("existing EAG preparation must be reused")
    destination.mkdir(parents=True)
    rows = []
    for name in sorted(set(names)):
        source = repo / name
        row = dict(path=name, bytes=source.stat().st_size, sha256=helper.digest(source))
        helper.copy_bound(source, destination / name, row)
        rows.append(row)
    helper.copy_bound(scale_root / scale_row["path"], destination / "fitting_field_scale.npz", scale_row)
    rows.append(dict(scale_row, path="fitting_field_scale.npz"))
    generation = verified_json(helper, full_unit_validation, full, "results/generation.json")
    replay = verified_json(helper, full_unit_validation, full, "results/replay.json")
    if generation["arrays"] != replay["arrays"] or generation["generated_unit_count_values"] != 160000000:
        raise ValueError("full-unit independent replay arrays differ")
    checkpoint = full_rows["results/checkpoint.pt"]
    record = dict(status="VALIDATED_EAG_TRAINING_AND_EVALUATION", training_seeds=[41, 42, 43],
        campaign=str(campaign.resolve()), files=rows, cpu_validation=str(cpu_validation),
        gpu_validation=str(gpu_validation), full_unit_validation=str(full_unit_validation),
        validation_deliveries={str(path): helper.digest(path / "delivery.json")
            for path in (cpu_validation, gpu_validation, full_unit_validation)},
        fitting_scale_source=str(scale_root / scale_row["path"]),
        fitting_scale_delivery_sha256=helper.digest(scale_root / "delivery.json"),
        serialized_full_model_bytes=checkpoint["bytes"], measured_microbatch=measured,
        scientific_claim_qualified=False, created_utc=datetime.now(timezone.utc).isoformat())
    helper.write_json(destination / "preparation.json", record)
    return record


def stage_resume(helper, parent, work, seed, preparation_digest):
    proof = json.loads((parent / "delivery.json").read_text())
    if (proof.get("training_seed") != seed or proof.get("preparation_sha256") != preparation_digest
            or proof.get("status") == "PASS"):
        raise ValueError("same-seed incomplete EAG resource attempt required")
    rows = helper.manifest_rows(proof)
    if "training/committed.json" not in rows:
        raise ValueError("resource recovery requires a committed training state")
    pointer = verified_json(helper, parent, proof, "training/committed.json")
    if "training/" + pointer["checkpoint"] not in rows:
        raise ValueError("committed recovery checkpoint is not delivery bound")
    completed = {}
    for phase_name in ("train", "evaluate", "replay"):
        name = "phases/" + phase_name + "/receipt.json"
        receipt = verified_json(helper, parent, proof, name) if name in rows else None
        completed[phase_name] = bool(receipt and receipt.get("status") == "PASS" and receipt.get("exit_code") == 0)
        if receipt:
            helper.copy_bound(parent / name, work / "prerequisites" / ("parent_" + phase_name + ".json"), rows[name])
    training_report = verified_json(helper, parent, proof, "training/run.json")
    completed["train"] = pointer["stage"] == "complete" and training_report.get("status") == "FITTED_DEVELOPMENT"
    if (completed["evaluate"] or completed["replay"]) and not completed["train"]:
        raise ValueError("completed evaluation requires terminal training in resource recovery")
    for name, row in rows.items():
        if name.startswith("training/") or completed["evaluate"] and name.startswith("forecasts/"):
            helper.copy_bound(parent / name, work / name, row)
    helper.write_json(work / "prerequisites/resource_parent.json", dict(parent=str(parent.resolve()),
        delivery_sha256=helper.digest(parent / "delivery.json"), completed_phases=completed,
        parent_job=proof["job_id"], committed=pointer))
    return completed


def storage_reserve(helper, repo, work, preparation, config):
    source = json.loads((repo / helper.COHORT / "delivery.json").read_text())
    # The shared reader normalizes native destination_bytes and archive bytes.
    # stage_inputs verifies those bound lengths and source digests before use.
    inputs = sum(row["bytes"] for row in helper.manifest_rows(source).values())
    selections = sum((config[stage]["maximum_epochs"] + config[stage]["validation_every"] - 1)
                     // config[stage]["validation_every"] for stage in ("autoencoder", "energy"))
    # At most one selected checkpoint per validation plus two full optimizer
    # commits. The output reserve includes one complete16-draw unit archive.
    model_bytes = preparation["serialized_full_model_bytes"]
    outputs = (selections + 6) * model_bytes + 3 * 1024 ** 3
    required = inputs + outputs + 20 * 1024 ** 3
    free = shutil.disk_usage(work).free
    report = dict(status="PASS" if free >= required else "BLOCKED", free_bytes=free,
        required_free_bytes=required, input_upper_bound_bytes=inputs, output_reserve_bytes=outputs,
        storage_path=str(work), serialized_full_model_bytes=model_bytes)
    helper.write_json(work / "storage_reserve.json", report)
    if report["status"] != "PASS":
        raise helper.ResourceBlocked(["EAG node-local input/checkpoint/output reserve unavailable"])


def run(repo, preparation_path, seed, resume_from=None, stop_after_commits=None):
    if seed not in (41, 42, 43):
        raise ValueError("registered EAG seed41/42/43 required")
    if os.environ.get("SLURMD_NODENAME") not in ALLOWED_NODES or not os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("EAG execution requires a Slurm GPU allocation on an authorized node")
    if subprocess.check_output(["findmnt", "-n", "-T", "/tmp", "-o", "FSTYPE"], text=True).strip() not in ("xfs", "ext4"):
        raise ValueError("node-local scratch required")
    helper = load_script(preparation_path / "Libs/scripts/run_population_trajectory_campaign.py", "eag_execution_helper")
    preparation_digest = helper.digest(preparation_path / "preparation.json")
    prepared = json.loads((preparation_path / "preparation.json").read_text())
    if prepared.get("status") != "VALIDATED_EAG_TRAINING_AND_EVALUATION" or prepared.get("training_seeds") != [41, 42, 43]:
        raise ValueError("validated EAG preparation required")
    campaign = Path(prepared["campaign"])
    if os.path.lexists(campaign / f"seed_{seed}"):
        raise FileExistsError("completed EAG seed already exists; reuse its evidence")
    scratch = Path("/tmp") / os.environ["USER"]
    scratch.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"neuronsim-energy-{os.environ['SLURM_JOB_ID']}-", dir=scratch))
    target = campaign / "attempts" / f"seed_{seed}_job_{os.environ['SLURM_JOB_ID']}"
    record = dict(status="PREPARING", exit_code=2, job_id=os.environ["SLURM_JOB_ID"], training_seed=seed,
        preparation_sha256=preparation_digest, preparation=str(preparation_path), node=os.environ["SLURMD_NODENAME"],
        partition=os.environ["SLURM_JOB_PARTITION"], source_units=100000, complete_source_episodes=72,
        fit_episodes=56, validation_episodes=16, scientific_claim_qualified=False,
        work=str(work), started_utc=datetime.now(timezone.utc).isoformat())
    print(json.dumps(record), flush=True)
    try:
        for row in prepared["files"]:
            helper.copy_bound(preparation_path / row["path"], work / row["path"], row)
        shutil.copy2(preparation_path / "preparation.json", work / "preparation.json")
        (work / "prerequisites").mkdir()
        (work / "scratch").mkdir()
        probe = load_script(work / "Libs/population_dynamics/resource_probe.py", "eag_probe").probe
        health = probe(work / "input_preflight.json", required_memory_gib=32)
        if health["status"] != "PASS":
            raise helper.ResourceBlocked(health["reasons"])
        config = yaml.safe_load((work / CONFIG).read_text())
        storage_reserve(helper, repo, work, prepared, config)
        complete = dict(train=False, evaluate=False, replay=False)
        if resume_from is not None:
            complete = stage_resume(helper, resume_from, work, seed, preparation_digest)
            record["resource_parent"] = str(resume_from)
        helper.stage_inputs(repo, work)
        for key, suffix in (("XDG_CACHE_HOME", "cache"), ("CUDA_CACHE_PATH", "cuda"),
                            ("TORCH_EXTENSIONS_DIR", "extensions"), ("PYKEOPS_CACHE_DIR", "keops")):
            directory = work / "scratch" / suffix
            directory.mkdir()
            os.environ[key] = str(directory)
        os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", NVIDIA_TF32_OVERRIDE="0",
                          OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
        entry = work / "entry.py"
        entry.write_text("import sys,runpy\nfrom pathlib import Path\np=Path(__file__).parent\nsys.path[:0]=[str(p),str(p/'Libs/external/LDNS')]\nmodule=sys.argv.pop(1)\nrunpy.run_module(module,run_name='__main__')\n")
        common = ["--acquisition", str(work / "inputs/train"), "--fields", str(work / "inputs/train_fields"),
                  "--config", str(work / CONFIG), "--seed", str(seed)]
        train = [sys.executable, str(entry), "Libs.population_dynamics.latent_energy_training", *common,
                 "--output", str(work / "training")]
        if resume_from is not None:
            train.append("--resume")
        if stop_after_commits is not None:
            train += ["--stop-after-commits", str(stop_after_commits)]
        evaluate = [sys.executable, str(entry), "Libs.population_dynamics.latent_energy_evaluation", *common,
            "--training", str(work / "training"), "--fitting-scale", str(work / "fitting_field_scale.npz"),
            "--output", str(work / "forecasts")]
        commands = [("train", train)] if stop_after_commits is not None else [
            ("train", train), ("evaluate", evaluate), ("replay", [*evaluate, "--replay"])]
        for phase_name, command in commands:
            if complete[phase_name]:
                continue
            code = helper.phase(work, phase_name, command, probe)
            if code:
                record.update(status="RESOURCE_BLOCKED" if code == 3 else "FAILED", exit_code=code, failed_phase=phase_name)
                break
        else:
            if stop_after_commits is not None:
                pointer = json.loads((work / "training/committed.json").read_text())
                record.update(status="COMMITTED_TRAINING_SLICE", exit_code=0, committed=pointer)
            else:
                replay = json.loads((work / "forecasts/replay.json").read_text())
                if replay.get("status") != "PASS" or replay.get("count_values_digest_replayed") != 7680000000:
                    raise ValueError("complete EAG48-origin/144-score independent replay required")
                record.update(status="PASS", exit_code=0, complete_origins=48, score_rows=144,
                              count_values_digest_replayed=7680000000)
    except helper.ResourceBlocked as error:
        record.update(status="RESOURCE_BLOCKED", exit_code=3, reasons=error.reasons)
    except BaseException as error:
        record.update(status="FAILED", exit_code=2, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    helper.archive(work, target, record)
    if record["status"] == "PASS":
        (campaign / f"seed_{seed}").symlink_to(target.resolve(), target_is_directory=True)
    shutil.rmtree(work)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    for name in ("cpu-validation", "gpu-validation", "full-unit-validation", "campaign", "resume-from"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop-after-commits", type=int)
    args = parser.parse_args()
    if args.prepare_only:
        if any(getattr(args, name) is None for name in ("cpu_validation", "gpu_validation", "full_unit_validation", "campaign")):
            parser.error("preparation requires test, GPU, full-unit evidence and campaign output")
        result = prepare(args.repo, args.preparation, args.cpu_validation, args.gpu_validation,
                         args.full_unit_validation, args.campaign)
    else:
        if args.seed is None:
            parser.error("execution requires a registered seed")
        result = run(args.repo, args.preparation, args.seed, args.resume_from, args.stop_after_commits)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}))
    raise SystemExit(result.get("exit_code", 0))


if __name__ == "__main__":
    main()
