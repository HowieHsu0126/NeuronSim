"""Transfer registered LDNS scratch to an allowed alternate compute node."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def migrate(repo, destination):
    if os.environ.get("SLURMD_NODENAME") != "cpu03" or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("LDNS migration requires CPU03 Slurm")
    destination = destination.resolve()
    if destination != Path("/data/neuronsim/neuronsim-ldns-recovery-20261013"):
        raise ValueError("registered migration destination required")
    destination.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("ldns_launcher", repo / "Libs/scripts/run_latent_spike_diffusion_campaign.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    preparation = repo / "Output/neural_population_dynamics/baselines/latent_spike_diffusion/preparation_20261012"
    original = json.loads((preparation / "preparation.json").read_text())
    for seed in (41, 42, 43):
        work = destination / f"seed_{seed}"
        if (work / "migration.json").exists():
            raise FileExistsError("reuse completed migration instead of repeating it")
        source = ("/data/neuronsim/neuronsim-ldns-617154-41" if seed == 41
                  else f"/tmp/neuronsim/neuronsim-ldns-611529-{seed}")
        # All three exact scratch roots contain fewer than 1000 declared files.
        # Dereference retained input links. Exclude only disposable scratch and
        # interrupted atomic writes; preserve original files on GPU03.
        subprocess.run(["rsync", "-aL", "--exclude=scratch/", "--exclude=*.writing",
                        "--exclude=*.tmp", f"gpu03:{source}/", str(work) + "/"], check=True)
        if seed == 41:
            # The recovered trainer read its earlier retained input cell
            # directly; that cell is separate from the training work directory.
            subprocess.run(["rsync", "-aL", "gpu03:/tmp/neuronsim/neuronsim-ldns-611529-41/inputs/",
                            str(work / "inputs") + "/"], check=True)
        (work / "scratch").mkdir(exist_ok=True)
        if launcher.digest(work / "preparation.json") != launcher.digest(preparation / "preparation.json"):
            raise ValueError("transferred preparation identity changed")
        files = []
        for name, expected in original["files"].items():
            path = work / name
            if launcher.digest(path) != expected:
                raise ValueError("transferred software changed: " + name)
            files.append(dict(path=name, bytes=path.stat().st_size, sha256=expected))
        inputs = json.loads((work / "inputs.json").read_text())
        for row in inputs["files"]:
            path = work / "inputs" / row["path"]
            if path.stat().st_size != row["bytes"] or launcher.digest(path) != row["sha256"]:
                raise ValueError("transferred input changed: " + row["path"])
            files.append(dict(row, path="inputs/" + row["path"]))
        committed = None
        if seed == 41:
            committed = json.loads((work / "training/committed.json").read_text())
            expected = "9e53bc3f53e8653fb5fc910f4555ba8c46495193c4b57da74e17ca6a09e7615a"
            if (committed["stage"] != "autoencoder" or committed["next_epoch"] != 198
                    or committed["sha256"] != expected
                    or launcher.digest(work / "training" / committed["checkpoint"]) != expected):
                raise ValueError("transferred epoch198 checkpoint changed")
            selected = work / "training/checkpoints/selected_autoencoder_0179.pt"
            if launcher.digest(selected) != "c55669874bb54451b9e0508db641cd7b184c0e60fe563c619954f51fb55f1866":
                raise ValueError("transferred selected autoencoder changed")
            for root in ("training", "phases", "prerequisites"):
                for path in sorted((work / root).rglob("*")):
                    if path.is_file():
                        files.append(dict(path=str(path.relative_to(work)), bytes=path.stat().st_size,
                                          sha256=launcher.digest(path)))
        else:
            if (work / "training").exists():
                raise ValueError("unstarted seed input cell unexpectedly contains training")
        launcher.write(work / "migration.json", dict(status="PASS", seed=seed,
            source_node="gpu03", source_job="617154" if seed == 41 else "611529",
            source_path=source, destination_node="cpu03", committed=committed,
            preparation_sha256=launcher.digest(preparation / "preparation.json"), files=files,
            inputs_verified=len(inputs["files"]), input_bytes=sum(r["bytes"] for r in inputs["files"]),
            source_modified=False, scientific_claim_qualified=False))
        summary = json.loads((work / "migration.json").read_text())
        launcher.write(repo / "Output/neural_population_dynamics/baselines/latent_spike_diffusion/alternate_gpu_recovery_20261013"
                       / f"migration_seed_{seed}.json", {k: v for k, v in summary.items() if k != "files"})
        print(json.dumps(dict(seed=seed, status="PASS", work=str(work), files=len(files))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    migrate(args.repo, args.destination)
