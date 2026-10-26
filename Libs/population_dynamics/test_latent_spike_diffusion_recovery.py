"""Transferred LDNS recovery must preserve source and epoch identities."""
import importlib.util
import json
from pathlib import Path

import pytest


def launcher():
    path = Path(__file__).parents[1] / "scripts/run_latent_spike_diffusion_campaign.py"
    spec = importlib.util.spec_from_file_location("ldns_recovery", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def cell(tmp_path):
    run = launcher()
    prep, work = tmp_path / "prep", tmp_path / "work"
    prep.mkdir(); work.mkdir()
    name = "Libs/population_dynamics/latent_spike_diffusion_training.py"
    for root in (prep, work):
        p = root / name; p.parent.mkdir(parents=True); p.write_text("frozen trainer\n")
        (root / "preparation.json").write_text(json.dumps(dict(files={name: run.digest(p)})))
    checkpoint = work / "training/checkpoints/autoencoder_0198.pt"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"committed optimizer state")
    pointer = dict(checkpoint="checkpoints/autoencoder_0198.pt", sha256=run.digest(checkpoint),
                   stage="autoencoder", next_epoch=198)
    (work / "training/committed.json").write_text(json.dumps(pointer))
    (work / "prerequisites").mkdir()
    (work / "prerequisites/cuda_delivery.json").write_text(json.dumps(dict(status="PASS", trainer_sha256=run.digest(work / name))))
    migration = dict(status="PASS", source_node="gpu03", source_job="617154", committed=pointer,
                     files=[dict(path=str(checkpoint.relative_to(work)), bytes=checkpoint.stat().st_size,
                                 sha256=run.digest(checkpoint))])
    (work / "migration.json").write_text(json.dumps(migration))
    return run, prep, work


def test_transferred_epoch198_is_admitted(cell):
    run, prep, work = cell
    _, migration = run.validate_recovery(work, prep, 41)
    assert migration["committed"]["next_epoch"] == 198


def test_transferred_checkpoint_corruption_is_rejected(cell):
    run, prep, work = cell
    (work / "training/checkpoints/autoencoder_0198.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="migrated file changed"):
        run.validate_recovery(work, prep, 41)


def test_recovery_never_admits_other_seed_or_epoch(cell):
    run, prep, work = cell
    with pytest.raises(ValueError, match="cancelled seed41"):
        run.validate_recovery(work, prep, 42)
    pointer = json.loads((work / "training/committed.json").read_text())
    pointer["next_epoch"] = 0
    (work / "training/committed.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="epoch identity"):
        run.validate_recovery(work, prep, 41)


def test_recovery_frozen_trainer_change_is_rejected(cell):
    run, prep, work = cell
    (work / "Libs/population_dynamics/latent_spike_diffusion_training.py").write_text("changed\n")
    with pytest.raises(ValueError, match="software changed"):
        run.validate_recovery(work, prep, 41)
