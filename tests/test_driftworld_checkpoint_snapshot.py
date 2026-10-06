import torch

from scripts.snapshot_driftworld_checkpoints import inspect_and_snapshot


def test_periodic_snapshot_is_independent_and_idempotent(tmp_path):
    source = tmp_path / "ckpt-latest.pth"
    torch.save({"step": 20_000, "model": {"weight": torch.ones(2)}}, source)

    destination = inspect_and_snapshot(source, tmp_path, multiple=20_000)
    assert destination == tmp_path / "ckpt-step20000-periodic20k.pth"
    assert destination.stat().st_ino != source.stat().st_ino
    assert torch.load(destination, weights_only=False)["step"] == 20_000
    assert inspect_and_snapshot(source, tmp_path, multiple=20_000) == destination


def test_nonperiodic_checkpoint_is_not_copied(tmp_path):
    source = tmp_path / "ckpt-latest.pth"
    torch.save({"step": 20_500, "model": {}}, source)
    assert inspect_and_snapshot(source, tmp_path, multiple=20_000) is None
    assert not list(tmp_path.glob("ckpt-step*-periodic20k.pth"))
