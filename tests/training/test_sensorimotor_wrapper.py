from anygoal.runners.train import train_sensorimotor


class FakeTrainer:
    def __init__(self):
        self.callbacks = []
        self.saved = None

    def save_checkpoint(self, path):
        self.saved = path


def test_missing_legacy_checkpoint_path_becomes_a_save_callback(tmp_path):
    trainer = FakeTrainer()
    captured = {}

    def manager(*args, **kwargs):
        captured.update(kwargs)
        return object()

    original = train_sensorimotor._compatible_manager.original
    train_sensorimotor._compatible_manager.original = manager
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    try:
        train_sensorimotor._compatible_manager(
            trainer=trainer, module=object(), data=object(), ckpt_path=checkpoint
        )
    finally:
        train_sensorimotor._compatible_manager.original = original

    assert captured["ckpt_path"] is None
    assert len(trainer.callbacks) == 1
    trainer.callbacks[0].on_train_end(trainer, object())
    assert trainer.saved == str(checkpoint)


def test_existing_checkpoint_remains_a_resume_path(tmp_path):
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.touch()
    trainer = FakeTrainer()
    captured = {}

    def manager(*args, **kwargs):
        captured.update(kwargs)
        return object()

    original = train_sensorimotor._compatible_manager.original
    train_sensorimotor._compatible_manager.original = manager
    try:
        train_sensorimotor._compatible_manager(
            trainer=trainer, module=object(), data=object(), ckpt_path=checkpoint
        )
    finally:
        train_sensorimotor._compatible_manager.original = original

    assert captured["ckpt_path"] == checkpoint.resolve()
    assert trainer.callbacks == []
