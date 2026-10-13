from anygoal.runners.train import train_sensorimotor


class FakeTrainer:
    def __init__(self):
        self.callbacks = []
        self.saved = None

    def save_checkpoint(self, path):
        self.saved = path


class FakeSchedulerTrainer:
    num_training_batches = 200
    accumulate_grad_batches = 2
    max_epochs = 10
    max_steps = -1


class FakeDataModule:
    @staticmethod
    def train_dataloader():
        return range(40)


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


def test_local_training_path_aliases_train_and_eval_to_same_file(tmp_path):
    dataset = (tmp_path / "cube.h5").resolve()
    arguments, aliases = train_sensorimotor._alias_local_training_dataset(
        [f"data.dataset.name={dataset}", "trainer.max_epochs=1"]
    )

    assert arguments == [
        "data.dataset.name=__anygoal_cube_train",
        "trainer.max_epochs=1",
    ]
    assert aliases == {
        "__anygoal_cube_train": dataset,
        "__anygoal_cube_eval": dataset,
    }


def test_dataset_alias_uses_explicit_path_without_cache_dir(tmp_path):
    dataset = (tmp_path / "cube.h5").resolve()
    captured = {}

    def dataset_factory(*args, **kwargs):
        captured.update(kwargs)
        return object()

    original = train_sensorimotor._local_path_hdf5_dataset.original
    original_aliases = dict(train_sensorimotor._DATASET_ALIASES)
    train_sensorimotor._local_path_hdf5_dataset.original = dataset_factory
    train_sensorimotor._DATASET_ALIASES.update({"__anygoal_cube_eval": dataset})
    try:
        train_sensorimotor._local_path_hdf5_dataset(
            name="__anygoal_cube_eval", cache_dir="unused"
        )
    finally:
        train_sensorimotor._local_path_hdf5_dataset.original = original
        train_sensorimotor._DATASET_ALIASES.clear()
        train_sensorimotor._DATASET_ALIASES.update(original_aliases)

    assert captured == {"path": str(dataset)}


def test_scheduler_defaults_match_upstream_step_based_schedule():
    module = type("FakeModule", (), {"trainer": FakeSchedulerTrainer()})()

    defaults = train_sensorimotor._sensorimotor_scheduler_defaults(module, None)

    assert defaults == {
        "warmup_steps": 10,
        "max_steps": 1000,
        "warmup_start_lr": 0.0,
        "eta_min": 0.0,
    }


def test_scheduler_defaults_respect_trainer_max_steps():
    trainer = FakeSchedulerTrainer()
    trainer.max_steps = 50
    module = type("FakeModule", (), {"trainer": trainer})()

    defaults = train_sensorimotor._sensorimotor_scheduler_defaults(module, None)

    assert defaults["warmup_steps"] == 1
    assert defaults["max_steps"] == 50


def test_scheduler_defaults_fall_back_to_dataloader_length():
    trainer = FakeSchedulerTrainer()
    trainer.num_training_batches = float("inf")
    trainer.train_dataloader = None
    trainer.datamodule = FakeDataModule()
    trainer.limit_train_batches = 0.5
    module = type("FakeModule", (), {"trainer": trainer})()

    defaults = train_sensorimotor._sensorimotor_scheduler_defaults(module, None)

    assert defaults["warmup_steps"] == 1
    assert defaults["max_steps"] == 100
