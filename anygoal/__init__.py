from .envs.registration import register_envs
from .collection import collect
from .registry import get_registry
from .version import __version__

register_envs()


def train(
    model,
    *,
    data,
    task,
    learning_rate=None,
    batch_size=None,
    seed=None,
    num_workers=None,
    output_dir=None,
    max_steps=None,
    max_epochs=None,
    device=None,
    **kwargs,
):
    """Load training dependencies only when the training API is used."""
    from .training import train as run_training

    return run_training(
        model,
        data=data,
        task=task,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        output_dir=output_dir,
        max_steps=max_steps,
        max_epochs=max_epochs,
        device=device,
        **kwargs,
    )

__all__ = ["__version__", "collect", "get_registry", "register_envs", "train"]
