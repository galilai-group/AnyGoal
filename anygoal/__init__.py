from .envs.registration import register_envs
from .collection import collect
from .registry import get_registry
from .paths import register_omegaconf_resolvers
from .version import __version__

register_envs()
register_omegaconf_resolvers()


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


def plan(
    policy,
    *,
    data,
    task=None,
    seed=0,
    num_evals=50,
    goal_offset_steps=25,
    eval_budget=50,
    start_selection="random",
    output_dir=None,
    save_video=False,
    save_artifacts=False,
    overwrite=False,
    world=None,
):
    """Load planning dependencies only when the planning API is used."""
    from .planning import plan as run_planning

    return run_planning(
        policy,
        data=data,
        task=task,
        seed=seed,
        num_evals=num_evals,
        goal_offset_steps=goal_offset_steps,
        eval_budget=eval_budget,
        start_selection=start_selection,
        output_dir=output_dir,
        save_video=save_video,
        save_artifacts=save_artifacts,
        overwrite=overwrite,
        world=world,
    )


def evaluate_rollout(
    model_or_adapter,
    *,
    data,
    task=None,
    checkpoint=None,
    seed=0,
    history_length=3,
    frame_skip=5,
    horizons=(15, 25),
    num_episodes=100,
    batch_size=32,
    max_starts_per_episode=None,
    output_dir=None,
    overwrite=False,
    device=None,
    **kwargs,
):
    """Load rollout dependencies only when rollout evaluation is used."""
    from .rollout import evaluate_rollout as run_rollout_evaluation

    return run_rollout_evaluation(
        model_or_adapter,
        data=data,
        task=task,
        checkpoint=checkpoint,
        seed=seed,
        history_length=history_length,
        frame_skip=frame_skip,
        horizons=horizons,
        num_episodes=num_episodes,
        batch_size=batch_size,
        max_starts_per_episode=max_starts_per_episode,
        output_dir=output_dir,
        overwrite=overwrite,
        device=device,
        **kwargs,
    )


__all__ = [
    "__version__",
    "collect",
    "evaluate_rollout",
    "get_registry",
    "plan",
    "register_envs",
    "train",
]
