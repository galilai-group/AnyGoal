"""Run architecture, gradient, stochasticity, sensitivity, and overfit checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_toy_driftworld import _make_dataset  # noqa: E402
from anygoal.wm.toy_predictors import (  # noqa: E402
    count_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=("pusht", "cube", "tworoom"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overfit-steps", type=int, default=1000)
    parser.add_argument("--overfit-samples", type=int, default=100)
    parser.add_argument("--overfit-lr", type=float, default=3e-4)
    parser.add_argument("--max-sample-mse-ratio", type=float, default=0.5)
    return parser.parse_args()


def _instantiate_model(cfg, model_name: str):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "configs" / "toy_driftworld"),
    ):
        model_cfg = compose(
            config_name="train",
            overrides=[f"data={cfg.data.name}", f"model={model_name}"],
        ).model
    config = OmegaConf.to_container(model_cfg, resolve=True)
    config.pop("name")
    return hydra.utils.instantiate(config)


def _fixed_metrics(model, states, actions, seeds: tuple[int, ...]) -> tuple[float, float]:
    objectives = []
    sample_mses = []
    target = states[:, model.history_size, None]
    with torch.no_grad():
        for seed in seeds:
            generator = torch.Generator(device=states.device).manual_seed(seed)
            output = model.loss(states, actions, generator=generator)
            objectives.append(output["loss"])
            sample_mses.append((output["generated"] - target).square().mean())
    return (
        float(torch.stack(objectives).mean().cpu()),
        float(torch.stack(sample_mses).mean().cpu()),
    )


def main() -> None:
    args = parse_args()
    if args.overfit_steps < 0 or args.overfit_samples <= 0 or args.overfit_lr <= 0:
        raise ValueError(
            "Overfit steps must be nonnegative, and sample count/LR must be positive."
        )
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "configs" / "toy_driftworld"),
    ):
        cfg = compose(config_name="train", overrides=[f"data={args.data}"])

    device = torch.device(args.device)
    model = _instantiate_model(cfg, "default").to(device)
    large_model = _instantiate_model(cfg, "large")
    default_parameters = count_trainable_parameters(model)
    large_parameters = count_trainable_parameters(large_model)
    del large_model

    generator = torch.Generator(device=device).manual_seed(17)
    batch_size = 8
    history = torch.randn(
        batch_size,
        model.history_size,
        model.state_dim,
        generator=generator,
        device=device,
    )
    actions = torch.randn(
        batch_size,
        model.action_dim,
        generator=generator,
        device=device,
    )
    noise = torch.randn(
        batch_size,
        model.state_dim,
        generator=generator,
        device=device,
    )
    lower_output = model(noise, history, actions)
    inference_output = model.predict_next_state(history, actions, noise=noise)
    expected_shape = (batch_size, model.state_dim)
    if lower_output.shape != expected_shape or inference_output.shape != expected_shape:
        raise RuntimeError("Shape check failed.")

    sequence_states = torch.randn(
        batch_size,
        model.history_size + 1,
        model.state_dim,
        generator=generator,
        device=device,
    )
    sequence_actions = torch.randn(
        batch_size,
        model.history_size + 1,
        model.action_dim,
        generator=generator,
        device=device,
    )
    model.zero_grad(set_to_none=True)
    training_output = model.loss(sequence_states, sequence_actions, generator=generator)
    training_output["loss"].backward()
    missing_gradients = []
    nonfinite_gradients = []
    for name, parameter in model.inner_model.named_parameters():
        if parameter.grad is None:
            missing_gradients.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite_gradients.append(name)
    if missing_gradients or nonfinite_gradients:
        raise RuntimeError(
            f"Gradient check failed: missing={missing_gradients}, "
            f"nonfinite={nonfinite_gradients}."
        )

    low_action = torch.full_like(actions, -2.0)
    high_action = torch.full_like(actions, 2.0)
    low_prediction = model.predict_next_state(history, low_action, noise=noise)
    high_prediction = model.predict_next_state(history, high_action, noise=noise)
    sensitivity = float((low_prediction - high_prediction).square().mean().sqrt().cpu())
    if sensitivity <= 0.0:
        raise RuntimeError("Action-sensitivity check failed.")

    first_generator = torch.Generator(device=device).manual_seed(91)
    second_generator = torch.Generator(device=device).manual_seed(91)
    seeded_first = model.predict_next_state(history, actions, generator=first_generator)
    seeded_second = model.predict_next_state(history, actions, generator=second_generator)
    stochastic_first = model.predict_next_state(history, actions)
    stochastic_second = model.predict_next_state(history, actions)
    if not torch.equal(seeded_first, seeded_second):
        raise RuntimeError("Seeded inference is not reproducible.")
    if torch.equal(stochastic_first, stochastic_second):
        raise RuntimeError("Fresh-noise inference unexpectedly became deterministic.")

    result = {
        "data": args.data,
        "state_dim": model.state_dim,
        "blocked_action_dim": model.action_dim,
        "shape": list(inference_output.shape),
        "gradient_parameters": sum(1 for _ in model.inner_model.parameters()),
        "action_sensitivity_rms": sensitivity,
        "default_trainable_parameters": default_parameters,
        "large_trainable_parameters": large_parameters,
        "seeded_inference_reproducible": True,
        "fresh_noise_is_stochastic": True,
    }

    if args.overfit_steps:
        train_set, _, _ = _make_dataset(cfg)
        sample_count = min(args.overfit_samples, len(train_set))
        batch = torch.utils.data.default_collate(
            [train_set[index] for index in range(sample_count)]
        )
        states = torch.nan_to_num(batch["state"].float()).to(device)
        actions = torch.nan_to_num(batch["action"].float()).to(device)
        overfit_model = _instantiate_model(cfg, "default").to(device).train()
        optimizer = torch.optim.AdamW(
            overfit_model.parameters(),
            lr=args.overfit_lr,
            betas=(0.9, float(cfg.optimizer.beta2)),
            weight_decay=float(cfg.optimizer.weight_decay),
        )
        seeds = (301, 302, 303, 304)
        initial_objective, initial_sample_mse = _fixed_metrics(
            overfit_model, states, actions, seeds
        )
        for _ in range(args.overfit_steps):
            output = overfit_model.loss(states, actions)
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                overfit_model.parameters(), float(cfg.optimizer.grad_clip)
            )
            optimizer.step()
            overfit_model.update_ema()
        final_objective, final_sample_mse = _fixed_metrics(
            overfit_model, states, actions, seeds
        )
        objective_ratio = final_objective / initial_objective
        sample_mse_ratio = final_sample_mse / initial_sample_mse
        result["overfit"] = {
            "samples": sample_count,
            "steps": args.overfit_steps,
            "lr": args.overfit_lr,
            "initial_objective": initial_objective,
            "final_objective": final_objective,
            "objective_ratio": objective_ratio,
            "initial_sample_mse": initial_sample_mse,
            "final_sample_mse": final_sample_mse,
            "sample_mse_ratio": sample_mse_ratio,
        }
        if sample_mse_ratio >= args.max_sample_mse_ratio:
            raise RuntimeError(
                f"Overfit sample-MSE ratio {sample_mse_ratio:.4f} did not beat "
                f"{args.max_sample_mse_ratio:.4f}."
            )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
