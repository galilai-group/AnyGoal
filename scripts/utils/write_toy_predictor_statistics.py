"""Write canonical state/action statistics for a toy-predictor data config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anygoal.datasets import (  # noqa: E402
    adapted_minmax_statistics,
    adapted_zscore_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Hydra data config name")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_directory = REPO_ROOT / "scripts/train/config/toy_predictors"
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_directory),
    ):
        cfg = compose(config_name="train", overrides=[f"data={args.data}"])

    adapter = hydra.utils.instantiate(cfg.data.adapter)
    dataset = swm.data.load_dataset(
        cfg.data.dataset_name,
        cache_dir=cfg.data.cache_dir,
        num_steps=1,
        frameskip=1,
        keys_to_load=list(cfg.data.keys_to_load),
        keys_to_cache=[],
        transform=None,
    )
    state_normalization = str(cfg.data.get("state_normalization", "zscore"))
    if state_normalization == "minmax":
        fixed_minimum = cfg.data.get("fixed_state_minimum")
        fixed_maximum = cfg.data.get("fixed_state_maximum")
        if (fixed_minimum is None) != (fixed_maximum is None):
            raise ValueError(
                "fixed_state_minimum and fixed_state_maximum must be set together."
            )
        if fixed_minimum is None:
            state_min, state_max = adapted_minmax_statistics(
                dataset, adapter, "state"
            )
        else:
            state_min = torch.full((adapter.state_dim,), float(fixed_minimum))
            state_max = torch.full((adapter.state_dim,), float(fixed_maximum))
            if (state_max <= state_min).any():
                raise ValueError("Fixed state maximum must exceed minimum.")
        state_statistics = {
            "type": "minmax",
            "min": state_min.tolist(),
            "max": state_max.tolist(),
        }
    elif state_normalization == "zscore":
        state_mean, state_std = adapted_zscore_statistics(dataset, adapter, "state")
        state_statistics = {
            "type": "zscore",
            "mean": state_mean.tolist(),
            "std": state_std.tolist(),
        }
    else:
        raise ValueError(
            f"Unsupported state normalization type: {state_normalization!r}."
        )
    action_mean, action_std = adapted_zscore_statistics(dataset, adapter, "action")
    payload = {
        "dataset": str(cfg.data.dataset_name),
        "representation": list(adapter.state_names),
        "state": state_statistics,
        "action": {
            "type": "zscore",
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
        },
    }

    output_path = args.output or Path(cfg.data.normalization_path)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite statistics: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
