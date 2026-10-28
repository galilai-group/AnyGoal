import os

os.environ["MUJOCO_GL"] = "egl"

import sys
import time
import types
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
import anygoal  # noqa: F401
from anygoal.paths import checkpoint_root, data_root, output_root
from anygoal.envs.cube_hop.env import (
    add_cube_evaluation_metrics,
    prepare_cube_evaluation,
)
from anygoal.envs.insert_peg.env import (
    add_insert_peg_evaluation_metrics,
)
from anygoal._planning_artifacts import PlanningArtifactRecorder


def patch_transformers_vit_pickle_compat() -> None:
    """Restore old ViT class names used by serialized checkpoints."""
    try:
        from transformers.models.vit import modeling_vit
        from transformers.pytorch_utils import apply_chunking_to_forward
    except ImportError:
        return

    if hasattr(modeling_vit, "ViTMLP"):
        return

    class ViTMLP(torch.nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.chunk_size_feed_forward = config.chunk_size_feed_forward
            self.seq_len_dim = 1
            self.intermediate = modeling_vit.ViTIntermediate(config)
            self.output = modeling_vit.ViTOutput(config)
            self.layernorm = torch.nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )

        def forward(
            self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
        ) -> torch.Tensor:
            if hasattr(self, "fc1") and hasattr(self, "fc2"):
                hidden_states = self.fc1(hidden_states)
                hidden_states = self.activation_fn(hidden_states)
                return self.fc2(hidden_states)

            layer_output = self.layernorm(hidden_states)
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                layer_output,
            )
            return self.output(layer_output, input_tensor)

        def feed_forward_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.intermediate(hidden_states)

    ViTMLP.__module__ = modeling_vit.__name__
    modeling_vit.ViTMLP = ViTMLP


def patch_loaded_model_compat(model: torch.nn.Module) -> None:
    """Patch serialized HF ViT objects across Transformers minor versions."""
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None:
            continue
        if not hasattr(config, "output_attentions"):
            config.output_attentions = getattr(config, "_output_attentions", False)
        if not hasattr(config, "output_hidden_states"):
            config.output_hidden_states = getattr(config, "_output_hidden_states", False)
        if not hasattr(config, "use_return_dict"):
            config.__dict__["use_return_dict"] = getattr(config, "return_dict", True)

    vit = getattr(model, "encoder", None)
    if vit is None or not hasattr(vit, "layers"):
        return
    if hasattr(vit, "encoder"):
        return

    def legacy_attention_forward(attention, hidden_states, output_attentions: bool):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_heads = attention.num_attention_heads
        head_dim = attention.head_dim

        def shape(states):
            return states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        query = shape(attention.q_proj(hidden_states))
        key = shape(attention.k_proj(hidden_states))
        value = shape(attention.v_proj(hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2)) * attention.scaling
        probs = torch.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=attention.attention_dropout, training=attention.training)
        context = torch.matmul(probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = attention.o_proj(context)
        return output, probs if output_attentions else None

    def legacy_layer_forward(layer, hidden_states, output_attentions: bool):
        attention_input = layer.layernorm_before(hidden_states)
        attention_output, attention_probs = legacy_attention_forward(
            layer.attention,
            attention_input,
            output_attentions,
        )
        hidden_states = hidden_states + layer.dropout(attention_output)
        mlp_input = layer.layernorm_after(hidden_states)
        mlp_output = layer.mlp(mlp_input)
        hidden_states = hidden_states + layer.dropout(mlp_output)
        return hidden_states, attention_probs

    def legacy_vit_forward(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = (
            getattr(self.config, "use_return_dict", getattr(self.config, "return_dict", True))
            if return_dict is None
            else return_dict
        )
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states, attention_probs = legacy_layer_forward(
                layer, hidden_states, output_attentions
            )
            if output_attentions:
                all_attentions = all_attentions + (attention_probs,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        sequence_output = self.layernorm(hidden_states)

        if not return_dict:
            output = (sequence_output, None)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            if output_attentions:
                output = output + (all_attentions,)
            return output

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    vit.pooler = None
    vit.forward = types.MethodType(legacy_vit_forward, vit)


class ReverseChunkDataset:
    """Dataset view that reverses only the chunks used for init/goal extraction."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def load_chunk(self, episodes_idx, start, end):
        chunks = self.dataset.load_chunk(episodes_idx, start, end)
        reversed_chunks = []
        for chunk in chunks:
            reversed_chunk = {}
            for key, value in chunk.items():
                if torch.is_tensor(value):
                    reversed_chunk[key] = torch.flip(value, dims=[0])
                elif isinstance(value, np.ndarray) and value.ndim > 0:
                    reversed_chunk[key] = np.flip(value, axis=0).copy()
                else:
                    reversed_chunk[key] = value
            reversed_chunks.append(reversed_chunk)
        return reversed_chunks


def parse_bool_cli_value(name: str, value: str) -> str:
    value = value.lower()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be true or false when a value is provided")
    return "true" if value in {"true", "1", "yes"} else "false"


def normalize_cli_aliases() -> None:
    """Allow short aliases for Hydra overrides."""
    normalized = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg == "--reverse":
            normalized.append("eval.reverse=true")
        elif arg == "--no-reverse":
            normalized.append("eval.reverse=false")
        elif arg.startswith("--reverse="):
            value = arg.split("=", 1)[1].lower()
            reverse_value = parse_bool_cli_value("--reverse", value)
            normalized.append(f"eval.reverse={reverse_value}")
        elif arg in {"--early_agent_moving", "--early-agent-moving"}:
            normalized.append("eval.early_agent_moving=true")
        elif arg in {"--no-early_agent_moving", "--no-early-agent-moving"}:
            normalized.append("eval.early_agent_moving=false")
        elif arg.startswith("--early_agent_moving=") or arg.startswith(
            "--early-agent-moving="
        ):
            name, value = arg.split("=", 1)
            early_agent_moving = parse_bool_cli_value(name, value)
            normalized.append(f"eval.early_agent_moving={early_agent_moving}")
        else:
            normalized.append(arg)

    sys.argv[:] = normalized


def filter_early_agent_moving_indices(dataset, valid_indices, cfg):
    """Filter forward start candidates before optional reverse eval swapping."""
    goal_indices = valid_indices + int(cfg.eval.goal_offset_steps)
    state = dataset.get_col_data("state")
    proprio = dataset.get_col_data("proprio")

    start_block = state[valid_indices, 2:4]
    goal_block = state[goal_indices, 2:4]
    block_distance = np.linalg.norm(goal_block - start_block, axis=1)

    start_ball = proprio[valid_indices, :2]
    goal_ball = proprio[goal_indices, :2]
    start_ball_block_distance = np.linalg.norm(start_ball - start_block, axis=1)
    goal_ball_block_distance = np.linalg.norm(goal_ball - goal_block, axis=1)

    finite = (
        np.isfinite(block_distance)
        & np.isfinite(start_ball_block_distance)
        & np.isfinite(goal_ball_block_distance)
    )
    ball_moves_toward_block = start_ball_block_distance > goal_ball_block_distance

    stationary_eps = float(cfg.eval.get("early_agent_moving_stationary_eps", 1e-6))
    block_threshold = float(cfg.eval.get("early_agent_moving_block_threshold", 30.0))
    stationary = finite & (block_distance <= stationary_eps)
    micro_moving = finite & (block_distance < block_threshold)

    stationary_candidates = valid_indices[stationary & ball_moves_toward_block]
    relaxed_candidates = valid_indices[micro_moving & ball_moves_toward_block]
    if len(stationary_candidates) >= cfg.eval.num_eval:
        filtered_indices = stationary_candidates
        block_filter_mode = f"stationary <= {stationary_eps:g}"
    else:
        filtered_indices = relaxed_candidates
        block_filter_mode = f"micro < {block_threshold:g}"

    print(
        "early_agent_moving filter is ON. "
        f"stationary candidates: {len(stationary_candidates)}, "
        f"micro-moving candidates: {len(relaxed_candidates)}, "
        f"using {len(filtered_indices)} candidates ({block_filter_mode})."
    )

    if len(filtered_indices) < cfg.eval.num_eval:
        raise ValueError(
            "Not enough early_agent_moving candidates for evaluation: "
            f"{len(filtered_indices)} found, need {cfg.eval.num_eval}."
        )

    return filtered_indices


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(str(dataset_name)).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = data_root() / dataset_path
    if not dataset_path.suffix:
        dataset_path = dataset_path.with_suffix(".h5")
    dataset = swm.data.HDF5Dataset(
        path=dataset_path,
        keys_to_cache=cfg.dataset.keys_to_cache,
    )
    return dataset


def load_cost_model(cfg: DictConfig, checkpoint_root: str | Path):
    """Load a legacy LeWM pickle with its original source modules visible."""
    source_root = Path(cfg.model_source_root).expanduser().resolve()
    if not (source_root / "jepa.py").is_file():
        raise FileNotFoundError(f"LeWM source not found: {source_root}")
    sys.path.insert(0, str(source_root))
    try:
        return swm.policy.AutoCostModel(cfg.policy, cache_dir=checkpoint_root)
    finally:
        sys.path.pop(0)

def evaluate(cfg: DictConfig, model_loader=load_cost_model):
    """Run the shared dataset-replay MPC evaluation pipeline."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"
    reverse_eval = bool(cfg.eval.get("reverse", False))

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = get_dataset(cfg, cfg.dataset.get("stats", cfg.eval.dataset_name))
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model_checkpoint_root = cfg.get("cache_dir") or checkpoint_root()
        patch_transformers_vit_pickle_compat()
        model = model_loader(cfg, model_checkpoint_root)
        patch_loaded_model_compat(model)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    output_directory = cfg.output.get("directory")
    results_path = (
        Path(output_directory)
        if output_directory
        else (
            output_root()
            if cfg.policy != "random"
            else Path(__file__).parent
        )
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")
    if bool(cfg.eval.get("early_agent_moving", False)):
        valid_indices = filter_early_agent_moving_indices(dataset, valid_indices, cfg)

    if str(cfg.eval.get("start_selection", "random")) == "first_episodes":
        step_indices = dataset.get_col_data("step_idx")
        first_rows = [
            np.flatnonzero(
                (dataset.get_col_data(col_name) == episode) & (step_indices == 0)
            )[0]
            for episode in np.sort(ep_indices)[: int(cfg.eval.num_eval)]
            if max_start_idx_dict[episode] >= 0
        ]
        if len(first_rows) < int(cfg.eval.num_eval):
            raise ValueError("Not enough eligible episodes for first_episodes selection.")
        random_episode_indices = np.asarray(first_rows, dtype=np.int64)
    else:
        g = np.random.default_rng(cfg.seed)
        random_episode_indices = g.choice(
            len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
        )
        # sort increasingly to avoid issues with HDF5Dataset indexing
        random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    eval_dataset = ReverseChunkDataset(dataset) if reverse_eval else dataset
    video_path = (
        results_path / "reverse"
        if reverse_eval and results_path is not None
        else results_path
    )
    if reverse_eval:
        print(
            "Running reverse evaluation: init frame is original goal frame, "
            "goal frame is original init frame."
        )

    callables = prepare_cube_evaluation(
        world,
        OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
    )
    artifact_recorder = None
    if bool(cfg.output.get("save_artifacts", False)):
        artifact_recorder = PlanningArtifactRecorder(
            world, eval_dataset, eval_episodes, eval_start_idx,
            cfg.eval.goal_offset_steps, results_path / "artifacts", cfg.seed,
        )
        artifact_recorder.install()
    metrics = world.evaluate(
        dataset=eval_dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=callables,
        video=video_path,
    )
    metrics = add_cube_evaluation_metrics(world, metrics)
    metrics = add_insert_peg_evaluation_metrics(world, metrics)
    if artifact_recorder is not None:
        artifact_recorder.save(metrics)
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")

    return metrics, end_time - start_time


@hydra.main(
    version_base=None,
    config_path="../../resources/plan/lewm",
    config_name="pusht",
)
def run(cfg: DictConfig):
    """Run LeWM or random-policy evaluation."""
    return evaluate(cfg)


if __name__ == "__main__":
    normalize_cli_aliases()
    run()
