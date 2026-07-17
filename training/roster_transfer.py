from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .roster_observation import ROSTER_OBSERVATION_SIZE
from .v5_env import V5_OBSERVATION_SIZE


EXPANDED_INPUT_KEYS = {
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.value_net.0.weight",
}


def new_roster_model(
    MaskablePPO: Any,
    env: Any,
    *,
    seed: int,
    device: str,
    log_dir: Path,
    rollout_steps: int = 1024,
) -> Any:
    rollout_steps = max(64, int(rollout_steps))
    batch_size = next(
        value
        for value in (256, 128, 64, 32, 16, 8, 4, 2, 1)
        if value <= rollout_steps and rollout_steps % value == 0
    )
    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=2.0e-4,
        n_steps=rollout_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.998,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0015,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=str(log_dir),
    )


def copy_v5_policy_weights(source_model: Any, target_model: Any) -> dict[str, Any]:
    """Copy v5 exactly and initialize all 186 new input columns to zero."""

    source_shape = tuple(source_model.observation_space.shape)
    target_shape = tuple(target_model.observation_space.shape)
    if source_shape != (V5_OBSERVATION_SIZE,):
        raise ValueError(f"source must use frozen v5 observation, got {source_shape}")
    if target_shape != (ROSTER_OBSERVATION_SIZE,):
        raise ValueError(f"target must use roster v6 observation, got {target_shape}")
    if source_model.action_space.n != target_model.action_space.n:
        raise ValueError("v5 and v6 action counts differ")

    source = source_model.policy.state_dict()
    target = target_model.policy.state_dict()
    copied: list[str] = []
    expanded: list[str] = []
    with torch.no_grad():
        for key, target_value in target.items():
            source_value = source.get(key)
            if source_value is None:
                continue
            if source_value.shape == target_value.shape:
                target_value.copy_(source_value)
                copied.append(key)
                continue
            if key in EXPANDED_INPUT_KEYS and tuple(source_value.shape) == (
                target_value.shape[0],
                V5_OBSERVATION_SIZE,
            ):
                target_value.zero_()
                target_value[:, :V5_OBSERVATION_SIZE].copy_(source_value)
                expanded.append(key)
                continue
            raise ValueError(
                f"unsupported policy shape migration for {key}: "
                f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
            )
    target_model.policy.load_state_dict(target, strict=True)
    return {
        "source_observation_size": V5_OBSERVATION_SIZE,
        "target_observation_size": ROSTER_OBSERVATION_SIZE,
        "copied_tensors": tuple(copied),
        "expanded_input_tensors": tuple(expanded),
        "new_input_columns_initialized_to_zero": ROSTER_OBSERVATION_SIZE
        - V5_OBSERVATION_SIZE,
    }


def initialize_roster_from_v5(
    MaskablePPO: Any,
    *,
    source_path: Path,
    env: Any,
    seed: int,
    device: str,
    log_dir: Path,
    rollout_steps: int = 1024,
) -> tuple[Any, dict[str, Any]]:
    source = MaskablePPO.load(str(source_path), device="cpu")
    target = new_roster_model(
        MaskablePPO,
        env,
        seed=seed,
        device=device,
        log_dir=log_dir,
        rollout_steps=rollout_steps,
    )
    report = copy_v5_policy_weights(source, target)
    report["source_path"] = str(source_path)
    return target, report
