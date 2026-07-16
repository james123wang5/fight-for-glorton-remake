from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .human_replay import build_human_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "training" / "checkpoints" / "peach_purpose_v5"
DEFAULT_REPLAYS = ROOT / "training" / "replays" / "human_v5"


def _accuracy(policy: Any, observations: Any, actions: Any, masks: Any) -> float:
    import torch

    if int(actions.shape[0]) == 0:
        return 0.0
    with torch.no_grad():
        distribution = policy.get_distribution(observations, action_masks=masks)
        predicted = distribution.distribution.probs.argmax(dim=1)
    return float((predicted == actions).float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从可验证的v2真人录像小步模仿，生成独立22级候选模型"
    )
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_DIRECTORY / "champion_level22_model.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DIRECTORY / "human_candidate_level22_model.zip",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--anchor", type=float, default=0.15)
    parser.add_argument("--entropy", type=float, default=0.001)
    parser.add_argument("--minimum-examples", type=int, default=64)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--wins-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    replay_dir = args.replay_dir.expanduser().resolve()
    paths = sorted(replay_dir.glob("human_vs_ai_*.json"))[-max(1, args.max_files) :]
    dataset = build_human_dataset(paths, wins_only=args.wins_only)
    print(f"可学习v2录像: {len(dataset.accepted_files)}，跳过: {len(dataset.skipped_files)}")
    for path, reason in dataset.skipped_files.items():
        print(f"- 跳过 {Path(path).name}: {reason}")
    print(f"真人目的样本: {dataset.size} / {dict(dataset.action_counts)}")
    if dataset.size < max(1, args.minimum_examples):
        raise SystemExit(
            f"有效样本不足 {args.minimum_examples}；请先用 --record-human 完成更多对局。"
        )

    try:
        import torch
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "缺少训练依赖，请运行: .venv-train/bin/python -m pip install -r requirements-training.txt"
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    base = args.base.expanduser().resolve()
    if not base.is_file():
        raise SystemExit(f"找不到冻结22级基础模型: {base}")
    model = MaskablePPO.load(str(base), device="cpu")
    reference = MaskablePPO.load(str(base), device="cpu")
    policy = model.policy
    reference.policy.set_training_mode(False)
    device = policy.device

    observations = torch.as_tensor(dataset.observations, device=device)
    actions = torch.as_tensor(dataset.actions, device=device, dtype=torch.long)
    masks = torch.as_tensor(dataset.masks, device=device, dtype=torch.bool)
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(dataset.size, generator=generator)
    validation_size = max(1, dataset.size // 10) if dataset.size >= 20 else 0
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    if training_indices.numel() == 0:
        training_indices = permutation
        validation_indices = permutation[:0]

    counts = torch.bincount(actions[training_indices], minlength=int(model.action_space.n)).float()
    class_weights = torch.zeros_like(counts)
    present = counts > 0
    class_weights[present] = counts[present].rsqrt()
    class_weights[present] /= class_weights[present].mean()
    with torch.no_grad():
        reference_distribution = reference.policy.get_distribution(
            observations,
            action_masks=masks,
        )
        reference_probs = reference_distribution.distribution.probs.detach().clamp_min(1e-8)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    validation_slice = validation_indices if validation_indices.numel() else training_indices
    before_accuracy = _accuracy(
        policy,
        observations[validation_slice],
        actions[validation_slice],
        masks[validation_slice],
    )
    policy.set_training_mode(True)
    losses: list[float] = []
    for epoch in range(max(1, args.epochs)):
        shuffled = training_indices[
            torch.randperm(training_indices.numel(), generator=generator)
        ]
        epoch_losses: list[float] = []
        for start in range(0, shuffled.numel(), max(1, args.batch_size)):
            indices = shuffled[start : start + max(1, args.batch_size)]
            _values, log_prob, entropy = policy.evaluate_actions(
                observations[indices],
                actions[indices],
                action_masks=masks[indices],
            )
            sample_weights = class_weights[actions[indices]]
            imitation = -(log_prob * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
            distribution = policy.get_distribution(
                observations[indices],
                action_masks=masks[indices],
            )
            probs = distribution.distribution.probs.clamp_min(1e-8)
            anchor = (
                reference_probs[indices]
                * (reference_probs[indices].log() - probs.log())
            ).sum(dim=1).mean()
            entropy_value = entropy.mean() if entropy is not None else torch.zeros((), device=device)
            loss = imitation + max(0.0, args.anchor) * anchor - max(0.0, args.entropy) * entropy_value
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        losses.append(mean_loss)
        print(f"模仿 epoch {epoch + 1}/{max(1, args.epochs)} loss={mean_loss:.5f}")

    policy.set_training_mode(False)
    after_accuracy = _accuracy(
        policy,
        observations[validation_slice],
        actions[validation_slice],
        masks[validation_slice],
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output.with_suffix("")))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(base),
        "output_model": str(output),
        "accepted_files": list(dataset.accepted_files),
        "skipped_files": dict(dataset.skipped_files),
        "examples": dataset.size,
        "action_counts": dict(dataset.action_counts),
        "epochs": max(1, args.epochs),
        "learning_rate": args.learning_rate,
        "anchor": args.anchor,
        "validation_accuracy_before": before_accuracy,
        "validation_accuracy_after": after_accuracy,
        "losses": losses,
        "champion_overwritten": False,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"真人模仿候选: {output}")
    print(f"验证准确率: {before_accuracy:.1%} -> {after_accuracy:.1%}")
    print("冻结22级冠军未改写；下一步用该候选进行一轮联赛巩固。")


if __name__ == "__main__":
    main()
