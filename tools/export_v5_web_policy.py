from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "training" / "checkpoints" / "peach_purpose_v5" / "foundation_model.zip"
DEFAULT_OUTPUT = ROOT / "assets" / "ai" / "v5_purpose_policy.npz"


def export(model_path: Path, output_path: Path) -> None:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit("请使用 .venv-train/bin/python 运行导出器") from exc
    model = MaskablePPO.load(str(model_path), device="cpu")
    policy = model.policy
    layers = policy.mlp_extractor.policy_net
    tensors = {
        "w1": layers[0].weight.detach().cpu().numpy().astype(np.float32),
        "b1": layers[0].bias.detach().cpu().numpy().astype(np.float32),
        "w2": layers[2].weight.detach().cpu().numpy().astype(np.float32),
        "b2": layers[2].bias.detach().cpu().numpy().astype(np.float32),
        "wa": policy.action_net.weight.detach().cpu().numpy().astype(np.float32),
        "ba": policy.action_net.bias.detach().cpu().numpy().astype(np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **tensors)
    print(f"Exported web v5 policy: {output_path} ({output_path.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出无需Torch的网页v5策略")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.model.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
