from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def configure_playable_ai() -> None:
    """Expose the current approved 21/22 models to normal RuntimeApp loading."""

    lightweight = ROOT / "assets" / "ai" / "online"
    level21 = lightweight / "peach" / "level21.npz"
    level22 = lightweight / "peach" / "level22.npz"
    if level21.is_file() and level22.is_file():
        os.environ.setdefault("GLORTON_AI21_MODEL", str(level21))
        os.environ.setdefault("GLORTON_AI22_MODEL", str(level22))
        os.environ.setdefault("GLORTON_AI_LIGHT", "1")
        return

    # Development fallback before lightweight exports exist.
    peach = ROOT / "training" / "checkpoints" / "peach_purpose_v5"
    roster = ROOT / "training" / "checkpoints" / "roster_v6"
    source21 = peach / "candidate_level21_model.zip"
    source22 = peach / "candidate_level22_model.zip"
    if source21.is_file() and source22.is_file():
        os.environ.setdefault("GLORTON_AI21_MODEL", str(source21))
        os.environ.setdefault("GLORTON_AI22_MODEL", str(source22))
        os.environ.setdefault("GLORTON_AI_V5", "1")
        if roster.is_dir():
            os.environ.setdefault("GLORTON_AI_ROSTER", "1")
            os.environ.setdefault("GLORTON_AI_ROSTER_ROOT", str(roster))
            os.environ.setdefault("GLORTON_AI_ROSTER_RUN_ID", "roster_b1")
            os.environ.setdefault("GLORTON_AI_ROSTER_STAGE", "Mogadishu")
