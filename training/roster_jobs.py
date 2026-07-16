from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import socket
from typing import Iterator, Mapping

from .roster_contract import FIGHTER_ORDER, STAGE_ORDER
from .roster_observation import ROSTER_OBSERVATION_VERSION


ROOT = Path(__file__).resolve().parents[1]
ROSTER_ACTION_VERSION = "glorton-purpose-v5-14"


def _slug(value: str) -> str:
    return value.removesuffix("Player").lower().replace("_", "-")


def deterministic_job_seed(base_seed: int, fighter_name: str, stage_name: str) -> int:
    payload = f"{int(base_seed)}:{fighter_name}:{stage_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFF_FFFF


@dataclass(frozen=True)
class TrainingScenario:
    fighter_name: str
    opponent_fighter_name: str
    stage_name: str
    run_id: str
    seed: int

    def __post_init__(self) -> None:
        if self.fighter_name not in FIGHTER_ORDER:
            raise ValueError(f"unknown fighter: {self.fighter_name}")
        if self.opponent_fighter_name not in FIGHTER_ORDER:
            raise ValueError(f"unknown opponent fighter: {self.opponent_fighter_name}")
        if self.stage_name not in STAGE_ORDER:
            raise ValueError(f"unknown stage: {self.stage_name}")
        if not self.run_id or any(character in self.run_id for character in "/\\\0"):
            raise ValueError("run_id must be a non-empty path-safe name")

    @property
    def scenario_slug(self) -> str:
        versus = (
            _slug(self.fighter_name)
            if self.fighter_name == self.opponent_fighter_name
            else f"{_slug(self.fighter_name)}-vs-{_slug(self.opponent_fighter_name)}"
        )
        return f"{versus}/{self.stage_name.lower()}/{self.run_id}"

    def checkpoint_dir(self, root: Path = ROOT) -> Path:
        return root / "training/checkpoints/roster_v6" / self.scenario_slug

    def log_dir(self, root: Path = ROOT) -> Path:
        return root / "training/logs/roster_v6" / self.scenario_slug

    def manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "scenario_slug": self.scenario_slug,
            "observation_version": ROSTER_OBSERVATION_VERSION,
            "action_version": ROSTER_ACTION_VERSION,
        }


def plan_parallel_jobs(
    fighters: tuple[str, ...],
    *,
    stage_name: str,
    run_id: str,
    base_seed: int,
) -> tuple[TrainingScenario, ...]:
    if len(set(fighters)) != len(fighters):
        raise ValueError("parallel fighter list contains duplicates")
    return tuple(
        TrainingScenario(
            fighter_name=fighter,
            opponent_fighter_name=fighter,
            stage_name=stage_name,
            run_id=run_id,
            seed=deterministic_job_seed(base_seed, fighter, stage_name),
        )
        for fighter in fighters
    )


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_job(scenario: TrainingScenario, root: Path = ROOT) -> tuple[Path, Path]:
    checkpoint_dir = scenario.checkpoint_dir(root)
    log_dir = scenario.log_dir(root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "scenario.json"
    expected = scenario.manifest()
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current != expected:
            raise RuntimeError(
                f"scenario mismatch in {manifest_path}; use a different run_id"
            )
    else:
        _atomic_json(manifest_path, expected)
    return checkpoint_dir, log_dir


@contextmanager
def claim_training_job(
    scenario: TrainingScenario,
    root: Path = ROOT,
) -> Iterator[tuple[Path, Path]]:
    checkpoint_dir, log_dir = prepare_job(scenario, root)
    lock_path = checkpoint_dir / ".train.lock"
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario.manifest(),
    }
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        owner = lock_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"training job is already claimed: {lock_path}\n{owner}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        yield checkpoint_dir, log_dir
    finally:
        lock_path.unlink(missing_ok=True)
