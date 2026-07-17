from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from .preflight_roster import TRAINABLE_ROSTER, preflight_roster
from .roster_jobs import plan_parallel_jobs


ROOT = Path(__file__).resolve().parents[1]


def _tail(path: Path, lines: int = 30) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def main() -> None:
    parser = argparse.ArgumentParser(description="预检并并发训练五个非桃子角色")
    parser.add_argument("--fighters", nargs="+", choices=TRAINABLE_ROSTER, default=TRAINABLE_ROSTER)
    parser.add_argument("--stage", choices=("Mogadishu",), default="Mogadishu")
    parser.add_argument("--run-id", default="roster_b1")
    parser.add_argument("--base-seed", type=int, default=20260716)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--foundation-steps", type=int, default=160_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--steps-per-level", type=int, default=60_000)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--eval-max-seconds", type=float, default=60.0)
    parser.add_argument("--autosave-steps", type=int, default=10_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    fighters = tuple(args.fighters)
    workers = max(1, min(int(args.workers), len(fighters)))

    if not args.skip_preflight:
        report = preflight_roster(
            fighters,
            stage_name=args.stage,
            seed=args.base_seed,
            rollout_decisions=40,
        )
        report_path = ROOT / "artifacts" / "training" / f"roster_preflight_{args.run_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"五角色预检通过: {report_path}")
    if args.preflight_only:
        return

    jobs = list(
        plan_parallel_jobs(
            fighters,
            stage_name=args.stage,
            run_id=args.run_id,
            base_seed=args.base_seed,
        )
    )
    pending = jobs[:]
    active: dict[int, tuple[Any, subprocess.Popen[str], Any, Path]] = {}
    failures: list[tuple[str, int, Path]] = []
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    def launch(job: Any) -> None:
        _checkpoint_dir, log_dir = job.checkpoint_dir(), job.log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "launcher.log"
        handle = log_path.open("a" if args.resume else "w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "training.train_roster",
            "--fighter",
            job.fighter_name,
            "--stage",
            job.stage_name,
            "--run-id",
            job.run_id,
            "--seed",
            str(job.seed),
            "--foundation-steps",
            str(args.foundation_steps),
            "--rounds",
            str(args.rounds),
            "--steps-per-level",
            str(args.steps_per_level),
            "--eval-episodes",
            str(args.eval_episodes),
            "--max-seconds",
            str(args.max_seconds),
            "--eval-max-seconds",
            str(args.eval_max_seconds),
            "--autosave-steps",
            str(args.autosave_steps),
            "--rollout-steps",
            str(args.rollout_steps),
            "--device",
            args.device,
            "--skip-preflight",
        ]
        if args.resume:
            command.append("--resume")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        active[process.pid] = (job, process, handle, log_path)
        print(f"启动 {job.fighter_name}: PID {process.pid}，日志 {log_path}")

    interrupted = False
    try:
        while pending or active:
            while pending and len(active) < workers:
                launch(pending.pop(0))
            for pid, (job, process, handle, log_path) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[pid]
                if code == 0:
                    print(f"完成 {job.fighter_name}")
                else:
                    failures.append((job.fighter_name, code, log_path))
                    print(f"失败 {job.fighter_name}: 退出码 {code}")
            if pending or active:
                time.sleep(0.5)
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到中断，通知所有角色保存当前检查点……")
        for _job, process, _handle, _log_path in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and any(
            process.poll() is None for _job, process, _handle, _log_path in active.values()
        ):
            time.sleep(0.25)
        for _job, process, handle, _log_path in active.values():
            if process.poll() is None:
                process.terminate()
            handle.close()

    if failures:
        print("\n以下任务失败：")
        for fighter, code, log_path in failures:
            print(f"\n--- {fighter} / {code} / {log_path} ---")
            print(_tail(log_path))
        raise SystemExit(1)
    if interrupted:
        print("候选均已请求保存。使用相同命令并加 --resume --skip-preflight 继续。")


if __name__ == "__main__":
    main()
