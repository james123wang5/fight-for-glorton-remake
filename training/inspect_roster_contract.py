from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runtime import Stage, load_manifest
from src.simulation import BattleSimulation

from .roster_contract import (
    FIGHTER_ORDER,
    STAGE_ORDER,
    StageNavigationGraph,
    capability_report,
    encode_roster_context,
    make_training_match_config,
)
from .roster_jobs import plan_parallel_jobs


def build_report(
    fighters: tuple[str, ...],
    stages: tuple[str, ...],
    *,
    run_id: str,
    base_seed: int,
) -> dict[str, object]:
    manifest = load_manifest()
    first_fighter = fighters[0]
    first_stage = stages[0]
    simulation = BattleSimulation.headless(
        seed=base_seed,
        match_config=make_training_match_config(first_fighter, first_fighter, first_stage),
    )
    runtime = simulation.runtime
    scenarios: dict[str, object] = {}
    for stage_name in stages:
        for fighter_name in fighters:
            config = make_training_match_config(fighter_name, fighter_name, stage_name)
            runtime.match_config = config
            runtime.stage = Stage(runtime.manifest, stage_name)
            simulation.reset(base_seed)
            agent, opponent = runtime.fighters[:2]
            graph = StageNavigationGraph(runtime, agent)
            context = encode_roster_context(runtime, agent, opponent)
            key = f"{fighter_name}/{stage_name}"
            scenarios[key] = {
                "surface_nodes": len(graph.nodes),
                "broad_edges": len(graph.edges),
                "dynamic_edges": sum(edge.dynamic for edge in graph.edges),
                "context_size": int(context.shape[0]),
                "finite": bool((context == context).all()),
            }
    jobs = plan_parallel_jobs(
        fighters,
        stage_name=stages[0],
        run_id=run_id,
        base_seed=base_seed,
    )
    return {
        "schema": "glorton-roster-contract-a1",
        "capabilities": capability_report(manifest),
        "scenarios": scenarios,
        "parallel_jobs": [
            {
                **job.manifest(),
                "checkpoint_dir": str(job.checkpoint_dir()),
                "log_dir": str(job.log_dir()),
            }
            for job in jobs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查全角色/地图训练数据协议")
    parser.add_argument("--fighters", nargs="+", choices=FIGHTER_ORDER, default=FIGHTER_ORDER)
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=STAGE_ORDER)
    parser.add_argument("--run-id", default="first_mogadishu")
    parser.add_argument("--base-seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(
        tuple(args.fighters),
        tuple(args.stages),
        run_id=args.run_id,
        base_seed=args.base_seed,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
