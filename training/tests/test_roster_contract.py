from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from src.runtime import Stage
from src.simulation import BattleSimulation
from training.roster_contract import (
    FIGHTER_ORDER,
    MAX_SURFACE_NODES,
    STAGE_ORDER,
    StageNavigationGraph,
    action_availability,
    attack_timing,
    capability_report,
    encode_roster_context,
    make_training_match_config,
)
from training.roster_jobs import (
    TrainingScenario,
    claim_training_job,
    plan_parallel_jobs,
    prepare_job,
)
from training.roster_observation import (
    ROSTER_CONTEXT_SIZE,
    ROSTER_OBSERVATION_SIZE,
    encode_roster_observation,
)
from training.v5_env import V5_OBSERVATION_SIZE, V5PeachEnv


class RosterContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = make_training_match_config("PeachPlayer", "PeachPlayer", "Mogadishu")
        cls.simulation = BattleSimulation.headless(seed=20260716, match_config=config)
        cls.runtime = cls.simulation.runtime

    def configure(self, fighter_name: str, stage_name: str) -> tuple[object, object]:
        self.runtime.match_config = make_training_match_config(
            fighter_name, fighter_name, stage_name
        )
        self.runtime.stage = Stage(self.runtime.manifest, stage_name)
        self.simulation.reset(20260716)
        return self.runtime.fighters[0], self.runtime.fighters[1]

    def test_all_six_capability_profiles_classify_specials(self) -> None:
        report = capability_report(self.runtime.manifest)
        self.assertEqual(set(report), set(FIGHTER_ORDER))
        self.assertEqual(report["PeachPlayer"]["ground_special_mode"], "projectile")
        self.assertEqual(report["SBLPlayer"]["ground_special_mode"], "beam")
        self.assertEqual(report["TrashPlayer"]["ground_special_mode"], "lob_or_trap")
        self.assertEqual(report["CoffeePlayer"]["ground_special_mode"], "lob_or_trap")
        self.assertEqual(report["DefaultPlayer"]["ground_special_mode"], "projectile")
        self.assertEqual(report["AuberginePlayer"]["ground_special_mode"], "projectile")
        self.assertEqual(report["PeachPlayer"]["up_special_mode"], "projectile")
        self.assertEqual(report["TrashPlayer"]["up_special_mode"], "burst")
        for name in {"SBLPlayer", "CoffeePlayer", "DefaultPlayer", "AuberginePlayer"}:
            self.assertEqual(report[name]["up_special_mode"], "strike")
        self.assertFalse(report["AuberginePlayer"]["has_dodge"])
        self.assertTrue(report["PeachPlayer"]["has_dodge"])

    def test_dynamic_availability_and_attack_remaining_frames(self) -> None:
        for fighter_name in FIGHTER_ORDER:
            fighter, opponent = self.configure(fighter_name, "Mogadishu")
            platform = max(self.runtime.stage.platforms, key=lambda item: item.rect.w)
            fighter.on_ground = True
            fighter.ground_platform = platform
            fighter.pos.x = platform.rect.centerx
            fighter.pos.y = platform.rect.top
            fighter.state = "stop"
            fighter.has_control = True
            fighter.ctrl_loss = 0
            fighter.current_attack = ""
            fighter.shielded = False
            fighter.shield_size = 100
            fighter.spec_up_ok = True
            fighter.jumpstate = 0
            fighter.xinc = 0
            opponent.pos.update(fighter.pos.x - 15, fighter.pos.y)
            fighter.facing = 1
            grounded = action_availability(fighter, opponent)
            self.assertTrue(grounded.jump)
            self.assertTrue(grounded.ground_punch)
            self.assertTrue(grounded.uppercut)
            self.assertTrue(grounded.back_throw)
            self.assertTrue(grounded.ground_special)
            self.assertTrue(grounded.up_special)
            self.assertTrue(grounded.shield)

            fighter.on_ground = False
            fighter.ground_platform = None
            fighter.jumpstate = 1
            fighter.yinc = 0
            airborne = action_availability(fighter, opponent)
            self.assertTrue(airborne.double_jump)
            self.assertTrue(airborne.air_punch)
            self.assertTrue(airborne.air_special)

            fighter.current_attack = "punchAir"
            fighter.attack_frame = 2
            timing = attack_timing(fighter)
            self.assertEqual(timing.label, "punchAir")
            self.assertEqual(timing.remaining_frames, timing.total_frames - 2)

    def test_every_fighter_stage_pair_has_finite_context_and_graph(self) -> None:
        for stage_name in STAGE_ORDER:
            for fighter_name in FIGHTER_ORDER:
                with self.subTest(stage=stage_name, fighter=fighter_name):
                    fighter, opponent = self.configure(fighter_name, stage_name)
                    graph = StageNavigationGraph(self.runtime, fighter)
                    context = encode_roster_context(self.runtime, fighter, opponent)
                    self.assertLessEqual(len(graph.nodes), MAX_SURFACE_NODES)
                    self.assertEqual(len({node.name for node in graph.nodes}), len(graph.nodes))
                    self.assertGreater(len(graph.edges), 0)
                    self.assertEqual(context.shape, (ROSTER_CONTEXT_SIZE,))
                    self.assertTrue(np.isfinite(context).all())

    def test_mogadishu_floor_is_split_by_solid_buildings(self) -> None:
        fighter, _opponent = self.configure("PeachPlayer", "Mogadishu")
        graph = StageNavigationGraph(self.runtime, fighter)
        fixed1 = [node for node in graph.nodes if node.platform_name == "Fixed1"]
        self.assertGreaterEqual(len(fixed1), 2)
        self.assertTrue(any(node.platform_name == "Fixed12" for node in graph.nodes))
        self.assertTrue(any(node.platform_name == "Fixed13" for node in graph.nodes))

    def test_v6_keeps_the_frozen_v5_prefix_exact(self) -> None:
        environment = V5PeachEnv(seed=20260716, curriculum_strength=0.0)
        environment.reset(seed=20260716, options={"curriculum": "duel"})
        controller = environment.intent_controllers[environment.agent_slot]
        legacy = environment._current_observation(environment.agent_slot)
        extended = encode_roster_observation(
            environment.runtime,
            environment.agent,
            environment.opponent,
            controller,
            episode_ticks=environment._episode_ticks,
            max_ticks=environment.max_ticks,
            spawns_swapped=environment._spawns_swapped_by_slot[environment.agent_slot],
            curriculum=environment.curriculum,
            wall_stall_steps=controller.no_progress_steps,
        )
        self.assertEqual(legacy.shape, (V5_OBSERVATION_SIZE,))
        self.assertEqual(extended.shape, (ROSTER_OBSERVATION_SIZE,))
        np.testing.assert_array_equal(extended[:V5_OBSERVATION_SIZE], legacy)
        environment.close()


class RosterJobIsolationTests(unittest.TestCase):
    def test_parallel_jobs_have_deterministic_unique_seeds_and_paths(self) -> None:
        jobs = plan_parallel_jobs(
            ("DefaultPlayer", "TrashPlayer", "CoffeePlayer"),
            stage_name="Mogadishu",
            run_id="phase_b1",
            base_seed=20260716,
        )
        self.assertEqual(len({job.seed for job in jobs}), len(jobs))
        self.assertEqual(len({job.scenario_slug for job in jobs}), len(jobs))
        self.assertEqual(jobs, plan_parallel_jobs(
            ("DefaultPlayer", "TrashPlayer", "CoffeePlayer"),
            stage_name="Mogadishu",
            run_id="phase_b1",
            base_seed=20260716,
        ))

    def test_same_job_cannot_be_claimed_twice_or_reused_for_another_seed(self) -> None:
        scenario = TrainingScenario(
            "TrashPlayer", "TrashPlayer", "Mogadishu", "lock_test", 101
        )
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            with claim_training_job(scenario, root) as (checkpoint_dir, log_dir):
                self.assertTrue((checkpoint_dir / ".train.lock").is_file())
                self.assertTrue(log_dir.is_dir())
                with self.assertRaises(RuntimeError):
                    with claim_training_job(scenario, root):
                        pass
            self.assertFalse((checkpoint_dir / ".train.lock").exists())
            changed_seed = TrainingScenario(
                "TrashPlayer", "TrashPlayer", "Mogadishu", "lock_test", 202
            )
            with self.assertRaises(RuntimeError):
                prepare_job(changed_seed, root)


if __name__ == "__main__":
    unittest.main()
