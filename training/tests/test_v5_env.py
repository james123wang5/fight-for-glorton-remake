from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from gymnasium.utils.env_checker import check_env

from training.v5_deployment import V5TrainedAIController
from training.v5_env import V5_OBSERVATION_SIZE, V5PeachEnv, encode_v5_observation
from training.v5_options import (
    PURPOSE_COUNT,
    Purpose,
    predicted_vertical_apex_y,
    purpose_action_mask,
)
from training.v5_runtime_observation import encode_v5_runtime_observation
from src.v5_web_deployment import WebV5Policy


class FakeV5Policy:
    observation_space = SimpleNamespace(shape=(V5_OBSERVATION_SIZE,))
    action_space = SimpleNamespace(n=PURPOSE_COUNT)

    def __init__(self) -> None:
        self.observation: np.ndarray | None = None
        self.mask: np.ndarray | None = None

    def predict(
        self,
        observation: np.ndarray,
        *,
        action_masks: np.ndarray,
        deterministic: bool = False,
    ):
        self.observation = observation.copy()
        self.mask = action_masks.copy()
        return np.asarray(np.flatnonzero(action_masks)[0], dtype=np.int64), None


class V5PurposeEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = V5PeachEnv(seed=61, max_episode_seconds=5, curriculum_strength=0.0)

    def tearDown(self) -> None:
        self.env.close()

    def test_discrete_purpose_contract_and_gym_contract(self) -> None:
        observation, info = self.env.reset(
            seed=61,
            options={"curriculum": "duel", "agent_slot": 0, "items_enabled": False},
        )
        self.assertEqual(observation.shape, (V5_OBSERVATION_SIZE,))
        self.assertEqual(self.env.action_masks().shape, (PURPOSE_COUNT,))
        self.assertEqual(info["observation_version"], "glorton-peach-purpose-v5")
        check_env(self.env, skip_render_check=True)

    def test_web_export_matches_desktop_observation_and_masked_action(self) -> None:
        from pathlib import Path

        from sb3_contrib import MaskablePPO

        root = Path(__file__).resolve().parents[2]
        desktop = MaskablePPO.load(
            str(root / "training/checkpoints/peach_purpose_v5/foundation_model.zip"),
            device="cpu",
        )
        web = WebV5Policy(root / "assets/ai/v5_purpose_policy.npz")
        self.env.reset(
            seed=91,
            options={"curriculum": "duel", "agent_slot": 0, "items_enabled": False},
        )
        slot = self.env.agent_slot
        fighter = self.env.runtime.fighters[slot]
        opponent = self.env.runtime.fighters[1 - slot]
        controller = self.env.intent_controllers[slot]
        common = {
            "episode_ticks": self.env._episode_ticks,
            "max_ticks": self.env.max_ticks,
            "spawns_swapped": self.env._spawns_swapped_by_slot[slot],
            "wall_stall_steps": controller.no_progress_steps,
        }
        expected = encode_v5_observation(
            self.env.runtime,
            fighter,
            opponent,
            controller,
            curriculum="duel",
            **common,
        )
        actual = encode_v5_runtime_observation(
            self.env.runtime,
            fighter,
            opponent,
            controller,
            **common,
        )
        np.testing.assert_array_equal(actual, expected)
        mask = purpose_action_mask(
            self.env.runtime,
            fighter,
            opponent,
            controller,
            curriculum="duel",
        )
        desktop_action = int(
            np.asarray(
                desktop.predict(expected, action_masks=mask, deterministic=True)[0]
            ).reshape(-1)[0]
        )
        self.assertEqual(web.predict(actual, mask), desktop_action)

    def test_navigation_lesson_crosses_fixed13_without_wasted_second_jump(self) -> None:
        self.env.reset(
            seed=2,
            options={"curriculum": "v5_navigation", "agent_slot": 0, "items_enabled": False},
        )
        first_mask = self.env.action_masks()
        self.assertTrue(first_mask[Purpose.NAVIGATE])
        self.assertEqual(np.flatnonzero(first_mask).tolist(), [int(Purpose.NAVIGATE)])
        info = {}
        for _ in range(20):
            mask = self.env.action_masks()
            action = Purpose.NAVIGATE if mask[Purpose.NAVIGATE] else Purpose.CONTINUE
            _observation, _reward, _terminated, truncated, info = self.env.step(int(action))
            if truncated:
                break
        self.assertTrue(info["lesson_success"])
        self.assertEqual(info["purpose_metrics"]["purposeful_second_jumps"], 0)
        self.assertEqual(info["purpose_metrics"]["jump_down_reversals"], 0)

    def test_air_chase_holds_horizontal_direction_and_hits_punch_air(self) -> None:
        self.env.reset(
            seed=3,
            options={"curriculum": "v5_air_chase", "agent_slot": 0, "items_enabled": False},
        )
        info = {}
        # Randomized velocity/height lessons may need one extra 100 ms
        # decision compared with the former fixed trajectory.
        for _ in range(24):
            mask = self.env.action_masks()
            action = Purpose.AIR_CHASE if mask[Purpose.AIR_CHASE] else Purpose.CONTINUE
            _observation, _reward, _terminated, truncated, info = self.env.step(int(action))
            if truncated:
                break
        self.assertTrue(info["lesson_success"])
        self.assertGreaterEqual(info["air_chase_hits"], 1)
        self.assertEqual(info["successful_attacks"]["punchAir"], 1)
        self.assertNotEqual(self.env.agent.xinc, 0.0)

    def test_air_chase_does_not_punch_itself_through_the_source_top_boundary(self) -> None:
        self.env.reset(
            seed=33,
            options={"curriculum": "duel", "agent_slot": 0, "items_enabled": False},
        )
        fighter = self.env.agent
        opponent = self.env.opponent
        fighter.pos.update(500.0, -180.0)
        fighter.prev_pos.update(fighter.pos)
        fighter.on_ground = False
        fighter.ground_platform = None
        fighter.jumpstate = 1
        fighter.yinc = -2.0
        opponent.pos.update(520.0, -190.0)
        opponent.prev_pos.update(opponent.pos)
        opponent.on_ground = False
        opponent.ground_platform = None
        controller = self.env.intent_controllers[self.env.agent_slot]

        controls = controller.begin_decision(
            Purpose.AIR_CHASE,
            fighter=fighter,
            opponent=opponent,
            action_mask=np.ones(PURPOSE_COUNT, dtype=bool),
        )

        self.assertLess(predicted_vertical_apex_y(fighter, fighter.yinc - 5.0), -200.0)
        self.assertFalse(any(control.get("punch_pressed") for control in controls))
        self.assertFalse(any(control.get("jump_pressed") for control in controls))
        self.assertEqual(controller.events.top_risk_preventions, 1)

    def test_hitstun_escape_is_buffered_and_fires_on_first_legal_decision(self) -> None:
        self.env.reset(
            seed=4,
            options={"curriculum": "v5_escape", "agent_slot": 0, "items_enabled": False},
        )
        info = {}
        for _ in range(8):
            mask = self.env.action_masks()
            action = (
                Purpose.HITSTUN_ESCAPE
                if mask[Purpose.HITSTUN_ESCAPE]
                else Purpose.CONTINUE
            )
            _observation, _reward, _terminated, truncated, info = self.env.step(int(action))
            if truncated:
                break
        self.assertTrue(info["lesson_success"])
        self.assertEqual(info["purpose_metrics"]["buffered_escapes"], 1)
        self.assertEqual(info["purpose_metrics"]["first_frame_escapes"], 1)
        self.assertNotEqual(self.env.agent.state, "thrown")

    def test_wall_blocks_air_chase_until_navigation_clears_it(self) -> None:
        self.env.reset(
            seed=5,
            options={"curriculum": "duel", "agent_slot": 0, "items_enabled": False},
        )
        by_name = {platform.name: platform for platform in self.env.runtime.stage.platforms}
        base = by_name["Fixed1"]
        wall = by_name["Fixed12"]
        self.env._place_fighter(self.env.agent, base, wall.rect.left - 55)
        self.env._place_fighter(self.env.opponent, base, wall.rect.right + 100)
        self.env.opponent.state = "thrown"
        self.env.opponent.on_ground = False
        self.env.opponent.ground_platform = None
        mask = self.env.action_masks()
        self.assertTrue(mask[Purpose.NAVIGATE])
        self.assertFalse(mask[Purpose.AIR_CHASE])

    def test_rocket_remains_available_for_a_valid_front_arc_but_not_behind(self) -> None:
        self.env.reset(
            seed=66,
            options={"curriculum": "duel", "agent_slot": 0, "items_enabled": False},
        )
        platform = self.env.agent.ground_platform or self.env._lesson_platform()
        self.env._place_fighter(self.env.agent, platform, float(platform.rect.centerx) - 50)
        self.env._place_fighter(self.env.opponent, platform, float(platform.rect.centerx) + 50)
        self.env.agent.facing = 1
        self.env.opponent.pos.y = self.env.agent.pos.y - 70
        self.env.opponent.prev_pos.update(self.env.opponent.pos)
        self.env.opponent.on_ground = False
        self.env.opponent.ground_platform = None
        self.assertTrue(self.env.action_masks()[Purpose.ROCKET])

        self.env.opponent.pos.x = self.env.agent.pos.x - 100
        self.env.opponent.prev_pos.update(self.env.opponent.pos)
        self.assertFalse(self.env.action_masks()[Purpose.ROCKET])

    def test_randomized_air_lessons_vary_but_stay_inside_source_bounds(self) -> None:
        starts: set[tuple[int, int, int, int]] = set()
        for seed in range(70, 76):
            self.env.reset(
                seed=seed,
                options={
                    "curriculum": "v5_air_chase",
                    "agent_slot": 0,
                    "items_enabled": False,
                },
            )
            starts.add(
                (
                    round(self.env.agent.pos.x),
                    round(self.env.opponent.pos.x),
                    round(self.env.opponent.pos.y),
                    round(self.env.opponent.yinc),
                )
            )
            self.assertGreater(self.env.agent.pos.y, self.env.runtime.stage.bounds.top)
            self.assertGreater(self.env.opponent.pos.y, self.env.runtime.stage.bounds.top)
        self.assertGreater(len(starts), 1)

    def test_scripted_probe_accepts_rare_legal_purposes_without_crashing_rollout(self) -> None:
        self.env.reset(
            seed=81,
            options={"curriculum": "duel", "agent_slot": 0, "items_enabled": False},
        )
        rocket_only = np.zeros(PURPOSE_COUNT, dtype=bool)
        rocket_only[Purpose.ROCKET] = True
        with patch.object(self.env, "_action_mask_for_slot", return_value=rocket_only):
            self.assertEqual(
                self.env._scripted_opponent_purpose("active"),
                int(Purpose.ROCKET),
            )

        land_only = np.zeros(PURPOSE_COUNT, dtype=bool)
        land_only[Purpose.LAND] = True
        with patch.object(self.env, "_action_mask_for_slot", return_value=land_only):
            self.assertEqual(
                self.env._scripted_opponent_purpose("melee"),
                int(Purpose.LAND),
            )

    def test_deployment_uses_v5_observation_and_purpose_mask(self) -> None:
        self.env.reset(
            seed=7,
            options={"curriculum": "duel", "agent_slot": 0, "items_enabled": False},
        )
        fake = FakeV5Policy()
        controller = V5TrainedAIController(
            self.env.runtime,
            self.env.agent,
            self.env.runtime.stage,
            "unused.zip",
            level=22,
            model=fake,
        )
        controller.controls_for_tick(self.env.runtime.fighters)
        self.assertIsNotNone(fake.observation)
        self.assertEqual(fake.observation.shape, (V5_OBSERVATION_SIZE,))
        self.assertEqual(fake.mask.shape, (PURPOSE_COUNT,))


if __name__ == "__main__":
    unittest.main()
