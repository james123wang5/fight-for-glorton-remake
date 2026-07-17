from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from training.preflight_roster import TRAINABLE_ROSTER, preflight_fighter
from training.play_roster_battle import main as play_roster_main
from training.roster_deployment import RosterTrainedAIController
from training.roster_env import RosterPurposeEnv
from training.roster_observation import ROSTER_OBSERVATION_SIZE
from training.roster_transfer import copy_v5_policy_weights, new_roster_model
from training.v5_env import V5_OBSERVATION_SIZE
from training.v5_options import PURPOSE_COUNT, Purpose


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "training"
    / "checkpoints"
    / "peach_purpose_v5"
    / "champion_level22_model.zip"
)


class FakeRosterPolicy:
    observation_space = SimpleNamespace(shape=(ROSTER_OBSERVATION_SIZE,))
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


class RosterPhaseBEnvironmentTests(unittest.TestCase):
    def test_playable_launcher_restores_native_window_and_keeps_peach_v5(self) -> None:
        observed: dict[str, str | None] = {}

        def fake_run_game() -> None:
            for name in (
                "SDL_VIDEODRIVER",
                "SDL_AUDIODRIVER",
                "GLORTON_AI_ROSTER",
                "GLORTON_AI_V5",
                "GLORTON_AI21_MODEL",
                "GLORTON_AI22_MODEL",
            ):
                observed[name] = os.environ.get(name)

        environment = {
            "SDL_VIDEODRIVER": "dummy",
            "SDL_AUDIODRIVER": "dummy",
        }
        with (
            patch.dict(os.environ, environment, clear=False),
            patch.object(sys, "argv", ["training.play_roster_battle"]),
            patch("src.runtime.main", side_effect=fake_run_game),
        ):
            play_roster_main()

        self.assertIsNone(observed["SDL_VIDEODRIVER"])
        self.assertIsNone(observed["SDL_AUDIODRIVER"])
        self.assertEqual(observed["GLORTON_AI_ROSTER"], "1")
        self.assertEqual(observed["GLORTON_AI_V5"], "1")
        self.assertTrue(str(observed["GLORTON_AI21_MODEL"]).endswith("champion_level21_model.zip"))
        self.assertTrue(str(observed["GLORTON_AI22_MODEL"]).endswith("champion_level22_model.zip"))

    def test_each_new_role_passes_ground_up_and_rollout_preflight(self) -> None:
        for index, fighter_name in enumerate(TRAINABLE_ROSTER):
            with self.subTest(fighter=fighter_name):
                report = preflight_fighter(
                    fighter_name,
                    seed=20260716 + index * 100,
                    rollout_decisions=4,
                )
                self.assertTrue(report["passed"])
                self.assertTrue(report["ground_special"]["started"])
                self.assertTrue(report["up_special"]["started"])
                self.assertTrue(report["finite_rewards"])

    def test_role_environment_never_changes_the_normal_game_manifest(self) -> None:
        env = RosterPurposeEnv(
            fighter_name="TrashPlayer",
            seed=17,
            curriculum_strength=0.0,
            max_episode_seconds=2.0,
        )
        try:
            observation, info = env.reset(
                seed=17,
                options={"curriculum": "duel", "items_enabled": False},
            )
            self.assertEqual(env.agent.fighter_name, "TrashPlayer")
            self.assertEqual(observation.shape, (ROSTER_OBSERVATION_SIZE,))
            self.assertTrue(np.isfinite(observation).all())
            self.assertEqual(info["fighter_name"], "TrashPlayer")
            self.assertFalse(env.runtime.match_config["players"][0]["computer"])
        finally:
            env.close()

    def test_rare_empty_base_mask_gets_state_appropriate_safe_fallback(self) -> None:
        env = RosterPurposeEnv(
            fighter_name="CoffeePlayer",
            seed=23,
            curriculum_strength=0.0,
            max_episode_seconds=2.0,
        )
        try:
            env.reset(
                seed=23,
                options={"curriculum": "roster_special", "agent_slot": 0},
            )
            env.agent.has_control = False
            with patch(
                "training.roster_options.purpose_action_mask",
                return_value=np.zeros(PURPOSE_COUNT, dtype=bool),
            ):
                mask = env.action_masks()
            self.assertEqual(mask.shape, (PURPOSE_COUNT,))
            self.assertEqual(int(mask.sum()), 1)
            self.assertTrue(mask[Purpose.CHASE])

            env.agent.state = "thrown"
            env.agent.ctrl_loss = 100
            with patch(
                "training.roster_options.purpose_action_mask",
                return_value=np.zeros(PURPOSE_COUNT, dtype=bool),
            ):
                mask = env.action_masks()
            self.assertEqual(int(mask.sum()), 1)
            self.assertTrue(mask[Purpose.HITSTUN_ESCAPE])
        finally:
            env.close()

    def test_rendered_controller_uses_v6_observation_and_role_mask(self) -> None:
        env = RosterPurposeEnv(
            fighter_name="SBLPlayer",
            seed=31,
            curriculum_strength=0.0,
            max_episode_seconds=2.0,
        )
        try:
            env.reset(
                seed=31,
                options={"curriculum": "duel", "agent_slot": 0},
            )
            fake = FakeRosterPolicy()
            controller = RosterTrainedAIController(
                env.runtime,
                env.agent,
                env.runtime.stage,
                "unused.zip",
                level=22,
                model=fake,
            )
            controller.controls_for_tick(env.runtime.fighters)
            self.assertIsNotNone(fake.observation)
            self.assertEqual(fake.observation.shape, (ROSTER_OBSERVATION_SIZE,))
            self.assertEqual(fake.mask.shape, (PURPOSE_COUNT,))
            self.assertTrue(fake.mask.any())
        finally:
            env.close()

    def test_rendered_controller_breaks_a_mutual_static_navigation_loop(self) -> None:
        env = RosterPurposeEnv(
            fighter_name="AuberginePlayer",
            seed=37,
            curriculum_strength=0.0,
            max_episode_seconds=2.0,
        )
        try:
            env.reset(
                seed=37,
                options={"curriculum": "duel", "agent_slot": 0},
            )
            controller = RosterTrainedAIController(
                env.runtime,
                env.agent,
                env.runtime.stage,
                "unused.zip",
                level=22,
                model=FakeRosterPolicy(),
            )
            # Do not advance physics: this deliberately imitates two fighters
            # whose repeated navigation plans produce no movement or attack.
            for _tick in range(80):
                controller.controls_for_tick(env.runtime.fighters)
            self.assertGreaterEqual(controller.stalemate_breaks, 1)
            self.assertGreaterEqual(controller.option.events.forced_replans, 1)
        finally:
            env.close()


class RosterWeightTransferTests(unittest.TestCase):
    @unittest.skipUnless(SOURCE.is_file(), "frozen v5 champion is not installed")
    def test_v5_outputs_are_exact_before_new_context_learns(self) -> None:
        from sb3_contrib import MaskablePPO

        source = MaskablePPO.load(str(SOURCE), device="cpu")
        env = RosterPurposeEnv(
            fighter_name="DefaultPlayer",
            seed=29,
            curriculum_strength=0.0,
            max_episode_seconds=2.0,
        )
        try:
            with tempfile.TemporaryDirectory() as temp_name:
                target = new_roster_model(
                    MaskablePPO,
                    env,
                    seed=29,
                    device="cpu",
                    log_dir=Path(temp_name),
                    rollout_steps=64,
                )
                report = copy_v5_policy_weights(source, target)
                self.assertEqual(report["new_input_columns_initialized_to_zero"], 186)
                source_state = source.policy.state_dict()
                target_state = target.policy.state_dict()
                for key, source_value in source_state.items():
                    target_value = target_state[key]
                    if key in report["expanded_input_tensors"]:
                        np.testing.assert_array_equal(
                            target_value[:, :V5_OBSERVATION_SIZE].cpu().numpy(),
                            source_value.cpu().numpy(),
                        )
                        self.assertEqual(
                            int(np.count_nonzero(target_value[:, V5_OBSERVATION_SIZE:].cpu().numpy())),
                            0,
                        )
                    else:
                        np.testing.assert_array_equal(
                            target_value.cpu().numpy(), source_value.cpu().numpy()
                        )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
