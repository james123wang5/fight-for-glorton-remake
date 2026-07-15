from __future__ import annotations

import math
from typing import Any

import numpy as np

from .tactical_input import COMBAT_LABELS, TacticalInputAdapter
from .v5_options import PurposefulOptionController
from .v5_runtime_helpers import (
    edge_danger,
    enemy_projectile_threats,
    is_offstage,
    route_distance,
    tactical_context,
    wall_probe,
)


V5_OBSERVATION_SIZE = 294
V5_FRAME_SKIP = 4
STATE_LABELS = ("stop", "goright", "goleft", "crouch", "thrown", "ko", "spawn", "dead")
ATTACK_LABELS = (
    "",
    "punchGround",
    "punchRun",
    "punchUp",
    "punchAir",
    "specialGround",
    "specialAir",
    "specialUp",
    "specialBackThrow",
    "koAttack",
)


def _nearest_platform(runtime: Any, fighter: Any) -> list[float]:
    bounds = runtime.stage.bounds
    if not runtime.stage.platforms:
        return [0.0] * 6

    def distance(platform: Any) -> float:
        gap_x = max(
            platform.rect.left - fighter.pos.x,
            0,
            fighter.pos.x - platform.rect.right,
        )
        return math.hypot(gap_x, platform.rect.top - fighter.pos.y)

    platform = min(runtime.stage.platforms, key=distance)
    gap_x = max(
        platform.rect.left - fighter.pos.x,
        0,
        fighter.pos.x - platform.rect.right,
    )
    return [
        gap_x / max(1.0, bounds.w),
        (platform.rect.top - fighter.pos.y) / max(1.0, bounds.h),
        platform.rect.w / max(1.0, bounds.w),
        float(bool(platform.moving)),
        float(fighter.ground_platform is platform),
        (platform.rect.centerx - fighter.pos.x) / max(1.0, bounds.w),
    ]


def _fighter_observation(runtime: Any, fighter: Any) -> list[float]:
    bounds = runtime.stage.bounds
    state = str(fighter.state)
    attack = str(fighter.current_attack)
    values = [
        (fighter.pos.x - bounds.centerx) / max(1.0, bounds.w / 2),
        (fighter.pos.y - bounds.centery) / max(1.0, bounds.h / 2),
        fighter.xinc / 30.0,
        fighter.yinc / 30.0,
        fighter.damage_amnt / 300.0,
        fighter.lives / 3.0,
        fighter.shield_size / 100.0,
        float(fighter.facing),
        float(fighter.attack_facing),
        float(bool(fighter.on_ground)),
        float(bool(fighter.has_control)),
        float(bool(fighter.invincible)),
        float(bool(fighter.shielded)),
        float(bool(fighter.spec_up_ok)),
        float(bool(fighter.out_of_camera)),
        float(bool(fighter.dead)),
        fighter.ctrl_loss / 1000.0,
        fighter.paralized / 1000.0,
        fighter.electrocuted_ms / 1000.0,
        fighter.spawn_invincible_ms / 3000.0,
        edge_danger(runtime, fighter),
    ]
    values.extend(float(fighter.jumpstate == index) for index in range(3))
    values.extend(float(state == label) for label in STATE_LABELS)
    values.extend(float(attack == label) for label in ATTACK_LABELS)
    attack_frames = max(
        1,
        len(fighter.animations.get(attack, {}).get("frames", [])) if attack else 1,
    )
    values.append(float(fighter.attack_frame) / attack_frames)
    values.extend(
        [
            (fighter.pos.x - bounds.left) / max(1.0, bounds.w),
            (bounds.right - fighter.pos.x) / max(1.0, bounds.w),
            (fighter.pos.y - bounds.top) / max(1.0, bounds.h),
            (bounds.bottom - fighter.pos.y) / max(1.0, bounds.h),
        ]
    )
    values.extend(_nearest_platform(runtime, fighter))
    return values


def _projectile_observation(runtime: Any, agent: Any, *, limit: int) -> list[float]:
    bounds = runtime.stage.bounds
    projectiles = [
        *((projectile, 0) for projectile in runtime.bullets),
        *((projectile, 1) for projectile in runtime.rockets),
        *((projectile, 2) for projectile in runtime.special_projectiles),
    ]
    projectiles.sort(
        key=lambda item: math.hypot(
            item[0].pos.x - agent.pos.x,
            item[0].pos.y - agent.pos.y,
        )
    )
    values: list[float] = []
    for projectile, kind_index in projectiles[:limit]:
        life = max(
            1,
            int(
                getattr(
                    projectile,
                    "life",
                    getattr(projectile, "config", {}).get("life_ms", 3000),
                )
            ),
        )
        values.extend(
            [
                (projectile.pos.x - agent.pos.x) / max(1.0, bounds.w),
                (projectile.pos.y - agent.pos.y) / max(1.0, bounds.h),
                float(getattr(projectile, "xinc", 0.0)) / 30.0,
                float(getattr(projectile, "yinc", 0.0)) / 30.0,
                float(getattr(projectile, "sender", None) is agent),
                float(getattr(projectile, "age", 0)) / life,
                float(kind_index == 0),
                float(kind_index == 1),
                float(kind_index == 2),
            ]
        )
    values.extend([0.0] * ((limit - min(limit, len(projectiles))) * 9))
    return values


def _base_observation(
    runtime: Any,
    agent: Any,
    opponent: Any,
    *,
    episode_ticks: int,
    max_ticks: int,
    spawns_swapped: bool,
) -> list[float]:
    bounds = runtime.stage.bounds
    values: list[float] = [
        episode_ticks / max(1, max_ticks),
        math.sin(runtime.stage_time_ms * math.tau / 10_000),
        math.cos(runtime.stage_time_ms * math.tau / 10_000),
        1.0 if spawns_swapped else -1.0,
    ]
    values.extend(_fighter_observation(runtime, agent))
    values.extend(_fighter_observation(runtime, opponent))
    values.extend(
        [
            (opponent.pos.x - agent.pos.x) / max(1.0, bounds.w),
            (opponent.pos.y - agent.pos.y) / max(1.0, bounds.h),
            (opponent.xinc - agent.xinc) / 30.0,
            (opponent.yinc - agent.yinc) / 30.0,
            math.hypot(opponent.pos.x - agent.pos.x, opponent.pos.y - agent.pos.y)
            / max(1.0, math.hypot(bounds.w, bounds.h)),
        ]
    )
    values.extend(_projectile_observation(runtime, agent, limit=3))
    if len(values) != 142:
        raise RuntimeError(f"v5 runtime base observation changed: {len(values)}")
    return values


def _environmental_combat_mask(runtime: Any, fighter: Any, opponent: Any) -> list[bool]:
    context = tactical_context(runtime, fighter, opponent)
    ready = bool(
        not fighter.dead
        and fighter.has_control
        and fighter.ctrl_loss <= 0
        and fighter.state not in {"ko", "thrown", "spawn", "dead"}
        and not fighter.current_attack
    )
    if fighter.shielded:
        return [True, False, False, False, False, False, False, False, True]
    if not ready:
        return [True] + [False] * (len(COMBAT_LABELS) - 1)
    distance = float(context["distance"])
    dx = float(context["dx"])
    dy = float(context["dy"])
    return [
        True,
        bool(fighter.jumpstate < 2),
        True,
        bool(distance <= 75 and abs(dy) <= 55 and bool(context["target_in_front"])),
        bool(fighter.on_ground and dy < -8 and abs(dx) <= 90),
        bool(
            fighter.on_ground
            and distance <= 30
            and abs(dy) <= 35
            and bool(context["target_behind"])
        ),
        bool(
            distance >= 45
            and bool(context["clear_shot"])
            and bool(context["target_in_front"])
            and abs(float(context["lead_y"])) <= 85
        ),
        bool(
            fighter.spec_up_ok
            and (bool(context["offstage"]) or bool(context["rocket_opportunity"]))
        ),
        True,
    ]


def _league_extras(runtime: Any, agent: Any, opponent: Any, adapter: TacticalInputAdapter) -> list[float]:
    bounds = runtime.stage.bounds
    diagonal = max(1.0, math.hypot(bounds.w, bounds.h))
    extras = adapter.features()
    items = [item for item in runtime.items if item.alive and item.state != 3]
    items.sort(key=lambda item: agent.pos.distance_to(item.pos))
    for item in items[:2]:
        kind = str(item.kind).lower()
        extras.extend(
            [
                (item.pos.x - agent.pos.x) / max(1.0, bounds.w),
                (item.pos.y - agent.pos.y) / max(1.0, bounds.h),
                float(kind == "mine"),
                float(kind == "grenade"),
                float(item.state) / 3.0,
                float(item.age_ms) / max(1.0, float(item.life_ms)),
                1.0,
            ]
        )
    extras.extend([0.0] * ((2 - min(2, len(items))) * 7))
    held = str(agent.current_item).lower()
    extras.extend([float(not held), float(held == "mine"), float(held == "grenade")])
    dx = float(opponent.pos.x - agent.pos.x)
    dy = float(opponent.pos.y - agent.pos.y)
    distance = math.hypot(dx, dy)
    target_is_behind = agent.facing * (agent.pos.x - opponent.pos.x) > 0
    extras.extend(
        [
            dx / max(1.0, bounds.w),
            dy / max(1.0, bounds.h),
            distance / diagonal,
            float(distance <= 24 and abs(dy) <= 30 and target_is_behind),
            float(dy < -18 and abs(dx) < 100),
            float(distance > 100 and abs(dy) < 70),
            float(is_offstage(runtime, agent)),
            float(bool(agent.spec_up_ok)),
        ]
    )
    return extras


def _tactical_extras(runtime: Any, agent: Any, opponent: Any, adapter: TacticalInputAdapter) -> list[float]:
    bounds = runtime.stage.bounds
    diagonal = max(1.0, math.hypot(bounds.w, bounds.h))
    extras = adapter.tactical_features()
    extras.extend(float(value) for value in _environmental_combat_mask(runtime, agent, opponent))
    threats = enemy_projectile_threats(runtime, agent, limit=2)
    for threat in threats:
        kind = str(threat["kind"])
        extras.extend(
            [
                float(threat["dx"]) / max(1.0, bounds.w),
                float(threat["dy"]) / max(1.0, bounds.h),
                float(threat["vx"]) / 30.0,
                float(threat["vy"]) / 30.0,
                float(threat["time_ticks"]) / 40.0,
                float(threat["miss"]) / diagonal,
                float(bool(threat["approaching"])),
                float(kind == "bullet"),
                float(kind == "rocket"),
                float(kind == "special"),
            ]
        )
    extras.extend([0.0] * ((2 - len(threats)) * 10))
    context = tactical_context(runtime, agent, opponent)
    extras.extend(
        [
            float(bool(context["clear_shot"])),
            float(bool(context["melee_threat"])),
            float(context["threat_score"]),
            float(context["lead_y"]) / max(1.0, bounds.h),
            float(context["intercept_ticks"]) / 60.0,
            float(float(context["distance"]) > 150),
        ]
    )
    return extras


def _navigation_extras(
    runtime: Any,
    fighter: Any,
    opponent: Any,
    adapter: TacticalInputAdapter,
    *,
    wall_stall_steps: int,
) -> list[float]:
    bounds = runtime.stage.bounds
    left = wall_probe(runtime, fighter, -1)
    right = wall_probe(runtime, fighter, 1)
    dx = float(opponent.pos.x - fighter.pos.x)
    dy = float(opponent.pos.y - fighter.pos.y)
    distance = math.hypot(dx, dy)
    diagonal = max(1.0, math.hypot(bounds.w, bounds.h))
    same_surface = bool(
        fighter.ground_platform is not None
        and fighter.ground_platform is opponent.ground_platform
    )
    values = [*adapter.v4_features()]
    for probe in (left, right):
        values.extend(
            [
                float(bool(probe["blocked"])),
                float(probe["distance"]) / 120.0,
                float(probe["top_delta"]) / max(1.0, bounds.h),
                float(probe["height"]) / max(1.0, bounds.h),
                float(bool(probe["moving"])),
            ]
        )
    values.extend(
        [
            float(same_surface),
            float(dy < -30.0),
            float(dy > 30.0),
            float(not bool(tactical_context(runtime, fighter, opponent)["clear_shot"])),
            route_distance(runtime, fighter, opponent) / diagonal,
            min(1.0, wall_stall_steps / 10.0),
            float(distance <= 75.0),
            float(75.0 < distance <= 180.0),
        ]
    )
    return values


def encode_v5_runtime_observation(
    runtime: Any,
    fighter: Any,
    opponent: Any,
    controller: PurposefulOptionController,
    *,
    episode_ticks: int,
    max_ticks: int,
    spawns_swapped: bool,
    wall_stall_steps: int = 0,
) -> np.ndarray:
    values = _base_observation(
        runtime,
        fighter,
        opponent,
        episode_ticks=episode_ticks,
        max_ticks=max_ticks,
        spawns_swapped=spawns_swapped,
    )
    values.extend(_league_extras(runtime, fighter, opponent, controller.adapter))
    values.extend(_tactical_extras(runtime, fighter, opponent, controller.adapter))
    values.extend(
        _navigation_extras(
            runtime,
            fighter,
            opponent,
            controller.adapter,
            wall_stall_steps=wall_stall_steps,
        )
    )
    values.extend(controller.features(fighter, opponent))
    if len(values) != V5_OBSERVATION_SIZE:
        raise RuntimeError(f"v5 runtime observation changed: {len(values)}")
    return np.clip(np.asarray(values, dtype=np.float32), -5.0, 5.0)
