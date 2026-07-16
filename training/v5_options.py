from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np

from .tactical_input import TacticalInputAdapter
from .v5_runtime_helpers import is_offstage as _is_offstage
from .v5_runtime_helpers import tactical_context, wall_probe


class Purpose(IntEnum):
    CONTINUE = 0
    CHASE = 1
    NAVIGATE = 2
    MELEE = 3
    AIR_CHASE = 4
    ANTI_AIR = 5
    BACK_THROW = 6
    AIMED_SHOT = 7
    ROCKET = 8
    HITSTUN_ESCAPE = 9
    EVADE = 10
    RECOVER = 11
    SHIELD = 12
    LAND = 13


PURPOSE_LABELS = tuple(item.name.lower() for item in Purpose)
PURPOSE_COUNT = len(PURPOSE_LABELS)
TOP_SURVIVAL_MARGIN = 12.0


@dataclass(frozen=True)
class RoutePlan:
    target_x: float
    target_y: float
    takeoff_x: float
    direction: int
    requires_jump: bool
    blocked: bool
    obstacle: Any | None
    target_platform: Any | None
    path: tuple[str, ...]
    cost: float


def _horizontal_gap(left: Any, right: Any) -> float:
    if left.rect.right < right.rect.left:
        return float(right.rect.left - left.rect.right)
    if right.rect.right < left.rect.left:
        return float(left.rect.left - right.rect.right)
    return 0.0


class StageNavigator:
    """Small platform graph plus a local obstacle waypoint for Mogadishu."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.platforms = list(runtime.stage.platforms)
        self.by_name = {platform.name: platform for platform in self.platforms}
        self.graph = self._build_graph()

    def _build_graph(self) -> dict[str, list[tuple[str, float]]]:
        graph: dict[str, list[tuple[str, float]]] = {
            platform.name: [] for platform in self.platforms
        }
        for source in self.platforms:
            for target in self.platforms:
                if source is target:
                    continue
                gap = _horizontal_gap(source, target)
                rise = float(source.rect.top - target.rect.top)
                fall = float(target.rect.top - source.rect.top)
                # Peach rises roughly 85 px per jump. Keep a margin for the
                # body/landing probe and allow a two-jump route up to 165 px.
                if rise > 165.0 or fall > 290.0:
                    continue
                horizontal_limit = 270.0 if rise > 15.0 else 380.0
                if gap > horizontal_limit:
                    continue
                cost = gap + 0.8 * max(0.0, rise) + 0.25 * max(0.0, fall) + 20.0
                graph[source.name].append((target.name, cost))
        return graph

    def support_platform(self, fighter: Any) -> Any | None:
        if fighter.ground_platform is not None:
            return fighter.ground_platform
        candidates: list[tuple[float, Any]] = []
        for platform in self.platforms:
            horizontal_gap = max(
                float(platform.rect.left) - float(fighter.pos.x),
                0.0,
                float(fighter.pos.x) - float(platform.rect.right),
            )
            vertical_gap = float(platform.rect.top) - float(fighter.pos.y)
            if horizontal_gap <= 90.0 and -20.0 <= vertical_gap <= 260.0:
                candidates.append((horizontal_gap + abs(vertical_gap) * 0.65, platform))
        return min(candidates, default=(0.0, None), key=lambda item: item[0])[1]

    def _blocking_platform(self, fighter: Any, target_x: float) -> Any | None:
        direction = 1 if target_x >= fighter.pos.x else -1
        middle_y = float(fighter.pos.y) - float(fighter.body_height) * 0.5
        entries: list[tuple[float, Any]] = []
        for platform in self.platforms:
            if platform is fighter.ground_platform:
                continue
            if not platform.rect.top <= middle_y < platform.rect.bottom:
                continue
            wall_x = float(platform.rect.left if direction > 0 else platform.rect.right)
            distance = (wall_x - float(fighter.pos.x)) * direction
            target_distance = (float(target_x) - float(fighter.pos.x)) * direction
            if 0.0 <= distance <= target_distance:
                entries.append((distance, platform))
        return min(entries, default=(0.0, None), key=lambda item: item[0])[1]

    def _shortest_path(self, start: Any, goal: Any) -> tuple[tuple[str, ...], float]:
        if start is None or goal is None:
            return (), float("inf")
        if start is goal:
            return (start.name,), 0.0
        queue: list[tuple[float, str]] = [(0.0, start.name)]
        costs = {start.name: 0.0}
        previous: dict[str, str] = {}
        while queue:
            cost, name = heapq.heappop(queue)
            if cost != costs.get(name):
                continue
            if name == goal.name:
                break
            for next_name, edge_cost in self.graph.get(name, ()):
                next_cost = cost + edge_cost
                if next_cost >= costs.get(next_name, float("inf")):
                    continue
                costs[next_name] = next_cost
                previous[next_name] = name
                heapq.heappush(queue, (next_cost, next_name))
        if goal.name not in costs:
            return (), float("inf")
        names = [goal.name]
        while names[-1] != start.name:
            names.append(previous[names[-1]])
        names.reverse()
        return tuple(names), costs[goal.name]

    @staticmethod
    def _landing_x(platform: Any, direction: int, half_width: float) -> float:
        margin = float(half_width) + 10.0
        if direction > 0:
            return min(float(platform.rect.centerx), float(platform.rect.left) + margin)
        return max(float(platform.rect.centerx), float(platform.rect.right) - margin)

    @staticmethod
    def _takeoff_x(platform: Any | None, direction: int, half_width: float, fallback: float) -> float:
        if platform is None:
            return fallback
        margin = float(half_width) + 5.0
        if direction > 0:
            return float(platform.rect.right) - margin
        return float(platform.rect.left) + margin

    def route(self, fighter: Any, opponent: Any) -> RoutePlan:
        target_x = float(opponent.pos.x)
        direction = 1 if target_x >= fighter.pos.x else -1
        start = self.support_platform(fighter)
        goal = self.support_platform(opponent)
        obstacle = self._blocking_platform(fighter, target_x)

        # This is the important Fixed12/Fixed13 case: both fighters may be on
        # Fixed1, but the direct route crosses a narrow solid rectangle. Treat
        # that rectangle's roof as an explicit waypoint instead of returning
        # the same start/goal node and walking forever.
        if obstacle is not None:
            rise = float(fighter.pos.y - obstacle.rect.top)
            if rise <= 165.0:
                landing_x = self._landing_x(obstacle, direction, fighter.body_half_width)
                takeoff_x = self._takeoff_x(
                    start, direction, fighter.body_half_width, float(fighter.pos.x)
                )
                return RoutePlan(
                    target_x=landing_x,
                    target_y=float(obstacle.rect.top),
                    takeoff_x=takeoff_x,
                    direction=direction,
                    requires_jump=True,
                    blocked=True,
                    obstacle=obstacle,
                    target_platform=obstacle,
                    path=tuple(
                        name
                        for name in (
                            getattr(start, "name", ""),
                            getattr(obstacle, "name", ""),
                            getattr(goal, "name", ""),
                        )
                        if name
                    ),
                    cost=abs(target_x - fighter.pos.x) + max(0.0, rise),
                )

        path, cost = self._shortest_path(start, goal)
        next_platform = self.by_name.get(path[1]) if len(path) > 1 else goal
        if next_platform is None:
            return RoutePlan(
                target_x=target_x,
                target_y=float(opponent.pos.y),
                takeoff_x=float(fighter.pos.x),
                direction=direction,
                requires_jump=False,
                blocked=False,
                obstacle=None,
                target_platform=None,
                path=path,
                cost=abs(target_x - fighter.pos.x),
            )
        waypoint_direction = 1 if next_platform.rect.centerx >= fighter.pos.x else -1
        waypoint_x = self._landing_x(
            next_platform, waypoint_direction, fighter.body_half_width
        )
        rise = float(fighter.pos.y - next_platform.rect.top)
        platform_gap = (
            _horizontal_gap(start, next_platform)
            if start is not None
            else abs(float(next_platform.rect.centerx) - float(fighter.pos.x))
        )
        requires_jump = bool(
            next_platform is not start
            and (rise > 12.0 or platform_gap > 20.0)
        )
        return RoutePlan(
            target_x=waypoint_x if requires_jump else target_x,
            target_y=float(next_platform.rect.top if requires_jump else opponent.pos.y),
            takeoff_x=self._takeoff_x(
                start, waypoint_direction, fighter.body_half_width, float(fighter.pos.x)
            ),
            direction=waypoint_direction,
            requires_jump=requires_jump,
            blocked=False,
            obstacle=None,
            target_platform=next_platform,
            path=path,
            cost=cost if math.isfinite(cost) else abs(target_x - fighter.pos.x),
        )

    def safe_landing(self, fighter: Any) -> Any | None:
        candidates: list[tuple[float, Any]] = []
        for platform in self.platforms:
            if platform.rect.top < fighter.pos.y + 12:
                continue
            gap_x = max(
                float(platform.rect.left) - float(fighter.pos.x),
                0.0,
                float(fighter.pos.x) - float(platform.rect.right),
            )
            if gap_x <= 120.0:
                candidates.append((float(platform.rect.top - fighter.pos.y) + gap_x, platform))
        return min(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def predict_position(fighter: Any, ticks: int) -> tuple[float, float]:
    ticks = max(0, int(ticks))
    x = float(fighter.pos.x) + float(fighter.xinc) * ticks
    y = float(fighter.pos.y)
    yinc = float(fighter.yinc)
    gravity = float(getattr(fighter, "gravity", 0.5))
    max_fall = float(getattr(fighter, "max_fall", 6.0))
    for _ in range(ticks):
        y += yinc
        yinc = min(max_fall, yinc + gravity)
    return x, y


def predicted_vertical_apex_y(fighter: Any, initial_yinc: float | None = None) -> float:
    """Predict the fighter origin's apex using the runtime's tick physics."""

    y = float(fighter.pos.y)
    yinc = float(fighter.yinc if initial_yinc is None else initial_yinc)
    gravity = max(0.01, float(getattr(fighter, "gravity", 0.5)))
    while yinc < 0.0:
        y += yinc
        yinc += gravity
    return y


def upward_action_is_safe(
    runtime: Any,
    fighter: Any,
    *,
    action: str,
    margin: float = TOP_SURVIVAL_MARGIN,
) -> bool:
    """Keep trained AI combos clear of the source map's upper KO line."""

    if action == "jump":
        initial_yinc = float(fighter.jump_yinc)
    elif action == "air_punch":
        current = float(fighter.yinc)
        initial_yinc = current if current <= -5.0 else current - 5.0
    else:
        raise ValueError(f"unknown upward action: {action}")
    safe_top = float(runtime.stage.bounds.top) + max(0.0, float(margin))
    return predicted_vertical_apex_y(fighter, initial_yinc) >= safe_top


def air_intercept(runtime: Any, fighter: Any, opponent: Any) -> dict[str, float | bool]:
    dx = float(opponent.pos.x - fighter.pos.x)
    horizon = int(np.clip(abs(dx) / max(1.0, float(fighter.move_xinc)), 4, 16))
    target_x, target_y = predict_position(opponent, horizon)
    predicted_dx = target_x - float(fighter.pos.x)
    predicted_dy = target_y - float(fighter.pos.y)
    jumps_left = max(0, 2 - int(fighter.jumpstate))
    vertical_reach = 65.0 + jumps_left * 82.0
    horizontal_reach = 70.0 + horizon * float(fighter.move_xinc)
    reachable = bool(
        abs(predicted_dx) <= horizontal_reach + 120.0
        and -vertical_reach <= predicted_dy <= 130.0
        and not _is_offstage(runtime, opponent)
    )
    strike_window = bool(
        not fighter.on_ground
        and abs(predicted_dx) <= 72.0
        and abs(predicted_dy) <= 62.0
    )
    return {
        "horizon": float(horizon),
        "target_x": target_x,
        "target_y": target_y,
        "dx": predicted_dx,
        "dy": predicted_dy,
        "reachable": reachable,
        "strike_window": strike_window,
    }


def safe_fast_fall(navigator: StageNavigator, fighter: Any, opponent: Any) -> bool:
    if fighter.on_ground or fighter.yinc >= fighter.max_fall:
        return False
    landing = navigator.safe_landing(fighter)
    if landing is None:
        return False
    attacker_below = bool(
        opponent.pos.y > fighter.pos.y
        and abs(opponent.pos.x - fighter.pos.x) < 90.0
        and opponent.current_attack
    )
    return not attacker_below


def purpose_action_mask(
    runtime: Any,
    fighter: Any,
    opponent: Any,
    controller: "PurposefulOptionController",
    *,
    curriculum: str = "duel",
) -> np.ndarray:
    mask = np.zeros(PURPOSE_COUNT, dtype=bool)
    context = tactical_context(runtime, fighter, opponent)
    route = controller.navigator.route(fighter, opponent)
    intercept = air_intercept(runtime, fighter, opponent)
    threat = float(context["threat_score"]) >= 0.20

    if fighter.dead or fighter.state in {"spawn", "dead"}:
        mask[Purpose.CONTINUE] = True
        return mask
    if controller.is_locked:
        mask[Purpose.CONTINUE] = True
        if _is_offstage(runtime, fighter):
            mask[Purpose.RECOVER] = True
        if fighter.state == "thrown" or fighter.ctrl_loss > 0:
            mask[Purpose.HITSTUN_ESCAPE] = True
        if threat and controller.intent not in {Purpose.RECOVER, Purpose.HITSTUN_ESCAPE}:
            mask[Purpose.EVADE] = True
            mask[Purpose.SHIELD] = bool(fighter.xinc == 0 and not fighter.current_attack)
        return mask
    if fighter.current_attack:
        mask[Purpose.CONTINUE] = True
        return mask
    if fighter.state == "thrown" or fighter.ctrl_loss > 0:
        mask[Purpose.CONTINUE] = True
        mask[Purpose.HITSTUN_ESCAPE] = True
        return mask
    if _is_offstage(runtime, fighter):
        mask[Purpose.RECOVER] = True
        mask[Purpose.HITSTUN_ESCAPE] = True
        return mask

    distance = float(context["distance"])
    dy = float(context["dy"])
    navigation_required = bool(route.requires_jump or route.blocked)
    above_route_obstacle = bool(
        not fighter.on_ground
        and (
            route.obstacle is None
            or fighter.pos.y <= route.obstacle.rect.top - 5.0
        )
    )
    mask[Purpose.CONTINUE] = controller.has_active_plan
    mask[Purpose.NAVIGATE] = navigation_required
    mask[Purpose.CHASE] = bool(not mask[Purpose.NAVIGATE] and distance > 58.0)
    mask[Purpose.MELEE] = bool(
        fighter.on_ground
        and distance <= 82.0
        and abs(dy) <= 58.0
        and context["target_in_front"]
    )
    mask[Purpose.AIR_CHASE] = bool(
        intercept["reachable"]
        and (not navigation_required or above_route_obstacle)
        and (
            opponent.state != "thrown"
            or opponent.last_sender is fighter
            or distance <= 125.0
        )
        and (
            opponent.state == "thrown"
            or not opponent.on_ground
            or (not fighter.on_ground and distance <= 150.0)
        )
    )
    mask[Purpose.ANTI_AIR] = bool(
        fighter.on_ground and dy < -12.0 and abs(float(context["dx"])) <= 105.0
    )
    mask[Purpose.BACK_THROW] = bool(
        fighter.on_ground
        and distance <= 30.0
        and abs(dy) <= 35.0
        and context["target_behind"]
    )
    mask[Purpose.AIMED_SHOT] = bool(
        75.0 <= distance <= 360.0
        and context["clear_shot"]
        and context["target_in_front"]
        and abs(float(context["lead_y"])) <= 32.0
        and controller.adapter.shoot_rearm <= 0
    )
    vertical_rocket = bool(
        dy < -55.0
        and abs(float(context["dx"])) <= 150.0
        and context["target_in_front"]
    )
    mask[Purpose.ROCKET] = bool(
        fighter.spec_up_ok
        and distance >= 45.0
        and (context["rocket_opportunity"] or vertical_rocket)
    )
    mask[Purpose.HITSTUN_ESCAPE] = False
    mask[Purpose.EVADE] = threat
    mask[Purpose.SHIELD] = bool(
        threat and fighter.xinc == 0 and controller.adapter.shield_rearm <= 0
    )
    mask[Purpose.LAND] = bool(
        not fighter.on_ground and safe_fast_fall(controller.navigator, fighter, opponent)
    )

    allowed_by_lesson: dict[str, set[Purpose]] = {
        "v5_navigation": {Purpose.CONTINUE, Purpose.NAVIGATE},
        "v5_air_chase": {Purpose.CONTINUE, Purpose.CHASE, Purpose.AIR_CHASE},
        "v5_escape": {
            Purpose.CONTINUE,
            Purpose.HITSTUN_ESCAPE,
            Purpose.EVADE,
            Purpose.LAND,
        },
        "v5_combo": {
            Purpose.CONTINUE,
            Purpose.CHASE,
            Purpose.MELEE,
            Purpose.AIR_CHASE,
            Purpose.ANTI_AIR,
        },
    }
    allowed = allowed_by_lesson.get(curriculum)
    if allowed is not None:
        mask = np.asarray(
            [value and Purpose(index) in allowed for index, value in enumerate(mask)],
            dtype=bool,
        )
    if not mask.any():
        mask[Purpose.CONTINUE if controller.has_active_plan else Purpose.CHASE] = True
    return mask


@dataclass
class PurposeEventTotals:
    plan_starts: int = 0
    plan_completions: int = 0
    plan_failures: int = 0
    forced_replans: int = 0
    purposeless_jumps: int = 0
    purposeful_jumps: int = 0
    purposeful_second_jumps: int = 0
    jump_down_reversals: int = 0
    air_chase_attempts: int = 0
    buffered_escapes: int = 0
    first_frame_escapes: int = 0
    wall_stall_decisions: int = 0
    top_risk_preventions: int = 0


class PurposefulOptionController:
    """Execute one semantically named plan as smooth source-style controls."""

    MIN_COMMIT = {
        Purpose.CHASE: 3,
        Purpose.NAVIGATE: 6,
        Purpose.MELEE: 2,
        Purpose.AIR_CHASE: 5,
        Purpose.ANTI_AIR: 2,
        Purpose.BACK_THROW: 2,
        Purpose.AIMED_SHOT: 2,
        Purpose.ROCKET: 2,
        Purpose.HITSTUN_ESCAPE: 4,
        Purpose.EVADE: 3,
        Purpose.RECOVER: 6,
        Purpose.SHIELD: 2,
        Purpose.LAND: 3,
    }
    DEADLINE = {
        Purpose.CHASE: 10,
        Purpose.NAVIGATE: 18,
        Purpose.MELEE: 5,
        Purpose.AIR_CHASE: 12,
        Purpose.ANTI_AIR: 4,
        Purpose.BACK_THROW: 4,
        Purpose.AIMED_SHOT: 4,
        Purpose.ROCKET: 5,
        Purpose.HITSTUN_ESCAPE: 10,
        Purpose.EVADE: 7,
        Purpose.RECOVER: 18,
        Purpose.SHIELD: 5,
        Purpose.LAND: 8,
    }

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.navigator = StageNavigator(runtime)
        self.adapter = TacticalInputAdapter(
            frame_skip=4,
            movement_commitment_decisions=3,
            combat_cooldown_decisions=2,
            shield_min_hold_decisions=1,
            shield_rearm_decisions=7,
            shield_max_hold_decisions=5,
            shoot_rearm_decisions=8,
        )
        self.events = PurposeEventTotals()
        self.reset()

    def reset(self) -> None:
        self.adapter.reset()
        self.intent = Purpose.CONTINUE
        self.plan_age = 0
        self.deadline = 0
        self.target_x = 0.0
        self.target_y = 0.0
        self.desired_direction = 0
        self.no_progress_steps = 0
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_jump_decision = -1000
        self.decision_index = 0
        self.escape_was_buffered = False
        self.last_route: RoutePlan | None = None
        self.events = PurposeEventTotals()

    @property
    def has_active_plan(self) -> bool:
        return self.intent != Purpose.CONTINUE and self.plan_age <= self.deadline

    @property
    def is_locked(self) -> bool:
        return bool(
            self.has_active_plan
            and self.plan_age < self.MIN_COMMIT.get(self.intent, 0)
        )

    def _start_plan(self, intent: Purpose, fighter: Any, opponent: Any) -> None:
        self.intent = intent
        self.plan_age = 0
        self.deadline = self.DEADLINE.get(intent, 4)
        self.no_progress_steps = 0
        self.last_x = float(fighter.pos.x)
        self.last_y = float(fighter.pos.y)
        self.last_route = None
        self.events.plan_starts += 1
        if intent in {Purpose.HITSTUN_ESCAPE, Purpose.RECOVER, Purpose.EVADE}:
            self.adapter.movement_lock = 0
        if intent == Purpose.AIR_CHASE:
            self.events.air_chase_attempts += 1
        if intent == Purpose.HITSTUN_ESCAPE and fighter.ctrl_loss > 0:
            self.escape_was_buffered = True
            self.events.buffered_escapes += 1

    @staticmethod
    def _movement_for_direction(direction: int, fighter: Any, opponent: Any) -> int:
        if direction == 0:
            return 0
        target_right = opponent.pos.x >= fighter.pos.x
        requested_right = direction > 0
        return 1 if requested_right == target_right else 2

    def _tactical_controls(
        self,
        fighter: Any,
        opponent: Any,
        *,
        direction: int,
        combat: int,
    ) -> tuple[dict[str, bool], ...]:
        movement = self._movement_for_direction(direction, fighter, opponent)
        mask = np.ones(13, dtype=bool)
        return self.adapter.begin_decision(
            np.asarray([movement, combat], dtype=np.int64),
            fighter=fighter,
            opponent=opponent,
            action_mask=mask,
        )

    def _safe_upward_combat(self, fighter: Any, combat: int) -> int:
        action = "jump" if combat == 1 else "air_punch"
        if upward_action_is_safe(self.runtime, fighter, action=action):
            return combat
        self.events.top_risk_preventions += 1
        return 0

    def _navigate(self, fighter: Any, opponent: Any) -> tuple[int, int]:
        keep_waypoint = bool(
            self.last_route is not None
            and self.last_route.target_platform is not None
            and not (
                fighter.on_ground
                and fighter.ground_platform is self.last_route.target_platform
            )
            and self.no_progress_steps < 6
        )
        route = self.last_route if keep_waypoint else self.navigator.route(fighter, opponent)
        self.last_route = route
        self.target_x = route.target_x
        self.target_y = route.target_y
        horizontal_error = float(route.target_x - fighter.pos.x)
        direction = 0 if abs(horizontal_error) <= 12.0 else (1 if horizontal_error > 0 else -1)
        if direction == 0:
            self.adapter.movement_lock = 0
        self.desired_direction = direction
        combat = 0
        horizontal_to_takeoff = (
            (route.takeoff_x - float(fighter.pos.x)) * direction if direction else 0.0
        )
        vertical_need = float(fighter.pos.y - route.target_y)
        if fighter.on_ground:
            if route.requires_jump and (
                horizontal_to_takeoff <= 24.0
                or route.blocked
                or wall_probe(self.runtime, fighter, direction)["distance"] <= 45.0
            ):
                combat = self._safe_upward_combat(fighter, 1)
        elif route.requires_jump and fighter.jumpstate < 2:
            predicted_apex = float(fighter.pos.y) - max(0.0, float(fighter.yinc) ** 2)
            cannot_clear = predicted_apex > route.target_y + 20.0
            obstacle_close = bool(
                route.obstacle is not None
                and abs(float(route.obstacle.rect.centerx) - float(fighter.pos.x)) < 90.0
                and fighter.pos.y > route.target_y + 10.0
            )
            if fighter.yinc >= -3.0 and (cannot_clear or obstacle_close or vertical_need > 80.0):
                combat = self._safe_upward_combat(fighter, 1)
        return direction, combat

    def _air_chase(self, fighter: Any, opponent: Any) -> tuple[int, int]:
        intercept = air_intercept(self.runtime, fighter, opponent)
        self.target_x = float(intercept["target_x"])
        self.target_y = float(intercept["target_y"])
        direction = 1 if self.target_x >= fighter.pos.x else -1
        self.desired_direction = direction
        if fighter.on_ground:
            return direction, self._safe_upward_combat(fighter, 1)
        if bool(intercept["strike_window"]):
            return direction, self._safe_upward_combat(fighter, 3)
        target_above = self.target_y < fighter.pos.y - 38.0
        if fighter.jumpstate < 2 and fighter.yinc >= -3.0 and target_above:
            return direction, self._safe_upward_combat(fighter, 1)
        return direction, 0

    def _escape_direction(self, fighter: Any, opponent: Any) -> int:
        bounds = self.runtime.stage.bounds
        toward_center = 1 if fighter.pos.x < bounds.centerx else -1
        edge_danger = abs(float(fighter.pos.x - bounds.centerx)) > bounds.w * 0.35
        if edge_danger:
            return toward_center
        return -1 if opponent.pos.x >= fighter.pos.x else 1

    def _recover(self, fighter: Any) -> tuple[int, int]:
        bounds = self.runtime.stage.bounds
        direction = 1 if fighter.pos.x < bounds.centerx else -1
        self.target_x = float(bounds.centerx)
        landing = self.navigator.safe_landing(fighter)
        self.target_y = float(landing.rect.top) if landing is not None else float(bounds.centery)
        self.desired_direction = direction
        if fighter.jumpstate < 2 and fighter.yinc >= -3.0:
            return direction, self._safe_upward_combat(fighter, 1)
        if fighter.spec_up_ok and (fighter.yinc > 1.0 or fighter.out_of_camera):
            return direction, 7
        return direction, 0

    def _land(self, fighter: Any, opponent: Any) -> tuple[int, int]:
        landing = self.navigator.safe_landing(fighter)
        if landing is None:
            return self._recover(fighter)
        target_x = float(np.clip(fighter.pos.x, landing.rect.left + 20, landing.rect.right - 20))
        self.target_x = target_x
        self.target_y = float(landing.rect.top)
        direction = 0 if abs(target_x - fighter.pos.x) < 10 else (1 if target_x > fighter.pos.x else -1)
        self.desired_direction = direction
        purposeful_fast_fall = safe_fast_fall(self.navigator, fighter, opponent)
        recently_jumped = self.decision_index - self.last_jump_decision <= 4
        return direction, 2 if purposeful_fast_fall and not recently_jumped else 0

    def begin_decision(
        self,
        requested: int | np.integer[Any],
        *,
        fighter: Any,
        opponent: Any,
        action_mask: Sequence[bool],
    ) -> tuple[dict[str, bool], ...]:
        self.decision_index += 1
        candidate = int(requested)
        mask = np.asarray(action_mask, dtype=bool)
        if candidate < 0 or candidate >= PURPOSE_COUNT or not mask[candidate]:
            candidate = int(Purpose.CONTINUE if self.has_active_plan else np.flatnonzero(mask)[0])
        requested_intent = Purpose(candidate)
        if requested_intent != Purpose.CONTINUE:
            emergency = requested_intent in {
                Purpose.HITSTUN_ESCAPE,
                Purpose.EVADE,
                Purpose.RECOVER,
                Purpose.SHIELD,
            }
            if not self.is_locked or emergency or requested_intent == self.intent:
                if requested_intent != self.intent or not self.has_active_plan:
                    self._start_plan(requested_intent, fighter, opponent)
        elif not self.has_active_plan:
            self.intent = Purpose.CONTINUE

        direction = 0
        combat = 0
        intent = self.intent
        context = tactical_context(self.runtime, fighter, opponent)
        if intent == Purpose.CHASE:
            self.last_route = self.navigator.route(fighter, opponent)
            if self.last_route.requires_jump or self.last_route.blocked:
                self.events.forced_replans += 1
                self.intent = Purpose.NAVIGATE
                self.deadline = self.DEADLINE[Purpose.NAVIGATE]
                direction, combat = self._navigate(fighter, opponent)
            else:
                self.target_x, self.target_y = float(opponent.pos.x), float(opponent.pos.y)
                direction = 1 if opponent.pos.x >= fighter.pos.x else -1
                self.desired_direction = direction
        elif intent == Purpose.NAVIGATE:
            direction, combat = self._navigate(fighter, opponent)
        elif intent == Purpose.MELEE:
            self.target_x, self.target_y = float(opponent.pos.x), float(opponent.pos.y)
            direction = 1 if opponent.pos.x >= fighter.pos.x else -1
            combat = 3 if float(context["distance"]) <= 82.0 else 0
        elif intent == Purpose.AIR_CHASE:
            direction, combat = self._air_chase(fighter, opponent)
        elif intent == Purpose.ANTI_AIR:
            direction, combat = 0, 4
        elif intent == Purpose.BACK_THROW:
            direction, combat = 0, 5
        elif intent == Purpose.AIMED_SHOT:
            direction, combat = 0, 6
        elif intent == Purpose.ROCKET:
            direction, combat = 0, 7
        elif intent == Purpose.HITSTUN_ESCAPE:
            direction = self._escape_direction(fighter, opponent)
            self.target_x = float(fighter.pos.x + direction * 100.0)
            self.target_y = float(fighter.pos.y)
            if fighter.ctrl_loss <= 0 and fighter.has_control:
                combat = (
                    self._safe_upward_combat(fighter, 1)
                    if fighter.jumpstate < 2
                    else 0
                )
                if self.escape_was_buffered:
                    self.events.first_frame_escapes += 1
                    self.escape_was_buffered = False
        elif intent == Purpose.EVADE:
            direction = self._escape_direction(fighter, opponent)
            combat = 1 if fighter.on_ground and fighter.jumpstate < 2 else 0
        elif intent == Purpose.RECOVER:
            direction, combat = self._recover(fighter)
        elif intent == Purpose.SHIELD:
            direction, combat = 0, 8
        elif intent == Purpose.LAND:
            direction, combat = self._land(fighter, opponent)

        before_jumpstate = int(fighter.jumpstate)
        controls = self._tactical_controls(
            fighter,
            opponent,
            direction=direction,
            combat=combat,
        )
        accepted_combat = int(self.adapter.last_action[1])
        if accepted_combat == 1:
            if intent in {
                Purpose.NAVIGATE,
                Purpose.AIR_CHASE,
                Purpose.HITSTUN_ESCAPE,
                Purpose.EVADE,
                Purpose.RECOVER,
            }:
                self.events.purposeful_jumps += 1
                if before_jumpstate == 1:
                    self.events.purposeful_second_jumps += 1
            else:
                self.events.purposeless_jumps += 1
            self.last_jump_decision = self.decision_index
        if accepted_combat == 2 and self.decision_index - self.last_jump_decision <= 4:
            self.events.jump_down_reversals += 1
        self.plan_age += 1
        return controls

    def observe_result(self, fighter: Any, opponent: Any) -> float:
        reward = 0.0
        moved = math.hypot(float(fighter.pos.x) - self.last_x, float(fighter.pos.y) - self.last_y)
        previous_distance = math.hypot(self.target_x - self.last_x, self.target_y - self.last_y)
        current_distance = math.hypot(
            self.target_x - float(fighter.pos.x), self.target_y - float(fighter.pos.y)
        )
        if self.intent in {Purpose.CHASE, Purpose.NAVIGATE, Purpose.AIR_CHASE}:
            if current_distance < previous_distance - 1.0 or moved >= 2.0:
                self.no_progress_steps = max(0, self.no_progress_steps - 1)
            else:
                self.no_progress_steps += 1
            if self.no_progress_steps >= 3:
                self.events.wall_stall_decisions += 1
                reward -= 0.015
            if self.no_progress_steps >= 6:
                self.events.plan_failures += 1
                self.events.forced_replans += 1
                self.plan_age = self.deadline + 1
                reward -= 0.05

        complete = False
        if self.intent == Purpose.NAVIGATE and self.last_route is not None:
            target_platform = self.last_route.target_platform
            complete = bool(
                target_platform is not None
                and fighter.on_ground
                and fighter.ground_platform is target_platform
            )
        elif self.intent == Purpose.CHASE:
            complete = fighter.pos.distance_to(opponent.pos) <= 75.0
        elif self.intent == Purpose.LAND:
            complete = bool(fighter.on_ground)
        elif self.intent == Purpose.HITSTUN_ESCAPE:
            complete = bool(fighter.ctrl_loss <= 0 and fighter.state != "thrown")
        elif self.intent == Purpose.RECOVER:
            complete = bool(fighter.on_ground and not _is_offstage(self.runtime, fighter))
        if complete:
            self.events.plan_completions += 1
            reward += 0.03 if self.intent == Purpose.NAVIGATE else 0.01
            self.plan_age = self.deadline + 1
        elif self.plan_age > self.deadline and self.intent != Purpose.CONTINUE:
            self.events.plan_failures += 1
            reward -= 0.02

        self.last_x = float(fighter.pos.x)
        self.last_y = float(fighter.pos.y)
        return reward

    def complete_current_plan(self) -> None:
        if self.intent == Purpose.CONTINUE or self.plan_age > self.deadline:
            return
        self.events.plan_completions += 1
        self.plan_age = self.deadline + 1

    def features(self, fighter: Any, opponent: Any) -> list[float]:
        bounds = self.runtime.stage.bounds
        one_hot = [float(self.intent == Purpose(index)) for index in range(PURPOSE_COUNT)]
        route = self.navigator.route(fighter, opponent)
        obstacle_height = (
            float(fighter.pos.y - route.obstacle.rect.top) if route.obstacle is not None else 0.0
        )
        values = [
            *one_hot,
            float(self.has_active_plan),
            self.plan_age / max(1, self.deadline),
            min(1.0, self.no_progress_steps / 6.0),
            (self.target_x - fighter.pos.x) / max(1.0, bounds.w),
            (self.target_y - fighter.pos.y) / max(1.0, bounds.h),
            float(self.desired_direction),
            float(self.is_locked),
            (route.target_x - fighter.pos.x) / max(1.0, bounds.w),
            (route.target_y - fighter.pos.y) / max(1.0, bounds.h),
            float(route.requires_jump),
            min(1.0, len(route.path) / 6.0),
            float(route.blocked),
            obstacle_height / max(1.0, bounds.h),
        ]
        for ticks in (4, 8, 12):
            target_x, target_y = predict_position(opponent, ticks)
            values.extend(
                [
                    (target_x - fighter.pos.x) / max(1.0, bounds.w),
                    (target_y - fighter.pos.y) / max(1.0, bounds.h),
                ]
            )
        intercept = air_intercept(self.runtime, fighter, opponent)
        values.extend(
            [
                float(fighter.state == "thrown"),
                float(fighter.ctrl_loss) / 1000.0,
                float(opponent.state == "thrown"),
                float(opponent.ctrl_loss) / 1000.0,
                float(opponent.yinc < 0.0),
                float(not fighter.on_ground),
                float(not opponent.on_ground),
                float(intercept["strike_window"]),
                float(safe_fast_fall(self.navigator, fighter, opponent)),
            ]
        )
        return values

    def metrics(self) -> Mapping[str, float | int]:
        starts = max(1, self.events.plan_starts)
        jumps = self.events.purposeful_jumps + self.events.purposeless_jumps
        return {
            **vars(self.events),
            "plan_completion_rate": self.events.plan_completions / starts,
            "purposeful_jump_rate": (
                self.events.purposeful_jumps / jumps if jumps else 1.0
            ),
            "escape_buffer_success_rate": (
                self.events.first_frame_escapes / self.events.buffered_escapes
                if self.events.buffered_escapes
                else 1.0
            ),
        }
