from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FIGHTER_ORDER = (
    "PeachPlayer",
    "DefaultPlayer",
    "TrashPlayer",
    "CoffeePlayer",
    "SBLPlayer",
    "AuberginePlayer",
)
STAGE_ORDER = ("Mogadishu", "Rooftop", "B52", "Space")
MAX_SURFACE_NODES = 32
SPECIAL_PROJECTILE_KINDS = ("Pencil", "Poop", "Garbage", "EnergyBall")

CAPABILITY_FIELDS = (
    "ground_melee",
    "air_melee",
    "uppercut",
    "back_throw",
    "shield",
    "dodge",
    "ranged_projectile",
    "ranged_beam",
    "ranged_lob_or_trap",
    "up_projectile",
    "up_strike",
    "up_burst",
)
AVAILABILITY_FIELDS = (
    "jump",
    "double_jump",
    "ground_punch",
    "air_punch",
    "uppercut",
    "back_throw",
    "ground_special",
    "air_special",
    "up_special",
    "shield",
)


def _one_hot(value: str, labels: Sequence[str]) -> list[float]:
    return [float(value == label) for label in labels]


@dataclass(frozen=True)
class FighterCapabilities:
    fighter_name: str
    special_kind: str
    ground_special_mode: str
    up_special_mode: str
    attack_labels: tuple[str, ...]
    has_dodge: bool

    def vector(self) -> tuple[float, ...]:
        labels = set(self.attack_labels)
        values = (
            "punchGround" in labels,
            "punchAir" in labels,
            "punchUp" in labels,
            "specialBackThrow" in labels,
            True,
            self.has_dodge,
            self.ground_special_mode == "projectile",
            self.ground_special_mode == "beam",
            self.ground_special_mode == "lob_or_trap",
            self.up_special_mode == "projectile",
            self.up_special_mode == "strike",
            self.up_special_mode == "burst",
        )
        return tuple(float(value) for value in values)


@dataclass(frozen=True)
class ActionAvailability:
    jump: bool
    double_jump: bool
    ground_punch: bool
    air_punch: bool
    uppercut: bool
    back_throw: bool
    ground_special: bool
    air_special: bool
    up_special: bool
    shield: bool

    def vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in AVAILABILITY_FIELDS)


@dataclass(frozen=True)
class AttackTiming:
    label: str
    frame: int
    total_frames: int
    remaining_frames: int

    @property
    def remaining_fraction(self) -> float:
        return self.remaining_frames / max(1, self.total_frames)


@dataclass(frozen=True)
class PlatformNode:
    index: int
    name: str
    platform_name: str
    segment_index: int
    moving: bool
    rect: tuple[float, float, float, float]


@dataclass(frozen=True)
class PlatformEdge:
    source: int
    target: int
    cost: float
    horizontal_gap: float
    rise: float
    fall: float
    requires_jump: bool
    dynamic: bool


@dataclass(frozen=True)
class RouteContext:
    current_node: int | None
    target_node: int | None
    next_node: int | None
    reachable: bool
    cost: float
    direct_blocked: bool


def fighter_capabilities(source: Any, fighter_name: str | None = None) -> FighterCapabilities:
    if hasattr(source, "fighter_data"):
        data = source.fighter_data
        name = str(source.fighter_name)
    else:
        manifest = source
        name = str(fighter_name or "")
        if name not in manifest.get("fighters", {}):
            raise ValueError(f"unknown fighter: {name}")
        data = manifest["fighters"][name]

    attacks = data.get("attacks", {})
    labels = tuple(sorted(set(attacks) | set(data.get("state_animations", {}))))
    ground = attacks.get("specialGround", {})
    if str(ground.get("kind", "")) == "kamehameha":
        ground_mode = "beam"
    elif str(ground.get("spawns", "")) in {"Garbage", "Poop"}:
        ground_mode = "lob_or_trap"
    elif ground.get("spawns"):
        ground_mode = "projectile"
    else:
        ground_mode = "strike"

    upward = attacks.get("specialUp", {})
    upward_spawn = str(upward.get("spawns", ""))
    if upward_spawn == "Rocket":
        up_mode = "projectile"
    elif upward_spawn == "GarbageBurst":
        up_mode = "burst"
    else:
        up_mode = "strike"
    return FighterCapabilities(
        fighter_name=name,
        special_kind=str(data.get("special_kind", "")),
        ground_special_mode=ground_mode,
        up_special_mode=up_mode,
        attack_labels=labels,
        has_dodge="dodge" in data.get("state_animations", {}),
    )


def action_availability(fighter: Any, opponent: Any | None = None) -> ActionAvailability:
    capabilities = fighter_capabilities(fighter)
    labels = set(capabilities.attack_labels)
    ready = bool(
        not fighter.dead
        and fighter.has_control
        and fighter.ctrl_loss <= 0
        and fighter.state not in {"ko", "thrown", "spawn", "dead"}
        and not fighter.current_attack
        and not fighter.shielded
    )
    can_first_jump = bool(ready and fighter.jumpstate == 0)
    can_second_jump = bool(ready and fighter.jumpstate == 1 and fighter.yinc >= -3.0)
    target_behind = False
    if opponent is not None:
        dx = float(opponent.pos.x - fighter.pos.x)
        dy = float(opponent.pos.y - fighter.pos.y)
        target_behind = bool(
            math.hypot(dx, dy) <= 30.0
            and abs(dy) <= 35.0
            and fighter.facing * (fighter.pos.x - opponent.pos.x) > 0
        )
    return ActionAvailability(
        jump=can_first_jump,
        double_jump=can_second_jump,
        ground_punch=bool(ready and fighter.on_ground and "punchGround" in labels),
        air_punch=bool(ready and not fighter.on_ground and "punchAir" in labels),
        uppercut=bool(
            ready
            and fighter.on_ground
            and abs(float(fighter.xinc)) <= 0.0
            and "punchUp" in labels
        ),
        back_throw=bool(
            ready
            and fighter.on_ground
            and target_behind
            and "specialBackThrow" in labels
        ),
        ground_special=bool(ready and fighter.on_ground and "specialGround" in labels),
        air_special=bool(ready and not fighter.on_ground and "specialAir" in labels),
        up_special=bool(ready and fighter.spec_up_ok and "specialUp" in labels),
        shield=bool(ready and fighter.xinc == 0 and fighter.shield_size > 0),
    )


def attack_timing(fighter: Any) -> AttackTiming:
    label = str(fighter.current_attack)
    if not label:
        return AttackTiming("", 0, 0, 0)
    animation = fighter.animations.get(label, {})
    total = max(1, int(animation.get("frame_count", 1)))
    frame = max(0, min(total, int(fighter.attack_frame)))
    return AttackTiming(label, frame, total, max(0, total - frame))


def make_training_match_config(
    agent_fighter: str,
    opponent_fighter: str,
    stage_name: str,
) -> dict[str, Any]:
    if agent_fighter not in FIGHTER_ORDER or opponent_fighter not in FIGHTER_ORDER:
        raise ValueError("training fighters must be one of FIGHTER_ORDER")
    if stage_name not in STAGE_ORDER:
        raise ValueError("training stage must be one of STAGE_ORDER")
    return {
        "type": "vsmode",
        "selected_stage": stage_name,
        "stage": stage_name,
        "limit_mode": "stock",
        "limit_value": 3,
        "players": [
            {
                "fighter": agent_fighter,
                "color": 0,
                "computer": False,
                "enabled": True,
                "level": 1,
                "team_index": 0,
            },
            {
                "fighter": opponent_fighter,
                "color": 1,
                "computer": False,
                "enabled": True,
                "level": 1,
                "team_index": 1,
            },
        ],
    }


def _horizontal_gap(left: Any, right: Any) -> float:
    if left.rect.right < right.rect.left:
        return float(right.rect.left - left.rect.right)
    if right.rect.right < left.rect.left:
        return float(left.rect.left - right.rect.right)
    return 0.0


def _single_jump_rise(fighter: Any) -> float:
    velocity = float(fighter.jump_yinc)
    gravity = max(0.01, float(fighter.gravity))
    rise = 0.0
    for _ in range(200):
        if velocity >= 0.0:
            break
        rise -= velocity
        velocity += gravity
    return rise


class StageNavigationGraph:
    """Current-phase broad graph over exposed, fighter-sized top surfaces."""

    def __init__(self, runtime: Any, fighter: Any) -> None:
        self.runtime = runtime
        self.fighter = fighter
        self.platforms = list(runtime.stage.platforms)
        self.nodes, self.node_platforms = self._build_surface_nodes()
        if len(self.nodes) > MAX_SURFACE_NODES:
            raise ValueError(
                f"{runtime.stage.name} has {len(self.nodes)} exposed surfaces; "
                f"contract allows {MAX_SURFACE_NODES}"
            )
        names = [node.name for node in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate surface names on {runtime.stage.name}")
        self.indices_by_platform_identity: dict[int, list[int]] = {}
        for node, platform in zip(self.nodes, self.node_platforms, strict=True):
            self.indices_by_platform_identity.setdefault(id(platform), []).append(node.index)
        self.edges = self._build_edges()
        self.adjacency: dict[int, list[PlatformEdge]] = {node.index: [] for node in self.nodes}
        for edge in self.edges:
            self.adjacency[edge.source].append(edge)

    def _build_surface_nodes(self) -> tuple[tuple[PlatformNode, ...], tuple[Any, ...]]:
        half_width = float(self.fighter.body_half_width)
        nodes: list[PlatformNode] = []
        owners: list[Any] = []
        for platform in self.platforms:
            intervals = [
                (
                    float(platform.rect.left) + half_width,
                    float(platform.rect.right) - half_width,
                )
            ]
            top = float(platform.rect.top)
            # Split a large platform wherever a fixed solid rises through its
            # top.  This turns Mogadishu's huge Fixed1 floor into disconnected
            # exposed segments instead of claiming fighters can walk through
            # buildings and Fixed12/Fixed13.
            for blocker in self.platforms:
                if blocker is platform or blocker.moving:
                    continue
                if not float(blocker.rect.top) < top <= float(blocker.rect.bottom) + 1.0:
                    continue
                blocked_left = float(blocker.rect.left) - half_width
                blocked_right = float(blocker.rect.right) + half_width
                remaining: list[tuple[float, float]] = []
                for left, right in intervals:
                    if blocked_right <= left or blocked_left >= right:
                        remaining.append((left, right))
                        continue
                    if left < blocked_left:
                        remaining.append((left, min(right, blocked_left)))
                    if blocked_right < right:
                        remaining.append((max(left, blocked_right), right))
                intervals = remaining
            for segment_index, (left, right) in enumerate(intervals):
                if right - left < 2.0:
                    continue
                index = len(nodes)
                nodes.append(
                    PlatformNode(
                        index=index,
                        name=f"{platform.name}#{segment_index}",
                        platform_name=str(platform.name),
                        segment_index=segment_index,
                        moving=bool(platform.moving),
                        rect=(left, top, right - left, float(platform.rect.h)),
                    )
                )
                owners.append(platform)
        return tuple(nodes), tuple(owners)

    def _build_edges(self) -> tuple[PlatformEdge, ...]:
        jump_rise = _single_jump_rise(self.fighter)
        max_rise = jump_rise * 1.92
        max_fall = max(290.0, float(self.runtime.stage.bounds.h) * 0.48)
        speed_scale = max(0.5, float(self.fighter.move_xinc) / 4.0)
        edges: list[PlatformEdge] = []
        for source_index, source in enumerate(self.nodes):
            for target_index, target in enumerate(self.nodes):
                if source_index == target_index:
                    continue
                source_left, source_top, source_width, _source_height = source.rect
                target_left, target_top, target_width, _target_height = target.rect
                source_right = source_left + source_width
                target_right = target_left + target_width
                if source_right < target_left:
                    gap = target_left - source_right
                elif target_right < source_left:
                    gap = source_left - target_right
                else:
                    gap = 0.0
                rise = source_top - target_top
                fall = target_top - source_top
                if rise > max_rise or fall > max_fall:
                    continue
                horizontal_limit = (270.0 if rise > 15.0 else 380.0) * speed_scale
                if gap > horizontal_limit:
                    continue
                edges.append(
                    PlatformEdge(
                        source=source_index,
                        target=target_index,
                        cost=gap
                        + 0.8 * max(0.0, rise)
                        + 0.25 * max(0.0, fall)
                        + 20.0,
                        horizontal_gap=gap,
                        rise=max(0.0, rise),
                        fall=max(0.0, fall),
                        requires_jump=bool(rise > 12.0 or gap > 20.0),
                        dynamic=bool(source.moving or target.moving),
                    )
                )
        return tuple(edges)

    def support_index(self, fighter: Any) -> int | None:
        ground_nodes = self.indices_by_platform_identity.get(id(fighter.ground_platform), [])
        if ground_nodes:
            return min(
                ground_nodes,
                key=lambda index: max(
                    self.nodes[index].rect[0] - float(fighter.pos.x),
                    0.0,
                    float(fighter.pos.x)
                    - (self.nodes[index].rect[0] + self.nodes[index].rect[2]),
                ),
            )
        candidates: list[tuple[float, int]] = []
        for node in self.nodes:
            left, top, width, _height = node.rect
            right = left + width
            gap_x = max(
                left - float(fighter.pos.x),
                0.0,
                float(fighter.pos.x) - right,
            )
            gap_y = top - float(fighter.pos.y)
            if gap_x <= 90.0 and -20.0 <= gap_y <= 260.0:
                candidates.append((gap_x + abs(gap_y) * 0.65, node.index))
        return min(candidates, default=(0.0, None), key=lambda item: item[0])[1]

    def _direct_blocked(self, fighter: Any, opponent: Any) -> bool:
        direction = 1 if opponent.pos.x >= fighter.pos.x else -1
        middle_y = float(fighter.pos.y) - float(fighter.body_height) * 0.5
        target_distance = abs(float(opponent.pos.x - fighter.pos.x))
        for platform in self.platforms:
            if platform is fighter.ground_platform:
                continue
            if not platform.rect.top <= middle_y < platform.rect.bottom:
                continue
            wall_x = float(platform.rect.left if direction > 0 else platform.rect.right)
            distance = (wall_x - float(fighter.pos.x)) * direction
            if 0.0 <= distance <= target_distance:
                return True
        return False

    def route(self, fighter: Any, opponent: Any) -> RouteContext:
        start = self.support_index(fighter)
        goal = self.support_index(opponent)
        blocked = self._direct_blocked(fighter, opponent)
        if start is None or goal is None:
            return RouteContext(start, goal, None, False, float("inf"), blocked)
        if start == goal:
            return RouteContext(start, goal, goal, True, 0.0, blocked)
        queue: list[tuple[float, int]] = [(0.0, start)]
        costs = {start: 0.0}
        previous: dict[int, int] = {}
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != costs.get(node):
                continue
            if node == goal:
                break
            for edge in self.adjacency.get(node, ()):
                next_cost = cost + edge.cost
                if next_cost >= costs.get(edge.target, float("inf")):
                    continue
                costs[edge.target] = next_cost
                previous[edge.target] = node
                heapq.heappush(queue, (next_cost, edge.target))
        if goal not in costs:
            return RouteContext(start, goal, None, False, float("inf"), blocked)
        path = [goal]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()
        return RouteContext(start, goal, path[1], True, costs[goal], blocked)


def _node_one_hot(index: int | None) -> list[float]:
    return [float(index == value) for value in range(MAX_SURFACE_NODES)]


def encode_roster_context(runtime: Any, agent: Any, opponent: Any) -> np.ndarray:
    """Encode the 186 new v6 fields without changing the frozen v5 prefix."""

    if agent.fighter_name not in FIGHTER_ORDER or opponent.fighter_name not in FIGHTER_ORDER:
        raise ValueError("unknown fighter in roster context")
    if runtime.stage.name not in STAGE_ORDER:
        raise ValueError("unknown stage in roster context")
    values: list[float] = []
    values.extend(_one_hot(agent.fighter_name, FIGHTER_ORDER))
    values.extend(_one_hot(opponent.fighter_name, FIGHTER_ORDER))
    values.extend(_one_hot(runtime.stage.name, STAGE_ORDER))
    values.extend(fighter_capabilities(agent).vector())
    values.extend(fighter_capabilities(opponent).vector())
    for fighter in (agent, opponent):
        values.extend(
            [
                float(fighter.weight),
                float(fighter.move_xinc) / 8.0,
                -float(fighter.jump_yinc) / 12.0,
                float(fighter.gravity),
                float(fighter.max_fall) / 10.0,
                float(fighter.body_half_width) / 30.0,
                float(fighter.body_height) / 50.0,
            ]
        )
    values.extend(action_availability(agent, opponent).vector())
    values.extend(action_availability(opponent, agent).vector())
    for fighter in (agent, opponent):
        timing = attack_timing(fighter)
        values.extend(
            [
                timing.remaining_fraction,
                min(1.0, timing.remaining_frames / 60.0),
            ]
        )

    graph = StageNavigationGraph(runtime, agent)
    route = graph.route(agent, opponent)
    values.extend(_node_one_hot(route.current_node))
    values.extend(_node_one_hot(route.target_node))
    values.extend(_node_one_hot(route.next_node))
    bounds = runtime.stage.bounds
    diagonal = max(1.0, math.hypot(bounds.w, bounds.h))
    next_node = graph.nodes[route.next_node] if route.next_node is not None else None
    next_dx = (
        0.0
        if next_node is None
        else float(next_node.rect[0] + next_node.rect[2] * 0.5 - agent.pos.x)
    )
    next_dy = 0.0 if next_node is None else float(next_node.rect[1] - agent.pos.y)
    current_moving = bool(
        route.current_node is not None and graph.nodes[route.current_node].moving
    )
    target_moving = bool(
        route.target_node is not None and graph.nodes[route.target_node].moving
    )
    values.extend(
        [
            float(route.current_node is not None),
            float(route.target_node is not None),
            float(route.reachable),
            min(5.0, route.cost / diagonal) if math.isfinite(route.cost) else 5.0,
            next_dx / max(1.0, bounds.w),
            next_dy / max(1.0, bounds.h),
            float(current_moving or target_moving or bool(next_node and next_node.moving)),
            float(route.direct_blocked),
        ]
    )
    special_projectiles = sorted(
        runtime.special_projectiles,
        key=lambda projectile: agent.pos.distance_to(projectile.pos),
    )
    nearest_kind = str(special_projectiles[0].kind) if special_projectiles else ""
    values.extend(_one_hot(nearest_kind, SPECIAL_PROJECTILE_KINDS))
    if len(values) != 186:
        raise RuntimeError(f"roster context changed: {len(values)}")
    return np.clip(np.asarray(values, dtype=np.float32), -5.0, 5.0)


def capability_report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "special_kind": (capability := fighter_capabilities(manifest, name)).special_kind,
            "ground_special_mode": capability.ground_special_mode,
            "up_special_mode": capability.up_special_mode,
            "has_dodge": capability.has_dodge,
            "capabilities": dict(zip(CAPABILITY_FIELDS, capability.vector(), strict=True)),
            "physics": {
                "weight": float(manifest["fighters"][name]["weight"]),
                "move_xinc": float(manifest["fighters"][name]["base_move_xinc"]),
                "jump_yinc": float(manifest["fighters"][name]["jump_yinc"]),
                "gravity": float(manifest["fighters"][name]["gravity_per_tick"]),
                "max_fall": float(manifest["fighters"][name]["max_fall_yinc"]),
            },
        }
        for name in FIGHTER_ORDER
    }
