from __future__ import annotations

import math
from typing import Any


def edge_danger(runtime: Any, fighter: Any) -> float:
    bounds = runtime.stage.bounds
    x = abs((float(fighter.pos.x) - bounds.centerx) / max(1.0, bounds.w / 2))
    y = abs((float(fighter.pos.y) - bounds.centery) / max(1.0, bounds.h / 2))
    return float(max(x, y))


def platform_distance(runtime: Any, fighter: Any) -> float:
    bounds = runtime.stage.bounds
    diagonal = max(1.0, math.hypot(bounds.w, bounds.h))
    if not runtime.stage.platforms:
        return 1.0
    distances: list[float] = []
    for platform in runtime.stage.platforms:
        gap_x = max(
            float(platform.rect.left) - float(fighter.pos.x),
            0.0,
            float(fighter.pos.x) - float(platform.rect.right),
        )
        gap_y = float(platform.rect.top) - float(fighter.pos.y)
        distances.append(math.hypot(gap_x, gap_y))
    return min(distances) / diagonal


def is_offstage(runtime: Any, fighter: Any) -> bool:
    return bool(
        not fighter.on_ground
        and (
            edge_danger(runtime, fighter) > 0.58
            or platform_distance(runtime, fighter) > 0.09
            or fighter.out_of_camera
        )
    )


def all_projectiles(runtime: Any) -> list[Any]:
    return [*runtime.bullets, *runtime.rockets, *runtime.special_projectiles]


def projectile_kind(projectile: Any) -> str:
    name = type(projectile).__name__.lower()
    if "bullet" in name:
        return "bullet"
    if "rocket" in name:
        return "rocket"
    return "special"


def projectile_threat(runtime: Any, fighter: Any, projectile: Any) -> dict[str, Any]:
    bounds = runtime.stage.bounds
    dx = float(projectile.pos.x - fighter.pos.x)
    dy = float(projectile.pos.y - fighter.pos.y)
    vx = float(getattr(projectile, "xinc", 0.0))
    vy = float(getattr(projectile, "yinc", 0.0))
    rvx = vx - float(fighter.xinc)
    rvy = vy - float(fighter.yinc)
    speed_sq = rvx * rvx + rvy * rvy
    raw_time = -(dx * rvx + dy * rvy) / speed_sq if speed_sq > 1e-6 else 999.0
    time_ticks = max(0.0, min(120.0, raw_time))
    miss = math.hypot(dx + rvx * time_ticks, dy + rvy * time_ticks)
    kind = projectile_kind(projectile)
    hit_radius = 62.0 if kind == "rocket" else 32.0
    approaching = raw_time > 0 and raw_time <= 24 and miss <= hit_radius
    time_factor = max(0.0, 1.0 - time_ticks / 24.0)
    miss_factor = max(0.0, 1.0 - miss / hit_radius)
    score = time_factor * miss_factor if approaching else 0.0
    return {
        "dx": dx,
        "dy": dy,
        "vx": vx,
        "vy": vy,
        "time_ticks": time_ticks,
        "miss": miss,
        "approaching": approaching,
        "score": score,
        "kind": kind,
        "bounds_w": float(bounds.w),
        "bounds_h": float(bounds.h),
    }


def enemy_projectile_threats(
    runtime: Any, fighter: Any, *, limit: int = 2
) -> list[dict[str, Any]]:
    entries = [
        projectile_threat(runtime, fighter, projectile)
        for projectile in all_projectiles(runtime)
        if bool(getattr(projectile, "alive", True))
        and getattr(projectile, "sender", None) is not fighter
    ]
    entries.sort(
        key=lambda item: (
            -float(item["score"]),
            float(item["time_ticks"]),
            float(item["miss"]),
        )
    )
    return entries[:limit]


def melee_threat(fighter: Any, opponent: Any) -> bool:
    if not str(opponent.current_attack) or str(opponent.current_attack).startswith("special"):
        return False
    dx = float(fighter.pos.x - opponent.pos.x)
    dy = float(fighter.pos.y - opponent.pos.y)
    facing = int(opponent.attack_facing if opponent.current_attack else opponent.facing)
    return bool(abs(dx) <= 75 and abs(dy) <= 55 and facing * dx >= -18)


def clear_shot(runtime: Any, fighter: Any, opponent: Any) -> bool:
    start = (round(fighter.pos.x), round(fighter.pos.y - 20))
    end = (round(opponent.pos.x), round(opponent.pos.y - 20))
    return not any(platform.rect.clipline(start, end) for platform in runtime.stage.platforms)


def rocket_opportunity(fighter: Any, opponent: Any) -> bool:
    dx = float(opponent.pos.x - fighter.pos.x)
    if fighter.facing * dx <= 20 or abs(dx) > 260:
        return False
    ticks = abs(dx) / 7.5
    if not 3.0 <= ticks <= 35.0:
        return False
    rocket_dy = -12.99 * ticks + 0.25 * ticks * (ticks + 1.0)
    target_dy = float(opponent.pos.y - fighter.pos.y) + float(opponent.yinc) * ticks
    return abs(target_dy - rocket_dy) <= 75.0


def tactical_context(runtime: Any, fighter: Any, opponent: Any) -> dict[str, Any]:
    dx = float(opponent.pos.x - fighter.pos.x)
    dy = float(opponent.pos.y - fighter.pos.y)
    distance = math.hypot(dx, dy)
    threats = enemy_projectile_threats(runtime, fighter, limit=2)
    projectile_score = max((float(item["score"]) for item in threats), default=0.0)
    close_attack = melee_threat(fighter, opponent)
    intercept_ticks = min(60.0, abs(dx) / 20.0)
    predicted_y = float(opponent.pos.y) + float(opponent.yinc) * intercept_ticks
    lead_y = predicted_y - float(fighter.pos.y)
    return {
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "clear_shot": clear_shot(runtime, fighter, opponent),
        "melee_threat": close_attack,
        "threat_score": max(projectile_score, 1.0 if close_attack else 0.0),
        "lead_y": lead_y,
        "intercept_ticks": intercept_ticks,
        "target_behind": fighter.facing * (fighter.pos.x - opponent.pos.x) > 0,
        "target_in_front": fighter.facing * dx >= 0,
        "rocket_opportunity": rocket_opportunity(fighter, opponent),
        "offstage": is_offstage(runtime, fighter),
        "opponent_above": dy < -18 and abs(dx) < 110,
    }


def wall_probe(
    runtime: Any, fighter: Any, direction: int, *, limit: float = 120.0
) -> dict[str, Any]:
    direction = 1 if direction >= 0 else -1
    middle_y = float(fighter.pos.y) - float(fighter.body_height) * 0.5
    entries: list[tuple[float, Any]] = []
    for platform in runtime.stage.platforms:
        if not platform.rect.top <= middle_y < platform.rect.bottom:
            continue
        stop_x = (
            float(platform.rect.left - fighter.body_half_width)
            if direction > 0
            else float(platform.rect.right + fighter.body_half_width)
        )
        distance = (stop_x - float(fighter.pos.x)) * direction
        if -1.0 <= distance <= limit:
            entries.append((max(0.0, distance), platform))
    if not entries:
        return {
            "blocked": False,
            "distance": limit,
            "top_delta": 0.0,
            "height": 0.0,
            "moving": False,
            "platform": None,
        }
    distance, platform = min(entries, key=lambda item: item[0])
    return {
        "blocked": True,
        "distance": distance,
        "top_delta": float(platform.rect.top) - float(fighter.pos.y),
        "height": float(platform.rect.h),
        "moving": bool(platform.moving),
        "platform": platform,
    }


def route_distance(runtime: Any, fighter: Any, opponent: Any) -> float:
    dx = float(opponent.pos.x - fighter.pos.x)
    dy = float(opponent.pos.y - fighter.pos.y)
    direction = 1 if dx >= 0 else -1
    probe = wall_probe(runtime, fighter, direction)
    distance = abs(dx) + 0.75 * abs(dy)
    if probe["blocked"] and float(probe["distance"]) < min(120.0, abs(dx)):
        distance += 60.0 + min(100.0, abs(float(probe["top_delta"])))
    return distance
