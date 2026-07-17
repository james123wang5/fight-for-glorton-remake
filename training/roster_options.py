from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .roster_contract import fighter_capabilities
from .v5_runtime_helpers import tactical_context
from .v5_options import (
    PURPOSE_COUNT,
    Purpose,
    PurposefulOptionController,
    purpose_action_mask,
)
from .v5_runtime_helpers import is_offstage


@dataclass(frozen=True)
class RoleCombatProfile:
    fighter_name: str
    ground_min_distance: float
    ground_max_distance: float
    ground_vertical_tolerance: float
    ground_requires_clear_path: bool
    up_horizontal_reach: float
    up_vertical_above: float
    up_max_distance: float


ROLE_COMBAT_PROFILES = {
    "PeachPlayer": RoleCombatProfile(
        "PeachPlayer", 75.0, 360.0, 32.0, True, 150.0, 55.0, 280.0
    ),
    "DefaultPlayer": RoleCombatProfile(
        "DefaultPlayer", 70.0, 330.0, 48.0, True, 95.0, 12.0, 145.0
    ),
    "TrashPlayer": RoleCombatProfile(
        "TrashPlayer", 58.0, 235.0, 120.0, False, 135.0, -35.0, 145.0
    ),
    "CoffeePlayer": RoleCombatProfile(
        "CoffeePlayer", 55.0, 225.0, 105.0, False, 100.0, 8.0, 145.0
    ),
    "SBLPlayer": RoleCombatProfile(
        "SBLPlayer", 85.0, 310.0, 40.0, True, 100.0, 8.0, 150.0
    ),
    "AuberginePlayer": RoleCombatProfile(
        "AuberginePlayer", 70.0, 345.0, 52.0, True, 100.0, 8.0, 150.0
    ),
}


def role_combat_profile(fighter: Any) -> RoleCombatProfile:
    try:
        return ROLE_COMBAT_PROFILES[str(fighter.fighter_name)]
    except KeyError as exc:
        raise ValueError(f"unsupported roster fighter: {fighter.fighter_name}") from exc


def ground_special_opportunity(runtime: Any, fighter: Any, opponent: Any) -> bool:
    profile = role_combat_profile(fighter)
    capabilities = fighter_capabilities(fighter)
    context = tactical_context(runtime, fighter, opponent)
    distance = float(context["distance"])
    lead_y = abs(float(context["lead_y"]))
    if not bool(context["target_in_front"]):
        return False
    if not profile.ground_min_distance <= distance <= profile.ground_max_distance:
        return False
    if capabilities.ground_special_mode == "lob_or_trap":
        # Trash and Coffee use an arc/trap. A straight clipline is not a
        # reliable legality test, but extreme vertical separation still is.
        return abs(float(context["dy"])) <= profile.ground_vertical_tolerance
    return bool(
        lead_y <= profile.ground_vertical_tolerance
        and (
            not profile.ground_requires_clear_path
            or bool(context["clear_shot"])
        )
    )


def offensive_up_special_opportunity(runtime: Any, fighter: Any, opponent: Any) -> bool:
    if not fighter.spec_up_ok:
        return False
    bounds = runtime.stage.bounds
    # All upward specials are commitments. Near the source top KO line they
    # are masked instead of letting a policy kill itself for one possible hit.
    if float(fighter.pos.y) < float(bounds.top) + 95.0:
        return False
    profile = role_combat_profile(fighter)
    capabilities = fighter_capabilities(fighter)
    context = tactical_context(runtime, fighter, opponent)
    dx = abs(float(context["dx"]))
    dy = float(context["dy"])
    distance = float(context["distance"])
    if capabilities.up_special_mode == "projectile":
        return bool(
            distance >= 45.0
            and bool(context["target_in_front"])
            and (bool(context["rocket_opportunity"]) or (dy < -55.0 and dx <= 150.0))
        )
    if capabilities.up_special_mode == "burst":
        return bool(dx <= profile.up_horizontal_reach and distance <= profile.up_max_distance)
    return bool(
        dx <= profile.up_horizontal_reach
        and distance <= profile.up_max_distance
        and dy <= profile.up_vertical_above
    )


def roster_purpose_action_mask(
    runtime: Any,
    fighter: Any,
    opponent: Any,
    controller: PurposefulOptionController,
    *,
    curriculum: str = "duel",
) -> np.ndarray:
    """Role-aware v6 mask while preserving the frozen fourteen action IDs."""

    mask = purpose_action_mask(
        runtime,
        fighter,
        opponent,
        controller,
        curriculum="duel" if curriculum == "roster_special" else curriculum,
    )
    ready_for_new_plan = bool(
        not fighter.dead
        and fighter.has_control
        and fighter.ctrl_loss <= 0
        and fighter.state not in {"ko", "thrown", "spawn", "dead"}
        and not fighter.current_attack
        and not controller.is_locked
        and not is_offstage(runtime, fighter)
    )
    if ready_for_new_plan:
        mask[Purpose.AIMED_SHOT] = bool(
            controller.adapter.shoot_rearm <= 0
            and ground_special_opportunity(runtime, fighter, opponent)
        )
        mask[Purpose.ROCKET] = offensive_up_special_opportunity(
            runtime, fighter, opponent
        )

    if curriculum == "roster_special":
        allowed = {
            Purpose.CONTINUE,
            Purpose.CHASE,
            Purpose.AIMED_SHOT,
            Purpose.ROCKET,
        }
        mask = np.asarray(
            [value and Purpose(index) in allowed for index, value in enumerate(mask)],
            dtype=bool,
        )
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != PURPOSE_COUNT:
        raise RuntimeError(
            f"roster purpose mask has {mask.size} entries instead of {PURPOSE_COUNT}"
        )
    if not mask.any():
        # This is a last-resort safety contract, not a learned preference.
        # A rare transition can combine a just-expired plan, a lesson filter,
        # hitstun/offstage state and an executor cooldown between simulation
        # ticks. MaskablePPO must still receive one harmless legal purpose.
        mask = np.zeros(PURPOSE_COUNT, dtype=bool)
        if fighter.dead or fighter.state in {"spawn", "dead"}:
            fallback = Purpose.CONTINUE
        elif fighter.state == "thrown" or fighter.ctrl_loss > 0:
            fallback = Purpose.HITSTUN_ESCAPE
        elif is_offstage(runtime, fighter):
            fallback = Purpose.RECOVER
        elif controller.has_active_plan or fighter.current_attack:
            fallback = Purpose.CONTINUE
        else:
            fallback = Purpose.CHASE
        mask[fallback] = True
    return mask


def role_purpose_labels(fighter: Any) -> tuple[str, ...]:
    capabilities = fighter_capabilities(fighter)
    labels = [purpose.name.lower() for purpose in Purpose]
    labels[Purpose.AIMED_SHOT] = f"ground_special:{capabilities.ground_special_mode}"
    labels[Purpose.ROCKET] = f"up_special:{capabilities.up_special_mode}"
    return tuple(labels)
