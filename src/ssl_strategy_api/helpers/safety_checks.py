from enum import Enum
from math import hypot
from typing import Any, Iterable, Optional, Tuple

from ssl_strategy_api.helpers.ball_spawner import BallSpawner
from ssl_strategy_api.models import RobotState

ROBOT_RADIUS_M = 0.09

_GEOMETRY = BallSpawner()


class TerminationReason(str, Enum):
    OUT_OF_FIELD = "out-of-field"
    OPPONENT_GOAL_AREA = "entered-opponent-goal-area"
    COLLISION = "collision"
    TARGET_REACHED = "target-reached"
    TIMEOUT = "timeout"


def _dimension(field: Any, primary: str, fallback: str) -> float:
    value = getattr(field, primary, None)
    if value is None:
        value = getattr(field, fallback)
    return float(value)


def is_out_of_field(x_m: float, y_m: float, field: Any, margin_m: float = 0.0) -> bool:
    """True when the point is outside the field lines shrunk by ``margin_m``."""
    half_length = _dimension(field, "length_m", "length") / 2 - margin_m
    half_width = _dimension(field, "width_m", "width") / 2 - margin_m
    return abs(x_m) > half_length or abs(y_m) > half_width


def is_in_opponent_goal_area(
    x_m: float, y_m: float, field: Any, attack_direction_x: int
) -> bool:
    """``field`` must expose penalty dimensions (the raw rSoccer field does)."""
    return _GEOMETRY.in_opponent_goal_area(field, x_m, y_m, attack_direction_x)


def find_collision(
    robot: RobotState,
    others: Iterable[RobotState],
    radius_m: float = ROBOT_RADIUS_M,
) -> Optional[int]:
    """Return the id of the first robot touching ``robot``, or None."""
    for other in others:
        if hypot(robot.x_m - other.x_m, robot.y_m - other.y_m) <= 2 * radius_m:
            return other.robot_id
    return None


def check_blue_robot(
    robot: RobotState,
    obstacles: Iterable[RobotState],
    field: Any,
    attack_direction_x: int,
) -> Tuple[Optional[TerminationReason], Optional[int]]:
    """Return the failure reason (and colliding robot id) or (None, None)."""
    if is_out_of_field(robot.x_m, robot.y_m, field):
        return TerminationReason.OUT_OF_FIELD, None
    if is_in_opponent_goal_area(robot.x_m, robot.y_m, field, attack_direction_x):
        return TerminationReason.OPPONENT_GOAL_AREA, None
    hit = find_collision(robot, obstacles)
    if hit is not None:
        return TerminationReason.COLLISION, hit
    return None, None
