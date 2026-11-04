from dataclasses import dataclass
from math import hypot, isfinite
import random
from typing import Dict, Optional, Sequence, Tuple

from ssl_strategy_api.helpers.safety_checks import (
    ROBOT_RADIUS_M,
    is_in_opponent_goal_area,
    is_out_of_field,
)

BLUE_START = (-3.5, 0.0)
DEFAULT_TARGET = (3.0, 2.0)
MIN_CLEARANCE_M = 4 * ROBOT_RADIUS_M
DEFAULT_RANDOM_OBSTACLES = 5
DEFAULT_DIFFICULTY = 1
# Difficulty level -> (chance that each yellow robot is placed on the straight
# blue-to-target path, minimum blue-to-target distance in metres).
DIFFICULTY_LEVELS: Dict[int, Tuple[float, float]] = {
    1: (0.10, 0.0),
    2: (0.25, 0.0),
    3: (0.50, 1.5),
    4: (0.70, 1.5),
    5: (0.80, 2.0),
}
# A robot placed "on the path" is offset sideways by up to this distance. Values
# above 2 * ROBOT_RADIUS_M spread the robots out, so some only sit near the line.
PATH_LATERAL_OFFSET_M = 0.5
PATH_RANGE = (0.2, 0.8)
# Only the first robots can be placed on the path; any further robots are uniform.
MAX_PATH_ROBOTS = 5


@dataclass(frozen=True)
class ObstacleSpec:
    """Initial position (m) and velocity (m/s) of one yellow robot, field frame."""

    x_m: float
    y_m: float
    vx_mps: float = 0.0
    vy_mps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "vx_mps", "vy_mps"):
            if not isfinite(getattr(self, name)):
                raise ValueError("{} must be finite".format(name))


@dataclass(frozen=True)
class Scenario:
    """A fixed layout: yellow obstacles plus the blue robot start and target."""

    obstacles: Tuple[ObstacleSpec, ...]
    blue_start: Tuple[float, float] = BLUE_START
    target: Tuple[float, float] = DEFAULT_TARGET


# Sample scenarios
SCENARIOS: Dict[str, Scenario] = {
    # Avoid opponent goal area
    "basic": Scenario(
        obstacles=(),
        blue_start=(4, 2.0),
        target=(4, -2.0),
    ),
    # Three stationary robots.
    "static": Scenario(
        obstacles=(
            ObstacleSpec(1.0, 0.0),
            ObstacleSpec(1.0, 1.0),
            ObstacleSpec(1.0, -1.0),
        ),
        blue_start=(0.5, 0.0),
        target=(3.0, 0.0),
    ),
    # Three static robots and one moving robot.
    "dynamic": Scenario(
        obstacles=(
            ObstacleSpec(0.8, -0.3),
            ObstacleSpec(1.0, -1.0),
            ObstacleSpec(1.3, 0.7),
            ObstacleSpec(0.0, 0.0, vx_mps=0.65, vy_mps=1),
        ),
        blue_start=(0.0, 1.0),
        target=(2.0, -1.2),
    ),
    
}


def validate_obstacles(
    obstacles: Sequence[ObstacleSpec],
    field,
    blue_start: Tuple[float, float] = BLUE_START,
) -> None:
    """Check starts are inside the field and not overlapping (goal area is allowed)."""
    for index, spec in enumerate(obstacles):
        if is_out_of_field(spec.x_m, spec.y_m, field, margin_m=ROBOT_RADIUS_M):
            raise ValueError("obstacle {} starts outside the field".format(index))
        if hypot(spec.x_m - blue_start[0], spec.y_m - blue_start[1]) < MIN_CLEARANCE_M:
            raise ValueError("obstacle {} starts on top of the blue robot".format(index))
        for other_index in range(index):
            other = obstacles[other_index]
            if hypot(spec.x_m - other.x_m, spec.y_m - other.y_m) < 2 * ROBOT_RADIUS_M:
                raise ValueError(
                    "obstacles {} and {} overlap".format(other_index, index)
                )


def validate_target(
    target: Tuple[float, float],
    obstacles: Sequence[ObstacleSpec],
    field,
    attack_direction_x: int = 1,
    blue_start: Optional[Tuple[float, float]] = None,
    min_clearance_m: float = MIN_CLEARANCE_M,
) -> None:
    x_m, y_m = target
    if not (isfinite(x_m) and isfinite(y_m)):
        raise ValueError("target must be finite")
    if is_out_of_field(x_m, y_m, field, margin_m=ROBOT_RADIUS_M):
        raise ValueError("target is outside the field")
    if is_in_opponent_goal_area(x_m, y_m, field, attack_direction_x):
        raise ValueError("target is inside the opponent goal area")
    for index, spec in enumerate(obstacles):
        if hypot(x_m - spec.x_m, y_m - spec.y_m) < min_clearance_m:
            raise ValueError(
                "target is too close to the start of obstacle {}".format(index)
            )
    if blue_start is not None and hypot(x_m - blue_start[0], y_m - blue_start[1]) < min_clearance_m:
        raise ValueError("target is too close to the blue robot start")


def _right_side_point(rng: random.Random, field) -> Tuple[float, float]:
    """Uniform point on the right half (x > 0), a robot radius inside the lines."""
    half_length = field.length / 2
    half_width = field.width / 2
    return (
        rng.uniform(ROBOT_RADIUS_M, half_length - ROBOT_RADIUS_M),
        rng.uniform(-half_width + ROBOT_RADIUS_M, half_width - ROBOT_RADIUS_M),
    )


def sample_target(
    obstacles: Sequence[ObstacleSpec],
    field,
    seed: Optional[int] = None,
    attack_direction_x: int = 1,
    rng: Optional[random.Random] = None,
    blue_start: Optional[Tuple[float, float]] = None,
    min_blue_distance_m: float = 0.0,
) -> Tuple[float, float]:
    """Random target on the right side, valid for ``obstacles`` (reproducible by seed).

    ``min_blue_distance_m`` is the minimum distance from ``blue_start`` to the target.
    """
    rng = rng if rng is not None else random.Random(seed)
    for _ in range(1000):
        candidate = _right_side_point(rng, field)
        if (
            blue_start is not None
            and hypot(candidate[0] - blue_start[0], candidate[1] - blue_start[1])
            < min_blue_distance_m
        ):
            continue
        try:
            validate_target(
                candidate, obstacles, field, attack_direction_x, blue_start, MIN_CLEARANCE_M
            )
        except ValueError:
            continue
        return candidate
    raise ValueError("could not find a valid random target")


def sample_blue_start(
    field, rng: random.Random, attack_direction_x: int = 1
) -> Tuple[float, float]:
    """Random blue start on the right side, outside the opponent goal area."""
    for _ in range(1000):
        x_m, y_m = _right_side_point(rng, field)
        if not is_in_opponent_goal_area(x_m, y_m, field, attack_direction_x):
            return x_m, y_m
    raise ValueError("could not find a valid random blue start")


def _path_point(
    rng: random.Random, start: Tuple[float, float], target: Tuple[float, float]
) -> Tuple[float, float]:
    """Point on the straight start-target segment, offset sideways by a small amount."""
    dx, dy = target[0] - start[0], target[1] - start[1]
    length = hypot(dx, dy)
    t = rng.uniform(*PATH_RANGE)
    offset = rng.uniform(-PATH_LATERAL_OFFSET_M, PATH_LATERAL_OFFSET_M)
    return (
        start[0] + t * dx - offset * dy / length,
        start[1] + t * dy + offset * dx / length,
    )


def sample_obstacles(
    count: int,
    field,
    rng: random.Random,
    blue_start: Tuple[float, float],
    max_speed_mps: float = 0.0,
    target: Optional[Tuple[float, float]] = None,
    path_probability: float = 0.0,
) -> Tuple[ObstacleSpec, ...]:
    """Random non-overlapping yellow robots on the right side (goal area allowed).

    With ``max_speed_mps`` > 0 each robot gets a random velocity up to that speed.
    With a ``target`` and ``path_probability`` > 0, each of the first
    ``MAX_PATH_ROBOTS`` robots is independently placed on the straight
    blue-to-target path with that probability (it falls back to a uniform position if the path has no free spot left).
    """
    keep_clear = [blue_start] + ([target] if target is not None else [])

    def free(x_m: float, y_m: float, placed) -> bool:
        if is_out_of_field(x_m, y_m, field, margin_m=ROBOT_RADIUS_M):
            return False
        if any(hypot(x_m - p[0], y_m - p[1]) < MIN_CLEARANCE_M for p in keep_clear):
            return False
        return all(hypot(x_m - p[0], y_m - p[1]) >= MIN_CLEARANCE_M for p in placed)

    placed = []
    for index in range(count):
        on_path = (
            target is not None
            and index < MAX_PATH_ROBOTS
            and rng.random() < path_probability
        )
        position = None
        if on_path:
            for _ in range(50):
                candidate = _path_point(rng, blue_start, target)
                if free(*candidate, placed):
                    position = candidate
                    break
        for _ in range(100):
            if position is not None:
                break
            candidate = _right_side_point(rng, field)
            if free(*candidate, placed):
                position = candidate
        if position is None:
            raise ValueError("could not place {} non-overlapping obstacles".format(count))
        placed.append(position)
    specs = []
    for x_m, y_m in placed:
        vx = rng.uniform(-max_speed_mps, max_speed_mps)
        vy = rng.uniform(-max_speed_mps, max_speed_mps)
        specs.append(ObstacleSpec(x_m, y_m, vx, vy))
    return tuple(specs)


def parse_obstacle(text: str) -> ObstacleSpec:
    """Parse 'x,y' or 'x,y,vx,vy'."""
    parts = [float(p) for p in text.split(",")]
    if len(parts) not in (2, 4):
        raise ValueError("obstacle must be 'x,y' or 'x,y,vx,vy'")
    return ObstacleSpec(*parts)


def parse_target(text: str) -> Tuple[float, float]:
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError("target must be 'x,y'")
    return parts[0], parts[1]
