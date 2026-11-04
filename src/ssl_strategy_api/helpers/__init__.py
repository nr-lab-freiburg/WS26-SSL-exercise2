"""Helpers for simulator runners and student strategies."""

from ssl_strategy_api.helpers.ball_spawner import BallSpawn, BallSpawner
from ssl_strategy_api.helpers.obstacle_test_scenarios_config import (
    BLUE_START,
    DEFAULT_DIFFICULTY,
    DEFAULT_RANDOM_OBSTACLES,
    DEFAULT_TARGET,
    DIFFICULTY_LEVELS,
    SCENARIOS,
    ObstacleSpec,
    Scenario,
    parse_obstacle,
    parse_target,
    sample_blue_start,
    sample_obstacles,
    sample_target,
    validate_obstacles,
    validate_target,
)
from ssl_strategy_api.helpers.print_scores import GoalTracker
from ssl_strategy_api.helpers.safety_checks import (
    ROBOT_RADIUS_M,
    TerminationReason,
    check_blue_robot,
    find_collision,
    is_in_opponent_goal_area,
    is_out_of_field,
)

__all__ = [
    "BLUE_START",
    "BallSpawn",
    "BallSpawner",
    "DEFAULT_DIFFICULTY",
    "DEFAULT_RANDOM_OBSTACLES",
    "DEFAULT_TARGET",
    "DIFFICULTY_LEVELS",
    "GoalTracker",
    "ObstacleSpec",
    "ROBOT_RADIUS_M",
    "SCENARIOS",
    "Scenario",
    "TerminationReason",
    "check_blue_robot",
    "find_collision",
    "is_in_opponent_goal_area",
    "is_out_of_field",
    "parse_obstacle",
    "parse_target",
    "sample_blue_start",
    "sample_obstacles",
    "sample_target",
    "validate_obstacles",
    "validate_target",
]
