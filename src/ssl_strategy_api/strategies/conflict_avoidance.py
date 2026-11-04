import math
from typing import Mapping, Optional, Tuple

from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState


class ConflictAvoidanceStrategy:
    """Implement your conflict avoidance strategy here for exercise 2.

    Drive the blue robot to ``self.target`` without touching any yellow robot,
    leaving the field, or entering the opponent goal area.

    The default below drives straight at the target and ignores the obstacles,
    so it will collide. Replace it with your own conflict avoidance.
    """

    def __init__(self) -> None:
        self.target: Optional[Tuple[float, float]] = None

    def set_target(self, x_m: float, y_m: float) -> None:
        """Called by the runner with the position the blue robot must reach."""
        self.target = (x_m, y_m)

    def reset(self, seed=None) -> None:
        pass

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        if not config.controlled_robot_ids or config.team != TeamColor.BLUE:
            return {}
        robot_id = config.controlled_robot_ids[0]
        robot = state.robots_blue.get(robot_id)
        if robot is None or self.target is None:
            return {}

        # state.robots_yellow holds the obstacles (positions and measured velocities).
        dx = self.target[0] - robot.x_m
        dy = self.target[1] - robot.y_m
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return {robot_id: RobotIntent()}
        speed = min(1.5, distance)
        return {
            robot_id: RobotIntent(vx_mps=speed * dx / distance, vy_mps=speed * dy / distance)
        }
