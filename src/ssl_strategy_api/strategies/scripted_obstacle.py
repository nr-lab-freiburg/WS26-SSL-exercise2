from typing import Dict, Mapping, Sequence, Tuple

from ssl_strategy_api.helpers.obstacle_test_scenarios_config import ObstacleSpec
from ssl_strategy_api.helpers.safety_checks import ROBOT_RADIUS_M
from ssl_strategy_api.models import RobotIntent, TeamConfig, WorldState


class ScriptedObstacleStrategy:
    """Moves each yellow robot at its constant velocity, bouncing off the field lines."""

    def __init__(self, obstacles: Sequence[ObstacleSpec] = ()) -> None:
        self._velocity: Dict[int, Tuple[float, float]] = {}
        self._initial: list = []
        self.set_obstacles(obstacles)

    def set_obstacles(self, obstacles: Sequence[ObstacleSpec]) -> None:
        """Load the velocities of a (new) layout and restart from them."""
        self._initial = [(s.vx_mps, s.vy_mps) for s in obstacles]
        self.reset()

    def reset(self, seed=None) -> None:
        self._velocity = dict(enumerate(self._initial))

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        intents: Dict[int, RobotIntent] = {}
        x_limit = state.field.length_m / 2 - ROBOT_RADIUS_M
        y_limit = state.field.width_m / 2 - ROBOT_RADIUS_M
        for robot_id in config.controlled_robot_ids:
            robot = state.robots_yellow.get(robot_id)
            if robot is None:
                continue
            vx, vy = self._velocity.get(robot_id, (0.0, 0.0))
            if (robot.x_m <= -x_limit and vx < 0) or (robot.x_m >= x_limit and vx > 0):
                vx = -vx
            if (robot.y_m <= -y_limit and vy < 0) or (robot.y_m >= y_limit and vy > 0):
                vy = -vy
            self._velocity[robot_id] = (vx, vy)
            intents[robot_id] = RobotIntent(vx_mps=vx, vy_mps=vy)
        return intents
