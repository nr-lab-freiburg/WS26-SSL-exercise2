import argparse
from collections import Counter
from importlib import import_module
from math import ceil, hypot, isfinite
import random
import time
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import pygame
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

from ssl_strategy_api import FieldGeometry, TeamColor, TeamConfig
from ssl_strategy_api.adapters.rsoccer import (
    to_rsoccer_commands,
    world_state_from_rsoccer_frame,
)
from ssl_strategy_api.helpers import (
    BLUE_START,
    DEFAULT_DIFFICULTY,
    DEFAULT_RANDOM_OBSTACLES,
    DIFFICULTY_LEVELS,
    DEFAULT_TARGET,
    ROBOT_RADIUS_M,
    SCENARIOS,
    Scenario,
    ObstacleSpec,
    TerminationReason,
    check_blue_robot,
    sample_blue_start,
    sample_obstacles,
    sample_target,
    validate_obstacles,
    validate_target,
)
from ssl_strategy_api.strategies.scripted_obstacle import ScriptedObstacleStrategy


SIMULATION_STEP_SECONDS = 0.025
DEFAULT_DURATION_SECONDS = 90.0
TARGET_COLOR = (255, 140, 0)
EPISODE_TIME_LIMIT_SECONDS = 30.0
TARGET_TOLERANCE_M = 0.10
BLUE_ROBOT_ID = 0


class Exercise2Env(SSLBaseEnv):
    """Conflict avoidance: one blue robot, yellow obstacles and a target.

    Two modes:
    - Random (``obstacles`` is None): every ``reset()`` places
      ``n_random_obstacles`` yellow robots, the blue robot and the target at new
      random positions on the right side of the field. Used for training/testing.
    - Fixed (``obstacles`` given, any number): the same layout on every reset.
      Used for the evaluation scenarios.
    """

    def __init__(
        self,
        obstacles: Optional[Sequence[ObstacleSpec]] = None,
        target: Union[Tuple[float, float], str, None] = None,
        blue_start: Tuple[float, float] = BLUE_START,
        n_random_obstacles: int = DEFAULT_RANDOM_OBSTACLES,
        max_obstacle_speed_mps: float = 0.0,
        seed: Optional[int] = None,
        render_mode=None,
        difficulty: int = DEFAULT_DIFFICULTY,
    ):
        if difficulty not in DIFFICULTY_LEVELS:
            raise ValueError("difficulty must be one of {}".format(sorted(DIFFICULTY_LEVELS)))
        self.difficulty = difficulty
        self.randomized = obstacles is None
        self.obstacles: Tuple[ObstacleSpec, ...] = () if obstacles is None else tuple(obstacles)
        random_target = target == "random"
        self.target = (
            target if target is not None and not random_target else DEFAULT_TARGET
        )
        self.blue_start = blue_start
        self.blue_theta_deg = 0.0
        self.max_obstacle_speed_mps = max_obstacle_speed_mps
        self._rng = random.Random(seed)
        n_yellow = n_random_obstacles if self.randomized else len(self.obstacles)

        self.blue_config = TeamConfig(
            team=TeamColor.BLUE,
            controlled_robot_ids=(BLUE_ROBOT_ID,),
            attack_direction_x=1,
        )
        self.yellow_config = TeamConfig(
            team=TeamColor.YELLOW,
            controlled_robot_ids=tuple(range(n_yellow)),
            attack_direction_x=-1,
        )
        super().__init__(
            field_type=1,
            n_robots_blue=1,
            n_robots_yellow=n_yellow,
            time_step=SIMULATION_STEP_SECONDS,
            render_mode=render_mode,
        )
        self.canonical_field = FieldGeometry(
            length_m=self.field.length,
            width_m=self.field.width,
            goal_width_m=self.field.goal_width,
        )
        if not self.randomized:
            validate_obstacles(self.obstacles, self.field, self.blue_start)
            if random_target:
                self.target = sample_target(
                    self.obstacles,
                    self.field,
                    attack_direction_x=self.blue_config.attack_direction_x,
                    rng=self._rng,
                    blue_start=self.blue_start,
                )
            validate_target(
                self.target,
                self.obstacles,
                self.field,
                self.blue_config.attack_direction_x,
                self.blue_start,
            )
        self.state = None
        self.termination_reason: Optional[TerminationReason] = None
        self.collided_with: Optional[int] = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = random.Random(seed)
        return super().reset(seed=seed, options=options)

    def _get_commands(self, commands):
        return commands

    def _render(self):
        # Draw the target right after the field so robots are drawn on top of it.
        renderer = self.field_renderer
        draw_field = renderer.draw

        def draw_field_and_target(surface):
            draw_field(surface)
            if self.target is not None:
                center = (
                    int(self.target[0] * renderer.scale + renderer.center_x),
                    int(self.target[1] * renderer.scale + renderer.center_y),
                )
                pygame.draw.circle(
                    surface, TARGET_COLOR, center, int(ROBOT_RADIUS_M * renderer.scale)
                )

        renderer.draw = draw_field_and_target
        try:
            super()._render()
        finally:
            del renderer.draw

    def _frame_to_observations(self):
        self.state = world_state_from_rsoccer_frame(
            self.frame,
            self.canonical_field,
            frame_number=self.steps,
        )
        return self.state

    def _calculate_reward_and_done(self):
        blue = self.state.robots_blue.get(BLUE_ROBOT_ID)
        if blue is None:
            return 0.0, True

        # Stops on: leaving the field, entering the opponent goal area, a collision,
        # or reaching the target.
        self.termination_reason, self.collided_with = check_blue_robot(
            blue,
            self.state.robots_yellow.values(),
            self.field,
            self.blue_config.attack_direction_x,
        )
        if (
            self.termination_reason is None
            and hypot(blue.x_m - self.target[0], blue.y_m - self.target[1])
            <= TARGET_TOLERANCE_M
        ):
            self.termination_reason = TerminationReason.TARGET_REACHED
        done = self.termination_reason is not None

        # Create here your reward function
        reward = 0.0
        return reward, done

    def _sample_random_layout(self) -> None:
        attack = self.blue_config.attack_direction_x
        self.blue_start = sample_blue_start(self.field, self._rng, attack)
        self.blue_theta_deg = self._rng.uniform(-180.0, 180.0)
        path_probability, min_target_distance_m = DIFFICULTY_LEVELS[self.difficulty]
        self.target = sample_target(
            (),
            self.field,
            attack_direction_x=attack,
            rng=self._rng,
            blue_start=self.blue_start,
            min_blue_distance_m=min_target_distance_m,
        )
        self.obstacles = sample_obstacles(
            len(self.yellow_config.controlled_robot_ids),
            self.field,
            self._rng,
            self.blue_start,
            self.max_obstacle_speed_mps,
            target=self.target,
            path_probability=path_probability,
        )

    def _get_initial_positions_frame(self):
        if self.randomized:
            self._sample_random_layout()

        frame = Frame()
        frame.ball = Ball(x=0.0, y=-self.field.width / 2 + 0.2, v_x=0.0, v_y=0.0)
        frame.robots_blue[BLUE_ROBOT_ID] = Robot(
            yellow=False,
            id=BLUE_ROBOT_ID,
            x=self.blue_start[0],
            y=self.blue_start[1],
            theta=self.blue_theta_deg,
        )
        for index, spec in enumerate(self.obstacles):
            frame.robots_yellow[index] = Robot(
                yellow=True,
                id=index,
                x=spec.x_m,
                y=spec.y_m,
                theta=180.0,
            )
        return frame


def _load_strategy(path: str) -> Any:
    module_name, separator, class_name = path.partition(":")
    if not separator:
        raise ValueError("strategy must use module.path:ClassName format")
    return getattr(import_module(module_name), class_name)()


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the rSoccer field (default: on; use --no-render for headless runs)",
    )
    parser.add_argument(
        "--blue-strategy",
        default="ssl_strategy_api.strategies.conflict_avoidance:ConflictAvoidanceStrategy",
        help="Blue strategy class as module.path:ClassName",
    )


def add_scenario_arguments(parser: argparse.ArgumentParser, allow_all: bool = False) -> None:
    add_common_arguments(parser)
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS) + (["all"] if allow_all else []),
        default="static",
        help="Built-in obstacle scenario"
        + ("; 'all' runs every scenario in a row" if allow_all else ""),
    )


def build_scenario(
    args: argparse.Namespace,
    scenarios: Mapping[str, Scenario] = SCENARIOS,
) -> Tuple[Sequence[ObstacleSpec], Tuple[float, float], Tuple[float, float]]:
    """Return the obstacles, blue start and target of the chosen scenario."""
    scenario = scenarios[args.scenario]
    return scenario.obstacles, scenario.blue_start, scenario.target


def run_demo(
    blue_strategy: Any,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    render: bool = True,
    seed: Optional[int] = None,
    num_obstacles: int = DEFAULT_RANDOM_OBSTACLES,
    max_obstacle_speed_mps: float = 0.0,
    difficulty: int = DEFAULT_DIFFICULTY,
) -> Counter:
    """Run random episodes back to back until ``duration_seconds`` have elapsed.

    Every episode ends on collision, leaving the field, entering the opponent
    goal area, reaching the target or ``EPISODE_TIME_LIMIT_SECONDS``, and the next
    one starts with new random positions for blue, yellow and the target.
    """
    if not isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    if num_obstacles < 0:
        raise ValueError("num_obstacles must be non-negative")

    env = Exercise2Env(
        n_random_obstacles=num_obstacles,
        max_obstacle_speed_mps=max_obstacle_speed_mps,
        seed=seed,
        render_mode="human" if render else None,
        difficulty=difficulty,
    )
    yellow_strategy = ScriptedObstacleStrategy()
    outcomes: Counter = Counter()

    def start_episode(first: bool):
        observation, _ = env.reset(seed=seed if first else None)
        if hasattr(blue_strategy, "set_target"):
            blue_strategy.set_target(*env.target)
        if hasattr(blue_strategy, "reset"):
            blue_strategy.reset(seed=seed)
        yellow_strategy.set_obstacles(env.obstacles)
        return observation

    observation = start_episode(first=True)
    started = time.monotonic()
    max_steps = ceil(duration_seconds / SIMULATION_STEP_SECONDS)
    episode_step_limit = ceil(EPISODE_TIME_LIMIT_SECONDS / SIMULATION_STEP_SECONDS)
    episode_step = 0
    episode_number = 1

    try:
        for step in range(1, max_steps + 1):
            episode_step += 1
            blue_intents = blue_strategy.decide(observation, env.blue_config)
            yellow_intents = yellow_strategy.decide(observation, env.yellow_config)

            commands = to_rsoccer_commands(observation, env.blue_config, blue_intents)
            commands.extend(
                to_rsoccer_commands(observation, env.yellow_config, yellow_intents)
            )
            observation, reward, done, _, _ = env.step(commands)

            reason = env.termination_reason if done else None
            if not done and episode_step >= episode_step_limit:
                reason = TerminationReason.TIMEOUT
            if reason is not None:
                outcomes[reason.value] += 1
                print(
                    "Episode {} ended after {:.2f}s: {}, reward={:.3f}".format(
                        episode_number,
                        episode_step * SIMULATION_STEP_SECONDS,
                        reason.value,
                        reward,
                    ),
                    flush=True,
                )
                episode_number += 1
                episode_step = 0
                observation = start_episode(first=False)

            wait = started + step * SIMULATION_STEP_SECONDS - time.monotonic()
            if wait > 0:
                time.sleep(wait)
    finally:
        env.close()

    print(
        "Finished: {} completed episodes {}".format(sum(outcomes.values()), dict(outcomes)),
        flush=True,
    )
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise 2 (conflict avoidance) for developing and testing a strategy: "
            "random episodes, restarted after every stop."
        )
    )
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    add_common_arguments(parser)
    parser.add_argument(
        "--num-obstacles",
        type=int,
        default=DEFAULT_RANDOM_OBSTACLES,
        help="Number of yellow robots placed randomly on the right side",
    )
    parser.add_argument(
        "--obstacle-speed",
        type=float,
        default=0.0,
        help="Max speed (m/s) of randomly moving yellow robots (default 0: static)",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=sorted(DIFFICULTY_LEVELS),
        default=DEFAULT_DIFFICULTY,
        help="1 (easy) to 5 (hard): how often yellow robots block the straight path to the target",
    )
    args = parser.parse_args()
    if not isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if not isfinite(args.obstacle_speed) or args.obstacle_speed < 0:
        parser.error("--obstacle-speed must be non-negative")

    try:
        run_demo(
            blue_strategy=_load_strategy(args.blue_strategy),
            duration_seconds=args.duration_seconds,
            render=args.render,
            seed=args.seed,
            num_obstacles=args.num_obstacles,
            max_obstacle_speed_mps=args.obstacle_speed,
            difficulty=args.difficulty,
        )
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
