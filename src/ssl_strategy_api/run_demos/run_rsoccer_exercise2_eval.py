import argparse
from math import hypot, isfinite
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import pygame

from ssl_strategy_api.adapters.rsoccer import to_rsoccer_commands
from ssl_strategy_api.helpers import SCENARIOS, ObstacleSpec, TerminationReason
from ssl_strategy_api.run_demos.run_rsoccer_exercise2 import (
    BLUE_ROBOT_ID,
    SIMULATION_STEP_SECONDS,
    TARGET_TOLERANCE_M,
    Exercise2Env,
    _load_strategy,
    add_scenario_arguments,
    build_scenario,
)
from ssl_strategy_api.strategies.scripted_obstacle import ScriptedObstacleStrategy

TIME_LIMIT_SECONDS = 30.0
FREEZE_SECONDS = 2.0


def run_evaluation(
    blue_strategy: Any,
    obstacles: Sequence[ObstacleSpec],
    blue_start: Tuple[float, float],
    target: Union[Tuple[float, float], str],
    render: bool = False,
    seed: Optional[int] = None,
    realtime: bool = True,
    name: str = "custom",
) -> Dict[str, Any]:
    """Run once with a 30 s simulated-time limit; stop on success or any failure.

    ``target`` is an (x, y) tuple or "random" (sampled by the env from ``seed``).
    """
    env = Exercise2Env(
        obstacles,
        target,
        blue_start=blue_start,
        seed=seed,
        render_mode="human" if render else None,
    )
    target = env.target
    print(
        "== scenario: {} ({} obstacles, target {:.2f},{:.2f}) ==".format(
            name, len(obstacles), *target
        ),
        flush=True,
    )
    yellow_strategy = ScriptedObstacleStrategy(obstacles)
    observation, _ = env.reset(seed=seed)

    if hasattr(blue_strategy, "set_target"):
        blue_strategy.set_target(*target)
    if hasattr(blue_strategy, "reset"):
        blue_strategy.reset(seed=seed)
    yellow_strategy.reset(seed=seed)

    max_steps = int(round(TIME_LIMIT_SECONDS / SIMULATION_STEP_SECONDS))
    reason = TerminationReason.TIMEOUT
    elapsed = TIME_LIMIT_SECONDS
    started = time.monotonic()

    try:
        for step in range(1, max_steps + 1):
            blue_intents = blue_strategy.decide(observation, env.blue_config)
            yellow_intents = yellow_strategy.decide(observation, env.yellow_config)
            commands = to_rsoccer_commands(observation, env.blue_config, blue_intents)
            commands.extend(
                to_rsoccer_commands(observation, env.yellow_config, yellow_intents)
            )
            observation, _, done, _, _ = env.step(commands)
            elapsed = step * SIMULATION_STEP_SECONDS

            blue = observation.robots_blue[BLUE_ROBOT_ID]

            if done:
                reason = env.termination_reason
                break
            if hypot(blue.x_m - target[0], blue.y_m - target[1]) <= TARGET_TOLERANCE_M:
                reason = TerminationReason.TARGET_REACHED
                break

            if realtime:
                wait = started + elapsed - time.monotonic()
                if wait > 0:
                    time.sleep(wait)

        blue = env.state.robots_blue.get(BLUE_ROBOT_ID)
        final_distance = (
            hypot(blue.x_m - target[0], blue.y_m - target[1]) if blue else None
        )
    except BaseException:
        env.close()
        raise

    success = reason == TerminationReason.TARGET_REACHED
    result = {
        "result": reason.value,
        "success": success,
        "elapsed_seconds": elapsed,
        "remaining_seconds": max(0.0, TIME_LIMIT_SECONDS - elapsed) if success else 0.0,
        "collided_with": env.collided_with,
        "final_distance_to_target_m": final_distance,
    }
    if success:
        print(
            "Target reached at {:.2f}s. Remaining time: {:.2f}s".format(
                elapsed, result["remaining_seconds"]
            ),
            flush=True,
        )
    else:
        detail = (
            " with yellow robot {}".format(env.collided_with)
            if env.collided_with is not None
            else ""
        )
        print(
            "Failed: {}{} at {:.2f}s. Remaining time: 0.00s".format(
                reason.value, detail, elapsed
            ),
            flush=True,
        )

    if render:
        # Keep the last frame on screen after the message so the final positions can be inspected.
        freeze_until = time.monotonic() + FREEZE_SECONDS
        while time.monotonic() < freeze_until:
            pygame.event.pump()
            time.sleep(0.05)
    env.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exercise 2 evaluation: reach the target within 30 s without conflicts."
    )
    add_scenario_arguments(parser, allow_all=True)
    args = parser.parse_args()

    try:
        results = {}
        names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        for name in names:
            args.scenario = name
            obstacles, blue_start, target = build_scenario(args)
            results[name] = run_evaluation(
                _load_strategy(args.blue_strategy),
                obstacles,
                blue_start,
                target,
                render=args.render,
                seed=args.seed,
                name=name,
            )
    except ValueError as error:
        parser.error(str(error))

    if len(results) > 1:
        print("\nSummary")
        for name, res in results.items():
            print("  {:8s} {:20s} remaining={:.2f}s".format(
                name, res["result"], res["remaining_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
