# SSL Student Strategy API

This package provides a simulator-independent SSL strategy framework, where strategy commands use physical units (for example, linear velocities in meters per second and angular velocities in radians per second). It has no runtime dependencies and requires Python 3.10 or newer.

Instructions for Ubuntu (For macOS and windows there are some extra files provided): First create a folder (for example `robocup_project`), then clone the rSoccer repository, then clone this `ssl_strategy_api` repository. After that, create `rsoccer_venv` inside the `rSoccer` repository, activate it, and install both packages:

```sh
mkdir -p robocup_project
cd robocup_project

git clone <RSOCCER_REPO_URL>
git clone <SSL_STRATEGY_API_REPO_URL>

cd rSoccer
python3.10 -m venv rsoccer_venv
source rsoccer_venv/bin/activate
python -m pip install -e .
python -m pip install --no-deps -e ../ssl_strategy_api
```

Replace `<RSOCCER_REPO_URL>` and `<SSL_STRATEGY_API_REPO_URL>` with the real repository URLs.

Note: use Python 3.10 or higher; ideally use python version 3.10 explicitly when creating the virtual environment.

`--no-deps` is safe for the shared API because it has no third-party dependencies; `-e` makes the venv import the package directly from this workspace, so local edits are immediately available.

## Strategy Interface

`WorldState` contains the ball, blue and yellow robot mappings, field geometry, and optional frame metadata. Positions are in meters, linear velocities are in m/s, headings are radians, and angular velocities are in rad/s. Velocity values may be absent; `velocity_source` distinguishes measured, estimated, and unavailable planar velocity. rSoccer provides you already with a velocity observation for the ball and robots. Though when you use other simulators (e.g. grSim) the velocities may need to be estimated by you.

Strategies implement the function `decide(state, config)`. This function gets the current state of the game (e.g. robot positions, ball position) and config (e.g. which team are we handling? for example YELLOW_TEAM, which robots are we controlling? e.g. controlled_robot_ids= (0,1)) as arguments and return a mapping from controlled robot IDs to `RobotIntent` (which is an action for each robot). Intents use field-frame velocities in physical units. Set `kick_speed_mps` to a value from 0 to 5; zero means no kick. The dribbler remains a boolean. For example, `RobotIntent(kick_speed_mps=2.5)` requests a 2.5 m/s kick.

`MotionLimits()` centrally caps planar speed at 2.0 m/s by vector magnitude and angular speed at 3.0 rad/s. For example, a request of `(vx_mps=2, vy_mps=2)` has magnitude about 2.83 m/s, so it is scaled to about `(1.41, 1.41)` while preserving direction. A task may configure lower limits, but not higher ones. Runners use this shared default rather than redefining limits at each command call. The rSoccer command adapter also rotates field-frame velocity into robot-relative command coordinates.

## Example Goalkeeper Environment

Initially, we provide a complete goalkeeper baseline in two versions:

- Rule-based goalkeeper: `src/ssl_strategy_api/strategies/goalkeeper.py`
- RL-based goalkeeper (Q-table style): `src/ssl_strategy_api/strategies/goalkeeper_rl.py`

You can run an example environment that uses the rule-based goalkeeper while a ball is repeatedly shot against it, so you can observe the baseline behavior:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_goalkeeper
```

## Run Exercise 1

Run Exercise 1 with:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1
```

In the default setup, blue controls robot id 0 with the striker strategy (student side), and yellow uses an idle goalkeeper strategy.

Choosing the goalkeeper strategy

In the Exercise 1 runner, the yellow strategy is selected through the `--yellow-strategy` argument.
By default, it uses the idle goalkeeper:

```text
default="ssl_strategy_api.strategies.idle_goalkeeper_rl:IdleGoalkeeperStrategy"
```

You can switch to other provided goalkeeper baselines by changing that default in the parser:

- Rule-based goalkeeper:

```text
default="ssl_strategy_api.strategies.goalkeeper:GoalkeeperStrategy"
```

- RL-based goalkeeper (Q-table):

```text
default="ssl_strategy_api.strategies.goalkeeper_rl:QTableGoalkeeperStrategy"
```

You can also choose the goalkeeper at runtime without editing code:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1 --yellow-strategy ssl_strategy_api.strategies.goalkeeper:GoalkeeperStrategy

python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1 --yellow-strategy ssl_strategy_api.strategies.goalkeeper_rl:QTableGoalkeeperStrategy
```

Rendering and duration options

If you do not want the simulator window to open, run in headless mode:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1 --no-render
```

This is useful for longer runs, batch evaluations, or RL training loops where visual rendering is unnecessary and can slow execution.

You can also control the maximum run duration:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1 --duration-seconds 300
```

The default duration is 90 seconds.

## Run Exercise 1 Evaluation

Run the Exercise 1 evaluation file with:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1_eval
```


### Exercise 1 vs Exercise 1 Eval

`run_rsoccer_exercise1.py` and `run_rsoccer_exercise1_eval.py` are very similar, but they are used for different purposes.

- `run_rsoccer_exercise1.py` is episode-based. The episode stops when the ball goes out of the field bounds, a goal is scored, or the ball is idle in the opponent goal area. Then a new episode is started and the blue robot along with the ball are respawned in random positions in the field. This is useful for RL training episodes and quick checks of rule-based strategies.
- `run_rsoccer_exercise1_eval.py` is the final evaluation runner. The episode does not stop on those events. Instead, when the ball goes out of the field bounds, a goal is scored, or the ball is idle in the opponent goal area, the ball is immediately respawned at a random valid position on the field, while robot positions remain unchanged.

## Exercise 2: Conflict Avoidance

The blue robot must reach its target position (orange circle on tha field) without touching any yellow robot, leaving the field, or entering the opponent goal area. Implement your strategy in `src/ssl_strategy_api/strategies/conflict_avoidance.py`. The runner calls `set_target(x, y)` on your strategy at the start of every episode; yellow robots are in `state.robots_yellow`. The reward function in `_calculate_reward_and_done` (`run_rsoccer_exercise2.py`) is left for you to implement in case you want to use an RL strategy.

**Robot size and collisions.** Every robot is a circle with a radius of 0.09 m (18 cm diameter). A collision is detected when the distance between the centres of the blue robot and any yellow robot is at most 0.18 m (0.09 m + 0.09 m), which means the robots touch. The collision check adds no safety margin: passing at 0.19 m is not a collision (if you want any extra margin, it has to come from your own strategy). The blue robot leaves the field when its centre crosses a field line.

```sh
# random episodes with static yellow robots
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise2
# random episodes with 10 yellow robots instead of 5
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise2 --num-obstacles 10
# random episodes with moving yellow robots (each axis of the velocity (vx and vy) is constant and random in [-1, 1] m/s per episode and per robot. For example yellow_robot1 can have a ux = 0.5 and uy = -0.2 and yellow_robot2 can have a ux = -0.7 and uy = 0.4)
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise2 --obstacle-speed 1
# harder random episodes: more yellow robots on the straight path to the target (levels 1-5)
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise2 --difficulty 3
# the following 3 fixed scenarios are provided to test your strategies
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise2_eval --scenario basic
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise2_eval --scenario static
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise2_eval --scenario dynamic
# all fixed scenarios in a row
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise2_eval --scenario all
```

### `run_rsoccer_exercise2.py` (develop and test your strategy)

Runs episodes back to back for `--duration-seconds` (default 90). Every episode uses new random positions on the right side of the field (x > 0):

- 5 yellow robots (`--num-obstacles`), not overlapping each other. They are static unless you set `--obstacle-speed` u (each velocity axis is drawn then once per episode from [-u, u] m/s and then stays constant; the yellow agents bounce back to the field if they reach the boundaries of the field).
- the blue robot with a random heading, outside the opponent goal area and on the right side of the field.
- a target in the right field, outside the opponent goal area and not on any robots' initial position.

**Difficulty levels (`--difficulty 1..5`, default 1).** Without any conflicting robots on the blue agent's path, the blue robot can simply drive straight to the target, so the level controls how often the yellow robots stand on that straight path. Each of the first 5 yellow robots is independently placed on the straight line from the blue start to the target with the probability shown on the table below (each robot is shifted sideways from the line by a random amount of up to 0.5 m on either side, so the robots are spread out around the path instead of standing exactly on it, and some only sit near the line). The other robots (if there are more than 5) are placed anywhere on the right side of the field. Thus they may also end up on the path of the blue robot by chance. At higher difficulty levels the target is also farther from the blue robot's starting position.

| Level | Chance that each yellow robot is on the path | Minimum blue-robot-to-target distance |
|---|---|---|
| 1 | 10% | none (random) |
| 2 | 25% | none (random) |
| 3 | 50% | 1.5 m |
| 4 | 70% | 1.5 m |
| 5 | 80% | 2.0 m |


An episode ends if the blue robot collides with any other robot, leaves the field, enters the opponent goal area, reaches the target, or after 30 seconds have passed. A new random episode then starts, and a summary of the outcomes is printed at the end. Use `--seed` for reproducible layouts. `Exercise2Env()` can also be used directly for RL: each `reset()` draws a new layout.

### `run_rsoccer_exercise2_eval.py` (fixed scenarios for testing your strategies)

- Scenarios (`--scenario`): `basic` (no yellow robots, just avoiding entering the opponent's goal area), `static` (3 fixed yellow robots), `dynamic` (3 static yellow robots and 1 yellow robot moving), or `all` to run the three in a row. Moving robots can roam anywhere in the field and bounce off the field lines.
- Each scenario defines its own blue start, yellow robots and target. The blue robot has 30 seconds to reach the target position. Reaching the target stops the run and prints the remaining seconds; any failure stops the run and reports the reason with 0 seconds remaining.
