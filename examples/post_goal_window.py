"""Train on the 3 seconds after a goal (goal celebrations) without playing whole games.

In Rocket League the cars keep driving for about 3 seconds after a goal; input stops only when the next
kickoff's countdown starts. catSim runs that window too, with `Arena(round_flow=True)`:

    arena.get_round().phase        2 while the window runs (0 kickoff countdown, 1 playing, 2 after a goal)
    arena.get_round().phase_left   seconds left in the window, from 3.0 down

This script runs post-goal episodes back to back:
  1. start_at_goal(): put the cars where you want them and the ball just inside a net, step one tick: goal.
  2. run_window(): your bot's controls every TICK_SKIP ticks, all through the window.
  3. celebration_reward(): an EXAMPLE reward (counts jumps and dodges). Swap in your own.
  4. The episode ends when the window does: the countdown starts, input is off, cars go back to kickoff spots.

In an RLGym-style env: start_at_goal is your state setter / reset, "phase is no longer 2" is your terminal
condition, celebration_reward is your reward function, and phase_left is a useful observation.

Run it:  python examples/post_goal_window.py path/to/catsim_data
"""
import math
import random
import sys

import catsim as rs

POST_GOAL = 2      # arena.get_round().phase while the post-goal window runs
TICK_SKIP = 8      # ticks between your bot's decisions (8 is what most bots use)
GOAL_Y = 5300      # a ball centre this far up the field is past the goal line (about 5215) and inside the net


def start_at_goal(arena: rs.Arena, scoring_team: rs.Team = rs.Team.BLUE) -> None:
    """Start an episode at a goal: the cars where you want them, then one tick with the ball inside the net."""
    arena.reset_kickoff()                    # a fresh round in play (phase 1), whatever the arena was doing
    for car in arena.get_cars():
        # EXAMPLE placement: anywhere on the field, on its wheels, facing any way, some speed and boost.
        # Put your own here, e.g. the scorer near the goal it just scored in, as after a real goal.
        rot = rs.Angle(yaw=random.uniform(-math.pi, math.pi)).to_rot_mat()
        car.set_state(rs.CarState(pos=rs.Vec(random.uniform(-3000, 3000), random.uniform(-4000, 4000), 17),
                                  rot_mat=rot, vel=rot.forward * random.uniform(0, 1500),
                                  boost=random.uniform(0, 100)))
    side = 1 if scoring_team == rs.Team.BLUE else -1      # blue scores in the goal at +y, orange at -y
    arena.ball.set_state(rs.BallState(pos=rs.Vec(0, side * GOAL_Y, 300), vel=rs.Vec(0, side * 500, 0)))
    arena.step(1)                            # the goal is scored on this tick; the window starts
    assert arena.get_round().phase == POST_GOAL


def bot(car: rs.Car, decision: int) -> rs.CarControls:
    """A stand-in for your bot (called every TICK_SKIP ticks): drive, jump, let go, dodge forward, drive, again.
    Replace it with your model's action."""
    step = decision % 10
    return rs.CarControls(throttle=1.0, jump=step in (0, 2), pitch=-1.0 if step == 2 else 0.0)


def celebration_reward(before: rs.CarHidden | None, after: rs.CarHidden | None) -> float:
    """EXAMPLE ONLY, swap in your own: +1 for each dodge the car starts, +0.5 for each jump.

    It reads catSim's hidden car state (docs/API.md, "CarHidden"): each component (jump, double_jump, dodge,
    boost, ...) has .active and .time (seconds since it changed), plus on_ground, jumped and double_jumped.
    For chained ground dashes you might reward dodges that start close to the ground (car.get_state().pos.z)
    or landing soon after a dodge, and only the scoring team's cars."""
    if before is None or after is None:      # a demolished car has no hidden state
        return 0.0
    r = 0.0
    if after.dodge.active and not before.dodge.active:
        r += 1.0
    if after.jump.active and not before.jump.active:
        r += 0.5
    return r


def run_window(arena: rs.Arena) -> tuple[list[float], int]:
    """Step every car's bot through the post-goal window. Returns each car's summed reward and the ticks run."""
    cars = arena.get_cars()
    totals = [0.0] * len(cars)
    prev = [c.get_hidden() for c in cars]
    tick = 0
    while True:
        if tick % TICK_SKIP == 0:            # a new decision for every car
            # arena.get_round().phase_left is a good observation: the seconds your bot has left
            for car in cars:
                car.set_controls(bot(car, tick // TICK_SKIP))
        arena.step(1)                        # one tick at a time, so the reward sees every tick
        tick += 1
        if arena.get_round().phase != POST_GOAL:
            # The window is over: this tick started the kickoff countdown (phase 0). Input is off and the cars
            # are back on kickoff spots, so end the episode here and don't score this tick.
            # (Stepping TICK_SKIP ticks at once? A short countdown can end inside one step, so check
            # "phase != 2", not "phase == 0".)
            return totals, tick
        now = [c.get_hidden() for c in cars]
        for i in range(len(cars)):
            totals[i] += celebration_reward(prev[i], now[i])   # per decision: sum its TICK_SKIP ticks
        prev = now


def main() -> None:
    rs.init(sys.argv[1] if len(sys.argv) > 1 else "catsim_data")
    arena = rs.Arena(round_flow=True)        # round_flow runs the post-goal window and the restart after it
    arena.add_car(rs.Team.BLUE, rs.CarConfig(rs.CarConfig.OCTANE))
    arena.add_car(rs.Team.ORANGE, rs.CarConfig(rs.CarConfig.OCTANE))
    for episode in range(5):
        start_at_goal(arena, rs.Team(episode % 2))
        r = arena.get_round()
        print(f"episode {episode}: goal, phase {r.phase}, {r.phase_left:.2f} s left, score {arena.score}")
        totals, ticks = run_window(arena)
        print(f"  window over after {ticks} ticks ({ticks / 120:.2f} s), phase now {arena.get_round().phase}, "
              f"rewards {[round(t, 2) for t in totals]}")


if __name__ == "__main__":
    main()
