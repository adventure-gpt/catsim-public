"""A fixed scenario through the Python API, hashed bit for bit. Run it on two machines (or OSes) with the same
data folder: the hashes must be equal.

    python tools/determinism_hash.py catsim_data [ticks]

Part 1: 1v1, 2v2 and 3v3 on Stadium_P with all six car types and kickoff resets. Part 2 (when the data folder has
items.txt): a 2v2 on UF_Night_P with the game's frame loop and the random-draw model on (the game's own
round flow, so goals restart the kickoff from the random stream).
"""
import os
import hashlib
import math
import struct
import sys

import catsim as rs


def main():
    data = sys.argv[1]
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    rs.init(data, silent=True)
    h = hashlib.sha256()
    for mode in (1, 2, 3):
        arena = rs.Arena(rs.GameMode.SOCCAR)
        configs = [rs.CarConfig.OCTANE, rs.CarConfig.DOMINUS, rs.CarConfig.PLANK, rs.CarConfig.BREAKOUT, rs.CarConfig.HYBRID, rs.CarConfig.MERC]
        cars = [arena.add_car(rs.Team(k % 2), rs.CarConfig(configs[k])) for k in range(2 * mode)]
        for t in range(ticks):
            if t % 8 == 0:
                for k, c in enumerate(cars):
                    ph = 0.001 * t * (k + 1)
                    c.set_controls(rs.CarControls(throttle=1.0 if (t // 240 + k) % 3 else -0.5, steer=math.sin(ph * 3.0),
                                                  pitch=math.cos(ph * 2.0), yaw=math.sin(ph), roll=math.cos(ph * 5.0),
                                                  jump=(t // 8 + k) % 37 < 3, boost=(t // 64 + k) % 4 == 0, handbrake=(t // 120 + k) % 9 == 0))
            arena.step(1)
            if t % 30 == 0:
                b = arena.ball.get_state()
                h.update(struct.pack("<9f", *b.pos, *b.vel, *b.ang_vel))
                for c in cars:
                    s = c.get_state()
                    h.update(struct.pack("<13f", *s.pos, *s.vel, *s.ang_vel, *s.rot_mat.forward, s.boost))
            if t % 1500 == 1499:
                arena.reset_kickoff()
        h.update(struct.pack("<2i", *arena.score))
    if os.path.isfile(os.path.join(data, "items.txt")):
        lo = dict(car_id=23, wheels_id=4842, antenna_id=12019, hat_id=12697, goal_explosion_id=12660)
        arena = rs.Arena(map="UF_Night_P", round_flow=True)
        cars = [arena.add_car(rs.Team(k % 2), loadout=lo) for k in range(4)]
        if arena.start_game_rng(0x1234567, 31.5) is None:
            raise SystemExit(f"draw model not covered: {arena.draw_model_note}")
        h.update(repr(arena.draw_parts()).encode())
        for f in range(ticks):
            if f % 8 == 0:
                for k, c in enumerate(cars):
                    ph = 0.002 * f * (k + 1)
                    c.set_controls(rs.CarControls(throttle=1.0, steer=math.sin(ph), jump=(f // 8 + k) % 29 < 2,
                                                  boost=(f // 32 + k) % 3 == 0, pitch=math.cos(ph)))
            if f % 1500 == 300:
                arena.ball.set_state(rs.BallState(rs.Vec(0, 4300, 300), rs.Vec(0, 3000, 0), rs.Vec()))
            steps = 0 if f % 97 == 0 else (2 if f % 89 == 0 else 1)
            arena.step_frame(steps, (1.0 + 0.01 * math.sin(f)) / 120.0, (f % 10) / 10.0)
            if f % 30 == 0:
                b = arena.ball.get_state()
                h.update(struct.pack("<9fI", *b.pos, *b.vel, *b.ang_vel, arena.crt_state))
                for c in cars:
                    s = c.get_state()
                    h.update(struct.pack("<10f", *s.pos, *s.vel, *s.ang_vel, s.boost))
        h.update(struct.pack("<2iI", *arena.score, arena.crt_state))
    print(h.hexdigest())


if __name__ == "__main__":
    main()
