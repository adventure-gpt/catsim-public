# The Python API

`import catsim as rs` gives the same names RocketSim's Python bindings use, so most scripts run unchanged.
This page lists what's there and, more importantly, what behaves differently.

## Quick reference

```python
rs.init("catsim_data")                          # RocketSim: rs.init("collision_meshes")
arena = rs.Arena(rs.GameMode.SOCCAR)            # tick rate is always 120
car = arena.add_car(rs.Team.BLUE, rs.CarConfig(rs.CarConfig.OCTANE))
car.set_controls(rs.CarControls(throttle=1, steer=0, pitch=0, yaw=0, roll=0, jump=False, boost=True, handbrake=False))
arena.step(8)                                   # ticks
s = car.get_state()                             # CarState: pos, vel, ang_vel, rot_mat (forward/right/up), boost, flags
car.set_state(s)
b = arena.ball.get_state(); arena.ball.set_state(b)
arena.reset_kickoff()
arena.get_cars(), arena.remove_car(car), arena.get_car_from_id(1)
arena.get_boost_pads()[0].get_state()           # is_active, cooldown
arena.set_goal_score_callback(fn, data)         # fn(arena, scoring_team, data)
arena.set_car_bump_callback(fn, data)           # fn(arena, bumper, victim, is_demo, data)
arena.set_car_ball_collision_callback(fn, data) # fn(arena, car, data)
arena.ball_prediction(ticks=360, every=1)       # catSim extra: list of BallState
arena.clone()                                   # same cars and states, hidden state included
arena.copy()                                    # catSim extra: an exact copy (contact caches and all)
car.get_hidden(), car.set_hidden(h)             # catSim extra: the hidden state (below)
car.get_body(), car.set_body(b)                 # catSim extra: the body exactly, in Bullet units
```

Types: `Vec`, `RotMat`, `Angle`, `CarControls`, `CarState`, `BallState`, `BoostPad`, `BoostPadState`,
`CarConfig`, `GameMode`, `Team`, `Loadout`.

Maps: `rs.Arena(map="Stadium_P")` (the default, the standard soccar arena) or `map="UF_Night_P"` (the same field,
with its own pad heights and no planes behind the goals). `Arena(round_flow=True)` runs the game's own goal
explosion, post-goal time and kickoff restart; with it off, the arena waits after a goal until you call
`reset_kickoff()`.

## Replaying real games: frames and the game's random numbers

The game runs physics inside an application frame loop: a frame of real duration takes 0, 1 or more 1/120 s
ticks. And the game's kickoff spawn choices come from the C runtime's rand() stream, which many other things
also draw from. To replay a real game exactly, catSim takes both:

```python
arena = rs.Arena(map="UF_Night_P", round_flow=True)
cats = dict(car_id=23, decal_id=5337, wheels_id=4842, boost_id=4994, antenna_id=12019, hat_id=12697,
                trails_id=8969, goal_explosion_id=12660)          # RLBot loadout names and product ids
for team in (0, 1, 0, 1):
    arena.add_car(rs.Team(team), loadout=cats)

arena.start_game_rng(crt_state, world_time)   # the game's rand() state and clock at the start, if you know them
arena.draw_model_note                          # "covered", or why the animated parts' timers are off
arena.step_frame(steps=1, frame_time=0.00833, render_alpha=0.4)   # one frame of the game's loop
arena.crt_state                                # the random state now
arena.draw_parts()                             # the animated parts' timers, in the game's update order
```

`start_game_rng` turns on the model of everything that draws from the stream (docs/PROVENANCE.md lists the
rules): the animated parts' level-of-detail timers (from your data folder's `items.txt` and the cars'
loadouts), effects (boost pickups, jumps, dodges, goal explosions, demolitions), two draws at every frame end,
the director camera's draws at each countdown (`director_camera=False` leaves them out), and the respawn spot
choice. When the data doesn't cover the match (an item the dumper couldn't read or has no rule for, or cars
whose items differ in their animated parts), it warns, says why in `draw_model_note`, and runs without the parts'
timers.

Some draws can't be predicted from the game's state (the online services', cache, crowd-chant and save timers).
If you know such a timer's phase, pass it: `start_game_rng(..., extra_timers=[(next_due_time, period,
random_extra_period)])`; each draws once when due if its random extra period is above 0.

Car configs: `CarConfig.OCTANE`, `DOMINUS`, `PLANK`, `BREAKOUT`, `HYBRID`, `MERC`, or any of the game's
handling preset names (`"Octane_Skunkz"`, `"LongCar_hero"`...) or a car body's name or product id
(`"Venom"`, `"23"`). `rs._lib().preset_names()` and `body_names()` list them.

## Hidden state: restoring a recorded or saved state exactly

New in 0.1.3: the hidden state below (`CarHidden`, `get_hidden` / `set_hidden`, `get_body` / `set_body`, `arena.copy`,
`get_round` / `set_round`, the pads' timers), the contact memory (`arena.get_contacts` / `set_contacts`) and the body
order (`arena.body_order` / `set_body_order`).

A packet shows where a car is and how it moves. The next ticks also read what it doesn't show: the jump and dodge
clocks, the input the car still holds, its wheels' contacts, the gates that stop a second ball hit or bump too soon.
catSim lets you read and set all of it, so a state you save (or one recorded from the game) can be put back exactly.

```python
s = car.get_state()                 # s.hidden is filled in: a CarHidden
car.set_state(s)                    # position, rotation, velocities, boost, then s.hidden when present

h = car.get_hidden()                # the hidden state alone
h.jump.active, h.jump.time          # each game component: on or off, and seconds since it changed
car.set_hidden(h)

b = car.get_body()                  # the body exactly as the physics holds it: Bullet units (1 = 50 uu), the
car.set_body(b)                     # rotation matrix by rows, no rounding anywhere; arena.ball.get_body() / set_body()

twin = arena.copy()                 # an exact copy: bodies, contact caches, every timer; stepping both gives the same
arena.get_round(), arena.set_round(r)   # the physics frame, the round's phase and timers, the score
pad.get_timer(), pad.set_timer(taken, length, elapsed)   # a boost pad's timer exactly; pad.set_state(BoostPadState)
```

`CarHidden` holds, with the game's meanings:

| Field | What it is |
|---|---|
| `driving` | the car takes input (off during a countdown and after a goal) |
| `on_ground`, `supersonic`, `supersonic_time` | the game's flags; seconds below supersonic speed while still counted supersonic |
| `jumped`, `double_jumped` | the game's own jump flags (set by a jump, a double jump or a dodge; cleared on landing) |
| `jump`, `jump_released` | the jump component, and whether the button was let go during it |
| `double_jump`, `dodge`, `auto_flip`, `boost`, `air_control` | the other components: `active` and `time` |
| `dodge_torque`, `dodge_direction` | the dodge's whole-number torque and its direction; `car.dodge_from_input(forward, right)` gives both the game's way |
| `auto_flip_right`, `auto_flip_duration` | the automatic flip back onto the wheels |
| `boost_amount` | boost 0 to 1, exactly (a packet shows it times 100, rounded) |
| `handbrake` | the smoothed handbrake, 0 to 1 |
| `held_input` | the controls the car holds from its last tick (axes as the game stores them, 8 bits each) |
| `wheels`, `world_contact`, `world_contact_normal` | each wheel's contact and normal; the body's contact with the arena |
| `bump_victim`, `bump_time` | the car it last bumped and when (physics time); the same car can be bumped again 0.25 s later |
| `ball_impact_frame`, `ball_touch_frame` | the physics frame of its last ball hit with the extra impulse, and of its last touch |

More for setting up a start: `car.demolish_now(respawn_in)`, `car.queue_respawn(pos, rotator)`,
`arena.queue_kickoff(spots)`, `car.set_driving(on)`, `car.add_boost(amount)`, `arena.ball.set_held(on)`,
`arena.set_pad_rule(on)` (off: pads change only when you say), `arena.use_game_countdown()` (later kickoffs run the
game's own 3-second countdown clock). The Rust API has the same in `catsim::game::hidden`.

A state copied from a car on its wheels or in the air, or a ball in the air, runs on identically (tests/test_hidden.py
checks this). One with a cached contact needs the contact memory too. `arena.copy()` keeps everything.

**Contact memory:** what a step reads from the last one besides the bodies. `arena.get_contacts()` gives a
`ContactMemory`: `boxes` (each moving body's broadphase box as the last bounds update left it: (body, min, max)), `pairs`
(the broadphase's overlapping pairs in their order: (body0, body1, dispatched)) and `manifolds` (each `Manifold`: its
`owner`, "pair" or "predictive", and `index`, `child`, `body0`, `body1`, and up to 4 cached `ContactPoint`s).
`arena.set_contacts(m)` puts one back; on an error nothing changes. Bodies are named "ball", "car<i>" (the car's place in
`get_cars()`) or "world<k>" (`arena.world_parts()`'s k-th plane or mesh). `ContactPoint.fields()` names a point's numbers,
and `arena.contact_point(body0, body1, local_a, local_b, normal, distance)` makes a point the game's way from the point on
each body, the normal and the distance. The native arena has the same as `contacts()` / `set_contacts()`, and the Rust API
as `Arena::contacts` / `set_contacts`.

**Body order:** `arena.body_order()` / `arena.set_body_order(order)` read and set the order of the moving bodies
("ball", "car<i>") in the collision world. Each tick the game numbers its islands in this order and the solver takes
the contact manifolds sorted by island, so it decides which of a body's contacts is solved first. It has a history: a
demolished car leaves the array (the last body takes its place) and its respawn goes on the end.

### RocketSim v3's CarState in catSim's terms

For tools that hold states in RocketSim v3's names (its rlpr_metric, the full-state suite's public rows):

| v3 field | catSim |
|---|---|
| `phys` (pos, rot_mat, vel, ang_vel) | the body: `set_body` with position x 0.02, the matrix by rows, velocity x 0.02 |
| `is_on_ground`, `is_supersonic`, `supersonic_grace_timer` | `on_ground`, `supersonic`, `supersonic_time` |
| `has_jumped` / `has_double_jumped`, `has_flipped` | `jumped` (any of the three) / `double_jumped` (double jump or flip) |
| `is_jumping`, `jump_ticks`, `jump_hold_broken` | `jump.active`, `jump.time = jump_ticks / 120` (`air_time_since_jump` when not jumping), `jump_released` |
| `is_flipping`, `flip_time`, `flip_rel_torque` (x, y) | `dodge.active`, `dodge.time`, torque (round(side torque x), round(forward torque y)) and direction (y, -x) |
| `is_auto_flipping`, `auto_flip_timer`, `auto_flip_torque_scale` | `auto_flip.active`, the time left as its duration, `auto_flip_right` from the sign |
| `boost` (0-100), `is_boosting`, `boosting_time`, `time_since_boosted` | `boost_amount / 100`; `boost.active`; `boost.time` (the first while boosting, the second when not) |
| `handbrake_val`, `prev_controls` / `controls` | `handbrake`, `held_input` |
| `wheels_with_contact` (v3's order: front left, front right, back left, back right) | `wheels` (catSim's order swaps left and right: 1, 0, 3, 2) |
| `bump_cooldown_timer`, `bump_cooldown_victim` | `bump_victim`, `bump_time = now - (0.25 - timer)` |
| `last_extra_hit_tick` | `ball_impact_frame`, the same number of frames before `arena.physics_frame` |
| `is_demoed`, `demo_respawn_timer` | `car.demolish_now(timer)` |
| `air_time` | no counterpart (not read by the game's physics) |

bench/adapters/catsim_py.py does exactly this, and is a working example.

### Replaying a recording from its start

To start from a real game's own kickoff (the benchmarks' levels C and D, docs/BENCHMARKS.md), the native arena takes
a few more calls: `NativeArena(..., kickoff=False, countdown_ticks=n)` skips catSim's own kickoff, then
`start_at_drop(spots)` / `restart(spots, steps)` build the bodies at given spots, `set_next_handles(order)` the
broadphase handles, `set_birth_delay`, `set_countdown(...)`, `set_spawn_selector(...)` the game's kickoff spot choice,
`set_match_seconds`, `round_gate`, `set_scripted`, and `packet_view()` / `clock()` read what a packet would show.
bench/simbench/game.py and bench/adapters/catsim_py.py (`game`) use them.

## Differences from RocketSim

| Topic | RocketSim | catSim | Why |
|---|---|---|---|
| Setup | `init(collision_meshes_dir)` | `init(catsim_data_dir)`, made by `catsim-dump` | catSim needs more of the game's data than the meshes |
| Game modes | Soccar, Hoops, Heatseeker, Snowday, Dropshot | **Soccar only** | Only soccar has been ported and proven |
| Tick rate | Any | **120 only** | The game's own rate |
| Adding or removing cars | Any time, state kept | Rebuilds the arena and starts a kickoff | The world is built in the game's order once its cars are known |
| A new arena | Cars at the origin until you place them | Cars at kickoff spots, ball in the middle, as the game starts a kickoff | Same flow as the game |
| Kickoff spots | Random, seedable | Chosen the game's way; the `seed` argument is ignored | |
| Setting a car's rotation | Exact matrix | Goes through the game's integer rotator (65536 units a turn), like the game's own state setting | Matches what the real game does when a state is set |
| Setting boost | Float | Rounded down to a whole number, 0 to 100 | Same as the game's state setting |
| Ball set at rest | Stays awake | `set_state(state, sleep_at_rest=True)` makes it sleep at once, like the game | Optional, default off |
| Goals | Reported; nothing else happens | Reported, and the goal explosion removes the ball as in the game; the arena waits for `reset_kickoff()`. With `Arena(round_flow=True)` the game's own post-goal time and restart run | |
| Maps | Soccar arena | Stadium_P and UF_Night_P | The two maps the engine has been checked on |
| `car.demolish()` | Available | Not available | Demolitions happen only through the game's rules |
| `get_gym_state()` and other array helpers | Some bindings have them | Not yet | |
| Ball prediction | Clone an arena and step it | `arena.ball_prediction()` (a ball-only arena, cars left out) | |
| Speed | Very fast | Slower: exactness costs time | Measured side by side in docs/BENCHMARKS.md |
| Car state fields | Many, set through `set_state` | The common ones in `CarState`, the rest in `CarState.hidden` / `CarHidden` (above) | catSim's fields follow the game; the table above maps v3's |

Determinism: the same arena, states and controls always give the same result, bit for bit, on Windows and
Linux. Threads are not used inside an arena.
