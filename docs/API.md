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
arena.clone()
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
| Speed | Very fast | Slower: exactness costs time | See docs/ACCURACY.md |
| Car state fields | Many | The common ones (see `CarState`) | Others to come as asked for |

Determinism: the same arena, states and controls always give the same result, bit for bit, on Windows and
Linux. Threads are not used inside an arena.
