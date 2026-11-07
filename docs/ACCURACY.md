# Accuracy: what is claimed, and how to check it

catSim claims only what has been measured. Each line says what was compared, how much, and how to run it. All the
checks below used game data made by this repo's dumper from a fresh Steam install, nothing else: from Steam build
25914621 on 2026-11-04, from build 26349479 on 2026-11-06. Both builds give byte-identical data, and
`catsim-dump --check catsim_data` tells you whether yours is that data.

## Checks you can run yourself

| Claim | Evidence | Run it |
|---|---|---|
| The fast engine is bit-identical to the scalar reference engine | Differential fuzzing: random 1v1/2v2/3v3 arenas, every car body and preset, random inputs, kickoffs, goals and demolitions; every body, contact manifold, car and arena field compared after every tick. 20,000,000 ticks with no difference; 4,000,000 more with the random-draw model on; 4,000,000 on UF_Night_P's layout (Windows). On Linux: 2,000,000 plus 1,000,000 with the draw model | `catsim_fuzz catsim_data/soccar 20000000 4 1` (add `rng-mesh` or `pad-map` as the fifth argument) |
| The differential notices a change | One changed velocity bit, one flipped event flag or one wrong random draw each fail at tick 64 | the fifth argument `velocity`, `event` or `rng-fault` |
| The fast engine's C runtime routines equal the reference engine's | 2,000,000 random float arguments, every routine, bit for bit | `catsim_crt 2000000` (`all` tries all 2^32 bit patterns) |
| The C runtime's tables are their mathematical definitions | All four tables recomputed with 300-bit arithmetic: 819 values, all equal | `python tools/gen_crt_tables.py --check` |
| The same inputs give the same result, on Windows and Linux | A fixed scenario hashed bit for bit: 1v1, 2v2 and 3v3 on Stadium_P with all six car types and kickoff resets, then a 2v2 on UF_Night_P with the game's frame loop, goals, restarts and the random-draw model. Windows 11 and Ubuntu 24.04 give the same hash, `c381a857bd0f...` | `python tools/determinism_hash.py catsim_data` on each machine |
| Ball prediction is the engine itself | A prediction 240 ticks ahead equals stepping the arena 240 ticks, bit for bit | tests/test_api.py |

(`catsim_fuzz` and `catsim_crt` are in `engine/`: `cargo run --release -p catsim_proof --bin catsim_fuzz -- ...`.)

## Checked by the maintainers against the development engine

These compare this repo with the private development engine it was copied from (at commit 6a45bcd1), which has
been checked against the real game for a long time and reads its game data its own way.

- **The dumper's data equals the development engine's**, every table and every mesh bit for bit, except two
  values where the dumper has the game's own value and the development engine doesn't: the small boost pads'
  height on Stadium_P (the game's is `0x428029fe`, measured in the game) and the default pickup box, which no car body uses (docs/PROVENANCE.md). Two more Steam
  builds give byte-identical data (25641912, and 26349479 from 2026-11-06). On 2026-11-06 the presets, the steer
  folds and the bodies were compared again with the development engine's current tables: all equal.
- **The whole engine**: 16,000,000 ticks of the same differential fuzzing, development engine (its own data)
  against this repo's engine (the dumper's data only): 10,000,000 plain, 3,000,000 with the random-draw model,
  3,000,000 on UF_Night_P. No difference.
- **The check is sensitive to the data**: one float step in the Octane's hitbox in the dumped data is caught at
  tick 0; one flipped bit in one floor vertex at tick 26,071.
- **The random-draw model's parts**: for the cat loadout (Octane, Yin Yang Cats wheels, Cat Toy antenna, Cosmic
  Kitty topper, Kitty Up goal explosion) on UF_Night_P in a 2v2, the parts read from the dumper's `items.txt`
  give the same 29 timers in the same order as the development engine's hand-written settings, and over the
  whole recorded game below both engines predict the same random stream (identical comparison statistics).

## Checked by the maintainers against the real game (recordings not published)

- **A whole recorded game replays exactly.** One of our own RLBot games (a 2v2 on UF_Night_P with the cat
  loadout, five minutes plus two goals, 38,285 ticks), replayed from its start state and the recorded inputs and
  frame timing through this repo's reference engine with the dumper's data and the draw model's settings from the
  dumper's `items.txt`: **every one of the 37,716 compared ticks is exact**, every car and the ball, including the
  ball's rotation, through both goals, to the end of the match. The recording is ours and stays private, so this
  one can't be re-run by others; it is stated so you know what has been checked. This replay checks the physics
  and the dumped data, not the random-draw model: with a deliberately wrong draw-model setting it is exact too
  (in this game the kickoff spots didn't depend on the random stream).
- **The random-draw model against the game's own random stream**, in the same game: the game's rand() state at
  the end of each frame (read while the game ran) equals the model's prediction for every frame from the start
  until the first draw the model doesn't cover (255 frames in, one draw short). With the antenna's animated part left out of the settings, the
  prediction goes wrong at that part's first update instead (39 frames in, four draws short: one per car). So the
  item data is right; how long a prediction stays exact is limited by the draws the model leaves out (below).
- Packets show the game's state after its own rounding (positions to 0.01, an integer rotation), and catSim
  reproduces that rounding.

## The game data on two sets of recordings (2026-11-06)

Run with the benchmark harness (`python -m simbench`, in `bench/`) on data from Steam build 26349479. Both sets of
recordings were made before that build, on earlier ones.

- **The public RocketGPU full-state suite**, its 36 recordings with a Plank-hitbox car, at the suite's level A
  (each segment started from the game's full internal state): 99.29% of 4,507,550 body-ticks and 96.08% of 7,090
  segments exact, the same to the last digit on Windows 11 and on Linux (WSL).
- **Our own 39 recorded RLBot games**, whole, at level C (the game's kickoff and respawn choices given): all 39 exact
  to the end, 1,642,789 of 1,642,789 compared rows. At level D (catSim's own choices from the random stream, on the 4
  games that have what it needs): 4 of 4 exact to the end.
- **The Darkcar rule is what the game does.** The Darkcar body has no handling preset; the dumper runs it on Plank's
  with the wheel rest positions rounded to 0.01 uu (docs/PROVENANCE.md). Left out (Plank's rests as stored, every
  other file the same), the same runs give 88.58% of body-ticks and 29.13% of segments at level A, and 24 of the 39
  games exact to the end at level C.

## What isn't claimed

- Other loadouts' random draws have not been checked against recordings: the model applies the same measured
  rules to any items the dumper describes. Known limits: the draws of the online services', cache, crowd-chant
  and save timers (not predictable from the game's state), and one loadout shape for all cars.
- Only soccar, on Stadium_P and UF_Night_P, with the 19 car bodies the data folder lists.
- The Epic edition's dump was compared with Steam's on one install (byte-identical, docs/DUMPER.md); other Epic
  builds haven't been.

## Speed

catSim does more work per tick than RocketSim (it follows the game's own order of operations), so it is slower.
No speed claim is made in this repo yet.
