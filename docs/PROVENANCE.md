# Where every piece of catSim comes from

For everything catSim uses from the game: where it comes from, and what this repo does with it.

The repo holds our own code, rules written in our own words, and math that can be recomputed. It holds no game
code, no data copied out of the game, no tools that touch the running game, and no recordings. Like RocketSim,
catSim was built by studying the game closely; the game's data comes from your own install through the dumper.

- **MIT licence, owner adventure-gpt.**
- **You bring the package key file.** The keys are not stored in plain form anywhere in the game's files;
  catSim never ships or downloads them.

## The game data: all of it comes from the user's own install

`catsim-dump` writes the arena meshes, every table the engine loads and the car items' data for the random-draw
model. Checked on a Windows Steam install (build 25914621) bit for bit against the development engine's tables. Builds
25641912 and 26349479 (2026-11-06) give byte-identical files, and on 2026-11-06 the presets, steer folds and bodies
were compared again with the development engine's current tables: equal.

| Data | Where the dumper reads it | Against the development engine |
|---|---|---|
| Arena meshes (16) and their placement | `Stadium_P.upk`: the level's cooked per-triangle collision cache; the components' mirror and translation | Bit-exact (three Steam builds give the same files) |
| Arena planes (6) | `Stadium_P.upk`: the plane mesh's bounds and its six placed components; the basis by the engine's own rotation chain (rule below) | Bit-exact |
| Goal volumes (2) | `Stadium_P.upk`: the goal volumes' brush polygons and locations | Bit-exact |
| Boost pads (34) | `Stadium_P.upk`: the placed pickups (inheriting the prefab's locations) | x, y, size, big pads' height exact; small pads' height: see below |
| UF_Night_P | `UF_Night_P.upk`: its own pad heights; checked to have Stadium_P's 16 meshes and Stadium_P's last four planes (the engine's rule for that map) | Bit-exact |
| Handling presets (18) | `Startup.upk` and the bodies' packages: preset axles and physics boxes; `TAGame.upk` defaults; the float arithmetic in `catsim/dump/cars.py` | Bit-exact (all 18) |
| Folded steer curves | `GameInfo_Soccar_SF.upk` car archetype's curves, folded per preset (correctly rounded tan/atan) | Bit-exact (17 presets x 8 keys; the dumper also folds an 18th row the engine doesn't use) |
| Car bodies (19) | Each body's package: preset, axles, offsets, skeletal mesh bounds, origin and wheel bones (composed leaf first) | Bit-exact (all 19) |
| Vehicle variant (Psyclops) | Its body package: the mutator's vehicle, wheels, components | Bit-exact |
| Pickup boxes | The presets' and bodies' physics boxes | Bit-exact (8) |
| Default mesh bounds | The Octane body's skeletal mesh | Bit-exact |
| Default pickup box | The Octane preset's physics box | See below |
| Car items for the draw model (7,882 products) | `TAGame.upk`'s product database (product id -> slot, package), then each item's package: the body's mesh, the wheels' settings, the attachments, the antenna's stick, the goal explosion's effects actor | For the cat loadout (Octane, Yin Yang Cats wheels, Cat Toy antenna, Cosmic Kitty topper, Kitty Up explosion) on UF_Night_P, the same 29 animated parts in the same order as the development engine |
| Maps' animated parts | Skeletal mesh components placed in the level and its always-loaded sublevels | UF_Night_P 5, Stadium_P 0 (UF_Night_P's 5 match an object list from the running game) |
| C runtime tables | Computed from their definitions (`tools/gen_crt_tables.py`) | Bit-exact (819 values) |
| Sine table | Computed at start | Math |

### Two places where the dumper gives the game's value and the development engine doesn't

1. **Small boost pads' height on Stadium_P.** The package says `0x428029fe` (64.08201599); the development engine
   has `0x428029fb` (64.08199310), three float steps lower. **The game uses `0x428029fe`.** Evidence: measured in the
   game on Stadium_P on three days, every small pad sits at `0x428029fe` and every big pad at `0x42900000` (72). Why they differ:
   the actors have no Location of their own; they inherit the prefab's, the prefab instance sits at the origin,
   and nothing at load time moves them, so the package's value is the game's. The development engine's constant
   is a slip made when it was first typed in: its own comment (64.082016) and its own notes (64.08201599121094)
   both give `0x428029fe`. catSim uses the package value.
2. **The default pickup box** (used only by handling presets that no car body uses, such as `Hybrid_Skunkz`).
   The development engine has rounded readings of the Octane's box (59.0035, 42.0995...); catSim uses the Octane's
   physics box exactly.

## Documented rules (our own words, in our own code)

Things the files don't say, each stated where it is used:

- **Goal pieces' origin:** a piece translated by +-5120 uu sits at +-0x42CCCCCB physics units (measured).
- **Mesh pieces' basis zero signs** follow their mirror (measured, the same on all 16 pieces).
- **Plane bases:** rotation matrix through the sine table, rows scaled by Scale3D and normalised (1/sqrt),
  through a quaternion; signed zeros become +0. Reproduces all six planes.
- **Spawn bases** for the six kickoff and respawn yaws: the axes the game leaves a spawned car with
  (measured; a constant with a doc comment in `game/arena.rs`; other yaws use the formula).
- **Bodies without a handling preset** (the Darkcar) run on the preset whose physics box equals theirs (Plank),
  with that preset's wheel rest positions rounded to 0.01 uu. Measured in the game: the Darkcar's wheels
  sit at y 27.8 and 20.28 where Plank's preset says 27.799904 and 20.280005. The recordings agree: with the rule
  left out, the full-state suite's 36 recordings with a Plank-hitbox car drop from 99.29% to 88.58% exact
  body-ticks, and our own whole games from 39 of 39 to 24 of 39 exact to the end (docs/ACCURACY.md).
- **Pickup boxes** of their own only for the eight presets checked in the game; other presets use the default.
- **Product ids** of the 19 bodies (the public numbers RLBot loadouts use).
- **Steer fold:** tan and atan correctly rounded (reproduces every folded key).
- **The random-draw model's rules** (`catsim/draw.py`, measured in the game, every draw attributed to its
  caller): each skeletal mesh component draws once every 0.75 to
  1.25 s; effects draw when they start (a boost pickup 5, a jump 6, a dodge 7, a goal explosion 7 three frame
  ends after the goal, a demolition 6 four frame ends after it); 2 draws at every frame end; the director
  camera's 2 at each countdown; a kickoff restart 2 + 7 per car; a respawned car's new body joins the update
  list two frame ends later and its other parts three; the in-frame order (map parts, pickup and demolition
  effects, the cars' parts in the order the cars were made, each car's jump and dodge effects after its own
  parts, the goal explosion's part, the frame end).
- **Default items** (id 0 in a loadout) add no animated parts (measured with the default loadout: only the car
  body is skeletal).

## Known limits of the draw model

 some draws come from things the game's state can't predict: the online
services' timers, the online-data cache timer, the crowd-chant timer and the post-match save timers. The model
leaves them out; a replay's random stream shifts when one fires. `Arena.start_game_rng(extra_timers=...)` is the
hook for adding such timers when their phases are known (for example from a start snapshot). The model also
takes one loadout shape for all cars, and items whose parts it has no rule for (particle attachments, effects
actors, several skeletal parts in one goal explosion...) leave a match uncovered.

## Recordings

None ship. The maintainers check catSim against their own recorded RLBot games; docs/ACCURACY.md reports those
checks. The public checks in this repo use synthetic scenes made by catSim.
