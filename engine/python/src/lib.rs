//! The Python binding: one arena at a time, plain lists in and out. The friendly, RocketSim-style
//! classes are written in Python on top of this (python/catsim/__init__.py).

use catsim::adapter::{self, ArenaSlot};
use catsim::game::arena::{Arena, Phase, RoundConfig};
use catsim::game::car::VehicleInputs;
use catsim::game::config::{body_spawn, preset, Preset};
use catsim::game::pads::PadLayout;
use catsim::game::rb_state::RbState;
use catsim::game::hidden::{self, BodyState, CarHidden, RoundHidden};
use catsim::math::rot::Rotator;
use catsim::math::v3;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::{Path, PathBuf};

fn rt(e: impl ToString) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn soccar_dir(dir: &Path) -> PathBuf {
    if dir.join("soccar").is_dir() { dir.join("soccar") } else { dir.to_path_buf() }
}

fn find_preset(name: &str) -> Option<(&'static Preset, Option<&'static catsim::game::config::BodySpawn>)> {
    if let Some(b) = body_spawn(name) {
        return preset(b.preset).map(|p| (p, Some(b)));
    }
    let mut c = name.chars();
    let cap = c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default();
    preset(name).or_else(|| preset(&cap)).map(|p| (p, None))
}

/// Load the game data from the dumper's folder. Every arena needs it; the first folder loaded wins.
#[pyfunction]
fn init(data_dir: PathBuf) -> PyResult<()> {
    let dir = soccar_dir(&data_dir);
    catsim::data::load(&dir).map_err(rt)?;
    catsim::bt::bvh::soccar_meshes(&dir).map_err(rt)?;
    Ok(())
}

#[pyfunction]
fn preset_names() -> Vec<String> {
    catsim::game::config::PRESETS.iter().map(|p| p.name.to_string()).collect()
}

#[pyfunction]
fn body_names() -> Vec<(u32, String, String)> {
    catsim::game::config::BODIES.iter().map(|b| (b.product, b.name.to_string(), b.preset.to_string())).collect()
}

/// Car presets' hitbox: full size (uu, x y z) and the box's offset from the car's origin (uu).
#[pyfunction]
fn preset_hitbox(name: &str) -> PyResult<([f32; 3], [f32; 3])> {
    let (p, _) = find_preset(name).ok_or_else(|| PyValueError::new_err(format!("no car preset {name}")))?;
    let f = f32::from_bits;
    Ok(([f(p.half[0]) * 100.0, f(p.half[1]) * 100.0, f(p.half[2]) * 100.0], [f(p.offset_xz[0]) * 50.0, 0.0, f(p.offset_xz[1]) * 50.0]))
}

#[pyfunction]
fn car_fields() -> Vec<String> {
    adapter::car_field_names()
}

#[pyfunction]
fn arena_fields() -> Vec<String> {
    adapter::arena_field_names()
}

#[pyfunction]
fn hidden_fields() -> Vec<String> {
    adapter::hidden_field_names()
}

/// The names of `NativeArena.car_hidden`'s numbers, in order.
#[pyfunction]
fn car_hidden_fields() -> Vec<String> {
    hidden::CAR_HIDDEN_FIELDS.iter().map(|s| s.to_string()).collect()
}

/// The names of `NativeArena.car_body`'s and `ball_body`'s numbers, in order.
#[pyfunction]
fn body_fields() -> Vec<String> {
    hidden::BODY_FIELDS.iter().map(|s| s.to_string()).collect()
}

fn layout(map: &str) -> PyResult<PadLayout> {
    PadLayout::from_map(map).ok_or_else(|| PyValueError::new_err(format!("catSim knows the maps Stadium_P and UF_Night_P, not {map}")))
}

/// The 34 boost pads of a map: ((x, y, z) in uu, big), in the packets' order (by y, then x).
#[pyfunction]
#[pyo3(signature = (map = "Stadium_P"))]
fn pad_locations(map: &str) -> PyResult<Vec<([f32; 3], bool)>> {
    let l = layout(map)?;
    Ok((0..catsim::game::pads::NUM_PADS).map(|i| l.definition(i)).map(|p| ([p.loc.x, p.loc.y, p.loc.z], p.big)).collect())
}

fn inputs(rows: &[[f32; 8]], n: usize) -> PyResult<Vec<VehicleInputs>> {
    if rows.len() != n {
        return Err(PyValueError::new_err(format!("need one control row per car ({n}), got {}", rows.len())));
    }
    Ok(rows.iter().map(|r| adapter::input_from_row(r)).collect())
}

#[pyclass(unsendable, module = "catsim._native")]
#[derive(Clone)]
pub struct NativeArena {
    slot: ArenaSlot,
    dir: PathBuf,
    ticks: u64,
    round_flow: bool,
    predictor: Option<Arena>,
}

impl NativeArena {
    fn kickoff(&mut self) {
        let a = &mut self.slot.arena;
        a.start_at_drop(None);
        let n = self.slot.num_cars;
        for _ in 0..2 {
            a.frame(&vec![VehicleInputs::default(); n], 1);
        }
        a.phase = Phase::Active;
        a.ball_held_2d = false;
        a.scripted = !self.round_flow;
        a.rule_respawns = true;
        for p in &mut a.players {
            if let Some(c) = p.car.as_mut() {
                c.driving = true;
                c.dodge.c.time = 0.0;
            }
        }
        self.slot.time_since_touch = 0.0;
        self.slot.episode_time = 0.0;
        self.slot.goal = -1.0;
    }

    fn check_car(&self, i: usize) -> PyResult<()> {
        if i < self.slot.num_cars { Ok(()) } else { Err(PyIndexError::new_err(format!("no car {i}"))) }
    }
}

#[pymethods]
impl NativeArena {
    /// An exact copy of this arena: bodies, contact caches, timers, everything (the next ticks are identical).
    fn copy(&self) -> Self {
        self.clone()
    }

    /// The physics frame number (one more each tick).
    #[getter]
    fn physics_frame(&self) -> i32 {
        self.slot.arena.physics_frame
    }

    /// A car's hidden state as numbers named by `car_hidden_fields()`, or None while it is demolished.
    fn car_hidden(&self, i: usize) -> PyResult<Option<Vec<f64>>> {
        self.check_car(i)?;
        Ok(self.slot.arena.car_hidden(i).map(|h| h.to_row()))
    }

    /// Set a car's hidden state from `car_hidden_fields()` numbers. The held input goes through the game's input
    /// store (8-bit axes) with no jump press or release.
    fn set_car_hidden(&mut self, i: usize, row: Vec<f64>) -> PyResult<()> {
        self.check_car(i)?;
        let h = CarHidden::from_row(&row).ok_or_else(|| PyValueError::new_err(format!("need {} numbers", hidden::CAR_HIDDEN_FIELDS.len())))?;
        if !self.slot.arena.set_car_hidden(i, &h) {
            return Err(PyRuntimeError::new_err("that car is demolished and has no hidden state to set"));
        }
        Ok(())
    }

    /// A car's body exactly as the physics holds it, as numbers named by `body_fields()` (Bullet units: 1 = 50 uu),
    /// or None while it is demolished.
    fn car_body(&self, i: usize) -> PyResult<Option<Vec<f64>>> {
        self.check_car(i)?;
        Ok(self.slot.arena.car_body(i).map(|b| b.to_row()))
    }

    /// Write a car's body exactly (no rounding of the rotation). A demolished car gets a body back.
    fn set_car_body(&mut self, i: usize, row: Vec<f64>) -> PyResult<()> {
        self.check_car(i)?;
        let b = BodyState::from_row(&row).ok_or_else(|| PyValueError::new_err(format!("need {} numbers", hidden::BODY_FIELDS.len())))?;
        let a = &mut self.slot.arena;
        if a.players[i].car.is_none() {
            let s = RbState { location: v3(b.origin.x * 50.0, b.origin.y * 50.0, b.origin.z * 50.0), ..RbState::default() };
            a.set_car_fresh(i, &s, 0.0);
        }
        a.set_car_body(i, &b);
        Ok(())
    }

    /// The ball's body exactly (Bullet units), or None when there is no ball.
    fn ball_body(&self) -> Option<Vec<f64>> {
        self.slot.arena.ball_body().map(|b| b.to_row())
    }

    /// Write the ball's body exactly.
    fn set_ball_body(&mut self, row: Vec<f64>) -> PyResult<()> {
        let b = BodyState::from_row(&row).ok_or_else(|| PyValueError::new_err(format!("need {} numbers", hidden::BODY_FIELDS.len())))?;
        self.slot.arena.set_ball_body(&b);
        Ok(())
    }

    /// The dodge torque and direction the game gives car `i` for a dodge input (forward, right).
    fn dodge_from_input(&self, i: usize, forward: f32, right: f32) -> PyResult<([f32; 3], [f32; 3])> {
        self.check_car(i)?;
        let (t, d) = self.slot.arena.dodge_from_input(i, forward, right).ok_or_else(|| PyRuntimeError::new_err("that car is demolished"))?;
        Ok(([t.x, t.y, t.z], [d.x, d.y, d.z]))
    }

    /// Demolish car `i` now; it respawns after `respawn_left` seconds.
    fn set_car_demolished(&mut self, i: usize, respawn_left: f32) -> PyResult<()> {
        self.check_car(i)?;
        self.slot.arena.set_car_demolished(i, respawn_left);
        Ok(())
    }

    /// Car `i`'s next respawn goes here: (x, y, z) uu and a rotator (pitch, yaw, roll; 65536 a turn).
    fn queue_respawn(&mut self, i: usize, loc: [f32; 3], rot: [i32; 3]) -> PyResult<()> {
        self.check_car(i)?;
        self.slot.arena.queue_respawn_spot(i, v3(loc[0], loc[1], loc[2]), Rotator { pitch: rot[0], yaw: rot[1], roll: rot[2] });
        Ok(())
    }

    /// The next kickoff puts the cars here, one ((x, y, z) uu, (pitch, yaw, roll)) per car in order.
    fn queue_kickoff(&mut self, spots: Vec<([f32; 3], [i32; 3])>) -> PyResult<()> {
        if spots.len() != self.slot.num_cars {
            return Err(PyValueError::new_err("need one spot per car"));
        }
        let q = spots.iter().map(|(l, r)| (v3(l[0], l[1], l[2]), Rotator { pitch: r[0], yaw: r[1], roll: r[2] })).collect();
        self.slot.arena.queue_kickoff_spots(q);
        Ok(())
    }

    /// The round's hidden state: (physics frame, phase 0 countdown / 1 active / 2 after a goal / 3 ended, ticks or
    /// seconds left in it, countdown clock (remaining, start timer elapsed, update timer elapsed) or None,
    /// ball has been hit, ball held, game time, score).
    #[allow(clippy::type_complexity)]
    fn round_hidden(&self) -> (i32, i32, f32, Option<(i32, f32, f32)>, bool, bool, f32, [i32; 2]) {
        let r = self.slot.arena.round_hidden();
        (r.physics_frame, r.phase, r.phase_left, r.countdown_clock, r.ball_has_been_hit, r.ball_held_2d, r.game_time, r.score)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_round_hidden(&mut self, physics_frame: i32, phase: i32, phase_left: f32, countdown_clock: Option<(i32, f32, f32)>,
                        ball_has_been_hit: bool, ball_held: bool, game_time: f32, score: [i32; 2]) {
        let r = RoundHidden { physics_frame, phase, phase_left, countdown_clock, ball_has_been_hit, ball_held_2d: ball_held, game_time, score };
        self.slot.arena.set_round_hidden(&r);
    }

    /// Use the game's own 3-second kickoff countdown for later kickoffs (with `round_flow`).
    fn use_game_countdown(&mut self, on: bool) {
        self.slot.arena.use_game_countdown(on);
    }

    /// A boost pad: (taken, its timer's length, the timer's elapsed seconds).
    fn pad(&self, i: usize) -> PyResult<(bool, f32, f32)> {
        let p = self.slot.arena.pads.pads.get(i).ok_or_else(|| PyIndexError::new_err(format!("no pad {i}")))?;
        Ok((p.picked_up, p.timer.rate, p.timer.elapsed))
    }

    /// Set a boost pad: taken or not, and its timer (length and elapsed seconds; a taken pad comes back when the
    /// elapsed time passes the length).
    fn set_pad(&mut self, i: usize, taken: bool, rate: f32, elapsed: f32) -> PyResult<()> {
        let p = self.slot.arena.pads.pads.get_mut(i).ok_or_else(|| PyIndexError::new_err(format!("no pad {i}")))?;
        *p = catsim::game::pads::Pad::cooldown(if taken { rate } else { 0.0 });
        if taken {
            p.timer.elapsed = elapsed;
            p.respawn_left = (rate - elapsed).max(0.0);
        }
        Ok(())
    }

    /// The game's boost pad pickups on or off (off: pads only change through `set_pad` and `add_boost`).
    fn set_pad_rule(&mut self, on: bool) {
        self.slot.arena.pad_rule_on = on;
    }

    /// Give car `i` this much boost (0 to 1 is empty to full), as a pickup does.
    fn add_boost(&mut self, i: usize, amount: f32) -> PyResult<()> {
        self.check_car(i)?;
        if let Some(c) = self.slot.arena.players[i].car.as_mut() {
            let v = c.boost.amount + amount;
            c.boost.set_amount(v);
        }
        Ok(())
    }

    /// Car `i` takes input or not (the game turns it off during countdowns and after goals).
    fn set_driving(&mut self, i: usize, on: bool) -> PyResult<()> {
        self.check_car(i)?;
        if let Some(c) = self.slot.arena.players[i].car.as_mut() {
            c.set_driving(on);
        }
        Ok(())
    }

    /// Hold the ball in place (the game does while every car is held at a kickoff), or let it go.
    fn set_ball_held(&mut self, on: bool) {
        self.slot.arena.ball_held_2d = on;
    }

    /// `cars`: (body or preset name, team 0 blue / 1 orange) in the order they join. `round_flow`: run
    /// the game's own goal, explosion and restart flow; off, a goal is only reported. `map`: Stadium_P or
    /// UF_Night_P (the same field; UF_Night_P has its own pad heights and no planes behind the goals).
    #[new]
    #[pyo3(signature = (cars, data_dir, round_flow=false, map="Stadium_P"))]
    fn new(cars: Vec<(String, u8)>, data_dir: PathBuf, round_flow: bool, map: &str) -> PyResult<Self> {
        let pad_layout = layout(map)?;
        let dir = soccar_dir(&data_dir);
        catsim::data::load(&dir).map_err(rt)?;
        let mut list = Vec::new();
        let mut bodies = Vec::new();
        for (name, team) in &cars {
            if *team > 1 {
                return Err(PyValueError::new_err("team must be 0 (blue) or 1 (orange)"));
            }
            let (p, b) = find_preset(name).ok_or_else(|| PyValueError::new_err(format!("no car body or preset named {name}")))?;
            list.push((p, *team));
            bodies.push(b);
        }
        let cfg = RoundConfig { countdown_ticks: 0, post_goal_time: 3.0, match_seconds: None };
        let mut arena = Arena::new_with_layout(&list, &dir, cfg, pad_layout).map_err(rt)?;
        for (i, b) in bodies.iter().enumerate() {
            if let Some(b) = b {
                arena.players[i].body = b;
            }
        }
        let n = list.len();
        let slot = ArenaSlot {
            arena,
            num_cars: n,
            time_since_touch: 0.0,
            episode_time: 0.0,
            applied: vec![VehicleInputs::default(); n],
            delay: vec![0; n],
            queue: vec![Vec::new(); n],
            events: vec![[0.0; adapter::E_DIM]; n],
            goal: -1.0,
        };
        let mut me = NativeArena { slot, dir, ticks: 0, round_flow, predictor: None };
        me.kickoff();
        Ok(me)
    }

    #[getter]
    fn num_cars(&self) -> usize {
        self.slot.num_cars
    }

    #[getter]
    fn tick_count(&self) -> u64 {
        self.ticks
    }

    #[getter]
    fn score(&self) -> [i32; 2] {
        self.slot.arena.clock.score
    }

    /// Advance `ticks` physics ticks (1/120 s each) with these controls, one row per car:
    /// [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]. Returns the team that scored
    /// on the last tick that had a goal, or -1.
    #[pyo3(signature = (controls, ticks=1))]
    fn step(&mut self, controls: Vec<[f32; 8]>, ticks: u32) -> PyResult<i32> {
        let inp = inputs(&controls, self.slot.num_cars)?;
        let mut goal = -1;
        for _ in 0..ticks {
            self.slot.step(&inp);
            self.ticks += 1;
            if self.slot.goal >= 0.0 {
                goal = self.slot.goal as i32;
            }
        }
        Ok(goal)
    }

    /// Put every car at a kickoff spot and the ball in the middle, as at a kickoff.
    fn reset_kickoff(&mut self) {
        self.kickoff();
    }

    fn arena_row(&self) -> Vec<f32> {
        let mut r = vec![0.0; adapter::A_DIM];
        self.slot.write_arena(&mut r);
        r
    }

    fn car_row(&self, i: usize) -> PyResult<Vec<f32>> {
        self.check_car(i)?;
        let mut r = vec![0.0; adapter::C_DIM];
        self.slot.write_car(i, &mut r);
        Ok(r)
    }

    fn hidden_row(&self, i: usize) -> PyResult<Vec<f32>> {
        self.check_car(i)?;
        let mut r = vec![0.0; adapter::H_DIM];
        self.slot.write_hidden(i, &mut r);
        Ok(r)
    }

    /// Per car: [touched ball, demo inflicted, demoed, bump inflicted] on the last tick.
    fn events(&self) -> Vec<[f32; 4]> {
        self.slot.events.iter().map(|e| [e[0], e[1], e[2], e[3]]).collect()
    }

    fn car_team(&self, i: usize) -> PyResult<u8> {
        self.check_car(i)?;
        Ok(self.slot.arena.players[i].team)
    }

    fn car_preset(&self, i: usize) -> PyResult<String> {
        self.check_car(i)?;
        Ok(self.slot.arena.players[i].preset.name.to_string())
    }

    /// Set a car: [x, y, z, vx, vy, vz, wx, wy, wz, yaw, pitch, roll (radians), boost 0..100]. The rotation goes
    /// through the game's integer rotator, as the game's own state setting does; boost is rounded down to a whole number.
    fn set_car(&mut self, i: usize, row: [f32; 13]) -> PyResult<()> {
        self.check_car(i)?;
        if self.slot.arena.players[i].car.is_none() {
            return Err(PyRuntimeError::new_err("that car is demolished and has no body to set"));
        }
        self.slot.set_car_row(i, &row);
        Ok(())
    }

    /// Set the ball: [x, y, z, vx, vy, vz, wx, wy, wz]. `sleep_at_rest`: a ball set with no
    /// velocity goes to sleep at once, as the game's own state setting does.
    #[pyo3(signature = (row, sleep_at_rest=false))]
    fn set_ball(&mut self, row: [f32; 9], sleep_at_rest: bool) {
        self.slot.set_ball_row(&row, sleep_at_rest);
    }

    /// Advance one application frame of `steps` physics ticks (0 or more) with these controls, with the
    /// frame's real duration and render alpha, as the game's frame loop does. For replaying real games.
    fn step_frame(&mut self, controls: Vec<[f32; 8]>, steps: i32, frame_time: f32, render_alpha: f32) -> PyResult<i32> {
        let inp = inputs(&controls, self.slot.num_cars)?;
        if !(0..=8).contains(&steps) || !frame_time.is_finite() || !(0.0..=1.0).contains(&render_alpha) {
            return Err(PyValueError::new_err("steps must be 0..8, frame_time finite, render_alpha in [0, 1]"));
        }
        self.slot.step_frame_with_time(&inp, steps, frame_time, render_alpha);
        self.ticks += steps as u64;
        Ok(self.slot.goal as i32)
    }

    /// Turn on the game's random-number stream and the model of everything outside physics that draws from it,
    /// so kickoff spawn choices follow the real game's. `crt_state`: the C runtime's rand() state; `world_time`:
    /// the game's clock (s). `mesh`: the animated-parts settings for this match (catsim.draw.settings), or None to
    /// leave the parts' timers out. `director_camera`: the countdown's two camera draws. `timers`: other timers
    /// that draw from the stream, as (next due time, period, random extra period) each; each draws once when due
    /// if its random extra period is above 0 (for timers known from a start snapshot).
    #[pyo3(signature = (crt_state, world_time, mesh = None, motion_fx = true, director_camera = true, timers = Vec::new()))]
    fn start_game_rng(&mut self, crt_state: u32, world_time: f32, mesh: Option<std::collections::HashMap<String, Vec<u32>>>,
                      motion_fx: bool, director_camera: bool, timers: Vec<(f32, f32, f32)>) -> PyResult<()> {
        use catsim::game::frame_rng::{FrameRng, LockChain, MeshConfig};
        if self.slot.arena.frame_rng.is_some() {
            return Err(PyRuntimeError::new_err("the random-number model is already running"));
        }
        if !world_time.is_finite() || timers.iter().any(|t| !(t.0.is_finite() && t.1.is_finite() && t.2.is_finite())) {
            return Err(PyValueError::new_err("world_time and timers must be finite"));
        }
        let cfg = match &mesh {
            None => None,
            Some(mesh) => {
                let one = |k: &str| -> PyResult<u32> {
                    mesh.get(k).and_then(|v| v.first().copied()).ok_or_else(|| PyValueError::new_err(format!("mesh settings need {k}")))
                };
                let two = |k: &str| -> PyResult<(u32, u32)> {
                    match mesh.get(k) {
                        Some(v) if v.len() == 2 => Ok((v[0], v[1])),
                        _ => Err(PyValueError::new_err(format!("mesh settings need {k} as two numbers"))),
                    }
                };
                Some(MeshConfig {
                    wheels: one("wheels")?, extras: one("extras")?, map: one("map")?, goal_mesh: one("goal_mesh")? != 0,
                    goal: two("goal")?, demolish: two("demolish")?, reset: two("reset")?, respawn: two("respawn")?,
                })
            }
        };
        let locks = timers.iter().map(|&(release, time, random_time)| LockChain { release, time, random_time, running: true }).collect();
        let a = &mut self.slot.arena;
        a.crt_state = Some(crt_state);
        let mut m = FrameRng::new(world_time, locks, motion_fx);
        m.countdown_entry = director_camera;
        if let Some(cfg) = cfg {
            let cars: Vec<usize> = a.players.iter().enumerate().filter(|(_, p)| p.car.is_some()).map(|(i, _)| i).collect();
            m.start_mesh(cfg, &cars);
        }
        a.frame_rng = Some(m);
        Ok(())
    }

    /// The animated parts' timers in the model's update order: (owner car or -1 for the map, -2 for the goal
    /// explosion; part: 0 body, 1 wheel, 2 other part, 3 goal explosion, 4 map).
    fn draw_parts(&self) -> Vec<(i32, u8)> {
        self.slot.arena.frame_rng.as_ref().map(|m| m.timers.iter().map(|t| (t.owner, t.kind)).collect()).unwrap_or_default()
    }

    /// Draws the model has counted since it started: [frame ends, effects, motion effects, timers, events].
    fn draw_counts(&self) -> Vec<u64> {
        self.slot.arena.frame_rng.as_ref().map(|m| m.draws.to_vec()).unwrap_or_default()
    }

    /// The C runtime's rand() state, when the random-number model is on.
    #[getter]
    fn crt_state(&self) -> Option<u32> {
        self.slot.arena.crt_state
    }

    /// The ball's path: `ticks` ticks ahead from the current ball, one row of
    /// [x, y, z, vx, vy, vz, wx, wy, wz] every `every` ticks. Cars are not in the prediction.
    #[pyo3(signature = (ticks, every=1))]
    fn predict_ball(&mut self, ticks: u32, every: u32) -> PyResult<Vec<[f32; 9]>> {
        if every == 0 {
            return Err(PyValueError::new_err("every must be at least 1"));
        }
        let mut row = vec![0.0; adapter::A_DIM];
        self.slot.write_arena(&mut row);
        if row[0].is_nan() {
            return Ok(Vec::new());
        }
        if self.predictor.is_none() {
            let cfg = RoundConfig { countdown_ticks: 0, post_goal_time: f32::MAX, match_seconds: None };
            let mut a = Arena::new(&[], &self.dir, cfg).map_err(rt)?;
            a.start_at_drop(None);
            self.predictor = Some(a);
        }
        let a = self.predictor.as_mut().unwrap();
        let s = RbState { location: v3(row[0], row[1], row[2]), velocity: v3(row[3], row[4], row[5]), spin: v3(row[6], row[7], row[8]), ..RbState::default() };
        a.phase = Phase::Ended;
        a.ball_held_2d = false;
        a.scripted = true;
        a.explosion = None;
        let ids = a.world.cw.manifolds().to_vec();
        for id in ids {
            a.world.cw.dispatcher.clear_manifold(id);
        }
        a.set_ball_state(&s, false);
        if let Some(b) = a.ball.as_mut() {
            b.first_frame = false;
            b.old_state = s;
        }
        let mut out = Vec::new();
        for t in 1..=ticks {
            a.phase = Phase::Ended;
            a.explosion = None;
            a.frame(&[], 1);
            if t % every == 0 {
                let b = a.ball.as_ref().expect("prediction ball");
                let st = b.read_memo.peek(&a.world.cw.bodies[b.body.0 as usize], &b.limits);
                out.push([st.location.x, st.location.y, st.location.z, st.velocity.x, st.velocity.y, st.velocity.z, st.spin.x, st.spin.y, st.spin.z]);
            }
        }
        Ok(out)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(preset_names, m)?)?;
    m.add_function(wrap_pyfunction!(body_names, m)?)?;
    m.add_function(wrap_pyfunction!(preset_hitbox, m)?)?;
    m.add_function(wrap_pyfunction!(car_fields, m)?)?;
    m.add_function(wrap_pyfunction!(arena_fields, m)?)?;
    m.add_function(wrap_pyfunction!(hidden_fields, m)?)?;
    m.add_function(wrap_pyfunction!(car_hidden_fields, m)?)?;
    m.add_function(wrap_pyfunction!(body_fields, m)?)?;
    m.add_function(wrap_pyfunction!(pad_locations, m)?)?;
    m.add_class::<NativeArena>()?;
    m.add("ENGINE", "catsim")?;
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
