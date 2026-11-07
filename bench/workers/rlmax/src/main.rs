//! simbench worker for RLMAX (github.com/akaAMAZING/RLMAX, MIT), over its crate (pinned in Cargo.toml).
//!
//! Speed (GPU): its own training path, `GpuBatch::decide`: every arena on the card, each decision one action-table row
//! per car (RLGym's 90-row table, the same rows the harness's controls come from), 8 ticks, then `extract` of every
//! arena's state into a device buffer; start states from `cpu::Arena::from_physical`, written to the card before each
//! repeat. `decide` resets an arena whose ball goes in (its training path does; the 8-second episodes rarely score).
//! Accuracy: one `cpu::Arena` (RLMAX's CPU build of the same physics) from `from_physical` with every CarState field
//! the level gives (flags, flip torque, jump ticks, clocks, bump and extra-hit gates); the contact manifolds, broadphase
//! pairs and pad timers start empty, as its docs say. Needs a CUDA toolkit and RLMAX_CUDA_PATH to build.

use rlmax::cpu::{self, flags, Arena, ArenaData, BallState, BodyState, CarState, Controls};
use rlmax::driver::{Buffer, Device};
use rlmax::gpu::{GpuArenaData, GpuBatch};
use serde_json::{Value, json};
use simbench_kit::{BODY, CarSpec, Hello, Sim, TICK_CAR, TICK_FRAME, b, f, has, serve, v3_from_packet};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

const RLMAX_REV: &str = "df9b37b73b";

struct Rlmax {
    data: Option<ArenaData>,
    gpu: Mutex<Option<(Device, GpuArenaData)>>,
}

struct RArena {
    a: Arena,
    presets: Vec<u32>,
    teams: Vec<u32>,
    pads_from_recording: bool,
    started: bool,                 // set from a whole state once (a later partial state builds on it)
}

fn preset(name: &str) -> u32 {
    match name { "Dominus" => 1, "Plank" => 2, "Breakout" => 3, "Hybrid" => 4, "Merc" => 5, "Psyclops" => 7, _ => 0 }
}

fn arr(v: &Value) -> [f32; 3] {
    simbench_kit::v3(v)
}

/// uu that RLMAX's set-state (/ 50) turns into exactly these Bullet words, where such a float exists.
fn uu_of_bt(w: f32) -> f32 {
    let u = w * 50.0;
    if u / 50.0 == w {
        return u;
    }
    let (mut up, mut dn) = (u, u);
    for _ in 0..4 {
        up = up.next_up();
        dn = dn.next_down();
        if up / 50.0 == w {
            return up;
        }
        if dn / 50.0 == w {
            return dn;
        }
    }
    u
}

fn body(c: &Value) -> BodyState {
    let (fw, rt, up) = if has(c, "fwd") { (arr(&c["fwd"]), arr(&c["right"]), arr(&c["up"])) } else { ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) };
    let mut s = BodyState { pos: arr(&c["pos"]), rot: [fw[0], fw[1], fw[2], rt[0], rt[1], rt[2], up[0], up[1], up[2]], vel: arr(&c["vel"]), ang_vel: arr(&c["ang_vel"]) };
    // the game's own Bullet words (a float-precision game): position and velocity through RLMAX's own / 50, exactly
    if let Some(w) = c.get("body_bt").and_then(Value::as_array) {
        let g = |k: usize| w[k].as_f64().unwrap_or(0.0) as f32;
        s.pos = [uu_of_bt(g(0)), uu_of_bt(g(1)), uu_of_bt(g(2))];
        s.vel = [uu_of_bt(g(12)), uu_of_bt(g(13)), uu_of_bt(g(14))];
    }
    s
}

/// The basis of a `body_bt` (origin, basis by rows, ...) for RLMAX's exact basis setter, which takes it by columns
/// (forward, right, up, as its bodies hold the rotation).
fn basis_bt(c: &Value) -> Option<[f32; 9]> {
    let w = c.get("body_bt").and_then(Value::as_array)?;
    let mut m = [0f32; 9];
    for r in 0..3 {
        for k in 0..3 {
            m[k * 3 + r] = w[3 + r * 3 + k].as_f64().unwrap_or(0.0) as f32;
        }
    }
    Some(m)
}

/// A state that only moves bodies or leaves cars out (the harness's flow mid-game: a respawn, a float-precision
/// sync): everything else stays as the arena has it.
fn partial(st: &Value) -> bool {
    st["cars"].as_array().is_some_and(|cs| cs.iter().any(|c| c.is_null() || !has(c, "boost")))
}

fn car_state(c: &Value, team: u32, preset: u32, next: Option<&Value>) -> CarState {
    let c = v3_from_packet(c);
    let c = &c;
    let mut s = CarState { body: body(c), team, preset, boost: f(c, "boost", 33.3), ..CarState::default() };
    let mut fl = 0u32;
    for (k, bit) in [("has_jumped", flags::HAS_JUMPED), ("has_double_jumped", flags::HAS_DOUBLE_JUMPED), ("has_flipped", flags::HAS_FLIPPED),
                     ("is_flipping", flags::IS_FLIPPING), ("is_jumping", flags::IS_JUMPING), ("jump_hold_broken", flags::JUMP_HOLD_BROKEN),
                     ("is_auto_flipping", flags::IS_AUTO_FLIPPING), ("is_supersonic", flags::IS_SUPERSONIC), ("is_boosting", flags::IS_BOOSTING),
                     ("is_demoed", flags::IS_DEMOED), ("on_ground", flags::ON_GROUND)] {
        if b(c, k) { fl |= bit; }
    }
    let wheels = c.get("wheels_contact").and_then(Value::as_array);
    for w in 0..4 {
        let on = match wheels { Some(a) => a.get(w).is_some_and(|x| x.as_bool().unwrap_or(false)), None => b(c, "on_ground") };
        if on { fl |= flags::WHEEL0 << w; }
    }
    let held = c.get("prev_controls").or(next);
    if let Some(p) = held.and_then(Value::as_array) {
        let g = |i: usize| p.get(i).and_then(Value::as_f64).unwrap_or(0.0) as f32;
        s.controls = Controls { throttle: g(0), steer: g(1), pitch: g(2), yaw: g(3), roll: g(4), jump: (g(5) > 0.5) as u32, boost: (g(6) > 0.5) as u32, handbrake: (g(7) > 0.5) as u32 };
        if g(5) > 0.5 { fl |= flags::PREV_JUMP; }
    }
    let r = arr(&c["flip_rel_torque"]);
    s.flip_torque = [r[0], r[1]];
    s.jump_ticks = c.get("jump_ticks").and_then(Value::as_f64).unwrap_or(0.0) as u32;
    s.flip_time = f(c, "flip_time", 0.0);
    s.air_time = f(c, "air_time", 0.0);
    s.air_time_since_jump = f(c, "air_time_since_jump", 0.0);
    s.handbrake_val = f(c, "handbrake_val", 0.0);
    s.supersonic_timer = f(c, "supersonic_grace_timer", 0.0);
    s.time_since_boosted = f(c, "time_since_boosted", 0.0);
    s.boosting_time = f(c, "boosting_time", 0.0);
    s.auto_flip_timer = f(c, "auto_flip_timer", 0.0);
    s.auto_flip_torque_scale = f(c, "auto_flip_torque_scale", 0.0);
    s.bump_cooldown = f(c, "bump_cooldown_timer", 0.0);
    if let Some(v) = c.get("bump_victim").and_then(Value::as_i64).filter(|&v| v >= 0) {
        s.bump_victim = v as u32;
        fl |= flags::HAS_BUMP_VICTIM;
    }
    if let Some(t) = c.get("last_extra_hit_tick").and_then(Value::as_i64).filter(|&t| t >= 0) {
        s.last_extra_hit_tick = t as u32;
        fl |= flags::HAS_EXTRA_HIT;
    }
    s.demo_respawn_timer = f(c, "demo_respawn_timer", 0.0);
    s.flags = fl;
    s
}

fn ball_state(bl: &Value) -> BallState {
    BallState { body: body(bl), sleeping: if b(bl, "asleep") { 1 } else { 0 }, deactivation_time: 0.0 }
}

impl Rlmax {
    fn data(&self) -> &ArenaData {
        self.data.as_ref().expect("setup first")
    }
}

impl Sim for Rlmax {
    type Arena = RArena;

    fn hello(&self) -> Hello {
        Hello {
            name: "rlmax".into(),
            version: format!("RLMAX {RLMAX_REV} (git, pinned in Cargo.toml)"),
            about: "RLMAX over its crate: GpuBatch::decide on the card for speed, its CPU build for accuracy".into(),
            device: "gpu",
            units: "bt",
            caps: json!({"levels": ["packet", "v3"], "frame_timing": false, "game_flow": false, "rng_stream": false, "spot_choices": false, "threads": true, "actions": true, "tick_frames": true}),
            choices: vec![
                "speed: GpuBatch::decide (one action-table row per car, 8 ticks) then extract into a device buffer; arenas whose ball goes in reset (its training path)".into(),
                "accuracy: its CPU build, cpu::Arena::from_physical with every CarState field the level gives; manifolds, broadphase pairs and pad timers start empty".into(),
                "its full-state loader (the suite's level A) is a test-only feature of its crate, not used here".into(),
                "whole games: a start (every car given) through from_physical; a respawn or a float-precision sync through set_car / set_ball onto the arena's own state, the game's basis words through set_body_basis; each tick's render fraction through set_pad_frame_fraction".into(),
            ],
        }
    }

    fn setup(&mut self, env: &Value) -> Result<(), String> {
        let meshes = env.get("collision_meshes").and_then(Value::as_str).ok_or("needs RocketSim's collision_meshes folder")?;
        self.data = Some(ArenaData::soccar(&PathBuf::from(meshes).join("soccar"))?);
        Ok(())
    }

    fn new_arena(&self, cars: &[CarSpec], _map: &str) -> Result<RArena, String> {
        Ok(RArena { a: Arena::new(cars.len() as u32), presets: cars.iter().map(|c| preset(&c.preset)).collect(),
                    teams: cars.iter().map(|c| c.team as u32).collect(), pads_from_recording: false, started: false })
    }

    fn set_state(&self, ra: &mut RArena, st: &Value) -> Result<(), String> {
        ra.pads_from_recording = b(st, "pads_from_recording");
        let cars = st["cars"].as_array().ok_or("state needs cars")?;
        let next = st.get("next_controls").and_then(Value::as_array);
        if ra.started && partial(st) {
            // set what is given, onto the arena's own state (contacts, broadphase and pad timers kept)
            for (i, c) in cars.iter().enumerate() {
                if c.is_null() {
                    continue;
                }
                let mut s = ra.a.car(i as u32);
                if has(c, "boost") {
                    s = car_state(c, ra.teams[i], ra.presets[i], next.and_then(|n| n.get(i)));
                } else {
                    s.body = body(c);
                }
                ra.a.set_car(i as u32, &s);
                if let Some(m) = basis_bt(c) {
                    ra.a.set_body_basis(i as u32, &m, None);
                }
            }
            let bl = &st["ball"];
            if bl.is_null() {
                // a ball given as null stays as it is
                return Ok(());
            }
            let cur = ra.a.ball();
            let mut bs = ball_state(bl);
            if !has(bl, "asleep") {
                // not given: the ball sleeps or not as the arena has it
                bs.sleeping = cur.sleeping;
                bs.deactivation_time = cur.deactivation_time;
            }
            ra.a.set_ball(&bs);
            if let Some(m) = basis_bt(bl) {
                ra.a.set_body_basis(cars.len() as u32, &m, None);
            }
            return Ok(());
        }
        let cs: Vec<CarState> = cars.iter().enumerate().map(|(i, c)| car_state(c, ra.teams[i], ra.presets[i], next.and_then(|n| n.get(i)))).collect();
        ra.a = Arena::from_physical(self.data(), &ball_state(&st["ball"]), &cs);
        ra.started = true;
        Ok(())
    }

    fn step(&self, ra: &mut RArena, inp: &[[f32; TICK_CAR]], ticks: u32, _frame: Option<[f32; 2]>) {
        for (i, r) in inp.iter().enumerate() {
            let c = if r[8] < 0.5 { Controls { steer: r[1], ..Controls::default() } } else {
                Controls { throttle: r[0], steer: r[1], pitch: r[2], yaw: r[3], roll: r[4], jump: (r[5] > 0.5) as u32, boost: (r[6] > 0.5) as u32, handbrake: (r[7] > 0.5) as u32 }
            };
            ra.a.set_controls(i as u32, &c);
        }
        ra.a.step(self.data(), ticks);
        if ra.pads_from_recording {
            for (i, r) in inp.iter().enumerate() {
                if r[9] > 0.0 {
                    let now = ra.a.car(i as u32).boost;
                    let m = (r[9] / 0.12).round();
                    let nb = if (1.0..=4.0).contains(&m) && (r[9] - 0.12 * m).abs() < 1e-3 { (now + 12.0 * m).min(100.0) } else { 100.0 };
                    ra.a.set_boost(i as u32, nb);
                }
            }
        }
    }

    fn step_tick(&self, ra: &mut RArena, inp: &[[f32; TICK_CAR]], row: &[f32; TICK_FRAME]) {
        // the game tests boost pads once per rendered frame, at its render fraction (negative: an earlier tick of it)
        if row[3].is_finite() {
            ra.a.set_pad_frame_fraction(row[3].min(1.0));
        }
        self.step(ra, inp, 1, None);
    }

    fn bodies(&self, ra: &RArena, out: &mut [[f32; BODY]]) {
        let n = ra.a.num_cars;
        for i in 0..=n {
            let (rot, ang, demoed) = if i < n {
                let c = ra.a.car(i);
                (c.body.rot, c.body.ang_vel, c.flags & flags::IS_DEMOED != 0)
            } else {
                let bl = ra.a.ball();
                (bl.body.rot, bl.body.ang_vel, false)
            };
            if demoed {
                out[i as usize] = [0.0; BODY];
                continue;
            }
            let bt = ra.a.body_bt(i);
            out[i as usize] = [bt[0], bt[1], bt[2], bt[3], bt[4], bt[5], ang[0], ang[1], ang[2],
                               rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8]];
        }
    }

    fn speed_batched(&self, job: &Value, starts: &[Value], _ctl: &[f32], actions: &[u8]) -> Option<Result<Value, String>> {
        Some((|| {
            let mut g = self.gpu.lock().unwrap();
            if g.is_none() {
                let dev = Device::open(0)?;
                let gd = GpuArenaData::upload(&dev, self.data()).map_err(|e| e.to_string())?;
                *g = Some((dev, gd));
            }
            let (dev, gd) = g.as_ref().unwrap();
            let cars = simbench_kit::cars_of(&job["cars"]);
            let nc = cars.len() as u32;
            let n_ar = job["arenas"].as_u64().unwrap() as usize;
            let n_dec = job["decisions"].as_u64().unwrap() as usize;
            let k = job["decision_ticks"].as_u64().unwrap() as u32;
            let reps = (job["warmup"].as_u64().unwrap() + job["repeats"].as_u64().unwrap()) as usize;
            let warm = job["warmup"].as_u64().unwrap() as usize;
            let presets: Vec<u32> = cars.iter().map(|c| preset(&c.preset)).collect();
            let mut words = Vec::new();
            for st in starts.iter().take(n_ar) {
                let cs: Vec<CarState> = st["cars"].as_array().unwrap().iter().enumerate()
                    .map(|(i, c)| car_state(c, cars[i].team as u32, presets[i], None)).collect();
                words.extend_from_slice(Arena::from_physical(self.data(), &ball_state(&st["ball"]), &cs).words());
            }
            let mut batch = GpuBatch::new(dev, nc, &words).map_err(|e| e.to_string())?;
            let table = Buffer::from_slice(dev, &simbench_kit::action_table()).map_err(|e| e.to_string())?;
            let per = n_ar * nc as usize;
            let rows = Buffer::from_slice(dev, &actions[..per]).map_err(|e| e.to_string())?;
            let goals = Buffer::from_slice(dev, &vec![0u32; n_ar]).map_err(|e| e.to_string())?;
            let ex = Buffer::<f32>::zeroed(dev, n_ar * cpu::extract_row_floats(nc) as usize).map_err(|e| e.to_string())?;
            // decisions arena-major per decision: actions[d][arena][car]
            let mut seconds = Vec::new();
            for r in 0..reps {
                batch.write(&words).map_err(|e| e.to_string())?;
                dev.synchronize().map_err(|e| e.to_string())?;
                let t0 = Instant::now();
                for d in 0..n_dec {
                    rows.write(&actions[d * per..(d + 1) * per]).map_err(|e| e.to_string())?;
                    batch.decide(gd, k, &rows, &table, &goals).map_err(|e| e.to_string())?;
                    batch.extract(&ex).map_err(|e| e.to_string())?;
                }
                dev.synchronize().map_err(|e| e.to_string())?;
                if r >= warm {
                    seconds.push(t0.elapsed().as_secs_f64());
                }
            }
            Ok(json!({"seconds": seconds, "threads": 1}))
        })())
    }
}

fn main() {
    serve(Rlmax { data: None, gpu: Mutex::new(None) });
}
