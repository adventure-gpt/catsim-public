//! simbench worker for catSim: the fast engine (engine/core), native, no Python in the loop.
//!
//! Its choices, printed with every result: bodies are written exactly as given (Bullet units when the level has them,
//! else uu x 0.02 and the rotation matrix as is); every hidden field the level gives is set (docs/API.md maps RocketSim
//! v3's fields onto catSim's); contact manifolds' cached points and the broadphase pair order can't be set (catSim makes
//! them at the first tick); pad pickups come from the recording on accuracy runs (the game's own pad rule off) and from
//! catSim's own pad rule on speed runs. Every tick is one frame of the game's loop, as at 120 fps.

use catsim::game::arena::{Arena, Phase, RoundConfig};
use catsim::game::car::VehicleInputs;
use catsim::game::config::{body_spawn, preset, Preset};
use catsim::game::hidden::{BodyState, CarHidden};
use catsim::game::pads::{PadLayout, Pad, NUM_PADS};
use catsim::math::{Mat3, v3};
use serde_json::{Value, json};
use simbench_kit::{BODY, CarSpec, Hello, Sim, TICK_CAR, b, f, has, serve};
use std::path::PathBuf;

mod full;

const WHEEL_FROM_V3: [usize; 4] = [1, 0, 3, 2];

struct Cat {
    dir: PathBuf,
}

struct CatArena {
    a: Arena,
    inputs: Vec<VehicleInputs>,
    timer_hold: bool,
}

/// A car configuration (the full-state suite's `CarConfig`) -> catSim's preset (by hitbox length) and body (by spawn
/// height, 36 uu + the body's mesh offset; the suite rounds spawn_z to 0.01 uu).
fn from_config(cfg: &Value) -> Option<(&'static Preset, Option<&'static catsim::game::config::BodySpawn>)> {
    let x = cfg.get("hitbox_size")?.get("x")?.as_f64()?;
    let len = (x * 100.0).round() / 100.0;
    let mut name = [(120.51, "Octane"), (130.43, "MuscleCar"), (133.99, "LongCar"), (129.52, "Hybrid"), (131.32, "Plank"), (123.22, "Merc")]
        .iter().find(|(l, _)| (l - len).abs() < 0.005).map(|(_, n)| *n)?;
    if cfg.get("three_wheels").is_some_and(|v| v.as_bool().unwrap_or(v.as_f64().unwrap_or(0.0) != 0.0)) {
        name = "Psyclops";
    }
    let mz = cfg.get("spawn_z").and_then(Value::as_f64).map(|z| z as f32 - 36.0);
    let mut best: Option<(&'static catsim::game::config::BodySpawn, f32)> = None;
    if let Some(mz) = mz {
        for b in catsim::game::config::BODIES.iter() {
            if b.preset == name || (name == "Plank" && b.preset == "Darkcar") {
                let d = (mz - f32::from_bits(b.mesh_z)).abs();
                if d < 0.0055 && best.is_none_or(|(_, e)| d < e) {
                    best = Some((b, d));
                }
            }
        }
    }
    match best {
        Some((b, _)) => Some((preset(b.preset)?, Some(b))),
        None => Some((preset(name)?, None)),
    }
}

fn preset_of(c: &CarSpec) -> Result<(&'static Preset, Option<&'static catsim::game::config::BodySpawn>), String> {
    if let Some(found) = c.config.as_ref().and_then(from_config) {
        return Ok(found);
    }
    if let Some(id) = c.body {
        if let Some(b) = body_spawn(&id.to_string()) {
            return Ok((preset(b.preset).ok_or("no preset")?, Some(b)));
        }
    }
    let name = match c.preset.as_str() {
        "Octane" => "Octane", "Dominus" => "MuscleCar", "Plank" => "Darkcar", "Breakout" => "LongCar",
        "Hybrid" => "Hybrid", "Merc" => "Merc", "Psyclops" => "Psyclops", other => other,
    };
    if let Some(b) = body_spawn(name) {
        return Ok((preset(b.preset).ok_or("no preset")?, Some(b)));
    }
    Ok((preset(name).ok_or_else(|| format!("catSim has no car preset {name}"))?, None))
}

fn vec3(v: &Value) -> catsim::math::Vec3 {
    let a = simbench_kit::v3(v);
    v3(a[0], a[1], a[2])
}

fn basis(c: &Value) -> Mat3 {
    let (fw, rt, up) = (vec3(&c["fwd"]), vec3(&c["right"]), vec3(&c["up"]));
    Mat3 { r: [v3(fw.x, rt.x, up.x), v3(fw.y, rt.y, up.y), v3(fw.z, rt.z, up.z)] }
}

fn body_of(c: &Value, default_up: bool) -> BodyState {
    if let Some(r) = c.get("body_bt").and_then(Value::as_array) {
        let r: Vec<f64> = r.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
        if let Some(bs) = BodyState::from_row(&r) {
            return bs;
        }
    }
    let k = 0.02f32;
    let (p, v) = (vec3(&c["pos"]), vec3(&c["vel"]));
    let m = if has(c, "fwd") { basis(c) } else if default_up { Mat3::IDENTITY } else { Mat3::IDENTITY };
    BodyState {
        origin: v3(p.x * k, p.y * k, p.z * k),
        basis: m,
        lin_vel: v3(v.x * k, v.y * k, v.z * k),
        ang_vel: vec3(&c["ang_vel"]),
        asleep: b(c, "asleep"),
        deactivation_time: 0.0,
    }
}

impl Cat {
    fn kickoff(a: &mut Arena, n: usize) {
        // the game's kickoff has five spots a team; a bigger team (the suite's uneven matches) gets spots in a row
        // instead (every car is moved to its start state next anyway)
        let big = (0..2).any(|t| a.players.iter().filter(|p| p.team == t).count() > 5);
        let spots = big.then(|| (0..n).map(|i| {
            let side = if a.players[i].team == 0 { -1.0 } else { 1.0 };
            (v3(-3000.0 + 750.0 * (i / 2) as f32, side * 4000.0, catsim::game::arena::SPAWN_ACTOR_Z), catsim::math::rot::Rotator::default())
        }).collect());
        a.start_at_drop(spots);
        for _ in 0..2 {
            a.frame(&vec![VehicleInputs::default(); n], 1);
        }
        a.phase = Phase::Active;
        a.ball_held_2d = false;
        a.scripted = true;
        a.rule_respawns = true;
        for p in &mut a.players {
            if let Some(c) = p.car.as_mut() {
                c.driving = true;
                c.dodge.c.time = 0.0;
            }
        }
    }

    /// RocketSim v3 / RLPR field meanings -> catSim's (docs/API.md).
    fn hidden(a: &Arena, i: usize, h: &mut CarHidden, c: &Value, tick: i64) {
        let frame = a.physics_frame;
        if has(c, "on_ground") { h.on_ground = b(c, "on_ground"); }
        // the game's own jump flag is set by a jump, a double jump and a dodge alike
        if has(c, "has_jumped") { h.jumped = b(c, "has_jumped") || b(c, "has_double_jumped") || b(c, "has_flipped"); }
        if has(c, "has_double_jumped") || has(c, "has_flipped") { h.double_jumped = b(c, "has_double_jumped") || b(c, "has_flipped"); }
        if has(c, "double_jumped_or_flipped") { h.double_jumped = b(c, "double_jumped_or_flipped") && !b(c, "has_flip"); }
        if has(c, "is_supersonic") { h.supersonic = b(c, "is_supersonic"); }
        if has(c, "supersonic_grace_timer") { h.supersonic_time = f(c, "supersonic_grace_timer", 0.0); }
        if has(c, "is_jumping") {
            h.jump.active = b(c, "is_jumping");
            h.jump.time = if has(c, "jump_time") {
                f(c, "jump_time", 0.0)
            } else if h.jump.active {
                f(c, "jump_ticks", 0.0) / 120.0
            } else {
                f(c, "air_time_since_jump", h.jump.time)
            };
        }
        if has(c, "jump_hold_broken") { h.jump_released = b(c, "jump_hold_broken"); }
        if has(c, "is_flipping") { h.dodge.active = b(c, "is_flipping"); }
        if has(c, "flip_time") { h.dodge.time = f(c, "flip_time", 0.0); }
        if has(c, "flip_rel_torque") {
            let r = simbench_kit::v3(&c["flip_rel_torque"]);
            if r[0] != 0.0 || r[1] != 0.0 {
                // RocketSim's relative flip torque (x side, y forward) -> the game's whole-number torque and direction
                if let Some(car) = a.players[i].car.as_ref() {
                    let d = &car.dodge;
                    let round = catsim::math::rot::app_round;
                    h.dodge_torque = v3(round(r[0] * d.side_torque * d.torque_scale) as f32, round(r[1] * d.forward_torque * d.torque_scale) as f32, 0.0);
                    h.dodge_direction = v3(r[1], -r[0], 0.0);
                }
            }
        }
        if has(c, "is_auto_flipping") {
            h.auto_flip.active = b(c, "is_auto_flipping");
            if h.auto_flip.active {
                let left = f(c, "auto_flip_timer", 0.0);
                h.auto_flip.time = 1.0 / 120.0;
                h.auto_flip_duration = left + 1.0 / 120.0;
                h.auto_flip_right = f(c, "auto_flip_torque_scale", 1.0) > 0.0;
            }
        }
        if has(c, "is_boosting") {
            h.boost.active = b(c, "is_boosting");
            h.boost.time = if h.boost.active { f(c, "boosting_time", 0.0) } else { f(c, "time_since_boosted", 0.0) };
        }
        if has(c, "boost") { h.boost_amount = f(c, "boost", 0.0) / 100.0; }
        if has(c, "boost01") { h.boost_amount = f(c, "boost01", 0.0); }
        if has(c, "handbrake_val") { h.handbrake = f(c, "handbrake_val", 0.0); }
        if let Some(p) = c.get("prev_controls").and_then(Value::as_array) {
            for (k, x) in p.iter().take(8).enumerate() {
                h.held_input[k] = x.as_f64().unwrap_or(0.0) as f32;
            }
        }
        // a contact normal the level doesn't give: straight up (measured best on the full-state suite's B2 starts)
        let up = v3(0.0, 0.0, 1.0);
        if let Some(w) = c.get("wheels_contact").and_then(Value::as_array) {
            for k in 0..4 {
                let src = WHEEL_FROM_V3[k];
                h.wheels[k].contact = w.get(src).is_some_and(|x| x.as_bool().unwrap_or(x.as_f64().unwrap_or(0.0) != 0.0));
                h.wheels[k].normal = c.get("wheel_normals").and_then(|n| n.get(src)).map(vec3).unwrap_or(up);
            }
        } else if has(c, "on_ground") {
            // a packet doesn't show the wheels: guess all four touching when the car is on the ground
            for w in h.wheels.iter_mut() {
                w.contact = h.on_ground;
                w.normal = up;
            }
        }
        if has(c, "world_contact") {
            h.world_contact = b(c, "world_contact");
            h.world_contact_normal = vec3(&c["world_contact_normal"]);
        }
        if has(c, "bump_cooldown_timer") {
            let left = f(c, "bump_cooldown_timer", 0.0);
            let v = c.get("bump_victim").and_then(Value::as_i64).unwrap_or(-1);
            h.bump_victim = if v >= 0 && left > 0.0 { Some(v as usize) } else { None };
            h.bump_time = if left > 0.0 { a.physics_time() - (0.25 - left) } else { -10.0 };
        }
        if has(c, "last_extra_hit_tick") {
            let hit = c["last_extra_hit_tick"].as_i64().unwrap_or(-1);
            h.ball_impact_frame = if hit >= 0 { (frame as i64 - (tick - hit)) as i32 } else { -100000 };
        }
    }
}

impl Sim for Cat {
    type Arena = CatArena;

    fn hello(&self) -> Hello {
        Hello {
            name: "catsim".into(),
            version: format!("catSim {} (this repository's engine)", engine_version()),
            about: "catSim's fast engine, native Rust, release build at catSim's own CPU level (x86-64-v2)".into(),
            device: "cpu",
            units: "bt",
            caps: json!({"levels": ["packet", "v3", "rlpr", "full"], "frame_timing": true, "game_flow": false, "rng_stream": false, "spot_choices": false, "threads": true}),
            choices: vec![
                "bodies written exactly as given (Bullet units when the level has them, else uu x 0.02 and the rotation matrix as is)".into(),
                "every hidden field the level gives: jump, flip, boost and supersonic clocks, the dodge torque from flip_rel_torque, wheel contacts, held controls, bump and ball-hit gates, demolitions".into(),
                "can't set: contact manifolds' cached points, the broadphase pair order (made at the first tick)".into(),
                "accuracy runs: pad pickups from the recording (catSim's pad rule off); speed runs: catSim's own pad rule".into(),
                "each tick is one frame of the game's loop".into(),
            ],
        }
    }

    fn setup(&mut self, env: &Value) -> Result<(), String> {
        let d = env.get("catsim_data").and_then(Value::as_str).ok_or("needs the catsim_data folder (catsim-dump)")?;
        let d = PathBuf::from(d);
        self.dir = if d.join("soccar").is_dir() { d.join("soccar") } else { d };
        catsim::data::load(&self.dir).map_err(|e| e.to_string())?;
        catsim::bt::bvh::soccar_meshes(&self.dir).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn new_arena(&self, cars: &[CarSpec], map: &str) -> Result<CatArena, String> {
        let layout = PadLayout::from_map(map).ok_or_else(|| format!("catSim knows Stadium_P and UF_Night_P, not {map}"))?;
        let mut list = Vec::new();
        let mut bodies = Vec::new();
        for c in cars {
            let (p, b) = preset_of(c)?;
            list.push((p, c.team));
            bodies.push(b);
        }
        let cfg = RoundConfig { countdown_ticks: 0, post_goal_time: 3.0, match_seconds: None };
        let mut a = Arena::new_with_layout(&list, &self.dir, cfg, layout).map_err(|e| e.to_string())?;
        for (i, b) in bodies.iter().enumerate() {
            if let Some(b) = b {
                a.players[i].body = b;
            }
        }
        Cat::kickoff(&mut a, cars.len());
        Ok(CatArena { a, inputs: vec![VehicleInputs::default(); cars.len()], timer_hold: false })
    }

    fn set_state(&self, ca: &mut CatArena, st: &Value) -> Result<(), String> {
        ca.timer_hold = b(st, "timer_hold");
        let a = &mut ca.a;
        if b(st, "no_goals") {
            // RLPR scores physics through the goal volumes (RocketSim doesn't remove its ball there): no goal flow
            a.phase = Phase::Ended;
            a.explosion = None;
            a.ball_held_2d = false;
            if a.ball.is_none() {
                a.reset_ball();
            }
            a.pad_rule_on = false;
        }
        if b(st, "pads_from_recording") {
            a.pad_rule_on = false;
        }
        let tick = st.get("tick").and_then(Value::as_i64).unwrap_or(0);
        let cars = st["cars"].as_array().ok_or("state needs cars")?;
        let next = st.get("next_controls").and_then(Value::as_array);
        let cars: Vec<Value> = cars.iter().map(simbench_kit::v3_from_packet).collect();
        for (i, c) in cars.iter().enumerate() {
            if b(c, "is_demoed") {
                a.set_car_demolished(i, f(c, "demo_respawn_timer", 3.0));
                continue;
            }
            let body = body_of(c, true);
            if a.players[i].car.is_none() {
                let s = catsim::game::rb_state::RbState { location: v3(body.origin.x * 50.0, body.origin.y * 50.0, body.origin.z * 50.0), ..Default::default() };
                a.set_car_fresh(i, &s, 0.0);
            }
            if let Some(car) = a.players[i].car.as_mut() {
                car.driving = true;
                car.game_active = true;
            }
            a.set_car_body(i, &body);
            if b(st, "no_goals") && !has(c, "body_bt") {
                // an RLPR record holds the game's own read of the body (uu): keep it as the car's last read, as the game has it
                if let Some(car) = a.players[i].car.as_mut() {
                    let s = catsim::game::rb_state::RbState { quat: catsim::game::rb_state::matrix_to_quat(&body.basis), location: vec3(&c["pos"]),
                                                              velocity: vec3(&c["vel"]), spin: vec3(&c["ang_vel"]), sleeping: false };
                    car.state = s;
                    car.old_state = s;
                }
            }
            let mut h = a.car_hidden(i).ok_or("car has no body")?;
            Cat::hidden(a, i, &mut h, c, tick);
            if !has(c, "prev_controls") {
                // not given: guess the car was already holding the controls of the first tick (no press or release)
                if let Some(nx) = next.and_then(|n| n.get(i)).and_then(Value::as_array) {
                    for (k, x) in nx.iter().take(8).enumerate() {
                        h.held_input[k] = x.as_f64().unwrap_or(0.0) as f32;
                    }
                }
            }
            a.set_car_hidden(i, &h);
        }
        let mut ball = body_of(&st["ball"], true);
        // RocketSim v3's `asleep` is set for the kickoff ball too, which the game keeps awake (resting on its spot)
        let bp = vec3(&st["ball"]["pos"]);
        if ball.asleep && bp.x.abs() < 1e-3 && bp.y.abs() < 1e-3 && (bp.z - 92.75).abs() < 0.01 {
            ball.asleep = false;
            ball.deactivation_time = 1.0;
        }
        a.set_ball_body(&ball);
        if let Some(pads) = st.get("pads").and_then(Value::as_array) {
            for p in pads {
                let p = simbench_kit::v3(p);
                let k = (0..NUM_PADS).min_by(|&x, &y| {
                    let d = |i: usize| { let q = a.pads.layout.definition(i).loc; (q.x - p[0]).powi(2) + (q.y - p[1]).powi(2) };
                    d(x).total_cmp(&d(y))
                }).unwrap();
                a.pads.pads[k] = Pad::cooldown(p[2].max(0.0));
            }
        }
        Ok(())
    }

    fn set_full_state(&self, ca: &mut CatArena, req: &Value, data: &[u8]) -> Result<(), String> {
        let f = full::Full::new(&req["start"]["full"], data);
        let nc = ca.a.players.len();
        if let Some(order) = f.handles(nc) {
            // remake the bodies so the ball and the cars hold the state's broadphase handles, through the game's own restart
            let a = &mut ca.a;
            a.set_next_handles(&order);
            a.start_like_restart(None, 0);
            a.phase = Phase::Active;
            a.countdown_clock = None;
            a.ball_held_2d = false;
            a.scripted = true;
            for p in a.players.iter_mut() {
                if let Some(c) = p.car.as_mut() {
                    c.set_driving(true);
                }
            }
        }
        ca.timer_hold = b(&req["start"], "timer_hold");
        ca.a.pad_rule_on = false;
        let dt0 = req.get("dt0").and_then(Value::as_f64).map(|x| x as f32).filter(|&x| x > 0.0).unwrap_or(1.0 / 120.0);
        f.apply(&mut ca.a, req, dt0)?;
        if let Some(pads) = req["start"].get("pads").and_then(Value::as_array) {
            let a = &mut ca.a;
            for p in pads {
                let p = simbench_kit::v3(p);
                let k = (0..NUM_PADS).min_by(|&x, &y| {
                    let d = |i: usize| { let q = a.pads.layout.definition(i).loc; (q.x - p[0]).powi(2) + (q.y - p[1]).powi(2) };
                    d(x).total_cmp(&d(y))
                }).unwrap();
                a.pads.pads[k] = Pad::cooldown(p[2].max(0.0));
            }
        }
        Ok(())
    }

    fn step(&self, ca: &mut CatArena, inp: &[[f32; TICK_CAR]], ticks: u32, _frame: Option<[f32; 2]>) {
        Cat::controls(ca, inp);
        for _ in 0..ticks {
            ca.a.frame(&ca.inputs, 1);
        }
        Cat::credit(ca, inp);
    }

    /// A frame of several ticks: the bodies after each tick are read from exact copies run that far (catSim's arena
    /// clones bit for bit), then the frame itself runs once, as the game's loop does.
    fn step_frame(&self, ca: &mut CatArena, ticks: &[Vec<[f32; TICK_CAR]>], frame_time: f32, alpha: f32, reads: &mut [Vec<[f32; BODY]>]) {
        Cat::controls(ca, &ticks[0]);
        let n = ticks.len() as i32;
        let alpha = alpha.clamp(0.0, 1.0);
        for j in 1..n {
            let mut c = ca.a.clone();
            c.frame_with_time(&ca.inputs, j, frame_time, alpha);
            Cat::read(&c, &mut reads[j as usize - 1]);
        }
        ca.a.frame_with_time(&ca.inputs, n, frame_time, alpha);
        Cat::read(&ca.a, &mut reads[n as usize - 1]);
        for rows in ticks {
            Cat::credit(ca, rows);
        }
    }

    fn bodies(&self, ca: &CatArena, out: &mut [[f32; BODY]]) {
        Cat::read(&ca.a, out);
    }
}

impl Cat {
    fn controls(ca: &mut CatArena, inp: &[[f32; TICK_CAR]]) {
        let held = !ca.timer_hold;
        let a = &mut ca.a;
        let mut anyone = false;
        for (i, r) in inp.iter().enumerate() {
            ca.inputs[i] = catsim::adapter::input_from_row(&r[..8]);
            let drive = r[8] >= 0.5;
            anyone |= drive;
            if held {
                if let Some(c) = a.players[i].car.as_mut() {
                    c.set_driving(drive);
                }
            }
        }
        if held {
            a.ball_held_2d = !inp.is_empty() && !anyone;
        }
    }

    /// The game's pad credits: a small pad gives 0.12 (the float the game adds), a big one fills the car.
    fn credit(ca: &mut CatArena, inp: &[[f32; TICK_CAR]]) {
        for (i, r) in inp.iter().enumerate() {
            let g = r[9];
            if g <= 0.0 {
                continue;
            }
            let Some(c) = ca.a.players[i].car.as_mut() else { continue };
            let m = (g / 0.12).round();
            if (1.0..=4.0).contains(&m) && (g - 0.12 * m).abs() < 1e-3 {
                for _ in 0..m as i32 {
                    let v = c.boost.amount + f32::from_bits(0x3df5_c28f);
                    c.boost.set_amount(v);
                }
            } else {
                c.boost.set_amount(1.0);
            }
        }
    }

    fn read(a: &Arena, out: &mut [[f32; BODY]]) {
        let n = a.players.len();
        let put = |o: &mut [f32; BODY], s: &BodyState| {
            let m = &s.basis.r;
            *o = [s.origin.x, s.origin.y, s.origin.z, s.lin_vel.x, s.lin_vel.y, s.lin_vel.z, s.ang_vel.x, s.ang_vel.y, s.ang_vel.z,
                  m[0].x, m[1].x, m[2].x, m[0].y, m[1].y, m[2].y, m[0].z, m[1].z, m[2].z];
        };
        for i in 0..n {
            match a.car_body(i) {
                Some(s) => put(&mut out[i], &s),
                None => out[i] = [0.0; BODY],
            }
        }
        match a.ball_body() {
            Some(s) => put(&mut out[n], &s),
            None => out[n] = [0.0; BODY],
        }
    }
}

/// catSim's version, from the engine workspace this worker is built against.
fn engine_version() -> &'static str {
    include_str!("../../../../engine/Cargo.toml").lines().find_map(|l| l.strip_prefix("version = \"")).map(|v| v.trim_end_matches('"')).unwrap_or("?")
}

fn main() {
    serve(Cat { dir: PathBuf::new() });
}
