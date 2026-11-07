//! simbench worker for RocketSim v3: the `rocketsim` crate (crates.io, pinned in Cargo.toml), native.
//!
//! Its choices, printed with every result: every field of v3's `CarState` the level gives is set through
//! `edit_car_state_full` (the previous controls and the contact flags through the state's own setters); the ball through
//! `edit_ball_state`, put to sleep with `sleep_ball` when asleep; pads by nearest location. A car held by the game (may
//! not drive) gets no controls. Pad pickups on accuracy runs come from the recording (its own pickups off); on speed runs from
//! v3's own pad rule. Release build with v3's own profile (lto, one codegen unit, abort on panic), default arena
//! settings (`ArenaMemWeightMode::Balanced`).

use glam::{Mat3A, Vec3A};
use rocketsim::{Arena, ArenaConfig, CarBodyConfig, CarControls, GameMode, Team};
use serde_json::{Value, json};
use simbench_kit::{BODY, CarSpec, Hello, Sim, TICK_CAR, b, f, has, serve};

struct V3;

struct V3Arena {
    a: Arena,
    ids: Vec<usize>,
    pads_from_recording: bool,
}

fn vec(v: &Value) -> Vec3A {
    Vec3A::from_array(simbench_kit::v3(v))
}

fn rot(c: &Value) -> Mat3A {
    if has(c, "fwd") {
        Mat3A::from_cols(vec(&c["fwd"]), vec(&c["right"]), vec(&c["up"]))
    } else {
        Mat3A::IDENTITY
    }
}

fn controls(v: &[f32]) -> CarControls {
    CarControls { throttle: v[0], steer: v[1], pitch: v[2], yaw: v[3], roll: v[4], jump: v[5] > 0.5, boost: v[6] > 0.5, handbrake: v[7] > 0.5 }
}

fn preset(name: &str) -> Result<CarBodyConfig, String> {
    Ok(match name {
        "Octane" => CarBodyConfig::OCTANE,
        "Dominus" => CarBodyConfig::DOMINUS,
        "Plank" => CarBodyConfig::PLANK,
        "Breakout" => CarBodyConfig::BREAKOUT,
        "Hybrid" => CarBodyConfig::HYBRID,
        "Merc" => CarBodyConfig::MERC,
        "Psyclops" => CarBodyConfig::PSYCLOPS,
        other => return Err(format!("RocketSim v3 has no preset {other}")),
    })
}

/// pos, vel in uu: from the level's uu fields, or a Bullet-units body (x 50).
fn phys(c: &Value) -> (Vec3A, Mat3A, Vec3A, Vec3A) {
    if !has(c, "pos") {
        if let Some(r) = c.get("body_bt").and_then(Value::as_array) {
            let g = |k: usize| r[k].as_f64().unwrap_or(0.0) as f32;
            let m = Mat3A::from_cols(Vec3A::new(g(3), g(6), g(9)), Vec3A::new(g(4), g(7), g(10)), Vec3A::new(g(5), g(8), g(11)));
            return (Vec3A::new(g(0), g(1), g(2)) * 50.0, m, Vec3A::new(g(12), g(13), g(14)) * 50.0, Vec3A::new(g(15), g(16), g(17)));
        }
    }
    (vec(&c["pos"]), rot(c), vec(&c["vel"]), vec(&c["ang_vel"]))
}

impl Sim for V3 {
    type Arena = V3Arena;

    fn hello(&self) -> Hello {
        Hello {
            name: "rocketsim-v3".into(),
            version: "rocketsim 0.3.5 (crates.io)".into(),
            about: "RocketSim v3, native Rust, release build with its own profile, default arena settings".into(),
            device: "cpu",
            units: "uu",
            caps: json!({"levels": ["packet", "v3"], "frame_timing": false, "game_flow": false, "rng_stream": false, "spot_choices": false, "threads": true}),
            choices: vec![
                "sets every v3 CarState field the level gives (edit_car_state_full; previous controls and contact flags through the state's own setters)".into(),
                "ball through edit_ball_state, sleep_ball when asleep; pads by nearest location".into(),
                "takes no frame timing (v3 steps fixed 1/120 s ticks) and no contact manifolds".into(),
                "accuracy runs: pad pickups from the recording; speed runs: v3's own pads".into(),
                "release build with v3's profile (lto, codegen-units 1, panic abort), ArenaMemWeightMode::Balanced (its default)".into(),
            ],
        }
    }

    fn setup(&mut self, env: &Value) -> Result<(), String> {
        let d = env.get("collision_meshes").and_then(Value::as_str).ok_or("needs RocketSim's collision_meshes folder")?;
        rocketsim::init(d, true).map_err(|e| format!("rocketsim::init({d}): {e}"))
    }

    fn new_arena(&self, cars: &[CarSpec], _map: &str) -> Result<V3Arena, String> {
        let mut a = Arena::new_with_config(ArenaConfig::new(GameMode::Soccar));
        let mut ids = Vec::new();
        for c in cars {
            let team = if c.team == 0 { Team::Blue } else { Team::Orange };
            ids.push(a.add_car(team, preset(&c.preset)?));
        }
        Ok(V3Arena { a, ids, pads_from_recording: false })
    }

    fn set_state(&self, va: &mut V3Arena, st: &Value) -> Result<(), String> {
        va.pads_from_recording = b(st, "pads_from_recording");
        let cars = st["cars"].as_array().ok_or("state needs cars")?;
        let next = st.get("next_controls").and_then(Value::as_array);
        let cars: Vec<Value> = cars.iter().map(simbench_kit::v3_from_packet).collect();
        for (k, c) in cars.iter().enumerate() {
            if c.is_null() {
                continue;
            }
            let id = va.ids[k];
            // not given: guess the car was already holding the controls of the first tick (no press or release)
            let guess: Option<Vec<f32>> = if has(c, "prev_controls") { None } else {
                next.and_then(|n| n.get(k)).and_then(Value::as_array).map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
            };
            let (pos, m, vel, ang) = phys(c);
            va.a.edit_car_state_full(id, |s| {
                s.phys.pos = pos;
                s.phys.rot_mat = m;
                s.phys.vel = vel;
                s.phys.ang_vel = ang;
                if has(c, "boost") { s.boost = f(c, "boost", s.boost); }
                let held: Option<Vec<f32>> = if has(c, "prev_controls") {
                    Some(c["prev_controls"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
                } else {
                    guess.clone()
                };
                if let Some(p) = held {
                    let ctl = controls(&p);
                    s.controls = ctl;
                    s.set_prev_controls(ctl);
                }
                if has(c, "on_ground") || has(c, "wheels_contact") {
                    let ground = b(c, "on_ground");
                    let w = c.get("wheels_contact").and_then(Value::as_array);
                    // a packet doesn't show the wheels: guess all four touching when the car is on the ground
                    let wheel = |i: usize| match w {
                        Some(w) => w.get(i).is_some_and(|x| x.as_bool().unwrap_or(x.as_f64().unwrap_or(0.0) != 0.0)),
                        None => ground,
                    };
                    s.set_contact_flags(ground, [wheel(0), wheel(1), wheel(2), wheel(3)]);
                }
                if has(c, "has_jumped") { s.has_jumped = b(c, "has_jumped"); }
                if has(c, "has_double_jumped") { s.has_double_jumped = b(c, "has_double_jumped"); }
                if has(c, "has_flipped") { s.has_flipped = b(c, "has_flipped"); }
                if has(c, "flip_rel_torque") { s.flip_rel_torque = vec(&c["flip_rel_torque"]); }
                if has(c, "jump_ticks") { s.jump_ticks = c["jump_ticks"].as_f64().unwrap_or(0.0) as u32; }
                if has(c, "flip_time") { s.flip_time = f(c, "flip_time", 0.0); }
                if has(c, "is_flipping") { s.is_flipping = b(c, "is_flipping"); }
                if has(c, "is_jumping") { s.is_jumping = b(c, "is_jumping"); }
                if has(c, "jump_hold_broken") { s.jump_hold_broken = b(c, "jump_hold_broken"); }
                if has(c, "air_time") { s.air_time = f(c, "air_time", 0.0); }
                if has(c, "air_time_since_jump") { s.air_time_since_jump = f(c, "air_time_since_jump", 0.0); }
                if has(c, "handbrake_val") { s.handbrake_val = f(c, "handbrake_val", 0.0); }
                if has(c, "is_supersonic") { s.is_supersonic = b(c, "is_supersonic"); }
                if has(c, "supersonic_grace_timer") { s.supersonic_grace_timer = f(c, "supersonic_grace_timer", 0.0); }
                if has(c, "time_since_boosted") { s.time_since_boosted = f(c, "time_since_boosted", 0.0); }
                if has(c, "boosting_time") { s.boosting_time = f(c, "boosting_time", 0.0); }
                if has(c, "is_boosting") { s.is_boosting = b(c, "is_boosting"); }
                if has(c, "is_auto_flipping") { s.is_auto_flipping = b(c, "is_auto_flipping"); }
                if has(c, "auto_flip_timer") { s.auto_flip_timer = f(c, "auto_flip_timer", 0.0); }
                if has(c, "auto_flip_torque_scale") { s.auto_flip_torque_scale = f(c, "auto_flip_torque_scale", 0.0); }
                if has(c, "bump_cooldown_timer") { s.bump_cooldown_timer = f(c, "bump_cooldown_timer", 0.0); }
                if has(c, "bump_victim") {
                    let v = c["bump_victim"].as_i64().unwrap_or(-1);
                    s.bump_cooldown_victim = (v >= 0).then_some(v as usize);
                }
                if has(c, "last_extra_hit_tick") {
                    let v = c["last_extra_hit_tick"].as_i64().unwrap_or(-1);
                    s.last_extra_hit_tick = (v >= 0).then_some(v as u64);
                }
                if has(c, "is_demoed") { s.is_demoed = b(c, "is_demoed"); }
                if has(c, "demo_respawn_timer") { s.demo_respawn_timer = f(c, "demo_respawn_timer", 0.0); }
            });
        }
        let ball = &st["ball"];
        // a ball given as null stays as it is
        if !ball.is_null() {
            let (pos, m, vel, ang) = phys(ball);
            va.a.edit_ball_state(|s| {
                s.phys.pos = pos;
                s.phys.rot_mat = m;
                s.phys.vel = vel;
                s.phys.ang_vel = ang;
            });
            if b(ball, "asleep") {
                va.a.sleep_ball();
            }
        }
        if let Some(pads) = st.get("pads").and_then(Value::as_array) {
            for p in pads {
                let p = simbench_kit::v3(p);
                let n = va.a.num_boost_pads();
                let k = (0..n).min_by(|&x, &y| {
                    let d = |i: usize| { let q = va.a.get_boost_pad_config(i).pos; (q.x - p[0]).powi(2) + (q.y - p[1]).powi(2) };
                    d(x).total_cmp(&d(y))
                }).unwrap();
                va.a.edit_boost_pad_state(k, |s| s.cooldown = p[2].max(0.0));
            }
        }
        if b(st, "new_round") {
            for k in 0..va.a.num_boost_pads() {
                va.a.edit_boost_pad_state(k, |s| s.cooldown = 0.0);
            }
        }
        if va.pads_from_recording {
            // on the suite's rows pads come only from the recording (its credits), as for every simulator: v3's own
            // pickups off, every pad taken with a cooldown that never runs out
            for k in 0..va.a.num_boost_pads() {
                va.a.edit_boost_pad_state(k, |s| s.cooldown = 1e9);
            }
        }
        Ok(())
    }

    fn step(&self, va: &mut V3Arena, inp: &[[f32; TICK_CAR]], ticks: u32, _frame: Option<[f32; 2]>) {
        for (k, r) in inp.iter().enumerate() {
            let mut c = controls(&r[..8]);
            if r[8] < 0.5 {
                c = CarControls::default();
            }
            va.a.set_car_controls(va.ids[k], c);
        }
        for _ in 0..ticks {
            va.a.step_tick();
        }
        if va.pads_from_recording {
            for (k, r) in inp.iter().enumerate() {
                let gain = r[9];
                let id = va.ids[k];
                if gain > 0.0 && !va.a.get_car_state(id).is_demoed {
                    let now = va.a.get_car_state(id).boost;
                    va.a.set_car_boost(id, (now + gain * 100.0).min(100.0));
                }
            }
        }
    }

    fn bodies(&self, va: &V3Arena, out: &mut [[f32; BODY]]) {
        let put = |o: &mut [f32; BODY], p: &rocketsim::PhysState| {
            let (x, y, z) = (p.rot_mat.x_axis, p.rot_mat.y_axis, p.rot_mat.z_axis);
            *o = [p.pos.x, p.pos.y, p.pos.z, p.vel.x, p.vel.y, p.vel.z, p.ang_vel.x, p.ang_vel.y, p.ang_vel.z,
                  x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z];
        };
        for (k, &id) in va.ids.iter().enumerate() {
            let s = va.a.get_car_state(id);
            if s.is_demoed {
                out[k] = [0.0; BODY];
            } else {
                put(&mut out[k], &s.phys);
            }
        }
        put(&mut out[va.ids.len()], &va.a.get_ball_state().phys);
    }
}

fn main() {
    serve(V3);
}
