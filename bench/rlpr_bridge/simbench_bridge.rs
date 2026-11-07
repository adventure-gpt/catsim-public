//! simbench bridge for RocketSim's rlpr_metric: runs any simbench worker (bench/simbench/protocol.py) as the metric's
//! backend, so RocketSim's own, unchanged scorer can score any simulator. Part of catSim's repository (MIT); the
//! harness (`python -m simbench rlpr`) copies it into a checkout of RocketSim that it fetches at a pinned commit and
//! wires it in with two small edits to rlpr_metric.rs (the backend it constructs and the label it prints).
//!
//! With SIMBENCH_WORKER unset the metric runs RocketSim v3's own backend, unchanged.
//! SIMBENCH_WORKER: the worker's command line as a JSON list of strings; SIMBENCH_SETUP: its setup request's env (JSON);
//! SIMBENCH_LABEL: the name printed in the metric's tables.
//!
//! An RLPR tick record goes to the worker as a simbench state at the "rlpr" level: every car's body (uu, the rotation
//! matrix as recorded), its jump and flip state and clocks, flip torque, boost (exactly, 0..1, as `boost01`), wheel
//! contacts and normals, world contact, held controls; the ball. The metric's hidden-state restores (handbrake, boost
//! latch, extra-hit gate, bump cooldown, jump hold) are added to that state before the next step.

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use glam::Vec3A;
use rocketsim_test::rlpr::{cpp_records::ControlsRecord, tick_record::TickRecord};

use super::common::{BodySnapshot, ReplayBackend, SimContactEvents, Snapshot};
use super::v3::V3Backend;

pub fn label() -> &'static str {
    match std::env::var("SIMBENCH_LABEL") {
        Ok(s) if std::env::var("SIMBENCH_WORKER").is_ok() => Box::leak(s.chars().take(7).collect::<String>().into_boxed_str()),
        _ => "v3",
    }
}

pub enum Either {
    V3(V3Backend),
    Bridge(Bridge),
}

impl Either {
    pub fn new(dodge_deadzone: f32) -> Self {
        match std::env::var("SIMBENCH_WORKER") {
            Ok(cmd) => Either::Bridge(Bridge::start(&cmd, &std::env::var("SIMBENCH_SETUP").unwrap_or_else(|_| "{}".into()))),
            Err(_) => Either::V3(V3Backend::with_dodge_deadzone(dodge_deadzone)),
        }
    }

    pub fn set_bodies(&mut self, bodies: &[usize]) {
        match self {
            Either::V3(b) => b.set_bodies(bodies),
            Either::Bridge(b) => b.set_bodies(bodies),
        }
    }
}

macro_rules! both {
    ($s:expr, $b:ident => $e:expr) => {
        match $s {
            Either::V3($b) => $e,
            Either::Bridge($b) => $e,
        }
    };
}

impl ReplayBackend for Either {
    fn reset(&mut self, start: &TickRecord) { both!(self, b => b.reset(start)) }
    fn set_state(&mut self, state: &TickRecord) { both!(self, b => b.set_state(state)) }
    fn reset_motion(&mut self, state: &TickRecord) { both!(self, b => b.reset_motion(state)) }
    fn set_handbrake_value(&mut self, i: usize, v: f32) { both!(self, b => b.set_handbrake_value(i, v)) }
    fn set_boost_state(&mut self, i: usize, armed: bool, t: f32) { both!(self, b => b.set_boost_state(i, armed, t)) }
    fn supports_cooldown_restore(&self) -> bool { both!(self, b => b.supports_cooldown_restore()) }
    fn suppress_next_extra_hit(&mut self, i: usize, s: bool) { both!(self, b => b.suppress_next_extra_hit(i, s)) }
    fn supports_bump_restore(&self) -> bool { both!(self, b => b.supports_bump_restore()) }
    fn set_bump_cooldown(&mut self, i: usize, v: usize, s: f32) { both!(self, b => b.set_bump_cooldown(i, v, s)) }
    fn set_jump_hold_broken(&mut self, i: usize, br: bool) { both!(self, b => b.set_jump_hold_broken(i, br)) }
    fn refresh_sticky_gates(&mut self) { both!(self, b => b.refresh_sticky_gates()) }
    fn step(&mut self, c: &[ControlsRecord]) -> Vec<SimContactEvents> { both!(self, b => b.step(c)) }
    fn snapshot(&mut self, i: usize) -> Snapshot { both!(self, b => b.snapshot(i)) }
}

const NAMES: [&str; 7] = ["Octane", "Dominus", "Plank", "Breakout", "Hybrid", "Merc", "Psyclops"];

pub struct Bridge {
    _child: Child,
    inp: ChildStdin,
    out: ChildStdout,
    bodies: Vec<usize>,
    built: Vec<usize>,
    arena: Option<u64>,
    units_bt: bool,
    cars: Vec<String>,
    ball: String,
    extra: Vec<Vec<String>>,
    dirty: bool,
    last: Vec<[f32; 18]>,
    tick: u64,
}

fn f(x: f32) -> String {
    if x.is_finite() { format!("{x:?}") } else { "0.0".into() }
}

fn v(x: impl Into<Vec3A>) -> String {
    let a: Vec3A = x.into();
    format!("[{},{},{}]", f(a.x), f(a.y), f(a.z))
}

fn controls(c: &ControlsRecord) -> [f32; 8] {
    let b = |x: bool| if x { 1.0 } else { 0.0 };
    [c.throttle, c.steer, c.pitch, c.yaw, c.roll, b(c.jump), b(c.boost), b(c.handbrake)]
}

impl Bridge {
    fn start(cmd: &str, setup: &str) -> Bridge {
        let parts: Vec<String> = cmd.trim().trim_start_matches('[').trim_end_matches(']').split("\",")
            .map(|s| s.trim().trim_matches('"').replace("\\\\", "\\").to_string()).filter(|s| !s.is_empty()).collect();
        let mut child = Command::new(&parts[0]).args(&parts[1..]).stdin(Stdio::piped()).stdout(Stdio::piped())
            .spawn().unwrap_or_else(|e| panic!("simbench bridge: could not start {}: {e}", parts[0]));
        let inp = child.stdin.take().unwrap();
        let out = child.stdout.take().unwrap();
        let mut b = Bridge { _child: child, inp, out, bodies: vec![], built: vec![], arena: None, units_bt: false, cars: vec![],
                             ball: String::new(), extra: vec![], dirty: false, last: vec![], tick: 0 };
        let (h, _) = b.call("{\"op\":\"hello\"}", &[]);
        b.units_bt = h.contains("\"units\":\"bt\"");
        b.call(&format!("{{\"op\":\"setup\",\"env\":{setup}}}"), &[]);
        b
    }

    fn call(&mut self, header: &str, payload: &[u8]) -> (String, Vec<u8>) {
        let h = format!("{},\"bin\":{}}}", header.strip_suffix('}').unwrap_or(header), payload.len());
        self.inp.write_all(&(h.len() as u32).to_le_bytes()).unwrap();
        self.inp.write_all(h.as_bytes()).unwrap();
        self.inp.write_all(payload).unwrap();
        self.inp.flush().unwrap();
        let mut n = [0u8; 4];
        self.out.read_exact(&mut n).expect("simbench bridge: the worker stopped");
        let mut hb = vec![0u8; u32::from_le_bytes(n) as usize];
        self.out.read_exact(&mut hb).unwrap();
        let rep = String::from_utf8_lossy(&hb).to_string();
        if rep.contains("\"error\"") {
            panic!("simbench bridge: the worker failed: {rep}");
        }
        let k = rep.split("\"bin\":").nth(1).and_then(|s| s.trim_start().split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let mut b = vec![0u8; k];
        self.out.read_exact(&mut b).unwrap();
        (rep, b)
    }

    fn set_bodies(&mut self, bodies: &[usize]) {
        self.bodies = bodies.to_vec();
    }

    fn ensure(&mut self, n: usize) {
        let want: Vec<usize> = (0..n).map(|i| self.bodies.get(i).copied().unwrap_or(0)).collect();
        if self.arena.is_some() && self.built == want {
            return;
        }
        if let Some(a) = self.arena {
            self.call(&format!("{{\"op\":\"close\",\"arena\":{a}}}"), &[]);
        }
        let cars: Vec<String> = want.iter().enumerate()
            .map(|(i, &p)| format!("{{\"preset\":\"{}\",\"team\":{}}}", NAMES.get(p).copied().unwrap_or("Octane"), i % 2)).collect();
        let (rep, _) = self.call(&format!("{{\"op\":\"new\",\"cars\":[{}]}}", cars.join(",")), &[]);
        let id = rep.split("\"arena\":").nth(1).and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next()).and_then(|s| s.parse().ok()).unwrap_or(0);
        self.arena = Some(id);
        self.built = want;
    }

    fn flush(&mut self) {
        if !self.dirty {
            return;
        }
        let cars: Vec<String> = self.cars.iter().zip(&self.extra)
            .map(|(c, e)| if e.is_empty() { format!("{{{c}}}") } else { format!("{{{c},{}}}", e.join(",")) }).collect();
        let st = format!("{{\"cars\":[{}],\"ball\":{{{}}},\"tick\":{},\"no_goals\":true}}", cars.join(","), self.ball, self.tick);
        let a = self.arena.unwrap();
        self.call(&format!("{{\"op\":\"set\",\"arena\":{a},\"state\":{st}}}"), &[]);
        self.dirty = false;
    }
}

impl ReplayBackend for Bridge {
    fn reset(&mut self, start: &TickRecord) {
        self.set_state(start);
    }

    fn set_state(&mut self, t: &TickRecord) {
        self.ensure(t.car_records.len());
        self.cars.clear();
        self.extra = vec![Vec::new(); t.car_records.len()];
        for r in &t.car_records {
            if !r.is_available() {
                // a slot without physics in this row (demolished, or not captured): no body, and no respawn of its own
                self.cars.push("\"pos\":[0.0,0.0,0.0],\"vel\":[0.0,0.0,0.0],\"ang_vel\":[0.0,0.0,0.0],\"is_demoed\":true,\"demo_respawn_timer\":1000.0".into());
                continue;
            }
            let p = &r.phys;
            let m = &p.rot.rows;
            let fwd = Vec3A::new(m[0].x, m[1].x, m[2].x);
            let right = Vec3A::new(m[0].y, m[1].y, m[2].y);
            let up = Vec3A::new(m[0].z, m[1].z, m[2].z);
            let pc = controls(&r.prev_controls);
            let order: [usize; 4] = [0, 1, 2, 3];      // RLPR's wheels are in RocketSim's order (as v3's backend reads them)
            let w: Vec<String> = order.iter().map(|&k| (if r.wheels[k].has_contact { "true" } else { "false" }).to_string()).collect();
            let wn: Vec<String> = order.iter().map(|&k| v(r.wheels[k].contact_normal)).collect();
            self.cars.push(format!(
                "\"pos\":{},\"fwd\":{},\"right\":{},\"up\":{},\"vel\":{},\"ang_vel\":{},\"on_ground\":{},\"is_jumping\":{},\"is_flipping\":{},\
                 \"jump_time\":{},\"flip_time\":{},\"has_jumped\":{},\"double_jumped_or_flipped\":{},\"has_flip\":{},\"flip_rel_torque\":{},\
                 \"boost\":{},\"boost01\":{},\"prev_controls\":[{}],\"wheels_contact\":[{}],\"wheel_normals\":[{}],\"world_contact\":{},\
                 \"world_contact_normal\":{},\"is_demoed\":false",
                v(p.pos), v(fwd), v(right), v(up), v(p.lin_vel), v(p.ang_vel), r.is_on_ground, r.is_jumping, r.is_flipping,
                f(r.jump_time), f(r.flip_time), r.has_jumped, r.double_jumped_or_flipped, r.has_flip, v(r.flip_rel_torque),
                f(r.boost_amount * 100.0), f(r.boost_amount), pc.map(f).join(","), w.join(","), wn.join(","), p.has_world_contact,
                v(p.world_contact_normal)));
        }
        let b = &t.ball_record;
        let m = &b.rot.rows;
        self.ball = format!("\"pos\":{},\"fwd\":{},\"right\":{},\"up\":{},\"vel\":{},\"ang_vel\":{}", v(b.pos),
                            v(Vec3A::new(m[0].x, m[1].x, m[2].x)), v(Vec3A::new(m[0].y, m[1].y, m[2].y)), v(Vec3A::new(m[0].z, m[1].z, m[2].z)),
                            v(b.lin_vel), v(b.ang_vel));
        self.dirty = true;
    }

    fn set_handbrake_value(&mut self, i: usize, value: f32) {
        self.extra[i].push(format!("\"handbrake_val\":{}", f(value)));
        self.dirty = true;
    }

    fn set_boost_state(&mut self, i: usize, armed: bool, time: f32) {
        self.extra[i].push(format!("\"is_boosting\":{armed},\"boosting_time\":{0},\"time_since_boosted\":{0}", f(time)));
        self.dirty = true;
    }

    fn supports_cooldown_restore(&self) -> bool {
        true
    }

    fn suppress_next_extra_hit(&mut self, i: usize, suppress: bool) {
        let t = if suppress { self.tick as i64 } else { -1 };
        self.extra[i].push(format!("\"last_extra_hit_tick\":{t}"));
        self.dirty = true;
    }

    fn supports_bump_restore(&self) -> bool {
        true
    }

    fn set_bump_cooldown(&mut self, i: usize, victim: usize, seconds: f32) {
        self.extra[i].push(format!("\"bump_cooldown_timer\":{},\"bump_victim\":{victim}", f(seconds.max(0.0))));
        self.dirty = true;
    }

    fn set_jump_hold_broken(&mut self, i: usize, broken: bool) {
        self.extra[i].push(format!("\"jump_hold_broken\":{broken}"));
        self.dirty = true;
    }

    fn step(&mut self, c: &[ControlsRecord]) -> Vec<SimContactEvents> {
        self.flush();
        let mut payload = Vec::with_capacity(c.len() * 40);
        for r in c {
            for x in controls(r).iter().chain([1.0f32, 0.0].iter()) {
                payload.extend_from_slice(&x.to_le_bytes());
            }
        }
        let a = self.arena.unwrap();
        let (_, data) = self.call(&format!("{{\"op\":\"step\",\"arena\":{a},\"cars\":{},\"ticks\":1}}", c.len()), &payload);
        let fl: Vec<f32> = data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        self.last = fl.chunks_exact(18).map(|b| b.try_into().unwrap()).collect();
        self.tick += 1;
        vec![SimContactEvents::default(); c.len()]
    }

    fn snapshot(&mut self, i: usize) -> Snapshot {
        let k = if self.units_bt { 50.0f32 } else { 1.0 };
        let body = |r: &[f32; 18]| BodySnapshot {
            pos: Vec3A::new(r[0] * k, r[1] * k, r[2] * k),
            vel: Vec3A::new(r[3] * k, r[4] * k, r[5] * k),
            ang_vel: Vec3A::new(r[6], r[7], r[8]),
            forward: Vec3A::new(r[9], r[10], r[11]),
            up: Vec3A::new(r[15], r[16], r[17]),
        };
        let n = self.last.len();
        Snapshot { car: body(&self.last[i]), ball: body(&self.last[n - 1]) }
    }
}
