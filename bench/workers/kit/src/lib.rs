//! simbench worker kit for simulators written in Rust (or with a Rust API).
//!
//! Implement `Sim` for your simulator and call `serve(sim)` from `main`: the kit speaks the harness's wire protocol
//! (bench/simbench/protocol.py) on stdin/stdout and runs the benchmarks' loops: the speed benchmark across threads with
//! one barrier per decision, and accuracy segments tick by tick. Anything your simulator prints should go to stderr.

use serde_json::{Value, json};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Instant;

pub const TICK_CAR: usize = 10;
/// Per tick when frames are given: [render alpha of its frame, the frame's duration (s), physics steps in the frame on
/// its first tick (0 on the frame's later ticks), the recording's raw render fraction for the tick (-1 on a frame's
/// earlier ticks), its raw frame start time (-1 when not recorded)].
pub const TICK_FRAME: usize = 5;
pub const BODY: usize = 18;

/// One car of an arena: RocketSim's preset name, the game's body product id when known, and the team.
#[derive(Clone, Debug)]
pub struct CarSpec {
    pub preset: String,
    pub body: Option<u32>,
    pub team: u8,
    /// The car's whole configuration when the benchmark has it (the full-state suite's `CarConfig`: hitbox, wheels,
    /// spawn height), for simulators that take a configuration rather than a preset.
    pub config: Option<Value>,
}

/// What the harness prints about the simulator, and which benchmarks it can run (see the Python kit's `Adapter`).
pub struct Hello {
    pub name: String,
    pub version: String,
    pub about: String,
    pub device: &'static str,
    pub units: &'static str,
    pub caps: Value,
    pub choices: Vec<String>,
}

pub trait Sim: Sync {
    type Arena;
    fn hello(&self) -> Hello;
    fn setup(&mut self, _env: &Value) -> Result<(), String> {
        Ok(())
    }
    fn new_arena(&self, cars: &[CarSpec], map: &str) -> Result<Self::Arena, String>;
    /// Put the arena in this state (bench/simbench/state.py): set every field the simulator can take.
    fn set_state(&self, a: &mut Self::Arena, state: &Value) -> Result<(), String>;
    /// `inputs`: per car the 8 controls, the may-drive flag and the pad boost gain (0..1) given after the tick.
    fn step(&self, a: &mut Self::Arena, inputs: &[[f32; TICK_CAR]], ticks: u32, frame: Option<[f32; 2]>);
    /// Per car then the ball: pos, vel (in `units`), ang_vel, forward, right, up. A car without a body is zeros.
    fn bodies(&self, a: &Self::Arena, out: &mut [[f32; BODY]]);
    /// One game frame of `ticks.len()` physics ticks with the frame's real duration and render alpha (only called when
    /// the simulator's caps say `frame_timing`). `ticks[t]` are the inputs recorded for each tick of the frame (the game
    /// reads controls once, at the frame's start; pad credits come after the frame). `reads[t]` must get the bodies after
    /// each tick. The default steps the ticks one by one.
    fn step_frame(&self, a: &mut Self::Arena, ticks: &[Vec<[f32; TICK_CAR]>], frame_time: f32, alpha: f32, reads: &mut [Vec<[f32; BODY]>]) {
        for (t, rows) in ticks.iter().enumerate() {
            self.step(a, rows, 1, Some([alpha, frame_time / ticks.len() as f32]));
            self.bodies(a, &mut reads[t]);
        }
    }
    /// The speed benchmark on the simulator's own batched path (GPU simulators, simulators with their own threading).
    /// `ctl`: float32 [arenas][decisions][cars][8]; `actions`: the same controls as rows of RLGym's 90-action table
    /// (`action_table()`), one byte per car, [decisions][arenas][cars]. Return None to use the kit's threaded loop.
    fn speed_batched(&self, _job: &Value, _starts: &[Value], _ctl: &[f32], _actions: &[u8]) -> Option<Result<Value, String>> {
        None
    }
    /// Level A: the full-state suite's start state (`start["full"]` names byte ranges of `full`). Default: unsupported.
    fn set_full_state(&self, _a: &mut Self::Arena, _req: &Value, _full: &[u8]) -> Result<(), String> {
        Err("this simulator does not take the full-state suite's start state".into())
    }
}

fn read_msg(r: &mut impl Read) -> io::Result<(Value, Vec<u8>)> {
    let mut n = [0u8; 4];
    r.read_exact(&mut n)?;
    let mut h = vec![0u8; u32::from_le_bytes(n) as usize];
    r.read_exact(&mut h)?;
    let v: Value = serde_json::from_slice(&h).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let k = v.get("bin").and_then(Value::as_u64).unwrap_or(0) as usize;
    let mut b = vec![0u8; k];
    r.read_exact(&mut b)?;
    Ok((v, b))
}

fn write_msg(w: &mut impl Write, mut h: Value, payload: &[u8]) -> io::Result<()> {
    h["bin"] = json!(payload.len());
    let s = serde_json::to_vec(&h).unwrap();
    w.write_all(&(s.len() as u32).to_le_bytes())?;
    w.write_all(&s)?;
    w.write_all(payload)?;
    w.flush()
}

pub fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn bytes(f: &[f32]) -> Vec<u8> {
    f.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn cars_of(v: &Value) -> Vec<CarSpec> {
    v.as_array().map(|a| a.iter().map(|c| CarSpec {
        preset: c["preset"].as_str().unwrap_or("Octane").to_string(),
        body: c.get("body").and_then(Value::as_u64).map(|x| x as u32),
        team: c["team"].as_u64().unwrap_or(0) as u8,
        config: c.get("config").cloned(),
    }).collect()).unwrap_or_default()
}

/// The speed benchmark: arenas split over `threads` threads (each builds its own), one barrier per decision.
fn speed<S: Sim>(sim: &S, job: &Value, starts: &[Value], ctl: &[f32]) -> Result<Value, String> {
    let n_ar = job["arenas"].as_u64().unwrap() as usize;
    let n_dec = job["decisions"].as_u64().unwrap() as usize;
    let k = job["decision_ticks"].as_u64().unwrap() as u32;
    let reps = (job["warmup"].as_u64().unwrap() + job["repeats"].as_u64().unwrap()) as usize;
    let warm = job["warmup"].as_u64().unwrap() as usize;
    let cars = cars_of(&job["cars"]);
    let map = job.get("map").and_then(Value::as_str).unwrap_or("Stadium_P").to_string();
    let nc = cars.len();
    let threads = (job["threads"].as_u64().unwrap_or(1) as usize).clamp(1, n_ar.max(1));
    let per = ctl.len() / n_ar.max(1);
    let barrier = Arc::new(Barrier::new(threads + 1));
    let err: Mutex<Option<String>> = Mutex::new(None);
    let mut seconds = Vec::new();
    std::thread::scope(|s| {
        for t in 0..threads {
            let (lo, hi) = (n_ar * t / threads, n_ar * (t + 1) / threads);
            let (barrier, cars, map, err) = (barrier.clone(), &cars, &map, &err);
            s.spawn(move || {
                let mut arenas = Vec::new();
                for _ in lo..hi {
                    match sim.new_arena(cars, map) {
                        Ok(a) => arenas.push(a),
                        Err(e) => *err.lock().unwrap() = Some(e),
                    }
                }
                let mut inp = vec![[0f32; TICK_CAR]; nc];
                let mut out = vec![[0f32; BODY]; nc + 1];
                for _ in 0..reps {
                    for (j, a) in arenas.iter_mut().enumerate() {
                        if let Err(e) = sim.set_state(a, &starts[lo + j]) {
                            *err.lock().unwrap() = Some(e);
                        }
                    }
                    barrier.wait();
                    for d in 0..n_dec {
                        for (j, a) in arenas.iter_mut().enumerate() {
                            let base = (lo + j) * per + d * nc * 8;
                            for (c, row) in inp.iter_mut().enumerate() {
                                row[..8].copy_from_slice(&ctl[base + c * 8..base + c * 8 + 8]);
                                row[8] = 1.0;
                                row[9] = 0.0;
                            }
                            sim.step(a, &inp, k, None);
                            sim.bodies(a, &mut out);
                        }
                        barrier.wait();
                    }
                }
            });
        }
        for r in 0..reps {
            barrier.wait();
            let t0 = Instant::now();
            for _ in 0..n_dec {
                barrier.wait();
            }
            if r >= warm {
                seconds.push(t0.elapsed().as_secs_f64());
            }
        }
    });
    if let Some(e) = err.into_inner().unwrap() {
        return Err(e);
    }
    Ok(json!({ "seconds": seconds, "threads": threads }))
}

fn segment<S: Sim>(sim: &S, req: &Value, payload: &[u8], frame_timing: bool) -> Result<Vec<u8>, String> {
    let cars = cars_of(&req["cars"]);
    let nc = cars.len();
    let n = req["ticks"].as_u64().unwrap() as usize;
    let map = req.get("map").and_then(Value::as_str).unwrap_or("Stadium_P");
    let mut a = sim.new_arena(&cars, map)?;
    let need = n * nc * TICK_CAR * 4;
    let inp = floats(&payload[..need]);
    let mut at = need;
    let frames = if req.get("frames").and_then(Value::as_bool).unwrap_or(false) {
        let f = floats(&payload[at..at + n * TICK_FRAME * 4]);
        at += n * TICK_FRAME * 4;
        Some(f)
    } else {
        None
    };
    if req["start"].get("full").is_some() {
        sim.set_full_state(&mut a, req, &payload[at..])?;
    } else {
        sim.set_state(&mut a, &req["start"])?;
    }
    let row = |t: usize| -> Vec<[f32; TICK_CAR]> {
        (0..nc).map(|c| inp[(t * nc + c) * TICK_CAR..(t * nc + c + 1) * TICK_CAR].try_into().unwrap()).collect()
    };
    let mut res = Vec::with_capacity(n * (nc + 1) * BODY);
    let mut t = 0;
    while t < n {
        let (len, ft, alpha) = match &frames {
            Some(f) if frame_timing => {
                let steps = (f[t * TICK_FRAME + 2] as usize).clamp(1, n - t);
                (steps, f[t * TICK_FRAME + 1], f[t * TICK_FRAME])
            }
            _ => (1, -1.0, -1.0),
        };
        let ticks: Vec<Vec<[f32; TICK_CAR]>> = (t..t + len).map(row).collect();
        let mut reads = vec![vec![[0f32; BODY]; nc + 1]; len];
        if frame_timing && frames.is_some() {
            sim.step_frame(&mut a, &ticks, ft, alpha, &mut reads);
        } else {
            sim.step(&mut a, &ticks[0], 1, None);
            sim.bodies(&a, &mut reads[0]);
        }
        for r in &reads {
            for b in r {
                res.extend_from_slice(b);
            }
        }
        t += len;
    }
    Ok(bytes(&res))
}

/// Serve the harness until it says quit.
pub fn serve<S: Sim>(mut sim: S) {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut r = BufReader::new(stdin.lock());
    let mut w = BufWriter::new(stdout.lock());
    let mut arenas: Vec<Option<S::Arena>> = Vec::new();
    let mut ncars: Vec<usize> = Vec::new();
    loop {
        let (req, payload) = match read_msg(&mut r) {
            Ok(m) => m,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return,
            Err(e) => {
                eprintln!("simbench worker: unreadable request ({e})");
                return;
            }
        };
        let op = req["op"].as_str().unwrap_or("").to_string();
        let res: Result<(Value, Vec<u8>), String> = (|| match op.as_str() {
            "hello" => {
                let h = sim.hello();
                Ok((json!({"name": h.name, "version": h.version, "about": h.about, "device": h.device, "units": h.units,
                           "caps": h.caps, "choices": h.choices, "protocol": 1}), vec![]))
            }
            "setup" => sim.setup(&req["env"]).map(|_| (json!({}), vec![])),
            "speed" => {
                let starts = req["starts"].as_array().cloned().unwrap_or_default();
                let job = &req["job"];
                let n = (job["arenas"].as_u64().unwrap() * job["decisions"].as_u64().unwrap()) as usize * cars_of(&job["cars"]).len();
                let ctl = floats(&payload[..(n * 8 * 4).min(payload.len())]);
                let actions = payload.get(n * 8 * 4..).unwrap_or(&[]);
                match sim.speed_batched(job, &starts, &ctl, actions) {
                    Some(r) => r.map(|v| (v, vec![])),
                    None => speed(&sim, job, &starts, &ctl).map(|v| (v, vec![])),
                }
            }
            "segment" => {
                let ft = sim.hello().caps.get("frame_timing").and_then(Value::as_bool).unwrap_or(false);
                segment(&sim, &req, &payload, ft).map(|b| (json!({}), b))
            }
            "new" => {
                let cars = cars_of(&req["cars"]);
                let a = sim.new_arena(&cars, req.get("map").and_then(Value::as_str).unwrap_or("Stadium_P"))?;
                arenas.push(Some(a));
                ncars.push(cars.len());
                Ok((json!({"arena": arenas.len() - 1}), vec![]))
            }
            "set" => {
                let i = req["arena"].as_u64().unwrap() as usize;
                let a = arenas.get_mut(i).and_then(Option::as_mut).ok_or("no such arena")?;
                sim.set_state(a, &req["state"]).map(|_| (json!({}), vec![]))
            }
            "step" | "bodies" => {
                let i = req["arena"].as_u64().unwrap() as usize;
                let nc = ncars[i];
                let a = arenas.get_mut(i).and_then(Option::as_mut).ok_or("no such arena")?;
                if op == "step" {
                    let f = floats(&payload);
                    let rows: Vec<[f32; TICK_CAR]> = (0..nc).map(|c| f[c * TICK_CAR..(c + 1) * TICK_CAR].try_into().unwrap()).collect();
                    let frame = req.get("frame").and_then(Value::as_array).map(|v| [v[0].as_f64().unwrap() as f32, v[1].as_f64().unwrap() as f32]);
                    sim.step(a, &rows, req.get("ticks").and_then(Value::as_u64).unwrap_or(1) as u32, frame);
                }
                let mut out = vec![[0f32; BODY]; nc + 1];
                sim.bodies(a, &mut out);
                Ok((json!({}), bytes(&out.concat())))
            }
            "close" => {
                let i = req["arena"].as_u64().unwrap() as usize;
                if let Some(slot) = arenas.get_mut(i) {
                    *slot = None;
                }
                Ok((json!({}), vec![]))
            }
            "quit" => Ok((json!({"quit": true}), vec![])),
            other => Err(format!("unknown request {other:?}")),
        })();
        let quit = op == "quit";
        let ok = match res {
            Ok((h, b)) => write_msg(&mut w, h, &b),
            Err(e) => write_msg(&mut w, json!({"error": e}), &[]),
        };
        if ok.is_err() || quit {
            return;
        }
    }
}

// ------------------------------------------------------------ small helpers for state JSON

pub fn v3(v: &Value) -> [f32; 3] {
    let a = v.as_array().map(|a| a.as_slice()).unwrap_or(&[]);
    let g = |i: usize| a.get(i).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    [g(0), g(1), g(2)]
}

pub fn f(v: &Value, k: &str, d: f32) -> f32 {
    v.get(k).and_then(Value::as_f64).map(|x| x as f32).unwrap_or(d)
}

pub fn b(v: &Value, k: &str) -> bool {
    match v.get(k) {
        Some(Value::Bool(x)) => *x,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) != 0.0,
        _ => false,
    }
}

pub fn has(v: &Value, k: &str) -> bool {
    v.get(k).is_some_and(|x| !x.is_null())
}

/// A packet-level car (level B's packet variant) as RocketSim v3 CarState fields, for adapters that map those: on the
/// ground from the air state, jumping / dodging from it, the dodge's clock and direction, the held controls from the
/// last input, the time since the jump from the dodge timeout. Fields a packet doesn't have stay absent (the adapter
/// guesses them). Cars that already have v3 fields come back unchanged.
pub fn v3_from_packet(c: &Value) -> Value {
    let Some(air) = c.get("air_state").and_then(Value::as_i64) else { return c.clone() };
    let mut o = c.clone();
    o["on_ground"] = json!(air == 0);
    o["is_jumping"] = json!(air == 1);
    o["is_flipping"] = json!(air == 3);
    let d = c.get("dodge_dir").and_then(Value::as_array).map(|a| [a[0].as_f64().unwrap_or(0.0), a[1].as_f64().unwrap_or(0.0)]).unwrap_or([0.0, 0.0]);
    if air == 3 {
        o["flip_time"] = c["dodge_elapsed"].clone();
        o["flip_rel_torque"] = json!([-d[1], d[0], 0.0]);
    } else {
        // not dodging: guess the last dodge (if any) ended long ago (its clock gates the next dodge)
        o["flip_time"] = json!(10.0);
    }
    if let Some(li) = c.get("last_input") {
        o["prev_controls"] = li.clone();
        // guesses from the held input: the smoothed handbrake at its end, boosting when boost is held (and there is some)
        let held = |k: usize| li.get(k).and_then(Value::as_f64).unwrap_or(0.0) > 0.5;
        o["handbrake_val"] = json!(if held(7) { 1.0 } else { 0.0 });
        let boosting = held(6) && c.get("boost").and_then(Value::as_f64).unwrap_or(0.0) > 0.0;
        o["is_boosting"] = json!(boosting);
        o["boosting_time"] = json!(if boosting { 0.5 } else { 0.0 });
        o["time_since_boosted"] = json!(if boosting { 0.0 } else { 0.5 });
    }
    let t = c.get("dodge_timeout").and_then(Value::as_f64).unwrap_or(-1.0);
    if t >= 0.0 {
        o["air_time_since_jump"] = json!(1.25 - t);
    }
    o
}

/// RLGym's 90-action lookup table, 8 floats a row (throttle, steer, pitch, yaw, roll, jump, boost, handbrake), in the
/// order the harness's speed controls index it.
pub fn action_table() -> Vec<f32> {
    let mut t = Vec::new();
    for throttle in [-1.0f32, 0.0, 1.0] {
        for steer in [-1.0f32, 0.0, 1.0] {
            for boost in [0.0f32, 1.0] {
                for handbrake in [0.0f32, 1.0] {
                    if boost == 1.0 && throttle != 1.0 {
                        continue;
                    }
                    t.extend([if throttle != 0.0 { throttle } else { boost }, steer, 0.0, steer, 0.0, 0.0, boost, handbrake]);
                }
            }
        }
    }
    for pitch in [-1.0f32, 0.0, 1.0] {
        for yaw in [-1.0f32, 0.0, 1.0] {
            for roll in [-1.0f32, 0.0, 1.0] {
                for jump in [0.0f32, 1.0] {
                    for boost in [0.0f32, 1.0] {
                        if (jump == 1.0 && yaw != 0.0) || (pitch == 0.0 && roll == 0.0 && jump == 0.0) {
                            continue;
                        }
                        let hb = if jump == 1.0 && (pitch != 0.0 || yaw != 0.0 || roll != 0.0) { 1.0 } else { 0.0 };
                        t.extend([boost, yaw, pitch, yaw, roll, jump, boost, hb]);
                    }
                }
            }
        }
    }
    t
}
