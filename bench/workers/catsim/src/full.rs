//! Level A for catSim: the RocketGPU full-state suite's start state (its `s.*` arrays, RocketGPU's field layout) mapped into
//! catSim's arena, the contact memory included (the broadphase's pair order, the manifolds' order and their cached
//! points, translated from RocketGPU's layout into catSim's, `Full::contacts`). docs/BENCHMARKS.md lists every field.

use catsim::bt::{ACTIVE, ISLAND_SLEEPING};
use catsim::game::arena::{Actor, Arena, Phase, SPAWN_ACTOR_Z};
use catsim::game::contacts::{BodyRef, Contacts, Manifold, ManifoldOwner, Pair};
use catsim::game::hidden::{BodyState, CarHidden, WheelGround};
use catsim::game::round::{CountdownClock, FrameTimer};
use catsim::math::rot::Rotator;
use catsim::math::{Mat3, Vec3, v3};
use serde_json::Value;

const SLOTS: usize = 8;
/// catSim's wheel w is the suite's (RocketSim order) wheel SUITE_WHEEL[w]: left and right swapped.
const SUITE_WHEEL: [usize; 4] = [1, 0, 3, 2];

/// The clock value one frame of `dt` earlier: the float `x` whose sum `x + dt` is exactly `v` (catSim's frame adds `dt`
/// to it again), the nearest to `v - dt`; `v - dt` when no float adds up to `v`.
fn before_add(v: f32, dt: f32) -> f32 {
    let x = v - dt;
    let step = |x: f32, k: i32| f32::from_bits((x.to_bits() as i32 + if x >= 0.0 { k } else { -k }) as u32);
    for k in [0, -1, 1, -2, 2, -3, 3] {
        let y = step(x, k);
        if y + dt == v {
            return y;
        }
    }
    x
}

pub struct Full<'a> {
    index: &'a Value,
    data: &'a [u8],
}

impl<'a> Full<'a> {
    pub fn new(index: &'a Value, data: &'a [u8]) -> Self {
        Full { index, data }
    }

    fn raw(&self, name: &str) -> &'a [u8] {
        match self.index.get(name).and_then(Value::as_array) {
            Some(r) => {
                let (o, n) = (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize);
                &self.data[o..o + n]
            }
            None => &[],
        }
    }

    fn slot(&self, name: &str, k: usize) -> &'a [u8] {
        let r = self.raw(name);
        let per = r.len() / SLOTS;
        &r[k * per..(k + 1) * per]
    }

    fn f(&self, name: &str, k: usize) -> Vec<f32> {
        self.slot(name, k).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn f1(&self, name: &str, k: usize) -> f32 {
        self.f(name, k).first().copied().unwrap_or(0.0)
    }

    fn u8(&self, name: &str, k: usize) -> u8 {
        self.slot(name, k).first().copied().unwrap_or(0)
    }

    fn u32(&self, name: &str, k: usize) -> u32 {
        let s = self.slot(name, k);
        if s.len() < 4 { 0 } else { u32::from_le_bytes([s[0], s[1], s[2], s[3]]) }
    }

    fn u64(&self, name: &str, k: usize) -> u64 {
        let s = self.slot(name, k);
        if s.len() < 8 { u64::MAX } else { u64::from_le_bytes(s[..8].try_into().unwrap()) }
    }

    fn whole_f(&self, name: &str) -> Vec<f32> {
        self.raw(name).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn whole_u(&self, name: &str) -> u64 {
        let r = self.raw(name);
        let mut b = [0u8; 8];
        b[..r.len().min(8)].copy_from_slice(&r[..r.len().min(8)]);
        u64::from_le_bytes(b)
    }

    fn vec(v: &[f32], at: usize) -> Vec3 {
        v3(v[at], v[at + 1], v[at + 2])
    }

    /// The Bullet basis by rows from its three columns.
    fn basis(fwd: &[f32], right: &[f32], up: &[f32]) -> Mat3 {
        Mat3 { r: [v3(fwd[0], right[0], up[0]), v3(fwd[1], right[1], up[1]), v3(fwd[2], right[2], up[2])] }
    }

    /// The broadphase handles the start state's bodies hold (ball, then the cars), when the state has them.
    pub fn handles(&self, nc: usize) -> Option<Vec<u16>> {
        let r = self.raw("solver_scratch.bp.sweep.sap_body_handle");
        if r.len() < nc + 1 {
            return None;
        }
        let h: Vec<u16> = r[..nc + 1].iter().map(|&x| x as u16).collect();
        let distinct = h.iter().collect::<std::collections::HashSet<_>>().len() == h.len();
        (h.iter().all(|&x| x != 0) && distinct).then_some(h)
    }

    /// Map the whole start state in. `dt0`: the start tick's frame time (the suite's frame timers already hold it; catSim's
    /// frame adds it again, so the timers go back one frame).
    pub fn apply(&self, a: &mut Arena, req: &Value, dt0: f32) -> Result<(), String> {
        let nc = a.players.len();
        let phys = self.whole_u("env_states.tick") as i64 - 1;
        a.physics_frame = phys as i32;
        let ids: Vec<u64> = req.get("ids").and_then(Value::as_array).map(|v| v.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
        let demoed: Vec<bool> = (0..nc).map(|k| self.u8("car_states.is_demoed", k) != 0).collect();
        let mut grounds = Vec::new();
        for k in 0..nc {
            if demoed[k] {
                continue;
            }
            let body = BodyState {
                origin: Self::vec(&self.f("car_states.pos_bt", k), 0),
                basis: Self::basis(&self.f("car_states.basis_forward", k), &self.f("car_states.basis_right", k), &self.f("car_states.basis_up", k)),
                lin_vel: Self::vec(&self.f("car_states.vel_bt", k), 0),
                ang_vel: Self::vec(&self.f("car_states.internal_ang_vel", k), 0),
                asleep: false,
                deactivation_time: 0.0,
            };
            if a.players[k].car.is_none() {
                let s = catsim::game::rb_state::RbState { location: v3(body.origin.x * 50.0, body.origin.y * 50.0, body.origin.z * 50.0), ..Default::default() };
                a.set_car_fresh(k, &s, 0.0);
            }
            let act = a.players[k].car.as_ref().map(|c| a.world.cw.bodies[c.body.0 as usize].activation_state).unwrap_or(ACTIVE);
            let body = BodyState { asleep: act == ISLAND_SLEEPING, ..body };
            a.set_car_body(k, &body);
            if self.u8("car_states.restored_inv_inertia_valid", k) != 0 {
                let m = self.f("car_states.restored_inv_inertia_bt", k);
                if m.len() == 9 {
                    let c = a.players[k].car.as_ref().unwrap();
                    a.world.cw.bodies[c.body.0 as usize].inv_inertia_world = Mat3 { r: [Self::vec(&m, 0), Self::vec(&m, 3), Self::vec(&m, 6)] };
                }
            }
            let mut h = a.car_hidden(k).ok_or("car has no body")?;
            self.car_hidden(a, k, &mut h, phys, &ids);
            a.set_car_hidden(k, &h);
            grounds.push(k);
            a.players[k].last_wheels_hit_ball_time = -1000.0;
        }
        // the ball
        let bp = self.whole_f("ball_states.pos_bt");
        let bv = self.whole_f("ball_states.vel_bt");
        let asleep = self.whole_u("ball_states.sleeping") as u32 != 0;
        let dtime = self.whole_f("ball_states.deactivation_time").first().copied().unwrap_or(0.0);
        // the state's sleep flag is Bullet's: catSim run into a segment's start from the one before sleeps the ball exactly
        // when the flag is set (147 of 147 run-ins checked, kickoff countdowns included)
        let sleeping = asleep;
        let ball = BodyState {
            origin: Self::vec(&bp, 0),
            basis: Self::basis(&self.whole_f("ball_states.basis_forward"), &self.whole_f("ball_states.basis_right"), &self.whole_f("ball_states.basis_up")),
            lin_vel: Self::vec(&bv, 0),
            ang_vel: Self::vec(&self.whole_f("ball_states.ang_vel"), 0),
            asleep: sleeping,
            deactivation_time: dtime,
        };
        a.set_ball_body(&ball);
        if self.whole_u("ball_states.restored_inv_inertia_valid") as u32 != 0 {
            let m = self.whole_f("ball_states.restored_inv_inertia_bt");
            if m.len() == 9 {
                let b = a.ball.as_ref().unwrap();
                a.world.cw.bodies[b.body.0 as usize].inv_inertia_world = Mat3 { r: [Self::vec(&m, 0), Self::vec(&m, 3), Self::vec(&m, 6)] };
            }
        }
        a.ball_held_2d = (self.whole_u("ball_states.constrained_2d") & 0xffff) != 0;
        // what each touching wheel stands on (the arena, a car or the ball). `set_wheel_normals_mask` marks the wheels
        // whose contact is the game's world contact (bit 5: as recorded); a touching wheel it leaves out stands on a car
        // or the ball, which catSim's own wheel ray in the start state tells apart (else the nearer of the two). The
        // ray's contact step pushes a dynamic ground it is inside of (a ball under a wheel): only the rays are wanted
        // here, so the bodies are put back as they were afterwards
        let bodies_before = a.world.cw.bodies.clone();
        for k in grounds {
            let Some(mut h) = a.car_hidden(k) else { continue };
            let Some(c) = a.players[k].car.as_ref() else { continue };
            let mut v = c.vehicle.clone();
            catsim::bt::vehicle::update_rays_and_contacts(&mut v, &mut a.world.cw);
            let mask = self.u8("car_states.set_wheel_normals_mask", k);
            let recorded = mask >> 5 & 1 != 0;
            let ball_at = a.ball.as_ref().map(|b| a.world.cw.bodies[b.body.0 as usize].transform.origin);
            let mut changed = false;
            for w in 0..4 {
                if !h.wheels[w].contact {
                    continue;
                }
                let ray = match v.wheels[w].ground.and_then(|id| a.body_ref(id)) {
                    Some(BodyRef::Car(_)) => Some(WheelGround::Car),
                    Some(BodyRef::Ball) => Some(WheelGround::Ball),
                    Some(BodyRef::World(_)) => Some(WheelGround::Arena),
                    None => None,
                };
                let g = if !recorded {
                    ray.unwrap_or(WheelGround::Arena)
                } else if mask >> SUITE_WHEEL[w] & 1 != 0 {
                    WheelGround::Arena
                } else {
                    match ray {
                        Some(g @ (WheelGround::Car | WheelGround::Ball)) => g,
                        _ => {
                            let hp = v.wheels[w].hard_point;
                            let near_ball = ball_at.is_some_and(|b| ((hp.x - b.x).powi(2) + (hp.y - b.y).powi(2) + (hp.z - b.z).powi(2)).sqrt() < 1.825 + 0.6 + 2.0);
                            if near_ball { WheelGround::Ball } else { WheelGround::Car }
                        }
                    }
                };
                changed |= g != h.wheels[w].ground;
                h.wheels[w].ground = g;
            }
            if changed {
                a.set_car_hidden(k, &h);
            }
        }
        a.world.cw.bodies = bodies_before;
        // the broadphase boxes as the game's last bounds update left them (its predictive contacts read them first)
        let boxes = self.whole_f("solver_scratch.bp.sweep.sap_body_aabb");
        let valid = self.whole_u("solver_scratch.bp.sweep.sap_body_aabb_valid") & 0xffff;
        for obj in 0..=nc {
            if valid >> obj & 1 == 0 || boxes.len() < (obj + 1) * 6 {
                continue;
            }
            let id = if obj == 0 { a.ball.as_ref().map(|b| b.body) } else if demoed[obj - 1] { None } else { a.players[obj - 1].car.as_ref().map(|c| c.body) };
            if let Some(id) = id {
                let cw = &mut a.world.cw;
                let h = cw.bodies[id.0 as usize].proxy as u16;
                cw.broadphase.set_aabb(h, Self::vec(&boxes, obj * 6), Self::vec(&boxes, obj * 6 + 3), &mut cw.dispatcher);
            }
        }
        for k in 0..nc {
            if demoed[k] {
                a.set_car_demolished(k, self.f1("car_states.demo_respawn_timer", k) - dt0);
            }
        }
        // the visit order: the ball's notify first, then the cars in the game's order
        for call in req.get("calls").and_then(Value::as_array).into_iter().flatten() {
            if call[0].as_str() == Some("set_car_visit_order") {
                let order: Vec<usize> = call[1][0].as_array().into_iter().flatten().filter_map(Value::as_u64)
                    .filter_map(|x| ids.iter().position(|&i| i == x)).filter(|&k| k < nc && !demoed[k]).collect();
                if order.len() == a.vehicles.len() {
                    a.notifies = std::iter::once(Actor::Ball).chain(order.iter().map(|&k| Actor::Car(k))).collect();
                    a.vehicles = order;
                }
            }
        }
        // a countdown in progress: stage 1 the 1 s start timer, stage 2 the update loop ("3 - fires" left)
        let stage = self.whole_u("env_states.countdown_phase") as u8;
        if stage == 1 || stage == 2 {
            // catSim's first frame adds that frame's time to the clock again: start it where that sum gives the state's
            let cnt = before_add(self.whole_f("env_states.countdown_count").first().copied().unwrap_or(0.0), dt0);
            let fires = self.whole_u("env_states.countdown_fires") as u8 as i32;
            let mut c = CountdownClock::new(3, 3);
            if stage == 1 {
                c.start_timer = FrameTimer { rate: 1.0, elapsed: cnt, active: true };
            } else {
                c.start_timer = FrameTimer::default();
                c.remaining = 3 - fires;
                c.replicated_remaining = c.remaining;
                c.update_timer = FrameTimer { rate: 1.0, elapsed: cnt, active: true };
            }
            a.countdown_clock = Some(c);
            a.rule_countdown = true;
            a.phase = Phase::Countdown { left: 1 << 20 };
            a.ball_held_2d = true;
            for p in a.players.iter_mut() {
                if let Some(c) = p.car.as_mut() {
                    c.set_driving(false);
                }
            }
        }
        // the moving bodies in the state's object order (`sap_obj`: 0 the ball, k + 1 car k): the islands are numbered
        // in it, and the solver takes a body's contacts in the island order it gives
        let so = self.raw("solver_scratch.bp.sweep.sap_obj");
        let n = (self.whole_u("solver_scratch.bp.sweep.sap_obj_count") as u8 as usize).min(so.len());
        let order: Vec<BodyRef> = so[..n].iter().filter_map(|&o| match o as usize {
            0 => Some(BodyRef::Ball),
            k if k <= nc && !demoed[k - 1] => Some(BodyRef::Car(k - 1)),
            _ => None,
        }).collect();
        if order.len() == a.body_order().len() {
            a.set_body_order(&order)?;
        }
        // the contact memory, in catSim's layout
        if std::env::var_os("SIMBENCH_CATSIM_NO_CONTACTS").is_none() {
            let c = self.contacts(a, nc, &demoed)?;
            a.set_contacts(&c)?;
        }
        // respawn spots for the demolished cars, from the recording (the suite hands its simulators these choices)
        if let Some(sp) = req["start"].get("respawn_spots").and_then(Value::as_array) {
            for (k, s) in sp.iter().enumerate() {
                if let Some(s) = s.as_array() {
                    let g = |i: usize| s[i].as_f64().unwrap_or(0.0) as f32;
                    a.queue_respawn_spot(k, v3(g(0), g(1), SPAWN_ACTOR_Z), Rotator { pitch: 0, yaw: g(2) as i32, roll: 0 });
                }
            }
        }
        Ok(())
    }

    fn u32s(&self, name: &str) -> Vec<u32> {
        self.raw(name).chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// RocketGPU's pair and manifold keys (a kind, two bodies, a static id) and its manifolds' points, in catSim's
    /// terms. Bodies in a key: 0 the ball, k + 1 car k; a static id names an arena part through the broadphase handle it
    /// holds (`sap_static_id`; the handles are catSim's too: planes 1-6, meshes 7-22); static id 99 marks a predictive
    /// contact. The pairs keep catSim's own static-static pairs (made with the arena, always first), then the state's.
    fn contacts(&self, a: &Arena, nc: usize, demoed: &[bool]) -> Result<Contacts, String> {
        let now = a.contacts()?;
        let parts = a.world_parts();
        let statics = self.raw("solver_scratch.bp.sweep.sap_static_id");
        // (handle 0 is the broadphase's unused first handle: its id reads 0 too, which is also the first mesh's)
        let static_of = |id: i16| -> Option<BodyRef> {
            let h = statics.chunks_exact(2).enumerate().skip(1).find(|(_, c)| i16::from_le_bytes([c[0], c[1]]) == id)?.0;
            parts.iter().position(|p| p.handle as usize == h).map(BodyRef::World)
        };
        let car = |raw: u32| -> Option<BodyRef> {
            let k = raw as usize;
            (k >= 1 && k <= nc && !demoed[k - 1]).then(|| BodyRef::Car(k - 1))
        };
        // (body0, body1, child) of a key's manifold, oriented as catSim's algorithms make it
        let decode = |key: u32| -> Option<(BodyRef, BodyRef, (i32, i32))> {
            let (kind, ra, rb, sid) = (key & 3, (key >> 2) & 15, (key >> 6) & 15, (key >> 16) as u16 as i16);
            match kind {
                0 => Some((BodyRef::Ball, static_of(sid)?, (-1, -1))),
                1 => Some((car(ra)?, BodyRef::Ball, (0, -1))),
                2 => Some((car(ra)?, static_of(sid)?, (0, -1))),
                _ => {
                    let (x, y) = (car(ra)?, car(rb)?);
                    let (hx, hy) = (a.broadphase_handle(x)?, a.broadphase_handle(y)?);
                    Some(if hx < hy { (x, y, (0, 0)) } else { (y, x, (0, 0)) })
                }
            }
        };
        let mut pairs: Vec<Pair> = now.pairs.iter().copied().filter(|p| matches!((p.body0, p.body1), (BodyRef::World(_), BodyRef::World(_)))).collect();
        let np = self.whole_u("solver_scratch.bp.bp_pair_count") as u8 as usize;
        let keys = self.u32s("solver_scratch.bp.bp_pairs");
        let mut pair_of_key = std::collections::HashMap::new();
        for &k in keys.iter().take(np) {
            let Some((x, y, _)) = decode(k) else { continue };
            pair_of_key.insert(k, pairs.len());
            pairs.push(Pair { body0: x, body1: y, dispatched: false });
        }
        // a pair catSim's broadphase holds and the state lacks stays, after the state's (its boxes overlap in catSim)
        let same = |p: &Pair, q: &Pair| (p.body0, p.body1) == (q.body0, q.body1) || (p.body0, p.body1) == (q.body1, q.body0);
        for p in &now.pairs {
            if !pairs.iter().any(|q| same(p, q)) {
                pairs.push(Pair { dispatched: false, ..*p });
            }
        }
        let npred = self.whole_u("solver_scratch.bp.bp_predictive_count") as u8 as usize;
        let pred = self.u32s("solver_scratch.bp.bp_predictive");
        let nm = self.whole_u("solver_scratch.bp.bp_manifold_count") as u8 as usize;
        let mut manifolds = Vec::new();
        let mut used_pred = vec![false; npred];
        for &key in self.u32s("solver_scratch.bp.bp_manifolds").iter().take(nm) {
            if (key >> 16) as u16 == 99 {
                let Some(k) = (0..npred.min(pred.len())).find(|&k| pred[k] == key && !used_pred[k]) else { continue };
                let body = match key & 3 {
                    0 | 1 => Some(BodyRef::Ball),
                    _ => car((key >> 2) & 15),
                };
                let Some(body) = body else { continue };
                used_pred[k] = true;
                manifolds.push(Manifold { owner: ManifoldOwner::Predictive(k), body0: body, body1: BodyRef::World(0), points: Vec::new() });
                continue;
            }
            let (Some(&pair), Some((b0, b1, child))) = (pair_of_key.get(&key), decode(key)) else { continue };
            let points = self.points(a, key, b0, b1);
            manifolds.push(Manifold { owner: ManifoldOwner::Pair { pair, child }, body0: b0, body1: b1, points });
        }
        // the predictive contacts are numbered in the state's order; one that didn't map leaves no gap
        let mut remap = vec![usize::MAX; npred];
        let mut n = 0;
        for (k, u) in used_pred.iter().enumerate() {
            if *u {
                remap[k] = n;
                n += 1;
            }
        }
        for m in manifolds.iter_mut() {
            if let ManifoldOwner::Predictive(k) = m.owner {
                m.owner = ManifoldOwner::Predictive(remap[k]);
            }
        }
        Ok(Contacts { boxes: now.boxes, pairs, manifolds })
    }

    /// One manifold's cached points from RocketGPU's per-kind storage.
    fn points(&self, a: &Arena, key: u32, b0: BodyRef, b1: BodyRef) -> Vec<catsim::bt::ManifoldPoint> {
        let (kind, ra, rb, sid) = (key & 3, ((key >> 2) & 15) as usize, ((key >> 6) & 15) as usize, (key >> 16) as u16 as i16);
        let mut out = Vec::new();
        let body = |r: BodyRef| a.body_id(r).map(|id| a.world.cw.bodies[id.0 as usize].transform);
        let (Some(t0), Some(t1)) = (body(b0), body(b1)) else { return out };
        let mut push = |la: Vec3, lb: Vec3, n: Vec3, d: f32| {
            if let Some(p) = a.contact_point(b0, b1, la, lb, n, d) {
                out.push(p);
            }
        };
        let fv = |name: &str| self.whole_f(name);
        let cnt_of = |name: &str, slot: usize| self.raw(name).get(slot).copied().unwrap_or(0) as usize;
        let at = |v: &[f32], i: usize| v3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
        match kind {
            // the ball and a plane or a mesh: the points' A is on the ball, B on the arena part
            0 => {
                let (pre, slot) = if sid >= 100 {
                    ("solver_scratch.ball_world_plane_manifolds", [0usize, 1, 3, 2, 4, 5].get((sid - 100) as usize).copied())
                } else {
                    ("solver_scratch.ball_world_mesh_manifolds", Some(sid as usize))
                };
                let Some(slot) = slot else { return out };
                let cnt = cnt_of(&format!("{pre}.count"), slot);
                let (la, lb, nn, d) = (fv(&format!("{pre}.points.local_point_a_bt")), fv(&format!("{pre}.points.local_point_b_bt")), fv(&format!("{pre}.points.normal")), fv(&format!("{pre}.points.distance_bt")));
                for i in 0..cnt.min(4) {
                    let q = slot * 4 + i;
                    if 3 * q + 2 < la.len() && q < d.len() {
                        if sid >= 100 {
                            // a plane's point on B is in world axes in the state: into the plane's own frame (catSim's
                            // planes are turned bodies), and the plane's normal, as catSim's convex-plane contact makes them
                            let r = &t1.basis.r;
                            let w = at(&lb, q);
                            let (dx, dy, dz) = (w.x - t1.origin.x, w.y - t1.origin.y, w.z - t1.origin.z);
                            let local = v3(dy * r[1].x + dx * r[0].x + dz * r[2].x, dy * r[1].y + dx * r[0].y + dz * r[2].y, dx * r[0].z + dy * r[1].z + dz * r[2].z);
                            let n = v3(r[0].y * 0.0 + r[0].x * 0.0 + r[0].z, r[1].y * 0.0 + r[1].x * 0.0 + r[1].z, r[2].y * 0.0 + r[2].x * 0.0 + r[2].z);
                            push(at(&la, q), local, n, d[q]);
                            continue;
                        }
                        push(at(&la, q), at(&lb, q), at(&nn, q), d[q]);
                    }
                }
            }
            // a car and the ball, or two cars: A and B as the state stores them, matched to catSim's body0 and body1 by
            // where A's local point lands (pos_a_bt is A's point in the world). The normal must point from B to A (a
            // point on A is its point on B plus normal times distance): the state's car-ball normals do, its car-car
            // ones point the other way (over the suite's starts, by the sign of normal . (pos_a - pos_b) * distance:
            // 531 of 531 car-ball points, and the opposite at 126 of 126 car-car ones)
            1 | 3 => {
                let (pre, slot) = if kind == 1 {
                    ("ball_car", ra - 1)
                } else {
                    let (lo, hi) = ((ra - 1).min(rb - 1), (ra - 1).max(rb - 1));
                    ("car_car", hi * (hi.max(1) - 1) / 2 + lo)
                };
                let cnt = cnt_of(&format!("{pre}.count"), slot);
                let (la, lb, nn, d, pa) = (fv(&format!("{pre}.points.local_a_bt")), fv(&format!("{pre}.points.local_b_bt")), fv(&format!("{pre}.points.normal")), fv(&format!("{pre}.points.distance_bt")), fv(&format!("{pre}.points.pos_a_bt")));
                for i in 0..cnt.min(4) {
                    let q = slot * 4 + i;
                    if 3 * q + 2 >= la.len() || 3 * q + 2 >= pa.len() || q >= d.len() {
                        continue;
                    }
                    let (x, y, n, p) = (at(&la, q), at(&lb, q), at(&nn, q), at(&pa, q));
                    let n = if kind == 3 { v3(-n.x, -n.y, -n.z) } else { n };
                    let gap = |t: &catsim::math::Transform, l: Vec3| {
                        let r = &t.basis.r;
                        let w = v3(r[0].x * l.x + r[0].y * l.y + r[0].z * l.z + t.origin.x, r[1].x * l.x + r[1].y * l.y + r[1].z * l.z + t.origin.y, r[2].x * l.x + r[2].y * l.y + r[2].z * l.z + t.origin.z);
                        (w.x - p.x).powi(2) + (w.y - p.y).powi(2) + (w.z - p.z).powi(2)
                    };
                    if gap(&t0, x) <= gap(&t1, x) {
                        push(x, y, n, d[q]);
                    } else {
                        push(y, x, v3(-n.x, -n.y, -n.z), d[q]);
                    }
                }
            }
            // a car and an arena part
            _ => {
                let c = ra - 1;
                if sid >= 100 {
                    let slot = c * 6 + (sid - 100) as usize;
                    let cnt = cnt_of("car_world_plane.count", slot);
                    let (la, pb, d) = (fv("car_world_plane.points.local_point_a_bt"), fv("car_world_plane.points.point_on_b_bt"), fv("car_world_plane.points.distance_bt"));
                    let r = &t1.basis.r;
                    // the plane's normal, and the world point in the plane's frame, as catSim's convex-plane contact makes them
                    let n = v3(r[0].y * 0.0 + r[0].x * 0.0 + r[0].z, r[1].y * 0.0 + r[1].x * 0.0 + r[1].z, r[2].y * 0.0 + r[2].x * 0.0 + r[2].z);
                    for i in 0..cnt.min(4) {
                        let q = slot * 4 + i;
                        if 3 * q + 2 >= la.len() || 3 * q + 2 >= pb.len() || q >= d.len() {
                            continue;
                        }
                        let w = at(&pb, q);
                        let (dx, dy, dz) = (w.x - t1.origin.x, w.y - t1.origin.y, w.z - t1.origin.z);
                        let lb = v3(dy * r[1].x + dx * r[0].x + dz * r[2].x, dy * r[1].y + dx * r[0].y + dz * r[2].y, dx * r[0].z + dy * r[1].z + dz * r[2].z);
                        push(at(&la, q), lb, n, d[q]);
                    }
                } else {
                    let cnts = self.raw("car_world_mesh.count");
                    let mesh = self.raw("car_world_mesh.mesh");
                    let Some(slot) = (c * 2..c * 2 + 2).find(|&j| cnts.get(j).copied().unwrap_or(0) > 0 && mesh.get(j).copied() == Some(sid as u8)) else { return out };
                    let (la, lb, nn, d) = (fv("car_world_mesh.points.local_point_a_bt"), fv("car_world_mesh.points.local_point_b_bt"), fv("car_world_mesh.points.normal"), fv("car_world_mesh.points.distance_bt"));
                    for i in 0..(cnts[slot] as usize).min(4) {
                        let q = slot * 4 + i;
                        if 3 * q + 2 < la.len() && q < d.len() {
                            push(at(&la, q), at(&lb, q), at(&nn, q), d[q]);
                        }
                    }
                }
            }
        }
        out
    }

    fn car_hidden(&self, a: &Arena, k: usize, h: &mut CarHidden, phys: i64, ids: &[u64]) {
        let u = |n: &str| self.u8(n, k) != 0;
        let f = |n: &str| self.f1(n, k);
        // the game's own flags as the suite calibrates them
        h.jumped = self.u8("wheel_runtime.cal_game_jumped", k) != 0;
        h.double_jumped = self.u8("wheel_runtime.cal_game_double_jumped", k) != 0;
        h.on_ground = u("car_states.is_on_ground");
        h.supersonic = u("car_states.internal_is_supersonic") || u("car_states.is_supersonic");
        h.supersonic_time = f("car_states.supersonic_time");
        h.boost_amount = f("car_states.boost") / 100.0;
        h.boost.active = u("car_states.is_boosting");
        h.boost.time = if h.boost.active { f("car_states.boosting_time") } else { f("car_states.time_since_boosted") };
        h.jump.active = u("car_states.is_jumping");
        h.jump.time = f("car_states.jump_time");
        h.jump_released = u("car_states.jump_hold_broken");
        // the double jump's clock is not in the state; only the dodge window reads it, as the least of three clocks
        h.double_jump.time = 10.0;
        h.dodge.active = u("car_states.is_flipping");
        h.dodge.time = f("car_states.flip_time");
        let rel = self.f("car_states.flip_rel_torque", k);
        if rel.len() >= 2 {
            let c = a.players[k].car.as_ref().unwrap();
            let d = &c.dodge;
            let round = catsim::math::rot::app_round;
            h.dodge_torque = v3(round(rel[0] * d.side_torque * d.torque_scale) as f32, round(rel[1] * d.forward_torque * d.torque_scale) as f32, 0.0);
            h.dodge_direction = v3(rel[1], -rel[0], 0.0);
        }
        h.auto_flip.active = u("car_states.is_auto_flipping");
        if h.auto_flip.active {
            h.auto_flip_right = f("car_states.auto_flip_torque_scale") > 0.0;
        }
        h.handbrake = f("car_states.handbrake_val");
        let lc = self.slot("car_states.last_controls", k);
        if lc.len() >= 23 {
            for i in 0..5 {
                h.held_input[i] = f32::from_le_bytes(lc[4 * i..4 * i + 4].try_into().unwrap());
            }
            h.held_input[5] = lc[20] as f32;
            h.held_input[6] = lc[21] as f32;
            h.held_input[7] = lc[22] as f32;
        }
        let wc = self.slot("car_states.wheels_with_contact", k);
        let nrm = self.f("car_states.set_wheel_normals", k);
        for w in 0..4 {
            let q = SUITE_WHEEL[w];
            h.wheels[w].contact = wc.get(q).copied().unwrap_or(0) != 0;
            if nrm.len() >= 12 {
                h.wheels[w].normal = Self::vec(&nrm, 3 * q);
            }
        }
        h.world_contact = u("car_states.world_contact.has_contact");
        h.world_contact_normal = Self::vec(&self.f("car_states.world_contact.contact_normal", k), 0);
        let t_imp = self.u64("car_states.ball_hit_info.tick_count_when_extra_impulse_applied", k);
        h.ball_impact_frame = if t_imp < (1 << 31) { t_imp as i32 } else { -1_000_000 };
        let t_touch = self.u64("car_states.latest_touch_tick", k);
        h.ball_touch_frame = if t_touch < (1 << 31) { t_touch as i32 } else { -1_000_000 };
        let other = self.u32("car_states.car_contact.other_car_id", k) as u64;
        let cd = f("car_states.car_contact.cooldown_timer");
        if cd > 0.0 {
            if let Some(v) = ids.iter().position(|&i| i == other) {
                h.bump_victim = Some(v);
                // the state's cooldowns all sit half a tick off catSim's grid (460 of 460 over the suite): the gate
                // opens on the tick after the one rounding to the nearest would give
                let ticks_ago = ((0.25 - cd) * 120.0 - 0.5).round() as i64;
                h.bump_time = (1.0 / 120.0) * (phys - ticks_ago) as f32;
            }
        }
    }
}
