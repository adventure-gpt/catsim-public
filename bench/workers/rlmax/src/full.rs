//! The RocketGPU full-state suite's start state in RLMAX's arena, through its own loader (its crate's test-only feature
//! `full_state`, built here with this worker's feature of the same name). The start's public row builds the arena; then what
//! the suite records beyond it goes straight in, in Bullet units: the bodies, each car's hidden scalars, the wheels' contacts,
//! the contact manifolds, the visit order and the broadphase. The suite's own layout is RocketGPU's state fields.

use rlmax::cpu::{Arena, ArenaData};
use serde_json::Value;

const SLOTS: usize = 8;
const SW_BALL: u32 = 23; // RLMAX's broadphase handle of the ball; car k is SW_BALL + 1 + k; statics: meshes 1-16, planes 17-22

pub struct Full<'a> {
    index: &'a Value,
    data: &'a [u8],
}

fn fl(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// A body's frame: position and basis columns (forward, right, up).
#[derive(Clone, Copy)]
struct Frame {
    p: [f32; 3],
    c: [[f32; 3]; 3],
}

impl Frame {
    fn world(&self, l: [f32; 3]) -> [f32; 3] {
        let mut o = self.p;
        for k in 0..3 {
            for (a, col) in self.c.iter().enumerate() {
                o[k] += col[k] * l[a];
            }
        }
        o
    }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum()
}

fn v3at(v: &[f32], q: usize) -> [f32; 3] {
    [v[3 * q], v[3 * q + 1], v[3 * q + 2]]
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
        fl(self.slot(name, k))
    }

    fn f1(&self, name: &str, k: usize) -> f32 {
        self.f(name, k).first().copied().unwrap_or(0.0)
    }

    fn u8(&self, name: &str, k: usize) -> u8 {
        self.slot(name, k).first().copied().unwrap_or(0)
    }

    fn all(&self, name: &str) -> Vec<f32> {
        fl(self.raw(name))
    }

    fn u(&self, name: &str) -> u64 {
        let r = self.raw(name);
        let mut b = [0u8; 8];
        b[..r.len().min(8)].copy_from_slice(&r[..r.len().min(8)]);
        u64::from_le_bytes(b)
    }

    fn u32s(&self, name: &str) -> Vec<u32> {
        self.raw(name).chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn car_frame(&self, k: usize) -> Frame {
        let g = |n: &str| -> [f32; 3] { let v = self.f(n, k); [v[0], v[1], v[2]] };
        Frame { p: g("car_states.pos_bt"), c: [g("car_states.basis_forward"), g("car_states.basis_right"), g("car_states.basis_up")] }
    }

    fn ball_frame(&self) -> Frame {
        let g = |n: &str| -> [f32; 3] { let v = self.all(n); [v[0], v[1], v[2]] };
        Frame { p: g("ball_states.pos_bt"), c: [g("ball_states.basis_forward"), g("ball_states.basis_right"), g("ball_states.basis_up")] }
    }

    /// Everything beyond the public row, into an arena built from it.
    pub fn apply(&self, a: &mut Arena, data: &ArenaData, req: &Value) -> Result<(), String> {
        let nc = a.num_cars as usize;
        let ids: Vec<u64> = req.get("ids").and_then(Value::as_array).map(|v| v.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
        let demoed: Vec<bool> = (0..nc).map(|k| self.u8("car_states.is_demoed", k) != 0).collect();
        // the recorded spawn choices: 1-4 a respawn slot, 5 on a kickoff spot
        let mut over = vec![0f32; nc];
        for call in req.get("calls").and_then(Value::as_array).into_iter().flatten() {
            let id = call[1][0].as_u64().unwrap_or(0);
            let spot = call[1][1].as_i64().unwrap_or(-1);
            let Some(k) = ids.iter().position(|&x| x == id) else { continue };
            if k >= nc || spot < 0 {
                continue;
            }
            match call[0].as_str() {
                Some("set_respawn_slot_override") => over[k] = (spot + 1) as f32,
                Some("set_kickoff_spawn_override") => over[k] = (spot + 5) as f32,
                _ => {}
            }
        }
        let tick = self.u("env_states.tick") as f32;
        for k in 0..nc {
            if let Some(z) = req["cars"][k]["config"].get("spawn_z").and_then(Value::as_f64) {
                a.fs_spawn_z(k as u32, z as f32);
            }
            if !demoed[k] {
                let mut b = Vec::with_capacity(28);
                for n in ["car_states.pos_bt", "car_states.vel_bt", "car_states.internal_ang_vel", "car_states.basis_forward", "car_states.basis_right", "car_states.basis_up"] {
                    b.extend(self.f(n, k));
                }
                let inv = self.f("car_states.restored_inv_inertia_bt", k);
                let valid = self.u8("car_states.restored_inv_inertia_valid", k) != 0 && inv.len() == 9;
                b.extend(if valid { inv } else { vec![0.0; 9] });
                b.push(valid as u32 as f32);
                // (its loader seeds the basis; its quaternion comes from the basis too)
                let m: [f32; 9] = b[9..18].try_into().unwrap();
                a.set_body_basis(k as u32, &m, None);
                a.fs_car(k as u32, &b);
            }
            a.fs_jump(k as u32, self.u8("wheel_runtime.cal_game_jumped", k) as f32, self.u8("wheel_runtime.cal_game_double_jumped", k) as f32,
                      self.f1("wheel_runtime.cal_jump_inactive_time", k), self.u8("wheel_runtime.cal_cached_ground", k) as f32);
            let n = self.f("car_states.world_contact.contact_normal", k);
            let lc = self.slot("car_states.last_controls", k);
            let other = u32::from_le_bytes(self.slot("car_states.car_contact.other_car_id", k)[..4].try_into().unwrap()) as u64;
            let imp = u64::from_le_bytes(self.slot("car_states.ball_hit_info.tick_count_when_extra_impulse_applied", k)[..8].try_into().unwrap());
            let has_hit = self.u8("car_states.ball_hit_info.is_valid", k) != 0 && imp != u64::MAX;
            let e = [self.u8("car_states.world_contact.has_contact", k) as f32, n[0], n[1], n[2], lc.get(20).copied().unwrap_or(0) as f32,
                     self.f1("car_states.car_contact.cooldown_timer", k), ids.iter().position(|&x| x == other).unwrap_or(0) as f32, over[k], tick,
                     has_hit as u32 as f32, if has_hit { imp as f32 } else { 0.0 }];
            a.fs_car_extra(k as u32, &e);
        }
        let mut b = Vec::with_capacity(30);
        for n in ["ball_states.pos_bt", "ball_states.vel_bt", "ball_states.ang_vel", "ball_states.basis_forward", "ball_states.basis_right", "ball_states.basis_up"] {
            b.extend(self.all(n));
        }
        let inv = self.all("ball_states.restored_inv_inertia_bt");
        let valid = self.u("ball_states.restored_inv_inertia_valid") as u32 != 0 && inv.len() == 9;
        b.extend(if valid { inv } else { vec![0.0; 9] });
        b.push(valid as u32 as f32);
        b.push(self.u("ball_states.sleeping") as u32 as f32);
        b.push(self.all("ball_states.deactivation_time").first().copied().unwrap_or(0.0));
        let m: [f32; 9] = b[9..18].try_into().unwrap();
        a.set_body_basis(nc as u32, &m, None);
        a.fs_ball(&b);
        for k in 0..nc {
            if demoed[k] {
                continue;
            }
            // the wheels' contacts of the last tick, in RLMAX's wheel order (the suite's has left and right swapped)
            let mask = self.u8("car_states.set_wheel_normals_mask", k) as u32;
            let touch = self.slot("car_states.wheels_with_contact", k);
            let nrm = self.f("car_states.set_wheel_normals", k);
            let mut bits = if mask & 0x20 != 0 { 0x20 } else { 0 };
            let mut normals = [[0f32; 3]; 4];
            for w in 0..4 {
                let q = w ^ 1;
                let on = if mask & 0x30 != 0 { mask >> q & 1 != 0 } else { touch.get(q).copied().unwrap_or(0) != 0 };
                if on {
                    bits |= 1 << w;
                }
                normals[w] = v3at(&nrm, q);
            }
            a.seed_wheel_contacts(data, k as u32, bits, &normals);
        }
        self.manifolds(a, nc, &demoed)?;
        if a.fs_visit(self.u("env_states.car_visit_order_packed") as u32, (0..nc).fold(0u32, |m, k| if demoed[k] { m } else { m | 1 << k })) != 0 {
            return Err("the recorded visit order doesn't fit".into());
        }
        self.sweep(a, nc, &demoed)
    }

    /// RLMAX's key of a recorded pair or manifold key (a kind, two bodies, a static id). None: not a pair RLMAX keeps.
    fn key_of(&self, key: u32, nc: usize, demoed: &[bool]) -> Option<f32> {
        let (kind, ra, rb, sid) = (key & 3, (key >> 2) & 15, (key >> 6) & 15, (key >> 16) as u16 as i16);
        let stat = |s: i16| -> Option<u32> { if s >= 100 { Some(17 + (s - 100) as u32) } else if s >= 0 { Some(s as u32 + 1) } else { None } };
        let car = |r: u32| -> Option<u32> { let k = r as usize; (k >= 1 && k <= nc && !demoed[k - 1]).then_some(SW_BALL + r) };
        let (x, y) = match kind {
            0 => (SW_BALL, stat(sid)?),
            1 => (car(ra)?, SW_BALL),
            2 => (car(ra)?, stat(sid)?),
            _ => (car(ra)?, car(rb)?),
        };
        let (lo, hi) = (x.min(y), x.max(y));
        let stride = if nc == 8 { 31 } else { 29 };
        Some(((hi - SW_BALL) * stride + lo) as f32)
    }

    fn sweep(&self, a: &mut Arena, nc: usize, demoed: &[bool]) -> Result<(), String> {
        let pos: Vec<u16> = self.raw("solver_scratch.bp.sweep.sap_edge_pos").chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let hnd = self.raw("solver_scratch.bp.sweep.sap_edge_handle");
        let per = hnd.len() / 3;
        let body_h = self.raw("solver_scratch.bp.sweep.sap_body_handle");
        let free = self.raw("solver_scratch.bp.sweep.sap_free");
        let free_n = (self.u("solver_scratch.bp.sweep.sap_free_count") as u8 as usize).min(free.len()).min(1 + nc);
        let np = self.u("solver_scratch.bp.bp_pair_count") as u8 as usize;
        let keys = self.u32s("solver_scratch.bp.bp_pairs");
        let pairs: Vec<f32> = keys.iter().take(np).filter_map(|&k| self.key_of(k, nc, demoed)).collect();
        // the manifolds; the last tick's predictive ones (static id 99) go as the next tick's sweep lets them go: ball first,
        // then the cars in the visit order, each by swap-with-last
        let nm = self.u("solver_scratch.bp.bp_manifold_count") as u8 as usize;
        let mut mans: Vec<(u32, u32)> = self.u32s("solver_scratch.bp.bp_manifolds").iter().take(nm).map(|&k| (k, 0)).collect();
        let packed = self.u("env_states.car_visit_order_packed") as u32;
        let mut order: Vec<u32> = vec![0];
        for q in 0..8 {
            let s = packed >> (4 * q) & 15;
            if (s as usize) < nc {
                order.push(s + 1);
            }
        }
        for b in order {
            if let Some(i) = mans.iter().position(|&(k, _)| (k >> 16) as u16 == 99 && (if k & 3 < 2 { 0 } else { (k >> 2) & 15 }) == b) {
                let last = mans.pop().unwrap();
                if i < mans.len() {
                    mans[i] = last;
                }
            }
        }
        let man: Vec<f32> = mans.iter().filter_map(|&(k, _)| self.key_of(k, nc, demoed)).collect();
        let live = (0..=nc).all(|b| body_h[b] != 0);
        let mut f = vec![(pairs.len() + if live { 0 } else { 256 }) as f32, man.len() as f32, free_n as f32];
        for i in 0..1 + nc {
            f.push(if i < free_n { free[i] as f32 } else { 0.0 });
        }
        for b in 0..=nc {
            let h = body_h[b];
            if h == 0 {
                f.extend([255.0; 6]);
                f.extend([0.0; 6]);
                continue;
            }
            a.seed_handle(b as u32, h as u32);
            let (mut places, mut at) = ([0f32; 6], [0f32; 6]);
            for ax in 0..3 {
                for i in 0..per {
                    if hnd[ax * per + i] == h {
                        let p = pos[ax * per + i];
                        let max = (p & 1) as usize;
                        places[ax * 2 + max] = i as f32;
                        at[ax * 2 + max] = p as f32;
                    }
                }
            }
            f.extend(places);
            f.extend(at);
        }
        f.extend(pairs);
        f.extend(man);
        match a.fs_sweep(&f) {
            0 => Ok(()),
            e => Err(format!("the recorded broadphase doesn't fit ({e})")),
        }
    }

    /// The contact manifolds with their cached points, in the dispatcher's order.
    fn manifolds(&self, a: &mut Arena, nc: usize, demoed: &[bool]) -> Result<(), String> {
        let nm = self.u("solver_scratch.bp.bp_manifold_count") as u8 as usize;
        let ball = self.ball_frame();
        let cnt_of = |name: &str, s: usize| self.raw(name).get(s).copied().unwrap_or(0) as usize;
        for &key in self.u32s("solver_scratch.bp.bp_manifolds").iter().take(nm) {
            if (key >> 16) as u16 == 99 {
                continue;
            }
            let (kind, ra, rb, sid) = (key & 3, ((key >> 2) & 15) as usize, ((key >> 6) & 15) as usize, (key >> 16) as u16 as i16);
            let mut pts = Vec::new();
            let mut put = |la: [f32; 3], lb: [f32; 3], n: [f32; 3], d: f32| {
                pts.extend(la);
                pts.extend(lb);
                pts.extend(n);
                pts.push(d);
            };
            let (rk, ra_out, obj);
            match kind {
                0 => {
                    let (pre, s) = if sid >= 100 {
                        ("solver_scratch.ball_world_plane_manifolds", [0usize, 1, 3, 2, 4, 5][(sid - 100) as usize])
                    } else {
                        ("solver_scratch.ball_world_mesh_manifolds", sid as usize)
                    };
                    let (la, lb, nn, d) = (self.all(&format!("{pre}.points.local_point_a_bt")), self.all(&format!("{pre}.points.local_point_b_bt")),
                                           self.all(&format!("{pre}.points.normal")), self.all(&format!("{pre}.points.distance_bt")));
                    for i in 0..cnt_of(&format!("{pre}.count"), s).min(4) {
                        let q = s * 4 + i;
                        put(v3at(&la, q), v3at(&lb, q), v3at(&nn, q), d[q]);
                    }
                    (rk, ra_out, obj) = (1, 0, if sid >= 100 { 0x100 + (sid - 100) as u32 } else { sid as u32 });
                }
                1 => {
                    let c = ra - 1;
                    if demoed[c] {
                        continue;
                    }
                    let car = self.car_frame(c);
                    let (la, lb, nn, d, pa) = (self.all("ball_car.points.local_a_bt"), self.all("ball_car.points.local_b_bt"), self.all("ball_car.points.normal"),
                                               self.all("ball_car.points.distance_bt"), self.all("ball_car.points.pos_a_bt"));
                    for i in 0..cnt_of("ball_car.count", c).min(4) {
                        let q = c * 4 + i;
                        let (x, y, n, p) = (v3at(&la, q), v3at(&lb, q), v3at(&nn, q), v3at(&pa, q));
                        // RLMAX's A is the ball: the state's A is told by where its local point lands
                        if dist2(ball.world(x), p) <= dist2(car.world(x), p) {
                            put(x, y, n, d[q]);
                        } else {
                            put(y, x, [-n[0], -n[1], -n[2]], d[q]);
                        }
                    }
                    (rk, ra_out, obj) = (3, c as u32, 0);
                }
                2 => {
                    let c = ra - 1;
                    if demoed[c] {
                        continue;
                    }
                    if sid >= 100 {
                        let s = c * 6 + (sid - 100) as usize;
                        let (la, pb, d) = (self.all("car_world_plane.points.local_point_a_bt"), self.all("car_world_plane.points.point_on_b_bt"), self.all("car_world_plane.points.distance_bt"));
                        for i in 0..cnt_of("car_world_plane.count", s).min(4) {
                            let q = s * 4 + i;
                            put(v3at(&la, q), v3at(&pb, q), [0.0; 3], d[q]);
                        }
                        (rk, ra_out, obj) = (0, c as u32, 0x100 + (sid - 100) as u32);
                    } else {
                        let (cnts, mesh) = (self.raw("car_world_mesh.count"), self.raw("car_world_mesh.mesh"));
                        let s = (c * 2..c * 2 + 2).find(|&j| cnts.get(j).copied().unwrap_or(0) > 0 && mesh.get(j).copied() == Some(sid as u8));
                        if let Some(s) = s {
                            let (la, lb, nn, d) = (self.all("car_world_mesh.points.local_point_a_bt"), self.all("car_world_mesh.points.local_point_b_bt"),
                                                   self.all("car_world_mesh.points.normal"), self.all("car_world_mesh.points.distance_bt"));
                            for i in 0..(cnts[s] as usize).min(4) {
                                let q = s * 4 + i;
                                put(v3at(&la, q), v3at(&lb, q), v3at(&nn, q), d[q]);
                            }
                        }
                        (rk, ra_out, obj) = (0, c as u32, sid as u32);
                    }
                }
                _ => {
                    let (lo, hi) = ((ra - 1).min(rb - 1), (ra - 1).max(rb - 1));
                    if demoed[lo] || demoed[hi] {
                        continue;
                    }
                    let s = hi * (hi.max(1) - 1) / 2 + lo;
                    let body_h = self.raw("solver_scratch.bp.sweep.sap_body_handle");
                    // RLMAX's A is the pair's higher broadphase handle; the state's A is told by where its local point lands
                    let ia = if body_h[1 + lo] > body_h[1 + hi] { lo } else { hi };
                    let ib = lo + hi - ia;
                    let (fa, fb) = (self.car_frame(ia), self.car_frame(ib));
                    let (la, lb, nn, d, pa) = (self.all("car_car.points.local_a_bt"), self.all("car_car.points.local_b_bt"), self.all("car_car.points.normal"),
                                               self.all("car_car.points.distance_bt"), self.all("car_car.points.pos_a_bt"));
                    for i in 0..cnt_of("car_car.count", s).min(4) {
                        let q = s * 4 + i;
                        let (x, y, n, p) = (v3at(&la, q), v3at(&lb, q), v3at(&nn, q), v3at(&pa, q));
                        if dist2(fa.world(x), p) <= dist2(fb.world(x), p) {
                            put(x, y, n, d[q]);
                        } else {
                            put(y, x, [-n[0], -n[1], -n[2]], d[q]);
                        }
                    }
                    // RLMAX numbers its pairs (i < j) row by row
                    let p = lo * (2 * nc - lo - 1) / 2 + (hi - lo - 1);
                    (rk, ra_out, obj) = (2, p as u32, 0);
                }
            }
            // (an empty manifold has no observable effect but its place in the order, which the sweep's keys hold; RLMAX
            // releases its own empty ones)
            if pts.is_empty() {
                continue;
            }
            if a.fs_manifold(rk, ra_out, obj, &pts) != 0 {
                return Err(format!("a recorded manifold doesn't fit (kind {rk}, {ra_out}, {obj})"));
            }
        }
        Ok(())
    }
}
