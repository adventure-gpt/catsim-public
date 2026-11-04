use crate::bt::world::{IntegrateMemo, integrate_key_of};
use crate::bt::{CF_KINEMATIC, CF_STATIC, DISABLE_SIMULATION, ISLAND_SLEEPING};
use crate::game::arena::Arena;
use crate::game::rb_state::{RbState, read_key};
use crate::game::rules;
use crate::math::crt_pow::powf_memo;
use crate::math::{Mat3, Quat, Transform, Vec3, v3};
use std::sync::atomic::{AtomicU8, Ordering};

pub mod f8;
use f8::{F8, M8};

pub const GROUP: usize = 8;

static LANES: AtomicU8 = AtomicU8::new(0);

pub fn lanes_on() -> bool {
    match LANES.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = f8::available() && std::env::var("CATSIM_GROUP_LANES").map_or(true, |v| v != "0");
            LANES.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

pub fn set_lanes(on: bool) {
    LANES.store(if on && f8::available() { 1 } else { 2 }, Ordering::Relaxed);
}

thread_local! {
    static STATS: core::cell::Cell<[u64; 6]> = const { core::cell::Cell::new([0; 6]) };
}

#[inline(always)]
fn count(k: usize, tried: u32, good: u32) {
    STATS.with(|c| {
        let mut v = c.get();
        v[2 * k] += u64::from(tried.count_ones());
        v[2 * k + 1] += u64::from(good.count_ones());
        c.set(v);
    });
}

pub fn lane_stats() -> [u64; 6] {
    STATS.with(|c| c.get())
}

static PLANT: AtomicU8 = AtomicU8::new(0);

pub fn plant_lane_bug(kernel: u8) {
    PLANT.store(kernel, Ordering::Relaxed);
}

#[inline(always)]
fn planted(k: u8, x: f32) -> f32 {
    if PLANT.load(Ordering::Relaxed) == k { f32::from_bits(x.to_bits() ^ 1) } else { x }
}

pub fn frames(arenas: &mut [&mut Arena], steps: i32) {
    let lanes = lanes_on();
    for chunk in arenas.chunks_mut(GROUP) {
        frames_chunk(chunk, steps, lanes);
    }
}

fn frames_chunk(a: &mut [&mut Arena], steps: i32, lanes: bool) {
    let n = a.len();
    let mut ft = [0.0f32; GROUP];
    for j in 0..n {
        ft[j] = a[j].dt * steps as f32;
        let alpha = a[j].render_alpha;
        a[j].frame_begin(&[], steps, ft[j], alpha);
    }
    for _ in 0..steps {
        tick(a, lanes);
    }
    for j in 0..n {
        a[j].frame_end(ft[j]);
    }
}

fn tick(a: &mut [&mut Arena], lanes: bool) {
    let n = a.len();
    let mut ok = [false; GROUP];
    for j in 0..n {
        ok[j] = a[j].group_ok();
        if !ok[j] {
            a[j].tick(&[]);
        }
    }
    let lanes = lanes && ok[..n].iter().filter(|x| **x).count() >= 2;
    for j in 0..n {
        if ok[j] {
            a[j].group_tick_pre();
        }
    }
    for j in 0..n {
        if ok[j] {
            a[j].world.stage_gravity();
        }
    }
    if lanes {
        unsafe { predict_lanes(a, &ok) };
    }
    for j in 0..n {
        if ok[j] {
            let dt = a[j].dt;
            a[j].world.stage_predict(dt);
        }
    }
    for j in 0..n {
        if ok[j] {
            let dt = a[j].dt;
            a[j].world.stage_ccd(dt);
        }
    }
    for j in 0..n {
        if ok[j] {
            let held = a[j].hook_held();
            let mut hook = |cp: &mut crate::bt::types::ManifoldPoint, x: &crate::bt::manifold::Wrap, y: &crate::bt::manifold::Wrap| rules::contact_added(cp, x, y, held);
            a[j].world.stage_collide(Some(&mut hook));
        }
    }
    for j in 0..n {
        if ok[j] {
            let dt = a[j].dt;
            a[j].world.stage_solve(dt);
        }
    }
    if lanes {
        unsafe { integrate_lanes(a, &ok) };
    }
    for j in 0..n {
        if ok[j] {
            let dt = a[j].dt;
            a[j].world.stage_integrate(dt);
        }
    }
    for j in 0..n {
        if ok[j] {
            let dt = a[j].dt;
            a[j].world.stage_tail(dt);
        }
    }
    if lanes {
        unsafe { read_lanes(a, &ok) };
    }
    for j in 0..n {
        if ok[j] {
            a[j].physics_post();
        }
    }
}

#[derive(Default)]
struct Soa {
    o: [[f32; 8]; 3],
    m: [[f32; 8]; 9],
    lin: [[f32; 8]; 3],
    ang: [[f32; 8]; 3],
    p: [[f32; 8]; 4],
    dt: [f32; 8],
}

impl Soa {
    #[inline(always)]
    fn put(&mut self, j: usize, t: &Transform, lin: Vec3, ang: Vec3, dt: f32) {
        self.o[0][j] = t.origin.x;
        self.o[1][j] = t.origin.y;
        self.o[2][j] = t.origin.z;
        for r in 0..3 {
            self.m[r * 3][j] = t.basis.r[r].x;
            self.m[r * 3 + 1][j] = t.basis.r[r].y;
            self.m[r * 3 + 2][j] = t.basis.r[r].z;
        }
        self.lin[0][j] = lin.x;
        self.lin[1][j] = lin.y;
        self.lin[2][j] = lin.z;
        self.ang[0][j] = ang.x;
        self.ang[1][j] = ang.y;
        self.ang[2][j] = ang.z;
        self.dt[j] = dt;
    }
}

#[inline(always)]
fn ld3(a: &[[f32; 8]; 3]) -> [F8; 3] {
    [F8::load(&a[0]), F8::load(&a[1]), F8::load(&a[2])]
}

#[inline(always)]
fn ld9(a: &[[f32; 8]; 9]) -> [F8; 9] {
    core::array::from_fn(|i| F8::load(&a[i]))
}

#[inline(always)]
fn fin(v: &[F8], m: M8) -> M8 {
    let mut m = m;
    for x in v {
        m = m.and(x.finite());
    }
    m
}

#[inline(always)]
fn st<const K: usize>(v: &[F8; K], out: &mut [[f32; 8]; K]) {
    for i in 0..K {
        v[i].store(&mut out[i]);
    }
}

#[derive(Default)]
struct Out {
    o: [[f32; 8]; 3],
    m: [[f32; 8]; 9],
    a: [[f32; 8]; 3],
    b: [[f32; 8]; 3],
    q: [[f32; 8]; 4],
    ok: u32,
}

#[inline(always)]
fn transform_of(o: &Out, j: usize) -> Transform {
    Transform {
        basis: Mat3 { r: [v3(o.m[0][j], o.m[1][j], o.m[2][j]), v3(o.m[3][j], o.m[4][j], o.m[5][j]), v3(o.m[6][j], o.m[7][j], o.m[8][j])] },
        origin: v3(o.o[0][j], o.o[1][j], o.o[2][j]),
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn k_predict(s: &Soa, out: &mut Out) {
    let (o, m, lin, ang) = (ld3(&s.o), ld9(&s.m), ld3(&s.lin), ld3(&s.ang));
    let p = [F8::load(&s.p[0]), F8::load(&s.p[1]), F8::load(&s.p[2]), F8::load(&s.p[3])];
    let dt = F8::load(&s.dt);
    let all = M8(F8::splat(0.0).eq(F8::splat(0.0)).0);
    let good = fin(&o, fin(&m, fin(&lin, fin(&ang, fin(&p, fin(&[dt], all))))));
    let (l, a) = f8::damp(&lin, &ang, p[0], p[1], p[2], p[3]);
    let (o2, m2, trig) = f8::integrate_transform(&o, &m, &l, &a, dt);
    let good = fin(&o2, fin(&m2, fin(&l, fin(&a, good.and(trig)))));
    st(&o2, &mut out.o);
    st(&m2, &mut out.m);
    st(&l, &mut out.a);
    st(&a, &mut out.b);
    out.ok = good.bits();
}

#[target_feature(enable = "avx2,fma")]
unsafe fn k_integrate(s: &Soa, out: &mut Out) {
    let (o, m, lin, ang) = (ld3(&s.o), ld9(&s.m), ld3(&s.lin), ld3(&s.ang));
    let dt = F8::load(&s.dt);
    let all = M8(F8::splat(0.0).eq(F8::splat(0.0)).0);
    let good = fin(&o, fin(&m, fin(&lin, fin(&ang, fin(&[dt], all)))));
    let (o2, m2, trig) = f8::integrate_transform(&o, &m, &lin, &ang, dt);
    let good = fin(&o2, fin(&m2, good.and(trig)));
    st(&o2, &mut out.o);
    st(&m2, &mut out.m);
    out.ok = good.bits();
}

#[target_feature(enable = "avx2,fma")]
unsafe fn k_read(s: &Soa, quantize: bool, out: &mut Out) {
    let (o, m, lin, ang) = (ld3(&s.o), ld9(&s.m), ld3(&s.lin), ld3(&s.ang));
    let (ml, ma) = (F8::load(&s.p[0]), F8::load(&s.p[1]));
    let all = M8(F8::splat(0.0).eq(F8::splat(0.0)).0);
    let good = fin(&o, fin(&m, fin(&lin, fin(&ang, fin(&[ml, ma], all)))));
    let (loc, q, vel, spin) = f8::read_awake(&o, &m, &lin, &ang, ml, ma, quantize);
    let good = fin(&loc, fin(&q, fin(&vel, fin(&spin, good))));
    st(&loc, &mut out.o);
    st(&q, &mut out.q);
    st(&vel, &mut out.a);
    st(&spin, &mut out.b);
    out.ok = good.bits();
}

unsafe fn predict_lanes(a: &mut [&mut Arena], ok: &[bool; GROUP]) {
    let mut s = Soa::default();
    let mut used = 0u32;
    for j in 0..a.len() {
        if !ok[j] {
            continue;
        }
        let id = a[j].ball.as_ref().unwrap().body.0 as usize;
        let dt = a[j].dt;
        let b = &a[j].world.cw.bodies[id];
        if b.flags & (CF_STATIC | CF_KINEMATIC) != 0 || b.additional_damping {
            continue;
        }
        s.put(j, &b.transform, b.lin_vel, b.ang_vel, dt);
        s.p[0][j] = b.max_lin_speed;
        s.p[1][j] = b.max_ang_speed;
        s.p[2][j] = powf_memo(1.0 - b.lin_damping, dt);
        s.p[3][j] = powf_memo(1.0 - b.ang_damping, dt);
        used |= 1 << j;
    }
    if used == 0 {
        return;
    }
    let mut out = Out::default();
    unsafe { k_predict(&s, &mut out) };
    let good = used & out.ok;
    count(0, used, good);
    for j in 0..a.len() {
        if good & (1 << j) == 0 {
            continue;
        }
        let id = a[j].ball.as_ref().unwrap().body.0 as usize;
        let dt = a[j].dt;
        let (bodies, memo) = a[j].world.memo_mut();
        let t = bodies[id].transform;
        let lin = v3(out.a[0][j], out.a[1][j], out.a[2][j]);
        let ang = v3(out.b[0][j], out.b[1][j], out.b[2][j]);
        let mut out_t = transform_of(&out, j);
        out_t.origin.y = planted(1, out_t.origin.y);
        store_memo(memo, id, integrate_key_of(&t, &lin, &ang, dt), out_t);
    }
}

#[inline(always)]
fn store_memo(memo: &mut Vec<IntegrateMemo>, i: usize, key: [u32; 19], out: Transform) {
    if memo.len() <= i {
        memo.resize(i + 1, IntegrateMemo::default());
    }
    memo[i] = IntegrateMemo { key, out, valid: true };
}

unsafe fn integrate_lanes(a: &mut [&mut Arena], ok: &[bool; GROUP]) {
    let mut s = Soa::default();
    let mut used = 0u32;
    for j in 0..a.len() {
        if !ok[j] {
            continue;
        }
        let id = a[j].ball.as_ref().unwrap().body.0 as usize;
        let dt = a[j].dt;
        let (bodies, memo) = a[j].world.memo_mut();
        let b = &bodies[id];
        if b.activation_state == ISLAND_SLEEPING || b.activation_state == DISABLE_SIMULATION || b.flags & (CF_STATIC | CF_KINEMATIC) != 0 {
            continue;
        }
        let key = integrate_key_of(&b.transform, &b.lin_vel, &b.ang_vel, dt);
        if memo.get(id).is_some_and(|m| m.valid && m.key == key) {
            continue;
        }
        s.put(j, &b.transform, b.lin_vel, b.ang_vel, dt);
        used |= 1 << j;
    }
    if used == 0 {
        return;
    }
    let mut out = Out::default();
    unsafe { k_integrate(&s, &mut out) };
    let good = used & out.ok;
    count(1, used, good);
    for j in 0..a.len() {
        if good & (1 << j) == 0 {
            continue;
        }
        let id = a[j].ball.as_ref().unwrap().body.0 as usize;
        let dt = a[j].dt;
        let (bodies, memo) = a[j].world.memo_mut();
        let b = &bodies[id];
        let key = integrate_key_of(&b.transform, &b.lin_vel, &b.ang_vel, dt);
        let mut t = transform_of(&out, j);
        t.origin.x = planted(2, t.origin.x);
        store_memo(memo, id, key, t);
    }
}

unsafe fn read_lanes(a: &mut [&mut Arena], ok: &[bool; GROUP]) {
    let mut s = Soa::default();
    let mut used = 0u32;
    let mut quantize = None;
    for j in 0..a.len() {
        if !ok[j] {
            continue;
        }
        let Arena { ball, world, .. } = &mut *a[j];
        let bl = ball.as_ref().unwrap();
        let b = &world.cw.bodies[bl.body.0 as usize];
        if b.activation_state == ISLAND_SLEEPING || b.activation_state == DISABLE_SIMULATION {
            continue;
        }
        if *quantize.get_or_insert(bl.limits.quantize) != bl.limits.quantize {
            continue;
        }
        if bl.read_memo.valid && bl.read_memo.key == read_key(b, &bl.limits) {
            continue;
        }
        s.put(j, &b.transform, b.lin_vel, b.ang_vel, 0.0);
        s.p[0][j] = bl.limits.max_linear;
        s.p[1][j] = bl.limits.max_angular;
        used |= 1 << j;
    }
    if used == 0 {
        return;
    }
    let mut out = Out::default();
    unsafe { k_read(&s, quantize.unwrap_or(false), &mut out) };
    let good = used & out.ok;
    count(2, used, good);
    for j in 0..a.len() {
        if good & (1 << j) == 0 {
            continue;
        }
        let Arena { ball, world, .. } = &mut *a[j];
        let bl = ball.as_mut().unwrap();
        let b = &world.cw.bodies[bl.body.0 as usize];
        let key = read_key(b, &bl.limits);
        let value = RbState {
            quat: Quat { x: out.q[0][j], y: out.q[1][j], z: out.q[2][j], w: out.q[3][j] },
            location: v3(planted(3, out.o[0][j]), out.o[1][j], out.o[2][j]),
            velocity: v3(out.a[0][j], out.a[1][j], out.a[2][j]),
            spin: v3(out.b[0][j], out.b[1][j], out.b[2][j]),
            sleeping: false,
        };
        bl.read_memo.key = key;
        bl.read_memo.value = value;
        bl.read_memo.valid = true;
    }
}
