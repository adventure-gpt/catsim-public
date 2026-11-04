use super::rb_state::{self, ForceMode, quat_to_matrix};
use crate::bt::Body;
use crate::bt::convex_cast::PointCollector;
use crate::bt::gjk::{ClosestPointInput, GjkDetector};
use crate::bt::manifold::Wrap;
use crate::bt::rigid_body::integrate_transform;
use crate::bt::shapes::Shape;
use crate::bt::simplex::Simplex;
use crate::bt::types::ManifoldPoint;
use crate::math::rot::{Rotator, matrix_to_rotator, rotator_to_matrix};
use crate::math::{Mat3, Quat, Transform, Vec3, crt, v3};

#[inline]
pub fn quat_rotator(q: Quat) -> Rotator {
    matrix_to_rotator(&quat_to_matrix(q))
}

pub fn slerp(a: Quat, b: Quat, t: f32) -> Quat {
    let c = a.y * b.y + a.x * b.x + a.z * b.z + a.w * b.w;
    let ac = if c < 0.0 { -c } else { c };
    let (s0, mut s1);
    if ac >= 0.9999 {
        s0 = 1.0 - t;
        s1 = t;
    } else {
        let cl = if ac >= 1.0 { 1.0 } else { ac };
        let om = crt::acosf(cl);
        let so = crt::sinf(om);
        let k = 1.0 / so;
        s0 = crt::sinf((1.0 - t) * om) * k;
        s1 = crt::sinf(om * t) * k;
    }
    if c < 0.0 {
        s1 = -s1;
    }
    Quat { x: s1 * b.x + a.x * s0, y: s1 * b.y + a.y * s0, z: s1 * b.z + a.z * s0, w: s1 * b.w + a.w * s0 }
}

#[inline]
pub fn normal(v: Vec3) -> Vec3 {
    let s = v.len2_xy();
    if s == 1.0 {
        v
    } else if s < 1e-8 {
        Vec3::ZERO
    } else {
        let k = 1.0 / s.sqrt();
        v3(v.x * k, v.y * k, v.z * k)
    }
}

#[inline]
pub fn vsize(v: Vec3) -> f32 {
    v.len2_xy().sqrt()
}

#[inline]
pub fn dot(a: Vec3, b: Vec3) -> f32 {
    a.dot_xy(b)
}

#[inline]
pub fn div_f(v: Vec3, f: f32) -> Vec3 {
    let k = 1.0 / f;
    v3(k * v.x, k * v.y, k * v.z)
}

#[derive(Clone, Copy, Debug)]
pub struct Curve(pub &'static [(f32, f32)]);

impl Curve {

    pub fn eval(&self, x: f32) -> f32 {
        let p = self.0;
        if p.is_empty() {
            return 0.0;
        }
        if p.len() < 2 || x <= p[0].0 {
            return p[0].1;
        }
        if p[p.len() - 1].0 <= x {
            return p[p.len() - 1].1;
        }
        let mut i = 1;
        while p[i].0 <= x {
            i += 1;
        }
        let w = p[i].0 - p[i - 1].0;
        if 0.0 < w {
            let t = (x - p[i - 1].0) / w;
            (p[i].1 - p[i - 1].1) * t + p[i - 1].1
        } else {
            p[i - 1].1
        }
    }
}

pub const PUSH_FACTOR: Curve = Curve(&[(0.0, f32::from_bits(0x3f26_6666)), (500.0, f32::from_bits(0x3f26_6666)), (1400.0, f32::from_bits(0x3f19_999a)), (2300.0, f32::from_bits(0x3f0c_cccd)), (4600.0, f32::from_bits(0x3e99_999a))]);
pub const PUSH_Z_SCALE: f32 = f32::from_bits(0x3eb3_3333);
pub const PUSH_FORWARD_SCALE: f32 = f32::from_bits(0x3f26_6666);
pub const MAX_RELATIVE_SPEED: f32 = 4600.0;

pub const CAR_BALL_FRICTION: f32 = 2.0;
pub const CAR_BALL_RESTITUTION: f32 = 0.0;

pub const BUMPER_GROUND: Curve = Curve(&[(0.0, 150.0), (1400.0, 200000.0), (2200.0, 275000.0)]);
pub const BUMPER_AIR: Curve = Curve(&[(0.0, 150.0), (1400.0, 250000.0), (2200.0, 350000.0)]);
pub const BUMPER_Z: Curve = Curve(&[(0.0, 50.0), (1400.0, 50000.0), (2200.0, 75000.0)]);
pub const BUMP_INTERVAL: f32 = 0.25;
pub const BUMP_YAW: f32 = 70.0;
pub const BUMP_PITCH: f32 = 36.869896;
pub const DEMO_YAW: f32 = 45.572994;
pub const DEMO_PITCH: f32 = BUMP_PITCH;
pub const SUPERSONIC_SPEED: f32 = 2200.0;
pub const SUPERSONIC_TURNOFF_BUFFER: f32 = 100.0;

pub const DEMO_SPEED_REQUIRED: f32 = SUPERSONIC_SPEED - SUPERSONIC_TURNOFF_BUFFER;

pub const TOI_ALLOWED_PENETRATION: f32 = f32::from_bits(0x3d23_d70a);

#[derive(Clone, Copy, Debug)]
pub struct Held {
    pub quat: Quat,
    pub loc: Vec3,
    pub vel: Vec3,
    pub ang: Vec3,
}

pub fn rb_add_force(b: &mut Body, f: Vec3, mode: ForceMode) {
    if 0.0001 <= f.x.abs() || 0.0001 <= f.y.abs() || 0.0001 <= f.z.abs() {
        rb_state::add_force(b, f, mode);
    }
}

pub fn add_impulse(b: &mut Body, imp: Vec3, position: Vec3, vel_change: bool) {
    let l = v3(imp.x * 0.02, imp.y * 0.02, imp.z * 0.02);
    let rel = if position.x == 0.0 && position.y == 0.0 && position.z == 0.0 {
        Vec3::ZERO
    } else {
        let o = b.transform.origin;
        v3(position.x * 0.02 - o.x, position.y * 0.02 - o.y, position.z * 0.02 - o.z)
    };
    let i = if vel_change && 0.0 < b.inv_mass {
        let k = 1.0 / b.inv_mass;
        v3(k * l.x, k * l.y, k * l.z)
    } else {
        l
    };
    rb_state::apply_impulse(b, i, rel);
    rb_state::activate(b);
}

pub fn forward_cone(m: &Mat3, d: Vec3, yaw_limit: f32, pitch_limit: f32, reverse: bool) -> bool {
    let (mut fx, mut fy, mut fz) = (m.r[0].x, m.r[0].y, m.r[0].z);
    if reverse {
        fx = -fx;
        fy = -fy;
        fz = -fz;
    }
    let ang = |axis: Vec3| -> f32 {
        let mut s = d.y * axis.y + d.x * axis.x + d.z * axis.z;
        if 0.0 <= s {
            s /= 1.01;
        } else {
            s *= 1.01;
        }
        let (mut ax, mut ay, mut az) = (d.x - axis.x * s, d.y - axis.y * s, d.z - axis.z * s);
        let l = ay * ay + ax * ax + az * az;
        if l != 1.0 {
            if 1e-8 <= l {
                let k = 1.0 / l.sqrt();
                ay *= k;
                az *= k;
                ax *= k;
            } else {
                ax = 0.0;
                ay = 0.0;
                az = 0.0;
            }
        }
        let c = fy * ay + ax * fx + fz * az;
        if -1.0 <= c {
            if 1.0 <= c { 1.0 } else { c }
        } else {
            -1.0
        }
    };
    let a1 = crt::acosf(ang(m.r[1]));
    let c2 = ang(m.r[2]);
    if a1 * 57.29578 <= pitch_limit {
        return crt::acosf(c2) * 57.29578 <= yaw_limit;
    }
    false
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Impact {
    pub fraction: f32,
    pub location: Vec3,
    pub rotation: Rotator,
    pub other_location: Vec3,
    pub other_rotation: Rotator,
    pub impact_location: Vec3,
    pub impact_normal: Vec3,
}

pub fn car_within_cone(own_offset: Vec3, other_offset: Vec3, im: &Impact, yaw_limit: f32, pitch_limit: f32, reverse: bool) -> bool {
    let m = rotator_to_matrix(im.rotation);
    let o = rotator_to_matrix(im.other_rotation);
    let part = |off: Vec3, r: Vec3| off.y * r.y + off.x * r.x + off.z * r.z;
    let dx = (part(other_offset, o.r[0]) + im.other_location.x) - (part(own_offset, m.r[0]) + im.location.x);
    let dz = (part(other_offset, o.r[2]) + im.other_location.z) - (part(own_offset, m.r[2]) + im.location.z);
    let dy = (part(other_offset, o.r[1]) + im.other_location.y) - (part(own_offset, m.r[1]) + im.location.y);
    let l = dy * dy + dx * dx + dz * dz;
    let d = if l == 1.0 {
        v3(dx, dy, dz)
    } else if l < 1e-8 {
        Vec3::ZERO
    } else {
        let k = 1.0 / l.sqrt();
        v3(dx * k, dy * k, dz * k)
    };
    forward_cone(&m, d, yaw_limit, pitch_limit, reverse)
}

pub fn impact_from_old(own: &Held, other: &Held, hit_loc: Vec3, hit_normal: Vec3) -> Impact {
    Impact {
        fraction: 0.0,
        location: own.loc,
        rotation: quat_rotator(own.quat),
        other_location: other.loc,
        other_rotation: quat_rotator(other.quat),
        impact_location: hit_loc,
        impact_normal: hit_normal,
    }
}

pub fn toi_blend(start_o: Vec3, end_o: Vec3, start_q: Quat, end_q: Quat, f: f32) -> (Vec3, Rotator) {
    let lerp = |e: f32, s: f32| (e * 50.0 - s * 50.0) * f + s * 50.0;
    let loc = v3(lerp(end_o.x, start_o.x), lerp(end_o.y, start_o.y), lerp(end_o.z, start_o.z));
    let q = slerp(start_q, end_q, f);
    let n = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    let q = if n <= 1e-8 {
        Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    } else {
        let k = 1.0 / n.sqrt();
        Quat { x: q.x * k, y: k * q.y, z: k * q.z, w: k * q.w }
    };
    (loc, matrix_to_rotator(&quat_to_matrix(q)))
}

pub fn ball_impact(car: &Held, ball: &Held) -> Option<Vec3> {
    let rel_vel = car.vel - ball.vel;
    let mut speed = vsize(rel_vel);
    if !(speed > 0.0) {
        return None;
    }
    speed = if MAX_RELATIVE_SPEED < speed { MAX_RELATIVE_SPEED } else { speed };
    let rel_pos = ball.loc - car.loc;
    let mut hit = normal(rel_pos);
    hit.z *= PUSH_Z_SCALE * (1.0 + 0.0 * 0.5);
    hit = normal(hit);
    let forward = rotator_to_matrix(quat_rotator(car.quat)).r[0];
    let k = dot(hit, forward);
    let f = v3(forward.x * k, forward.y * k, forward.z * k);
    let s = 1.0 - PUSH_FORWARD_SCALE;
    hit -= v3(f.x * s, f.y * s, f.z * s);
    hit = normal(hit);
    let hit_xy = v3(hit.x, hit.y, 0.0);
    let curve = PUSH_FACTOR.eval(speed);
    let a = v3(hit.x * speed, hit.y * speed, hit.z * speed);
    let b = v3(a.x * curve, a.y * curve, a.z * curve);
    let c = v3(b.x * 1.0, b.y * 1.0, b.z * 1.0);
    let mut imp = c + v3(hit_xy.x * 0.0, hit_xy.y * 0.0, hit_xy.z * 0.0);
    imp += v3(imp.x * 0.0, imp.y * 0.0, imp.z * 0.0);
    Some(imp)
}

pub fn bump_impulse(speed: f32, vel: Vec3, in_cone: bool, victim_on_ground: bool, up: Vec3) -> Vec3 {
    let dir = div_f(vel, speed);
    let mut imp_z = Vec3::ZERO;
    let push;
    if in_cone {
        if victim_on_ground {
            push = BUMPER_GROUND.eval(speed);
            let z = BUMPER_Z.eval(speed);
            imp_z = v3(up.x * z, up.y * z, up.z * z);
        } else {
            push = BUMPER_AIR.eval(speed);
        }
    } else {
        push = 0.0;
    }
    let mut imp = v3(dir.x * push, dir.y * push, dir.z * push);
    imp += imp_z;
    imp += v3(imp.x * 0.0, imp.y * 0.0, imp.z * 0.0);
    imp
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CarImpactGate {
    Stop,
    Go { speed: f32, vel_dir: Vec3 },
}

pub fn car_impact_gate(att: &Held, victim: &Held) -> CarImpactGate {
    let speed = vsize(att.vel);
    if speed <= 0.0 {
        return CarImpactGate::Stop;
    }
    let hit_dir = normal(victim.loc - att.loc);
    let toward = dot(hit_dir, att.vel);
    if toward <= 0.0 {
        return CarImpactGate::Stop;
    }
    let vel_dir = div_f(att.vel, speed);
    let other_speed = dot(victim.vel, vel_dir);
    if other_speed >= toward {
        return CarImpactGate::Stop;
    }
    CarImpactGate::Go { speed, vel_dir }
}

pub fn contact_added(cp: &mut ManifoldPoint, a: &Wrap, b: &Wrap, ball_constrained: bool) {
    let (o0, o1) = (a.body.owner, b.body.owner);
    let car_ball = matches!((o0, o1), (Owner::Car(_), Owner::Ball) | (Owner::Ball, Owner::Car(_)));
    if car_ball {
        cp.combined_restitution = CAR_BALL_RESTITUTION;
        cp.combined_friction = CAR_BALL_FRICTION;
    }
    if ball_constrained && (o0 == Owner::Ball || o1 == Owner::Ball) {
        cp.combined_restitution = 0.0;
    }
}

pub use crate::bt::Owner;

#[derive(Clone, Copy, Debug, Default)]
pub struct NotifyPoint {
    pub loc: Vec3,
    pub normal: Vec3,
    pub dist: f32,
    pub vel0: Vec3,
    pub vel1: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct RbCollision {
    pub other: Owner,

    pub actor: u32,
    pub vel0: Vec3,
    pub vel1: Vec3,
    pub loc: Vec3,
    pub normal: Vec3,
    pub force: Vec3,
    pub friction: Vec3,
    pub vel_normal: Vec3,
    pub vel_tangent: Vec3,
    pub calls: i32,
    pub points: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WorldContact {
    pub has: bool,
    pub loc: Vec3,
    pub vel: Vec3,
    pub normal: Vec3,
}

pub fn notify_points(m: &crate::bt::types::Manifold, a: &Body, b: &Body, out: &mut Vec<NotifyPoint>) -> Option<(Vec3, Vec3)> {
    let n = m.cached as usize;
    if n == 0 || a.flags & 4 != 0 || b.flags & 4 != 0 {
        return None;
    }
    let quiet = |x: &Body| x.internal_type & 2 != 0 && (x.activation_state == 2 || x.activation_state == 5);
    if quiet(a) || quiet(b) {
        return None;
    }
    let thr = m.processing_threshold;
    let mut lo = thr + 1.0;
    let mut hi = 0.0f32;
    for p in &m.points[..n] {
        if p.distance <= lo {
            lo = p.distance;
        }
        if hi <= p.rest_speed {
            hi = p.rest_speed;
        }
    }
    if !(lo <= thr || 0.0 < hi) {
        return None;
    }
    out.clear();
    let (mut force, mut fric) = (Vec3::ZERO, Vec3::ZERO);
    for p in &m.points[..n] {
        if p.distance <= thr || 0.0 < p.rest_speed {
            let i = p.applied_impulse;
            force = v3(i * p.normal_b.x + force.x, i * p.normal_b.y + force.y, i * p.normal_b.z + force.z);
            let l1 = p.applied_impulse_lat1;
            let f1 = v3(l1 * p.lat_dir1.x + fric.x, l1 * p.lat_dir1.y + fric.y, l1 * p.lat_dir1.z + fric.z);
            let l2 = p.applied_impulse_lat2;
            fric = v3(l2 * p.lat_dir2.x + f1.x, l2 * p.lat_dir2.y + f1.y, l2 * p.lat_dir2.z + f1.z);
            out.push(NotifyPoint { loc: p.world_a * 50.0, normal: p.normal_b, dist: p.distance, vel0: p.vel_a * 50.0, vel1: p.vel_b * 50.0 });
        }
    }
    Some((force, fric))
}

pub fn contact_append(list: &mut Vec<RbCollision>, other: Owner, actor: u32, points: &[NotifyPoint], force: Vec3, friction: Vec3) {
    if points.is_empty() {
        return;
    }
    let i = match list.iter().position(|r| r.actor == actor) {
        Some(i) => i,
        None => {
            list.push(RbCollision {
                other,
                actor,
                vel0: Vec3::ZERO,
                vel1: Vec3::ZERO,
                loc: Vec3::ZERO,
                normal: Vec3::ZERO,
                force: Vec3::ZERO,
                friction: Vec3::ZERO,
                vel_normal: Vec3::ZERO,
                vel_tangent: Vec3::ZERO,
                calls: 0,
                points: 0,
            });
            list.len() - 1
        }
    };
    let r = &mut list[i];
    r.calls += 1;
    r.force = r.force + force;
    r.friction = r.friction + friction;
    for p in points {
        r.points += 1;
        r.loc = r.loc + p.loc;
        r.normal = r.normal + p.normal;
        r.vel0 = r.vel0 + p.vel0;
        r.vel1 = r.vel1 + p.vel1;
    }
}

pub fn contact_average(list: &mut [RbCollision], is_world: impl Fn(Owner) -> bool) -> WorldContact {
    let mut w = WorldContact::default();
    let mut n = 0;
    for r in list.iter_mut() {
        let k = 1.0 / r.calls as f32;
        let kp = 1.0 / r.points as f32;
        r.force = v3(r.force.x * k, r.force.y * k, r.force.z * k);
        r.friction.x *= k;
        r.friction.z = k * r.friction.z;
        r.friction.y = k * r.friction.y;
        r.loc = v3(kp * r.loc.x, kp * r.loc.y, kp * r.loc.z);
        r.normal = normal_contact(r.normal);
        let v = v3(kp * r.vel0.x, kp * r.vel0.y, kp * r.vel0.z);
        r.vel0 = v;
        r.vel1 = v3(kp * r.vel1.x, kp * r.vel1.y, kp * r.vel1.z);
        let (nx, ny, nz) = (r.normal.x, r.normal.y, r.normal.z);
        let d = ny * v.y + nx * v.x + nz * v.z;
        let vn = v3(d * nx, d * ny, d * nz);
        r.vel_normal = vn;
        r.vel_tangent = v3(v.x - vn.x, v.y - vn.y, v.z - vn.z);
        if is_world(r.other) {
            n += 1;
            w.loc.x += r.loc.x;
            w.loc.y = r.loc.y + w.loc.y;
            w.loc.z += r.loc.z;
            w.normal.x += r.normal.x;
            w.normal.y += r.normal.y;
            w.normal.z = r.normal.z + w.normal.z;
            w.vel.x = r.vel0.x + w.vel.x;
            w.vel.y = r.vel0.y + w.vel.y;
            w.vel.z = w.vel.z + r.vel0.z;
            w.has = true;
        }
    }
    if 1 < n {
        let k = 1.0 / n as f32;
        w.loc = v3(k * w.loc.x, k * w.loc.y, k * w.loc.z);
        w.vel = v3(k * w.vel.x, k * w.vel.y, k * w.vel.z);
        let l = w.normal.z * w.normal.z + w.normal.x * w.normal.x + w.normal.y * w.normal.y;
        w.normal = normalize_ordered(w.normal, l);
    }
    w
}

#[inline]
fn normal_contact(sum: Vec3) -> Vec3 {
    let (x, y, z) = (sum.x, sum.y, sum.z);
    let s = y * y + x * x + z * z;
    if s == 1.0 {
        v3(x, y, z)
    } else if 1e-8 <= s {
        let k = 1.0 / s.sqrt();
        v3(x * k, y * k, z * k)
    } else {
        Vec3::ZERO
    }
}

#[inline]
fn normalize_ordered(v: Vec3, s: f32) -> Vec3 {
    if s == 1.0 {
        v
    } else if 1e-8 <= s {
        let k = 1.0 / s.sqrt();
        v3(k * v.x, k * v.y, k * v.z)
    } else {
        Vec3::ZERO
    }
}

pub fn diff_axis_angle(a: &Mat3, b: &Mat3) -> (Vec3, f32) {
    let (m0, m1, m2) = (a.r[0].x, a.r[0].y, a.r[0].z);
    let (m4, m5, m6) = (a.r[1].x, a.r[1].y, a.r[1].z);
    let (m8, m9, m10) = (a.r[2].x, a.r[2].y, a.r[2].z);
    let c00 = m5 * m10 - m6 * m9;
    let c10 = m8 * m6 - m4 * m10;
    let c20 = m4 * m9 - m8 * m5;
    let k = 1.0 / (c10 * m1 + c00 * m0 + c20 * m2);
    let i20 = k * c20;
    let i22 = (m0 * m5 - m1 * m4) * k;
    let i21 = (m1 * m8 - m0 * m9) * k;
    let i12 = (m2 * m4 - m0 * m6) * k;
    let i11 = (m0 * m10 - m2 * m8) * k;
    let i01 = (m2 * m9 - m1 * m10) * k;
    let i02 = (m1 * m6 - m2 * m5) * k;
    let i10 = k * c10;
    let i00 = k * c00;
    let (n0, n1, n2) = (b.r[0].x, b.r[0].y, b.r[0].z);
    let (n4, n5, n6) = (b.r[1].x, b.r[1].y, b.r[1].z);
    let (n8, n9, n10) = (b.r[2].x, b.r[2].y, b.r[2].z);
    let d = Mat3 {
        r: [
            v3(i00 * n0 + i10 * n1 + i20 * n2, i01 * n0 + i11 * n1 + i21 * n2, i02 * n0 + i12 * n1 + i22 * n2),
            v3(i00 * n4 + i10 * n5 + i20 * n6, i01 * n4 + i11 * n5 + i21 * n6, i02 * n4 + i12 * n5 + i22 * n6),
            v3(i00 * n8 + i10 * n9 + i20 * n10, i01 * n8 + i11 * n9 + i21 * n10, i02 * n8 + i12 * n9 + i22 * n10),
        ],
    };
    let q = rb_state::matrix_to_quat(&d);
    let k = 1.0 / (q.y * q.y + q.x * q.x + q.z * q.z + q.w * q.w).sqrt();
    let (x, y, z) = (q.x * k, q.y * k, q.z * k);
    let w = q.w * k;
    let w = if -1.0 <= w { w } else { -1.0 };
    let w = if w <= 1.0 { w } else { 1.0 };
    let half = crt::acosf(w);
    let angle = half + half;
    let l = x * x + y * y + z * z;
    let axis = if 1.4210855e-14 <= l {
        let k = 1.0 / l.sqrt();
        v3(k * x, y * k, k * z)
    } else {
        v3(1.0, 0.0, 0.0)
    };
    (axis, angle)
}

pub fn calc_time_of_impact(a: &Shape, b: &Shape, from_a: &Transform, to_a: &Transform, from_b: &Transform, to_b: &Transform, allowed: f32) -> Option<(f32, Vec3, Vec3)> {
    let lin_a = v3(to_a.origin.x - from_a.origin.x, to_a.origin.y - from_a.origin.y, to_a.origin.z - from_a.origin.z);
    let (axis, angle) = diff_axis_angle(&from_a.basis, &to_a.basis);
    let ang_a = v3(axis.x * angle, axis.y * angle, axis.z * angle);
    let lin_b = v3(to_b.origin.x - from_b.origin.x, to_b.origin.y - from_b.origin.y, to_b.origin.z - from_b.origin.z);
    let (axis, angle) = diff_axis_angle(&from_b.basis, &to_b.basis);
    let ang_b = v3(axis.x * angle, axis.y * angle, axis.z * angle);
    let (rad_a, rad_b) = (a.angular_motion_disc(), b.angular_motion_disc());
    let bound = (ang_a.x * ang_a.x + ang_a.y * ang_a.y + ang_a.z * ang_a.z).sqrt() * rad_a + (ang_b.x * ang_b.x + ang_b.y * ang_b.y + ang_b.z * ang_b.z).sqrt() * rad_b;
    let rel = v3(lin_b.x - lin_a.x, lin_b.y - lin_a.y, lin_b.z - lin_a.z);
    if (rel.z * rel.z + rel.y * rel.y + rel.x * rel.x).sqrt() + bound == 0.0 {
        return None;
    }
    let mut simplex = Simplex::new();
    let mut closest = |ta: &Transform, tb: &Transform| {
        let mut out = PointCollector::new();
        let mut det = GjkDetector::new(a, b, true);
        det.closest_points(&mut simplex, &ClosestPointInput { a: *ta, b: *tb, max_distance_sqr: f32::from_bits(0x5d5e_0b6b) }, &mut out);
        out
    };
    let c = closest(from_a, from_b);
    if !c.has_result {
        return None;
    }
    let (mut dist, mut n, mut hit) = (c.distance + allowed, c.normal, c.point);
    let mut lambda = 0.0f32;
    let proj = |n: Vec3| rel.x * n.x + rel.y * n.y + rel.z * n.z + bound;
    if !(1.1920929e-07 < proj(n)) {
        return None;
    }
    let mut iter = 0;
    loop {
        if dist <= 0.001 {
            return Some((lambda, n, hit));
        }
        let p = proj(n);
        if p <= 1.1920929e-07 {
            return None;
        }
        let new = lambda + dist / p;
        if 1.0 < new || new < 0.0 || new <= lambda {
            return None;
        }
        let ta = integrate_transform(from_a, lin_a, ang_a, new);
        let tb = integrate_transform(from_b, lin_b, ang_b, new);
        let c = closest(&ta, &tb);
        if !c.has_result {
            return None;
        }
        dist = c.distance + allowed;
        n = c.normal;
        hit = c.point;
        lambda = new;
        iter += 1;
        if iter >= 0x41 {
            return None;
        }
    }
}

#[inline]
fn child_transform(p: &Transform, c: &Transform) -> Transform {
    let (r0, r1, r2) = (p.basis.r[0], p.basis.r[1], p.basis.r[2]);
    let (c0, c1, c2) = (c.basis.col(0), c.basis.col(1), c.basis.col(2));
    Transform {
        basis: Mat3 {
            r: [
                v3(r0.dot_xy(c0), r0.dot_xy(c1), r0.dot_xy(c2)),
                v3(r1.dot_xy(c0), r1.dot_xy(c1), r1.dot_xy(c2)),
                v3(r2.dot_xy(c0), r2.dot_xy(c1), r2.dot_xy(c2)),
            ],
        },
        origin: v3(r0.dot_xy(c.origin) + p.origin.x, r1.dot_xy(c.origin) + p.origin.y, r2.dot_xy(c.origin) + p.origin.z),
    }
}

pub fn cast_shapes(a: &Shape, b: &Shape, from_a: &Transform, to_a: &Transform, from_b: &Transform, to_b: &Transform, allowed: f32, best: &mut Option<(f32, Vec3, Vec3)>) {
    if let Shape::Compound(c) = a {
        for ch in &c.children {
            let (f, t) = (child_transform(from_a, &ch.transform), child_transform(to_a, &ch.transform));
            cast_shapes(&ch.shape, b, &f, &t, from_b, to_b, allowed, best);
        }
    } else if let Shape::Compound(c) = b {
        for ch in &c.children {
            let (f, t) = (child_transform(from_b, &ch.transform), child_transform(to_b, &ch.transform));
            cast_shapes(a, &ch.shape, from_a, to_a, &f, &t, allowed, best);
        }
    } else if let Some(r) = calc_time_of_impact(a, b, from_a, to_a, from_b, to_b, allowed) {
        if best.map_or(1.0, |x| x.0) > r.0 {
            *best = Some(r);
        }
    }
}
