use crate::bt::{ACTIVE, Body, DISABLE_DEACTIVATION, DISABLE_SIMULATION, ISLAND_SLEEPING};
use crate::math::rot::{app_round, cvtss2si, cvttss2si};
use crate::math::{Mat3, Quat, Vec3, v3};

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct RbState {
    pub quat: Quat,
    pub location: Vec3,
    pub velocity: Vec3,
    pub spin: Vec3,

    pub sleeping: bool,
}

impl Default for RbState {
    fn default() -> RbState {
        RbState { quat: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }, location: Vec3::ZERO, velocity: Vec3::ZERO, spin: Vec3::ZERO, sleeping: false }
    }
}

impl RbState {

    pub fn words(&self) -> [u32; 14] {
        let q = self.quat;
        let (l, v, s) = (self.location.bits(), self.velocity.bits(), self.spin.bits());
        [q.x.to_bits(), q.y.to_bits(), q.z.to_bits(), q.w.to_bits(), l[0], l[1], l[2], v[0], v[1], v[2], s[0], s[1], s[2], self.sleeping as u32]
    }
}

#[inline]
pub fn quat_to_matrix(q: Quat) -> Mat3 {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    let z2 = z + z;
    let y2 = y + y;
    let xx2 = (x + x) * x;
    let xw2 = (x + x) * w;
    Mat3 {
        r: [
            v3(1.0 - (z2 * z + y2 * y), z2 * w + y2 * x, z2 * x - y2 * w),
            v3(y2 * x - z2 * w, 1.0 - (z2 * z + xx2), xw2 + z2 * y),
            v3(y2 * w + z2 * x, z2 * y - xw2, 1.0 - (y2 * y + xx2)),
        ],
    }
}

#[inline]
pub fn quat_rotate(q: Quat, v: Vec3) -> Vec3 {
    let (qx, qy, qz, qw) = (q.x, q.y, q.z, q.w);
    let w2 = qw + qw;
    let k = qw * qw - (qy * qy + qx * qx + qz * qz);
    let d = v.x * qx + v.y * qy + v.z * qz;
    let d2 = d + d;
    v3(
        (v.z * qy - v.y * qz) * w2 + k * v.x + d2 * qx,
        (v.x * qz - v.z * qx) * w2 + k * v.y + d2 * qy,
        (v.y * qx - v.x * qy) * w2 + k * v.z + d2 * qz,
    )
}

#[inline]
pub fn forward_speed(s: &RbState) -> f32 {
    let f = quat_rotate(s.quat, v3(1.0, 0.0, 0.0));
    f.y * s.velocity.y + f.x * s.velocity.x + f.z * s.velocity.z
}

pub fn matrix_to_quat(m: &Mat3) -> Quat {
    let (r0, r1, r2) = (m.r[0], m.r[1], m.r[2]);
    let trace = r0.x + r1.y + r2.z;
    let mut q = [0.0f32; 4];
    if trace <= 0.0 {
        let i = if r1.y <= r0.x {
            if r0.x < r2.z { 2 } else { 0 }
        } else if r1.y < r2.z {
            2
        } else {
            1
        };
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let s = ((m.r[i][i] - m.r[j][j]) - m.r[k][k]) + 1.0;
        let s = s.sqrt();
        let f = 0.5 / s;
        q[i] = s * 0.5;
        q[3] = (m.r[k][j] - m.r[j][k]) * f;
        q[j] = (m.r[j][i] + m.r[i][j]) * f;
        q[k] = (m.r[i][k] + m.r[k][i]) * f;
    } else {
        let s = (trace + 1.0).sqrt();
        q[3] = s * 0.5;
        let f = 0.5 / s;
        q[1] = (r0.z - r2.x) * f;
        q[2] = (r1.x - r0.y) * f;
        q[0] = (r2.y - r1.z) * f;
    }
    Quat { x: q[0], y: q[1], z: q[2], w: q[3] }
}

pub fn quat_to_btmatrix(q: Quat) -> Mat3 {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    let s = 2.0 / (y * y + x * x + z * z + w * w);
    let zs = s * z;
    let ys = s * y;
    let xxs = s * x * x;
    let xws = s * x * w;
    Mat3 {
        r: [
            v3(1.0 - (zs * z + ys * y), ys * x - zs * w, zs * x + ys * w),
            v3(ys * x + zs * w, 1.0 - (zs * z + xxs), zs * y - xws),
            v3(zs * x - ys * w, zs * y + xws, 1.0 - (ys * y + xxs)),
        ],
    }
}

pub fn quat_encode(q: Quat) -> ([i32; 3], u32) {
    let (mut x, mut y, mut z, mut w) = (0.0f32, 0.0f32, 0.0f32, 1.0f32);
    let n2 = q.y * q.y + q.x * q.x + q.z * q.z + q.w * q.w;
    if 1e-08 < n2 {
        let k = 1.0 / n2.sqrt();
        w = k * q.w;
        x = k * q.x;
        y = k * q.y;
        z = k * q.z;
    }
    let (ax, ay, az, aw) = (x.abs(), y.abs(), z.abs(), w.abs());

    let (a, ia) = if ax < ay { (ay, 1u32) } else { (ax, 0) };
    let (b, ib) = if az < aw { (aw, 3u32) } else { (az, 2) };
    let which = if b <= a { ia } else { ib };
    let (c0, c1, c2, big) = match which {
        0 => (y, z, w, x),
        1 => (x, z, w, y),
        2 => (x, y, w, z),
        _ => (x, y, z, w),
    };
    (quat_code3(c0, c1, c2, big), which)
}

#[inline]
fn quat_code3(mut a: f32, mut b: f32, mut c: f32, big: f32) -> [i32; 3] {
    if big < 0.0 {
        a *= -1.0;
        b *= -1.0;
        c *= -1.0;
    }
    const MAX: i32 = (1 << 18) - 1;
    let one = |v: f32| -> i32 {
        let f = (v * f32::from_bits(0x3fb504f3) * 0.5 + 0.5) * MAX as f32;
        let k = cvtss2si(f + f + 0.5) >> 1;
        if k < 0 {
            0
        } else if k < MAX {
            k
        } else {
            MAX
        }
    };
    [one(a), one(b), one(c)]
}

pub fn quat_decode(codes: [i32; 3], which: u32) -> Quat {
    const MAX: f32 = ((1 << 18) - 1) as f32;
    let one = |k: i32| -> f32 {
        let f = k as f32 / MAX - 0.5;
        (f + f) * f32::from_bits(0x3f3504f3)
    };
    let (a, b, c) = (one(codes[0]), one(codes[1]), one(codes[2]));
    let big = f64::from(((1.0 - a * a) - b * b) - c * c).sqrt() as f32;
    let (x, y, z, w) = match which {
        0 => (big, a, b, c),
        1 => (a, big, b, c),
        2 => (a, b, big, c),
        _ => (a, b, c, big),
    };
    let n2 = x * x + y * y + z * z + w * w;
    if n2 <= 1e-08 {
        Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    } else {
        let k = 1.0 / n2.sqrt();
        Quat { x: x * k, y: y * k, z: z * k, w: w * k }
    }
}

#[inline]
fn grid_location(v: f32) -> f32 {
    let scale = 100.0f32;
    let k = app_round(scale * v);
    (1.0 / scale) * k as f32
}

#[inline]
fn grid_velocity(v: f32, scale: f32) -> f32 {
    let k = cvttss2si(scale * v);
    let inv = 1.0 / scale;
    let f = inv * k as f32;
    let sign = if f <= 0.0 {
        if f >= 0.0 { 0 } else { -1 }
    } else {
        1
    };
    sign as f32 * (inv * 0.1) + f
}

pub fn codec(s: &mut RbState) {
    s.location = v3(grid_location(s.location.x), grid_location(s.location.y), grid_location(s.location.z));
    let (codes, which) = quat_encode(s.quat);
    s.quat = quat_decode(codes, which);
    if s.sleeping {
        s.velocity = Vec3::ZERO;
        s.spin = Vec3::ZERO;
    } else {
        s.velocity = v3(grid_velocity(s.velocity.x, 100.0), grid_velocity(s.velocity.y, 100.0), grid_velocity(s.velocity.z, 100.0));
        s.spin = v3(grid_velocity(s.spin.x, 10000.0), grid_velocity(s.spin.y, 10000.0), grid_velocity(s.spin.z, 10000.0));
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadLimits {
    pub max_linear: f32,
    pub max_angular: f32,
    pub quantize: bool,
}

pub fn read_body(b: &Body, lim: &ReadLimits) -> RbState {
    let o = b.transform.origin;
    let mut s = RbState {
        location: v3(o.x * 50.0, o.y * 50.0, o.z * 50.0),
        quat: matrix_to_quat(&b.transform.basis),
        velocity: v3(b.lin_vel.x * 50.0, b.lin_vel.y * 50.0, b.lin_vel.z * 50.0),
        spin: b.ang_vel,
        sleeping: b.activation_state == ISLAND_SLEEPING || b.activation_state == DISABLE_SIMULATION,
    };
    if s.sleeping {
        s.velocity = Vec3::ZERO;
        s.spin = Vec3::ZERO;
    } else {
        let m = lim.max_linear;
        if 0.0 < m {
            let v = s.velocity;
            let n2 = v.y * v.y + v.x * v.x + v.z * v.z;
            if m * m < n2 {
                let d = cap_dir(v, n2);
                s.velocity = v3(m * d.x, m * d.y, m * d.z);
            }
        }
        let m = lim.max_angular;
        if 0.0 < m {
            let v = s.spin;
            let n2 = v.y * v.y + v.x * v.x + v.z * v.z;
            if m * m < n2 {
                let d = cap_dir(v, n2);
                s.spin = v3(m * d.x, m * d.y, m * d.z);
            }
        }
    }
    if lim.quantize {
        codec(&mut s);
    }
    s
}

#[inline]
fn cap_dir(v: Vec3, n2: f32) -> Vec3 {
    if n2 == 1.0 {
        v
    } else if 1e-08 <= n2 {
        let k = 1.0 / n2.sqrt();
        v3(k * v.x, k * v.y, k * v.z)
    } else {
        Vec3::ZERO
    }
}

pub fn write_body(b: &mut Body, s: &RbState) {
    let body_asleep = b.activation_state == ISLAND_SLEEPING || b.activation_state == DISABLE_SIMULATION;
    if s.sleeping != body_asleep {
        if s.sleeping {
            set_activation_state(b, ISLAND_SLEEPING);
        } else {
            activate(b);
        }
    }
    b.transform.origin = v3(s.location.x * 0.02, s.location.y * 0.02, s.location.z * 0.02);
    b.transform.basis = quat_to_btmatrix(s.quat);
    b.update_revision += 1;
    b.lin_vel = v3(s.velocity.x * 0.02, s.velocity.y * 0.02, s.velocity.z * 0.02);
    b.update_revision += 1;
    b.ang_vel = s.spin;
}

#[inline]
pub fn set_activation_state(b: &mut Body, st: i32) {
    if b.activation_state != DISABLE_DEACTIVATION && b.activation_state != DISABLE_SIMULATION {
        b.activation_state = st;
    }
}

#[inline]
pub fn activate(b: &mut Body) {
    if b.flags & 3 == 0 {
        if b.activation_state != DISABLE_DEACTIVATION && b.activation_state != DISABLE_SIMULATION {
            b.activation_state = ACTIVE;
        }
        b.deactivation_time = 0.0;
    }
}

#[inline]
pub fn apply_torque_impulse(b: &mut Body, t: Vec3) {
    let m = &b.inv_inertia_world;
    let (a, c, d) = (m.r[0], m.r[1], m.r[2]);
    let f = b.ang_factor;
    b.ang_vel.y = (t.y * c.y + t.x * c.x + t.z * c.z) * f.y + b.ang_vel.y;
    b.ang_vel.z = (t.y * d.y + t.x * d.x + t.z * d.z) * f.z + b.ang_vel.z;
    b.ang_vel.x = (t.y * a.y + t.x * a.x + t.z * a.z) * f.x + b.ang_vel.x;
}

pub fn apply_impulse(b: &mut Body, i: Vec3, r: Vec3) {
    let im = b.inv_mass;
    if im != 0.0 {
        let lf = b.lin_factor;
        b.lin_vel.x = i.x * lf.x * im + b.lin_vel.x;
        b.lin_vel.y = i.y * lf.y * im + b.lin_vel.y;
        b.lin_vel.z = i.z * lf.z * im + b.lin_vel.z;
        let fz = lf.z * i.z;
        let fy = lf.y * i.y;
        let fx = lf.x * i.x;
        let t = v3(r.y * fz - r.z * fy, r.z * fx - r.x * fz, r.x * fy - r.y * fx);
        apply_torque_impulse(b, t);
    }
}

pub fn apply_force(b: &mut Body, f: Vec3, r: Vec3) {
    let lf = b.lin_factor;
    b.total_force.y = f.y * lf.y + b.total_force.y;
    b.total_force.z = f.z * lf.z + b.total_force.z;
    b.total_force.x = f.x * lf.x + b.total_force.x;
    let fx = f.x * lf.x;
    let fz = f.z * lf.z;
    let fy = f.y * lf.y;
    let af = b.ang_factor;
    b.total_torque.y = (r.z * fx - r.x * fz) * af.y + b.total_torque.y;
    b.total_torque.x = (r.y * fz - r.z * fy) * af.x + b.total_torque.x;
    b.total_torque.z = (r.x * fy - r.y * fx) * af.z + b.total_torque.z;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ForceMode {
    Force = 0,
    Impulse = 1,
    VelocityChange = 2,
    Accel = 5,
}

pub fn add_force(b: &mut Body, f: Vec3, mode: ForceMode) {
    let mut v = v3(f.x * 0.02, f.y * 0.02, f.z * 0.02);
    match mode {
        ForceMode::Impulse => apply_impulse(b, v, Vec3::ZERO),
        ForceMode::VelocityChange => {
            if 0.0 < b.inv_mass {
                let k = 1.0 / b.inv_mass;
                v = v3(k * v.x, k * v.y, k * v.z);
            }
            apply_impulse(b, v, Vec3::ZERO);
        }
        ForceMode::Accel => {
            if 0.0 < b.inv_mass {
                let k = 1.0 / b.inv_mass;
                v = v3(k * v.x, k * v.y, k * v.z);
            }
            apply_force(b, v, Vec3::ZERO);
        }
        ForceMode::Force => apply_force(b, v, Vec3::ZERO),
    }
    activate(b);
}

pub fn add_torque(b: &mut Body, t: Vec3, mode: ForceMode) {
    let m = b.inv_inertia_world;
    let (m00, m01, m02) = (m.r[0].x, m.r[0].y, m.r[0].z);
    let (m10, m11, m12) = (m.r[1].x, m.r[1].y, m.r[1].z);
    let (m20, m21, m22) = (m.r[2].x, m.r[2].y, m.r[2].z);
    match mode {
        ForceMode::Impulse => apply_torque_impulse(b, t),
        ForceMode::VelocityChange => {
            let mut out = t;
            if 0.0 < b.inv_mass {
                let c0 = m11 * m22 - m12 * m21;
                let c1 = m20 * m12 - m10 * m22;
                let c2 = m10 * m21 - m20 * m11;
                let k = 1.0 / (m01 * c1 + m00 * c0 + m02 * c2);
                out = v3(
                    (m02 * m21 - m01 * m22) * k * t.y + k * c0 * t.x + (m01 * m12 - m02 * m11) * k * t.z,
                    (m00 * m22 - m02 * m20) * k * t.y + k * c1 * t.x + (m02 * m10 - m00 * m12) * k * t.z,
                    (m01 * m20 - m00 * m21) * k * t.y + k * c2 * t.x + (m00 * m11 - m01 * m10) * k * t.z,
                );
            }
            apply_torque_impulse(b, out);
        }
        ForceMode::Accel if 0.0 < b.inv_mass => {
            let af = b.ang_factor;
            let c0 = m11 * m22 - m12 * m21;
            let c1 = m20 * m12 - m10 * m22;
            let c2 = m10 * m21 - m20 * m11;
            let k = 1.0 / (c1 * m01 + c0 * m00 + c2 * m02);
            b.total_torque.y = ((m22 * m00 - m20 * m02) * k * t.y + k * c1 * t.x + (m10 * m02 - m12 * m00) * k * t.z) * af.y + b.total_torque.y;
            b.total_torque.z = ((m20 * m01 - m21 * m00) * k * t.y + k * c2 * t.x + (m11 * m00 - m10 * m01) * k * t.z) * af.z + b.total_torque.z;
            b.total_torque.x = ((m21 * m02 - m22 * m01) * k * t.y + k * c0 * t.x + (m12 * m01 - m11 * m02) * k * t.z) * af.x + b.total_torque.x;
        }
        _ => {
            let af = b.ang_factor;
            b.total_torque.y = t.y * af.y + b.total_torque.y;
            b.total_torque.z = t.z * af.z + b.total_torque.z;
            b.total_torque.x = t.x * af.x + b.total_torque.x;
        }
    }
    activate(b);
}
