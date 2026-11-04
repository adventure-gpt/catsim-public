use crate::bt::Body;
use crate::game::rb_state::{ForceMode, RbState, add_force, add_torque, quat_rotate, quat_to_matrix};
use crate::game::drive::sensitivity;
use crate::math::rot::{Rotator, matrix_to_rotator, rotator_to_matrix};
use crate::math::{Vec3, v3};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Component {
    pub active: bool,
    pub time: f32,
    pub min_inactive_time: f32,
}

impl Component {

    #[inline]
    pub fn set_active(&mut self, on: bool) {
        if on != self.active {
            self.time = 0.0;
            self.active = on;
        }
    }

    #[inline]
    pub fn inactive_time(&self) -> f32 {
        if self.active { 0.0 } else { self.time }
    }

    #[inline]
    pub fn active_time(&self) -> f32 {
        if self.active { self.time } else { 0.0 }
    }
}

#[inline(always)]
fn big(v: Vec3) -> bool {
    0.0001 <= v.x.abs() || 0.0001 <= v.y.abs() || 0.0001 <= v.z.abs()
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AirControl {
    pub c: Component,
    pub torque: [i32; 3],
    pub damping: [i32; 3],
    pub torque_scalar: f32,
    pub damping_scalar: f32,
    pub throttle_force: f32,
    pub sensitivity: f32,

    pub robin: Option<[(f32, f32); 3]>,
}

#[inline]
pub fn air_pre(a: &mut AirControl, driving: bool, wheels_touching: i32) {
    a.c.set_active(driving && wheels_touching == 0);
}

pub fn air_forces(a: &AirControl, b: &mut Body, st: &RbState, input: (f32, f32, f32), pitch_locked: bool, throttle: f32, boosting: bool) {
    let mut pitch = sensitivity(input.0, a.sensitivity, 10.0);
    let yaw = sensitivity(input.1, a.sensitivity, 10.0);
    let roll = sensitivity(input.2, a.sensitivity, 10.0);
    if pitch_locked {
        pitch = 0.0;
    }
    const K: f32 = f32::from_bits(0x38c90fdb);
    let m = quat_to_matrix(st.quat);
    let (f, r, u) = (m.r[0], m.r[1], m.r[2]);
    let s = a.torque_scalar;
    let ty = a.torque[1] as f32 * yaw * K;
    let tp = a.torque[0] as f32 * -pitch * K;
    let tr = (-a.torque[2]) as f32 * roll * K;
    let t = v3(
        (tp * r.x + 0.0 + ty * u.x + tr * f.x) * s,
        (tp * r.y + 0.0 + ty * u.y + tr * f.y) * s,
        (tp * r.z + 0.0 + ty * u.z + tr * f.z) * s,
    );
    if big(t) {
        add_torque(b, t, ForceMode::Accel);
    }
    let w = st.spin;
    let dp = (1.0 - pitch.abs()) * ((w.x * r.x + w.y * r.y + w.z * r.z) * (a.damping[0] as f32 * K));
    let dy = (1.0 - yaw.abs()) * ((w.x * u.x + w.y * u.y + w.z * u.z) * (a.damping[1] as f32 * K));
    let dr = (w.x * f.x + w.y * f.y + w.z * f.z) * (a.damping[2] as f32 * K) * 1.0;
    let ds = a.damping_scalar;
    let d = v3(
        (((0.0 - dp * r.x) - dy * u.x) - dr * f.x) * ds,
        (((0.0 - dp * r.y) - dy * u.y) - dr * f.y) * ds,
        (((0.0 - dp * r.z) - dy * u.z) - dr * f.z) * ds,
    );
    if big(d) {
        add_torque(b, d, ForceMode::Accel);
    }
    let tf = a.throttle_force;
    if 0.0 < tf {
        let k = if boosting { 1.0 } else { throttle };
        if boosting || k != 0.0 {
            let v = v3(tf * f.x * k, tf * f.y * k, tf * f.z * k);
            if big(v) {
                add_force(b, v, ForceMode::Force);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Boost {
    pub c: Component,
    pub consumption_rate: f32,
    pub max_amount: f32,
    pub start_amount: f32,
    pub used: f32,
    pub amount: f32,
    pub modifier: f32,
    pub no_boost: bool,
    pub recharge_ground_only: bool,
    pub force: f32,
    pub min_boost_time: f32,
    pub recharge_rate: f32,
    pub recharge_delay: f32,
    pub unlimited_refs: i32,
    pub restriction: u8,
}

impl Boost {

    #[inline]
    pub fn set_amount(&mut self, v: f32) {
        let v = if 0.0 <= v {
            if self.max_amount <= v { self.max_amount } else { v }
        } else {
            0.0
        };
        if 1e-08 <= (v - self.amount).abs() || v == 0.0 {
            self.amount = v;
        }
    }

    #[inline]
    fn restricted(&self, on_ground: bool) -> bool {
        match self.restriction {
            1 => !on_ground,
            2 => on_ground,
            _ => false,
        }
    }

    #[inline]
    fn can_deactivate(&self) -> bool {
        if self.amount == 0.0 {
            return true;
        }
        self.c.active_time() >= self.min_boost_time
    }

    #[inline]
    fn can_activate(&self, driving: bool, on_ground: bool) -> bool {
        driving && self.c.min_inactive_time <= self.c.inactive_time() && 0.0 < self.amount && !self.restricted(on_ground)
    }
}

pub fn boost_pre(b: &mut Boost, activate_input: bool, driving: bool, on_ground: bool, dt: f32) {
    if !b.c.active {
        if !b.no_boost && b.recharge_rate != 0.0 && b.recharge_delay < b.c.time && (!b.recharge_ground_only || on_ground) {
            b.set_amount(dt * b.recharge_rate + b.amount);
        }
        if !activate_input || !b.can_activate(driving, on_ground) {
            return;
        }
        b.c.set_active(true);
    } else {
        let d = dt * b.consumption_rate;
        if b.unlimited_refs == 0 {
            let old = b.amount;
            b.set_amount(old - d);
            b.used = (old - b.amount) + b.used;
        } else {
            b.used = d + b.used;
        }
        if b.amount != 0.0 && !b.restricted(on_ground) && (activate_input || !b.can_deactivate()) {
            return;
        }
        b.c.set_active(false);
    }
}

pub fn boost_forces(c: &Boost, b: &mut Body, st: &RbState) {
    let f = quat_rotate(st.quat, v3(1.0, 0.0, 0.0));
    let (m, k) = (c.modifier, c.force);
    let v = v3(k * f.x * m, k * f.y * m, k * f.z * m);
    if big(v) {
        add_force(b, v, ForceMode::Force);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Jump {
    pub c: Component,
    pub min_jump_time: f32,
    pub impulse: f32,
    pub force: f32,
    pub force_time: f32,
    pub podium_force_time: f32,
    pub deactivate: bool,

    pub robin: Option<[(f32, f32); 3]>,
}

pub fn jump_pre(j: &mut Jump, podium: bool) {
    if j.c.active {
        let t = j.c.time;
        if (j.deactivate && t > j.min_jump_time) || (podium && t > j.podium_force_time) || t > j.force_time {
            j.c.set_active(false);
        }
    }
}

pub fn robin_torque(b: &mut Body, st: &RbState, p: &[(f32, f32); 3], mode: ForceMode) {
    let l = st.location;
    let mut m = l.x.abs();
    if m <= l.y.abs() {
        m = l.y.abs();
    }
    if m <= l.z.abs() {
        m = l.z.abs();
    }
    let ta = crate::math::crt::sinf(m * p[0].0) * p[0].1;
    let tb = crate::math::crt::sinf(m * p[1].0) * p[1].1;
    let tc = crate::math::crt::sinf(m * p[2].0) * p[2].1;
    let r = quat_to_matrix(st.quat);
    let (r0, r1, r2) = (r.r[0], r.r[1], r.r[2]);
    let t = v3(tb * r2.x + ta * r1.x + tc * r0.x, tb * r2.y + ta * r1.y + tc * r0.y, tb * r2.z + ta * r1.z + tc * r0.z);
    if big(t) {
        add_torque(b, t, mode);
    }
}

pub fn jump_forces(j: &Jump, b: &mut Body, st: &RbState, t: f32) {
    if t == 0.0 {
        if let Some(p) = &j.robin {
            robin_torque(b, st, p, ForceMode::VelocityChange);
        }
        let u = quat_to_matrix(st.quat).r[2];
        let k = j.impulse;
        let v = v3(k * u.x, k * u.y, k * u.z);
        if big(v) {
            add_force(b, v, ForceMode::Impulse);
        }
    }
    if t <= j.force_time {
        let u = quat_to_matrix(st.quat).r[2];
        let k = j.force;
        let v = v3(k * u.x, k * u.y, k * u.z);
        if big(v) {
            add_force(b, v, ForceMode::Force);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DoubleJump {
    pub c: Component,
    pub impulse: f32,
    pub impulse_scale: f32,
    pub stored: Vec3,

    pub robin: Option<[(f32, f32); 3]>,
}

pub fn double_jump_pre(d: &mut DoubleJump) {
    if d.c.active && d.c.time > f32::from_bits(0x3dcccccd) {
        d.c.set_active(false);
    }
}

pub fn double_jump_forces(d: &mut DoubleJump, b: &mut Body, st: &RbState, t: f32) {
    if t != 0.0 {
        return;
    }
    if let Some(p) = &d.robin {
        robin_torque(b, st, p, ForceMode::VelocityChange);
    }
    let u = quat_to_matrix(st.quat).r[2];
    let (k, s) = (d.impulse, d.impulse_scale);
    d.stored = v3(k * u.x * s, k * u.y * s, k * u.z * s);
    if big(d.stored) {
        add_force(b, d.stored, ForceMode::Impulse);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Dodge {
    pub c: Component,
    pub input_threshold: f32,
    pub side_impulse: f32,
    pub side_impulse_scale: f32,
    pub forward_impulse: f32,
    pub forward_impulse_scale: f32,
    pub backward_impulse: f32,
    pub backward_impulse_scale: f32,
    pub side_torque: f32,
    pub forward_torque: f32,
    pub torque_time: f32,
    pub min_torque_time: f32,
    pub z_damping: f32,
    pub z_damping_delay: f32,
    pub z_damping_up_time: f32,
    pub impulse_scale: f32,
    pub torque_scale: f32,
    pub torque: Vec3,
    pub direction: Vec3,
}

pub fn dodge_pre(d: &mut Dodge, on_ground: bool) {
    if d.c.active {
        if !on_ground {
            let t = d.c.active_time();
            if t < d.torque_time || t == d.torque_time {
                return;
            }
        }
        d.c.set_active(false);
    }
}

pub fn dodge_impulse(d: &Dodge, st: &RbState, max_linear: f32, dir: Vec3) -> Vec3 {
    let x = if dir.x.abs() <= 0.1 { 0.0 } else { dir.x };
    let y = if dir.y.abs() <= 0.1 { 0.0 } else { dir.y };
    let f = quat_rotate(st.quat, v3(1.0, 0.0, 0.0));
    let speed = f.y * st.velocity.y + f.x * st.velocity.x + f.z * st.velocity.z;
    let abs = speed.abs();
    let backward = if 100.0 <= abs { (0.0 <= x) != (0.0 <= speed) } else { x < 0.0 };
    let ratio = abs / max_linear;
    let (scale, imp) = if backward { (d.backward_impulse_scale, d.backward_impulse) } else { (d.forward_impulse_scale, d.forward_impulse) };
    let fwd = ((scale - 1.0) * ratio + 1.0) * (x * imp * d.impulse_scale);
    let side = ((d.side_impulse_scale - 1.0) * ratio + 1.0) * (y * d.side_impulse * d.impulse_scale);
    let r = matrix_to_rotator(&quat_to_matrix(st.quat));
    let m = rotator_to_matrix(Rotator { pitch: 0, yaw: r.yaw, roll: 0 });
    v3(
        m.r[0].x * fwd + side * m.r[1].x + m.r[2].x * 0.0 + 0.0 * 0.0,
        m.r[0].y * fwd + side * m.r[1].y + m.r[2].y * 0.0 + 0.0 * 0.0,
        m.r[0].z * fwd + side * m.r[1].z + m.r[2].z * 0.0 + 0.0 * 0.0,
    )
}

pub fn dodge_torque(d: &Dodge, b: &mut Body, st: &RbState, dodge_forward: f32, body_scale: f32, t: f32) {
    let tq = d.torque;
    let mut ty = tq.y;
    if d.min_torque_time <= t {
        let fwd = dodge_forward;
        let k = if 0.0 <= tq.y {
            let c = if -1.0 <= fwd {
                if 0.0 <= fwd { 0.0 } else { fwd }
            } else {
                -1.0
            };
            c + 1.0
        } else if 0.0 <= fwd {
            let c = if 1.0 <= fwd { 1.0 } else { fwd };
            1.0 - c
        } else {
            1.0
        };
        ty = tq.y * k;
    }
    let m = quat_to_matrix(st.quat);
    let k = 1.0 / body_scale;
    let (tx, tz, ty) = (k * tq.x, k * tq.z, k * ty);
    let v = v3(
        ty * m.r[1].x + tx * m.r[0].x + tz * m.r[2].x + 0.0 * 0.0,
        ty * m.r[1].y + tx * m.r[0].y + tz * m.r[2].y + 0.0 * 0.0,
        ty * m.r[1].z + tx * m.r[0].z + tz * m.r[2].z + 0.0 * 0.0,
    );
    if big(v) {
        add_torque(b, v, ForceMode::Accel);
    }
    if d.z_damping_delay <= t {
        let vz = st.velocity.z;
        if vz < 0.0 || d.z_damping_up_time < 0.0 || t < d.z_damping_delay + d.z_damping_up_time {
            let dv = -vz * d.z_damping;
            if 0.0001 <= dv.abs() {
                add_force(b, v3(0.0, 0.0, dv), ForceMode::VelocityChange);
            }
        }
    }
}

pub fn dodge_forces(d: &Dodge, b: &mut Body, st: &RbState, max_linear: f32, dodge_forward: f32, body_scale: f32, t: f32) {
    if t == 0.0 {
        let v = dodge_impulse(d, st, max_linear, d.direction);
        if big(v) {
            add_force(b, v, ForceMode::Impulse);
        }
    }
    dodge_torque(d, b, st, dodge_forward, body_scale, t);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FlipCar {
    pub c: Component,
    pub impulse: f32,
    pub torque: f32,
    pub time: f32,
    pub default_time: f32,
    pub flip_right: bool,
}

pub fn flip_pre(c: &mut FlipCar, on_ground: bool) {
    if c.c.active {
        if !on_ground {
            let t = c.c.active_time();
            if t < c.time || t == c.time {
                return;
            }
        }
        c.c.set_active(false);
    }
}

pub fn flip_forces(c: &FlipCar, b: &mut Body, st: &RbState, t: f32) {
    if t == 0.0 {
        let u = quat_to_matrix(st.quat).r[2];
        let k = -c.impulse;
        let v = v3(k * u.x, k * u.y, k * u.z);
        if big(v) {
            add_force(b, v, ForceMode::VelocityChange);
        }
    }
    if t <= c.time {
        let f = quat_to_matrix(st.quat).r[0];
        let k = c.torque;
        let s = if c.flip_right { 1.0 } else { -1.0 };
        let v = v3(k * f.x * s, k * f.y * s, k * f.z * s);
        if big(v) {
            add_torque(b, v, ForceMode::VelocityChange);
        }
    }
}
