use super::types::*;
use crate::math::crt::{cosf, sinf};
use crate::math::crt_pow::powf_memo as powf;
use crate::math::{Mat3, Quat, Transform, Vec3, v3};

#[derive(Clone, Copy, Debug)]
pub struct RigidBodyInfo {
    pub mass: f32,
    pub transform: Transform,
    pub shape: ShapeId,
    pub local_inertia: Vec3,
    pub lin_damping: f32,
    pub ang_damping: f32,
    pub friction: f32,
    pub rolling_friction: f32,
    pub restitution: f32,
    pub lin_sleep_threshold: f32,
    pub ang_sleep_threshold: f32,
    pub additional_damping: bool,
    pub additional_damping_factor: f32,
    pub additional_lin_threshold_sqr: f32,
    pub additional_ang_threshold_sqr: f32,
    pub additional_ang_factor: f32,
    pub max_lin_speed: f32,
    pub max_ang_speed: f32,
}

impl RigidBodyInfo {
    pub fn new(mass: f32, transform: Transform, shape: ShapeId, local_inertia: Vec3) -> RigidBodyInfo {
        RigidBodyInfo {
            mass,
            transform,
            shape,
            local_inertia,
            lin_damping: 0.0,
            ang_damping: 0.0,
            friction: 0.5,
            rolling_friction: 0.0,
            restitution: 0.0,
            lin_sleep_threshold: 0.8,
            ang_sleep_threshold: 1.0,
            additional_damping: false,
            additional_damping_factor: f32::from_bits(0x3ba3d70a),
            additional_lin_threshold_sqr: f32::from_bits(0x3c23d70a),
            additional_ang_threshold_sqr: f32::from_bits(0x3c23d70a),
            additional_ang_factor: f32::from_bits(0x3c23d70a),
            max_lin_speed: 0.0,
            max_ang_speed: 0.0,
        }
    }
}

pub fn mat_to_quat(m: &Mat3) -> Quat {
    let (m00, m11, m22) = (m.r[0].x, m.r[1].y, m.r[2].z);
    let trace = m00 + m11 + m22;
    if trace <= 0.0 {
        let i = if m11 <= m00 {
            if m00 < m22 { 2 } else { 0 }
        } else if m11 < m22 {
            2
        } else {
            1
        };
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let s = (((m.r[i][i] - m.r[j][j]) - m.r[k][k]) + 1.0).sqrt();
        let t = 0.5 / s;
        let mut q = [0.0f32; 4];
        q[i] = s * 0.5;
        q[3] = (m.r[k][j] - m.r[j][k]) * t;
        q[j] = (m.r[j][i] + m.r[i][j]) * t;
        q[k] = (m.r[i][k] + m.r[k][i]) * t;
        Quat { x: q[0], y: q[1], z: q[2], w: q[3] }
    } else {
        let s = (trace + 1.0).sqrt();
        let t = 0.5 / s;
        Quat { x: (m.r[2].y - m.r[1].z) * t, y: (m.r[0].z - m.r[2].x) * t, z: (m.r[1].x - m.r[0].y) * t, w: s * 0.5 }
    }
}

pub fn quat_to_mat(q: Quat) -> Mat3 {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    let s = 2.0 / (y * y + x * x + z * z + w * w);
    let sz = s * z;
    let sy = s * y;
    let sxx = s * x * x;
    let sxw = s * x * w;
    Mat3 {
        r: [
            v3(1.0 - (sz * z + sy * y), sy * x - sz * w, sz * x + sy * w),
            v3(sy * x + sz * w, 1.0 - (sz * z + sxx), sz * y - sxw),
            v3(sz * x - sy * w, sz * y + sxw, 1.0 - (sy * y + sxx)),
        ],
    }
}

pub fn integrate_transform(cur: &Transform, lin: Vec3, ang: Vec3, dt: f32) -> Transform {
    let origin = v3(dt * lin.x + cur.origin.x, dt * lin.y + cur.origin.y, dt * lin.z + cur.origin.z);
    let (wx, wy, wz) = (ang.x, ang.y, ang.z);
    let mut w = (wx * wx + wy * wy + wz * wz).sqrt();

    if 0.7853982 < w * dt {
        w = 0.7853982 / dt;
    }
    let s = if 0.001 <= w { sinf(w * 0.5 * dt) / w } else { dt * 0.5 - dt * dt * dt * 0.020833334 * w * w };
    let (ax, az, ay) = (wx * s, wz * s, wy * s);
    let c = cosf(w * dt * 0.5);
    let q = mat_to_quat(&cur.basis);
    let x = (ax * q.w + c * q.x + q.z * ay) - az * q.y;
    let y = (c * q.y + q.w * ay + az * q.x) - ax * q.z;
    let z = (c * q.z + az * q.w + ax * q.y) - ay * q.x;
    let w2 = ((c * q.w - ax * q.x) - ay * q.y) - az * q.z;
    let n = y * y + x * x + z * z + w2 * w2;
    let k = 1.0 / n.sqrt();
    let basis = quat_to_mat(Quat { x: x * k, y: y * k, z: z * k, w: w2 * k });
    Transform { basis, origin }
}

impl Body {

    pub fn new_rigid(info: &RigidBodyInfo) -> Body {
        let clamp01 = |d: f32| {
            if 0.0 <= d {
                if d <= 1.0 { d } else { 1.0 }
            } else {
                0.0
            }
        };
        let mut b = Body {
            internal_type: 2,
            lin_damping: clamp01(info.lin_damping),
            ang_damping: clamp01(info.ang_damping),
            lin_sleep_threshold: info.lin_sleep_threshold,
            ang_sleep_threshold: info.ang_sleep_threshold,
            additional_damping: info.additional_damping,
            additional_damping_factor: info.additional_damping_factor,
            additional_lin_threshold_sqr: info.additional_lin_threshold_sqr,
            additional_ang_threshold_sqr: info.additional_ang_threshold_sqr,
            additional_ang_factor: info.additional_ang_factor,
            transform: info.transform,
            interp_transform: info.transform,
            friction: info.friction,
            rolling_friction: info.rolling_friction,
            restitution: info.restitution,
            shape: info.shape,

            update_revision: 1,
            max_lin_speed: info.max_lin_speed,
            max_ang_speed: info.max_ang_speed,
            ..Body::default()
        };
        b.set_mass_props(info.mass, info.local_inertia);

        b.inv_mass_factor = v3(b.inv_mass * b.lin_factor.x, b.inv_mass * b.lin_factor.y, b.inv_mass * b.lin_factor.z);
        b
    }

    pub fn set_gravity(&mut self, g: Vec3) {
        if self.inv_mass != 0.0 {
            let m = 1.0 / self.inv_mass;
            self.gravity = v3(m * g.x, m * g.y, m * g.z);
        }
        self.gravity_accel = g;
    }

    pub fn set_mass_props(&mut self, mass: f32, inertia: Vec3) {
        if mass == 0.0 {
            self.flags |= CF_STATIC;
            self.inv_mass = 0.0;
        } else {
            self.flags &= !CF_STATIC;
            self.inv_mass = 1.0 / mass;
        }
        self.gravity = v3(mass * self.gravity_accel.x, mass * self.gravity_accel.y, mass * self.gravity_accel.z);
        let inv = |v: f32| if v == 0.0 { 0.0 } else { 1.0 / v };
        self.inv_inertia_local = v3(inv(inertia.x), inv(inertia.y), inv(inertia.z));
        self.inv_mass_factor = v3(self.inv_mass * self.lin_factor.x, self.inv_mass * self.lin_factor.y, self.inv_mass * self.lin_factor.z);
        self.update_inertia_tensor();
    }

    pub fn apply_gravity(&mut self) {
        if self.flags & (CF_STATIC | CF_KINEMATIC) == 0 && self.activation_state != ISLAND_SLEEPING && self.activation_state != DISABLE_SIMULATION && self.rb_flags & 1 == 0 {
            self.total_force.y = self.lin_factor.y * self.gravity.y + self.total_force.y;
            self.total_force.z = self.lin_factor.z * self.gravity.z + self.total_force.z;
            self.total_force.x = self.lin_factor.x * self.gravity.x + self.total_force.x;
        }
    }

    pub fn apply_damping(&mut self, dt: f32) {
        let m = self.max_lin_speed;
        if 0.0 < m && m * m < self.lin_vel.len2_xy() {
            let v = self.lin_vel;
            let k = 1.0 / v.len2_xy().sqrt();
            self.lin_vel = v3(v.x * k * m, v.y * k * m, v.z * k * m);
        }
        let m = self.max_ang_speed;
        if 0.0 < m && m * m < self.ang_vel.len2_xy() {
            let v = self.ang_vel;
            let k = 1.0 / v.len2_xy().sqrt();
            self.ang_vel = v3(v.x * k * m, v.y * k * m, v.z * k * m);
        }
        let d = powf(1.0 - self.lin_damping, dt);
        self.lin_vel = v3(d * self.lin_vel.x, d * self.lin_vel.y, d * self.lin_vel.z);
        let d = powf(1.0 - self.ang_damping, dt);
        self.ang_vel = v3(d * self.ang_vel.x, d * self.ang_vel.y, d * self.ang_vel.z);
        if self.additional_damping {
            let w = self.ang_vel;
            if w.y * w.y + w.x * w.x + w.z * w.z < self.additional_ang_threshold_sqr && self.lin_vel.len2_xy() < self.additional_lin_threshold_sqr {
                let f = self.additional_damping_factor;
                self.ang_vel = v3(w.x * f, f * w.y, w.z * f);
                let v = self.lin_vel;
                self.lin_vel = v3(f * v.x, f * v.y, v.z * f);
            }
            let l = self.lin_vel.len2_xy().sqrt();
            if l < self.lin_damping {
                if l <= 0.005 {
                    self.lin_vel = Vec3::ZERO;
                } else {
                    let v = self.lin_vel;
                    let k = 1.0 / v.len2_xy().sqrt();
                    self.lin_vel = v3(v.x - v.x * k * 0.005, v.y - v.y * k * 0.005, v.z - v.z * k * 0.005);
                }
            }
            let l = self.ang_vel.len2_xy().sqrt();
            if l < self.ang_damping {
                if l <= 0.005 {
                    self.ang_vel = Vec3::ZERO;
                } else {
                    let v = self.ang_vel;
                    let k = 1.0 / v.len2_xy().sqrt();
                    self.ang_vel = v3(v.x - v.x * k * 0.005, v.y - v.y * k * 0.005, v.z - v.z * k * 0.005);
                }
            }
        }
    }

    pub fn predict_integrated_transform(&self, dt: f32) -> Transform {
        integrate_transform(&self.transform, self.lin_vel, self.ang_vel, dt)
    }

    pub fn proceed_to_transform(&mut self, t: &Transform) {
        self.interp_transform = if self.flags & CF_KINEMATIC == 0 { *t } else { self.transform };
        self.interp_lin_vel = self.lin_vel;
        self.interp_ang_vel = self.ang_vel;
        self.transform = *t;
        self.update_inertia_tensor();
    }

    pub fn update_inertia_tensor(&mut self) {
        let b = &self.transform.basis;
        let (ix, iy, iz) = (self.inv_inertia_local.x, self.inv_inertia_local.y, self.inv_inertia_local.z);
        let (m00, m01, m02) = (b.r[0].x, b.r[0].y, b.r[0].z);
        let (m10, m11, m12) = (b.r[1].x, b.r[1].y, b.r[1].z);
        let (m20, m21, m22) = (b.r[2].x, b.r[2].y, b.r[2].z);
        let (a0x, a0y, a0z) = (ix * m00, iy * m01, iz * m02);
        let (a1x, a1y, a1z) = (ix * m10, iy * m11, iz * m12);
        let (a2x, a2y, a2z) = (ix * m20, iy * m21, iz * m22);
        self.inv_inertia_world = Mat3 {
            r: [
                crate::math::lanes::rows_yx(b,v3(a0x,a0y,a0z)),
                crate::math::lanes::rows_yx(b,v3(a1x,a1y,a1z)),
                crate::math::lanes::rows_yx(b,v3(a2x,a2y,a2z)),
            ],
        };
    }

    pub fn apply_impulse(&mut self, imp: Vec3, rel: Vec3) {
        let im = self.inv_mass;
        if im != 0.0 {
            self.lin_vel.x = imp.x * self.lin_factor.x * im + self.lin_vel.x;
            self.lin_vel.y = imp.y * self.lin_factor.y * im + self.lin_vel.y;
            self.lin_vel.z = imp.z * self.lin_factor.z * im + self.lin_vel.z;
            let fz = self.lin_factor.z * imp.z;
            let fy = self.lin_factor.y * imp.y;
            let fx = self.lin_factor.x * imp.x;
            let t = v3(rel.y * fz - rel.z * fy, rel.z * fx - rel.x * fz, rel.x * fy - rel.y * fx);
            self.apply_torque_impulse(t);
        }
    }

    pub fn apply_torque_impulse(&mut self, t: Vec3) {
        let i = &self.inv_inertia_world;
        let wy = (t.y * i.r[1].y + t.x * i.r[1].x + t.z * i.r[1].z) * self.ang_factor.y;
        let wz = (t.y * i.r[2].y + t.x * i.r[2].x + t.z * i.r[2].z) * self.ang_factor.z;
        let wx = (t.y * i.r[0].y + t.x * i.r[0].x + t.z * i.r[0].z) * self.ang_factor.x;
        self.ang_vel = v3(wx + self.ang_vel.x, wy + self.ang_vel.y, wz + self.ang_vel.z);
    }

    pub fn apply_force(&mut self, f: Vec3, rel: Vec3) {
        self.total_force.y = f.y * self.lin_factor.y + self.total_force.y;
        self.total_force.z = f.z * self.lin_factor.z + self.total_force.z;
        self.total_force.x = f.x * self.lin_factor.x + self.total_force.x;
        let fx = f.x * self.lin_factor.x;
        let fz = f.z * self.lin_factor.z;
        let fy = f.y * self.lin_factor.y;
        self.total_torque.y = (rel.z * fx - rel.x * fz) * self.ang_factor.y + self.total_torque.y;
        self.total_torque.x = (rel.y * fz - rel.z * fy) * self.ang_factor.x + self.total_torque.x;
        self.total_torque.z = (rel.x * fy - rel.y * fx) * self.ang_factor.z + self.total_torque.z;
    }

    #[inline]
    pub fn set_activation_state(&mut self, s: i32) {
        if self.activation_state != DISABLE_DEACTIVATION && self.activation_state != DISABLE_SIMULATION {
            self.activation_state = s;
        }
    }

    #[inline]
    pub fn activate(&mut self, force: bool) {
        if force || self.flags & (CF_STATIC | CF_KINEMATIC) == 0 {
            if self.activation_state != DISABLE_DEACTIVATION && self.activation_state != DISABLE_SIMULATION {
                self.activation_state = ACTIVE;
            }
            self.deactivation_time = 0.0;
        }
    }

    #[inline]
    pub fn is_rigid(&self) -> bool {
        self.internal_type & 2 != 0
    }
}
