use crate::bt::vehicle::{self, Vehicle, VehicleWorld};
use crate::bt::{Body, BodyId, ISLAND_SLEEPING};
use crate::game::components::*;
use crate::game::drive::{self, DriveInputs, GroundInfo, VehicleSim, WheelSim};
use crate::game::rb_state::{self, ForceMode, RbState, ReadLimits, add_force, add_torque, quat_to_matrix};
use crate::math::rot::{self, Rotator, app_round};
use crate::math::{Vec3, v3};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VehicleInputs {
    pub throttle: f32,
    pub steer: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub dodge_forward: f32,
    pub dodge_right: f32,
    pub handbrake: bool,
    pub jump: bool,
    pub activate_boost: bool,
    pub holding_boost: bool,
    pub jumped: bool,
    pub grab: bool,
    pub button_mash: bool,
    pub air_roll: bool,
}

#[inline]
pub fn quantize_axis(v: f32) -> f32 {
    let mut x = if v < 0.0 {
        let x = v * 128.0;
        if x <= -128.0 { -128.0 } else { x }
    } else {
        let x = v * 127.0;
        if 127.0 <= x { 127.0 } else { x }
    };
    x = x + 128.0;
    let k = (crate::math::rot::cvtss2si(x + x + 0.5) >> 1) as u32;
    let f = (k & 0xff) as i32 as f32 - 128.0;
    if (k as u8) < 0x80 { f * 0.0078125 } else { f / 127.0 }
}

#[inline]
fn clamp_axis(v: f32) -> f32 {
    if -1.0 <= v {
        if 1.0 <= v { 1.0 } else { v }
    } else {
        -1.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldContact {
    pub has_contact: bool,
    pub location: Vec3,
    pub velocity: Vec3,
    pub normal: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct CarPhysicsConfig {
    pub supersonic_speed: f32,
    pub supersonic_hysteresis: f32,
    pub supersonic_time: f32,
    pub quantize_physics: bool,
    pub jump_leave_ground_time: f32,
}

#[derive(Clone, Debug)]
pub struct Car {
    pub body: BodyId,
    pub old_state: RbState,
    pub state: RbState,
    pub world_contact: WorldContact,
    pub limits: ReadLimits,
    pub rolling_friction: f32,
    pub config: CarPhysicsConfig,
    pub sticky_base: f32,
    pub sticky_full: f32,
    pub auto_flip_torque: f32,
    pub auto_flip_force: f32,
    pub driving: bool,
    pub jumped: bool,
    pub double_jumped: bool,
    pub on_ground: bool,
    pub supersonic: bool,
    pub podium: bool,
    pub input: VehicleInputs,
    pub input_restriction: u8,
    pub has_ai_controller: bool,
    pub num_wheels_for_grounded: i32,
    pub time_below_supersonic: f32,
    pub max_time_for_dodge: f32,
    pub max_num_jumps: i32,

    pub body_scale: f32,

    pub dodge_input_threshold: f32,

    pub gravity_z: f32,

    pub game_active: bool,
    pub sim: VehicleSim,
    pub wheels: [WheelSim; 4],
    pub vehicle: Vehicle,
    pub air: AirControl,
    pub boost: Boost,
    pub jump: Jump,
    pub double_jump: DoubleJump,
    pub dodge: Dodge,
    pub flip: FlipCar,

    pub wants_constraint: bool,
}

#[inline]
pub fn ground_info(b: &Body) -> GroundInfo {
    GroundInfo { world: matches!(b.owner, crate::bt::Owner::World), sticky: b.sticky_wheels }
}

impl Car {

    #[inline]
    pub fn is_on_ground(&self) -> bool {
        self.num_wheels_for_grounded <= drive::num_wheel_contacts(&self.wheels)
    }

    pub fn read_state(&mut self, b: &mut Body) {
        let s = rb_state::read_body(b, &self.limits);
        self.old_state = s;
        self.state = s;
        if self.config.quantize_physics {
            rb_state::write_body(b, &s);
        }
        let k = self.rolling_friction;
        if 0.0 < k && self.world_contact.has_contact {
            let w = self.state.spin;
            let len = f64::from(w.y * w.y + w.x * w.x + w.z * w.z).sqrt() as f32;
            let r = len / self.limits.max_angular;
            let t = v3(-w.x * k * r, -w.y * k * r, -w.z * k * r);
            if 0.0001 <= t.x.abs() || 0.0001 <= t.y.abs() || 0.0001 <= t.z.abs() {
                add_torque(b, t, ForceMode::VelocityChange);
            }
        }
    }

    pub fn auto_roll(&self, b: &mut Body) {
        let n = drive::ground_normal(&self.wheels, self.world_contact.normal);
        let m = quat_to_matrix(self.state.quat);
        let (fw, rt) = (m.r[0], m.r[1]);
        let cx = fw.z * n.y - fw.y * n.z;
        let (nny, nnz) = (-n.y, -n.z);
        let cy = fw.x * n.z - fw.z * n.x;
        let nnx = -n.x;
        let cz = fw.y * n.x - fw.x * n.y;
        let a = cx * rt.x + rt.y * cy + rt.z * cz;
        let a = if 0.0 <= a {
            if 1.0 <= a { 1.0 } else { a }
        } else {
            0.0
        };
        let c = (nny * cz - nnz * cy) * fw.x + (cx * nnz - nnx * cz) * fw.y + (nnx * cy - cx * nny) * fw.z;
        let c = if 0.0 <= c {
            if 1.0 <= c { 1.0 } else { c }
        } else {
            0.0
        };
        let (mut px, mut py, mut pz) = (fw.x, fw.y, fw.z);
        if 0.0 <= rt.y * n.y + rt.x * n.x + rt.z * n.z {
            px = -fw.x;
            py = -fw.y;
            pz = -fw.z;
        }
        let ta = (1.0 - a) * self.auto_flip_torque;
        let (mut qx, mut qy, mut qz) = (rt.x, rt.y, rt.z);
        if fw.y * n.y + fw.x * n.x + fw.z * n.z < 0.0 {
            qx = -rt.x;
            qy = -rt.y;
            qz = -rt.z;
        }
        let tb = (1.0 - c) * self.auto_flip_torque;
        let f = self.auto_flip_force;
        let v = v3(f * nnx, f * nny, f * nnz);
        if 0.0001 <= v.x.abs() || 0.0001 <= v.y.abs() || 0.0001 <= v.z.abs() {
            add_force(b, v, ForceMode::Accel);
        }
        add_torque(b, v3(tb * qx + ta * px, tb * qy + ta * py, tb * qz + ta * pz), ForceMode::Accel);
    }

    pub fn sticky_force(&self, b: &mut Body) {
        if !self.wheels.iter().any(|w| w.contact.sticky) {
            return;
        }
        let n = drive::ground_normal(&self.wheels, self.world_contact.normal);
        let s = if self.input.throttle != 0.0 || 25.0 < rb_state::forward_speed(&self.state).abs() { self.sticky_full } else { self.sticky_base };
        let base = self.sticky_base;
        let g = self.gravity_z;
        let sign = if g <= 0.0 { 1.0 } else { -1.0 };
        let k = g * (n.z.abs() * (base - s) + s) * sign;
        let v = v3(k * n.x, k * n.y, k * n.z);
        if 0.0001 <= v.x.abs() || 0.0001 <= v.y.abs() || 0.0001 <= v.z.abs() {
            add_force(b, v, ForceMode::Accel);
        }
    }

    pub fn update_on_ground(&mut self) {
        let new = self.is_on_ground();
        if new != self.on_ground {
            self.on_ground = new;
            if new {
                self.jumped = false;
                self.double_jumped = false;
            }
        }
    }

    #[inline]
    pub fn frame_tick(&mut self) {
        self.update_on_ground();
    }

    pub fn pre_tick(&mut self, b: &mut Body, dt: f32) {
        self.read_state(b);
        let flipping = self.dodge.c.active;
        if !flipping && self.input.throttle != 0.0 && drive::num_world_contacts(&self.wheels) < 4 {
            if 0 < drive::num_world_contacts(&self.wheels) || self.world_contact.has_contact {
                self.auto_roll(b);
            }
        }
        self.sticky_force(b);
        self.wants_constraint = !self.driving || self.podium;

        b.check_collide_with = self.wants_constraint as i32;

        let v = self.state.velocity;
        let sp2 = v.y * v.y + v.x * v.x + v.z * v.z;
        let sc = &self.config;
        if sp2 < sc.supersonic_speed * sc.supersonic_speed {
            if self.supersonic {
                let t = dt + self.time_below_supersonic;
                self.time_below_supersonic = t;
                let low = sc.supersonic_speed - sc.supersonic_hysteresis;
                if sc.supersonic_time <= t || sp2 < low * low {
                    self.supersonic = false;
                }
            }
        } else {
            self.supersonic = true;
            self.time_below_supersonic = 0.0;
        }

        if self.jumped && self.is_on_ground() {
            let t = self.jump.c.inactive_time();
            if self.config.jump_leave_ground_time <= t {
                self.jumped = false;
                self.double_jumped = false;
                if !self.on_ground {
                    self.update_on_ground();
                }
            }
        }
        self.run_components(b, dt);
    }

    pub fn run_components(&mut self, b: &mut Body, dt: f32) {
        let touching = drive::num_wheel_contacts(&self.wheels);
        let st = self.state;

        air_pre(&mut self.air, self.driving, touching);
        if self.air.c.active {
            let locked = self.dodge.c.active || 0.3 > self.dodge.c.time;
            match self.air.robin {

                Some(p) => {
                    if !self.world_contact.has_contact {
                        if self.boost.c.active {
                            robin_torque(b, &st, &p, ForceMode::Accel);
                        }
                        air_forces(&self.air, b, &st, (self.input.pitch, self.input.yaw, self.input.roll), locked, self.input.throttle, self.boost.c.active);
                    }
                }
                None => air_forces(&self.air, b, &st, (self.input.pitch, self.input.yaw, self.input.roll), locked, self.input.throttle, self.boost.c.active),
            }
        }
        self.air.c.time = dt + self.air.c.time;

        let on_ground = self.is_on_ground();
        boost_pre(&mut self.boost, self.input.activate_boost, self.driving, on_ground, dt);
        if self.boost.c.active {
            boost_forces(&self.boost, b, &st);
        }
        self.boost.c.time = dt + self.boost.c.time;

        jump_pre(&mut self.jump, self.podium);
        if self.jump.c.active {
            jump_forces(&self.jump, b, &st, self.jump.c.time);
        }
        self.jump.c.time = dt + self.jump.c.time;

        double_jump_pre(&mut self.double_jump);
        if self.double_jump.c.active {
            let t = self.double_jump.c.time;
            double_jump_forces(&mut self.double_jump, b, &st, t);
        }
        self.double_jump.c.time = dt + self.double_jump.c.time;

        dodge_pre(&mut self.dodge, on_ground);
        if self.dodge.c.active {
            dodge_forces(&self.dodge, b, &st, self.limits.max_linear, self.input.dodge_forward, self.body_scale, self.dodge.c.time);
        }
        self.dodge.c.time = dt + self.dodge.c.time;

        flip_pre(&mut self.flip, on_ground);
        if self.flip.c.active {
            flip_forces(&self.flip, b, &st, self.flip.c.time);
        }
        self.flip.c.time = dt + self.flip.c.time;
    }

    pub fn set_driving(&mut self, on: bool) {
        if on != self.driving {
            self.driving = on;
            if !on {
                self.set_vehicle_input(VehicleInputs::default());
            }
        }
    }

    pub fn push_input(&mut self, ctrl: &mut VehicleInputs) {
        ctrl.activate_boost = ctrl.holding_boost && 0.0 < self.boost.amount;
        let mut out = *ctrl;
        out.jump = ctrl.jump || ctrl.jumped;
        ctrl.jumped = false;
        self.set_vehicle_input(out);
    }

    pub fn set_vehicle_input(&mut self, mut new: VehicleInputs) {
        let old_jump = self.input.jump;
        if self.is_on_ground() && new.activate_boost && 0.0 < self.boost.amount {
            new.throttle = 1.0;
        }
        self.store_input(new);
        if self.input.jump != old_jump {
            if self.input.jump {
                self.on_jump_pressed();
            } else {
                self.on_jump_released();
            }
        }
    }

    pub fn store_input(&mut self, new: VehicleInputs) {
        let mut i = new;
        i.throttle = clamp_axis(new.throttle);
        i.steer = clamp_axis(new.steer);
        i.pitch = clamp_axis(new.pitch);
        i.yaw = clamp_axis(new.yaw);
        i.roll = clamp_axis(new.roll);
        i.dodge_forward = clamp_axis(new.dodge_forward);
        i.dodge_right = clamp_axis(new.dodge_right);
        if self.input_restriction == 1 {
            if 0.0 < i.dodge_forward {
                i.dodge_forward = i.dodge_forward * -1.0;
            }
            if self.has_ai_controller && 0.0 < i.throttle {
                i.throttle = i.throttle * -1.0;
            }
        }
        i.throttle = quantize_axis(i.throttle);
        i.steer = quantize_axis(i.steer);
        i.pitch = quantize_axis(i.pitch);
        i.yaw = quantize_axis(i.yaw);
        i.roll = quantize_axis(i.roll);
        i.dodge_forward = quantize_axis(i.dodge_forward);
        i.dodge_right = quantize_axis(i.dodge_right);
        self.input = i;
    }

    #[inline]
    pub fn has_dodge_input(&self) -> bool {
        self.input.dodge_forward.abs() + self.input.dodge_right.abs() + self.input.roll.abs() >= self.dodge_input_threshold
    }

    #[inline]
    pub fn within_dodge_window(&self) -> bool {
        let inner = fmin(self.dodge.c.inactive_time(), self.double_jump.c.inactive_time());
        fmin(self.jump.c.inactive_time(), inner) <= self.max_time_for_dodge
    }

    #[inline]
    fn component_can_activate(&self, c: &Component) -> bool {
        self.driving && c.min_inactive_time <= c.inactive_time()
    }

    pub fn on_jump_pressed(&mut self) {
        if !self.driving {
            return;
        }
        if !self.dodge.c.active {
            if self.flip.c.active {
                return;
            }
            if self.flip_can_activate() {
                self.flip_activate();
                return;
            }
        }
        let can_jump = !self.game_active || 0 < self.max_num_jumps;
        let grounded = self.is_on_ground();
        if can_jump && !self.jumped && grounded {
            self.jumped = true;
            if !self.jump.c.active {
                self.jump.c.set_active(true);
                self.jump.deactivate = false;
            }
        }

        if !self.double_jumped && 0 < self.max_num_jumps - 1 && !grounded {
            let within = self.within_dodge_window();
            if !self.jumped || within {
                self.jumped = true;
                self.double_jumped = true;
                if self.component_can_activate(&self.dodge.c) && self.has_dodge_input() {
                    self.dodge_activate();
                } else {
                    self.double_jump.c.set_active(true);
                }
            }
        }
    }

    #[inline]
    pub fn on_jump_released(&mut self) {
        if self.jump.c.active {
            self.jump.deactivate = true;
        }
    }

    pub fn dodge_activate(&mut self) {
        let was = self.dodge.c.active;
        self.dodge.c.set_active(true);
        if was {
            return;
        }
        let d = &mut self.dodge;
        let (x0, y0) = (self.input.dodge_forward, self.input.dodge_right + self.input.roll);
        let n2 = x0 * x0 + y0 * y0;
        let (dx, dy) = if n2 != 1.0 {
            if 1e-08 <= n2 {
                let k = 1.0 / n2.sqrt();
                (k * x0, k * y0)
            } else {
                (0.0, 0.0)
            }
        } else {
            (x0, y0)
        };
        let ty = d.forward_torque * dx * d.torque_scale;
        let tx = d.side_torque * -dy * d.torque_scale;
        let sign = if self.gravity_z > 0.0 { -1.0 } else { 1.0 };
        d.direction = v3(dx, dy * sign, 0.0);
        d.torque = v3(app_round(tx) as f32, app_round(ty) as f32, 0.0);
    }

    pub fn flip_can_activate(&self) -> bool {
        if self.flip.c.active || !self.component_can_activate(&self.flip.c) {
            return false;
        }
        if self.world_contact.has_contact && self.world_contact.normal.z > 0.707 {
            let m = quat_to_matrix(self.state.quat);
            let up = rot::rotator_to_matrix(rot::matrix_to_rotator(&m)).r[2];
            let n = self.world_contact.normal;
            return n.x * up.x + n.y * up.y + n.z * up.z < -0.966;
        }
        false
    }

    pub fn flip_activate(&mut self) {
        self.flip.c.set_active(true);
        let m = quat_to_matrix(self.state.quat);
        let r: Rotator = rot::matrix_to_rotator(&m);
        let mut roll = r.roll & 0xffff;
        if roll > 32767 {
            roll -= 65536;
        }
        let a = roll as f32;
        self.flip.time = self.flip.default_time * (a.abs() / 32768.0);
        self.flip.flip_right = a > 0.0;
    }

    pub fn vehicle_update<W: VehicleWorld + ?Sized>(&mut self, world: &mut W, dt: f32, frame_time: f32) {
        vehicle::update_rays_and_contacts(&mut self.vehicle, world);
        self.vehicle_callback(world, dt, frame_time);
        vehicle::update_suspension_and_friction(&mut self.vehicle, world, dt);
    }

    pub fn vehicle_callback<W: VehicleWorld + ?Sized>(&mut self, world: &mut W, dt: f32, frame_time: f32) {
        let body = world.body(self.body);
        self.state = rb_state::read_body(body, &self.limits);
        for i in 0..4 {
            let g = self.vehicle.wheels[i].ground.map(|id| ground_info(world.body(id)));
            drive::wheel_refresh(&mut self.wheels[i], &self.vehicle.wheels[i], g, frame_time);
        }
        let touching = drive::num_wheel_contacts(&self.wheels);
        let inp = DriveInputs { throttle: self.input.throttle, steer: self.input.steer, handbrake: self.input.handbrake, driving: self.driving, input_restriction: self.input_restriction };
        drive::update_outputs(&mut self.sim, &inp, &self.state, touching, dt);
        let body = world.body(self.body);
        drive::drive(&mut self.sim, &mut self.wheels, &mut self.vehicle.wheels, body, &self.state, self.driving, self.input.throttle);
    }

    pub fn post_tick(&mut self, b: &mut Body) {
        rb_state::activate(b);
        self.state = rb_state::read_body(b, &self.limits);
    }

    #[inline]
    pub fn body_asleep(b: &Body) -> bool {
        b.activation_state == ISLAND_SLEEPING
    }
}

#[inline]
fn fmin(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}
