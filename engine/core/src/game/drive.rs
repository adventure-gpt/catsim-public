use crate::bt::vehicle::WheelInfo;
use crate::bt::{Body, BodyId};
use crate::game::rb_state::{RbState, forward_speed, quat_rotate};
use crate::math::{Vec3, v3};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curve {
    pub keys: Vec<(f32, f32)>,
}

impl Curve {
    pub fn new(keys: &[(f32, f32)]) -> Curve {
        Curve { keys: keys.to_vec() }
    }

    pub fn eval(&self, x: f32, default: f32) -> f32 {
        let k = &self.keys;
        let n = k.len();
        if n == 0 {
            return default;
        }
        if n < 2 || x <= k[0].0 {
            return k[0].1;
        }
        if k[n - 1].0 <= x {
            return k[n - 1].1;
        }
        let mut i = 1;
        while k[i].0 <= x {
            i += 1;
            if n <= i {
                return k[n - 1].1;
            }
        }
        let width = k[i].0 - k[i - 1].0;
        if 0.0 < width {
            let t = (x - k[i - 1].0) / width;
            return (k[i].1 - k[i - 1].1) * t + k[i - 1].1;
        }
        k[i - 1].1
    }
}

#[inline]
pub fn sensitivity(mut x: f32, sens: f32, limit: f32) -> f32 {
    if limit < sens || (sens - limit).abs() < 0.001 {
        x = if x <= 0.0 {
            if 0.0 <= x { 0.0 } else { -1.0 }
        } else {
            1.0
        };
    }
    let y = x * sens;
    if y < -1.0 {
        return -1.0;
    }
    if 1.0 <= y { 1.0 } else { y }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WheelContact {
    pub has_contact: bool,
    pub world: bool,
    pub changed_time: f32,
    pub actor: Option<BodyId>,
    pub location: Vec3,
    pub normal: Vec3,
    pub lat_dir: Vec3,
    pub long_dir: Vec3,
    pub sticky: bool,
}

#[derive(Clone, Debug, Default)]
pub struct WheelSim {
    pub steer_factor: f32,
    pub lat_friction_curve: Curve,
    pub long_friction_curve: Curve,
    pub handbrake_lat_curve: Curve,
    pub handbrake_long_curve: Curve,
    pub wall_friction_curve: Curve,
    pub contact: WheelContact,
    pub contact_notified: bool,
    pub friction_input: f32,
}

#[derive(Clone, Debug, Default)]
pub struct VehicleSim {
    pub steer_curve: Curve,
    pub drive_torque: f32,
    pub drive_curve: Curve,
    pub brake_torque: f32,
    pub handbrake_steer_curve: Curve,
    pub handbrake_rise: f32,
    pub handbrake_fall: f32,
    pub stop_threshold: f32,
    pub idle_brake: f32,
    pub opposite_brake: f32,
    pub steering_sensitivity: f32,
    pub tire_friction_scale: f32,
    pub out_throttle: f32,
    pub out_steer: f32,
    pub out_brake: f32,
    pub out_handbrake: f32,
    pub wheels_steer: [f32; 4],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroundInfo {
    pub world: bool,
    pub sticky: bool,
}

pub fn wheel_refresh(s: &mut WheelSim, w: &WheelInfo, ground: Option<GroundInfo>, frame_time: f32) {
    let was = s.contact.has_contact;
    let keep_time = s.contact.changed_time;
    s.contact = WheelContact { changed_time: keep_time, ..WheelContact::default() };
    if let (Some(id), Some(g)) = (w.ground, ground) {
        s.contact.actor = Some(id);
        s.contact.location = v3(w.contact.x * 50.0, w.contact.y * 50.0, w.contact.z * 50.0);
        s.contact.normal = w.normal;
        s.contact.has_contact = true;
        s.contact.world = g.world;
        s.contact.sticky = g.sticky;
    }
    if was != s.contact.has_contact {
        s.contact.changed_time = frame_time;
    }
}

pub fn wheel_dirs(s: &mut WheelSim, w: &WheelInfo) {
    let lat = w.basis.col(1);
    let n = s.contact.normal;
    s.contact.lat_dir = lat;
    s.contact.long_dir = v3(n.z * lat.y - n.y * lat.z, n.x * lat.z - lat.x * n.z, lat.x * n.y - n.x * lat.y);
}

pub fn wheel_friction(s: &mut WheelSim, w: &mut WheelInfo, c: &Body, tire_scale: f32, throttle: f32, engine: f32, brake: f32, handbrake: f32) {
    s.friction_input = 0.0;
    if s.contact.has_contact {
        let o = c.transform.origin;
        let (rz, ry, rx) = (w.hard_point.z - o.z, w.hard_point.y - o.y, w.hard_point.x - o.x);
        let (a, v) = (c.ang_vel, c.lin_vel);
        let vx = ((a.y * rz - a.z * ry) + v.x) * 50.0;
        let vy = ((a.z * rx - a.x * rz) + v.y) * 50.0;
        let vz = ((a.x * ry - a.y * rx) + v.z) * 50.0;
        let (lat, lon) = (s.contact.lat_dir, s.contact.long_dir);
        let side = (vy * lat.y + vx * lat.x + vz * lat.z).abs();
        let mut input = 0.0;
        if 5.0 < side {
            input = side / ((vy * lon.y + vx * lon.x + vz * lon.z).abs() + side);
            s.friction_input = input;
        }
        let sticky = s.contact.sticky && throttle != 0.0;
        let mut flat = s.lat_friction_curve.eval(input, 1.0);
        let mut flong = s.long_friction_curve.eval(s.friction_input, 1.0);
        if 0.0 < handbrake {
            let a = s.handbrake_lat_curve.eval(s.friction_input, 1.0);
            flat = flat * ((a - 1.0) * handbrake + 1.0);
            let b = s.handbrake_long_curve.eval(s.friction_input, 1.0);
            flong = flong * ((b - 1.0) * handbrake + 1.0);
        }
        if !sticky {
            let k = s.wall_friction_curve.eval(s.contact.normal.z, 1.0);
            flat = flat * k;
            flong = flong * k;
        }
        w.long_friction = tire_scale * flong;
        w.lat_friction = tire_scale * flat;
    }
    w.engine_force = engine * 0.02;
    w.brake = brake * 0.02;
}

#[inline]
pub fn num_wheel_contacts(wheels: &[WheelSim]) -> i32 {
    wheels.iter().filter(|w| w.contact.has_contact).count() as i32
}

#[inline]
pub fn num_world_contacts(wheels: &[WheelSim]) -> i32 {
    wheels.iter().filter(|w| w.contact.world).count() as i32
}

pub fn ground_normal(wheels: &[WheelSim], world_contact_normal: Vec3) -> Vec3 {
    if 0 < num_world_contacts(wheels) {
        let (mut x, mut y, mut z) = (0.0f32, 0.0f32, 0.0f32);
        for w in wheels {
            if w.contact.world {
                x = x + w.contact.normal.x;
                y = y + w.contact.normal.y;
                z = z + w.contact.normal.z;
            }
        }
        let n2 = x * x + y * y + z * z;
        if n2 != 1.0 {
            if 1e-08 <= n2 {
                let k = 1.0 / n2.sqrt();
                return v3(k * x, k * y, k * z);
            }
            return Vec3::ZERO;
        }
        return v3(x, y, z);
    }
    world_contact_normal
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DriveInputs {
    pub throttle: f32,
    pub steer: f32,
    pub handbrake: bool,
    pub driving: bool,
    pub input_restriction: u8,
}

pub fn update_outputs(sim: &mut VehicleSim, inp: &DriveInputs, state: &RbState, wheels_touching: i32, dt: f32) {
    let steer = sensitivity(inp.steer, sim.steering_sensitivity, 10.0);
    let t = inp.throttle;
    let mut throttle = 0.0f32;
    let mut brake = 0.0f32;
    let mut hb_target = 0.0f32;
    if inp.driving {
        throttle = t;
        if !inp.handbrake {
            let speed = forward_speed(state);
            let stop = sim.stop_threshold;
            let mut coast = false;
            if t <= 0.01 {
                if t < -0.01 {
                    if stop < speed {
                        brake = sim.opposite_brake;
                    }
                } else {
                    coast = true;
                }
            } else if speed < -stop {
                brake = sim.opposite_brake;
            } else if inp.input_restriction == 1 {
                coast = true;
            }
            if coast {
                throttle = 0.0;
                brake = if stop <= speed.abs() { sim.idle_brake } else { 1.0 };
            }
        } else {
            hb_target = 1.0;
        }
    }
    if wheels_touching == 0 {
        brake = 0.0;
    } else if brake != 0.0 {
        throttle = 0.0;
    }
    let hb = sim.out_handbrake;
    sim.out_throttle = throttle;
    let d = hb_target - hb;
    sim.out_brake = brake;
    sim.out_steer = steer;
    let rate = if hb == 0.0 || (0.0 < d) == (0.0 < hb) { sim.handbrake_rise } else { sim.handbrake_fall };
    let r = rate * dt;
    let mut step = -r;
    if -r <= d {
        step = d;
        if r <= d {
            step = r;
        }
    }
    sim.out_handbrake = step + hb;
}

pub fn drive(sim: &mut VehicleSim, wheels: &mut [WheelSim; 4], bt: &mut [WheelInfo; 4], c: &Body, state: &RbState, driving: bool, throttle_input: f32) {
    let f = quat_rotate(state.quat, v3(1.0, 0.0, 0.0));
    let speed = f.y * state.velocity.y + f.x * state.velocity.x + f.z * state.velocity.z;
    let abs_speed = speed.abs();
    let mut angle = sim.steer_curve.eval(abs_speed, 0.0);
    if 0.0 < sim.out_handbrake {
        let b = sim.handbrake_steer_curve.eval(abs_speed, 0.0);
        angle = (b - angle) * sim.out_handbrake + angle;
    }
    let steer_rad = angle * sim.out_steer * 0.017453292;
    let wheel_steer = if driving { steer_rad } else { 0.0 };
    let mut engine = 0.0f32;
    let mut brake = 0.0f32;
    if sim.out_throttle != 0.0 {
        let k = sim.drive_curve.eval(speed, 0.0);
        engine = k * (sim.out_throttle * sim.drive_torque);
    }
    if sim.out_brake != 0.0 {
        brake = sim.out_brake * sim.brake_torque;
    }
    engine = engine * 0.25;
    if num_wheel_contacts(&wheels[..]) < 3 {
        engine = engine * 0.25;
    }
    for i in 0..4 {
        bt[i].steer = wheel_steer * wheels[i].steer_factor;
        sim.wheels_steer[i] = steer_rad;
    }
    crate::bt::vehicle::update_wheel_bases(bt, c);
    for i in 0..4 {
        wheel_dirs(&mut wheels[i], &bt[i]);
        wheel_friction(&mut wheels[i], &mut bt[i], c, sim.tire_friction_scale, throttle_input, engine, brake * 0.25, sim.out_handbrake);
    }
}
