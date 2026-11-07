use crate::bt::BodyId;
use crate::bt::vehicle::{Vehicle, WheelInfo};
use crate::game::car::{Car, CarPhysicsConfig, VehicleInputs, WorldContact};
use crate::game::components::*;
use crate::game::drive::{Curve, VehicleSim, WheelSim};
use crate::game::rb_state::{RbState, ReadLimits};
use crate::math::{Mat3, Vec3, v3};

#[derive(Clone, Copy, Debug)]
pub struct Axle {
    pub connection: [u32; 3],
    pub rest: u32,
    pub radius: u32,

    pub rest_pos: [u32; 2],
}

#[derive(Clone, Copy, Debug)]
pub struct Preset {
    pub name: &'static str,

    pub half: [u32; 3],

    pub offset_xz: [u32; 2],
    pub min_hit_dist2: u32,
    pub front: Axle,
    pub back: Axle,
}

#[inline(always)]
const fn fb(b: u32) -> f32 {
    f32::from_bits(b)
}

pub static PRESETS: std::sync::LazyLock<Vec<Preset>> = std::sync::LazyLock::new(|| {
    crate::data::table("presets")
        .iter()
        .map(|r| Preset {
            name: r.s(0),
            half: r.wn(0),
            offset_xz: r.wn(3),
            min_hit_dist2: r.w(5),
            front: Axle { connection: r.wn(6), rest: r.w(9), radius: r.w(10), rest_pos: r.wn(11) },
            back: Axle { connection: r.wn(13), rest: r.w(16), radius: r.w(17), rest_pos: r.wn(18) },
        })
        .collect()
});

pub const CAR_GROUP: u16 = 0x08;
pub const CAR_MASK: u16 = 0x81ed;
pub const BALL_GROUP: u16 = 0x80;
pub const BALL_MASK: u16 = 0x82e9;
pub const WORLD_GROUP: u16 = 0x01;
pub const WORLD_MASK: u16 = 0xffff;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodySpawn {
    pub product: u32,
    pub name: &'static str,
    pub preset: &'static str,
    pub mesh_z: u32,
    pub forward: f32,

    pub mesh_x: f32,

    pub mesh_bounds: [u32; 7],
}

impl BodySpawn {
    pub fn mesh_bounds(&self) -> [f32; 7] {
        self.mesh_bounds.map(f32::from_bits)
    }
}

pub static BODIES: std::sync::LazyLock<Vec<BodySpawn>> = std::sync::LazyLock::new(|| {
    crate::data::table("bodies")
        .iter()
        .map(|r| BodySpawn { product: r.w(0), name: r.s(0), preset: r.s(1), mesh_z: r.w(1), forward: r.f(2), mesh_x: r.f(3), mesh_bounds: r.wn(4) })
        .collect()
});

pub fn body_spawn(key: &str) -> Option<&'static BodySpawn> {
    BODIES.iter().find(|b| b.name.eq_ignore_ascii_case(key) || key.parse::<u32>().ok() == Some(b.product))
}

pub fn spawn_body_origin(spot: Vec3, yaw: i32, team: u8, b: &BodySpawn) -> Vec3 {
    let (mut x, mut y) = (spot.x, spot.y);
    if team == 1 {
        let t = crate::math::rot::sine_table();
        let (bx, by) = (-x, -y);
        let (c, s) = (crate::math::rot::tcos(t, 32768), crate::math::rot::tsin(t, 32768));
        x = c * bx - s * by;
        y = s * bx + c * by;
    }
    actor_body_origin(v3(x, y, spot.z), yaw, b)
}

/// A car body's first origin from its spawn point actor's own location (`spawn_body_origin` once the orange spot has
/// been turned): the spawn selector's spots are those floats already, an orange one turned half a turn through the
/// sine table, so it is not turned again (turned twice its x / y were 2-14 float steps off at every orange spawn).
pub fn actor_body_origin(loc: Vec3, yaw: i32, b: &BodySpawn) -> Vec3 {
    let t = crate::math::rot::sine_table();
    let (x, mut y) = (loc.x, loc.y);
    if b.forward == 0.0 && y == -2560.0 && (x == 2048.0 || x == -2048.0) {
        y = f32::from_bits(0xc51f_ffff);
    }
    let (cy, sy) = (crate::math::rot::tcos(t, yaw), crate::math::rot::tsin(t, yaw));
    v3((x + b.forward * cy) * 0.02, (y + b.forward * sy) * 0.02, (36.0 + fb(b.mesh_z)) * 0.02)
}

pub fn preset(name: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.name == name)
}

pub const STEER_RAW: [(f32, f32); 6] = [(0.0, 35.0), (500.0, 20.0), (1000.0, 11.0), (1500.0, 6.25), (1750.0, 5.0), (3000.0, 2.0)];
pub const HANDBRAKE_STEER_RAW: [(f32, f32); 2] = [(0.0, 25.0), (2500.0, 7.5)];

pub const DRIVE_CURVE: [(f32, f32); 5] = [(-1410.0, 0.0), (-1400.0, fb(0x3dcccccd)), (0.0, 1.0), (1400.0, fb(0x3dcccccd)), (1410.0, 0.0)];

pub static STEER_FOLDED: std::sync::LazyLock<Vec<[u32; 6]>> = std::sync::LazyLock::new(|| crate::data::table("steer_folded").iter().map(|r| r.wn(0)).collect());
pub static HANDBRAKE_FOLDED: std::sync::LazyLock<Vec<[u32; 2]>> = std::sync::LazyLock::new(|| crate::data::table("handbrake_folded").iter().map(|r| r.wn(0)).collect());

impl Preset {
    pub fn half_extents(&self) -> Vec3 {
        v3(fb(self.half[0]), fb(self.half[1]), fb(self.half[2]))
    }
    pub fn offset(&self) -> Vec3 {
        v3(fb(self.offset_xz[0]), 0.0, fb(self.offset_xz[1]))
    }
    pub fn index(&self) -> usize {
        PRESETS.iter().position(|p| p.name == self.name).unwrap()
    }

    pub fn steer_curve(&self) -> Curve {
        if let Some(v) = vehicle_variant(self.name) {
            return Curve { keys: v.steer.iter().map(|(x, y)| (fb(*x), fb(*y))).collect() };
        }
        let k = &STEER_FOLDED[self.index()];
        Curve { keys: STEER_RAW.iter().zip(k.iter()).map(|(r, b)| (r.0, fb(*b))).collect() }
    }
    pub fn handbrake_steer_curve(&self) -> Curve {
        if let Some(v) = vehicle_variant(self.name) {
            return Curve { keys: v.handbrake_steer.iter().map(|(x, y)| (fb(*x), fb(*y))).collect() };
        }
        let k = &HANDBRAKE_FOLDED[self.index()];
        Curve { keys: HANDBRAKE_STEER_RAW.iter().zip(k.iter()).map(|(r, b)| (r.0, fb(*b))).collect() }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VehicleVariant {
    pub steer: &'static [(u32, u32)],
    pub handbrake_steer: &'static [(u32, u32)],
    pub lat_friction: [(u32, u32); 2],
    pub sticky_base: f32,
    pub auto_flip_torque: f32,
    pub auto_flip_force: f32,

    pub wheel_up_offset: u32,
    pub robin_air: [(f32, f32); 3],
    pub robin_jump: [(f32, f32); 3],
    pub robin_double_jump: [(f32, f32); 3],
    pub flip_time: u32,
}

pub static VEHICLE_VARIANTS: std::sync::LazyLock<Vec<(&'static str, VehicleVariant)>> = std::sync::LazyLock::new(|| {
    let pairs = |r: &crate::data::Row, at: &mut usize| -> &'static [(u32, u32)] {
        let n = r.w(*at) as usize;
        let v: Vec<(u32, u32)> = (0..n).map(|k| (r.w(*at + 1 + 2 * k), r.w(*at + 2 + 2 * k))).collect();
        *at += 1 + 2 * n;
        Box::leak(v.into_boxed_slice())
    };
    let f3 = |r: &crate::data::Row, at: usize| -> [(f32, f32); 3] { std::array::from_fn(|k| (r.f(at + 2 * k), r.f(at + 2 * k + 1))) };
    crate::data::table_opt("vehicle_variants")
        .unwrap_or(&[])
        .iter()
        .map(|r| {
            let mut at = 0;
            let steer = pairs(r, &mut at);
            let handbrake_steer = pairs(r, &mut at);
            let v = VehicleVariant {
                steer,
                handbrake_steer,
                lat_friction: [(r.w(at), r.w(at + 1)), (r.w(at + 2), r.w(at + 3))],
                sticky_base: r.f(at + 4),
                auto_flip_torque: r.f(at + 5),
                auto_flip_force: r.f(at + 6),
                wheel_up_offset: r.w(at + 7),
                robin_air: f3(r, at + 8),
                robin_jump: f3(r, at + 14),
                robin_double_jump: f3(r, at + 20),
                flip_time: r.w(at + 26),
            };
            (r.s(0), v)
        })
        .collect()
});

pub fn vehicle_variant(preset: &str) -> Option<&'static VehicleVariant> {
    VEHICLE_VARIANTS.iter().find(|(n, _)| *n == preset).map(|(_, v)| v)
}

fn bt_wheel(a: &Axle, left: bool, min_hit_dist2: u32) -> WheelInfo {
    let sign = if left { 0x8000_0000u32 } else { 0 };
    WheelInfo {
        connection: v3(fb(a.connection[0]), fb(a.connection[1] ^ sign), fb(a.connection[2])),
        radius: fb(a.radius),
        rest: fb(a.rest),
        travel: fb(0x3e75c28f),
        max_raise: fb(0x3d4ccccc),
        stiffness: 500.0,
        damping_compression: 25.0,
        damping_relaxation: 40.0,
        min_hit_dist2: fb(min_hit_dist2),
        push_force: 1.0,
        lat_friction: 1.0,
        long_friction: 1.0,
        basis: Mat3::IDENTITY,
        ..WheelInfo::default()
    }
}

fn wheel_sim(steer_factor: f32, v: Option<&VehicleVariant>) -> WheelSim {
    WheelSim {
        steer_factor,
        lat_friction_curve: match v {
            Some(v) => Curve::new(&v.lat_friction.map(|(x, y)| (fb(x), fb(y)))),
            None => Curve::new(&[(0.0, 1.0), (1.0, fb(0x3e4ccccd))]),
        },
        long_friction_curve: Curve::default(),
        handbrake_lat_curve: Curve::new(&[(0.0, fb(0x3dcccccd))]),
        handbrake_long_curve: Curve::new(&[(0.0, 0.5), (1.0, fb(0x3f666666))]),
        wall_friction_curve: Curve::new(&[(0.0, fb(0x3dcccccd)), (fb(0x3f34fdf4), 0.5), (1.0, 1.0)]),
        ..WheelSim::default()
    }
}

pub const PHYSICS_CONFIG: CarPhysicsConfig = CarPhysicsConfig {
    supersonic_speed: 2200.0,
    supersonic_hysteresis: 100.0,
    supersonic_time: 1.0,
    quantize_physics: true,
    jump_leave_ground_time: 0.125,
};

pub fn new_car(p: &Preset, body: BodyId, inv_mass: f32) -> Car {
    let variant = vehicle_variant(p.name);
    let mut wheels = [
        bt_wheel(&p.front, true, p.min_hit_dist2),
        bt_wheel(&p.front, false, p.min_hit_dist2),
        bt_wheel(&p.back, true, p.min_hit_dist2),
        bt_wheel(&p.back, false, p.min_hit_dist2),
    ];
    if let Some(v) = variant {
        for w in wheels.iter_mut() {
            w.up_offset = fb(v.wheel_up_offset);
        }
    }
    crate::bt::vehicle::set_force_scales(&mut wheels, inv_mass);
    let sim = VehicleSim {
        steer_curve: p.steer_curve(),
        drive_torque: 288000.0,
        drive_curve: Curve::new(&DRIVE_CURVE),
        brake_torque: 10500.0,
        handbrake_steer_curve: p.handbrake_steer_curve(),
        handbrake_rise: 5.0,
        handbrake_fall: 2.0,
        stop_threshold: 25.0,
        idle_brake: fb(0x3e19999a),
        opposite_brake: 1.0,
        steering_sensitivity: 1.0,
        tire_friction_scale: 1.0,
        ..VehicleSim::default()
    };
    Car {
        body,
        old_state: RbState::default(),
        state: RbState::default(),
        world_contact: WorldContact::default(),
        limits: ReadLimits { max_linear: 2300.0, max_angular: 5.5, quantize: true },
        rolling_friction: 0.0,
        config: PHYSICS_CONFIG,
        sticky_base: variant.map_or(0.5, |v| v.sticky_base),
        sticky_full: 1.5,
        auto_flip_torque: variant.map_or(80.0, |v| v.auto_flip_torque),
        auto_flip_force: variant.map_or(100.0, |v| v.auto_flip_force),
        driving: false,
        jumped: false,
        double_jumped: false,
        on_ground: false,
        supersonic: false,
        podium: false,
        input: VehicleInputs::default(),
        input_restriction: 0,
        has_ai_controller: false,
        num_wheels_for_grounded: 3,
        time_below_supersonic: 0.0,
        max_time_for_dodge: 1.25,
        max_num_jumps: 2,
        body_scale: 1.0,
        dodge_input_threshold: 0.5,
        gravity_z: -650.0,
        game_active: true,
        sim,
        wheels: [wheel_sim(1.0, variant), wheel_sim(1.0, variant), wheel_sim(0.0, variant), wheel_sim(0.0, variant)],
        vehicle: Vehicle { friction_impulse: [Vec3::ZERO; 4], contact_mask: 0xff7f, ray_group: 0x81ad, chassis: body, keep_contacts: false, wheels },
        air: AirControl {
            c: Component::default(),
            torque: [130, 95, 400],
            damping: [30, 20, 50],
            torque_scalar: 1000.0,
            damping_scalar: 1000.0,
            throttle_force: 12000.0,
            sensitivity: 1.0,
            robin: variant.map(|v| v.robin_air),
        },
        boost: Boost {
            c: Component::default(),
            consumption_rate: fb(0x3eaa7efa),
            max_amount: 1.0,
            start_amount: fb(0x3eaa7efa),
            used: 0.0,
            amount: 0.0,
            modifier: 1.0,
            no_boost: false,
            recharge_ground_only: false,
            force: 178500.0,
            min_boost_time: fb(0x3dcccccd),
            recharge_rate: 0.0,
            recharge_delay: 0.0,
            unlimited_refs: 0,
            restriction: 0,
        },
        jump: Jump { c: Component::default(), min_jump_time: fb(0x3ccbfb16), impulse: 52500.0, force: 262500.0, force_time: fb(0x3e4ccccd), podium_force_time: fb(0x3dae147b), deactivate: false, robin: variant.map(|v| v.robin_jump) },
        double_jump: DoubleJump { c: Component::default(), impulse: 52500.0, impulse_scale: 1.0, stored: Vec3::ZERO, robin: variant.map(|v| v.robin_double_jump) },
        dodge: Dodge {
            c: Component::default(),
            input_threshold: 0.5,
            side_impulse: 90000.0,
            side_impulse_scale: fb(0x3ff33333),
            forward_impulse: 90000.0,
            forward_impulse_scale: 1.0,
            backward_impulse: 96000.0,
            backward_impulse_scale: 2.5,
            side_torque: 260.0,
            forward_torque: 224.0,
            torque_time: fb(0x3f266666),
            min_torque_time: fb(0x3d27ef9e),
            z_damping: fb(0x3eb33333),
            z_damping_delay: fb(0x3e19999a),
            z_damping_up_time: fb(0x3d75c28f),
            impulse_scale: 1.0,
            torque_scale: 1.0,
            torque: Vec3::ZERO,
            direction: Vec3::ZERO,
        },
        flip: FlipCar { c: Component::default(), impulse: 200.0, torque: 1000.0, time: fb(variant.map_or(0x3ecccccd, |v| v.flip_time)), default_time: fb(variant.map_or(0x3ecccccd, |v| v.flip_time)), flip_right: false },
        wants_constraint: true,
    }
}

#[cfg(feature = "dynamics_core")]
pub fn car_rigid_info(transform: crate::math::Transform, shape: crate::bt::ShapeId, inertia: Vec3) -> crate::bt::rigid_body::RigidBodyInfo {
    let mut info = crate::bt::rigid_body::RigidBodyInfo::new(CAR_MASS, transform, shape, inertia);
    info.friction = fb(0x3e99999a);
    info.restitution = fb(0x3dcccccd);
    info.max_lin_speed = 46.0;
    info.max_ang_speed = 5.5;
    info
}

pub const CAR_MASS: f32 = 180.0;

pub fn finish_car_body(b: &mut crate::bt::Body, index: u32) {
    b.flags |= crate::bt::CF_CUSTOM_MATERIAL;
    b.ccd_motion_threshold = fb(0x3ca3d70a);
    b.owner = crate::bt::Owner::Car(index);
}
