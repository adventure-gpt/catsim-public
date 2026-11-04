use crate::math::{Mat3, Transform, Vec3};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct BodyId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ShapeId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ManifoldId(pub u32);

pub const CF_STATIC: u32 = 1;
pub const CF_KINEMATIC: u32 = 2;
pub const CF_NO_RESPONSE: u32 = 4;
pub const CF_CUSTOM_MATERIAL: u32 = 8;

pub const ACTIVE: i32 = 1;
pub const ISLAND_SLEEPING: i32 = 2;
pub const WANTS_DEACTIVATION: i32 = 3;
pub const DISABLE_DEACTIVATION: i32 = 4;
pub const DISABLE_SIMULATION: i32 = 5;

#[derive(Clone, Debug)]
pub struct Body {

    pub transform: Transform,
    pub interp_transform: Transform,
    pub interp_lin_vel: Vec3,
    pub interp_ang_vel: Vec3,
    pub aniso_friction: Vec3,
    pub has_aniso_friction: i32,
    pub contact_processing_threshold: f32,
    pub proxy: u32,
    pub shape: ShapeId,
    pub flags: u32,
    pub island_tag: i32,
    pub companion_id: i32,
    pub activation_state: i32,
    pub deactivation_time: f32,
    pub friction: f32,
    pub restitution: f32,
    pub rolling_friction: f32,
    pub internal_type: i32,
    pub hit_fraction: f32,
    pub ccd_swept_radius: f32,
    pub ccd_motion_threshold: f32,
    pub check_collide_with: i32,
    pub update_revision: i32,

    pub inv_inertia_world: Mat3,
    pub lin_vel: Vec3,
    pub ang_vel: Vec3,
    pub inv_mass: f32,
    pub lin_factor: Vec3,
    pub gravity: Vec3,
    pub gravity_accel: Vec3,
    pub inv_inertia_local: Vec3,
    pub total_force: Vec3,
    pub total_torque: Vec3,
    pub lin_damping: f32,
    pub ang_damping: f32,
    pub additional_damping: bool,
    pub additional_damping_factor: f32,
    pub additional_lin_threshold_sqr: f32,
    pub additional_ang_threshold_sqr: f32,
    pub additional_ang_factor: f32,
    pub lin_sleep_threshold: f32,
    pub ang_sleep_threshold: f32,
    pub rb_flags: u32,
    pub debug_body_id: i32,

    pub merged: ManifoldPoint,
    pub merged_dist: f32,
    pub merged_count: i32,
    pub max_lin_speed: f32,
    pub max_ang_speed: f32,
    pub ang_factor: Vec3,
    pub inv_mass_factor: Vec3,

    pub owner: Owner,

    pub sticky_wheels: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Owner {
    #[default]
    World,
    Ball,
    Car(u32),
}

impl Default for Body {

    fn default() -> Body {
        Body {
            transform: Transform::IDENTITY,
            interp_transform: Transform::IDENTITY,
            interp_lin_vel: Vec3::ZERO,
            interp_ang_vel: Vec3::ZERO,
            aniso_friction: crate::math::v3(1.0, 1.0, 1.0),
            has_aniso_friction: 0,
            contact_processing_threshold: f32::from_bits(0x5d5e0b6b),
            proxy: 0,
            shape: ShapeId(u32::MAX),
            flags: CF_STATIC,
            island_tag: -1,
            companion_id: -1,
            activation_state: ACTIVE,
            deactivation_time: 0.0,
            friction: 0.5,
            restitution: 0.0,
            rolling_friction: 0.0,
            internal_type: 1,
            hit_fraction: 1.0,
            ccd_swept_radius: 0.0,
            ccd_motion_threshold: 0.0,
            check_collide_with: 0,
            update_revision: 0,
            inv_inertia_world: Mat3::ZERO,
            lin_vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            inv_mass: 0.0,
            lin_factor: crate::math::v3(1.0, 1.0, 1.0),
            gravity: Vec3::ZERO,
            gravity_accel: Vec3::ZERO,
            inv_inertia_local: Vec3::ZERO,
            total_force: Vec3::ZERO,
            total_torque: Vec3::ZERO,
            lin_damping: 0.0,
            ang_damping: 0.0,
            additional_damping: false,
            additional_damping_factor: 0.0,
            additional_lin_threshold_sqr: 0.0,
            additional_ang_threshold_sqr: 0.0,
            additional_ang_factor: 0.0,
            lin_sleep_threshold: 0.0,
            ang_sleep_threshold: 0.0,
            rb_flags: 0,
            debug_body_id: 0,
            merged: ManifoldPoint::default(),
            merged_dist: 0.0,
            merged_count: 0,
            max_lin_speed: 0.0,
            max_ang_speed: 0.0,
            ang_factor: crate::math::v3(1.0, 1.0, 1.0),
            inv_mass_factor: Vec3::ZERO,
            owner: Owner::World,
            sticky_wheels: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ManifoldPoint {
    pub local_a: Vec3,
    pub local_b: Vec3,
    pub world_b: Vec3,
    pub world_a: Vec3,
    pub normal_b: Vec3,
    pub offset_a: Vec3,
    pub offset_b: Vec3,
    pub distance: f32,
    pub combined_friction: f32,
    pub combined_rolling: f32,
    pub combined_restitution: f32,
    pub part0: i32,
    pub part1: i32,
    pub index0: i32,
    pub index1: i32,
    pub user_data: u64,
    pub lateral_initialized: bool,
    pub applied_impulse: f32,
    pub applied_impulse_lat1: f32,
    pub applied_impulse_lat2: f32,
    pub contact_motion1: f32,
    pub contact_motion2: f32,
    pub contact_cfm1: f32,
    pub contact_cfm2: f32,
    pub life_time: i32,
    pub lat_dir1: Vec3,
    pub lat_dir2: Vec3,
    pub vel_a: Vec3,
    pub vel_b: Vec3,
    pub rest_speed: f32,
    pub row_scale1: f32,
    pub row_scale2: f32,
    pub sphere_on_static: bool,
}

impl Default for ManifoldPoint {

    fn default() -> ManifoldPoint {
        ManifoldPoint {
            local_a: Vec3::ZERO,
            local_b: Vec3::ZERO,
            world_b: Vec3::ZERO,
            world_a: Vec3::ZERO,
            normal_b: Vec3::ZERO,
            offset_a: Vec3::ZERO,
            offset_b: Vec3::ZERO,
            distance: 0.0,
            combined_friction: 0.0,
            combined_rolling: 0.0,
            combined_restitution: 0.0,
            part0: 0,
            part1: 0,
            index0: 0,
            index1: 0,
            user_data: 0,
            lateral_initialized: false,
            applied_impulse: 0.0,
            applied_impulse_lat1: 0.0,
            applied_impulse_lat2: 0.0,
            contact_motion1: 0.0,
            contact_motion2: 0.0,
            contact_cfm1: 0.0,
            contact_cfm2: 0.0,
            life_time: 0,
            lat_dir1: Vec3::ZERO,
            lat_dir2: Vec3::ZERO,
            vel_a: Vec3::ZERO,
            vel_b: Vec3::ZERO,
            rest_speed: 0.0,
            row_scale1: 1.0,
            row_scale2: 1.0,
            sphere_on_static: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Manifold {
    pub points: [ManifoldPoint; 4],
    pub body0: BodyId,
    pub body1: BodyId,
    pub cached: i32,
    pub breaking_threshold: f32,
    pub processing_threshold: f32,
    pub companion_a: i32,
    pub companion_b: i32,
    pub index1a: i32,
}
