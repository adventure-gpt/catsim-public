use crate::bt::rigid_body::RigidBodyInfo;
use crate::bt::shapes::{Shape, SphereShape};
use crate::bt::{Body, BodyId, Owner, ShapeId};
use crate::game::rb_state::{self, RbState, ReadLimits};
use crate::game::rules::{RbCollision, WorldContact};
use crate::math::{Mat3, Transform, Vec3, v3};

pub const SHAPE_RADIUS: f32 = f32::from_bits(0x3fe9_9998);

pub const SCRIPT_RADIUS: f32 = f32::from_bits(0x42be_f970);
pub const MASS_PROP: f32 = 30.0;
pub const INV_MASS: f32 = f32::from_bits(0x3d08_8888);
pub const INV_INERTIA: f32 = f32::from_bits(0x3ccc_f771);
pub const FRICTION: f32 = f32::from_bits(0x3eb3_3333);
pub const RESTITUTION: f32 = f32::from_bits(0x3f19_999a);
pub const LIN_DAMPING: f32 = f32::from_bits(0x3cf5_c28f);
pub const MAX_LINEAR_UU: f32 = 6000.0;
pub const MAX_ANGULAR: f32 = 6.0;

pub const SPAWN_Z_BT: f32 = f32::from_bits(0x4000_9fa4);
pub const CCD_MOTION_THRESHOLD: f32 = f32::from_bits(0x3ca3_d70a);

#[inline]
pub fn read_limits(quantize: bool) -> ReadLimits {
    ReadLimits { max_linear: MAX_LINEAR_UU, max_angular: MAX_ANGULAR, quantize }
}

pub fn shape() -> Shape {
    Shape::Sphere(SphereShape::new(SHAPE_RADIUS))
}

pub fn body(shape: ShapeId, inertia_of_mass: Vec3, loc_bt: Vec3, rotation: Mat3) -> Body {
    let mut info = RigidBodyInfo::new(MASS_PROP, Transform { basis: rotation, origin: loc_bt }, shape, inertia_of_mass);
    info.friction = FRICTION;
    info.restitution = RESTITUTION;
    info.lin_damping = LIN_DAMPING;
    info.ang_damping = 0.0;
    info.max_lin_speed = MAX_LINEAR_UU * 0.02;
    info.max_ang_speed = MAX_ANGULAR;
    let mut b = Body::new_rigid(&info);
    b.inv_mass = INV_MASS;
    b.inv_inertia_local = v3(INV_INERTIA, INV_INERTIA, INV_INERTIA);
    b.inv_mass_factor = v3(INV_MASS, INV_MASS, INV_MASS);
    b.flags |= crate::bt::CF_CUSTOM_MATERIAL;
    b.ccd_motion_threshold = CCD_MOTION_THRESHOLD;
    b.owner = Owner::Ball;
    b
}

#[derive(Clone, Debug)]
pub struct Ball {
    pub body: BodyId,
    pub old_state: RbState,
    pub state: RbState,

    pub birth_state: RbState,
    pub limits: ReadLimits,
    pub contacts: Vec<RbCollision>,
    pub world_contact: WorldContact,

    pub seconds_since_grounded: f32,

    pub last_hit_world_time: f32,

    pub round_active: bool,

    pub actor_loc: Vec3,

    pub age: u32,

    pub first_frame: bool,
}

impl Ball {
    pub fn new(body: BodyId, limits: ReadLimits) -> Ball {
        Ball {
            body,
            old_state: RbState::default(),
            state: RbState::default(),
            birth_state: RbState::default(),
            limits,
            contacts: Vec::new(),
            world_contact: WorldContact::default(),
            seconds_since_grounded: 0.0,
            last_hit_world_time: 0.0,
            round_active: false,
            actor_loc: Vec3::ZERO,
            age: 0,
            first_frame: true,
        }
    }

    pub fn init_state(&mut self, b: &Body) {
        let s = rb_state::read_body(b, &self.limits);
        self.state = s;
        self.old_state = s;
        self.birth_state = s;
    }

    pub fn pre_tick(&mut self, b: &mut Body) {
        self.age += 1;
        let s = rb_state::read_body(b, &self.limits);
        self.old_state = s;
        self.state = s;
        if self.limits.quantize {
            rb_state::write_body(b, &s);
        }
        self.contacts.clear();
    }

    pub fn post_read(&mut self, b: &Body) -> bool {
        let was = self.state.sleeping;
        self.state = rb_state::read_body(b, &self.limits);
        was
    }

    pub fn grounded_clock(&mut self, was_sleeping: bool, dt: f32) {
        if was_sleeping == self.state.sleeping {
            self.seconds_since_grounded = dt + self.seconds_since_grounded;
        } else {
            self.seconds_since_grounded = 0.0;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GoalFace {
    pub normal: Vec3,
    pub vertex: Vec3,
}


#[derive(Clone, Copy, Debug)]
pub struct Goal {
    pub location: Vec3,
    pub faces: [GoalFace; 6],

    pub team_scoring: u8,
}

include!("goal_faces.rs");

impl Goal {

    pub fn fully_inside(&self, ball: Vec3, radius: f32) -> bool {
        let l = v3(ball.x - self.location.x, ball.y - self.location.y, ball.z - self.location.z);
        let r2 = radius + radius;
        for f in &self.faces {
            let d = (l.y - f.vertex.y) * f.normal.y + (l.x - f.vertex.x) * f.normal.x + (l.z - f.vertex.z) * f.normal.z;
            let pen = radius - d;
            if r2 > pen {
                return false;
            }
        }
        true
    }
}

#[inline]
pub fn goals() -> &'static [Goal; 2] {
    &GOALS
}
