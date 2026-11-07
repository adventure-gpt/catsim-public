//! The contact memory in the arena's terms (bt::contacts has it in body ids): the broadphase's overlapping pairs in
//! their order, the contact manifolds in the dispatcher's order with their cached points, and the last step's predictive
//! contacts. Bodies are named as the ball, a car or a part of the arena (its planes and meshes, in the order the arena
//! made them). State access only: an arena that never calls `set_contacts` steps exactly as before.

pub use crate::bt::contacts::ManifoldOwner;
use crate::bt::contacts::{BoxState, ContactMemory, ManifoldState, PairState};
use crate::bt::shapes::Shape;
use crate::bt::{BodyId, CF_CUSTOM_MATERIAL, CF_STATIC, ManifoldPoint};
use crate::game::arena::Arena;
use crate::math::{Vec3, v3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BodyRef {
    Ball,
    Car(usize),
    /// The k-th part of the arena (`Arena::world_parts`).
    World(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartKind {
    Plane,
    Mesh,
}

/// A part of the arena: a plane or a collision mesh, and its broadphase handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldPart {
    pub kind: PartKind,
    pub handle: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    pub body0: BodyRef,
    pub body1: BodyRef,
    pub dispatched: bool,
}

#[derive(Clone, Debug)]
pub struct Manifold {
    pub owner: ManifoldOwner,
    pub body0: BodyRef,
    pub body1: BodyRef,
    pub points: Vec<ManifoldPoint>,
}

/// A moving body's broadphase box as the last bounds update left it (Bullet units, with the contact margin).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyBox {
    pub body: BodyRef,
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Clone, Debug, Default)]
pub struct Contacts {
    pub boxes: Vec<BodyBox>,
    pub pairs: Vec<Pair>,
    pub manifolds: Vec<Manifold>,
}

/// Every field of a contact point, in `contact_point_row`'s order.
pub const CONTACT_POINT_FIELDS: [&str; 54] = [
    "local_a.x", "local_a.y", "local_a.z", "local_b.x", "local_b.y", "local_b.z",
    "world_a.x", "world_a.y", "world_a.z", "world_b.x", "world_b.y", "world_b.z",
    "normal_b.x", "normal_b.y", "normal_b.z", "distance",
    "combined_friction", "combined_rolling", "combined_restitution",
    "part0", "index0", "part1", "index1",
    "applied_impulse", "applied_impulse_lat1", "applied_impulse_lat2", "lateral_initialized", "life_time",
    "lat_dir1.x", "lat_dir1.y", "lat_dir1.z", "lat_dir2.x", "lat_dir2.y", "lat_dir2.z",
    "offset_a.x", "offset_a.y", "offset_a.z", "offset_b.x", "offset_b.y", "offset_b.z",
    "vel_a.x", "vel_a.y", "vel_a.z", "vel_b.x", "vel_b.y", "vel_b.z",
    "contact_motion1", "contact_motion2", "contact_cfm1", "contact_cfm2", "rest_speed", "row_scale1", "row_scale2",
    "sphere_on_static",
];

/// A contact point as numbers (every value exact in an f64), in `CONTACT_POINT_FIELDS`' order.
pub fn contact_point_row(p: &ManifoldPoint) -> Vec<f64> {
    let mut r = Vec::with_capacity(CONTACT_POINT_FIELDS.len());
    let v = |r: &mut Vec<f64>, x: Vec3| r.extend([x.x as f64, x.y as f64, x.z as f64]);
    v(&mut r, p.local_a);
    v(&mut r, p.local_b);
    v(&mut r, p.world_a);
    v(&mut r, p.world_b);
    v(&mut r, p.normal_b);
    r.extend([p.distance, p.combined_friction, p.combined_rolling, p.combined_restitution].map(|x| x as f64));
    r.extend([p.part0, p.index0, p.part1, p.index1].map(|x| x as f64));
    r.extend([p.applied_impulse, p.applied_impulse_lat1, p.applied_impulse_lat2].map(|x| x as f64));
    r.push(p.lateral_initialized as u8 as f64);
    r.push(p.life_time as f64);
    v(&mut r, p.lat_dir1);
    v(&mut r, p.lat_dir2);
    v(&mut r, p.offset_a);
    v(&mut r, p.offset_b);
    v(&mut r, p.vel_a);
    v(&mut r, p.vel_b);
    r.extend([p.contact_motion1, p.contact_motion2, p.contact_cfm1, p.contact_cfm2, p.rest_speed, p.row_scale1, p.row_scale2].map(|x| x as f64));
    r.push(p.sphere_on_static as u8 as f64);
    r
}

pub fn contact_point_from_row(r: &[f64]) -> Option<ManifoldPoint> {
    if r.len() != CONTACT_POINT_FIELDS.len() {
        return None;
    }
    let f = |i: usize| r[i] as f32;
    let v = |i: usize| v3(f(i), f(i + 1), f(i + 2));
    let n = |i: usize| r[i] as i32;
    Some(ManifoldPoint {
        local_a: v(0),
        local_b: v(3),
        world_a: v(6),
        world_b: v(9),
        normal_b: v(12),
        distance: f(15),
        combined_friction: f(16),
        combined_rolling: f(17),
        combined_restitution: f(18),
        part0: n(19),
        index0: n(20),
        part1: n(21),
        index1: n(22),
        applied_impulse: f(23),
        applied_impulse_lat1: f(24),
        applied_impulse_lat2: f(25),
        lateral_initialized: r[26] != 0.0,
        life_time: n(27),
        lat_dir1: v(28),
        lat_dir2: v(31),
        offset_a: v(34),
        offset_b: v(37),
        vel_a: v(40),
        vel_b: v(43),
        contact_motion1: f(46),
        contact_motion2: f(47),
        contact_cfm1: f(48),
        contact_cfm2: f(49),
        rest_speed: f(50),
        row_scale1: f(51),
        row_scale2: f(52),
        sphere_on_static: r[53] != 0.0,
        user_data: 0,
    })
}

fn clamp10(f: f32) -> f32 {
    let lo = if -10.0 <= f { f } else { -10.0 };
    if lo <= 10.0 { lo } else { 10.0 }
}

impl Arena {
    fn static_bodies(&self) -> Vec<BodyId> {
        let cw = &self.world.cw;
        let mut v: Vec<BodyId> = cw.objects.iter().copied().filter(|id| cw.bodies[id.0 as usize].flags & CF_STATIC != 0).collect();
        v.sort();
        v
    }

    /// The arena's planes and meshes, in the order the arena made them (`BodyRef::World(k)` is the k-th).
    pub fn world_parts(&self) -> Vec<WorldPart> {
        let cw = &self.world.cw;
        self.static_bodies()
            .into_iter()
            .map(|id| {
                let b = &cw.bodies[id.0 as usize];
                let kind = if matches!(cw.shapes[b.shape.0 as usize], Shape::Plane(_)) { PartKind::Plane } else { PartKind::Mesh };
                WorldPart { kind, handle: b.proxy as u16 }
            })
            .collect()
    }

    pub fn body_id(&self, r: BodyRef) -> Option<BodyId> {
        match r {
            BodyRef::Ball => self.ball.as_ref().map(|b| b.body),
            BodyRef::Car(i) => self.players.get(i)?.car.as_ref().map(|c| c.body),
            BodyRef::World(k) => self.static_bodies().get(k).copied(),
        }
    }

    pub fn body_ref(&self, id: BodyId) -> Option<BodyRef> {
        if self.ball.as_ref().is_some_and(|b| b.body == id) {
            return Some(BodyRef::Ball);
        }
        if let Some(i) = self.players.iter().position(|p| p.car.as_ref().is_some_and(|c| c.body == id)) {
            return Some(BodyRef::Car(i));
        }
        self.static_bodies().iter().position(|s| *s == id).map(BodyRef::World)
    }

    /// A body's broadphase handle (the number its pairs are ordered by).
    pub fn broadphase_handle(&self, r: BodyRef) -> Option<u16> {
        self.body_id(r).map(|id| self.world.cw.bodies[id.0 as usize].proxy as u16)
    }

    /// The contact memory as it is now.
    pub fn contacts(&self) -> Result<Contacts, String> {
        let m = self.world.contact_memory()?;
        let name = |id: BodyId| self.body_ref(id).ok_or_else(|| format!("body {} is not the ball, a car or the arena", id.0));
        let mut boxes = Vec::with_capacity(m.boxes.len());
        for b in &m.boxes {
            boxes.push(BodyBox { body: name(b.body)?, min: b.min, max: b.max });
        }
        let mut pairs = Vec::with_capacity(m.pairs.len());
        for p in &m.pairs {
            pairs.push(Pair { body0: name(p.body0)?, body1: name(p.body1)?, dispatched: p.dispatched });
        }
        let mut manifolds = Vec::with_capacity(m.manifolds.len());
        for x in m.manifolds {
            manifolds.push(Manifold { owner: x.owner, body0: name(x.body0)?, body1: name(x.body1)?, points: x.points });
        }
        Ok(Contacts { boxes, pairs, manifolds })
    }

    /// Replace the contact memory (bt::contacts::World::set_contact_memory says how). On an error nothing has changed.
    pub fn set_contacts(&mut self, c: &Contacts) -> Result<(), String> {
        let id = |r: BodyRef| self.body_id(r).ok_or_else(|| format!("no body {r:?}"));
        let mut mem = ContactMemory::default();
        for b in &c.boxes {
            mem.boxes.push(BoxState { body: id(b.body)?, min: b.min, max: b.max });
        }
        for p in &c.pairs {
            mem.pairs.push(PairState { body0: id(p.body0)?, body1: id(p.body1)?, dispatched: p.dispatched });
        }
        for m in &c.manifolds {
            mem.manifolds.push(ManifoldState { owner: m.owner, body0: id(m.body0)?, body1: id(m.body1)?, points: m.points.clone() });
        }
        self.world.set_contact_memory(&mem)
    }

    /// A contact point as the narrow phase adds it, from what a recording keeps of one: the point on each body in that
    /// body's frame, the normal on `body1` in world axes and the distance. The rest (world positions, the combined
    /// friction and restitution with the game's own rules, the friction directions) is made the way the game makes it.
    pub fn contact_point(&self, body0: BodyRef, body1: BodyRef, local_a: Vec3, local_b: Vec3, normal_b: Vec3, distance: f32) -> Option<ManifoldPoint> {
        let cw = &self.world.cw;
        let (b0, b1) = (&cw.bodies[self.body_id(body0)?.0 as usize], &cw.bodies[self.body_id(body1)?.0 as usize]);
        let (ra, oa) = (&b0.transform.basis.r, b0.transform.origin);
        let (rb, ob) = (&b1.transform.basis.r, b1.transform.origin);
        let (lx, ly, lz) = (local_a.x, local_a.y, local_a.z);
        let world_a = v3(lx * ra[0].x + ly * ra[0].y + lz * ra[0].z + oa.x, lx * ra[1].x + ly * ra[1].y + lz * ra[1].z + oa.y, lx * ra[2].x + ly * ra[2].y + lz * ra[2].z + oa.z);
        let (lx, ly, lz) = (local_b.x, local_b.y, local_b.z);
        let world_b = v3(ly * rb[0].y + lx * rb[0].x + lz * rb[0].z + ob.x, ly * rb[1].y + lx * rb[1].x + lz * rb[1].z + ob.y, lx * rb[2].x + ly * rb[2].y + lz * rb[2].z + ob.z);
        let (lat_dir1, lat_dir2) = crate::bt::manifold::plane_space(normal_b);
        let mut p = ManifoldPoint {
            local_a,
            local_b,
            world_a,
            world_b,
            normal_b,
            distance,
            combined_friction: clamp10(b1.friction * b0.friction),
            combined_restitution: b1.restitution * b0.restitution,
            combined_rolling: clamp10(b1.rolling_friction * b0.rolling_friction),
            lat_dir1,
            lat_dir2,
            ..ManifoldPoint::default()
        };
        if b0.flags & CF_CUSTOM_MATERIAL != 0 || b1.flags & CF_CUSTOM_MATERIAL != 0 {
            let e = if b0.restitution <= b1.restitution { b1.restitution } else { b0.restitution };
            p.combined_restitution = if 0.0 <= e { if 1.0 <= e { 1.0 } else { e } } else { 0.0 };
            let s1 = &cw.shapes[b1.shape.0 as usize];
            p.sphere_on_static = matches!(cw.shapes[b0.shape.0 as usize], Shape::Sphere(_)) && matches!(s1, Shape::Plane(_) | Shape::Mesh(_));
            let car_ball = matches!((body0, body1), (BodyRef::Car(_), BodyRef::Ball) | (BodyRef::Ball, BodyRef::Car(_)));
            if car_ball {
                p.combined_restitution = crate::game::rules::CAR_BALL_RESTITUTION;
                p.combined_friction = crate::game::rules::CAR_BALL_FRICTION;
            }
            if self.ball_held_2d && (body0 == BodyRef::Ball || body1 == BodyRef::Ball) {
                p.combined_restitution = 0.0;
            }
        }
        Some(p)
    }
}
