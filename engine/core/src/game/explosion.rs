use super::rules::normal;
use crate::game::rb_state;
use crate::math::rot::{Rotator, rotator_to_matrix};
use crate::math::{Vec3, v3};

pub const RB_VELOCITY_CHANGE: f32 = 5000.0;
pub const END_RADIUS: f32 = 3000.0;
pub const START_RADIUS: f32 = 0.0;
pub const FALLOFF: f32 = 0.1;
pub const SPEED: f32 = 10000.0;
pub const MOMENTUM_OFFSET: Vec3 = v3(-500.0, 0.0, -500.0);

pub const FIRST_TEST_TICK: i32 = 4;
pub const MARGIN: f32 = 160.0;

pub fn orientation_for_goal(y: f32) -> Rotator {
    Rotator { pitch: 0, yaw: if y < 0.0 { 16384 } else { -16384 }, roll: 0 }
}

#[derive(Clone, Debug)]
pub struct Explosion {
    pub location: Vec3,
    pub rotation: Rotator,
    pub life: f32,
    pub ticks: i32,
    pub hit: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct ExplosionHit {
    pub car: u32,
    pub impulse: Vec3,
    pub position: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct ExplosionPrimitive {
    pub actor: u32,
    pub actor_location: Vec3,
    pub bounds_origin: Vec3,
    pub bounds_extent: Vec3,
    pub eligible: bool,
    pub processable: bool,
    pub physics_is_rigid_body: bool,
}

pub fn fixed_mesh_bounds(mut local: [f32; 7], offset: [f32; 3], m: [f32; 16], bounds_scale: f32) -> [f32; 7] {
    for axis in 0..3 { local[axis] += offset[axis]; }
    let mut out = [0.0f32; 7];
    for axis in 0..3 {
        out[axis] = ((m[4 + axis] * local[1] + m[axis] * local[0]) + m[8 + axis] * local[2]) + m[12 + axis];
    }
    for sx in [-1.0f32, 1.0] {
        let x = local[3] * sx;
        for sy in [-1.0f32, 1.0] {
            let y = local[4] * sy;
            for axis in 0..3 {
                let xy = y * m[4 + axis] + x * m[axis];
                let low = ((local[5] * -1.0) * m[8 + axis] + xy) + m[12 + axis] * 0.0;
                let high = (local[5] * m[8 + axis] + xy) + m[12 + axis] * 0.0;
                let a = if low <= out[3 + axis] { out[3 + axis] } else { low };
                out[3 + axis] = if high <= a { a } else { high };
            }
        }
    }
    let length2 = |row: usize| (m[row] * m[row] + m[row + 1] * m[row + 1]) + m[row + 2] * m[row + 2];
    let yz = if length2(4) <= length2(8) { length2(8) } else { length2(4) };
    let max2 = if length2(0) <= yz { yz } else { length2(0) };
    out[6] = (max2 as f64).sqrt() as f32 * local[6];
    for value in &mut out[3..] { *value *= bounds_scale; }
    out
}

pub fn car_mesh_bounds(local: [f32; 7], offset: [f32; 3], location: Vec3, rotation: Rotator, bounds_scale: f32) -> [f32; 7] {
    let r = rotator_to_matrix(rotation).r;
    let m = [r[0].x, r[0].y, r[0].z, 0.0, r[1].x, r[1].y, r[1].z, 0.0, r[2].x, r[2].y, r[2].z, 0.0, location.x, location.y, location.z, 1.0];
    fixed_mesh_bounds(local, offset, m, bounds_scale)
}

pub static OCTANE_MESH_BOUNDS: std::sync::LazyLock<[f32; 7]> = std::sync::LazyLock::new(|| crate::data::table("default_mesh_bounds")[0].fnn(0));

pub fn sphere_overlaps_bounds(center: Vec3, radius_squared: f32, origin: Vec3, extent: Vec3) -> bool {
    let mut distance_squared = 0.0f32;
    for (c, o, e) in [(center.x, origin.x, extent.x), (center.y, origin.y, extent.y), (center.z, origin.z, extent.z)] {
        let low = o - e;
        let high = o + e;
        if c < low {
            let d = c - low;
            distance_squared += d * d;
        } else if c > high {
            let d = c - high;
            distance_squared += d * d;
        }
    }
    distance_squared <= radius_squared
}

impl Explosion {
    pub fn new(location: Vec3, rotation: Rotator) -> Explosion {
        Explosion { location, rotation, life: 0.0, ticks: 0, hit: Vec::new() }
    }

    #[inline]
    pub fn scale(grow: f32) -> f32 {
        let w = END_RADIUS - START_RADIUS;
        let t = if w <= 0.0 { 1.0 } else { grow / w };
        crate::math::crt_pow::powf(1.0 - t, FALLOFF)
    }

    pub fn frame_tick_bounds(&mut self, dt: f32, primitives: &[ExplosionPrimitive], out: &mut Vec<ExplosionHit>) {
        self.ticks += 1;
        let grow = self.life * SPEED;
        let radius = grow + START_RADIUS;
        self.life += dt;
        if radius > END_RADIUS {
            return;
        }
        let scale = Self::scale(grow);
        let mut admitted: Vec<&ExplosionPrimitive> = Vec::new();
        for p in primitives {
            if p.eligible && !admitted.iter().any(|old| old.actor == p.actor)
                && sphere_overlaps_bounds(self.location, radius * radius, p.bounds_origin, p.bounds_extent) {
                admitted.push(p);
            }
        }
        for p in admitted.into_iter().rev() {
            if !p.processable || self.hit.contains(&p.actor) {
                continue;
            }
            self.hit.push(p.actor);
            if p.physics_is_rigid_body && scale > 0.0 {
                out.push(self.process_hit(p.actor, p.actor_location, scale));
            }
        }
    }

    pub fn frame_tick(&mut self, dt: f32, cars: &[(u32, Vec3)], out: &mut Vec<ExplosionHit>) {
        self.ticks += 1;
        let grow = self.life * SPEED;
        let radius = grow + START_RADIUS;
        self.life += dt;
        if radius > END_RADIUS || self.ticks < FIRST_TEST_TICK {
            return;
        }
        let k = Self::scale(grow);
        for &(id, loc) in cars {
            if self.hit.contains(&id) {
                continue;
            }
            let d = loc - self.location;
            if d.len2_xy().sqrt() <= radius + MARGIN {
                self.hit.push(id);
                out.push(self.process_hit(id, loc, k));
            }
        }
    }

    pub fn process_hit(&self, car: u32, hit_location: Vec3, damage_scale: f32) -> ExplosionHit {
        let m = rotator_to_matrix(self.rotation);
        let o = MOMENTUM_OFFSET;

        let turned = v3(
            o.x * m.r[0].x + o.y * m.r[1].x + o.z * m.r[2].x,
            o.x * m.r[0].y + o.y * m.r[1].y + o.z * m.r[2].y,
            o.x * m.r[0].z + o.y * m.r[1].z + o.z * m.r[2].z,
        );
        let dir = normal(hit_location - (self.location + turned));
        let a = dir * RB_VELOCITY_CHANGE;
        ExplosionHit { car, impulse: a * damage_scale, position: hit_location }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_closed_faces_and_disjoint_components() {
        assert!(sphere_overlaps_bounds(Vec3::ZERO, 1.0, v3(2.0, 0.0, 0.0), v3(1.0, 1.0, 1.0)));
        assert!(!sphere_overlaps_bounds(Vec3::ZERO, 1.0, v3(2.01, 0.0, 0.0), v3(1.0, 1.0, 1.0)));

        for x in [-3.0, 3.0] {
            assert!(!sphere_overlaps_bounds(Vec3::ZERO, 1.0, v3(x, 0.0, 0.0), v3(1.0, 1.0, 1.0)));
        }
    }

    #[test]
    fn bounds_hits_once_in_reverse_admission_order() {
        let p = |actor| ExplosionPrimitive { actor, actor_location: v3(1.0, 2.0, 3.0), bounds_origin: Vec3::ZERO,
            bounds_extent: v3(2.0, 2.0, 2.0), eligible: true, processable: true, physics_is_rigid_body: true };
        let mut ex = Explosion::new(Vec3::ZERO, Rotator::default());
        let mut out = Vec::new();
        ex.frame_tick_bounds(0.001, &[p(1), p(2), p(1)], &mut out);
        assert_eq!(out.iter().map(|h| h.car).collect::<Vec<_>>(), [2, 1]);
        assert_eq!(out[0].position, p(2).actor_location);
        ex.frame_tick_bounds(0.001, &[p(1), p(2)], &mut out);
        assert_eq!(out.len(), 2);
        let mut nonphysical = p(3);
        nonphysical.physics_is_rigid_body = false;
        ex.frame_tick_bounds(0.001, &[nonphysical], &mut out);
        assert!(ex.hit.contains(&3));
        assert_eq!(out.len(), 2);
    }
}

pub fn apply_hit(b: &mut crate::bt::Body, hit: &ExplosionHit) {
    super::rules::add_impulse(b, hit.impulse, hit.position, true);
    let _ = rb_state::activate;
}
