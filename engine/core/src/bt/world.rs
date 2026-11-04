use super::types::*;
use crate::math::Transform;

pub const DEACTIVATION_TIME: f32 = 2.0;

pub fn apply_gravity(bodies: &mut [Body], rigid: &[BodyId]) {
    for id in rigid {
        let b = &mut bodies[id.0 as usize];
        if b.activation_state != ISLAND_SLEEPING && b.activation_state != DISABLE_SIMULATION {
            b.apply_gravity();
        }
    }
}

pub fn predict_unconstrained_motion(bodies: &mut [Body], rigid: &[BodyId], dt: f32) {
    for id in rigid {
        let b = &mut bodies[id.0 as usize];
        if b.flags & (CF_STATIC | CF_KINEMATIC) == 0 {
            b.apply_damping(dt);
            b.interp_transform = b.predict_integrated_transform(dt);
        }
    }
}

pub fn integrate_transforms(bodies: &mut [Body], rigid: &[BodyId], dt: f32) {
    for id in rigid {
        let b = &mut bodies[id.0 as usize];
        b.hit_fraction = 1.0;
        if b.activation_state != ISLAND_SLEEPING && b.activation_state != DISABLE_SIMULATION && b.flags & (CF_STATIC | CF_KINEMATIC) == 0 {
            let t = b.predict_integrated_transform(dt);
            b.proceed_to_transform(&t);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct IntegrateMemo {
    pub(crate) key: [u32; 19],
    pub(crate) out: Transform,
    pub(crate) valid: bool,
}

#[inline(always)]
pub(crate) fn integrate_key(b: &Body, dt: f32) -> [u32; 19] {
    integrate_key_of(&b.transform, &b.lin_vel, &b.ang_vel, dt)
}

#[inline(always)]
pub(crate) fn integrate_key_of(t: &Transform, v: &crate::math::Vec3, w: &crate::math::Vec3, dt: f32) -> [u32; 19] {
    let (o, r) = (&t.origin, &t.basis.r);
    [
        o.x.to_bits(), o.y.to_bits(), o.z.to_bits(),
        r[0].x.to_bits(), r[0].y.to_bits(), r[0].z.to_bits(),
        r[1].x.to_bits(), r[1].y.to_bits(), r[1].z.to_bits(),
        r[2].x.to_bits(), r[2].y.to_bits(), r[2].z.to_bits(),
        v.x.to_bits(), v.y.to_bits(), v.z.to_bits(),
        w.x.to_bits(), w.y.to_bits(), w.z.to_bits(),
        dt.to_bits(),
    ]
}

#[inline]
pub fn predicted_transform_memo(b: &Body, i: usize, dt: f32, memo: &mut Vec<IntegrateMemo>) -> Transform {
    let key = integrate_key(b, dt);
    if let Some(m) = memo.get(i) {
        if m.valid && m.key == key {
            return m.out;
        }
    }
    let t = b.predict_integrated_transform(dt);
    if memo.len() <= i {
        memo.resize(i + 1, IntegrateMemo::default());
    }
    memo[i] = IntegrateMemo { key, out: t, valid: true };
    t
}

pub fn predict_unconstrained_motion_memo(bodies: &mut [Body], rigid: &[BodyId], dt: f32, memo: &mut Vec<IntegrateMemo>) {
    for id in rigid {
        let i = id.0 as usize;
        let b = &mut bodies[i];
        if b.flags & (CF_STATIC | CF_KINEMATIC) == 0 {
            b.apply_damping(dt);
            b.interp_transform = predicted_transform_memo(b, i, dt, memo);
        }
    }
}

pub fn integrate_transforms_memo(bodies: &mut [Body], rigid: &[BodyId], dt: f32, memo: &mut Vec<IntegrateMemo>) {
    for id in rigid {
        let i = id.0 as usize;
        let b = &mut bodies[i];
        b.hit_fraction = 1.0;
        if b.activation_state != ISLAND_SLEEPING && b.activation_state != DISABLE_SIMULATION && b.flags & (CF_STATIC | CF_KINEMATIC) == 0 {
            let t = predicted_transform_memo(b, i, dt, memo);
            b.proceed_to_transform(&t);
        }
    }
}

pub fn update_activation_state(bodies: &mut [Body], rigid: &[BodyId], dt: f32, deactivation_time: f32, disable_deactivation: bool) {
    for id in rigid {
        let b = &mut bodies[id.0 as usize];
        if b.activation_state != ISLAND_SLEEPING && b.activation_state != DISABLE_DEACTIVATION {
            let (v, w) = (b.lin_vel, b.ang_vel);
            let (ls, as_) = (b.lin_sleep_threshold, b.ang_sleep_threshold);
            if ls * ls <= v.x * v.x + v.y * v.y + v.z * v.z || as_ * as_ <= w.x * w.x + w.y * w.y + w.z * w.z {
                b.deactivation_time = 0.0;
                b.set_activation_state(0);
            } else {
                b.deactivation_time = dt + b.deactivation_time;
            }
        }
        let s = b.activation_state;
        if s != DISABLE_DEACTIVATION {
            if disable_deactivation || deactivation_time == 0.0 || (!(s == ISLAND_SLEEPING || s == WANTS_DEACTIVATION) && b.deactivation_time <= deactivation_time) {
                b.set_activation_state(ACTIVE);
            } else if b.flags & (CF_STATIC | CF_KINEMATIC) == 0 {
                if s == ACTIVE {
                    b.set_activation_state(WANTS_DEACTIVATION);
                }
                if b.activation_state == ISLAND_SLEEPING {
                    b.update_revision += 2;
                    b.ang_vel = crate::math::Vec3::ZERO;
                    b.lin_vel = crate::math::Vec3::ZERO;
                }
            } else {
                b.set_activation_state(ISLAND_SLEEPING);
            }
        }
    }
}

pub fn clear_forces(bodies: &mut [Body], rigid: &[BodyId]) {
    for id in rigid {
        let b = &mut bodies[id.0 as usize];
        b.total_force = crate::math::Vec3::ZERO;
        b.total_torque = crate::math::Vec3::ZERO;
    }
}

#[cfg(feature = "dynamics")]
pub use with_collision::*;

#[cfg(feature = "dynamics")]
mod with_collision {
    use super::*;
    use crate::bt::algo_convex::transform_inverse;
    use crate::bt::collision_world::{CollisionWorld, LocalRayResult, RayCallback};
    use crate::bt::dispatch::ActorHook;
    use crate::bt::islands::IslandManager;
    use crate::bt::rigid_body::RigidBodyInfo;
    use crate::bt::shapes::Shape;
    use crate::bt::solver::{Solver, SolverInfo};
    use crate::bt::sweep::Handle;
    use crate::math::{Transform, Vec3, v3};

    struct PredictiveRay<'a> {
        bodies: &'a [Body],
        closest: f32,
        body: Option<BodyId>,
        group: u16,
        mask: u16,
        from: Vec3,
        to: Vec3,
        normal: Vec3,
        point: Vec3,
    }

    impl RayCallback for PredictiveRay<'_> {
        fn closest_fraction(&self) -> f32 {
            self.closest
        }
        fn flags(&self) -> u32 {
            0
        }

        fn add_single_result(&mut self, hit: &LocalRayResult, normal_in_world: bool) -> f32 {
            self.closest = hit.fraction;
            self.body = Some(hit.body);
            self.normal = if normal_in_world {
                hit.normal
            } else {
                let m = &self.bodies[hit.body.0 as usize].transform.basis.r;
                let n = hit.normal;
                v3(n.y * m[0].y + n.x * m[0].x + n.z * m[0].z, n.y * m[1].y + n.x * m[1].x + n.z * m[1].z, n.y * m[2].y + n.x * m[2].x + n.z * m[2].z)
            };
            let t = hit.fraction;
            let s = 1.0 - t;
            self.point = v3(s * self.from.x + t * self.to.x, s * self.from.y + t * self.to.y, s * self.from.z + t * self.to.z);
            hit.fraction
        }

        fn needs_collision(&self, h: &Handle, b: &Body) -> bool {
            b.flags & (CF_STATIC | CF_KINEMATIC) != 0 && h.group & self.mask != 0 && self.group & h.mask != 0
        }
    }

    #[derive(Clone)]
    pub struct World {
        pub cw: CollisionWorld,

        pub rigid_bodies: Vec<BodyId>,
        pub gravity: Vec3,
        pub info: SolverInfo,
        pub islands: IslandManager,
        pub solver: Solver,

        pub predictive_manifolds: Vec<ManifoldId>,

        pub use_continuous: bool,

        pub update_vehicles_before_physics: bool,
        pub deactivation_time: f32,
        pub disable_deactivation: bool,

        batch_bodies: Vec<BodyId>,
        batch_manifolds: Vec<ManifoldId>,

        integrate_memo: Vec<IntegrateMemo>,
    }

    impl Default for World {
        fn default() -> Self {
            Self::new()
        }
    }

    impl World {

        pub fn new() -> World {
            World {
                cw: CollisionWorld::new(),
                rigid_bodies: Vec::new(),

                gravity: Vec3::ZERO,
                info: SolverInfo::game(),
                islands: IslandManager::new(),
                solver: Solver::new(),
                predictive_manifolds: Vec::new(),
                use_continuous: false,
                update_vehicles_before_physics: true,
                deactivation_time: DEACTIVATION_TIME,
                disable_deactivation: false,
                batch_bodies: Vec::new(),
                batch_manifolds: Vec::new(),
                integrate_memo: Vec::new(),
            }
        }

        pub fn set_gravity(&mut self, g: Vec3) {
            self.gravity = g;
            for id in &self.rigid_bodies {
                self.cw.bodies[id.0 as usize].set_gravity(g);
            }
        }

        pub fn set_use_continuous(&mut self, on: bool) {
            if on != self.use_continuous {
                self.use_continuous = on;
                if !on {
                    self.release_predictive();
                }
            }
        }

        fn release_predictive(&mut self) {
            for k in 0..self.predictive_manifolds.len() {
                let m = self.predictive_manifolds[k];
                self.cw.dispatcher.release_manifold(m);
            }
            self.predictive_manifolds.clear();
        }

        pub fn new_body(&mut self, info: &RigidBodyInfo) -> BodyId {
            self.cw.insert_body(Body::new_rigid(info))
        }

        pub fn add_rigid_body(&mut self, id: BodyId, group: u16, mask: u16) {
            let g = self.gravity;
            let b = &mut self.cw.bodies[id.0 as usize];
            b.set_gravity(g);
            if b.flags & CF_STATIC == 0 {
                self.rigid_bodies.push(id);
            } else {
                b.set_activation_state(ISLAND_SLEEPING);
            }
            self.cw.add_object(id, group, mask);
        }

        pub fn add_rigid_body_default(&mut self, id: BodyId) {
            let dynamic = self.cw.bodies[id.0 as usize].flags & (CF_STATIC | CF_KINEMATIC) == 0;
            if dynamic {
                self.add_rigid_body(id, 1, 0xffff);
            } else {
                self.add_rigid_body(id, 2, 0xfffd);
            }
        }

        pub fn remove_rigid_body(&mut self, id: BodyId) {
            if let Some(at) = self.rigid_bodies.iter().position(|b| *b == id) {
                self.rigid_bodies.swap_remove(at);
            }
            self.cw.remove_object(id);
        }

        fn predictive_ray(&self, from: Vec3, to: Vec3, group: u16, mask: u16, verified:u64) -> (f32, Option<BodyId>, Vec3) {
            let mut cb = PredictiveRay { bodies: &self.cw.bodies, closest: 1.0, body: None, group, mask, from, to, normal: Vec3::ZERO, point: Vec3::ZERO };
            self.cw.ray_test_preverified(from, to, &mut cb, verified);
            (cb.closest, cb.body, cb.normal)
        }

        pub fn create_predictive_contacts(&mut self, dt: f32) {
            self.release_predictive();
            let mut verified=None;
            let back = f32::from_bits(0x3d23d70a);
            for k in 0..self.rigid_bodies.len() {
                let id = self.rigid_bodies[k];
                let b = &mut self.cw.bodies[id.0 as usize];
                b.hit_fraction = 1.0;
                if b.activation_state == ISLAND_SLEEPING || b.activation_state == DISABLE_SIMULATION || b.flags & (CF_STATIC | CF_KINEMATIC) != 0 || b.ccd_motion_threshold == 0.0 {
                    continue;
                }
                let pred = predicted_transform_memo(b, id.0 as usize, dt, &mut self.integrate_memo);
                let (mut cur, mut end) = (b.transform.origin, pred.origin);
                if let Shape::Compound(c) = &self.cw.shapes[b.shape.0 as usize] {
                    let l = c.children[0].transform.origin;
                    let mul = |t: &Transform| {
                        let r = &t.basis.r;
                        v3(
                            l.y * r[0].y + l.x * r[0].x + l.z * r[0].z + t.origin.x,
                            l.y * r[1].y + l.x * r[1].x + l.z * r[1].z + t.origin.y,
                            l.y * r[2].y + l.x * r[2].x + l.z * r[2].z + t.origin.z,
                        )
                    };
                    cur = mul(&b.transform);
                    end = mul(&pred);
                }
                let (mx, my, mz) = (end.x - cur.x, end.y - cur.y, end.z - cur.z);
                let m2 = mx * mx + my * my + mz * mz;
                let th = b.ccd_motion_threshold;
                if !(m2 > th * th) {
                    continue;
                }
                let k_ = 1.0 / m2.sqrt();
                let from = v3(cur.x - k_ * mx * back, cur.y - k_ * my * back, cur.z - k_ * mz * back);
                let (rx, ry, rz) = (end.x - from.x, end.y - from.y, end.z - from.z);
                let h = &self.cw.broadphase.handles[b.proxy as usize];
                let (group, mask) = (h.group, h.mask);
                let _ray_span = crate::prof::span(26);
                if matches!(self.cw.shapes[b.shape.0 as usize], Shape::Compound(_)) { crate::prof::count(39); }
                let (f, hit, n) = if let Some(t) = self.cw.statics_ready() {

                    if t.clear.0.as_ref().is_some_and(|g| g.segment_clear(from, end)) {
                        crate::prof::count(30);
                        continue;
                    }
                    let mut cb = PredictiveRay { bodies: &self.cw.bodies, closest: 1.0, body: None, group, mask, from, to: end, normal: Vec3::ZERO, point: Vec3::ZERO };
                    self.cw.ray_test_statics(from, end, &mut cb, group, mask);
                    (cb.closest, cb.body, cb.normal)
                } else {
                    let valid=*verified.get_or_insert_with(||self.cw.verified_ray_filters());
                    self.predictive_ray(from, end, group, mask,valid)
                };
                if !(1.0 > f) {
                    continue;
                }
                let Some(other) = hit else { continue };
                crate::prof::count(27);
                let (hx, hy, hz) = (rx * f, ry * f, rz * f);
                let d = (hx * hx + hy * hy + hz * hz).sqrt();
                let t = if d > back { d - back } else { 0.0 };
                let i_ = 1.0 / d;
                let (sx, sy, sz) = (i_ * hx * t, i_ * hy * t, i_ * hz * t);
                let distance = (ry - sy) * n.y + (rx - sx) * n.x + (rz - sz) * n.z;
                let CollisionWorld { bodies, shapes, dispatcher, .. } = &mut self.cw;
                let m = dispatcher.get_new_manifold(bodies, shapes, id, other);
                self.predictive_manifolds.push(m);
                let p = v3(sx + from.x, sy + from.y, sz + from.z);
                let (o0, o1) = (&bodies[id.0 as usize], &bodies[other.0 as usize]);
                let inv = transform_inverse(&o1.transform);
                let r = &inv.basis.r;
                let local_b = v3(
                    p.y * r[0].y + p.x * r[0].x + p.z * r[0].z + inv.origin.x,
                    p.y * r[1].y + p.x * r[1].x + p.z * r[1].z + inv.origin.y,
                    p.y * r[2].y + p.x * r[2].x + p.z * r[2].z + inv.origin.z,
                );
                let fr = o0.friction * o1.friction;
                let fr = if -10.0 <= fr { fr } else { -10.0 };
                let fr = if fr <= 10.0 { fr } else { 10.0 };
                let origin0 = o0.transform.origin;
                let manifold = &mut dispatcher.slab[m.0 as usize];
                let pt = ManifoldPoint { local_b, normal_b: n, distance, ..ManifoldPoint::default() };
                let idx = manifold.add_point(&pt) as usize;
                let q = &mut manifold.points[idx];
                q.combined_restitution = 0.0;
                q.combined_friction = fr;
                q.world_a = origin0;
                q.world_b = p;
            }
        }

        pub fn calculate_simulation_islands(&mut self) {
            let CollisionWorld { bodies, objects, broadphase, dispatcher, statics, static_epoch, .. } = &mut self.cw;
            let (pairs, handles) = (&broadphase.pairs.pairs, &broadphase.handles);

            let t = (*static_epoch == statics.epoch && statics.ok).then_some(&*statics);
            let start = t.filter(|t| t.prefix && t.islands_settled).map_or(0, |t| t.order.len());
            let settle = t.is_some_and(|t| t.prefix && !t.islands_settled);
            self.islands.update_activation_state_from(bodies, objects, start, pairs[t.map_or(0, |t| t.ss_prefix)..].iter().filter(|p| t.is_none_or(|t| !t.is_static_handle(p.proxy0) && !t.is_static_handle(p.proxy1))).map(|p| (BodyId(handles[p.proxy0 as usize].client), BodyId(handles[p.proxy1 as usize].client))));
            for mid in &self.predictive_manifolds {
                let m = &dispatcher.slab[mid.0 as usize];
                let (b0, b1) = (&bodies[m.body0.0 as usize], &bodies[m.body1.0 as usize]);
                if b0.flags & (CF_STATIC | CF_KINEMATIC) == 0 && b1.flags & (CF_STATIC | CF_KINEMATIC) == 0 {
                    let (t0, t1) = (b0.island_tag, b1.island_tag);
                    self.islands.unite(t0, t1);
                }
            }
            self.islands.store_island_activation_state_from(bodies, objects, start);
            if settle {
                statics.islands_settled = true;
            }
        }

        pub fn solve_constraints(&mut self) {
            let info = self.info;
            let solver = &mut self.solver;
            let batch_bodies = &mut self.batch_bodies;
            let batch_manifolds = &mut self.batch_manifolds;
            batch_bodies.clear();
            batch_manifolds.clear();
            let CollisionWorld { bodies, objects, dispatcher, .. } = &mut self.cw;
            let (slab, array) = (&mut dispatcher.slab, &dispatcher.array);
            self.islands.build_and_process_islands(bodies, objects, slab, array, |bodies, manifolds, ib, im, _id| {
                batch_bodies.extend_from_slice(ib);
                batch_manifolds.extend_from_slice(im);
                if info.min_batch < batch_manifolds.len() as i32 {
                    solver.solve_group(bodies, batch_bodies, manifolds, batch_manifolds, &info);
                    batch_bodies.clear();
                    batch_manifolds.clear();
                }
            });
            solver.solve_group(bodies, batch_bodies, slab, batch_manifolds, &info);
            batch_bodies.clear();
            batch_manifolds.clear();
        }

        pub fn step_simulation_internal(&mut self, dt: f32, actions: &mut dyn FnMut(&mut World, f32)) {
            self.step_hooked(dt, actions, None);
        }

        pub fn step_hooked(&mut self, dt: f32, actions: &mut dyn FnMut(&mut World, f32), hook: Option<&mut ActorHook<'_>>) {
            self.stage_gravity();
            self.stage_predict(dt);
            if self.update_vehicles_before_physics {
                { let _span=crate::prof::span(2); actions(self, dt); }
            }
            self.stage_ccd(dt);
            self.stage_collide(hook);
            self.stage_solve(dt);
            self.stage_integrate(dt);
            if !self.update_vehicles_before_physics {
                { let _span=crate::prof::span(2); actions(self, dt); }
            }
            self.stage_tail(dt);
        }

        #[inline]
        pub(crate) fn stage_gravity(&mut self) {
            self.cw.ensure_statics();
            let _span=crate::prof::span(0); apply_gravity(&mut self.cw.bodies, &self.rigid_bodies);
        }
        #[inline]
        pub(crate) fn stage_predict(&mut self, dt: f32) {
            let _span=crate::prof::span(1); predict_unconstrained_motion_memo(&mut self.cw.bodies, &self.rigid_bodies, dt, &mut self.integrate_memo);
        }
        #[inline]
        pub(crate) fn stage_ccd(&mut self, dt: f32) {
            if self.use_continuous {
                let _span=crate::prof::span(3); self.create_predictive_contacts(dt);
            }
        }
        #[inline]
        pub(crate) fn stage_collide(&mut self, hook: Option<&mut ActorHook<'_>>) {
            let _span=crate::prof::span(4); self.cw.perform_discrete_collision_detection_hooked(hook);
        }
        #[inline]
        pub(crate) fn stage_solve(&mut self, dt: f32) {
            { let _span=crate::prof::span(5); self.calculate_simulation_islands(); }
            self.info.time_step = dt;
            { let _span=crate::prof::span(6); self.solve_constraints(); }
        }
        #[inline]
        pub(crate) fn stage_integrate(&mut self, dt: f32) {
            let _span=crate::prof::span(7); integrate_transforms_memo(&mut self.cw.bodies, &self.rigid_bodies, dt, &mut self.integrate_memo);
        }
        #[inline]
        pub(crate) fn stage_tail(&mut self, dt: f32) {
            { let _span=crate::prof::span(8); update_activation_state(&mut self.cw.bodies, &self.rigid_bodies, dt, self.deactivation_time, self.disable_deactivation); }
            clear_forces(&mut self.cw.bodies, &self.rigid_bodies);
        }

        #[inline]
        pub(crate) fn memo_mut(&mut self) -> (&mut [Body], &mut Vec<IntegrateMemo>) {
            (&mut self.cw.bodies, &mut self.integrate_memo)
        }
    }
}
