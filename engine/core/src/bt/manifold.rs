use crate::bt::gjk::ContactResult;
use crate::bt::internal_edge::{TriangleInfoMap, adjust_internal_edge};
use crate::bt::shapes::{SHAPE_PLANE, SHAPE_SPHERE, SHAPE_TRIANGLE, Shape};
use crate::bt::types::{Body, BodyId, CF_CUSTOM_MATERIAL, Manifold, ManifoldId, ManifoldPoint};
use crate::math::{Transform, Vec3, v3};

const SQRT1_2: f32 = f32::from_bits(0x3f3504f3);
const EPS_PLANE: f32 = f32::from_bits(0x25800000);

impl Manifold {

    pub fn new(body0: BodyId, body1: BodyId, breaking_threshold: f32, processing_threshold: f32) -> Manifold {
        Manifold { points: [ManifoldPoint::default(); 4], body0, body1, cached: 0, breaking_threshold, processing_threshold, companion_a: -1, companion_b: -1, index1a: 0 }
    }

    pub fn cache_entry(&self, pt: &ManifoldPoint) -> i32 {
        let mut shortest = self.breaking_threshold * self.breaking_threshold;
        let mut near = -1;
        let p = pt.local_a;
        for i in 0..self.cached as usize {
            let c = self.points[i].local_a;
            let (dy, dx, dz) = (c.y - p.y, c.x - p.x, c.z - p.z);
            let d = dy * dy + dx * dx + dz * dz;
            if !(shortest <= d) {
                near = i as i32;
                shortest = d;
            }
        }
        near
    }

    pub fn sort_cached_points(&self, pt: &ManifoldPoint) -> i32 {
        let pts = &self.points;
        let mut max_pen = pt.distance;
        let mut idx: i32 = -1;
        if pts[0].distance < max_pen {
            idx = 0;
            max_pen = pts[0].distance;
        }
        if pts[1].distance < max_pen {
            idx = 1;
            max_pen = pts[1].distance;
        }
        if pts[2].distance < max_pen {
            idx = 2;
            max_pen = pts[2].distance;
        }
        if pts[3].distance < max_pen {
            idx = 3;
        }
        let (mut r0, mut r1, mut r2, mut r3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let a = pt.local_a;
        let (l0, l1, l2, l3) = (pts[0].local_a, pts[1].local_a, pts[2].local_a, pts[3].local_a);
        if idx != 0 {
            let ax = a.x - l1.x;
            let ay = a.y - l1.y;
            let az = a.z - l1.z;
            let by = l3.y - l2.y;
            let bx = l3.x - l2.x;
            let bz = l3.z - l2.z;
            let cx = bz * ay - by * az;
            let cy = bx * az - bz * ax;
            let cz = by * ax - bx * ay;
            r0 = cy * cy + cx * cx + cz * cz;
        }
        if idx != 1 {
            let az = a.z - l0.z;
            let bx = l3.x - l2.x;
            let by = l3.y - l2.y;
            let ax = a.x - l0.x;
            let ay = a.y - l0.y;
            let bz = l3.z - l2.z;
            let cx = ay * bz - az * by;
            let cz = ax * by - ay * bx;
            let cy = az * bx - ax * bz;
            r1 = cy * cy + cx * cx + cz * cz;
        }
        if idx != 2 {
            let by = l3.y - l1.y;
            let bx = l3.x - l1.x;
            let bz = l3.z - l1.z;
            let ax = a.x - l0.x;
            let ay = a.y - l0.y;
            let az = a.z - l0.z;
            let cz = by * ax - bx * ay;
            let cy = bx * az - bz * ax;
            let cx = bz * ay - by * az;
            r2 = cy * cy + cx * cx + cz * cz;
        }
        if idx != 3 {
            let bz = l2.z - l1.z;
            let bx = l2.x - l1.x;
            let by = l2.y - l1.y;
            let ax = a.x - l0.x;
            let az = a.z - l0.z;
            let ay = a.y - l0.y;
            let cy = bx * az - bz * ax;
            let cz = by * ax - bx * ay;
            let cx = bz * ay - by * az;
            r3 = cx * cx + cy * cy + cz * cz;
        }

        let mut biggest: i32 = -1;
        let mut m = f32::from_bits(0xdd5e0b6b);
        for (i, r) in [r0, r1, r2, r3].into_iter().enumerate() {
            if m < r.abs() {
                biggest = i as i32;
                m = r.abs();
            }
        }
        biggest
    }

    pub fn add_point(&mut self, pt: &ManifoldPoint) -> i32 {
        let idx = if self.cached == 4 {
            self.sort_cached_points(pt)
        } else {
            let i = self.cached;
            self.cached += 1;
            i
        };
        let i = idx.max(0);
        self.points[i as usize] = *pt;
        i
    }

    pub fn remove_point(&mut self, i: usize) {
        let last = self.cached as usize - 1;
        if i != last {
            self.points[i] = self.points[last];
            let l = &mut self.points[last];
            l.user_data = 0;
            l.applied_impulse = 0.0;
            l.applied_impulse_lat1 = 0.0;
            l.lateral_initialized = false;
            l.applied_impulse_lat2 = 0.0;
            l.life_time = 0;
        }
        self.cached -= 1;
    }

    pub fn clear(&mut self) {
        self.cached = 0;
    }

    pub fn refresh(&mut self, tr_a: &Transform, tr_b: &Transform) {
        let (ra, oa) = (&tr_a.basis.r, tr_a.origin);
        let (rb, ob) = (&tr_b.basis.r, tr_b.origin);
        let mut i = self.cached - 1;
        while i >= 0 {
            let p = &mut self.points[i as usize];
            let (lx, ly, lz) = (p.local_a.x, p.local_a.y, p.local_a.z);
            p.world_a = v3(lx * ra[0].x + ly * ra[0].y + lz * ra[0].z + oa.x, lx * ra[1].x + ly * ra[1].y + lz * ra[1].z + oa.y, lx * ra[2].x + ly * ra[2].y + lz * ra[2].z + oa.z);
            let (lx, ly, lz) = (p.local_b.x, p.local_b.y, p.local_b.z);
            p.world_b = v3(ly * rb[0].y + lx * rb[0].x + lz * rb[0].z + ob.x, ly * rb[1].y + lx * rb[1].x + lz * rb[1].z + ob.y, lx * rb[2].x + ly * rb[2].y + lz * rb[2].z + ob.z);
            p.life_time += 1;
            p.distance = (p.world_a.y - p.world_b.y) * p.normal_b.y + (p.world_a.x - p.world_b.x) * p.normal_b.x + (p.world_a.z - p.world_b.z) * p.normal_b.z;
            i -= 1;
        }
        let mut i = self.cached - 1;
        while i >= 0 {
            let thr = self.breaking_threshold;
            let p = &self.points[i as usize];
            let d = p.distance;
            if d <= thr {
                let dx = p.world_b.x - (p.world_a.x - d * p.normal_b.x);
                let dy = p.world_b.y - (p.world_a.y - d * p.normal_b.y);
                let dz = p.world_b.z - (p.world_a.z - d * p.normal_b.z);
                if thr * thr < dy * dy + dx * dx + dz * dz {
                    self.remove_point(i as usize);
                }
            } else {
                self.remove_point(i as usize);
            }
            i -= 1;
        }
    }
}

#[derive(Clone, Copy)]
pub struct Wrap<'a> {
    pub shape: &'a Shape,
    pub body: &'a Body,
    pub id: BodyId,
    pub transform: &'a Transform,
    pub part_id: i32,
    pub index: i32,
    pub info_map: Option<&'a TriangleInfoMap>,
}

pub struct ManifoldResult<'a> {
    pub manifolds: &'a mut [Manifold],
    pub current: usize,
    pub wrap0: Wrap<'a>,
    pub wrap1: Wrap<'a>,
    pub part0: i32,
    pub part1: i32,
    pub index0: i32,
    pub index1: i32,

    pub actor_hook: Option<&'a mut dyn FnMut(&mut ManifoldPoint, &Wrap, &Wrap)>,
}

impl<'a> ManifoldResult<'a> {

    pub fn new(manifolds: &'a mut [Manifold], current: ManifoldId, wrap0: Wrap<'a>, wrap1: Wrap<'a>) -> ManifoldResult<'a> {
        ManifoldResult { manifolds, current: current.0 as usize, wrap0, wrap1, part0: 0, part1: 0, index0: 0, index1: 0, actor_hook: None }
    }

    #[inline]
    pub fn manifold(&mut self) -> &mut Manifold {
        &mut self.manifolds[self.current]
    }

    pub fn set_shape_identifiers_a(&mut self, part: i32, index: i32) {
        self.part0 = part;
        self.index0 = index;
    }

    pub fn set_shape_identifiers_b(&mut self, part: i32, index: i32) {
        self.part1 = part;
        self.index1 = index;
    }

    pub fn refresh(&mut self) {
        let (tr0, tr1) = (&self.wrap0.body.transform, &self.wrap1.body.transform);
        let id0 = self.wrap0.id;
        let m = &mut self.manifolds[self.current];
        if m.cached != 0 {
            if m.body0 == id0 {
                m.refresh(tr0, tr1);
            } else {
                m.refresh(tr1, tr0);
            }
        }
    }
}

fn plane_space(n: Vec3) -> (Vec3, Vec3) {
    let mut k = 0.0f32;
    if !(SQRT1_2 < n.z.abs()) {
        let a = n.x * n.x + n.y * n.y;
        if EPS_PLANE < a {
            k = 1.0 / a.sqrt();
        }
        let d1x = -n.y * k;
        let d1y = n.x * k;
        (v3(d1x, d1y, 0.0), v3(-n.z * d1y, d1x * n.z, a * k))
    } else {
        let a = n.y * n.y + n.z * n.z;
        if EPS_PLANE < a {
            k = 1.0 / a.sqrt();
        }
        let d1z = k * n.y;
        (v3(0.0, -n.z * k, d1z), v3(k * a, -n.x * d1z, -n.z * k * n.x))
    }
}

pub fn contact_added(cp: &mut ManifoldPoint, w0: &Wrap, w1: &Wrap) {
    let (r0, r1) = (w0.body.restitution, w1.body.restitution);

    let f = w1.body.friction * w0.body.friction;
    let mut fr = -10.0f32;
    if -10.0 <= f {
        fr = f;
        if 10.0 <= f {
            fr = 10.0;
        }
    }
    let e = if r0 <= r1 { r1 } else { r0 };
    cp.combined_friction = fr;
    let mut rest = 0.0f32;
    if 0.0 <= e {
        rest = e;
        if 1.0 <= e {
            rest = 1.0;
        }
    }
    cp.combined_restitution = rest;
    let (part1, index1) = (cp.part1, cp.index1);
    adjust_internal_edge(cp, w1, part1, index1);
    if w0.shape.type_id() == SHAPE_SPHERE {
        let t = w1.shape.type_id();
        if t == SHAPE_TRIANGLE || t == SHAPE_PLANE {
            cp.sphere_on_static = true;
        }
    }
}

impl ContactResult for ManifoldResult<'_> {

    fn add_contact_point(&mut self, normal: Vec3, point: Vec3, depth: f32) {
        let current = self.current;
        let m = &mut self.manifolds[current];
        if !(depth <= m.breaking_threshold) {
            return;
        }
        let (b0, b1) = (self.wrap0.body, self.wrap1.body);
        let swapped = m.body0 != self.wrap0.id;
        let pa_z = depth * normal.z + point.z;
        let pa_x = depth * normal.x + point.x;
        let pa_y = depth * normal.y + point.y;
        let mut new = ManifoldPoint::default();
        if !swapped {
            let t0 = &b0.transform;
            let t1 = &b1.transform;
            let (r0, o0) = (&t0.basis.r, t0.origin);
            let (r1, o1) = (&t1.basis.r, t1.origin);
            let (dx, dz) = (point.x - o1.x, point.z - o1.z);
            let (ax, az, ay) = (pa_x - o0.x, pa_z - o0.z, pa_y - o0.y);
            new.local_a = v3(ay * r0[1].x + ax * r0[0].x + az * r0[2].x, ay * r0[1].y + ax * r0[0].y + az * r0[2].y, ay * r0[1].z + ax * r0[0].z + az * r0[2].z);
            let dy = point.y - o1.y;
            new.local_b = v3(dy * r1[1].x + dx * r1[0].x + dz * r1[2].x, dx * r1[0].y + dy * r1[1].y + dz * r1[2].y, dy * r1[1].z + dx * r1[0].z + dz * r1[2].z);
        } else {
            let t0 = &b0.transform;
            let t1 = &b1.transform;
            let (r0, o0) = (&t0.basis.r, t0.origin);
            let (r1, o1) = (&t1.basis.r, t1.origin);
            let (dx, dy, dz) = (point.x - o0.x, point.y - o0.y, point.z - o0.z);
            let (az, ax, ay) = (pa_z - o1.z, pa_x - o1.x, pa_y - o1.y);
            new.local_a = v3(ax * r1[0].x + ay * r1[1].x + az * r1[2].x, ax * r1[0].y + ay * r1[1].y + az * r1[2].y, ax * r1[0].z + ay * r1[1].z + az * r1[2].z);
            new.local_b = v3(dy * r0[1].x + dx * r0[0].x + dz * r0[2].x, dy * r0[1].y + dx * r0[0].y + dz * r0[2].y, dx * r0[0].z + dy * r0[1].z + dz * r0[2].z);
        }
        new.normal_b = normal;
        new.world_b = point;
        new.world_a = v3(pa_x, pa_y, pa_z);
        new.distance = depth;
        let idx = m.cache_entry(&new);
        let f = b1.friction * b0.friction;
        let clamp10 = |f: f32| {
            let lo = if -10.0 <= f { f } else { -10.0 };
            if lo <= 10.0 { lo } else { 10.0 }
        };
        new.combined_friction = clamp10(f);
        new.combined_restitution = b1.restitution * b0.restitution;
        new.combined_rolling = clamp10(b1.rolling_friction * b0.rolling_friction);
        let (d1, d2) = plane_space(normal);
        new.lat_dir1 = d1;
        new.lat_dir2 = d2;
        if !swapped {
            new.part0 = self.part0;
            new.part1 = self.part1;
            new.index0 = self.index0;
            new.index1 = self.index1;
        } else {
            new.part0 = self.part1;
            new.part1 = self.part0;
            new.index0 = self.index1;
            new.index1 = self.index0;
        }
        let i = if idx < 0 {
            m.add_point(&new)
        } else {

            let old = m.points[idx as usize];
            new.user_data = old.user_data;
            new.applied_impulse = old.applied_impulse;
            new.applied_impulse_lat1 = old.applied_impulse_lat1;
            new.applied_impulse_lat2 = old.applied_impulse_lat2;
            new.life_time = old.life_time;
            m.points[idx as usize] = new;
            idx
        };
        if b0.flags & CF_CUSTOM_MATERIAL != 0 || b1.flags & CF_CUSTOM_MATERIAL != 0 {
            let (w0, w1) = if !swapped { (self.wrap0, self.wrap1) } else { (self.wrap1, self.wrap0) };
            let cp = &mut self.manifolds[current].points[i as usize];
            contact_added(cp, &w0, &w1);
            if let Some(h) = self.actor_hook.as_mut() {
                h(cp, &w0, &w1);
            }
        }
    }
}
