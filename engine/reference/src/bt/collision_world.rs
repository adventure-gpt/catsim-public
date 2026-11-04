use super::algo_compound::{Dbvt, child_world_gate};
use super::algo_convex::transform_inverse;
use super::bvh::{MeshShape, soccar_meshes};
use super::convex_cast::{CastResult, gjk_cast, subsimplex_cast};
use super::simplex::Simplex;
use super::dispatch::{ActorHook, CONTACT_BREAKING_THRESHOLD, Dispatcher};
use super::shapes::*;
use super::sweep::{AxisSweep, Handle};
use super::types::*;
use crate::math::{Mat3, Transform, Vec3, v3};

#[derive(Clone, Debug)]
pub struct CollisionWorld {

    pub bodies: Vec<Body>,
    pub free_bodies: Vec<u32>,

    pub objects: Vec<BodyId>,
    pub shapes: Vec<Shape>,

    pub broadphase: AxisSweep,

    pub dispatcher: Dispatcher,

    pub use_continuous: bool,

    pub force_update_all_aabbs: bool,
}

impl Default for CollisionWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl CollisionWorld {

    pub fn new() -> CollisionWorld {
        CollisionWorld {
            bodies: Vec::new(),
            free_bodies: Vec::new(),
            objects: Vec::new(),
            shapes: Vec::new(),
            broadphase: AxisSweep::new_game(),
            dispatcher: Dispatcher::new(),
            use_continuous: true,
            force_update_all_aabbs: false,
        }
    }

    pub fn add_shape(&mut self, s: Shape) -> ShapeId {
        self.shapes.push(s);
        ShapeId(self.shapes.len() as u32 - 1)
    }

    pub fn insert_body(&mut self, b: Body) -> BodyId {
        match self.free_bodies.pop() {
            Some(i) => {
                self.bodies[i as usize] = b;
                BodyId(i)
            }
            None => {
                self.bodies.push(b);
                BodyId(self.bodies.len() as u32 - 1)
            }
        }
    }

    pub fn add_object(&mut self, id: BodyId, group: u16, mask: u16) {
        self.objects.push(id);
        let b = &self.bodies[id.0 as usize];
        let (lo, hi) = self.shapes[b.shape.0 as usize].broadphase_aabb(&b.transform);
        let h = self.broadphase.create_proxy(lo, hi, id.0, group, mask, &mut self.dispatcher);
        self.bodies[id.0 as usize].proxy = h as u32;
    }

    pub fn remove_object(&mut self, id: BodyId) {
        let h = self.bodies[id.0 as usize].proxy as u16;
        if h != 0 {
            self.broadphase.pairs.clean_proxy_from_pairs(h, &mut self.dispatcher);
            self.broadphase.destroy_proxy(h, &mut self.dispatcher);
            self.bodies[id.0 as usize].proxy = 0;
        }
        if let Some(at) = self.objects.iter().position(|o| *o == id) {
            self.objects.swap_remove(at);
        }
    }

    pub fn free_body(&mut self, id: BodyId) {
        self.free_bodies.push(id.0);
    }

    pub fn update_aabbs(&mut self) {
        for n in 0..self.objects.len() {
            let id = self.objects[n];
            let b = &self.bodies[id.0 as usize];
            if !self.force_update_all_aabbs && b.activation_state == DISABLE_SIMULATION {
                continue;
            }
            let shape = &self.shapes[b.shape.0 as usize];
            let (mut lo, mut hi) = shape.aabb(&b.transform);
            let th = CONTACT_BREAKING_THRESHOLD;
            lo = v3(lo.x - th, lo.y - th, lo.z - th);
            hi = v3(hi.x + th, hi.y + th, hi.z + th);
            if self.use_continuous && b.internal_type == 2 && b.flags & (CF_STATIC | CF_KINEMATIC) == 0 {
                let (lo2, hi2) = shape.aabb(&b.interp_transform);
                let lo2 = v3(lo2.x - th, lo2.y - th, lo2.z - th);
                let hi2 = v3(hi2.x + th, hi2.y + th, hi2.z + th);
                lo = lo.min(lo2);
                hi = hi.max(hi2);
            }
            let (dx, dy, dz) = (hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
            if b.flags & CF_STATIC != 0 || dy * dy + dx * dx + dz * dz < 1e12 {
                let h = b.proxy as u16;
                self.broadphase.set_aabb(h, lo, hi, &mut self.dispatcher);
            } else {
                let b = &mut self.bodies[id.0 as usize];
                if b.activation_state != DISABLE_DEACTIVATION && b.activation_state != DISABLE_SIMULATION {
                    b.activation_state = DISABLE_SIMULATION;
                }
            }
        }
    }

    pub fn perform_discrete_collision_detection(&mut self) {
        self.perform_discrete_collision_detection_hooked(None);
    }

    pub fn perform_discrete_collision_detection_hooked(&mut self, hook: Option<&mut ActorHook<'_>>) {
        self.update_aabbs();
        let CollisionWorld { bodies, shapes, broadphase, dispatcher, .. } = self;
        let AxisSweep { pairs, handles, .. } = broadphase;
        dispatcher.dispatch_all_collision_pairs(pairs, handles, bodies, shapes, hook);
    }

    pub fn manifolds(&self) -> &[ManifoldId] {
        &self.dispatcher.array
    }

    pub fn manifold(&self, id: ManifoldId) -> &Manifold {
        &self.dispatcher.slab[id.0 as usize]
    }
}

const fn t(b: [u32; 9], o: [u32; 3]) -> ([u32; 9], [u32; 3]) {
    (b, o)
}
const ONE: u32 = 0x3f80_0000;
const NEG0: u32 = 0x8000_0000;
const IDENT: [u32; 9] = [ONE, 0, 0, 0, ONE, 0, 0, 0, ONE];



static SOCCAR_PLANES: std::sync::LazyLock<Vec<(u32, u32, ([u32; 9], [u32; 3]))>> =
    std::sync::LazyLock::new(|| crate::data::table("planes").iter().map(|r| (r.w(0), r.w(1), (r.wn(2), r.wn(11)))).collect());


fn transform_from_bits(b: &[u32; 9], o: &[u32; 3]) -> Transform {
    let f = f32::from_bits;
    Transform {
        basis: Mat3 { r: [v3(f(b[0]), f(b[1]), f(b[2])), v3(f(b[3]), f(b[4]), f(b[5])), v3(f(b[6]), f(b[7]), f(b[8]))] },
        origin: Vec3::from_bits(*o),
    }
}

pub const WORLD_FRICTION: f32 = 1.0;
pub const WORLD_RESTITUTION: f32 = f32::from_bits(0x3e99_999a);

pub const WORLD_GROUP: u16 = 1;

impl CollisionWorld {

    pub fn add_static(&mut self, shape: ShapeId, transform: Transform, group: u16, mask: u16) -> BodyId {
        let b = Body {
            transform,
            shape,
            flags: CF_STATIC,
            activation_state: DISABLE_SIMULATION,
            friction: WORLD_FRICTION,
            restitution: WORLD_RESTITUTION,
            ..Body::default()
        };
        let id = self.insert_body(b);
        self.add_object(id, group, mask);
        id
    }

    pub fn add_soccar_statics_with(&mut self, meshes: Vec<(Vec3, MeshShape)>) {
        self.add_soccar_statics_selected(meshes, true);
    }

    fn add_soccar_statics_selected(&mut self, meshes: Vec<(Vec3, MeshShape)>, goal_plates: bool) {
        for (hx, hy, (b, o)) in SOCCAR_PLANES.iter().skip(if goal_plates { 0 } else { 2 }) {
            let s = self.add_shape(Shape::Plane(PlaneShape::new(f32::from_bits(*hx), f32::from_bits(*hy))));
            self.add_static(s, transform_from_bits(b, o), WORLD_GROUP, 0xffff);
        }
        for (n, (origin, mesh)) in meshes.into_iter().enumerate() {
            let s = self.add_shape(Shape::Mesh(mesh));
            let mut tr = transform_from_bits(&super::bvh::mesh_bases()[n], &[0, 0, 0]);
            tr.origin = origin;
            self.add_static(s, tr, WORLD_GROUP, 0xffff);
        }
    }

    pub fn add_soccar_statics(&mut self, dir: &std::path::Path) -> std::io::Result<()> {
        self.add_soccar_statics_for_map(dir, true)
    }

    pub fn add_soccar_statics_for_map(&mut self, dir: &std::path::Path, goal_plates: bool) -> std::io::Result<()> {
        self.add_soccar_statics_selected(soccar_meshes(dir)?, goal_plates);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LocalRayResult {
    pub body: BodyId,
    pub info: Option<(i32, i32)>,
    pub normal: Vec3,
    pub fraction: f32,
}

pub trait RayCallback {
    fn closest_fraction(&self) -> f32;
    fn flags(&self) -> u32;
    fn add_single_result(&mut self, hit: &LocalRayResult, normal_in_world: bool) -> f32;

    fn needs_collision(&self, handle: &Handle, body: &Body) -> bool;
}

#[derive(Clone, Copy, Debug)]
pub struct ClosestRay {
    pub closest: f32,
    pub body: Option<BodyId>,
    pub group: u16,
    pub mask: u16,
    pub flags: u32,
    pub from: Vec3,
    pub to: Vec3,
    pub normal: Vec3,
    pub point: Vec3,
}

impl ClosestRay {
    pub fn new(from: Vec3, to: Vec3) -> ClosestRay {
        ClosestRay { closest: 1.0, body: None, group: 1, mask: 0xffff, flags: 0, from, to, normal: Vec3::ZERO, point: Vec3::ZERO }
    }
}

impl RayCallback for ClosestRay {
    fn closest_fraction(&self) -> f32 {
        self.closest
    }
    fn flags(&self) -> u32 {
        self.flags
    }

    fn add_single_result(&mut self, hit: &LocalRayResult, _normal_in_world: bool) -> f32 {
        self.closest = hit.fraction;
        self.body = Some(hit.body);
        self.normal = hit.normal;
        let t = hit.fraction;
        let s = 1.0 - t;
        self.point = v3(s * self.from.x + t * self.to.x, s * self.from.y + t * self.to.y, s * self.from.z + t * self.to.z);
        hit.fraction
    }

    fn needs_collision(&self, handle: &Handle, _body: &Body) -> bool {
        handle.group & self.mask != 0 && self.group & handle.mask != 0
    }
}

struct ChildAdder<'a> {
    next: &'a mut dyn RayCallback,
    child: i32,
    closest: f32,
    flags: u32,
}

impl RayCallback for ChildAdder<'_> {
    fn closest_fraction(&self) -> f32 {
        self.closest
    }
    fn flags(&self) -> u32 {
        self.flags
    }
    fn add_single_result(&mut self, hit: &LocalRayResult, normal_in_world: bool) -> f32 {
        let mut h = *hit;
        if h.info.is_none() {
            h.info = Some((-1, self.child));
        }
        let f = self.next.add_single_result(&h, normal_in_world);
        self.closest = self.next.closest_fraction();
        f
    }

    fn needs_collision(&self, handle: &Handle, body: &Body) -> bool {
        self.next.needs_collision(handle, body)
    }
}

struct RayBox {
    from: Vec3,
    inv: Vec3,
    lambda: f32,
}

impl RayBox {

    #[inline(always)]
    fn new(from: Vec3, to: Vec3) -> RayBox {
        let (dx, dy, dz) = (to.x - from.x, to.y - from.y, to.z - from.z);
        let k = 1.0 / (dy * dy + dx * dx + dz * dz).sqrt();
        let (rx, ry, rz) = (k * dx, k * dy, k * dz);
        let large = 1e18f32;
        let inv = v3(if rx == 0.0 { large } else { 1.0 / rx }, if ry == 0.0 { large } else { 1.0 / ry }, if rz == 0.0 { large } else { 1.0 / rz });
        RayBox { from, inv, lambda: 0.0 }.with_lambda(rx, ry, rz, dx, dy, dz)
    }
    #[inline(always)]
    fn with_lambda(mut self, rx: f32, ry: f32, rz: f32, dx: f32, dy: f32, dz: f32) -> RayBox {
        self.lambda = rx * dx + ry * dy + rz * dz;
        self
    }

    #[inline(always)]
    fn hits(&self, lo: Vec3, hi: Vec3) -> bool {
        let (f, i) = (self.from, self.inv);
        let (x0, x1) = if i.x < 0.0 { (hi.x, lo.x) } else { (lo.x, hi.x) };
        let (y0, y1) = if i.y < 0.0 { (hi.y, lo.y) } else { (lo.y, hi.y) };
        let (z0, z1) = if i.z < 0.0 { (hi.z, lo.z) } else { (lo.z, hi.z) };
        let tmin = (x0 - f.x) * i.x;
        let tmax = (x1 - f.x) * i.x;
        let tymin = (y0 - f.y) * i.y;
        let tymax = (y1 - f.y) * i.y;
        if !(tmin <= tymax && tymin <= tmax) {
            return false;
        }
        let t0 = if tymin <= tmin { tmin } else { tymin };
        let t1 = if tmax <= tymax { tmax } else { tymax };
        let tzmin = (z0 - f.z) * i.z;
        let tzmax = (z1 - f.z) * i.z;
        if !(t0 <= tzmax && tzmin <= t1) {
            return false;
        }
        let a = if tzmin <= t0 { t0 } else { tzmin };
        if !(a < self.lambda) {
            return false;
        }
        let b = if t1 <= tzmax { t1 } else { tzmax };
        0.0 < b
    }
}

#[allow(clippy::too_many_arguments)]
fn ray_triangle(v: &[Vec3; 3], part: i32, index: i32, from: Vec3, to: Vec3, flags: u32, hit_fraction: &mut f32, body: BodyId, t: &Transform, cb: &mut dyn RayCallback) {
    let (ex2, ey2, ez2) = (v[2].x - v[0].x, v[2].y - v[0].y, v[2].z - v[0].z);
    let ny = ex2 * (v[1].z - v[0].z) - ez2 * (v[1].x - v[0].x);
    let nz = ey2 * (v[1].x - v[0].x) - ex2 * (v[1].y - v[0].y);
    let nx = ez2 * (v[1].y - v[0].y) - ey2 * (v[1].z - v[0].z);
    let d = ny * v[0].y + nx * v[0].x + nz * v[0].z;
    let da = (ny * from.y + nx * from.x + nz * from.z) - d;
    let db = (ny * to.y + nx * to.x + nz * to.z) - d;
    if !(db * da < 0.0) || (flags & 1 != 0 && !(0.0 < da)) {
        return;
    }
    let f = da / (da - db);
    if !(f < *hit_fraction) {
        return;
    }
    let len2 = ny * ny + nx * nx + nz * nz;
    let tol = len2 * f32::from_bits(0xb8d1_b717);
    let s = 1.0 - f;
    let px = s * from.x + to.x * f;
    let py = s * from.y + to.y * f;
    let pz = s * from.z + to.z * f;
    let (ax, ay, az) = (v[0].x - px, v[0].y - py, v[0].z - pz);
    let (bx, by, bz) = (v[1].x - px, v[1].y - py, v[1].z - pz);
    if !(tol <= (bz * ay - by * az) * nx + (bx * az - bz * ax) * ny + (by * ax - bx * ay) * nz) {
        return;
    }
    let (cx, cy, cz) = (v[2].x - px, v[2].y - py, v[2].z - pz);
    if !(tol <= (cz * by - cy * bz) * nx + (cx * bz - cz * bx) * ny + (cy * bx - cx * by) * nz) {
        return;
    }
    if !(tol <= (cz * ax - cx * az) * ny + (cy * az - cz * ay) * nx + (cx * ay - cy * ax) * nz) {
        return;
    }
    let k = 1.0 / len2.sqrt();
    let (mut ox, mut oy, mut oz) = (nx * k, ny * k, nz * k);
    if !(flags & 2 != 0 || 0.0 < da) {
        ox = -ox;
        oy = -oy;
        oz = -oz;
    }
    let r = &t.basis.r;
    let world = v3(oy * r[0].y + ox * r[0].x + oz * r[0].z, oy * r[1].y + ox * r[1].x + oz * r[1].z, oy * r[2].y + ox * r[2].x + oz * r[2].z);
    let hit = LocalRayResult { body, info: Some((part, index)), normal: world, fraction: f };
    *hit_fraction = cb.add_single_result(&hit, true);
}

impl MeshRay for MeshShape {

    fn ray_walk(&self, from: Vec3, to: Vec3, mut f: impl FnMut(i32, i32)) {
        let ray = RayBox::new(from, to);
        let zero = Vec3::ZERO;
        let lo = v3(if to.x < from.x { to.x } else { from.x }, if to.y < from.y { to.y } else { from.y }, if to.z < from.z { to.z } else { from.z });
        let hi = v3(if from.x < to.x { to.x } else { from.x }, if from.y < to.y { to.y } else { from.y }, if from.z < to.z { to.z } else { from.z });
        let b = &self.bvh;
        let (qlo, qhi) = b.quantize_query(lo + zero, hi + zero);
        let (q, mn) = (b.quantization, b.aabb_min);
        let mut i = 0i32;
        while i < b.cur_node_index {
            let n = &b.nodes[i as usize];
            let e = n.escape_or_triangle;
            let mut hit = false;
            if n.min[1] <= qhi[1] && qlo[2] <= n.max[2] && qlo[1] <= n.max[1] && n.min[0] <= qhi[0] && n.min[2] <= qhi[2] && qlo[0] <= n.max[0] {
                let nlo = v3((n.min[0] as f32 / q.x + mn.x) - 0.0, (n.min[1] as f32 / q.y + mn.y) - 0.0, (n.min[2] as f32 / q.z + mn.z) - 0.0);
                let nhi = v3((n.max[0] as f32 / q.x + mn.x) - 0.0, (n.max[1] as f32 / q.y + mn.y) - 0.0, (n.max[2] as f32 / q.z + mn.z) - 0.0);
                hit = ray.hits(nlo, nhi);
            }
            if e < 0 {
                if hit {
                    i += 1;
                } else {
                    i += -e;
                }
            } else {
                if hit {
                    f(e >> 21, e & 0x1f_ffff);
                }
                i += 1;
            }
        }
    }
}

trait MeshRay {
    fn ray_walk(&self, from: Vec3, to: Vec3, f: impl FnMut(i32, i32));
}

fn dbvt_ray(tree: &Dbvt, from: Vec3, to: Vec3, mut f: impl FnMut(i32)) {
    if tree.root < 0 {
        return;
    }
    let ray = RayBox::new(from, to);
    let mut stack = vec![tree.root];
    while let Some(i) = stack.pop() {
        let n = &tree.nodes[i as usize];
        if ray.hits(n.min, n.max) {
            if n.child1 < 0 {
                f(n.data);
            } else {
                stack.push(n.child0);
                stack.push(n.child1);
            }
        }
    }
}

pub fn ray_test_single(from: &Transform, to: &Transform, body: BodyId, shape: &Shape, t: &Transform, cb: &mut dyn RayCallback) {
    let r = &t.basis.r;
    match shape {
        Shape::Plane(_) => {
            let nx = r[0].y * 0.0 + r[0].x * 0.0 + r[0].z;
            let ny = r[1].y * 0.0 + r[1].x * 0.0 + r[1].z;
            let nz = r[2].y * 0.0 + r[2].x * 0.0 + r[2].z;
            let (f, g, o) = (from.origin, to.origin, t.origin);
            let span = -((g.y - f.y) * ny + (g.x - f.x) * nx + (g.z - f.z) * nz);
            let reach = -((o.y - f.y) * ny + (o.x - f.x) * nx + (o.z - f.z) * nz);
            let eps = f32::from_bits(0x3400_0000);
            if eps < span && eps < reach && reach <= span {
                let fraction = reach / span;
                if fraction < cb.closest_fraction() {
                    cb.add_single_result(&LocalRayResult { body, info: None, normal: v3(nx, ny, nz), fraction }, true);
                }
            }
        }
        Shape::Mesh(m) => {
            let inv = transform_inverse(t);
            let i = &inv.basis.r;
            let local = |p: Vec3| v3(p.x * i[0].x + p.y * i[0].y + p.z * i[0].z + inv.origin.x, p.x * i[1].x + p.y * i[1].y + p.z * i[1].z + inv.origin.y, p.x * i[2].x + p.y * i[2].y + p.z * i[2].z + inv.origin.z);
            let (lf, lt) = (local(from.origin), local(to.origin));
            let flags = cb.flags();
            let mut hit_fraction = cb.closest_fraction();
            m.ray_walk(lf, lt, |part, index| {
                let v = m.triangle(index);
                ray_triangle(&v, part, index, lf, lt, flags, &mut hit_fraction, body, t, cb);
            });
        }
        Shape::Compound(c) => {
            let o = t.origin;
            let local = |p: Vec3| {
                let (x, y, z) = (p.x - o.x, p.y - o.y, p.z - o.z);
                v3(r[1].x * y + r[0].x * x + r[2].x * z, r[1].y * y + r[0].y * x + r[2].y * z, r[1].z * y + r[0].z * x + r[2].z * z)
            };
            let (lf, lt) = (local(from.origin), local(to.origin));
            let mut hits: Vec<i32> = Vec::new();
            dbvt_ray(&c.tree, lf, lt, |k| hits.push(k));
            for k in hits {
                let child = &c.children[k as usize];
                let world = child_world_gate(t, &child.transform);
                let (closest, flags) = (cb.closest_fraction(), cb.flags());
                let mut adder = ChildAdder { next: &mut *cb, child: k, closest, flags };
                ray_test_single(from, to, body, &child.shape, &world, &mut adder);
            }
        }
        _ => {

            let point = Shape::Sphere(SphereShape::new(0.0));
            let mut result = CastResult::new(cb.closest_fraction());
            let mut simplex = Simplex::new();
            let found = if cb.flags() & 4 != 0 {
                subsimplex_cast(&point, shape, &mut simplex, from, to, t, t, &mut result)
            } else {
                gjk_cast(&point, shape, &mut simplex, from, to, t, t, &mut result)
            };
            if found {
                let n = result.normal;
                let len2 = n.x * n.x + n.y * n.y + n.z * n.z;
                if f32::from_bits(0x38d1_b717) < len2 && result.fraction < cb.closest_fraction() {
                    let k = 1.0 / len2.sqrt();
                    cb.add_single_result(&LocalRayResult { body, info: None, normal: v3(n.x * k, n.y * k, n.z * k), fraction: result.fraction }, true);
                }
            }
        }
    }
}

impl CollisionWorld {

    pub fn ray_test_object(&self, from: Vec3, to: Vec3, object: BodyId) -> ClosestRay {
        let b = &self.bodies[object.0 as usize];
        let mut cb = ClosestRay::new(from, to);
        let (f, g) = (Transform { basis: Mat3::IDENTITY, origin: from }, Transform { basis: Mat3::IDENTITY, origin: to });
        ray_test_single(&f, &g, object, &self.shapes[b.shape.0 as usize], &b.transform, &mut cb);
        cb
    }

    pub fn ray_test(&self, from: Vec3, to: Vec3, cb: &mut dyn RayCallback) {
        let (f, g) = (Transform { basis: Mat3::IDENTITY, origin: from }, Transform { basis: Mat3::IDENTITY, origin: to });
        let bp = &self.broadphase;
        for i in 1..(bp.num_handles as usize * 2 + 1) {
            let e = bp.edges[0][i];
            if e.pos & 1 == 0 {
                continue;
            }
            let h = &bp.handles[e.handle as usize];
            if cb.closest_fraction() == 0.0 {
                continue;
            }
            let id = BodyId(h.client);
            let b = &self.bodies[id.0 as usize];
            if cb.needs_collision(h, b) {
                ray_test_single(&f, &g, id, &self.shapes[b.shape.0 as usize], &b.transform, cb);
            }
        }
    }
}

#[cfg(feature = "car")]
impl super::vehicle::VehicleWorld for CollisionWorld {
    fn body(&self, id: BodyId) -> &Body {
        &self.bodies[id.0 as usize]
    }
    fn body_mut(&mut self, id: BodyId) -> &mut Body {
        &mut self.bodies[id.0 as usize]
    }
    fn aabb_test(&self, min: Vec3, max: Vec3, visit: &mut dyn FnMut(BodyId, u16, u16)) {
        self.broadphase.aabb_test(min, max, |h| visit(BodyId(h.client), h.group, h.mask));
    }
    fn proxy_group(&self, id: BodyId) -> u16 {
        self.broadphase.handles[self.bodies[id.0 as usize].proxy as usize].group
    }
    fn ray_test(&self, from: Vec3, to: Vec3, object: BodyId) -> Option<super::vehicle::RayHit> {
        let c = self.ray_test_object(from, to, object);
        c.body.map(|_| super::vehicle::RayHit { fraction: c.closest, normal: c.normal, point: c.point })
    }
}
