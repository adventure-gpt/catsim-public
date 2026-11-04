use super::algo_compound::{CompoundAlgo, CompoundCompoundAlgo};
use super::algo_concave::ConvexConcaveAlgo;
use super::manifold::Wrap;
use super::pairs::PairCache;
use super::shapes::*;
use super::sweep::Handle;
use super::types::*;

pub const CONTACT_BREAKING_THRESHOLD: f32 = f32::from_bits(0x3ca3_d70a);

#[derive(Clone, Copy)]
pub struct ResultOut<'w> {
    pub manifold: Option<ManifoldId>,
    pub wrap0: Wrap<'w>,
    pub wrap1: Wrap<'w>,
    pub part0: i32,
    pub part1: i32,
    pub index0: i32,
    pub index1: i32,
}

impl<'w> ResultOut<'w> {

    pub fn new(w0: &Wrap<'w>, w1: &Wrap<'w>) -> ResultOut<'w> {
        ResultOut { manifold: None, wrap0: *w0, wrap1: *w1, part0: 0, part1: 0, index0: 0, index1: 0 }
    }

    #[inline]
    pub fn take_ids(&mut self, inner: &ResultOut) {
        self.manifold = inner.manifold;
        self.part0 = inner.part0;
        self.part1 = inner.part1;
        self.index0 = inner.index0;
        self.index1 = inner.index1;
    }
}

#[derive(Clone, Debug)]
pub enum Algorithm {

    Empty,

    ConvexConvex { manifold: Option<ManifoldId>, own: bool },

    SphereSphere { manifold: Option<ManifoldId>, own: bool },

    SphereTriangle { manifold: Option<ManifoldId>, own: bool, swapped: bool },

    BoxBox { manifold: Option<ManifoldId>, own: bool },

    ConvexPlane { manifold: Option<ManifoldId>, own: bool, swapped: bool },

    ConvexConcave(Box<ConvexConcaveAlgo>),

    Compound(Box<CompoundAlgo>),

    CompoundCompound(Box<CompoundCompoundAlgo>),
}

pub type ActorHook<'h> = dyn FnMut(&mut ManifoldPoint, &Wrap, &Wrap) + 'h;

#[inline(always)]
fn rehook<'b>(h: &'b mut Option<&mut ActorHook<'_>>) -> Option<&'b mut ActorHook<'b>> {
    match h {
        Some(x) => Some(&mut **x),
        None => None,
    }
}

pub struct Ctx<'a> {
    pub disp: &'a mut Dispatcher,
    pub bodies: &'a [Body],
    pub shapes: &'a [Shape],
    pub hook: Option<&'a mut ActorHook<'a>>,
}

#[derive(Clone, Debug, Default)]
pub struct Dispatcher {

    pub slab: Vec<Manifold>,
    pub free: Vec<u32>,

    pub array: Vec<ManifoldId>,
}

#[inline]
pub fn needs_collision(a: &Body, b: &Body) -> bool {
    !((a.activation_state == ISLAND_SLEEPING || a.activation_state == DISABLE_SIMULATION)
        && (b.activation_state == ISLAND_SLEEPING || b.activation_state == DISABLE_SIMULATION))
}

#[inline]
pub fn needs_response(a: &Body, b: &Body) -> bool {
    if a.flags & CF_NO_RESPONSE != 0 || b.flags & CF_NO_RESPONSE != 0 {
        return false;
    }
    a.flags & (CF_STATIC | CF_KINEMATIC) == 0 || b.flags & (CF_STATIC | CF_KINEMATIC) == 0
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pick {
    SphereSphere,
    SphereTriangle,
    TriangleSphere,
    BoxBox,
    ConvexPlane,
    PlaneConvex,
    ConvexConvex,
    ConvexConcave,
    ConcaveConvex,
    CompoundCompound,
    Compound,
    SwappedCompound,
    Empty,
}

fn pick(t0: i32, t1: i32) -> Pick {
    if t0 == SHAPE_SPHERE && t1 == SHAPE_SPHERE {
        return Pick::SphereSphere;
    }
    if t0 == SHAPE_SPHERE && t1 == SHAPE_TRIANGLE {
        return Pick::SphereTriangle;
    }
    if t0 == SHAPE_TRIANGLE && t1 == SHAPE_SPHERE {
        return Pick::TriangleSphere;
    }
    if t0 == SHAPE_BOX && t1 == SHAPE_BOX {
        return Pick::BoxBox;
    }
    if t0 < 0x14 && t1 == SHAPE_PLANE {
        return Pick::ConvexPlane;
    }
    if t1 < 0x14 && t0 == SHAPE_PLANE {
        return Pick::PlaneConvex;
    }
    if t0 < 0x14 {
        if t1 < 0x14 {
            return Pick::ConvexConvex;
        }
        if (0x15..=0x1d).contains(&t1) {
            return Pick::ConvexConcave;
        }
    } else {
        if t1 < 0x14 && (0x15..=0x1d).contains(&t0) {
            return Pick::ConcaveConvex;
        }
        if t0 == SHAPE_COMPOUND {
            return if t1 == SHAPE_COMPOUND { Pick::CompoundCompound } else { Pick::Compound };
        }
    }
    if t1 == SHAPE_COMPOUND { Pick::SwappedCompound } else { Pick::Empty }
}

impl Dispatcher {
    pub fn new() -> Dispatcher {
        Dispatcher::default()
    }

    pub fn get_new_manifold(&mut self, bodies: &[Body], shapes: &[Shape], b0: BodyId, b1: BodyId) -> ManifoldId {
        let (a, b) = (&bodies[b0.0 as usize], &bodies[b1.0 as usize]);
        let t1 = shapes[b.shape.0 as usize].contact_breaking_threshold(CONTACT_BREAKING_THRESHOLD);
        let t0 = shapes[a.shape.0 as usize].contact_breaking_threshold(CONTACT_BREAKING_THRESHOLD);
        let breaking = if t1 <= t0 { t1 } else { t0 };
        let processing = if b.contact_processing_threshold <= a.contact_processing_threshold { b.contact_processing_threshold } else { a.contact_processing_threshold };
        let mut m = Manifold::new(b0, b1, breaking, processing);
        m.index1a = self.array.len() as i32;
        let id = match self.free.pop() {
            Some(i) => {
                self.slab[i as usize] = m;
                ManifoldId(i)
            }
            None => {
                self.slab.push(m);
                ManifoldId(self.slab.len() as u32 - 1)
            }
        };
        self.array.push(id);
        id
    }

    pub fn clear_manifold(&mut self, id: ManifoldId) {
        self.slab[id.0 as usize].clear();
    }

    pub fn release_manifold(&mut self, id: ManifoldId) {
        self.clear_manifold(id);
        let at = self.slab[id.0 as usize].index1a as usize;
        let last = self.array.len() - 1;
        self.array.swap(at, last);
        let moved = self.array[at];
        self.slab[moved.0 as usize].index1a = at as i32;
        self.array.pop();
        self.free.push(id.0);
    }

    pub fn free_algorithm(&mut self, a: Algorithm) {
        match a {
            Algorithm::Empty => {}
            Algorithm::ConvexConvex { manifold, own }
            | Algorithm::SphereSphere { manifold, own }
            | Algorithm::SphereTriangle { manifold, own, .. }
            | Algorithm::BoxBox { manifold, own }
            | Algorithm::ConvexPlane { manifold, own, .. } => {
                if own {
                    if let Some(m) = manifold {
                        self.release_manifold(m);
                    }
                }
            }
            Algorithm::ConvexConcave(c) => {
                self.clear_manifold(c.manifold);
                self.release_manifold(c.manifold);
            }
            Algorithm::Compound(c) => {
                for child in c.children.into_iter().flatten() {
                    self.free_algorithm(*child);
                }
            }
            Algorithm::CompoundCompound(c) => {
                for p in c.pairs {
                    self.free_algorithm(*p.algorithm);
                }
            }
        }
    }

    pub fn find_algorithm(&mut self, bodies: &[Body], shapes: &[Shape], w0: &Wrap, w1: &Wrap, shared: Option<ManifoldId>) -> Algorithm {
        let (b0, b1) = (w0.id, w1.id);
        let needs = |bodies: &[Body]| needs_collision(&bodies[b0.0 as usize], &bodies[b1.0 as usize]);
        match pick(w0.shape.type_id(), w1.shape.type_id()) {
            Pick::Empty => Algorithm::Empty,
            Pick::ConvexConvex => Algorithm::ConvexConvex { manifold: shared, own: false },
            Pick::SphereSphere => Algorithm::SphereSphere { manifold: Some(self.get_new_manifold(bodies, shapes, b0, b1)), own: true },
            Pick::SphereTriangle | Pick::TriangleSphere => {
                let swapped = pick(w0.shape.type_id(), w1.shape.type_id()) == Pick::TriangleSphere;
                match shared {
                    Some(m) => Algorithm::SphereTriangle { manifold: Some(m), own: false, swapped },
                    None => Algorithm::SphereTriangle { manifold: Some(self.get_new_manifold(bodies, shapes, b0, b1)), own: true, swapped },
                }
            }
            Pick::BoxBox => {

                if needs(bodies) {
                    Algorithm::BoxBox { manifold: Some(self.get_new_manifold(bodies, shapes, b0, b1)), own: true }
                } else {
                    Algorithm::BoxBox { manifold: None, own: false }
                }
            }
            Pick::ConvexPlane => {
                if needs(bodies) {
                    Algorithm::ConvexPlane { manifold: Some(self.get_new_manifold(bodies, shapes, b0, b1)), own: true, swapped: false }
                } else {
                    Algorithm::ConvexPlane { manifold: None, own: false, swapped: false }
                }
            }
            Pick::PlaneConvex => {
                if needs(bodies) {
                    Algorithm::ConvexPlane { manifold: Some(self.get_new_manifold(bodies, shapes, b1, b0)), own: true, swapped: true }
                } else {
                    Algorithm::ConvexPlane { manifold: None, own: false, swapped: true }
                }
            }
            Pick::ConvexConcave => Algorithm::ConvexConcave(Box::new(ConvexConcaveAlgo::new(self.get_new_manifold(bodies, shapes, b0, b1), false))),
            Pick::ConcaveConvex => Algorithm::ConvexConcave(Box::new(ConvexConcaveAlgo::new(self.get_new_manifold(bodies, shapes, b1, b0), true))),
            Pick::Compound => Algorithm::Compound(Box::new(CompoundAlgo::new(w0, false, shared))),
            Pick::SwappedCompound => Algorithm::Compound(Box::new(CompoundAlgo::new(w1, true, shared))),
            Pick::CompoundCompound => Algorithm::CompoundCompound(Box::new(CompoundCompoundAlgo::new(shared))),
        }
    }

    pub fn dispatch_all_collision_pairs(&mut self, cache: &mut PairCache, handles: &[Handle], bodies: &[Body], shapes: &[Shape], hook: Option<&mut ActorHook<'_>>) {
        self.dispatch_all_collision_pairs_statics(cache, handles, bodies, shapes, hook, None);
    }

    pub(super) fn dispatch_all_collision_pairs_statics(&mut self, cache: &mut PairCache, handles: &[Handle], bodies: &[Body], shapes: &[Shape], mut hook: Option<&mut ActorHook<'_>>, statics: Option<&super::statics::StaticTable>) {
        for i in statics.map_or(0, |t| t.ss_prefix)..cache.pairs.len() {
            if let Some(t) = statics {
                let p = &cache.pairs[i];
                if t.is_sleeping_static(p.proxy0) && t.is_sleeping_static(p.proxy1) {
                    continue;
                }
            }
            if self.sphere_plane_far(&cache.pairs[i], handles, bodies, shapes) {
                continue;
            }
            self.near_callback(cache, i, handles, bodies, shapes, rehook(&mut hook));
        }
    }

    #[inline]
    fn sphere_plane_far(&self, pair: &super::pairs::Pair, handles: &[Handle], bodies: &[Body], shapes: &[Shape]) -> bool {
        let Some(Algorithm::ConvexPlane { manifold: Some(m), swapped, .. }) = pair.algorithm.as_deref() else { return false };
        let man = &self.slab[m.0 as usize];
        if man.cached != 0 {
            return false;
        }
        let (b0, b1) = (&bodies[handles[pair.proxy0 as usize].client as usize], &bodies[handles[pair.proxy1 as usize].client as usize]);
        let (conv, plane) = if *swapped { (b1, b0) } else { (b0, b1) };
        let Shape::Sphere(sph) = &shapes[conv.shape.0 as usize] else { return false };
        if !matches!(shapes[plane.shape.0 as usize], Shape::Plane(_)) {
            return false;
        }
        let near_unit = |a: crate::math::Vec3, b: crate::math::Vec3, want: f32| {
            let d = a.x * b.x + a.y * b.y + a.z * b.z;
            (d - want).abs() <= 1e-3
        };
        let (r, p) = (&conv.transform.basis.r, &plane.transform.basis.r);
        if !(near_unit(r[0], r[0], 1.0) && near_unit(r[1], r[1], 1.0) && near_unit(r[2], r[2], 1.0)
            && near_unit(r[0], r[1], 0.0) && near_unit(r[0], r[2], 0.0) && near_unit(r[1], r[2], 0.0))
        {
            return false;
        }

        let n = crate::math::v3(p[0].z, p[1].z, p[2].z);
        if !near_unit(n, n, 1.0) {
            return false;
        }
        let (c, o) = (conv.transform.origin, plane.transform.origin);
        let d = (c.x - o.x) * n.x + (c.y - o.y) * n.y + (c.z - o.z) * n.z;
        let reach = sph.radius * sph.scale.x + man.breaking_threshold + 0.5;
        d.is_finite() && reach.is_finite() && c.x.abs() < 1e4 && c.y.abs() < 1e4 && c.z.abs() < 1e4 && d > reach
    }

    fn near_callback(&mut self, cache: &mut PairCache, index: usize, handles: &[Handle], bodies: &[Body], shapes: &[Shape], mut hook: Option<&mut ActorHook<'_>>) {
        let pair = &mut cache.pairs[index];

        if matches!(pair.algorithm.as_deref(), Some(Algorithm::Empty)) {
            return;
        }
        let (b0, b1) = (BodyId(handles[pair.proxy0 as usize].client), BodyId(handles[pair.proxy1 as usize].client));
        let (o0, o1) = (&bodies[b0.0 as usize], &bodies[b1.0 as usize]);
        if !needs_collision(o0, o1) {
            return;
        }
        let w0 = Wrap { shape: &shapes[o0.shape.0 as usize], body: o0, id: b0, transform: &o0.transform, part_id: -1, index: -1, info_map: None };
        let w1 = Wrap { shape: &shapes[o1.shape.0 as usize], body: o1, id: b1, transform: &o1.transform, part_id: -1, index: -1, info_map: None };
        if pair.algorithm.is_none() {
            let a = self.find_algorithm(bodies, shapes, &w0, &w1, None);
            pair.algorithm = Some(Box::new(a));
        }
        let mut result = ResultOut::new(&w0, &w1);
        #[cfg(feature = "profile")]
        let (_span, pid, before) = {
            let a = pair.algorithm.as_deref().unwrap();
            let other = |s: &Shape| match s { Shape::Plane(_) => 0, Shape::Mesh(_) => 1, _ => 2 };
            let id = match a {
                Algorithm::ConvexConvex { .. } => 16,
                Algorithm::SphereSphere { .. } => 17,
                Algorithm::SphereTriangle { .. } => 18,
                Algorithm::BoxBox { .. } => 19,
                Algorithm::ConvexPlane { .. } => 20,
                Algorithm::ConvexConcave(_) => 21,
                Algorithm::Compound(_) => {
                    let o = if matches!(w0.shape, Shape::Compound(_)) { other(w1.shape) } else { other(w0.shape) };
                    22 + o
                }
                Algorithm::CompoundCompound(_) => 25,
                Algorithm::Empty => 25,
            };
            let mut n = 0;
            visit_contact_manifolds(a, &mut |m| n += self.slab[m.0 as usize].cached);
            (crate::prof::span(id), id, n)
        };
        let mut ctx = Ctx { disp: self, bodies, shapes, hook: rehook(&mut hook) };
        process_algorithm(pair.algorithm.as_mut().unwrap(), &w0, &w1, &mut ctx, &mut result);
        #[cfg(feature = "profile")]
        {
            let mut n = 0;
            visit_contact_manifolds(pair.algorithm.as_deref().unwrap(), &mut |m| n += self.slab[m.0 as usize].cached);
            if before == 0 && n == 0 {
                crate::prof::count(28);
            }
        }
    }
}

pub fn process_algorithm(a: &mut Algorithm, w0: &Wrap, w1: &Wrap, ctx: &mut Ctx, result: &mut ResultOut) {
    match a {
        Algorithm::Empty => {}
        Algorithm::ConvexConvex { manifold, own } => {

            let m = match *manifold {
                Some(m) => m,
                None => {
                    let m = ctx.disp.get_new_manifold(ctx.bodies, ctx.shapes, w0.id, w1.id);
                    *manifold = Some(m);
                    *own = true;
                    m
                }
            };
            result.manifold = Some(m);
            super::collide_narrow::convex_convex(m, *own, w0, w1, ctx, result);
        }
        Algorithm::SphereSphere { manifold, own } => {
            if let Some(m) = *manifold {
                result.manifold = Some(m);
                super::collide_narrow::sphere_sphere(m, *own, w0, w1, ctx, result);
            }
        }
        Algorithm::SphereTriangle { manifold, own, swapped } => {
            if let Some(m) = *manifold {
                result.manifold = Some(m);
                super::collide_narrow::sphere_triangle(m, *own, *swapped, w0, w1, ctx, result);
            }
        }
        Algorithm::BoxBox { manifold, own } => {
            if let Some(m) = *manifold {
                result.manifold = Some(m);
                super::collide_narrow::box_box(m, *own, w0, w1, ctx, result);
            }
        }
        Algorithm::ConvexPlane { manifold, own, swapped } => {
            if let Some(m) = *manifold {
                result.manifold = Some(m);
                super::collide_narrow::convex_plane(m, *own, *swapped, w0, w1, ctx, result);
            }
        }
        Algorithm::ConvexConcave(c) => c.process(w0, w1, ctx, result),
        Algorithm::Compound(c) => c.process(w0, w1, ctx, result),
        Algorithm::CompoundCompound(c) => c.process(w0, w1, ctx, result),
    }
}

pub fn all_contact_manifolds(a: &Algorithm, out: &mut Vec<ManifoldId>) {
    visit_contact_manifolds(a, &mut |m| out.push(m));
}

pub fn visit_contact_manifolds(a: &Algorithm, out: &mut impl FnMut(ManifoldId)) {
    match a {
        Algorithm::Empty => {}
        Algorithm::ConvexConvex { manifold, own }
        | Algorithm::SphereSphere { manifold, own }
        | Algorithm::SphereTriangle { manifold, own, .. }
        | Algorithm::BoxBox { manifold, own }
        | Algorithm::ConvexPlane { manifold, own, .. } => {
            if *own {
                if let Some(m) = *manifold {
                    out(m);
                }
            }
        }
        Algorithm::ConvexConcave(c) => out(c.manifold),
        Algorithm::Compound(c) => {
            for child in c.children.iter().flatten() {
                visit_contact_manifolds(child, out);
            }
        }
        Algorithm::CompoundCompound(c) => {
            for p in &c.pairs {
                visit_contact_manifolds(&p.algorithm, out);
            }
        }
    }
}
