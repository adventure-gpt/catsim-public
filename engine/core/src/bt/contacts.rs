//! The contact memory: what a physics step reads from the last one besides the bodies. The moving bodies' broadphase
//! boxes as the last bounds update left them, the broadphase's overlapping pairs in their order, the contact manifolds in
//! the dispatcher's order with their cached points, and the last step's predictive contacts (the next step releases them
//! first, which moves other manifolds in the order).
//!
//! State access only. Reading it changes nothing; setting it builds the pair cache's algorithms and the manifolds with
//! the dispatcher's own constructors, so a world that never calls the setter steps exactly as it always has.

use super::algo_compound::{SimplePair, child_world_gate};
use super::dispatch::Algorithm;
use super::manifold::Wrap;
use super::pairs::PairCache;
use super::shapes::Shape;
use super::types::*;
use super::world::World;

/// Who holds a manifold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifoldOwner {
    /// Pair `pair` of the pair cache. `child` names the compound children: (-1, -1) for two plain bodies, (k, -1) for a
    /// compound body's child k against the other body, (a, b) for child a of the pair's first body against child b of
    /// its second (two compounds).
    Pair { pair: usize, child: (i32, i32) },
    /// The k-th predictive contact the last step made (continuous collision).
    Predictive(usize),
}

/// One manifold: its owner, its two bodies (points' `local_a` is on `body0`, `normal_b` on `body1`) and its points.
#[derive(Clone, Debug)]
pub struct ManifoldState {
    pub owner: ManifoldOwner,
    pub body0: BodyId,
    pub body1: BodyId,
    pub points: Vec<ManifoldPoint>,
}

/// One overlapping pair, in the pair cache's order. `body0` is the body with the lower broadphase handle.
/// `dispatched`: the pair has its collision algorithm (it has been through the narrow phase).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairState {
    pub body0: BodyId,
    pub body1: BodyId,
    pub dispatched: bool,
}

/// A moving body's broadphase box (with the contact margin), as the last bounds update left it. A step's continuous
/// collision reads these before the step updates them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxState {
    pub body: BodyId,
    pub min: crate::math::Vec3,
    pub max: crate::math::Vec3,
}

#[derive(Clone, Debug, Default)]
pub struct ContactMemory {
    pub boxes: Vec<BoxState>,
    pub pairs: Vec<PairState>,
    /// In the dispatcher's order (the order the solver's islands take them in).
    pub manifolds: Vec<ManifoldState>,
}

fn owners_of(a: &Algorithm, pair: usize, child: (i32, i32), out: &mut Vec<Option<ManifoldOwner>>) {
    let mut put = |m: ManifoldId| {
        let i = m.0 as usize;
        if out.len() <= i {
            out.resize(i + 1, None);
        }
        out[i] = Some(ManifoldOwner::Pair { pair, child });
    };
    match a {
        Algorithm::Empty => {}
        Algorithm::ConvexConvex { manifold, own }
        | Algorithm::SphereSphere { manifold, own }
        | Algorithm::SphereTriangle { manifold, own, .. }
        | Algorithm::BoxBox { manifold, own }
        | Algorithm::ConvexPlane { manifold, own, .. } => {
            if let (true, Some(m)) = (*own, *manifold) {
                put(m);
            }
        }
        Algorithm::ConvexConcave(c) => put(c.manifold),
        Algorithm::Compound(c) => {
            for (k, ch) in c.children.iter().enumerate() {
                if let Some(ch) = ch {
                    owners_of(ch, pair, (k as i32, -1), out);
                }
            }
        }
        Algorithm::CompoundCompound(c) => {
            for p in &c.pairs {
                owners_of(&p.algorithm, pair, (p.index0, p.index1), out);
            }
        }
    }
}

/// The manifold a leaf algorithm holds, made the way its first narrow-phase call makes it when it has none yet.
fn leaf_manifold(a: &mut Algorithm, disp: &mut super::dispatch::Dispatcher, bodies: &[Body], shapes: &[Shape], b0: BodyId, b1: BodyId) -> Result<ManifoldId, String> {
    match a {
        Algorithm::ConvexConvex { manifold, own } => Ok(match *manifold {
            Some(m) => m,
            None => {
                let m = disp.get_new_manifold(bodies, shapes, b0, b1);
                *manifold = Some(m);
                *own = true;
                m
            }
        }),
        Algorithm::SphereSphere { manifold: Some(m), own: true }
        | Algorithm::SphereTriangle { manifold: Some(m), own: true, .. }
        | Algorithm::BoxBox { manifold: Some(m), own: true }
        | Algorithm::ConvexPlane { manifold: Some(m), own: true, .. } => Ok(*m),
        // Made while one of the bodies sleeps: the algorithm makes its manifold only when they collide. The manifold
        // being set was made while they did, so it is made as then.
        Algorithm::BoxBox { manifold: manifold @ None, own } => {
            let m = disp.get_new_manifold(bodies, shapes, b0, b1);
            (*manifold, *own) = (Some(m), true);
            Ok(m)
        }
        Algorithm::ConvexPlane { manifold: manifold @ None, own, swapped } => {
            let m = if *swapped { disp.get_new_manifold(bodies, shapes, b1, b0) } else { disp.get_new_manifold(bodies, shapes, b0, b1) };
            (*manifold, *own) = (Some(m), true);
            Ok(m)
        }
        Algorithm::ConvexConcave(c) => Ok(c.manifold),
        _ => Err("this pair's algorithm holds no manifold of its own".into()),
    }
}

fn is_free_of_manifolds(a: &Algorithm) -> bool {
    matches!(a, Algorithm::Empty | Algorithm::Compound(_) | Algorithm::CompoundCompound(_))
}

impl World {
    /// The contact memory as it is now.
    pub fn contact_memory(&self) -> Result<ContactMemory, String> {
        let cw = &self.cw;
        let handles = &cw.broadphase.handles;
        let mut owner: Vec<Option<ManifoldOwner>> = vec![None; cw.dispatcher.slab.len()];
        let mut boxes = Vec::new();
        for id in &cw.objects {
            let b = &cw.bodies[id.0 as usize];
            if b.flags & CF_STATIC == 0 && b.proxy != 0 {
                let h = &handles[b.proxy as usize];
                boxes.push(BoxState { body: *id, min: h.aabb_min, max: h.aabb_max });
            }
        }
        boxes.sort_by_key(|b| b.body);
        let mut pairs = Vec::with_capacity(cw.broadphase.pairs.pairs.len());
        for (i, p) in cw.broadphase.pairs.pairs.iter().enumerate() {
            pairs.push(PairState {
                body0: BodyId(handles[p.proxy0 as usize].client),
                body1: BodyId(handles[p.proxy1 as usize].client),
                dispatched: p.algorithm.is_some(),
            });
            if let Some(a) = p.algorithm.as_deref() {
                owners_of(a, i, (-1, -1), &mut owner);
            }
        }
        for (k, m) in self.predictive_manifolds.iter().enumerate() {
            owner[m.0 as usize] = Some(ManifoldOwner::Predictive(k));
        }
        let mut manifolds = Vec::with_capacity(cw.dispatcher.array.len());
        for id in &cw.dispatcher.array {
            let m = &cw.dispatcher.slab[id.0 as usize];
            let o = owner.get(id.0 as usize).copied().flatten().ok_or_else(|| format!("manifold {} has no owner", id.0))?;
            manifolds.push(ManifoldState { owner: o, body0: m.body0, body1: m.body1, points: m.points[..m.cached as usize].to_vec() });
        }
        Ok(ContactMemory { boxes, pairs, manifolds })
    }

    /// Replace the contact memory. The boxes go in first, through the broadphase's own update (which keeps its sorted
    /// edges). The pairs are the bodies whose boxes overlap (the broadphase adds and removes a pair only when two boxes
    /// start or stop overlapping, so a pair left out stays out until its boxes part and meet again); this sets their
    /// order. Each manifold is made by its owner's own algorithm, as the narrow phase makes it, in the order given, and
    /// gets the points given. On an error nothing has changed.
    pub fn set_contact_memory(&mut self, mem: &ContactMemory) -> Result<(), String> {
        let nb = self.cw.bodies.len();
        let handle_of = |cw: &super::collision_world::CollisionWorld, b: BodyId| -> Result<u16, String> {
            if b.0 as usize >= nb {
                return Err(format!("no body {}", b.0));
            }
            let h = cw.bodies[b.0 as usize].proxy as u16;
            if h == 0 || cw.broadphase.handles.get(h as usize).is_none_or(|x| x.client != b.0) {
                return Err(format!("body {} is not in the broadphase", b.0));
            }
            Ok(h)
        };
        let mut bp = self.cw.broadphase.clone();
        {
            let mut scratch = self.cw.dispatcher.clone();
            for b in &mem.boxes {
                let h = handle_of(&self.cw, b.body)?;
                bp.set_aabb(h, b.min, b.max, &mut scratch);
            }
        }
        let mut cache = PairCache::new();
        for p in &mem.pairs {
            let (h0, h1) = (handle_of(&self.cw, p.body0)?, handle_of(&self.cw, p.body1)?);
            let n = cache.pairs.len();
            cache.add_overlapping_pair(&bp.handles, h0, h1);
            if cache.pairs.len() != n + 1 {
                return Err(format!("pair ({}, {}) is listed twice or its bodies never collide", p.body0.0, p.body1.0));
            }
        }
        let npred = mem.manifolds.iter().filter(|m| matches!(m.owner, ManifoldOwner::Predictive(_))).count();
        for m in &mem.manifolds {
            if m.points.len() > 4 {
                return Err("a manifold holds at most 4 points".into());
            }
            match m.owner {
                ManifoldOwner::Pair { pair, .. } if pair >= cache.pairs.len() => return Err(format!("no pair {pair}")),
                ManifoldOwner::Predictive(k) if k >= npred => return Err(format!("predictive contact {k} of {npred}")),
                _ => {}
            }
            handle_of(&self.cw, m.body0)?;
            handle_of(&self.cw, m.body1)?;
        }

        let mut disp = super::dispatch::Dispatcher::new();
        let mut predictive = vec![None; npred];
        let mut made = Vec::with_capacity(mem.manifolds.len());
        {
            let (bodies, shapes, handles) = (&self.cw.bodies, &self.cw.shapes, &bp.handles);
            for m in &mem.manifolds {
                let id = match m.owner {
                    ManifoldOwner::Predictive(k) => {
                        if predictive[k].is_some() {
                            return Err(format!("predictive contact {k} is listed twice"));
                        }
                        let id = disp.get_new_manifold(bodies, shapes, m.body0, m.body1);
                        predictive[k] = Some(id);
                        id
                    }
                    ManifoldOwner::Pair { pair, child } => {
                        let p = &mut cache.pairs[pair];
                        let (b0, b1) = (BodyId(handles[p.proxy0 as usize].client), BodyId(handles[p.proxy1 as usize].client));
                        let (o0, o1) = (&bodies[b0.0 as usize], &bodies[b1.0 as usize]);
                        let w0 = Wrap { shape: &shapes[o0.shape.0 as usize], body: o0, id: b0, transform: &o0.transform, part_id: -1, index: -1, info_map: None };
                        let w1 = Wrap { shape: &shapes[o1.shape.0 as usize], body: o1, id: b1, transform: &o1.transform, part_id: -1, index: -1, info_map: None };
                        let before = disp.array.len();
                        if p.algorithm.is_none() {
                            p.algorithm = Some(Box::new(disp.find_algorithm(bodies, shapes, &w0, &w1, None)));
                        }
                        let alg = p.algorithm.as_deref_mut().unwrap();
                        let id = match (child, alg) {
                            ((-1, -1), a) => leaf_manifold(a, &mut disp, bodies, shapes, b0, b1)?,
                            ((k, -1), Algorithm::Compound(c)) => {
                                let (comp, other) = if c.swapped { (&w1, &w0) } else { (&w0, &w1) };
                                let Shape::Compound(cs) = comp.shape else { return Err("compound pair on another shape".into()) };
                                let ch = cs.children.get(k as usize).ok_or_else(|| format!("no compound child {k}"))?;
                                if c.children[k as usize].is_none() {
                                    let t = child_world_gate(comp.transform, &ch.transform);
                                    let cw = Wrap { shape: &ch.shape, body: comp.body, id: comp.id, transform: &t, part_id: -1, index: k, info_map: None };
                                    c.children[k as usize] = Some(Box::new(disp.find_algorithm(bodies, shapes, &cw, other, c.shared)));
                                }
                                leaf_manifold(c.children[k as usize].as_deref_mut().unwrap(), &mut disp, bodies, shapes, comp.id, other.id)?
                            }
                            ((a, b), Algorithm::CompoundCompound(c)) => {
                                let (Shape::Compound(c0), Shape::Compound(c1)) = (w0.shape, w1.shape) else { return Err("compound pair on other shapes".into()) };
                                let (ch0, ch1) = (c0.children.get(a as usize), c1.children.get(b as usize));
                                let (Some(ch0), Some(ch1)) = (ch0, ch1) else { return Err(format!("no compound children ({a}, {b})")) };
                                let at = match c.pairs.iter().position(|p| p.index0 == a && p.index1 == b) {
                                    Some(at) => at,
                                    None => {
                                        let (t0, t1) = (child_world_gate(w0.transform, &ch0.transform), child_world_gate(w1.transform, &ch1.transform));
                                        let x0 = Wrap { shape: &ch0.shape, body: o0, id: b0, transform: &t0, part_id: -1, index: a, info_map: None };
                                        let x1 = Wrap { shape: &ch1.shape, body: o1, id: b1, transform: &t1, part_id: -1, index: b, info_map: None };
                                        let alg = disp.find_algorithm(bodies, shapes, &x0, &x1, c.shared);
                                        c.pairs.push(SimplePair { index0: a, index1: b, algorithm: Box::new(alg) });
                                        c.pairs.len() - 1
                                    }
                                };
                                leaf_manifold(&mut c.pairs[at].algorithm, &mut disp, bodies, shapes, b0, b1)?
                            }
                            (c, _) => return Err(format!("pair {pair} has no compound children {c:?}")),
                        };
                        if disp.array.len() != before + 1 || disp.array[before] != id {
                            return Err(format!("pair {pair} child {child:?}: its manifold is listed twice"));
                        }
                        id
                    }
                };
                let man = &mut disp.slab[id.0 as usize];
                if (man.body0, man.body1) != (m.body0, m.body1) {
                    return Err(format!("manifold of bodies ({}, {}): the game makes it ({}, {})", m.body0.0, m.body1.0, man.body0.0, man.body1.0));
                }
                for (i, pt) in m.points.iter().enumerate() {
                    man.points[i] = *pt;
                }
                man.cached = m.points.len() as i32;
                made.push(id);
            }
            for (i, p) in mem.pairs.iter().enumerate() {
                let q = &mut cache.pairs[i];
                if p.dispatched && q.algorithm.is_none() {
                    let (b0, b1) = (BodyId(handles[q.proxy0 as usize].client), BodyId(handles[q.proxy1 as usize].client));
                    let (o0, o1) = (&bodies[b0.0 as usize], &bodies[b1.0 as usize]);
                    let w0 = Wrap { shape: &shapes[o0.shape.0 as usize], body: o0, id: b0, transform: &o0.transform, part_id: -1, index: -1, info_map: None };
                    let w1 = Wrap { shape: &shapes[o1.shape.0 as usize], body: o1, id: b1, transform: &o1.transform, part_id: -1, index: -1, info_map: None };
                    let n = disp.array.len();
                    let a = disp.find_algorithm(bodies, shapes, &w0, &w1, None);
                    if disp.array.len() != n || !is_free_of_manifolds(&a) {
                        return Err(format!("pair {i} is dispatched but its manifold is not listed"));
                    }
                    q.algorithm = Some(Box::new(a));
                }
            }
        }
        if disp.array != made {
            return Err("the manifolds' order could not be made".into());
        }
        bp.pairs = cache;
        self.cw.broadphase = bp;
        self.cw.dispatcher = disp;
        self.predictive_manifolds = predictive.into_iter().map(|m| m.unwrap()).collect();
        self.cw.refresh_static_pair_prefix();
        Ok(())
    }
}
