use super::algo_convex::transform_inverse;
use super::dispatch::{Algorithm, Ctx, ResultOut, visit_contact_manifolds, process_algorithm};
use super::manifold::Wrap;
use super::shapes::Shape;
use super::types::*;
use crate::math::{Mat3, Transform, Vec3, v3};

#[derive(Clone, Debug)]
pub struct CompoundChild {
    pub transform: Transform,
    pub shape: Shape,
    pub shape_type: i32,
    pub margin: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct DbvtNode {
    pub min: Vec3,
    pub max: Vec3,
    pub parent: i32,
    pub child0: i32,
    pub child1: i32,
    pub data: i32,
}

#[derive(Clone, Debug, Default)]
pub struct Dbvt {
    pub nodes: Vec<DbvtNode>,
    pub root: i32,
}

impl Dbvt {
    fn new() -> Dbvt {
        Dbvt { nodes: Vec::new(), root: -1 }
    }

    #[inline(always)]
    fn minps(a: Vec3, b: Vec3) -> Vec3 {
        v3(if a.x < b.x { a.x } else { b.x }, if a.y < b.y { a.y } else { b.y }, if a.z < b.z { a.z } else { b.z })
    }
    #[inline(always)]
    fn maxps(a: Vec3, b: Vec3) -> Vec3 {
        v3(if a.x > b.x { a.x } else { b.x }, if a.y > b.y { a.y } else { b.y }, if a.z > b.z { a.z } else { b.z })
    }

    pub fn insert(&mut self, min: Vec3, max: Vec3, data: i32) {
        let leaf = self.nodes.len() as i32;
        self.nodes.push(DbvtNode { min, max, parent: -1, child0: -1, child1: -1, data });
        if self.root < 0 {
            self.root = leaf;
            return;
        }
        let (lx, ly, lz) = (min.x + max.x, min.y + max.y, min.z + max.z);
        let prox = |n: &DbvtNode| ((n.max.z + n.min.z) - lz).abs() + ((n.max.x + n.min.x) - lx).abs() + ((n.max.y + n.min.y) - ly).abs();
        let mut at = self.root;
        while self.nodes[at as usize].child1 >= 0 {
            let n = self.nodes[at as usize];
            let (a, b) = (self.nodes[n.child0 as usize], self.nodes[n.child1 as usize]);
            at = if prox(&b) <= prox(&a) { n.child1 } else { n.child0 };
        }
        let sibling = at;
        let prev = self.nodes[sibling as usize].parent;
        let node = self.nodes.len() as i32;
        let (smin, smax) = (self.nodes[sibling as usize].min, self.nodes[sibling as usize].max);
        self.nodes.push(DbvtNode { min: Self::minps(min, smin), max: Self::maxps(max, smax), parent: prev, child0: sibling, child1: leaf, data: 0 });
        self.nodes[sibling as usize].parent = node;
        self.nodes[leaf as usize].parent = node;
        if prev < 0 {
            self.root = node;
            return;
        }
        if self.nodes[prev as usize].child1 == sibling {
            self.nodes[prev as usize].child1 = node;
        } else {
            self.nodes[prev as usize].child0 = node;
        }
        let (mut p, mut c) = (prev, node);
        loop {
            let pn = self.nodes[p as usize];
            let cn = self.nodes[c as usize];
            if pn.min.x <= cn.min.x && pn.min.y <= cn.min.y && pn.min.z <= cn.min.z && cn.max.x <= pn.max.x && cn.max.y <= pn.max.y && cn.max.z <= pn.max.z {
                return;
            }
            let (a, b) = (self.nodes[pn.child0 as usize], self.nodes[pn.child1 as usize]);
            self.nodes[p as usize].min = Self::minps(a.min, b.min);
            self.nodes[p as usize].max = Self::maxps(a.max, b.max);
            c = p;
            p = pn.parent;
            if p < 0 {
                return;
            }
        }
    }

    #[inline(always)]
    fn is_internal(&self, i: i32) -> bool {
        self.nodes[i as usize].child1 >= 0
    }

    pub fn collide_tv(&self, lo: Vec3, hi: Vec3, mut f: impl FnMut(i32)) {
        if self.root < 0 {
            return;
        }
        let n = &self.nodes[self.root as usize];
        if n.child1 < 0 {
            if !(n.max.x < lo.x || n.max.y < lo.y || n.max.z < lo.z
                || hi.x < n.min.x || hi.y < n.min.y || hi.z < n.min.z) { f(n.data); }
            return;
        }
        let mut stack = vec![self.root];
        while let Some(i) = stack.pop() {
            let n = &self.nodes[i as usize];
            let miss = n.max.x < lo.x || n.max.y < lo.y || n.max.z < lo.z || hi.x < n.min.x || hi.y < n.min.y || hi.z < n.min.z;
            if !miss {
                if n.child1 < 0 {
                    f(n.data);
                } else {
                    stack.push(n.child0);
                    stack.push(n.child1);
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompoundShape {
    pub children: Vec<CompoundChild>,
    pub local_aabb_min: Vec3,
    pub local_aabb_max: Vec3,
    pub tree: Dbvt,
    pub update_revision: i32,
    pub margin: f32,
}

impl Default for CompoundShape {
    fn default() -> Self {
        Self::new()
    }
}

impl CompoundShape {

    pub fn new() -> CompoundShape {
        let large = f32::from_bits(0x5d5e_0b6b);
        CompoundShape { children: Vec::new(), local_aabb_min: v3(large, large, large), local_aabb_max: v3(-large, -large, -large), tree: Dbvt::new(), update_revision: 1, margin: 0.0 }
    }

    pub fn add_child(&mut self, transform: Transform, shape: Shape) {
        self.update_revision += 1;
        let (lo, hi) = shape.aabb(&transform);
        if lo.x < self.local_aabb_min.x {
            self.local_aabb_min.x = lo.x;
        }
        if self.local_aabb_max.x < hi.x {
            self.local_aabb_max.x = hi.x;
        }
        if lo.y < self.local_aabb_min.y {
            self.local_aabb_min.y = lo.y;
        }
        if self.local_aabb_max.y < hi.y {
            self.local_aabb_max.y = hi.y;
        }
        if lo.z < self.local_aabb_min.z {
            self.local_aabb_min.z = lo.z;
        }
        if self.local_aabb_max.z < hi.z {
            self.local_aabb_max.z = hi.z;
        }
        self.tree.insert(lo, hi, self.children.len() as i32);
        let (shape_type, margin) = (shape.type_id(), shape.margin());
        self.children.push(CompoundChild { transform, shape, shape_type, margin });
    }

    pub fn aabb(&self, t: &Transform) -> (Vec3, Vec3) {
        let (lo, hi) = (self.local_aabb_min, self.local_aabb_max);
        let (mut cx, mut hx) = ((lo.x + hi.x) * 0.5, (hi.x - lo.x) * 0.5);
        let (mut cy, mut hy) = ((lo.y + hi.y) * 0.5, (hi.y - lo.y) * 0.5);
        let (mut cz, mut hz) = ((lo.z + hi.z) * 0.5, (hi.z - lo.z) * 0.5);
        if self.children.is_empty() {
            (cx, cy, cz, hx, hy, hz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let m = self.margin;
        let (ez, ey, ex) = (hz + m, hy + m, m + hx);
        let r = &t.basis.r;
        let o = t.origin;
        let wx = r[0].x * cx + r[0].y * cy + r[0].z * cz + o.x;
        let wy = cy * r[1].y + r[1].x * cx + cz * r[1].z + o.y;
        let wz = cy * r[2].y + r[2].x * cx + cz * r[2].z + o.z;
        let dx = r[0].y.abs() * ey + r[0].x.abs() * ex + r[0].z.abs() * ez;
        let dy = r[1].y.abs() * ey + r[1].x.abs() * ex + r[1].z.abs() * ez;
        let dz = r[2].y.abs() * ey + r[2].x.abs() * ex + r[2].z.abs() * ez;
        (v3(wx - dx, wy - dy, wz - dz), v3(dx + wx, dy + wy, dz + wz))
    }

    pub fn local_inertia(&self, mass: f32) -> Vec3 {
        let (lo, hi) = self.aabb(&Transform::IDENTITY);
        let k = mass / 12.0;
        let lz = (hi.z - lo.z) * 0.5;
        let lx = (hi.x - lo.x) * 0.5;
        let ly = (hi.y - lo.y) * 0.5;
        let (lz, lx, ly) = (lz + lz, lx + lx, ly + ly);
        let (z2, x2, y2) = (lz * lz, lx * lx, ly * ly);
        v3((z2 + y2) * k, (x2 + z2) * k, (x2 + y2) * k)
    }
}

#[inline(always)]
pub(crate) fn child_world_gate(o: &Transform, c: &Transform) -> Transform {
    let (r, m) = (&o.basis.r, &c.basis);
    let (c0, c1, c2) = (m.col(0), m.col(1), m.col(2));
    Transform {
        basis: Mat3 {
            r: [
                v3(r[0].dot_xy(c0), r[0].dot_xy(c1), r[0].dot_xy(c2)),
                v3(r[1].dot_xy(c0), r[1].dot_xy(c1), r[1].dot_xy(c2)),
                v3(r[2].dot_xy(c0), r[2].dot_xy(c1), r[2].dot_xy(c2)),
            ],
        },
        origin: v3(r[0].dot_xy(c.origin) + o.origin.x, r[1].dot_xy(c.origin) + o.origin.y, r[2].dot_xy(c.origin) + o.origin.z),
    }
}

#[inline(always)]
fn child_world_removal(o: &Transform, c: &Transform) -> Transform {
    let mut t = child_world_gate(o, c);
    let r = &o.basis.r;
    let c2 = c.basis.col(2);
    t.basis.r[1].z = c2.y * r[1].y + c2.x * r[1].x + c2.z * r[1].z;
    t
}

#[derive(Clone, Debug)]
pub struct CompoundAlgo {
    pub children: Vec<Option<Box<Algorithm>>>,
    pub swapped: bool,
    pub shared: Option<ManifoldId>,
}

impl CompoundAlgo {

    pub fn new(compound: &Wrap, swapped: bool, shared: Option<ManifoldId>) -> CompoundAlgo {
        let n = match compound.shape {
            Shape::Compound(c) => c.children.len(),
            _ => unreachable!("a compound algorithm on another shape"),
        };
        let mut children = Vec::with_capacity(n);
        children.resize_with(n, || None);
        CompoundAlgo { children, swapped, shared }
    }

    fn refresh(alg: &Algorithm, ctx: &mut Ctx, result: &ResultOut) {
        visit_contact_manifolds(alg, &mut |m| {
            super::collide_narrow::refresh_points_of(m, ctx, result);
        });
    }

    pub fn process(&mut self, w0: &Wrap, w1: &Wrap, ctx: &mut Ctx, result: &mut ResultOut) {
        let (comp_wrap, other) = if self.swapped { (w1, w0) } else { (w0, w1) };
        let comp = match comp_wrap.shape {
            Shape::Compound(c) => c,
            _ => unreachable!(),
        };
        for c in self.children.iter().flatten() {
            Self::refresh(c, ctx, result);
        }

        let inv = transform_inverse(comp_wrap.transform);
        let ot = other.transform;
        let i = &inv.basis.r;
        let or = &ot.basis.r;
        let in_comp = Transform {
            basis: Mat3 {
                r: [
                    v3(
                        i[0].x * or[0].x + i[0].y * or[1].x + i[0].z * or[2].x,
                        i[0].x * or[0].y + i[0].y * or[1].y + i[0].z * or[2].y,
                        i[0].y * or[1].z + i[0].x * or[0].z + i[0].z * or[2].z,
                    ),
                    v3(
                        or[0].x * i[1].x + or[1].x * i[1].y + or[2].x * i[1].z,
                        or[0].y * i[1].x + or[1].y * i[1].y + or[2].y * i[1].z,
                        or[0].z * i[1].x + or[1].z * i[1].y + or[2].z * i[1].z,
                    ),
                    v3(
                        or[0].x * i[2].x + or[1].x * i[2].y + or[2].x * i[2].z,
                        or[0].y * i[2].x + or[1].y * i[2].y + or[2].y * i[2].z,
                        or[0].z * i[2].x + or[1].z * i[2].y + or[2].z * i[2].z,
                    ),
                ],
            },
            origin: v3(
                i[0].x * ot.origin.x + i[0].y * ot.origin.y + i[0].z * ot.origin.z + inv.origin.x,
                ot.origin.x * i[1].x + ot.origin.y * i[1].y + ot.origin.z * i[1].z + inv.origin.y,
                ot.origin.x * i[2].x + ot.origin.y * i[2].y + ot.origin.z * i[2].z + inv.origin.z,
            ),
        };
        let (lo, hi) = other.shape.aabb(&in_comp);

        comp.tree.collide_tv(lo, hi, |index| {
            self.gate(comp, comp_wrap, other, index, ctx, result);
        });

        for (n, slot) in self.children.iter_mut().enumerate() {
            if slot.is_none() {
                continue;
            }
            let child = &comp.children[n];
            let t = child_world_removal(comp_wrap.transform, &child.transform);
            let (lo0, hi0) = child.shape.aabb(&t);
            let (lo1, hi1) = other.shape.aabb(other.transform);
            let mut overlap = true;
            if hi1.x < lo0.x || hi0.x < lo1.x {
                overlap = false;
            }
            if hi1.z < lo0.z || hi0.z < lo1.z {
                overlap = false;
            }
            if hi1.y < lo0.y || hi0.y < lo1.y || !overlap {
                let a = slot.take().unwrap();
                ctx.disp.free_algorithm(*a);
            }
        }
    }

    fn gate(&mut self, comp: &CompoundShape, comp_wrap: &Wrap, other: &Wrap, index: i32, ctx: &mut Ctx, result: &mut ResultOut) {
        let child = &comp.children[index as usize];
        let t = child_world_gate(comp_wrap.transform, &child.transform);
        let (lo0, hi0) = child.shape.aabb(&t);
        let (lo1, hi1) = other.shape.aabb(other.transform);
        let mut overlap = true;
        if hi1.x < lo0.x || hi0.x < lo1.x {
            overlap = false;
        }
        if hi1.z < lo0.z || hi0.z < lo1.z {
            overlap = false;
        }
        if !(lo0.y <= hi1.y && lo1.y <= hi0.y && overlap) {
            return;
        }
        let child_wrap = Wrap { shape: &child.shape, body: comp_wrap.body, id: comp_wrap.id, transform: &t, part_id: -1, index, info_map: None };
        if self.children[index as usize].is_none() {
            let a = ctx.disp.find_algorithm(ctx.bodies, ctx.shapes, &child_wrap, other, self.shared);
            self.children[index as usize] = Some(Box::new(a));
        }
        let mut inner = *result;
        if result.wrap0.id == comp_wrap.id {
            inner.wrap0 = child_wrap;
            inner.part0 = -1;
            inner.index0 = index;
        } else {
            inner.wrap1 = child_wrap;
            inner.part1 = -1;
            inner.index1 = index;
        }
        process_algorithm(self.children[index as usize].as_mut().unwrap(), &child_wrap, other, ctx, &mut inner);
        result.take_ids(&inner);
    }
}

#[derive(Clone, Debug)]
pub struct SimplePair {
    pub index0: i32,
    pub index1: i32,
    pub algorithm: Box<Algorithm>,
}

#[derive(Clone, Debug)]
pub struct CompoundCompoundAlgo {
    pub pairs: Vec<SimplePair>,
    pub shared: Option<ManifoldId>,
}

impl CompoundCompoundAlgo {

    pub fn new(shared: Option<ManifoldId>) -> CompoundCompoundAlgo {
        CompoundCompoundAlgo { pairs: Vec::new(), shared }
    }

    pub fn process(&mut self, w0: &Wrap, w1: &Wrap, ctx: &mut Ctx, result: &mut ResultOut) {
        let (c0, c1) = match (w0.shape, w1.shape) {
            (Shape::Compound(a), Shape::Compound(b)) => (a, b),
            _ => unreachable!(),
        };
        for p in &self.pairs {
            CompoundAlgo::refresh(&p.algorithm, ctx, result);
        }

        let inv = transform_inverse(w0.transform);
        let i = &inv.basis.r;
        let m = &w1.transform.basis.r;
        let o = w1.transform.origin;
        let xform = Transform {
            basis: Mat3 {
                r: [
                    v3(
                        m[0].x * i[0].x + m[1].x * i[0].y + m[2].x * i[0].z,
                        m[0].y * i[0].x + m[1].y * i[0].y + m[2].y * i[0].z,
                        m[0].z * i[0].x + m[1].z * i[0].y + m[2].z * i[0].z,
                    ),
                    v3(
                        m[0].x * i[1].x + m[1].x * i[1].y + m[2].x * i[1].z,
                        m[0].y * i[1].x + m[1].y * i[1].y + m[2].y * i[1].z,
                        m[0].z * i[1].x + m[1].z * i[1].y + m[2].z * i[1].z,
                    ),
                    v3(
                        m[0].x * i[2].x + m[1].x * i[2].y + m[2].x * i[2].z,
                        m[0].y * i[2].x + m[1].y * i[2].y + m[2].y * i[2].z,
                        m[1].z * i[2].y + m[0].z * i[2].x + m[2].z * i[2].z,
                    ),
                ],
            },
            origin: v3(
                i[0].x * o.x + i[0].y * o.y + i[0].z * o.z + inv.origin.x,
                o.x * i[1].x + o.y * i[1].y + i[1].z * o.z + inv.origin.y,
                o.x * i[2].x + o.y * i[2].y + o.z * i[2].z + inv.origin.z,
            ),
        };
        let mut hits: Vec<(i32, i32)> = Vec::new();
        collide_tt(&c0.tree, &c1.tree, &xform, &mut hits);
        for (a, b) in hits {
            self.leaf(c0, c1, w0, w1, a, b, ctx, result);
        }

        let mut remove: Vec<(i32, i32)> = Vec::new();
        let mut i = 0;
        while i < self.pairs.len() {
            let (a, b) = (self.pairs[i].index0, self.pairs[i].index1);
            let (ch0, ch1) = (&c0.children[a as usize], &c1.children[b as usize]);
            let t0 = removal_transform0(w0.transform, &ch0.transform);
            let (lo0, hi0) = ch0.shape.aabb(&t0);
            let t1 = removal_transform1(w1.transform, &ch1.transform);
            let (lo1, hi1) = ch1.shape.aabb(&t1);
            let mut overlap = true;
            if hi1.x < lo0.x || hi0.x < lo1.x {
                overlap = false;
            }
            if hi1.z < lo0.z || hi0.z < lo1.z {
                overlap = false;
            }
            if hi1.y < lo0.y || hi0.y < lo1.y || !overlap {
                let alg = core::mem::replace(&mut self.pairs[i].algorithm, Box::new(Algorithm::Empty));
                ctx.disp.free_algorithm(*alg);
                remove.push((a, b));
            }
            i += 1;
        }
        for (a, b) in remove {
            if let Some(at) = self.pairs.iter().position(|p| p.index0 == a && p.index1 == b) {
                self.pairs.swap_remove(at);
            }
        }
    }

    fn leaf(&mut self, c0: &CompoundShape, c1: &CompoundShape, w0: &Wrap, w1: &Wrap, a: i32, b: i32, ctx: &mut Ctx, result: &mut ResultOut) {
        let (ch0, ch1) = (&c0.children[a as usize], &c1.children[b as usize]);
        let t0 = child_world_gate(w0.transform, &ch0.transform);
        let t1 = leaf_transform1(w1.transform, &ch1.transform);
        let (lo0, hi0) = ch0.shape.aabb(&t0);
        let (lo1, hi1) = ch1.shape.aabb(&t1);
        let mut overlap = true;
        if hi1.x < lo0.x || hi0.x < lo1.x {
            overlap = false;
        }
        if hi1.z < lo0.z || hi0.z < lo1.z {
            overlap = false;
        }
        if !(lo0.y <= hi1.y && lo1.y <= hi0.y && overlap) {
            return;
        }
        let wrap0 = Wrap { shape: &ch0.shape, body: w0.body, id: w0.id, transform: &t0, part_id: -1, index: a, info_map: None };
        let wrap1 = Wrap { shape: &ch1.shape, body: w1.body, id: w1.id, transform: &t1, part_id: -1, index: b, info_map: None };
        let at = match self.pairs.iter().position(|p| p.index0 == a && p.index1 == b) {
            Some(at) => at,
            None => {
                let alg = ctx.disp.find_algorithm(ctx.bodies, ctx.shapes, &wrap0, &wrap1, self.shared);
                self.pairs.push(SimplePair { index0: a, index1: b, algorithm: Box::new(alg) });
                self.pairs.len() - 1
            }
        };
        let mut inner = *result;
        inner.wrap0 = wrap0;
        inner.wrap1 = wrap1;
        inner.part0 = -1;
        inner.index0 = a;
        inner.part1 = -1;
        inner.index1 = b;
        process_algorithm(&mut self.pairs[at].algorithm, &wrap0, &wrap1, ctx, &mut inner);
        result.take_ids(&inner);
    }
}

#[inline(always)]
fn leaf_transform1(o: &Transform, c: &Transform) -> Transform {
    let (r, m) = (&o.basis.r, &c.basis);
    let (c0, c1, c2) = (m.col(0), m.col(1), m.col(2));
    Transform {
        basis: Mat3 {
            r: [
                v3(r[0].dot_xy(c0), r[0].y * c1.y + r[0].x * c1.x + r[0].z * c1.z, r[0].y * c2.y + r[0].x * c2.x + r[0].z * c2.z),
                v3(r[1].dot_xy(c0), r[1].y * c1.y + r[1].x * c1.x + r[1].z * c1.z, r[1].dot_xy(c2)),
                v3(r[2].dot_xy(c0), r[2].y * c1.y + r[2].x * c1.x + r[2].z * c1.z, r[2].dot_xy(c2)),
            ],
        },
        origin: v3(r[0].dot_xy(c.origin) + o.origin.x, r[1].dot_xy(c.origin) + o.origin.y, r[2].dot_xy(c.origin) + o.origin.z),
    }
}

#[inline(always)]
fn removal_transform0(o: &Transform, c: &Transform) -> Transform {
    let (r, m) = (&o.basis.r, &c.basis);
    let (c0, c1, c2) = (m.col(0), m.col(1), m.col(2));
    Transform {
        basis: Mat3 {
            r: [
                v3(r[0].dot_xy(c0), r[0].dot_xy(c1), r[0].dot_xy(c2)),
                v3(r[1].dot_xy(c0), r[1].y * c1.y + r[1].x * c1.x + r[1].z * c1.z, r[1].y * c2.y + r[1].x * c2.x + r[1].z * c2.z),
                v3(r[2].dot_xy(c0), r[2].dot_xy(c1), r[2].dot_xy(c2)),
            ],
        },
        origin: v3(
            r[0].dot_xy(c.origin) + o.origin.x,
            r[1].y * c.origin.y + r[1].x * c.origin.x + r[1].z * c.origin.z + o.origin.y,
            r[2].dot_xy(c.origin) + o.origin.z,
        ),
    }
}

#[inline(always)]
fn removal_transform1(o: &Transform, c: &Transform) -> Transform {
    let (r, m) = (&o.basis.r, &c.basis);
    let (c0, c1, c2) = (m.col(0), m.col(1), m.col(2));
    Transform {
        basis: Mat3 {
            r: [
                v3(r[0].dot_xy(c0), r[0].y * c1.y + r[0].x * c1.x + r[0].z * c1.z, r[0].y * c2.y + r[0].x * c2.x + r[0].z * c2.z),
                v3(r[1].dot_xy(c0), r[1].y * c1.y + r[1].x * c1.x + r[1].z * c1.z, r[1].y * c2.y + r[1].x * c2.x + r[1].z * c2.z),
                v3(r[2].dot_xy(c0), r[2].y * c1.y + r[2].x * c1.x + r[2].z * c1.z, r[2].y * c2.y + r[2].x * c2.x + r[2].z * c2.z),
            ],
        },
        origin: v3(r[0].dot_xy(c.origin) + o.origin.x, r[1].dot_xy(c.origin) + o.origin.y, r[2].dot_xy(c.origin) + o.origin.z),
    }
}

fn collide_tt(a: &Dbvt, b: &Dbvt, x: &Transform, out: &mut Vec<(i32, i32)>) {
    if a.root < 0 || b.root < 0 {
        return;
    }
    let r = &x.basis.r;
    let mut stack: Vec<(i32, i32)> = vec![(a.root, b.root)];
    while let Some((ia, ib)) = stack.pop() {
        let (na, nb) = (&a.nodes[ia as usize], &b.nodes[ib as usize]);
        let ex = (nb.max.x - nb.min.x) * 0.5 + 0.0;
        let ey = (nb.max.y - nb.min.y) * 0.5 + 0.0;
        let ez = (nb.max.z - nb.min.z) * 0.5 + 0.0;
        let cz = (nb.min.z + nb.max.z) * 0.5;
        let cy = (nb.min.y + nb.max.y) * 0.5;
        let cx = (nb.min.x + nb.max.x) * 0.5;
        let wz = cy * r[2].y + r[2].x * cx + cz * r[2].z + x.origin.z;
        let wy = cy * r[1].y + r[1].x * cx + cz * r[1].z + x.origin.y;
        let wx = cy * r[0].y + r[0].x * cx + cz * r[0].z + x.origin.x;
        let dz = r[2].y.abs() * ey + r[2].x.abs() * ex + r[2].z.abs() * ez;
        let dy = r[1].y.abs() * ey + r[1].x.abs() * ex + r[1].z.abs() * ez;
        let dx = r[0].y.abs() * ey + r[0].x.abs() * ex + r[0].z.abs() * ez;
        let miss = wy + dy < na.min.y || dx + wx < na.min.x || wz + dz < na.min.z || na.max.y < wy - dy || na.max.x < wx - dx || na.max.z < wz - dz;
        if miss {
            continue;
        }
        match (a.is_internal(ia), b.is_internal(ib)) {
            (false, false) => out.push((na.data, nb.data)),
            (false, true) => {
                stack.push((ia, nb.child0));
                stack.push((ia, nb.child1));
            }
            (true, false) => {
                stack.push((na.child0, ib));
                stack.push((na.child1, ib));
            }
            (true, true) => {
                stack.push((na.child0, nb.child0));
                stack.push((na.child1, nb.child0));
                stack.push((na.child0, nb.child1));
                stack.push((na.child1, nb.child1));
            }
        }
    }
}
