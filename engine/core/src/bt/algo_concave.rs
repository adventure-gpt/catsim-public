use super::dispatch::{Ctx, ResultOut, process_algorithm};
use super::internal_edge::TriangleInfoMap;
use super::manifold::Wrap;
use super::shapes::{Shape, TriangleShape};
use super::types::*;
use crate::math::{Mat3, Transform, Vec3, v3};

#[derive(Clone, Debug)]
pub struct ConvexConcaveAlgo {
    pub swapped: bool,
    pub manifold: ManifoldId,
}

impl ConvexConcaveAlgo {

    pub fn new(manifold: ManifoldId, swapped: bool) -> ConvexConcaveAlgo {
        ConvexConcaveAlgo { swapped, manifold }
    }

    pub fn process(&mut self, w0: &Wrap, w1: &Wrap, ctx: &mut Ctx, result: &mut ResultOut) {
        let (convex, tri) = if self.swapped { (w1, w0) } else { (w0, w1) };
        let mesh = match tri.shape {
            Shape::Mesh(m) => m,
            _ => unreachable!("a concave algorithm on another shape"),
        };
        if convex.shape.type_id() >= 0x14 {
            return;
        }
        let margin = mesh.margin;
        result.manifold = Some(self.manifold);
        #[cfg(feature = "profile")]
        let is_box = matches!(convex.shape, Shape::Box(_));
        #[cfg(feature = "profile")]
        let _call = if is_box { Some(crate::prof::span(32)) } else { None };

        let inv = super::algo_convex::transform_inverse(tri.transform);
        let i = &inv.basis.r;
        let m = &convex.transform.basis.r;
        let o = convex.transform.origin;
        let in_mesh = Transform {
            basis: Mat3 {
                r: [
                    v3(i[0].dot_xy(m_col(m, 0)), i[0].dot_xy(m_col(m, 1)), i[0].dot_xy(m_col(m, 2))),
                    v3(i[1].dot_xy(m_col(m, 0)), i[1].dot_xy(m_col(m, 1)), i[1].dot_xy(m_col(m, 2))),
                    v3(i[2].dot_xy(m_col(m, 0)), i[2].dot_xy(m_col(m, 1)), i[2].dot_xy(m_col(m, 2))),
                ],
            },
            origin: v3(
                i[0].x * o.x + o.y * i[0].y + o.z * i[0].z + inv.origin.x,
                o.x * i[1].x + o.y * i[1].y + o.z * i[1].z + inv.origin.y,
                o.x * i[2].x + o.y * i[2].y + o.z * i[2].z + inv.origin.z,
            ),
        };
        let (lo, hi) = convex.shape.aabb(&in_mesh);
        let hi = v3(margin + hi.x, margin + hi.y, margin + hi.z);
        let lo = v3(lo.x - margin, lo.y - margin, lo.z - margin);

        {
            let man = &mut ctx.disp.slab[self.manifold.0 as usize];
            man.body0 = convex.id;
            man.body1 = tri.id;
        }

        #[cfg(feature = "profile")]
        let _walk = if is_box { Some(crate::prof::span(33)) } else { None };
        #[cfg(feature = "profile")]
        if is_box {
            let (qlo, qhi) = mesh.bvh.quantize_query(lo, hi);
            if mesh.bvh.grid.empty(qlo, qhi) { crate::prof::count(36); }
        }
        mesh.bvh.report_aabb_overlapping_nodes(lo, hi, |part, index| {
            #[cfg(feature = "profile")]
            if is_box { crate::prof::count(34); }
            self.triangle(mesh.triangle(index), part, index, margin, lo, hi, convex, tri, &in_mesh, &mesh.info_map, ctx, result);
        });
        #[cfg(feature = "profile")]
        drop(_walk);

        if ctx.disp.slab[self.manifold.0 as usize].cached != 0 {
            super::collide_narrow::refresh_points_of(self.manifold, ctx, result);
        }
    }

    fn triangle(&self, v: [Vec3; 3], part: i32, index: i32, margin: f32, lo: Vec3, hi: Vec3, convex: &Wrap, tri: &Wrap, in_mesh: &Transform, info_map: &TriangleInfoMap, ctx: &mut Ctx, result: &mut ResultOut) {
        let min3 = |a: f32, b: f32, c: f32| {
            let m = if a < b { a } else { b };
            if c < m { c } else { m }
        };
        let max3 = |a: f32, b: f32, c: f32| {
            let m = if b < a { a } else { b };
            if m < c { c } else { m }
        };
        if !(min3(v[0].x, v[1].x, v[2].x) <= hi.x
            && lo.x <= max3(v[0].x, v[1].x, v[2].x)
            && min3(v[0].z, v[1].z, v[2].z) <= hi.z
            && lo.z <= max3(v[0].z, v[1].z, v[2].z)
            && min3(v[0].y, v[1].y, v[2].y) <= hi.y
            && lo.y <= max3(v[0].y, v[1].y, v[2].y))
        {
            return;
        }
        #[cfg(feature = "profile")]
        let (_pass, before) = (if matches!(convex.shape, Shape::Box(_)) { Some(crate::prof::span(35)) } else { None }, ctx.disp.slab[self.manifold.0 as usize].cached);

        if let Shape::Box(bx) = convex.shape {
            let reach = ctx.disp.slab[self.manifold.0 as usize].breaking_threshold + bx.margin + margin;
            if box_triangle_far(bx, in_mesh, &v, 2.0 * reach + 0.01) {
                crate::prof::count(29);
                if result.wrap0.id == tri.id {
                    result.part0 = part;
                    result.index0 = index;
                } else {
                    result.part1 = part;
                    result.index1 = index;
                }
                result.manifold = Some(self.manifold);
                return;
            }
        }
        let shape = Shape::Triangle(TriangleShape { v, margin });
        let tri_wrap = Wrap { shape: &shape, body: tri.body, id: tri.id, transform: tri.transform, part_id: part, index, info_map: Some(info_map) };
        let mut alg = ctx.disp.find_algorithm(ctx.bodies, ctx.shapes, convex, &tri_wrap, Some(self.manifold));
        let mut inner = *result;
        if result.wrap0.id == tri.id {
            inner.wrap0 = tri_wrap;
            inner.part0 = part;
            inner.index0 = index;
        } else {
            inner.wrap1 = tri_wrap;
            inner.part1 = part;
            inner.index1 = index;
        }
        process_algorithm(&mut alg, convex, &tri_wrap, ctx, &mut inner);
        result.take_ids(&inner);
        ctx.disp.free_algorithm(alg);
        #[cfg(feature = "profile")]
        if _pass.is_some() && ctx.disp.slab[self.manifold.0 as usize].cached != before { crate::prof::count(37); }
    }
}

#[inline(always)]
fn m_col(m: &[Vec3; 3], c: usize) -> Vec3 {
    v3(m[0][c], m[1][c], m[2][c])
}

pub static PLANT_FAR_BUG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[inline]
fn box_triangle_far(b: &super::shapes::BoxShape, t: &Transform, v: &[Vec3; 3], gap: f32) -> bool {
    let r = &t.basis.r;
    let c = t.origin;
    let cols = [v3(r[0].x, r[1].x, r[2].x), v3(r[0].y, r[1].y, r[2].y), v3(r[0].z, r[1].z, r[2].z)];
    let h = [b.half.x, b.half.y, b.half.z];
    let d = |a: Vec3, u: Vec3| a.x * u.x + a.y * u.y + a.z * u.z;
    let gap = if PLANT_FAR_BUG.load(std::sync::atomic::Ordering::Relaxed) { -0.3 } else { gap };
    let far = |u: Vec3| {
        let l2 = d(u, u);
        if !(l2 > 1e-20 && l2 < 1e20) {
            return false;
        }
        let pc = d(c, u);
        let rad = h[0] * d(cols[0], u).abs() + h[1] * d(cols[1], u).abs() + h[2] * d(cols[2], u).abs();
        let (p0, p1, p2) = (d(v[0], u), d(v[1], u), d(v[2], u));
        let (tmin, tmax) = (p0.min(p1).min(p2), p0.max(p1).max(p2));
        let g = (tmin - (pc + rad)).max((pc - rad) - tmax);
        g.is_finite() && rad.is_finite() && g > gap * l2.sqrt()
    };
    let (e1, e2) = (v3(v[1].x - v[0].x, v[1].y - v[0].y, v[1].z - v[0].z), v3(v[2].x - v[0].x, v[2].y - v[0].y, v[2].z - v[0].z));
    let n = v3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
    far(n) || far(cols[0]) || far(cols[1]) || far(cols[2])
}
