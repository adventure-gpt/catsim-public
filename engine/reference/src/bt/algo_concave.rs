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

        let mut hits: Vec<(i32, i32)> = Vec::new();
        mesh.bvh.report_aabb_overlapping_nodes(lo, hi, |part, index| hits.push((part, index)));
        for (part, index) in hits {
            self.triangle(mesh.triangle(index), part, index, margin, lo, hi, convex, tri, &mesh.info_map, ctx, result);
        }

        if ctx.disp.slab[self.manifold.0 as usize].cached != 0 {
            super::collide_narrow::refresh_points_of(self.manifold, ctx, result);
        }
    }

    fn triangle(&self, v: [Vec3; 3], part: i32, index: i32, margin: f32, lo: Vec3, hi: Vec3, convex: &Wrap, tri: &Wrap, info_map: &TriangleInfoMap, ctx: &mut Ctx, result: &mut ResultOut) {
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
    }
}

#[inline(always)]
fn m_col(m: &[Vec3; 3], c: usize) -> Vec3 {
    v3(m[0][c], m[1][c], m[2][c])
}
