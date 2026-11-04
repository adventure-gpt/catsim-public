use super::algo_convex;
use super::dispatch::{ActorHook, Ctx, ResultOut};
use super::manifold::{ManifoldResult, Wrap};
use super::types::*;

#[inline]
fn with_result(m: ManifoldId, ctx: &mut Ctx, r: &mut ResultOut, f: impl FnOnce(&mut ManifoldResult)) {
    let hook: Option<&mut ActorHook> = match ctx.hook {
        Some(ref mut h) => Some(&mut **h),
        None => None,
    };
    let mut res = ManifoldResult { manifolds: &mut ctx.disp.slab, current: m.0 as usize, wrap0: r.wrap0, wrap1: r.wrap1, part0: r.part0, part1: r.part1, index0: r.index0, index1: r.index1, actor_hook: hook };
    f(&mut res);
    r.part0 = res.part0;
    r.part1 = res.part1;
    r.index0 = res.index0;
    r.index1 = res.index1;
}

pub fn convex_convex(m: ManifoldId, own: bool, w0: &Wrap, w1: &Wrap, ctx: &mut Ctx, r: &mut ResultOut) {
    with_result(m, ctx, r, |res| algo_convex::process_convex_convex(w0, w1, m, own, res));
}

pub fn sphere_sphere(m: ManifoldId, _own: bool, w0: &Wrap, w1: &Wrap, ctx: &mut Ctx, r: &mut ResultOut) {
    with_result(m, ctx, r, |res| algo_convex::process_sphere_sphere(w0, w1, m, res));
}

pub fn sphere_triangle(m: ManifoldId, own: bool, swapped: bool, w0: &Wrap, w1: &Wrap, ctx: &mut Ctx, r: &mut ResultOut) {
    with_result(m, ctx, r, |res| algo_convex::process_sphere_triangle(w0, w1, m, own, swapped, res));
}

pub fn box_box(m: ManifoldId, own: bool, w0: &Wrap, w1: &Wrap, ctx: &mut Ctx, r: &mut ResultOut) {
    with_result(m, ctx, r, |res| algo_convex::process_box_box(w0, w1, m, own, res));
}

pub fn convex_plane(m: ManifoldId, own: bool, swapped: bool, w0: &Wrap, w1: &Wrap, ctx: &mut Ctx, r: &mut ResultOut) {
    with_result(m, ctx, r, |res| algo_convex::process_convex_plane(w0, w1, m, own, swapped, res));
}

pub fn refresh_points_of(id: ManifoldId, ctx: &mut Ctx, r: &ResultOut) {
    let m = &mut ctx.disp.slab[id.0 as usize];
    if m.cached != 0 {
        if m.body0 == r.wrap0.id {
            m.refresh(&r.wrap0.body.transform, &r.wrap1.body.transform);
        } else {
            m.refresh(&r.wrap1.body.transform, &r.wrap0.body.transform);
        }
    }
}
