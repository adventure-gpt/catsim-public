use crate::bt::gjk::{ClosestPointInput, ContactResult, GjkDetector};
use crate::bt::manifold::{ManifoldResult, Wrap};
use crate::bt::shapes::{BoxShape, Shape, SphereShape, TriangleShape};
use crate::bt::simplex::Simplex;
use crate::bt::types::ManifoldId;
use crate::math::crt::atan2f;
use crate::math::{Mat3, Transform, Vec3, v3};

const EPS: f32 = f32::from_bits(0x34000000);
const LARGE: f32 = f32::from_bits(0x5d5e0b6b);

pub fn transform_inverse(t: &Transform) -> Transform {
    let r = &t.basis.r;
    let (ox, oy, oz) = (-t.origin.x, -t.origin.y, -t.origin.z);
    Transform {
        basis: Mat3 { r: [v3(r[0].x, r[1].x, r[2].x), v3(r[0].y, r[1].y, r[2].y), v3(r[0].z, r[1].z, r[2].z)] },
        origin: v3(r[1].x * oy + r[0].x * ox + r[2].x * oz, r[1].y * oy + r[0].y * ox + r[2].y * oz, r[1].z * oy + r[0].z * ox + r[2].z * oz),
    }
}

pub fn process_convex_convex(w0: &Wrap, w1: &Wrap, manifold: ManifoldId, own_manifold: bool, r: &mut ManifoldResult) {
    r.current = manifold.0 as usize;
    let mut simplex = Simplex::new();
    let (ma, mb) = (w0.shape.margin(), w1.shape.margin());
    let mut gjk = GjkDetector::new(w0.shape, w1.shape, true);
    let d = r.manifold().breaking_threshold + ma + mb;
    let input = ClosestPointInput { a: *w0.transform, b: *w1.transform, max_distance_sqr: d * d };
    gjk.closest_points(&mut simplex, &input, r);
    if own_manifold {
        r.refresh();
    }
}

pub fn process_convex_plane(w0: &Wrap, w1: &Wrap, manifold: ManifoldId, own_manifold: bool, swapped: bool, r: &mut ManifoldResult) {
    let (conv, plane) = if swapped { (w1, w0) } else { (w0, w1) };
    let Shape::Plane(pshape) = plane.shape else { unreachable!("second shape of convex-plane") };
    let c_t = conv.transform;
    let p_t = plane.transform;
    let cinv = transform_inverse(c_t);
    let (i0, i1, i2) = (cinv.basis.r[0], cinv.basis.r[1], cinv.basis.r[2]);
    let pm = p_t.basis.r;
    let (p00, p01, p02) = (pm[0].x, pm[0].y, pm[0].z);
    let (p10, p11, p12) = (pm[1].x, pm[1].y, pm[1].z);
    let (p20, p21, p22) = (pm[2].x, pm[2].y, pm[2].z);
    let pinv = transform_inverse(p_t);
    let (pi0, pi1, pi2) = (pinv.basis.r[0], pinv.basis.r[1], pinv.basis.r[2]);
    let cm = c_t.basis.r;
    let (c00, c01, c02) = (cm[0].x, cm[0].y, cm[0].z);
    let (c10, c11, c12) = (cm[1].x, cm[1].y, cm[1].z);
    let (c20, c21, c22) = (cm[2].x, cm[2].y, cm[2].z);
    let co = c_t.origin;
    let ox = pi0.y * co.y + pi0.x * co.x + pi0.z * co.z + pinv.origin.x;
    let oy = co.x * pi1.x + co.y * pi1.y + co.z * pi1.z + pinv.origin.y;
    let oz = co.y * pi2.y + pi2.x * co.x + pi2.z * co.z + pinv.origin.z;
    let dir = v3(
        ((i0.x * p00 + i0.y * p10 + p20 * i0.z) * 0.0 + (i0.x * p01 + i0.y * p11 + i0.z * p21) * 0.0) - (i0.x * p02 + i0.y * p12 + i0.z * p22) * 1.0,
        ((p00 * i1.x + p10 * i1.y + p20 * i1.z) * 0.0 + (p01 * i1.x + p11 * i1.y + p21 * i1.z) * 0.0) - (p02 * i1.x + p12 * i1.y + p22 * i1.z) * 1.0,
        ((p00 * i2.x + p10 * i2.y + p20 * i2.z) * 0.0 + (p01 * i2.x + p11 * i2.y + p21 * i2.z) * 0.0) - (p02 * i2.x + p12 * i2.y + p22 * i2.z) * 1.0,
    );
    let vt = conv.shape.local_support(dir);
    let x = (c00 * pi0.x + c10 * pi0.y + c20 * pi0.z) * vt.x + (c01 * pi0.x + c11 * pi0.y + c21 * pi0.z) * vt.y + (c02 * pi0.x + c12 * pi0.y + c22 * pi0.z) * vt.z + ox;
    let y = (c00 * pi1.x + c10 * pi1.y + c20 * pi1.z) * vt.x + (c11 * pi1.y + c01 * pi1.x + c21 * pi1.z) * vt.y + (c02 * pi1.x + c12 * pi1.y + c22 * pi1.z) * vt.z + oy;
    let z = (c00 * pi2.x + c10 * pi2.y + c20 * pi2.z) * vt.x + (c01 * pi2.x + c11 * pi2.y + c21 * pi2.z) * vt.y + (c02 * pi2.x + c12 * pi2.y + c22 * pi2.z) * vt.z + oz;
    r.current = manifold.0 as usize;
    let thr = r.manifold().breaking_threshold;
    let lim_x = thr + pshape.half_x;
    let lim_y = thr + pshape.half_y;
    if !(thr <= z || x < -lim_x || y < -lim_y || lim_x < x || lim_y < y) {
        let normal = v3(p01 * 0.0 + p00 * 0.0 + p02, p11 * 0.0 + p10 * 0.0 + p12, p21 * 0.0 + p20 * 0.0 + p22);
        let po = p_t.origin;
        let point = v3(y * p01 + x * p00 + p02 * 0.0 + po.x, y * p11 + x * p10 + p12 * 0.0 + po.y, y * p21 + x * p20 + p22 * 0.0 + po.z);
        r.add_contact_point(normal, point, z);
    }
    if own_manifold {
        r.refresh();
    }
}

pub fn process_sphere_sphere(w0: &Wrap, w1: &Wrap, manifold: ManifoldId, r: &mut ManifoldResult) {
    r.current = manifold.0 as usize;
    let (Shape::Sphere(s0), Shape::Sphere(s1)) = (w0.shape, w1.shape) else { unreachable!("sphere-sphere") };
    let (o0, o1) = (w0.transform.origin, w1.transform.origin);
    let dx = o0.x - o1.x;
    let dy = o0.y - o1.y;
    let dz = o0.z - o1.z;
    let len = (dy * dy + dx * dx + dz * dz).sqrt();
    let r1 = s1.radius * s1.scale.x;
    let sum = s0.radius * s0.scale.x + r1;
    if len <= sum {
        let (mut nx, mut ny, mut nz) = (1.0f32, 0.0f32, 0.0f32);
        if EPS < len {
            let k = 1.0 / len;
            nx = k * dx;
            ny = k * dy;
            nz = k * dz;
        }
        let point = v3(nx * r1 + o1.x, ny * r1 + o1.y, nz * r1 + o1.z);
        r.add_contact_point(v3(nx, ny, nz), point, len - sum);
    }
    r.refresh();
}

pub fn sphere_triangle_collide(sphere: &SphereShape, tri: &TriangleShape, c: Vec3, threshold: f32) -> Option<(Vec3, Vec3, f32)> {
    let v = &tri.v;
    let e1z = v[1].z - v[0].z;
    let e2y = v[2].y - v[0].y;
    let e1y = v[1].y - v[0].y;
    let e1x = v[1].x - v[0].x;
    let radius = sphere.radius * sphere.scale.x;
    let rt = radius + threshold;
    let nx0 = (v[2].z - v[0].z) * e1y - e2y * e1z;
    let ny0 = (v[2].x - v[0].x) * e1z - (v[2].z - v[0].z) * e1x;
    let nz0 = e2y * e1x - (v[2].x - v[0].x) * e1y;
    let k = 1.0 / (ny0 * ny0 + nx0 * nx0 + nz0 * nz0).sqrt();
    let (ny, nx, nz) = (k * ny0, k * nx0, k * nz0);
    let dist = (c.x - v[0].x) * nx + (c.y - v[0].y) * ny + (c.z - v[0].z) * nz;
    if !(0.0 <= dist) {
        return None;
    }
    if rt <= dist {
        return None;
    }
    let e20x = v[0].x - v[2].x;
    let e12y = v[2].y - v[1].y;
    let e20y = v[0].y - v[2].y;
    let e12x = v[2].x - v[1].x;
    let e20z = v[0].z - v[2].z;
    let r1 = (nz * e1y - ny * e1z) * (c.x - v[0].x) + (nx * e1z - nz * e1x) * (c.y - v[0].y) + (ny * e1x - nx * e1y) * (c.z - v[0].z);
    let e12z = v[2].z - v[1].z;
    let r2 = (e12z * nx - e12x * nz) * (c.y - v[1].y) + (e12y * nz - e12z * ny) * (c.x - v[1].x) + (e12x * ny - e12y * nx) * (c.z - v[1].z);
    let r3 = (e20z * nx - e20x * nz) * (c.y - v[2].y) + (e20y * nz - e20z * ny) * (c.x - v[2].x) + (e20x * ny - e20y * nx) * (c.z - v[2].z);
    let (mut px, mut py, mut pz);
    if (0.0 < r1 && 0.0 < r2 && 0.0 < r3) || (r1 <= 0.0 && r2 <= 0.0 && r3 <= 0.0) {
        px = c.x - nx * dist;
        py = c.y - ny * dist;
        pz = c.z - nz * dist;
    } else {
        let mut best = rt * rt;
        let mut has = false;
        px = 0.0;
        py = 0.0;
        pz = 0.0;
        for i in 0..3 {
            let (a, b) = (v[i], v[(i + 1) % 3]);
            let (bx, by, bz) = (b.x - a.x, b.y - a.y, b.z - a.z);
            let (mut dx, mut dy, mut dz) = (c.x - a.x, c.y - a.y, c.z - a.z);
            let mut t = by * dy + bx * dx + bz * dz;
            if t <= 0.0 {
                t = 0.0;
            } else {
                let vv = by * by + bx * bx + bz * bz;
                if vv <= t {
                    t = 1.0;
                    dx -= bx;
                    dy -= by;
                    dz -= bz;
                } else {
                    t /= vv;
                    dx -= t * bx;
                    dy -= t * by;
                    dz -= t * bz;
                }
            }
            let qx = bx * t + a.x;
            let d2 = dy * dy + dx * dx + dz * dz;
            let qy = by * t + a.y;
            let qz = bz * t + a.z;
            if d2 < best {
                has = true;
                px = qx;
                py = qy;
                pz = qz;
                best = d2;
            }
        }
        if !has {
            return None;
        }
    }
    let (tx, ty, tz) = (c.x - px, c.y - py, c.z - pz);
    let d2 = ty * ty + tx * tx + tz * tz;
    if rt * rt <= d2 {
        return None;
    }
    let point = v3(px, py, pz);
    if d2 <= EPS {
        Some((point, v3(nx, ny, nz), -radius))
    } else {
        let l = 1.0 / (tx * tx + ty * ty + tz * tz).sqrt();
        Some((point, v3(l * tx, ty * l, tz * l), -(radius - d2.sqrt())))
    }
}

pub fn sphere_triangle_detect(sphere: &SphereShape, tri: &TriangleShape, ta: &Transform, tb: &Transform, threshold: f32, swap: bool, out: &mut impl ContactResult) {
    let b = &tb.basis.r;
    let (dx, dy, dz) = (ta.origin.x - tb.origin.x, ta.origin.y - tb.origin.y, ta.origin.z - tb.origin.z);
    let c = v3(dx * b[0].x + dy * b[1].x + dz * b[2].x, dx * b[0].y + dy * b[1].y + dz * b[2].y, dx * b[0].z + dy * b[1].z + dz * b[2].z);
    let Some((p, n, depth)) = sphere_triangle_collide(sphere, tri, c, threshold) else { return };
    let o = tb.origin;
    let wp = v3(p.y * b[0].y + p.x * b[0].x + p.z * b[0].z + o.x, p.y * b[1].y + p.x * b[1].x + p.z * b[1].z + o.y, p.x * b[2].x + p.y * b[2].y + p.z * b[2].z + o.z);
    let wnx = n.y * b[0].y + n.x * b[0].x + n.z * b[0].z;
    let wny = n.y * b[1].y + n.x * b[1].x + n.z * b[1].z;
    let wnz = n.x * b[2].x + n.y * b[2].y + n.z * b[2].z;
    if !swap {
        out.add_contact_point(v3(wnx, wny, wnz), wp, depth);
    } else {
        let point = v3(wp.x + wnx * depth, wp.y + wny * depth, wp.z + wnz * depth);
        out.add_contact_point(v3(-wnx, -wny, -wnz), point, depth);
    }
}

pub fn process_sphere_triangle(w0: &Wrap, w1: &Wrap, manifold: ManifoldId, own_manifold: bool, swapped: bool, r: &mut ManifoldResult) {
    r.current = manifold.0 as usize;
    let (sw, tw) = if swapped { (w1, w0) } else { (w0, w1) };
    let (Shape::Sphere(s), Shape::Triangle(t)) = (sw.shape, tw.shape) else { unreachable!("sphere-triangle") };
    let thr = r.manifold().breaking_threshold;
    sphere_triangle_detect(s, t, sw.transform, tw.transform, thr, swapped, r);
    if own_manifold {
        r.refresh();
    }
}

pub fn process_box_box(w0: &Wrap, w1: &Wrap, manifold: ManifoldId, own_manifold: bool, r: &mut ManifoldResult) {
    r.current = manifold.0 as usize;
    let (Shape::Box(b0), Shape::Box(b1)) = (w0.shape, w1.shape) else { unreachable!("box-box") };
    let side = |b: &BoxShape| {
        let m = b.margin;
        let (hx, hy, hz) = (b.half.x + m, b.half.y + m, b.half.z + m);
        v3(hx + hx, hy + hy, hz + hz)
    };
    let (s1, s2) = (side(b1), side(b0));
    box_box_detect(&w0.transform.origin, &w0.transform.basis, &s2, &w1.transform.origin, &w1.transform.basis, &s1, r);
    if own_manifold {
        r.refresh();
    }
}

#[inline]
fn m(r: &Mat3, row: usize, col: usize) -> f32 {
    r.r[row][col]
}

pub fn box_box_detect(p1: &Vec3, r1: &Mat3, side1: &Vec3, p2: &Vec3, r2: &Mat3, side2: &Vec3, out: &mut impl ContactResult) {
    let dy = p2.y - p1.y;
    let dz = p2.z - p1.z;
    let dx = p2.x - p1.x;
    let (r1m, r2m) = (r1, r2);
    let a = |row: usize, col: usize| m(r1m, row, col);
    let b = |row: usize, col: usize| m(r2m, row, col);

    let pp0 = a(1, 0) * dy + a(0, 0) * dx + a(2, 0) * dz;
    let pp1 = a(1, 1) * dy + a(0, 1) * dx + a(2, 1) * dz;
    let pp2 = a(1, 2) * dy + a(0, 2) * dx + a(2, 2) * dz;
    let bx = side2.x * 0.5;
    let az = side1.z * 0.5;
    let ax = side1.x * 0.5;
    let ay = side1.y * 0.5;
    let by = side2.y * 0.5;
    let bz = side2.z * 0.5;
    let rr11 = b(0, 0) * a(0, 0) + b(1, 0) * a(1, 0) + b(2, 0) * a(2, 0);
    let rr12 = b(0, 1) * a(0, 0) + b(1, 1) * a(1, 0) + b(2, 1) * a(2, 0);
    let rr13 = b(0, 2) * a(0, 0) + b(1, 2) * a(1, 0) + b(2, 2) * a(2, 0);
    let rr21 = b(0, 0) * a(0, 1) + b(1, 0) * a(1, 1) + b(2, 0) * a(2, 1);
    let rr22 = b(0, 1) * a(0, 1) + b(1, 1) * a(1, 1) + b(2, 1) * a(2, 1);
    let rr23 = b(0, 2) * a(0, 1) + b(1, 2) * a(1, 1) + b(2, 2) * a(2, 1);
    let rr31 = b(0, 0) * a(0, 2) + b(1, 0) * a(1, 2) + b(2, 0) * a(2, 2);
    let rr32 = b(0, 1) * a(0, 2) + b(1, 1) * a(1, 2) + b(2, 1) * a(2, 2);
    let rr33 = b(0, 2) * a(0, 2) + b(1, 2) * a(1, 2) + b(2, 2) * a(2, 2);
    let (q11, q12, q13) = (rr11.abs(), rr12.abs(), rr13.abs());
    let (q21, q22, q23) = (rr21.abs(), rr22.abs(), rr23.abs());
    let (q31, q32, q33) = (rr31.abs(), rr32.abs(), rr33.abs());

    let mut s = -f32::MAX;
    let mut invert = false;

    let mut code = 0u32;

    let mut norm_col: Option<(u8, usize)> = None;

    let s2 = pp0.abs() - (q11 * bx + ax + q12 * by + q13 * bz);
    if 0.0 < s2 {
        return;
    }
    if s < s2 {
        s = s2;
        invert = pp0 < 0.0;
        norm_col = Some((1, 0));
        code = 1;
    }
    let s2 = pp1.abs() - (bx * q21 + ay + by * q22 + bz * q23);
    if 0.0 < s2 {
        return;
    }
    if s < s2 {
        s = s2;
        invert = pp1 < 0.0;
        norm_col = Some((1, 1));
        code = 2;
    }
    let s2 = pp2.abs() - (bx * q31 + az + by * q32 + bz * q33);
    if 0.0 < s2 {
        return;
    }
    if s < s2 {
        s = s2;
        invert = pp2 < 0.0;
        norm_col = Some((1, 2));
        code = 3;
    }
    let t = b(1, 0) * dy + b(0, 0) * dx + b(2, 0) * dz;
    let s2 = t.abs() - (ay * q21 + ax * q11 + az * q31 + bx);
    if 0.0 < s2 {
        return;
    }
    if s < s2 {
        s = s2;
        invert = t < 0.0;
        norm_col = Some((2, 0));
        code = 4;
    }
    let t = dx * b(0, 1) + dy * b(1, 1) + dz * b(2, 1);
    let s2 = t.abs() - (ay * q22 + ax * q12 + az * q32 + by);
    if 0.0 < s2 {
        return;
    }
    if s < s2 {
        s = s2;
        invert = t < 0.0;
        norm_col = Some((2, 1));
        code = 5;
    }
    let t = dx * b(0, 2) + dy * b(1, 2) + dz * b(2, 2);
    let s2 = t.abs() - (ay * q23 + ax * q13 + az * q33 + bz);
    if 0.0 < s2 {
        return;
    }
    if s < s2 {
        s = s2;
        invert = t < 0.0;
        norm_col = Some((2, 2));
        code = 6;
    }
    const FUDGE2: f32 = f32::from_bits(0x3727c5ac);
    let (f11, f12, f13) = (q11 + FUDGE2, q12 + FUDGE2, q13 + FUDGE2);
    let (f21, f22, f23) = (q21 + FUDGE2, q22 + FUDGE2, q23 + FUDGE2);
    let (f31, f32_, f33) = (q31 + FUDGE2, q32 + FUDGE2, q33 + FUDGE2);
    const FUDGE: f32 = f32::from_bits(0x3f866666);
    let (mut n1, mut n2, mut n3) = (0.0f32, 0.0f32, 0.0f32);

    let t = pp2 * rr21 - pp1 * rr31;
    let s2 = t.abs() - (ay * f31 + az * f21 + by * f13 + bz * f12);
    if EPS < s2 {
        return;
    }
    let neg = -rr31;
    let l = ((0.0 - neg * rr31) + rr21 * rr21).sqrt();
    if EPS < l {
        let sd = s2 / l;
        if s < sd * FUDGE {
            code = 7;
            norm_col = None;
            invert = t < 0.0;
            n1 = 0.0 / l;
            n2 = neg / l;
            n3 = rr21 / l;
            s = sd;
        }
    }
    let t = pp2 * rr22 - pp1 * rr32;
    let s2 = t.abs() - (ay * f32_ + az * f22 + bx * f13 + bz * f11);
    if EPS < s2 {
        return;
    }
    let neg = -rr32;
    let l = ((0.0 - neg * rr32) + rr22 * rr22).sqrt();
    if EPS < l && s < (s2 / l) * FUDGE {
        code = 8;
        norm_col = None;
        invert = t < 0.0;
        n1 = 0.0 / l;
        n2 = neg / l;
        n3 = rr22 / l;
        s = s2 / l;
    }
    let t = pp2 * rr23 - pp1 * rr33;
    let s2 = t.abs() - (ay * f33 + az * f23 + bx * f12 + by * f11);
    if EPS < s2 {
        return;
    }
    let neg = -rr33;
    let l = ((0.0 - neg * rr33) + rr23 * rr23).sqrt();
    if EPS < l && s < (s2 / l) * FUDGE {
        code = 9;
        norm_col = None;
        invert = t < 0.0;
        n1 = 0.0 / l;
        n2 = neg / l;
        n3 = rr23 / l;
        s = s2 / l;
    }
    let t = pp0 * rr31 - pp2 * rr11;
    let s2 = t.abs() - (ax * f31 + az * f11 + by * f23 + bz * f22);
    if EPS < s2 {
        return;
    }
    let neg = -rr11;
    let l = ((rr31 * rr31 + 0.0) - neg * rr11).sqrt();
    if EPS < l && s < (s2 / l) * FUDGE {
        code = 10;
        norm_col = None;
        invert = t < 0.0;
        n1 = rr31 / l;
        n2 = 0.0 / l;
        n3 = neg / l;
        s = s2 / l;
    }
    let t = pp0 * rr32 - pp2 * rr12;
    let s2 = t.abs() - (ax * f32_ + az * f12 + bx * f23 + bz * f21);
    if EPS < s2 {
        return;
    }
    let neg = -rr12;
    let l = ((rr32 * rr32 + 0.0) - neg * rr12).sqrt();
    if EPS < l && s < (s2 / l) * FUDGE {
        code = 11;
        norm_col = None;
        invert = t < 0.0;
        n1 = rr32 / l;
        n2 = 0.0 / l;
        n3 = neg / l;
        s = s2 / l;
    }
    let t = pp0 * rr33 - pp2 * rr13;
    let s2 = t.abs() - (ax * f33 + az * f13 + bx * f22 + by * f21);
    if EPS < s2 {
        return;
    }
    let neg = -rr13;
    let l = ((rr33 * rr33 + 0.0) - neg * rr13).sqrt();
    if EPS < l && s < (s2 / l) * FUDGE {
        code = 12;
        norm_col = None;
        invert = t < 0.0;
        n1 = rr33 / l;
        n2 = 0.0 / l;
        n3 = neg / l;
        s = s2 / l;
    }
    let t = pp1 * rr11 - pp0 * rr21;
    let s2 = t.abs() - (ax * f21 + ay * f11 + by * f33 + bz * f32_);
    if EPS < s2 {
        return;
    }
    let neg = -rr21;
    let l = ((rr11 * rr11 - neg * rr21) + 0.0).sqrt();
    if EPS < l {
        let sd = s2 / l;
        if s < sd * FUDGE {
            code = 13;
            norm_col = None;
            invert = t < 0.0;
            n1 = neg / l;
            n2 = rr11 / l;
            n3 = 0.0 / l;
            s = sd;
        }
    }
    let t = pp1 * rr12 - pp0 * rr22;
    let s2 = t.abs() - (ax * f22 + ay * f12 + bx * f33 + bz * f31);
    if EPS < s2 {
        return;
    }
    let neg = -rr22;
    let l = ((rr12 * rr12 - neg * rr22) + 0.0).sqrt();
    if EPS < l {
        let sd = s2 / l;
        if s < sd * FUDGE {
            code = 14;
            norm_col = None;
            invert = t < 0.0;
            n1 = neg / l;
            n2 = rr12 / l;
            n3 = 0.0 / l;
            s = sd;
        }
    }
    let t = pp1 * rr13 - pp0 * rr23;
    let s2 = t.abs() - (ax * f23 + ay * f13 + bx * f32_ + by * f31);
    if EPS < s2 {
        return;
    }
    let neg = -rr23;
    let l = ((rr13 * rr13 - neg * rr23) + 0.0).sqrt();
    let mut edge15 = false;
    if !(l <= EPS) {
        let sd = s2 / l;
        if !(sd * FUDGE <= s) {
            edge15 = true;
            code = 15;
            invert = t < 0.0;
            n1 = neg / l;
            n2 = rr13 / l;
            n3 = 0.0 / l;
            s = sd;
        }
    }
    if !edge15 && code == 0 {
        return;
    }

    let (mut nx, mut ny, mut nz);
    let col_norm = norm_col.filter(|_| !edge15);
    if let Some((bi, col)) = col_norm {
        let rr = if bi == 1 { r1m } else { r2m };
        nx = rr.r[0][col];
        ny = rr.r[1][col];
        nz = rr.r[2][col];
    } else {
        nx = n1 * a(0, 0) + n2 * a(0, 1) + n3 * a(0, 2);
        ny = n2 * a(1, 1) + n1 * a(1, 0) + n3 * a(1, 2);
        nz = n2 * a(2, 1) + n1 * a(2, 0) + n3 * a(2, 2);
    }
    if invert {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }
    let depth_s = s;

    if 6 < code {

        let sgn = |d: f32, pos: f32, neg: f32| if d <= 0.0 { neg } else { pos };
        let ua = ax * sgn(nx * a(0, 0) + a(1, 0) * ny + nz * a(2, 0), 1.0, -1.0);
        let ub = ay * sgn(ny * a(1, 1) + a(0, 1) * nx + nz * a(2, 1), 1.0, -1.0);
        let uc = az * sgn(a(0, 2) * nx + ny * a(1, 2) + nz * a(2, 2), 1.0, -1.0);
        let pax = ((p1.x + ua * a(0, 0)) + ub * a(0, 1)) + uc * a(0, 2);
        let pay = ((p1.y + ua * a(1, 0)) + ub * a(1, 1)) + uc * a(1, 2);
        let paz = ((p1.z + ua * a(2, 0)) + ub * a(2, 1)) + uc * a(2, 2);
        let va = bx * sgn(b(1, 0) * ny + b(0, 0) * nx + b(2, 0) * nz, -1.0, 1.0);
        let vb = by * sgn(ny * b(1, 1) + b(0, 1) * nx + nz * b(2, 1), -1.0, 1.0);
        let vc = bz * sgn(ny * b(1, 2) + b(0, 2) * nx + nz * b(2, 2), -1.0, 1.0);
        let qx = ((va * b(0, 0) + p2.x) + vb * b(0, 1)) + vc * b(0, 2);
        let qy = ((va * b(1, 0) + p2.y) + vb * b(1, 1)) + vc * b(1, 2);
        let qz = ((va * b(2, 0) + p2.z) + vb * b(2, 1)) + vc * b(2, 2);
        let c = (code - 7) as usize;
        let i1 = c / 3;
        let i2 = c % 3;
        let dpx = qx - pax;
        let (bu0, bu1, bu2) = (b(0, i2), b(1, i2), b(2, i2));
        let dpy = qy - pay;
        let dpz = qz - paz;
        let uaub = bu1 * a(1, i1) + bu0 * a(0, i1) + bu2 * a(2, i1);
        let d = 1.0 - uaub * uaub;
        let mut beta = 0.0f32;
        if f32::from_bits(0x38d1b717) < d {
            beta = (-(bu1 * dpy + bu0 * dpx + bu2 * dpz) + (a(1, i1) * dpy + a(0, i1) * dpx + a(2, i1) * dpz) * uaub) * (1.0 / d);
        }
        let point = v3(bu0 * beta + qx, bu1 * beta + qy, bu2 * beta + qz);
        out.add_contact_point(v3(-nx, -ny, -nz), point, -(-depth_s));
        return;
    }

    let (ra, rb, pa, pb, sa, sb, nrx, nry, nrz);
    if code <= 3 {
        ra = r1m;
        rb = r2m;
        pa = *p1;
        pb = *p2;
        sa = [ax, ay, az];
        sb = [bx, by, bz];
        nrx = nx;
        nry = ny;
        nrz = nz;
    } else {
        ra = r2m;
        rb = r1m;
        pa = *p2;
        pb = *p1;
        sa = [bx, by, bz];
        sb = [ax, ay, az];
        nrx = -nx;
        nry = -ny;
        nrz = -nz;
    }
    let rb_at = |row: usize, col: usize| rb.r[row][col];
    let ra_at = |row: usize, col: usize| ra.r[row][col];
    let nr0 = nry * rb_at(1, 0) + nrx * rb_at(0, 0) + nrz * rb_at(2, 0);
    let nr1 = nry * rb_at(1, 1) + nrx * rb_at(0, 1) + nrz * rb_at(2, 1);
    let nr2 = nry * rb_at(1, 2) + nrx * rb_at(0, 2) + nrz * rb_at(2, 2);
    let (an0, an1, an2) = (nr0.abs(), nr1.abs(), nr2.abs());

    let (lanr, a1, a2): (usize, usize, usize);
    if an0 < an1 {
        lanr = (an1 <= an2) as usize + 1;
        a1 = 0;
        a2 = (an2 < an1) as usize + 1;
    } else if an2 < an0 {
        lanr = 0;
        a1 = 1;
        a2 = 2;
    } else {
        lanr = 2;
        a1 = 0;
        a2 = 1;
    }
    let nr_l = [nr0, nr1, nr2][lanr];
    let sl = sb[lanr];
    let (cx, cy, cz);
    if 0.0 < nr_l || nr_l == 0.0 {
        cx = (pb.x - pa.x) - sl * rb_at(0, lanr);
        cy = (pb.y - pa.y) - sl * rb_at(1, lanr);
        cz = (pb.z - pa.z) - sl * rb_at(2, lanr);
    } else {
        cx = sl * rb_at(0, lanr) + (pb.x - pa.x);
        cy = (pb.y - pa.y) + sl * rb_at(1, lanr);
        cz = (pb.z - pa.z) + sl * rb_at(2, lanr);
    }
    let codeg = (if code < 4 { code - 1 } else { code - 4 }) as usize;
    let (c1, c2): (usize, usize) = if codeg == 0 {
        (1, 2)
    } else if codeg == 1 {
        (0, 2)
    } else {
        (0, 1)
    };
    let (a1x, a1y, a1z) = (ra_at(0, c1), ra_at(1, c1), ra_at(2, c1));
    let (a2x, a2y, a2z) = (ra_at(0, c2), ra_at(1, c2), ra_at(2, c2));
    let c1v = cx * a1x + cy * a1y + cz * a1z;
    let c2v = cx * a2x + cy * a2y + cz * a2z;
    let (ba1x, ba1y, ba1z) = (rb_at(0, a1), rb_at(1, a1), rb_at(2, a1));
    let (ba2x, ba2y, ba2z) = (rb_at(0, a2), rb_at(1, a2), rb_at(2, a2));
    let m11 = ba1x * a1x + ba1y * a1y + ba1z * a1z;
    let m12t = ba2x * a1x + ba2y * a1y;
    let m21 = ba1x * a2x + ba1y * a2y + ba1z * a2z;
    let m12 = m12t + ba2z * a1z;
    let k1 = sb[a2] * m12;
    let m22 = ba2x * a2x + ba2y * a2y + ba2z * a2z;
    let k3 = sb[a1] * m11;
    let k4 = sb[a1] * m21;
    let t1 = c1v - k3;
    let k2 = sb[a2] * m22;
    let t0 = t1 - k1;
    let t2 = c2v - k4;
    let mut quad = [0.0f32; 8];
    quad[0] = t0;
    quad[1] = t2 - k2;
    quad[2] = t1 + k1;
    quad[3] = t2 + k2;
    let t3 = c1v + k3;
    let t4 = c2v + k4;
    quad[4] = t3 + k1;
    quad[5] = t4 + k2;
    quad[6] = t3 - k1;
    quad[7] = t4 - k2;
    let rect = [sa[c1], sa[c2]];
    let mut ret = [0.0f32; 16];
    let n = intersect_rect_quad(rect, &quad, &mut ret);
    if n < 1 {
        return;
    }
    let n = n as usize;
    let det1 = 1.0 / (m22 * m11 - m12 * m21);
    let (m11d, m12d, m21d, m22d) = (m11 * det1, m12 * det1, m21 * det1, m22 * det1);
    let mut pts = [[0.0f32; 3]; 8];
    let mut deps = [0.0f32; 8];
    let mut cnum = 0usize;
    for j in 0..n {
        let u = ret[j * 2];
        let v = ret[j * 2 + 1];
        let k1_ = u - c1v;
        let d2 = v - c2v;
        let al = m11d * d2 - k1_ * m21d;
        let be = k1_ * m22d - m12d * d2;
        let px = be * ba1x + cx + al * ba2x;
        let py = be * ba1y + cy + al * ba2y;
        let pz = be * ba1z + cz + al * ba2z;
        pts[cnum] = [px, py, pz];
        let dp = sa[codeg] - (nry * py + nrx * px + nrz * pz);
        deps[cnum] = dp;
        if 0.0 <= dp {
            ret[cnum * 2] = u;
            ret[cnum * 2 + 1] = v;
            cnum += 1;
        }
    }
    if cnum < 1 {
        return;
    }

    let onx = -nx;
    let ony = -ny;
    let onz = -nz;
    let maxc = 4usize;
    if maxc < cnum {
        let i1 = {
            let mut best = 0usize;
            let mut bd = deps[0];
            for i in 1..cnum {
                if bd < deps[i] {
                    best = i;
                    bd = deps[i];
                }
            }
            best
        };
        let keep = cull_points(cnum, &ret, maxc, i1);
        for j in 0..maxc {
            let idx = keep[j];
            let d = deps[idx];
            let (qx, qy, qz) = (pts[idx][0] + pa.x, pts[idx][1] + pa.y, pts[idx][2] + pa.z);
            if code < 4 {
                out.add_contact_point(v3(onx, ony, onz), v3(qx, qy, qz), -d);
            } else {
                let dd = d;
                out.add_contact_point(v3(onx, ony, onz), v3(qx - nx * dd, qy - ny * dd, qz - nz * dd), -d);
            }
        }
    } else if code < 4 {
        for j in 0..cnum {
            out.add_contact_point(v3(onx, ony, onz), v3(pts[j][0] + pa.x, pts[j][1] + pa.y, pts[j][2] + pa.z), -deps[j]);
        }
    } else {
        for j in 0..cnum {
            let d = deps[j];
            let point = v3((pts[j][0] + pa.x) - nx * d, (pts[j][1] + pa.y) - ny * d, (pts[j][2] + pa.z) - nz * d);
            out.add_contact_point(v3(onx, ony, onz), point, -d);
        }
    }
}

pub fn intersect_rect_quad(h: [f32; 2], p: &[f32; 8], ret: &mut [f32; 16]) -> i32 {
    let mut q = [0.0f32; 16];
    q[..8].copy_from_slice(p);
    let mut nq: usize = 4;
    let mut nr: usize = 0;
    let mut r = [0.0f32; 16];
    'done: for dir in 0..2usize {
        let mut sign: i32 = -1;
        while sign <= 1 {
            nr = 0;
            let mut count = nq as i32;
            let mut pq = 0usize;
            let s = sign as f32;
            while 0 < count {
                if s * q[pq + dir] < h[dir] {
                    r[nr * 2] = q[pq];
                    r[nr * 2 + 1] = q[pq + 1];
                    nr += 1;
                    if nr & 8 != 0 {
                        break 'done;
                    }
                }
                let next = if 1 < count { pq + 2 } else { 0 };
                if (q[next + dir] * s < h[dir]) != (q[pq + dir] * s < h[dir]) {
                    let other = 1 - dir;
                    r[nr * 2 + other] = ((q[next + other] - q[pq + other]) / (q[next + dir] - q[pq + dir])) * (h[dir] * s - q[pq + dir]) + q[pq + other];
                    r[nr * 2 + dir] = s * h[dir];
                    nr += 1;
                    if nr & 8 != 0 {
                        break 'done;
                    }
                }
                pq += 2;
                count -= 1;
            }
            q = r;
            nq = nr;
            sign += 2;
        }
    }

    ret[..nr * 2].copy_from_slice(&r[..nr * 2]);
    nr as i32
}

pub fn cull_points(n: usize, p: &[f32; 16], m_keep: usize, i0: usize) -> [usize; 8] {
    let (cx, cy);
    if n == 1 {
        cx = p[0];
        cy = p[1];
    } else if n == 2 {
        cx = (p[2] + p[0]) * 0.5;
        cy = (p[3] + p[1]) * 0.5;
    } else {

        let m1 = n - 1;
        let mut a = 0.0f32;
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut i = 0usize;
        if m1 >= 4 {
            let groups = ((n - 5) >> 2) + 1;
            for _ in 0..groups {
                let (x0, y0, x1, y1, x2, y2, x3, y3, x4, y4) = (p[i * 2], p[i * 2 + 1], p[i * 2 + 2], p[i * 2 + 3], p[i * 2 + 4], p[i * 2 + 5], p[i * 2 + 6], p[i * 2 + 7], p[i * 2 + 8], p[i * 2 + 9]);
                let q0 = y1 * x0 - y0 * x1;
                let q1 = y2 * x1 - y1 * x2;
                let q2 = y3 * x2 - y2 * x3;
                let q3 = y4 * x3 - x4 * y3;
                a = a + q0 + q1 + q2 + q3;
                sx = (x1 + x0) * q0 + sx + (x1 + x2) * q1 + (x3 + x2) * q2 + (x4 + x3) * q3;
                sy = (y0 + y1) * q0 + sy + (y2 + y1) * q1 + (y2 + y3) * q2 + (y4 + y3) * q3;
                i += 4;
            }
        }
        while i < m1 {
            let (x0, y0, x1, y1) = (p[i * 2], p[i * 2 + 1], p[i * 2 + 2], p[i * 2 + 3]);
            let q = x0 * y1 - y0 * x1;
            a += q;
            sx += (x0 + x1) * q;
            sy += (y0 + y1) * q;
            i += 1;
        }
        let (xl, yl) = (p[(n - 1) * 2], p[(n - 1) * 2 + 1]);
        let q = xl * p[1] - yl * p[0];
        let k = if (q + a).abs() <= EPS { LARGE } else { 1.0 / ((q + a) * 3.0) };
        cx = ((xl + p[0]) * q + sx) * k;
        cy = ((yl + p[1]) * q + sy) * k;
    }
    let mut ang = [0.0f32; 8];
    for i in 0..n {
        ang[i] = atan2f(p[i * 2 + 1] - cy, p[i * 2] - cx);
    }
    let mut avail = [true; 8];
    let mut iret = [0usize; 8];
    iret[0] = i0;
    avail[i0] = false;
    const TWO_PI: f32 = f32::from_bits(0x40c90fdb);
    const PI: f32 = f32::from_bits(0x40490fdb);
    let step = TWO_PI / m_keep as f32;
    for j in 1..m_keep {
        let mut target = j as f32 * step + ang[i0];
        if PI < target {
            target += -TWO_PI;
        }
        let mut best = 1e9f32;
        let mut found = i0;
        for i in 0..n {
            if avail[i] {
                let mut d = (ang[i] - target).abs();
                if PI < d {
                    d = TWO_PI - d;
                }
                if d < best {
                    found = i;
                    best = d;
                }
            }
        }
        iret[j] = found;
        avail[found] = false;
    }
    iret
}
