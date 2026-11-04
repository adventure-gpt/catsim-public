use crate::bt::gjk::{ClosestPointInput, ContactResult, GjkDetector};
use crate::bt::shapes::Shape;
use crate::bt::simplex::Simplex;
use crate::math::{Transform, Vec3, v3};

const EPS2: f32 = f32::from_bits(0x28800000);
const LARGE: f32 = f32::from_bits(0x5d5e0b6b);

#[derive(Clone, Copy, Debug)]
pub struct CastResult {
    pub normal: Vec3,
    pub hit_point: Vec3,
    pub fraction: f32,
    pub allowed_penetration: f32,
}

impl CastResult {

    pub fn new(fraction: f32) -> CastResult {
        CastResult { normal: Vec3::ZERO, hit_point: Vec3::ZERO, fraction, allowed_penetration: 0.0 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointCollector {
    pub normal: Vec3,
    pub point: Vec3,
    pub distance: f32,
    pub has_result: bool,
}

impl PointCollector {
    pub fn new() -> PointCollector {
        PointCollector { normal: Vec3::ZERO, point: Vec3::ZERO, distance: LARGE, has_result: false }
    }
}

impl Default for PointCollector {
    fn default() -> PointCollector {
        PointCollector::new()
    }
}

impl ContactResult for PointCollector {
    fn add_contact_point(&mut self, normal: Vec3, point: Vec3, depth: f32) {
        if depth < self.distance {
            self.has_result = true;
            self.normal = normal;
            self.point = point;
            self.distance = depth;
        }
    }
}

pub fn gjk_cast(a: &Shape, b: &Shape, simplex: &mut Simplex, from_a: &Transform, to_a: &Transform, from_b: &Transform, to_b: &Transform, res: &mut CastResult) -> bool {
    simplex.reset();
    let rz = (to_a.origin.z - from_a.origin.z) - (to_b.origin.z - from_b.origin.z);
    let ry = (to_a.origin.y - from_a.origin.y) - (to_b.origin.y - from_b.origin.y);
    let rx = (to_a.origin.x - from_a.origin.x) - (to_b.origin.x - from_b.origin.x);
    let mut col = PointCollector::new();
    let mut gjk = GjkDetector::new(a, b, false);
    let mut input = ClosestPointInput { a: *from_a, b: *from_b, max_distance_sqr: LARGE };
    gjk.closest_points(simplex, &input, &mut col);
    if !col.has_result {
        return false;
    }
    let mut lambda = 0.0f32;
    let mut iter = 0;
    loop {
        if col.distance <= f32::from_bits(0x3a83126f) {
            if -res.allowed_penetration <= ry * col.normal.y + col.normal.x * rx + rz * col.normal.z {
                return false;
            }
            break;
        }
        iter += 1;
        if 64 < iter {
            return false;
        }
        let l = lambda - col.distance / (ry * col.normal.y + col.normal.x * rx + rz * col.normal.z);
        if 1.0 < l || l < 0.0 || l <= lambda {
            return false;
        }
        let s = 1.0 - l;
        input.a.origin = v3(l * to_a.origin.x + s * from_a.origin.x, l * to_a.origin.y + s * from_a.origin.y, s * from_a.origin.z + l * to_a.origin.z);
        input.b.origin = v3(l * to_b.origin.x + s * from_b.origin.x, s * from_b.origin.y + l * to_b.origin.y, l * to_b.origin.z + s * from_b.origin.z);
        gjk.closest_points(simplex, &input, &mut col);
        if !col.has_result {
            return false;
        }
        lambda = l;
        if !(0.0 <= col.distance) {
            break;
        }
    }
    res.hit_point = col.point;
    res.normal = col.normal;
    res.fraction = lambda;
    true
}

pub fn subsimplex_cast(a: &Shape, b: &Shape, simplex: &mut Simplex, from_a: &Transform, to_a: &Transform, from_b: &Transform, to_b: &Transform, res: &mut CastResult) -> bool {
    simplex.reset();
    let ra = from_a.basis.r;
    let rb = from_b.basis.r;
    let (fa, fb) = (from_a.origin, from_b.origin);
    let (mut oa, mut ob) = (fa, fb);
    let rx = (to_a.origin.x - fa.x) - (to_b.origin.x - fb.x);
    let ry = (to_a.origin.y - fa.y) - (to_b.origin.y - fb.y);
    let rz = (to_a.origin.z - fa.z) - (to_b.origin.z - fb.z);
    let (nx, ny, nz) = (-rx, -ry, -rz);
    let d = v3(nx * ra[0].x + ny * ra[1].x + nz * ra[2].x, ny * ra[1].y + nx * ra[0].y + nz * ra[2].y, ny * ra[1].z + nx * ra[0].z + nz * ra[2].z);
    let sa = a.local_support(d);
    let d = v3(ry * rb[1].x + rx * rb[0].x + rz * rb[2].x, ry * rb[1].y + rx * rb[0].y + rz * rb[2].y, ry * rb[1].z + rx * rb[0].z + rz * rb[2].z);
    let sb = b.local_support(d);
    let mut vx = (sa.y * ra[0].y + sa.x * ra[0].x + sa.z * ra[0].z + fa.x) - (sb.x * rb[0].x + sb.y * rb[0].y + sb.z * rb[0].z + fb.x);
    let mut vy = (sa.y * ra[1].y + sa.x * ra[1].x + sa.z * ra[1].z + fa.y) - (sb.y * rb[1].y + sb.x * rb[1].x + sb.z * rb[1].z + fb.y);
    let mut vz = (sa.y * ra[2].y + sa.x * ra[2].x + sa.z * ra[2].z + fa.z) - (sb.y * rb[2].y + sb.x * rb[2].x + sb.z * rb[2].z + fb.z);
    let mut lambda = 0.0f32;
    let (mut hx, mut hy, mut hz) = (0.0f32, 0.0f32, 0.0f32);
    let eps = f32::from_bits(0x38d1b717);
    if eps < vy * vy + vx * vx + vz * vz {
        let mut max_iter = 64;
        while max_iter != 0 {
            max_iter -= 1;
            let (ny, nx, nz) = (-vy, -vx, -vz);
            let d = v3(ra[1].x * ny + ra[0].x * nx + ra[2].x * nz, ra[1].y * ny + ra[0].y * nx + ra[2].y * nz, ra[1].z * ny + ra[0].z * nx + ra[2].z * nz);
            let s = a.local_support(d);
            let pw = v3(ra[0].x * s.x + ra[0].y * s.y + ra[0].z * s.z + oa.x, ra[1].x * s.x + ra[1].y * s.y + ra[1].z * s.z + oa.y, ra[2].x * s.x + ra[2].y * s.y + ra[2].z * s.z + oa.z);
            let d = v3(rb[0].x * vx + rb[1].x * vy + rb[2].x * vz, rb[0].y * vx + rb[1].y * vy + rb[2].y * vz, rb[0].z * vx + rb[1].z * vy + rb[2].z * vz);
            let s = b.local_support(d);
            let qw = v3(rb[0].x * s.x + rb[0].y * s.y + rb[0].z * s.z + ob.x, rb[1].x * s.x + rb[1].y * s.y + rb[1].z * s.z + ob.y, rb[2].x * s.x + rb[2].y * s.y + rb[2].z * s.z + ob.z);
            let w = v3(pw.x - qw.x, pw.y - qw.y, pw.z - qw.z);
            let vw = w.y * vy + w.x * vx + w.z * vz;
            if 1.0 < lambda {
                return false;
            }
            if 0.0 < vw {
                let vr = rx * vx + ry * vy + rz * vz;
                if -EPS2 <= vr {
                    return false;
                }
                lambda -= vw / vr;
                let s = 1.0 - lambda;
                oa = v3(s * fa.x + lambda * to_a.origin.x, lambda * to_a.origin.y + s * fa.y, s * fa.z + lambda * to_a.origin.z);
                ob = v3(s * fb.x + lambda * to_b.origin.x, lambda * to_b.origin.y + s * fb.y, s * fb.z + lambda * to_b.origin.z);
                hx = vx;
                hy = vy;
                hz = vz;
            }
            if !simplex.in_simplex(w) {
                simplex.add_vertex(w, pw, qw);
            }
            let ok = simplex.update();
            let c = simplex.cached_v;
            vx = c.x;
            vy = c.y;
            vz = c.z;
            if !ok || vx * vx + vy * vy + vz * vz <= eps {
                break;
            }
        }
    }
    res.fraction = lambda;
    let l2 = hx * hx + hy * hy + hz * hz;
    let n = if EPS2 <= l2 {
        let k = 1.0 / l2.sqrt();
        v3(hx * k, hy * k, hz * k)
    } else {
        Vec3::ZERO
    };
    res.normal = n;
    if ry * n.y + rx * n.x + rz * n.z < -res.allowed_penetration {
        simplex.update();
        res.hit_point = simplex.cached_p2;
        return true;
    }
    false
}
