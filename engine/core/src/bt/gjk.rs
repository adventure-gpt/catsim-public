use crate::bt::epa;
use crate::bt::shapes::{SHAPE_TRIANGLE, Shape};
use crate::bt::simplex::Simplex;
use crate::math::{Transform, Vec3, v3};

const EPS2: f32 = f32::from_bits(0x28800000);
const LARGE: f32 = f32::from_bits(0x5d5e0b6b);

pub trait ContactResult {
    fn add_contact_point(&mut self, normal: Vec3, point: Vec3, depth: f32);
}

#[derive(Clone, Copy, Debug)]
pub struct ClosestPointInput {
    pub a: Transform,
    pub b: Transform,
    pub max_distance_sqr: f32,
}

pub struct GjkDetector<'a> {
    pub a: &'a Shape,
    pub b: &'a Shape,
    pub use_epa: bool,
    pub cached_axis: Vec3,
    pub margin_a: f32,
    pub margin_b: f32,
    pub ignore_margin: bool,
    pub cached_distance: f32,
    pub last_method: i32,
    pub cur_iter: i32,
    pub degenerate: i32,
    pub catch_degeneracies: bool,
    pub fix_normal: bool,
}

impl<'a> GjkDetector<'a> {

    pub fn new(a: &'a Shape, b: &'a Shape, use_epa: bool) -> GjkDetector<'a> {
        GjkDetector {
            a,
            b,
            use_epa,
            cached_axis: v3(0.0, 1.0, 0.0),
            margin_a: a.margin(),
            margin_b: b.margin(),
            ignore_margin: false,
            cached_distance: 0.0,
            last_method: -1,
            cur_iter: 0,
            degenerate: 0,
            catch_degeneracies: true,
            fix_normal: true,
        }
    }

    pub fn closest_points(&mut self, simplex: &mut Simplex, input: &ClosestPointInput, out: &mut impl ContactResult) {
        self.cached_distance = 0.0;
        let ra = input.a.basis.r;
        let rb = input.b.basis.r;
        let (oa, ob) = (input.a.origin, input.b.origin);
        let off = v3((ob.x + oa.x) * 0.5, (oa.y + ob.y) * 0.5, (oa.z + ob.z) * 0.5);
        let ta = v3(oa.x - off.x, oa.y - off.y, oa.z - off.z);
        let tb = v3(ob.x - off.x, ob.y - off.y, ob.z - off.z);
        let local_a = Transform { basis: input.a.basis, origin: ta };
        let local_b = Transform { basis: input.b.basis, origin: tb };
        let (mut margin_b, mut margin_a) = (self.margin_b, self.margin_a);
        if self.ignore_margin {
            margin_a = 0.0;
            margin_b = 0.0;
        }
        let margin = margin_b + margin_a;
        self.cur_iter = 0;
        self.cached_axis = v3(0.0, 1.0, 0.0);
        self.last_method = -1;
        self.degenerate = 0;
        simplex.reset();

        let mut sq_dist = LARGE;
        let mut distance = 0.0f32;
        let mut is_valid = false;
        let mut normal = Vec3::ZERO;
        let mut point;
        let mut w;
        let mut check_simplex = false;
        loop {
            let v = self.cached_axis;
            let (nvy, nvx, nvz) = (-v.y, -v.x, -v.z);
            let sep_a = v3(nvy * ra[1].x + nvx * ra[0].x + nvz * ra[2].x, nvy * ra[1].y + nvx * ra[0].y + nvz * ra[2].y, nvy * ra[1].z + nvx * ra[0].z + nvz * ra[2].z);
            let sep_b = v3(v.y * rb[1].x + v.x * rb[0].x + v.z * rb[2].x, v.y * rb[1].y + v.x * rb[0].y + v.z * rb[2].y, v.y * rb[1].z + v.x * rb[0].z + v.z * rb[2].z);
            let p = self.a.support_no_margin(sep_a);
            let q = self.b.support_no_margin(sep_b);
            let pw = v3(p.x * ra[0].x + p.y * ra[0].y + p.z * ra[0].z + ta.x, ra[1].x * p.x + ra[1].y * p.y + ra[1].z * p.z + ta.y, ra[2].x * p.x + ra[2].y * p.y + ra[2].z * p.z + ta.z);
            let qw = v3(q.x * rb[0].x + q.y * rb[0].y + q.z * rb[0].z + tb.x, rb[1].x * q.x + rb[1].y * q.y + rb[1].z * q.z + tb.y, rb[2].x * q.x + rb[2].y * q.y + rb[2].z * q.z + tb.z);
            w = v3(pw.x - qw.x, pw.y - qw.y, pw.z - qw.z);
            let delta = w.y * v.y + w.x * v.x + w.z * v.z;
            if 0.0 < delta && sq_dist * input.max_distance_sqr < delta * delta {
                self.degenerate = 10;
                check_simplex = true;
                break;
            }
            if simplex.in_simplex(w) {
                self.degenerate = 1;
                check_simplex = true;
                break;
            }
            let f0 = sq_dist - delta;
            if f0 <= sq_dist * f32::from_bits(0x358637bd) {
                self.degenerate = if f0 <= 0.0 { 2 } else { 11 };
                check_simplex = true;
                break;
            }
            simplex.add_vertex(w, pw, qw);
            if !simplex.update() {
                self.degenerate = 3;
                check_simplex = true;
                break;
            }
            let nv = simplex.cached_v;
            let l2 = nv.x * nv.x + nv.y * nv.y + nv.z * nv.z;
            if l2 < f32::from_bits(0x358637bd) {
                self.cached_axis = nv;
                self.degenerate = 6;
                check_simplex = true;
                break;
            }
            if sq_dist - l2 <= sq_dist * f32::from_bits(0x34000000) {
                self.degenerate = 12;
                sq_dist = l2;
                check_simplex = true;
                break;
            }
            let it = self.cur_iter;
            self.cached_axis = nv;
            self.cur_iter = it + 1;
            if 1000 < it {
                break;
            }
            sq_dist = l2;
            if simplex.n == 4 {
                self.degenerate = 13;
                break;
            }
        }
        if check_simplex {
            simplex.update();
            let pb = simplex.cached_p2;
            let axis = self.cached_axis;
            let len2 = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
            normal = axis;
            if len2 < f32::from_bits(0x38d1b717) {
                self.degenerate = 5;
            }
            if len2 <= EPS2 {
                self.last_method = 2;
                point = pb;
            } else {
                let rlen = 1.0 / len2.sqrt();
                normal = v3(axis.x * rlen, axis.y * rlen, axis.z * rlen);
                let s = margin_b / sq_dist.sqrt();
                is_valid = true;
                self.last_method = 1;
                point = v3(pb.x + axis.x * s, pb.y + s * axis.y, pb.z + s * axis.z);
                distance = 1.0 / rlen - margin;
            }
        } else {

            point = w;
        }
        let catch_deg = self.catch_degeneracies && self.use_epa && self.degenerate != 0 && !(f32::from_bits(0x3c23d70a) <= margin + distance);
        if !is_valid || catch_deg {
            if self.use_epa {
                self.cached_axis = Vec3::ZERO;
                let guess = v3(tb.x - ta.x, tb.y - ta.y, tb.z - ta.z);

                let (ok, res) = epa::penetration(self.a, &local_a, self.b, &local_b, guess, true);
                let mut valid2 = false;
                let (mut wa, mut wb) = (Vec3::ZERO, Vec3::ZERO);
                if ok {
                    wa = res.witness[0];
                    wb = res.witness[1];
                    self.cached_axis = res.normal;
                    valid2 = true;
                } else {
                    let (ok2, res2) = epa::distance(self.a, &local_a, self.b, &local_b, guess);
                    if ok2 {
                        wa = res2.witness[0];
                        wb = res2.witness[1];
                        self.cached_axis = res2.normal;
                    }
                }
                if !valid2 {
                    let c = self.cached_axis;
                    if 0.0 < c.x * c.x + c.y * c.y + c.z * c.z {
                        let (dy, dx, dz) = (wa.y - wb.y, wa.x - wb.x, wa.z - wb.z);
                        let d2 = (dy * dy + dx * dx + dz * dz).sqrt() - margin;
                        if !is_valid || d2 < distance {
                            point = v3(c.x * margin_b + wb.x, wb.y + c.y * margin_b, wb.z + c.z * margin_b);
                            let k = 1.0 / (c.y * c.y + c.x * c.x + c.z * c.z).sqrt();
                            normal = v3(c.x * k, c.y * k, c.z * k);
                            is_valid = true;
                            distance = d2;
                            self.last_method = 6;
                        } else {
                            self.last_method = 5;
                        }
                    }
                } else {
                    let (mut nx, mut ny, mut nz) = (wb.x - wa.x, wb.y - wa.y, wb.z - wa.z);
                    let mut l2 = ny * ny + nx * nx + nz * nz;
                    if l2 <= EPS2 {
                        let c = self.cached_axis;
                        nx = c.x;
                        ny = c.y;
                        nz = c.z;
                        l2 = c.x * c.x + c.y * c.y + c.z * c.z;
                    }
                    if l2 <= EPS2 {
                        self.last_method = 9;
                    } else {
                        let k = 1.0 / l2.sqrt();
                        let (dy, dx, dz) = (wa.y - wb.y, wa.x - wb.x, wa.z - wb.z);
                        let len = (dy * dy + dx * dx + dz * dz).sqrt();
                        if !is_valid || -len < distance {
                            is_valid = true;
                            normal = v3(nx * k, ny * k, nz * k);
                            point = wb;
                            distance = -len;
                            self.last_method = 3;
                        } else {
                            self.last_method = 8;
                        }
                    }
                }
            }
            if !is_valid {
                return;
            }
        }
        if 0.0 <= distance && input.max_distance_sqr <= distance * distance {
            return;
        }
        if self.fix_normal {
            if self.b.type_id() == SHAPE_TRIANGLE {
                let Shape::Triangle(tr) = self.b else { unreachable!() };
                let v = &tr.v;
                let e2z = v[2].z - v[0].z;
                let e2y = v[2].y - v[0].y;
                let e2x = v[2].x - v[0].x;
                let e1x = v[1].x - v[0].x;
                let e1z = v[1].z - v[0].z;
                let e1y = v[1].y - v[0].y;
                let ny = e2x * e1z - e2z * e1x;
                let nx = e2z * e1y - e2y * e1z;
                let nz = e2y * e1x - e2x * e1y;
                let wx = rb[0].y * ny + rb[0].x * nx + rb[0].z * nz;
                let wy = rb[1].y * ny + rb[1].x * nx + rb[1].z * nz;
                let wz = rb[2].y * ny + rb[2].x * nx + rb[2].z * nz;
                let k = 1.0 / (wy * wy + wx * wx + wz * wz).sqrt();
                normal = v3(k * wx, k * wy, k * wz);
            } else {
                let (amin, amax) = self.a.aabb(&local_a);
                let (sx, sy, sz) = (amin.x + amax.x, amin.y + amax.y, amin.z + amax.z);
                let (bmin, bmax) = self.b.aabb(&local_b);
                let d = (sy * 0.5 - (bmin.y + bmax.y) * 0.5) * normal.y + (sx * 0.5 - (bmin.x + bmax.x) * 0.5) * normal.x + (sz * 0.5 - (bmin.z + bmax.z) * 0.5) * normal.z;
                if !(0.0 <= d) {
                    normal = v3(normal.x * -1.0, normal.y * -1.0, normal.z * -1.0);
                }
            }
        }
        let pw = v3(point.x + off.x, point.y + off.y, point.z + off.z);
        self.cached_distance = distance;
        self.cached_axis = normal;
        out.add_contact_point(normal, pw, distance);
    }
}
