use crate::bt::manifold::Wrap;
use crate::bt::shapes::{SHAPE_TRIANGLE, Shape};
use crate::bt::types::ManifoldPoint;
use crate::math::crt::{atan2f, cosf, sinf};
use crate::math::{Vec3, v3};
use std::collections::BTreeMap;

const EPS: f32 = f32::from_bits(0x34000000);
const LARGE: f32 = f32::from_bits(0x5d5e0b6b);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TriangleInfo {
    pub flags: u32,
    pub edge0: f32,
    pub edge1: f32,
    pub edge2: f32,
}

#[derive(Clone, Debug)]
pub struct TriangleInfoMap {
    pub map: BTreeMap<u32, TriangleInfo>,
    pub convex_epsilon: f32,
    pub planar_epsilon: f32,
    pub equal_vertex_threshold: f32,
    pub edge_distance_threshold: f32,
    pub max_edge_angle_threshold: f32,
    pub zero_area_threshold: f32,
}

impl Default for TriangleInfoMap {

    fn default() -> TriangleInfoMap {
        TriangleInfoMap {
            map: BTreeMap::new(),
            convex_epsilon: 0.0,
            planar_epsilon: f32::from_bits(0x38d1b717),
            equal_vertex_threshold: f32::from_bits(0x322bcc76),
            edge_distance_threshold: f32::from_bits(0x3dcccccd),
            max_edge_angle_threshold: f32::from_bits(0x40c90fdb),
            zero_area_threshold: f32::from_bits(0x322bcc76),
        }
    }
}

#[inline]
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 3] {

    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [(ax * bw + aw * bx + ay * bz) - az * by, (ay * bw + aw * by + az * bx) - ax * bz, (az * bw + aw * bz + ax * by) - ay * bx]
}

pub fn quat_to_rows(x: f32, y: f32, z: f32, w: f32) -> [Vec3; 3] {
    let d = 2.0 / (y * y + x * x + z * z + w * w);
    let sz = d * z;
    let sy = d * y;
    let xs = d * x * x;
    let wx = d * x * w;
    [
        v3(1.0 - (sz * z + sy * y), sy * x - sz * w, sz * x + sy * w),
        v3(sy * x + sz * w, 1.0 - (sz * z + xs), sz * y - wx),
        v3(sz * x - sy * w, sz * y + wx, 1.0 - (sy * y + xs)),
    ]
}

pub fn clamp_normal(axis: Vec3, edge_normal: Vec3, n: Vec3, correction: f32) -> Option<Vec3> {
    let cx = axis.y * edge_normal.z - axis.z * edge_normal.y;
    let cy = axis.z * edge_normal.x - axis.x * edge_normal.z;
    let cz = axis.x * edge_normal.y - axis.y * edge_normal.x;
    let k = 1.0 / (cy * cy + cx * cx + cz * cz).sqrt();
    let angle = atan2f(k * cy * n.y + k * cx * n.x + k * cz * n.z, edge_normal.x * n.x + edge_normal.y * n.y + edge_normal.z * n.z);
    if (correction < 0.0 && angle < correction) || (0.0 <= correction && correction < angle) {
        let half = (correction - angle) * 0.5;
        let s = sinf(half) / (axis.x * axis.x + axis.y * axis.y + axis.z * axis.z).sqrt();
        let (qx, qy, qz) = (axis.x * s, axis.y * s, axis.z * s);
        let r = quat_to_rows(qx, qy, qz, cosf(half));
        Some(v3(n.x * r[0].x + n.y * r[0].y + n.z * r[0].z, n.x * r[1].x + n.y * r[1].y + n.z * r[1].z, n.x * r[2].x + n.y * r[2].y + n.z * r[2].z))
    } else {
        None
    }
}

pub fn nearest_on_segment(p: Vec3, a: Vec3, b: Vec3, lo: f32, hi: f32) -> Vec3 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let dz = b.z - a.z;
    let l2 = dy * dy + dx * dx + dz * dz;
    if EPS <= l2 {
        let t = ((p.y - a.y) * dy + (p.x - a.x) * dx + (p.z - a.z) * dz) / l2;
        let t = if lo <= t {
            if t <= hi { t } else { hi }
        } else {
            lo
        };
        v3(dx * t + a.x, dy * t + a.y, dz * t + a.z)
    } else {
        a
    }
}

fn point_dist(p: Vec3, q: Vec3) -> f32 {
    let (dy, dx, dz) = (p.y - q.y, p.x - q.x, p.z - q.z);
    (dy * dy + dx * dx + dz * dz).sqrt()
}

fn write_back(cp: &mut ManifoldPoint, w: &Wrap, world_n: Vec3) {
    cp.normal_b = world_n;
    set_b_from_a(cp, w);
}

fn set_b_from_a(cp: &mut ManifoldPoint, w: &Wrap) {
    let d = cp.distance;
    let n = cp.normal_b;
    let by = cp.world_a.y - d * n.y;
    let bz = cp.world_a.z - d * n.z;
    cp.world_b = v3(cp.world_a.x - d * n.x, by, bz);
    let t = w.transform;
    let r = &t.basis.r;
    let (y, x, z) = (by - t.origin.y, cp.world_b.x - t.origin.x, bz - t.origin.z);
    cp.local_b = v3(y * r[1].x + x * r[0].x + z * r[2].x, y * r[1].y + x * r[0].y + z * r[2].y, y * r[1].z + x * r[0].z + z * r[2].z);
}

#[inline]
fn local_to_world_yxz(w: &Wrap, n: Vec3) -> Vec3 {
    let r = &w.transform.basis.r;
    v3(n.y * r[0].y + n.x * r[0].x + n.z * r[0].z, n.y * r[1].y + n.x * r[1].x + n.z * r[1].z, n.y * r[2].y + n.x * r[2].x + n.z * r[2].z)
}

fn current_local_normal(cp: &ManifoldPoint, w: &Wrap) -> Vec3 {
    let r = &w.transform.basis.r;
    let (ny, nx, nz) = (cp.normal_b.y, cp.normal_b.x, cp.normal_b.z);
    v3(ny * r[1].x + nx * r[0].x + nz * r[2].x, nx * r[0].y + ny * r[1].y + nz * r[2].y, nx * r[0].z + ny * r[1].z + nz * r[2].z)
}

pub fn adjust_internal_edge(cp: &mut ManifoldPoint, tri: &Wrap, part: i32, index: i32) {
    if tri.shape.type_id() != SHAPE_TRIANGLE {
        return;
    }
    let Some(map) = tri.info_map else { return };
    let key = (part as u32) << 21 | index as u32;
    let Some(info) = map.map.get(&key).copied() else { return };
    let Shape::Triangle(shape) = tri.shape else { unreachable!() };
    let (v0, v1, v2) = (shape.v[0], shape.v[1], shape.v[2]);
    let tn = shape.normal();
    let (nx, ny, nz) = (tn.x, tn.y, tn.z);
    let lb = cp.local_b;
    let near0 = nearest_on_segment(lb, v0, v1, 0.0, 1.0);

    let r = &tri.transform.basis.r;
    let cn = cp.normal_b;
    let lnx = cn.y * r[1].x + cn.x * r[0].x + cn.z * r[2].x;
    let lny = cn.x * r[0].y + cn.y * r[1].y + cn.z * r[2].y;
    let lnz = cn.x * r[0].z + cn.y * r[1].z + cn.z * r[2].z;
    let k = 1.0 / (lny * lny + lnx * lnx + lnz * lnz).sqrt();
    let (lnx, lny, lnz) = (lnx * k, lny * k, lnz * k);
    let ln = v3(lnx, lny, lnz);

    const E: f32 = f32::from_bits(0x3727c5ac);
    let e_hi = 1.0 - E;
    let max_angle = map.max_edge_angle_threshold;
    let mut best_edge = -1;
    let mut best = LARGE;
    let (mut hit0, mut hit1, mut hit2) = (false, false, false);
    if info.edge0.abs() < max_angle {
        let d = point_dist(lb, nearest_on_segment(lb, v0, v1, E, e_hi));
        if d < LARGE {
            best = d;
            best_edge = 0;
        }
        if d < EPS {
            hit0 = true;
        }
    }
    if info.edge1.abs() < max_angle {
        let d = point_dist(lb, nearest_on_segment(lb, v1, v2, E, e_hi));
        if d < best {
            best_edge = 1;
            best = d;
        }
        if d < EPS {
            hit1 = true;
        }
    }
    if info.edge2.abs() < max_angle {
        let d = point_dist(lb, nearest_on_segment(lb, v2, v0, E, e_hi));
        if d < best {
            best_edge = 2;
        }
        if d < EPS {
            hit2 = true;
        }
    }

    let mut concave = 0;
    let mut touched = false;
    let convex = map.convex_epsilon;
    let edge_dist = map.edge_distance_threshold;

    if info.edge0.abs() < max_angle && point_dist(lb, near0) < edge_dist && (best_edge == 0 || hit0) {
        let (ey, ex, ez) = (v0.y - v1.y, v0.x - v1.x, v0.z - v1.z);
        touched = true;
        if info.edge0 == 0.0 {
            concave = 1;
        } else {
            let s = if info.flags & 1 == 0 { -1.0f32 } else { 1.0 };
            let (snx, sny) = (nx * s, ny * s);
            let half = info.edge0 * 0.5;
            let q = sinf(half) / (ey * ey + ex * ex + ez * ez).sqrt();
            let (qz, qx, qy) = (q * ez, q * ex, q * ey);
            let c = cosf(half);
            let a = [(c * nx + qy * nz) - qz * ny, (qz * nx + c * ny) - qx * nz, (c * nz + qx * ny) - qy * nx, (-qx * nx - qy * ny) - qz * nz];
            let [mut rx, mut ry, mut rz] = quat_mul(a, [-qx, -qy, -qz, c]);
            if info.flags & 8 != 0 {
                rx *= -1.0;
                ry *= -1.0;
                rz *= -1.0;
            }
            if convex <= lny * sny + lnx * snx + lnz * nz * s || convex <= lnx * rx * s + lny * ry * s + lnz * rz * s {
                if let Some(o) = clamp_normal(v3(ex, ey, ez), v3(snx, sny, nz * s), ln, info.edge0) {
                    if 0.0 < nx * o.x + ny * o.y + nz * o.z {
                        write_back(cp, tri, local_to_world_yxz(tri, o));
                    }
                }
            } else {
                concave = 1;
            }
        }
    }

    let near1 = nearest_v1v2(lb, v1, v2);
    if info.edge1.abs() < max_angle && point_dist(lb, near1) < edge_dist && (best_edge == 1 || hit1) {
        let (ex, ey, ez) = (v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
        touched = true;
        if info.edge1 == 0.0 {
            concave += 1;
        } else {
            let s = if info.flags & 2 == 0 { -1.0f32 } else { 1.0 };
            let sny = ny * s;
            let half = info.edge1 * 0.5;
            let q = sinf(half) / (ey * ey + ex * ex + ez * ez).sqrt();
            let (qx, qy, qz) = (ex * q, ey * q, ez * q);
            let c = cosf(half);
            let a = [(nx * c + nz * qy) - ny * qz, (nx * qz + ny * c) - nz * qx, (nz * c + ny * qx) - nx * qy, (nx * -qx - ny * qy) - nz * qz];
            let [mut rx, mut ry, mut rz] = quat_mul(a, [-qx, -qy, -qz, c]);
            if info.flags & 0x10 != 0 {
                rx *= -1.0;
                ry *= -1.0;
                rz *= -1.0;
            }
            if convex <= sny * lny + nx * s * lnx + nz * s * lnz || convex <= lnx * s * rx + lny * ry * s + lnz * rz * s {
                let l = current_local_normal(cp, tri);
                if let Some(o) = clamp_normal(v3(ex, ey, ez), v3(nx * s, sny, nz * s), l, info.edge1) {
                    if 0.0 < nx * o.x + ny * o.y + nz * o.z {
                        write_back(cp, tri, local_to_world_yxz(tri, o));
                    }
                }
            } else {
                concave += 1;
            }
        }
    }

    let near2 = nearest_v2v0(lb, v2, v0);
    if info.edge2.abs() < max_angle && point_dist(lb, near2) < edge_dist && (best_edge == 2 || hit2) {
        let (ez, ex, ey) = (v2.z - v0.z, v2.x - v0.x, v2.y - v0.y);
        if info.edge2 == 0.0 {
            concave += 1;
        } else {
            let s = if info.flags & 4 == 0 { -1.0f32 } else { 1.0 };
            let (snx, snz, sny) = (nx * s, s * nz, s * ny);
            let half = info.edge2 * 0.5;
            let q = sinf(half) / (ey * ey + ex * ex + ez * ez).sqrt();
            let (qx, qy, qz) = (q * ex, q * ey, q * ez);
            let c = cosf(half);
            let a = [(qy * nz + nx * c) - qz * ny, (c * ny + nx * qz) - qx * nz, (c * nz + qx * ny) - nx * qy, (nx * -qx - qy * ny) - qz * nz];
            let [mut rx, mut ry, mut rz] = quat_mul(a, [-qx, -qy, -qz, c]);
            if info.flags & 0x20 != 0 {
                rx *= -1.0;
                ry *= -1.0;
                rz *= -1.0;
            }
            if convex <= sny * lny + snx * lnx + snz * lnz || convex <= lnx * rx * s + lny * ry * s + lnz * rz * s {
                let l = current_local_normal(cp, tri);
                if let Some(o) = clamp_normal(v3(ex, ey, ez), v3(snx, sny, snz), l, info.edge2) {
                    if 0.0 < nx * o.x + ny * o.y + nz * o.z {
                        write_back(cp, tri, local_to_world_yxz(tri, o));
                    }
                }
            } else {
                concave += 1;
            }
        }
    } else if !touched {
        return;
    }

    if concave == 0 {
        return;
    }
    if ny * lny + lnx * nx + nz * lnz < 0.0 {
        return;
    }
    cp.normal_b = local_to_world_yxz(tri, v3(nx, ny, nz));
    set_b_from_a(cp, tri);
}

#[inline]
fn nearest_v1v2(p: Vec3, v1: Vec3, v2: Vec3) -> Vec3 {
    let (ey, ex, ez) = (v2.y - v1.y, v2.x - v1.x, v2.z - v1.z);
    let l2 = ey * ey + ex * ex + ez * ez;
    if EPS <= l2 {
        let t = ((p.x - v1.x) * ex + (p.y - v1.y) * ey + (p.z - v1.z) * ez) / l2;
        let t = if 0.0 <= t {
            if t <= 1.0 { t } else { 1.0 }
        } else {
            0.0
        };
        v3(t * ex + v1.x, t * ey + v1.y, t * ez + v1.z)
    } else {
        v1
    }
}

#[inline]
fn nearest_v2v0(p: Vec3, v2: Vec3, v0: Vec3) -> Vec3 {
    let (ey, ex, ez) = (v0.y - v2.y, v0.x - v2.x, v0.z - v2.z);
    let l2 = ey * ey + ex * ex + ez * ez;
    if EPS <= l2 {
        let t = ((p.y - v2.y) * ey + (p.x - v2.x) * ex + (p.z - v2.z) * ez) / l2;
        let t = if 0.0 <= t {
            if t <= 1.0 { t } else { 1.0 }
        } else {
            0.0
        };
        v3(t * ex + v2.x, t * ey + v2.y, t * ez + v2.z)
    } else {
        v2
    }
}
