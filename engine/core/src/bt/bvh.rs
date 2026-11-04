use super::internal_edge::{TriangleInfo, TriangleInfoMap};
use crate::math::crt::{atan2f, cosf, sinf};
use crate::math::{Vec3, v3};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct QNode {
    pub min: [u16; 3],
    pub max: [u16; 3],
    pub escape_or_triangle: i32,
}

#[derive(Clone, Debug)]
pub struct Bvh {
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    pub quantization: Vec3,
    pub cur_node_index: i32,
    pub nodes: Vec<QNode>,

    pub fbox: Vec<[Vec3; 2]>,

    pub grid: LeafGrid,
}

#[derive(Clone, Debug, Default)]
pub struct LeafGrid {

    cols: Vec<u64>,

    sum: Vec<u64>,
}

const GRID_SHIFT: u32 = 10;

impl LeafGrid {
    pub fn build(nodes: &[QNode]) -> LeafGrid {
        let mut cols = vec![0u64; 64 * 64];
        for n in nodes {
            if n.escape_or_triangle < 0 {
                continue;
            }
            let c = |v: u16| (v >> GRID_SHIFT) as usize;
            let zm = Self::zmask(c(n.min[2]), c(n.max[2]));
            for x in c(n.min[0])..=c(n.max[0]) {
                for y in c(n.min[1])..=c(n.max[1]) {
                    cols[x * 64 + y] |= zm;
                }
            }
        }
        let mut g = LeafGrid { cols, sum: Vec::new() };
        g.summarize();
        g
    }

    fn summarize(&mut self) {
        let mut sum = vec![0u64; 16 * 16];
        for (i, c) in self.cols.iter().enumerate() {
            sum[(i / 64 >> 2) * 16 + ((i % 64) >> 2)] |= *c;
        }
        self.sum = sum;
    }

    #[inline(always)]
    fn zmask(a: usize, b: usize) -> u64 {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        (u64::MAX >> (63 - b)) & (u64::MAX << a)
    }

    #[inline]
    pub fn empty(&self, qlo: [u16; 3], qhi: [u16; 3]) -> bool {
        if self.cols.is_empty() {
            return false;
        }
        let r = |a: usize| {
            let (l, h) = if qlo[a] <= qhi[a] { (qlo[a], qhi[a]) } else { (qhi[a], qlo[a]) };
            ((l >> GRID_SHIFT) as usize, (h >> GRID_SHIFT) as usize)
        };
        let ((x0, x1), (y0, y1), (z0, z1)) = (r(0), r(1), r(2));
        let zm = Self::zmask(z0, z1);

        let mut any = false;
        for sx in (x0 >> 2)..=(x1 >> 2) {
            let row = &self.sum[sx * 16..sx * 16 + 16];
            for sy in (y0 >> 2)..=(y1 >> 2) {
                any |= row[sy] & zm != 0;
            }
        }
        if !any {
            return true;
        }
        for x in x0..=x1 {
            let row = &self.cols[x * 64..x * 64 + 64];
            for y in y0..=y1 {
                if row[y] & zm != 0 {
                    return false;
                }
            }
        }
        true
    }

    pub fn plant_bug(&mut self) {
        for (i, c) in self.cols.iter_mut().enumerate() {
            if (i / 64) % 2 == 0 {
                *c = 0;
            }
        }
        self.summarize();
    }
}

impl Bvh {
    #[inline(always)]
    fn qmin(&self, p: Vec3) -> [u16; 3] {
        [
            ((p.x - self.aabb_min.x) * self.quantization.x) as i32 as u16 & 0xfffe,
            ((p.y - self.aabb_min.y) * self.quantization.y) as i32 as u16 & 0xfffe,
            ((p.z - self.aabb_min.z) * self.quantization.z) as i32 as u16 & 0xfffe,
        ]
    }
    #[inline(always)]
    fn qmax(&self, p: Vec3) -> [u16; 3] {
        [
            ((p.x - self.aabb_min.x) * self.quantization.x + 1.0) as i32 as u16 | 1,
            ((p.y - self.aabb_min.y) * self.quantization.y + 1.0) as i32 as u16 | 1,
            ((p.z - self.aabb_min.z) * self.quantization.z + 1.0) as i32 as u16 | 1,
        ]
    }
    #[inline(always)]
    fn unquantize(&self, q: [u16; 3]) -> Vec3 {
        v3(
            q[0] as f32 / self.quantization.x + self.aabb_min.x,
            q[1] as f32 / self.quantization.y + self.aabb_min.y,
            q[2] as f32 / self.quantization.z + self.aabb_min.z,
        )
    }

    fn set_quantization_values(&mut self, bvh_min: Vec3, bvh_max: Vec3, margin: f32) {
        self.aabb_min = v3(bvh_min.x - margin, bvh_min.y - margin, bvh_min.z - margin);
        self.aabb_max = v3(margin + bvh_max.x, margin + bvh_max.y, margin + bvh_max.z);
        let s = 65533.0f32;
        self.quantization = v3(
            s / (self.aabb_max.x - self.aabb_min.x),
            s / (self.aabb_max.y - self.aabb_min.y),
            s / (self.aabb_max.z - self.aabb_min.z),
        );

        let (mn, q) = (self.aabb_min, self.quantization);
        let back_x = ((((mn.x - mn.x) * q.x) as i32 as u16 & 0xfffe) as f32 / q.x + mn.x) - margin;
        let back_y = ((((mn.y - mn.y) * q.y) as i32 as u16 & 0xfffe) as f32 / q.y + mn.y) - margin;
        let back_z = ((((mn.z - mn.z) * q.z) as i32 as u16 & 0xfffe) as f32 / q.z + mn.z) - margin;
        if back_x < self.aabb_min.x {
            self.aabb_min.x = back_x;
        }
        if back_y < self.aabb_min.y {
            self.aabb_min.y = back_y;
        }
        if back_z < self.aabb_min.z {
            self.aabb_min.z = back_z;
        }
        let (mn, mx) = (self.aabb_min, self.aabb_max);
        let up_x = (((mx.x - mn.x) * q.x + 1.0) as i32 as u16 | 1) as f32 / q.x + mn.x + margin;
        let up_y = (((mx.y - mn.y) * q.y + 1.0) as i32 as u16 | 1) as f32 / q.y + mn.y + margin;
        let up_z = (((mx.z - mn.z) * q.z + 1.0) as i32 as u16 | 1) as f32 / q.z + mn.z + margin;
        if self.aabb_max.x < up_x {
            self.aabb_max.x = up_x;
        }
        if self.aabb_max.y < up_y {
            self.aabb_max.y = up_y;
        }
        if self.aabb_max.z < up_z {
            self.aabb_max.z = up_z;
        }
        self.quantization = v3(
            s / (self.aabb_max.x - self.aabb_min.x),
            s / (self.aabb_max.y - self.aabb_min.y),
            s / (self.aabb_max.z - self.aabb_min.z),
        );
    }

    fn leaf(&self, tri: &[Vec3; 3], part: i32, index: i32) -> QNode {
        let (mut lo, mut hi) = (v3(1e18, 1e18, 1e18), v3(-1e18, -1e18, -1e18));
        for v in tri {
            lo.x = if v.x < lo.x { v.x } else { lo.x };
            lo.y = if v.y < lo.y { v.y } else { lo.y };
            lo.z = if v.z < lo.z { v.z } else { lo.z };
            hi.x = if hi.x < v.x { v.x } else { hi.x };
            hi.y = if hi.y < v.y { v.y } else { hi.y };
            hi.z = if hi.z < v.z { v.z } else { hi.z };
        }
        let (small, half) = (f32::from_bits(0x3b03126f), f32::from_bits(0x3a83126f));
        if hi.x - lo.x < small {
            hi.x += half;
            lo.x += -half;
        }
        if hi.y - lo.y < small {
            hi.y += half;
            lo.y += -half;
        }
        if hi.z - lo.z < small {
            hi.z += half;
            lo.z += -half;
        }
        QNode { min: self.qmin(lo), max: self.qmax(hi), escape_or_triangle: (part << 21) | index }
    }

    fn calc_splitting_axis(&self, leaves: &[QNode], start: i32, end: i32) -> usize {
        let mut means = Vec3::ZERO;
        let mut variance = Vec3::ZERO;
        let n = end - start;
        for i in start..end {
            let l = &leaves[i as usize];
            let (lo, hi) = (self.unquantize(l.min), self.unquantize(l.max));
            means.x += (hi.x + lo.x) * 0.5;
            means.y += (hi.y + lo.y) * 0.5;
            means.z += (hi.z + lo.z) * 0.5;
        }
        let f = 1.0 / n as f32;
        for i in start..end {
            let l = &leaves[i as usize];
            let (lo, hi) = (self.unquantize(l.min), self.unquantize(l.max));
            let dx = (lo.x + hi.x) * 0.5 - f * means.x;
            let dy = (hi.y + lo.y) * 0.5 - means.y * f;
            let dz = (hi.z + lo.z) * 0.5 - means.z * f;
            variance.x += dx * dx;
            variance.y += dy * dy;
            variance.z += dz * dz;
        }
        let g = 1.0 / (n as f32 - 1.0);
        if g * variance.y <= variance.x * g {
            if variance.x * g < variance.z * g { 2 } else { 0 }
        } else if g * variance.y < variance.z * g {
            2
        } else {
            1
        }
    }

    fn sort_and_calc_splitting_index(&self, leaves: &mut [QNode], start: i32, end: i32, axis: usize) -> i32 {
        let n = end - start;
        let mut means = Vec3::ZERO;
        for i in start..end {
            let l = &leaves[i as usize];
            let (lo, hi) = (self.unquantize(l.min), self.unquantize(l.max));
            means.x += (lo.x + hi.x) * 0.5;
            means.y += (hi.y + lo.y) * 0.5;
            means.z += (hi.z + lo.z) * 0.5;
        }
        let f = 1.0 / n as f32;
        means.y *= f;
        means.x *= f;
        means.z *= f;
        let split_value = means[axis];
        let mut split = start;
        for i in start..end {
            let l = &leaves[i as usize];
            let (lo, hi) = (self.unquantize(l.min), self.unquantize(l.max));
            let center = v3((lo.x + hi.x) * 0.5, (hi.y + lo.y) * 0.5, (hi.z + lo.z) * 0.5);
            if split_value < center[axis] {
                leaves.swap(i as usize, split as usize);
                split += 1;
            }
        }
        let range = n / 3;
        if range + start < split && split < (end - range) - 1 {
            return split;
        }
        (n >> 1) + start
    }

    fn build_tree(&mut self, leaves: &mut [QNode], start: i32, end: i32) {
        let cur = self.cur_node_index;
        if end - start == 1 {
            self.nodes[cur as usize] = leaves[start as usize];
            self.cur_node_index += 1;
            return;
        }
        let axis = self.calc_splitting_axis(leaves, start, end);
        let split = self.sort_and_calc_splitting_index(leaves, start, end, axis);
        let internal = self.cur_node_index as usize;
        let (mn, mx) = (self.aabb_min, self.aabb_max);
        self.nodes[internal].min = self.qmin(mx);
        self.nodes[internal].max = self.qmax(mn);
        for i in start..end {
            let l = leaves[i as usize];
            let (lo, hi) = (self.unquantize(l.min), self.unquantize(l.max));
            let (qlo, qhi) = (self.qmin(lo), self.qmax(hi));
            let node = &mut self.nodes[internal];
            for a in 0..3 {
                if qlo[a] < node.min[a] {
                    node.min[a] = qlo[a];
                }
                if node.max[a] < qhi[a] {
                    node.max[a] = qhi[a];
                }
            }
        }
        self.cur_node_index += 1;
        self.build_tree(leaves, start, split);
        self.build_tree(leaves, split, end);
        let escape = self.cur_node_index - cur;
        self.nodes[internal].escape_or_triangle = -escape;
    }

    pub fn build(triangles: impl Iterator<Item = [Vec3; 3]>, bvh_min: Vec3, bvh_max: Vec3) -> Bvh {
        let mut b = Bvh { aabb_min: Vec3::ZERO, aabb_max: Vec3::ZERO, quantization: Vec3::ZERO, cur_node_index: 0, nodes: Vec::new(), fbox: Vec::new(), grid: LeafGrid::default() };
        b.set_quantization_values(bvh_min, bvh_max, 1.0);
        let mut leaves: Vec<QNode> = Vec::new();
        for (i, t) in triangles.enumerate() {
            leaves.push(b.leaf(&t, 0, i as i32));
        }
        let n = leaves.len() as i32;
        b.nodes = vec![QNode::default(); leaves.len() * 2];
        b.build_tree(&mut leaves, 0, n);
        let (q, mn) = (b.quantization, b.aabb_min);
        b.fbox = b.nodes[..b.cur_node_index as usize]
            .iter()
            .map(|n| {
                [
                    v3((n.min[0] as f32 / q.x + mn.x) - 0.0, (n.min[1] as f32 / q.y + mn.y) - 0.0, (n.min[2] as f32 / q.z + mn.z) - 0.0),
                    v3((n.max[0] as f32 / q.x + mn.x) - 0.0, (n.max[1] as f32 / q.y + mn.y) - 0.0, (n.max[2] as f32 / q.z + mn.z) - 0.0),
                ]
            })
            .collect();
        b.grid = LeafGrid::build(&b.nodes[..b.cur_node_index as usize]);
        b
    }

    #[inline]
    pub fn quantize_query(&self, lo: Vec3, hi: Vec3) -> ([u16; 3], [u16; 3]) {
        let clamp = |p: Vec3| {
            let (mn, mx) = (self.aabb_min, self.aabb_max);
            let mut c = v3(if p.x < mn.x { mn.x } else { p.x }, if p.y < mn.y { mn.y } else { p.y }, if p.z < mn.z { mn.z } else { p.z });
            c.x = if mx.x < c.x { mx.x } else { c.x };
            c.y = if mx.y < c.y { mx.y } else { c.y };
            c.z = if mx.z < c.z { mx.z } else { c.z };
            c
        };
        (self.qmin(clamp(lo)), self.qmax(clamp(hi)))
    }

    pub fn report_aabb_overlapping_nodes(&self, lo: Vec3, hi: Vec3, f: impl FnMut(i32, i32)) {
        let (qlo, qhi) = self.quantize_query(lo, hi);

        if self.grid.empty(qlo, qhi) {
            return;
        }
        self.walk_quantized(qlo, qhi, f);
    }

    pub fn walk_quantized(&self, qlo: [u16; 3], qhi: [u16; 3], mut f: impl FnMut(i32, i32)) {
        let mut i = 0i32;
        let end = self.cur_node_index;
        while i < end {
            let n = &self.nodes[i as usize];
            let overlap = n.min[2] <= qhi[2]
                && qlo[2] <= n.max[2]
                && qlo[1] <= n.max[1]
                && qlo[0] <= n.max[0]
                && n.min[1] <= qhi[1]
                && n.min[0] <= qhi[0];
            let e = n.escape_or_triangle;
            if e < 0 {
                if overlap {
                    i += 1;
                } else {
                    i += -e;
                }
            } else {
                if overlap {
                    f(e >> 21, e & 0x1f_ffff);
                }
                i += 1;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct MeshShape {

    pub vertices: Vec<Vec3>,

    pub triangles: Vec<[u32; 3]>,
    pub local_aabb_min: Vec3,
    pub local_aabb_max: Vec3,
    pub margin: f32,
    pub bvh: Bvh,
    pub info_map: TriangleInfoMap,
}

impl MeshShape {

    pub fn new(vertices: Vec<Vec3>, triangles: Vec<[u32; 3]>) -> MeshShape {
        let margin = 0.0f32;
        let (mut local_aabb_min, mut local_aabb_max) = (Vec3::ZERO, Vec3::ZERO);
        let support = |dir: Vec3| -> Vec3 {
            let mut best = -1e18f32;
            let mut out = Vec3::ZERO;
            for t in &triangles {
                for k in 0..3 {
                    let v = vertices[t[k] as usize];
                    let d = dir.x * v.x + dir.y * v.y + dir.z * v.z;
                    if best < d {
                        best = d;
                        out = v;
                    }
                }
            }
            out
        };
        for a in 0..3 {
            let mut d = Vec3::ZERO;
            d[a] = 1.0;
            local_aabb_max[a] = margin + support(d)[a];
            d[a] = -1.0;
            local_aabb_min[a] = support(d)[a] - margin;
        }
        let tris = triangles.iter().map(|t| [vertices[t[0] as usize], vertices[t[1] as usize], vertices[t[2] as usize]]);
        let bvh = Bvh::build(tris, local_aabb_min, local_aabb_max);
        let mut m = MeshShape { vertices, triangles, local_aabb_min, local_aabb_max, margin, bvh, info_map: TriangleInfoMap::default() };
        m.generate_internal_edge_info();
        m
    }

    pub fn aabb(&self, t: &crate::math::Transform) -> (Vec3, Vec3) {
        let (lo, hi) = (self.local_aabb_min, self.local_aabb_max);
        let m = self.margin;
        let ex = m + (hi.x - lo.x) * 0.5;
        let ey = (hi.y - lo.y) * 0.5 + m;
        let ez = (hi.z - lo.z) * 0.5 + m;
        let cy = (hi.y + lo.y) * 0.5;
        let cx = (hi.x + lo.x) * 0.5;
        let cz = (lo.z + hi.z) * 0.5;
        let r = &t.basis.r;
        let o = t.origin;
        let wx = cx * r[0].x + cy * r[0].y + cz * r[0].z + o.x;
        let wz = cy * r[2].y + cx * r[2].x + cz * r[2].z + o.z;
        let wy = cy * r[1].y + cx * r[1].x + cz * r[1].z + o.y;
        let dx = r[0].y.abs() * ey + r[0].x.abs() * ex + r[0].z.abs() * ez;
        let dy = r[1].y.abs() * ey + r[1].x.abs() * ex + r[1].z.abs() * ez;
        let dz = r[2].y.abs() * ey + r[2].x.abs() * ex + r[2].z.abs() * ez;
        (v3(wx - dx, wy - dy, wz - dz), v3(dx + wx, dy + wy, dz + wz))
    }

    #[inline(always)]
    pub fn triangle(&self, index: i32) -> [Vec3; 3] {
        let t = self.triangles[index as usize];
        [self.vertices[t[0] as usize], self.vertices[t[1] as usize], self.vertices[t[2] as usize]]
    }
}

#[inline(always)]
fn edge_tri_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3 {
    let z = (v2.y - v0.y) * (v1.x - v0.x) - (v2.x - v0.x) * (v1.y - v0.y);
    let y = (v2.x - v0.x) * (v1.z - v0.z) - (v2.z - v0.z) * (v1.x - v0.x);
    let x = (v2.z - v0.z) * (v1.y - v0.y) - (v2.y - v0.y) * (v1.z - v0.z);
    let k = 1.0 / (x * x + y * y + z * z).sqrt();
    v3(x * k, k * y, k * z)
}

#[inline(always)]
fn rotated_dot(edge: Vec3, half: f32, n: Vec3, other: Vec3) -> f32 {
    let s = sinf(half) / (edge.x * edge.x + edge.y * edge.y + edge.z * edge.z).sqrt();
    let c = cosf(half);
    let (qz, qy, qx) = (s * edge.z, s * edge.y, s * edge.x);
    let w1 = (-qx * n.x - qy * n.y) - qz * n.z;
    let x1 = (qy * n.z + c * n.x) - qz * n.y;
    let y1 = (qz * n.x + c * n.y) - qx * n.z;
    let z1 = (qx * n.y + c * n.z) - qy * n.x;
    let (x2, y2, z2) = (-qx, -qy, -qz);
    let x = (x1 * c + w1 * x2 + y1 * z2) - z1 * y2;
    let y = (y1 * c + w1 * y2 + z1 * x2) - x1 * z2;
    let z = (z1 * c + w1 * z2 + x1 * y2) - y1 * x2;
    y * other.y + x * other.x + z * other.z
}

impl MeshShape {

    fn generate_internal_edge_info(&mut self) {
        for t in 0..self.triangles.len() as i32 {
            let a = self.triangle(t);
            let (mut lo, mut hi) = (v3(1e18, 1e18, 1e18), v3(-1e18, -1e18, -1e18));
            for v in &a {
                lo.x = if v.x < lo.x { v.x } else { lo.x };
                lo.y = if v.y < lo.y { v.y } else { lo.y };
                lo.z = if v.z < lo.z { v.z } else { lo.z };
                hi.x = if hi.x < v.x { v.x } else { hi.x };
                hi.y = if hi.y < v.y { v.y } else { hi.y };
                hi.z = if hi.z < v.z { v.z } else { hi.z };
            }
            let MeshShape { bvh, vertices, triangles, info_map, .. } = self;
            bvh.report_aabb_overlapping_nodes(lo, hi, |part, other| {
                let i = triangles[other as usize];
                let b = [vertices[i[0] as usize], vertices[i[1] as usize], vertices[i[2] as usize]];
                connectivity(info_map, 0, t, &a, part, other, &b);
            });
        }
    }
}

fn connectivity(map: &mut TriangleInfoMap, part_a: i32, tri_a: i32, a: &[Vec3; 3], part_b: i32, tri_b: i32, b: &[Vec3; 3]) {
    if part_a == part_b && tri_a == tri_b {
        return;
    }
    let cross2 = |t: &[Vec3; 3]| {
        let z = (t[2].y - t[0].y) * (t[1].x - t[0].x) - (t[2].x - t[0].x) * (t[1].y - t[0].y);
        let y = (t[2].x - t[0].x) * (t[1].z - t[0].z) - (t[2].z - t[0].z) * (t[1].x - t[0].x);
        let x = (t[2].z - t[0].z) * (t[1].y - t[0].y) - (t[2].y - t[0].y) * (t[1].z - t[0].z);
        x * x + y * y + z * z
    };
    let equal = map.equal_vertex_threshold;
    if cross2(b) < equal || cross2(a) < equal {
        return;
    }
    let (mut sa, mut sb) = ([-1i32; 3], [-1i32; 3]);
    let mut shared = 0usize;
    for i in 0..3 {
        for j in 0..3 {
            let (dx, dy, dz) = (a[i].x - b[j].x, a[i].y - b[j].y, a[i].z - b[j].z);
            if dx * dx + dy * dy + dz * dz < equal {
                sa[shared] = i as i32;
                sb[shared] = j as i32;
                shared += 1;
                if shared > 2 {
                    return;
                }
            }
        }
    }
    if shared != 2 {
        return;
    }

    let (s0, s1, b0, b1);
    if sa[0] == 0 && sa[1] == 2 {
        s0 = 2usize;
        s1 = 0usize;
        b0 = sb[0] as usize;
        b1 = sb[1] as usize;
    } else {
        s0 = sa[0] as usize;
        s1 = sa[1] as usize;
        b0 = sb[1] as usize;
        b1 = sb[0] as usize;
    }
    let key = ((part_a as u32) << 21) | tri_a as u32;
    let sum = s0 + s1;
    let other_a = 3 - sum;
    let other_b = 3 - b0 - b1;
    let e = v3(a[s1].x - a[s0].x, a[s1].y - a[s0].y, a[s1].z - a[s0].z);
    let na = edge_tri_normal(a[0], a[1], a[2]);
    let nb = edge_tri_normal(b[b0], b[b1], b[other_b]);
    let k = 1.0 / (e.x * e.x + e.y * e.y + e.z * e.z).sqrt();
    let e = v3(e.x * k, e.y * k, e.z * k);

    let (cz, cy, cx) = (na.y * e.x - e.y * na.x, e.z * na.x - na.z * e.x, na.z * e.y - na.y * e.z);
    let k = 1.0 / (cx * cx + cy * cy + cz * cz).sqrt();
    let mut eca = v3(k * cx, k * cy, k * cz);
    let t = v3(a[other_a].x - a[s0].x, a[other_a].y - a[s0].y, a[other_a].z - a[s0].z);
    if t.y * eca.y + t.x * eca.x + t.z * eca.z < 0.0 {
        eca = v3(eca.x * -1.0, eca.y * -1.0, eca.z * -1.0);
    }
    let (cz, cy, cx) = (e.x * nb.y - e.y * nb.x, e.z * nb.x - e.x * nb.z, e.y * nb.z - e.z * nb.y);
    let k = 1.0 / (cx * cx + cy * cy + cz * cz).sqrt();
    let mut ecb = v3(k * cx, k * cy, k * cz);
    let t = v3(b[other_b].x - b[b1].x, b[other_b].y - b[b1].y, b[other_b].z - b[b1].z);
    if t.y * ecb.y + t.x * ecb.x + t.z * ecb.z < 0.0 {
        ecb = v3(ecb.x * -1.0, ecb.y * -1.0, ecb.z * -1.0);
    }
    let (ez, ey, ex) = (eca.x * ecb.y - ecb.x * eca.y, eca.z * ecb.x - eca.x * ecb.z, ecb.z * eca.y - ecb.y * eca.z);
    let len2 = ex * ex + ey * ey + ez * ez;
    let mut corrected = 0.0f32;
    let mut convex = false;
    if map.planar_epsilon <= len2 {
        let k = 1.0 / len2.sqrt();
        let (cex, cey, cez) = (k * ex, k * ey, k * ez);
        let nz = cex * eca.y - cey * eca.x;
        let ny = cez * eca.x - cex * eca.z;
        let nx = cey * eca.z - cez * eca.y;
        let k = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt();
        let angle = atan2f(k * ny * ecb.y + k * nx * ecb.x + k * nz * ecb.z, ecb.y * eca.y + ecb.x * eca.x + ecb.z * eca.z);
        corrected = f32::from_bits(0x40490fdb) - angle;
        convex = na.y * ecb.y + na.x * ecb.x + ecb.z * na.z < 0.0;
        if !convex {
            corrected = -corrected;
        }
    }
    let half = -corrected * 0.5;
    let negative = -corrected;
    let info = map.map.entry(key).or_insert_with(TriangleInfo::default);
    match sum {
        1 => {
            let edge = v3(a[0].x - a[1].x, a[0].y - a[1].y, a[0].z - a[1].z);
            if rotated_dot(edge, half, na, nb) < 0.0 {
                info.flags |= 8;
            }
            info.edge0 = negative;
            if convex {
                info.flags |= 1;
            }
        }
        2 => {
            let edge = v3(a[2].x - a[0].x, a[2].y - a[0].y, a[2].z - a[0].z);
            if rotated_dot(edge, half, na, nb) < 0.0 {
                info.flags |= 0x20;
            }
            info.edge2 = negative;
            if convex {
                info.flags |= 4;
            }
        }
        3 => {
            let edge = v3(a[1].x - a[2].x, a[1].y - a[2].y, a[1].z - a[2].z);
            if rotated_dot(edge, half, na, nb) < 0.0 {
                info.flags |= 0x10;
            }
            info.edge1 = negative;
            if convex {
                info.flags |= 2;
            }
        }
        _ => {}
    }
}

pub fn read_cmf(bytes: &[u8]) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let word = |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    let (t, v) = (word(0) as usize, word(4) as usize);
    let mut tris = Vec::with_capacity(t);
    for i in 0..t {
        let at = 8 + i * 12;
        tris.push([word(at), word(at + 4), word(at + 8)]);
    }
    let vo = 8 + t * 12;
    let mut verts = Vec::with_capacity(v);
    for i in 0..v {
        let at = vo + i * 12;
        verts.push(v3(f32::from_bits(word(at)), f32::from_bits(word(at + 4)), f32::from_bits(word(at + 8))));
    }
    (verts, tris)
}



/// The arena's pieces from the data folder: `arena.txt` (one line per mesh file: name, mirror Scale3D,
/// Translation in uu, all as float bits) and `mesh_<i>.cmf` (local vertices in physics units).
///
/// Placement rules, in our own words:
/// - a piece at the origin stays there; a piece translated by +-5120 uu along y (the goal pieces) sits at
///   +-102.39999 physics units (bits 0x42CCCCCB, two float steps under 5120 * 0.02), as measured in the
///   running game. Any other translation is refused rather than guessed.
/// - the basis is the identity, with the zero signs the game's own transform leaves: a piece mirrored in y has
///   -0 at (0,1) and (2,0); one mirrored in x only has -0 at (1,2) and (2,0).
fn read_arena(dir: &std::path::Path) -> std::io::Result<(Vec<Vec<u8>>, Vec<(Vec3, [u32; 9])>)> {
    let bad = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
    let table = std::fs::read_to_string(dir.join("arena.txt"))?;
    let mut files = vec![table.clone().into_bytes()];
    let mut place = Vec::new();
    for line in table.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() != 8 {
            return Err(bad(format!("arena.txt: bad line {line:?}")));
        }
        let i: usize = t[0].parse().map_err(|_| bad(format!("arena.txt: bad index in {line:?}")))?;
        if i != place.len() {
            return Err(bad("arena.txt: lines out of order".into()));
        }
        let w = |k: usize| u32::from_str_radix(t[k], 16).map(f32::from_bits).map_err(|_| bad(format!("arena.txt: bad word in {line:?}")));
        let (sx, sy, tx, ty, tz) = (w(2)?, w(3)?, w(5)?, w(6)?, w(7)?);
        let origin = if tx == 0.0 && tz == 0.0 && ty == 0.0 {
            Vec3::ZERO
        } else if tx == 0.0 && tz == 0.0 && ty.abs() == 5120.0 {
            v3(0.0, f32::from_bits(0x42CC_CCCB).copysign(ty), 0.0)
        } else {
            return Err(bad(format!("arena.txt: piece {i} is placed where catSim has no placement rule")));
        };
        const ONE: u32 = 0x3f80_0000;
        const NEG0: u32 = 0x8000_0000;
        let basis = if sy < 0.0 {
            [ONE, NEG0, 0, 0, ONE, 0, NEG0, 0, ONE]
        } else if sx < 0.0 {
            [ONE, 0, 0, 0, ONE, NEG0, NEG0, 0, ONE]
        } else {
            [ONE, 0, 0, 0, ONE, 0, 0, 0, ONE]
        };
        files.push(std::fs::read(dir.join(format!("mesh_{i}.cmf")))?);
        place.push((origin, basis));
    }
    if place.len() != 16 {
        return Err(bad(format!("arena.txt lists {} pieces, not 16", place.len())));
    }
    let _ = MESH_BASES.set(place.iter().map(|p| p.1).collect());
    Ok((files, place))
}

static MESH_BASES: std::sync::OnceLock<Vec<[u32; 9]>> = std::sync::OnceLock::new();

/// The pieces' bases, in the world's order, once the arena has been read.
pub fn mesh_bases() -> &'static [[u32; 9]] {
    MESH_BASES.get().expect("the arena has not been read yet")
}

pub fn soccar_meshes(dir: &std::path::Path) -> std::io::Result<Vec<(Vec3, std::sync::Arc<MeshShape>)>> {
    use std::sync::{Arc, Mutex, OnceLock};
    type Shared = Vec<(Vec3, Arc<MeshShape>)>;
    static CACHE: OnceLock<Mutex<Vec<(Vec<Vec<u8>>, Shared)>>> = OnceLock::new();
    crate::data::load(dir)?;
    let (files, place) = read_arena(dir)?;
    let mut cache = CACHE.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, meshes)) = cache.iter().find(|(bytes, _)| *bytes == files) {
        return Ok(meshes.clone());
    }
    let mut out = Vec::with_capacity(16);
    for (i, (origin, _)) in place.iter().enumerate() {
        let (verts, tris) = read_cmf(&files[i + 1]);
        out.push((*origin, Arc::new(MeshShape::new(verts, tris))));
    }
    cache.push((files, out.clone()));
    Ok(out)
}
