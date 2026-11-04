use super::bvh::MeshShape;
use crate::math::{Transform, Vec3};
use std::sync::{Arc, Mutex, OnceLock};

const H: f64 = 0.5;
const INV_H: f32 = 2.0;
const GROW: f32 = 0.01;
const PLANE_REACH: f64 = 0.05;
const MAX_COLUMNS: usize = 64;

#[derive(Debug)]
pub struct ClearGrid {
    lo: [f32; 3],
    n: [usize; 3],

    cols: Vec<u128>,

    sum: Vec<u128>,
    sn1: usize,

    pub stats: (usize, usize, f64),
}

pub enum StaticGeom<'a> {
    Plane { n: [f32; 3], o: [f32; 3] },
    Mesh { mesh: &'a Arc<MeshShape>, t: Transform },
}

#[derive(Clone, PartialEq, Debug)]
enum Key {
    Plane([u32; 6]),
    Mesh(usize, [u32; 12]),
}

fn tbits(t: &Transform) -> [u32; 12] {
    let r = &t.basis.r;
    [
        r[0].x.to_bits(), r[0].y.to_bits(), r[0].z.to_bits(), r[1].x.to_bits(), r[1].y.to_bits(), r[1].z.to_bits(),
        r[2].x.to_bits(), r[2].y.to_bits(), r[2].z.to_bits(), t.origin.x.to_bits(), t.origin.y.to_bits(), t.origin.z.to_bits(),
    ]
}

pub fn shared(statics: &[StaticGeom]) -> Option<Arc<ClearGrid>> {
    type Entry = (Vec<Key>, Vec<Arc<MeshShape>>, Option<Arc<ClearGrid>>);
    static CACHE: OnceLock<Mutex<Vec<Entry>>> = OnceLock::new();
    let key: Vec<Key> = statics
        .iter()
        .map(|s| match s {
            StaticGeom::Plane { n, o } => Key::Plane([n[0].to_bits(), n[1].to_bits(), n[2].to_bits(), o[0].to_bits(), o[1].to_bits(), o[2].to_bits()]),
            StaticGeom::Mesh { mesh, t } => Key::Mesh(Arc::as_ptr(mesh) as usize, tbits(t)),
        })
        .collect();
    let mut cache = CACHE.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, _, g)) = cache.iter().find(|(k, _, _)| *k == key) {
        return g.clone();
    }
    let keep = statics.iter().filter_map(|s| if let StaticGeom::Mesh { mesh, .. } = s { Some((*mesh).clone()) } else { None }).collect();
    let g = ClearGrid::build(statics).map(Arc::new);
    cache.push((key, keep, g.clone()));
    g
}

type V = [f64; 3];
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V, b: V) -> V {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn len(a: V) -> f64 {
    dot(a, a).sqrt()
}

fn tri_box(c: V, e: f64, v: [V; 3]) -> bool {
    let p = [sub(v[0], c), sub(v[1], c), sub(v[2], c)];
    let slack = 1e-9 * (1.0 + e);
    let sep = |ax: V| {
        let (a, b, d) = (dot(p[0], ax), dot(p[1], ax), dot(p[2], ax));
        let r = e * (ax[0].abs() + ax[1].abs() + ax[2].abs());
        a.min(b).min(d) > r + slack * len(ax) || a.max(b).max(d) < -r - slack * len(ax)
    };
    for k in 0..3 {
        let mut ax = [0.0; 3];
        ax[k] = 1.0;
        if sep(ax) {
            return false;
        }
    }
    let ed = [sub(p[1], p[0]), sub(p[2], p[1]), sub(p[0], p[2])];
    if sep(cross(ed[0], ed[1])) {
        return false;
    }
    for k in 0..3 {
        let mut ax = [0.0; 3];
        ax[k] = 1.0;
        for d in &ed {
            let a = cross(ax, *d);
            if dot(a, a) > 0.0 && sep(a) {
                return false;
            }
        }
    }
    true
}

impl ClearGrid {
    fn build(statics: &[StaticGeom]) -> Option<ClearGrid> {

        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        let mut tris: Vec<[V; 3]> = Vec::new();
        for s in statics {
            if let StaticGeom::Mesh { mesh, t } = s {
                let r = &t.basis.r;
                let w = |p: Vec3| -> V {
                    let p = [p.x as f64, p.y as f64, p.z as f64];
                    let row = |k: usize| [r[k].x as f64, r[k].y as f64, r[k].z as f64];
                    [dot(row(0), p) + t.origin.x as f64, dot(row(1), p) + t.origin.y as f64, dot(row(2), p) + t.origin.z as f64]
                };
                for tri in &mesh.triangles {
                    let v = [w(mesh.vertices[tri[0] as usize]), w(mesh.vertices[tri[1] as usize]), w(mesh.vertices[tri[2] as usize])];
                    for p in &v {
                        for a in 0..3 {
                            lo[a] = lo[a].min(p[a]);
                            hi[a] = hi[a].max(p[a]);
                        }
                    }
                    tris.push(v);
                }
            }
        }
        if tris.is_empty() || lo.iter().chain(hi.iter()).any(|x| !x.is_finite() || x.abs() > 1e4) {
            return None;
        }
        let mut n = [0usize; 3];
        let mut glo = [0f32; 3];
        for a in 0..3 {
            let l = ((lo[a] - 2.0) / H).floor() * H;
            let h = ((hi[a] + 2.0) / H).ceil() * H;
            glo[a] = l as f32;
            n[a] = ((h - l) / H).round() as usize;
        }
        if n[2] > 128 || n[0] * n[1] > 4_000_000 {
            return None;
        }
        let mut g = ClearGrid { lo: glo, n, cols: vec![0u128; n[0] * n[1]], sum: Vec::new(), sn1: 0, stats: (0, tris.len(), 0.0) };
        let centre = |a: usize, i: usize| glo[a] as f64 + (i as f64 + 0.5) * H;
        let e = H * 0.5;

        for s in statics {
            if let StaticGeom::Plane { n: pn, o } = s {
                let nv = [pn[0] as f64, pn[1] as f64, pn[2] as f64];
                let nl = len(nv);
                if !(nl > 0.5) {
                    return None;
                }
                let ov = [o[0] as f64, o[1] as f64, o[2] as f64];
                let reach = e * (nv[0].abs() + nv[1].abs() + nv[2].abs()) + PLANE_REACH * nl;
                for x in 0..n[0] {
                    for y in 0..n[1] {
                        let col = &mut g.cols[x * n[1] + y];
                        for z in 0..n[2] {
                            let c = [centre(0, x), centre(1, y), centre(2, z)];
                            if dot(sub(c, ov), nv).abs() <= reach {
                                *col |= 1u128 << z;
                            }
                        }
                    }
                }
            }
        }

        for v in &tris {
            let (e0, e1, e2) = (len(sub(v[1], v[0])), len(sub(v[2], v[1])), len(sub(v[0], v[2])));
            let d = e0.max(e1).max(e2);
            let q = len(cross(sub(v[1], v[0]), sub(v[2], v[0])));
            if !(q >= 1e-4 * d * d) || !(d > 0.0) {
                return None;
            }
            let tau = 0.02 + 2.0 * std::f64::consts::PI * d * d * d / q * (1e-4 + 2e-6 * (d + 3.0) * (d + 3.0) / q);
            if !(tau < 4.0) {
                return None;
            }
            g.stats.2 = g.stats.2.max(tau);
            let mut r = [(0usize, 0usize); 3];
            let mut inside = true;
            for a in 0..3 {
                let l = v[0][a].min(v[1][a]).min(v[2][a]) - tau - e;
                let h = v[0][a].max(v[1][a]).max(v[2][a]) + tau + e;
                let i0 = ((l - glo[a] as f64) / H).floor().max(0.0) as usize;
                let i1 = ((h - glo[a] as f64) / H).floor();
                if i1 < 0.0 || i0 >= n[a] {
                    inside = false;
                }
                r[a] = (i0, (i1 as usize).min(n[a] - 1));
            }
            if !inside {
                continue;
            }
            for x in r[0].0..=r[0].1 {
                for y in r[1].0..=r[1].1 {
                    let col = &mut g.cols[x * n[1] + y];
                    for z in r[2].0..=r[2].1 {
                        if *col >> z & 1 == 0 && tri_box([centre(0, x), centre(1, y), centre(2, z)], e + tau, *v) {
                            *col |= 1u128 << z;
                        }
                    }
                }
            }
        }
        g.stats.0 = g.cols.iter().map(|c| c.count_ones() as usize).sum();
        g.summarize();
        Some(g)
    }

    #[inline]
    pub fn segment_clear(&self, a: Vec3, b: Vec3) -> bool {
        let mut r = [(0usize, 0usize); 3];
        for k in 0..3 {
            let (p, q) = (a[k], b[k]);
            let (l, h) = if p < q { (p, q) } else { (q, p) };
            let l = (l - GROW - self.lo[k]) * INV_H;
            let h = (h + GROW - self.lo[k]) * INV_H;

            if !(l >= 0.0 && h < self.n[k] as f32) {
                return false;
            }
            r[k] = (l as usize, h as usize);
        }
        if (r[0].1 - r[0].0 + 1) * (r[1].1 - r[1].0 + 1) > MAX_COLUMNS {
            return false;
        }
        let zm = (u128::MAX >> (127 - (r[2].1 - r[2].0))) << r[2].0;

        let mut any = false;
        for sx in (r[0].0 >> 2)..=(r[0].1 >> 2) {
            let row = &self.sum[sx * self.sn1..];
            for sy in (r[1].0 >> 2)..=(r[1].1 >> 2) {
                any |= row[sy] & zm != 0;
            }
        }
        if !any {
            return true;
        }
        for x in r[0].0..=r[0].1 {
            let row = &self.cols[x * self.n[1]..];
            for y in r[1].0..=r[1].1 {
                if row[y] & zm != 0 {
                    return false;
                }
            }
        }
        true
    }

    pub fn with_bug(&self) -> ClearGrid {
        let mut cols = self.cols.clone();
        for (i, c) in cols.iter_mut().enumerate() {
            if (i / self.n[1]) % 2 == 0 {
                *c = 0;
            }
        }
        let mut g = ClearGrid { lo: self.lo, n: self.n, cols, sum: Vec::new(), sn1: 0, stats: self.stats };
        g.summarize();
        g
    }

    fn summarize(&mut self) {
        let (s0, s1) = (self.n[0].div_ceil(4), self.n[1].div_ceil(4));
        let mut sum = vec![0u128; s0 * s1];
        for x in 0..self.n[0] {
            for y in 0..self.n[1] {
                sum[(x >> 2) * s1 + (y >> 2)] |= self.cols[x * self.n[1] + y];
            }
        }
        self.sum = sum;
        self.sn1 = s1;
    }

    pub fn extent(&self) -> ([f32; 3], [usize; 3]) {
        (self.lo, self.n)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ClearRef(pub Option<Arc<ClearGrid>>);

impl PartialEq for ClearRef {
    fn eq(&self, o: &ClearRef) -> bool {
        match (&self.0, &o.0) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}
