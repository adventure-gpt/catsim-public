use crate::bt::shapes::Shape;
use crate::math::{Mat3, Transform, Vec3, v3};

const EPS_0001: f32 = f32::from_bits(0x38d1b717);
const PLANE_EPS: f32 = f32::from_bits(0x3727c5ac);
const MAX_VERTICES: usize = 64;
const MAX_FACES: usize = 128;
const NONE: u16 = u16::MAX;

#[derive(Clone, Copy, Debug, Default)]
pub struct Sv {
    pub d: Vec3,
    pub w: Vec3,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EpaResults {
    pub status: i32,
    pub witness: [Vec3; 2],
    pub normal: Vec3,
    pub distance: f32,
}

pub struct MinkowskiDiff<'a> {
    pub a: &'a Shape,
    pub b: &'a Shape,
    pub to1: Mat3,
    pub to0: Mat3,
    pub to0_origin: Vec3,
    pub with_margins: bool,
}

impl<'a> MinkowskiDiff<'a> {

    pub fn new(a: &'a Shape, wa: &Transform, b: &'a Shape, wb: &Transform, with_margins: bool) -> MinkowskiDiff<'a> {
        let (a0, a1, a2) = (wa.basis.r[0], wa.basis.r[1], wa.basis.r[2]);
        let (b0, b1, b2) = (wb.basis.r[0], wb.basis.r[1], wb.basis.r[2]);
        let to1 = Mat3 {
            r: [
                v3(b0.x * a0.x + b1.x * a1.x + b2.x * a2.x, b0.x * a0.y + b1.x * a1.y + b2.x * a2.y, b1.x * a1.z + b0.x * a0.z + b2.x * a2.z),
                v3(b0.y * a0.x + b1.y * a1.x + b2.y * a2.x, b0.y * a0.y + b1.y * a1.y + b2.y * a2.y, b1.y * a1.z + b0.y * a0.z + b2.y * a2.z),
                v3(b0.z * a0.x + b1.z * a1.x + b2.z * a2.x, b0.z * a0.y + b1.z * a1.y + b2.z * a2.y, b1.z * a1.z + b0.z * a0.z + b2.z * a2.z),
            ],
        };
        let vy = wb.origin.y - wa.origin.y;
        let vx = wb.origin.x - wa.origin.x;
        let vz = wb.origin.z - wa.origin.z;
        let to0 = Mat3 {
            r: [
                v3(b1.x * a1.x + b0.x * a0.x + b2.x * a2.x, b1.y * a1.x + b0.y * a0.x + b2.y * a2.x, b1.z * a1.x + b0.z * a0.x + b2.z * a2.x),
                v3(b1.x * a1.y + b0.x * a0.y + b2.x * a2.y, b1.y * a1.y + b0.y * a0.y + b2.y * a2.y, b1.z * a1.y + b0.z * a0.y + b2.z * a2.y),
                v3(b1.x * a1.z + b0.x * a0.z + b2.x * a2.z, b1.y * a1.z + b0.y * a0.z + b2.y * a2.z, b1.z * a1.z + b0.z * a0.z + b2.z * a2.z),
            ],
        };
        let to0_origin = v3(vy * a1.x + vx * a0.x + vz * a2.x, vy * a1.y + vx * a0.y + vz * a2.y, vy * a1.z + vx * a0.z + vz * a2.z);
        MinkowskiDiff { a, b, to1, to0, to0_origin, with_margins }
    }

    #[inline]
    fn ls(&self, s: &Shape, d: Vec3) -> Vec3 {
        if self.with_margins { s.support_with_margin(d) } else { s.support_no_margin(d) }
    }

    pub fn support1(&self, d: Vec3) -> Vec3 {
        let m = &self.to1.r;
        let dd = v3(d.y * m[0].y + d.x * m[0].x + d.z * m[0].z, d.y * m[1].y + d.x * m[1].x + d.z * m[1].z, d.y * m[2].y + d.x * m[2].x + d.z * m[2].z);
        let s = self.ls(self.b, dd);
        let t = &self.to0.r;
        let o = self.to0_origin;
        v3(s.y * t[0].y + s.x * t[0].x + s.z * t[0].z + o.x, s.y * t[1].y + s.x * t[1].x + s.z * t[1].z + o.y, s.y * t[2].y + s.x * t[2].x + s.z * t[2].z + o.z)
    }

    pub fn get_support(&self, d: Vec3) -> Sv {
        let k = 1.0 / (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
        let n = v3(k * d.x, k * d.y, k * d.z);
        let neg = v3(-(k * d.x), -(k * d.y), -(k * d.z));
        let p1 = self.support1(neg);
        let p0 = self.ls(self.a, n);
        Sv { d: n, w: v3(p0.x - p1.x, p0.y - p1.y, p0.z - p1.z) }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SSimplex {
    pub c: [usize; 4],
    pub p: [f32; 4],
    pub rank: u32,
}

pub const GJK_VALID: i32 = 0;
pub const GJK_INSIDE: i32 = 1;
pub const GJK_FAILED: i32 = 2;

pub fn det(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    ((((b.z * a.y * c.x + b.x * a.z * c.y) - b.z * a.x * c.y) - a.y * b.x * c.z) + a.x * b.y * c.z) - a.z * b.y * c.x
}

pub fn project_origin3(a: Vec3, b: Vec3, c: Vec3) -> (f32, [f32; 3], u32) {
    let mut w = [0.0f32; 3];
    let mut m = 0u32;
    let dl0z = a.z - b.z;
    let dl1z = b.z - c.z;
    let dl2z = c.z - a.z;
    let dl0y = a.y - b.y;
    let dl1y = b.y - c.y;
    let dl2y = c.y - a.y;
    let dl0x = a.x - b.x;
    let dl1x = b.x - c.x;
    let dl2x = c.x - a.x;
    let nx = dl1z * dl0y - dl1y * dl0z;
    let ny = dl1x * dl0z - dl1z * dl0x;
    let nz = dl1y * dl0x - dl1x * dl0y;
    let l = ny * ny + nx * nx + nz * nz;
    let mut mind = -1.0f32;
    if l <= 0.0 {
        return (mind, w, m);
    }
    let vt = [a, b, c];
    let dl = [v3(dl0x, dl0y, dl0z), v3(dl1x, dl1y, dl1z), v3(dl2x, dl2y, dl2z)];
    const IMD3: [usize; 3] = [1, 2, 0];

    let (mut use0, mut use1) = (false, false);
    let mut t = 0.0f32;
    let mut w0 = 0.0f32;
    for i in 0..3 {
        let v = vt[i];
        let e = dl[i];
        if 0.0 < (e.z * nx - e.x * nz) * v.y + (e.y * nz - ny * e.z) * v.x + (e.x * ny - e.y * nx) * v.z {
            let j = IMD3[i];
            let o = vt[j];
            let (dx, dy, dz) = (o.x - v.x, o.y - v.y, o.z - v.z);
            let ll = dy * dy + dx * dx + dz * dz;
            let subd;
            if ll <= 0.0 {
                subd = -1.0;
            } else {
                t = -(dx * v.x + dy * v.y + dz * v.z) / ll;
                if t < 1.0 {
                    if 0.0 < t {
                        w0 = 1.0 - t;
                        use0 = true;
                        use1 = true;
                        let px = dx * t + v.x;
                        let pz = dz * t + v.z;
                        let py = dy * t + v.y;
                        subd = py * py + px * px + pz * pz;
                    } else {
                        use0 = true;
                        use1 = false;
                        t = 0.0;
                        w0 = 1.0;
                        subd = v.y * v.y + v.x * v.x + v.z * v.z;
                    }
                } else {
                    use0 = false;
                    use1 = true;
                    t = 1.0;
                    w0 = 0.0;
                    subd = o.y * o.y + o.x * o.x + o.z * o.z;
                }
            }
            if mind < 0.0 || subd < mind {
                m = (if use1 { 1 << j } else { 0 }) + (if use0 { 1 << i } else { 0 });
                w[i] = w0;
                w[j] = t;
                w[IMD3[j]] = 0.0;
                mind = subd;
            }
        }
    }
    if mind < 0.0 {
        let s = l.sqrt();
        m = 7;
        let k = (ny * a.y + nx * a.x + nz * a.z) / l;
        let px = k * nx;
        let py = k * ny;
        let pz = k * nz;
        mind = py * py + px * px + pz * pz;
        let c0 = dl1y * (b.z - pz) - dl1z * (b.y - py);
        let c1 = dl1z * (b.x - px) - dl1x * (b.z - pz);
        let c2 = dl1x * (b.y - py) - dl1y * (b.x - px);
        let s0 = (c1 * c1 + c0 * c0 + c2 * c2).sqrt();
        w[0] = s0 / s;
        let d0 = dl2y * (c.z - pz) - dl2z * (c.y - py);
        let d1 = dl2z * (c.x - px) - dl2x * (c.z - pz);
        let d2 = dl2x * (c.y - py) - dl2y * (c.x - px);
        let s1 = (d1 * d1 + d0 * d0 + d2 * d2).sqrt();
        w[1] = s1 / s;
        w[2] = 1.0 - (s1 / s + s0 / s);
    }
    (mind, w, m)
}

pub fn project_origin4(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> (f32, [f32; 4], u32) {
    let mut w = [0.0f32; 4];
    let mut m = 0u32;
    let dl0y = a.y - d.y;
    let dl0z = a.z - d.z;
    let dl0x = a.x - d.x;
    let dl1z = b.z - d.z;
    let dl1y = b.y - d.y;
    let dl1x = b.x - d.x;
    let dl2y = c.y - d.y;
    let dl2x = c.x - d.x;
    let dl2z = c.z - d.z;
    let bcy = b.y - c.y;
    let bcx = b.x - c.x;
    let bcz = b.z - c.z;
    let vl = ((((dl1x * dl0z * dl2y + dl1z * dl0y * dl2x) - dl1z * dl0x * dl2y) - dl1x * dl0y * dl2z) + dl1y * dl0x * dl2z) - dl1y * dl0z * dl2x;
    let abx = a.x - b.x;
    let aby = a.y - b.y;
    let abz = a.z - b.z;
    let s = ((bcz * abx - bcx * abz) * a.y + (bcy * abz - bcz * aby) * a.x + (bcx * aby - bcy * abx) * a.z) * vl;
    if 0.0 < s || vl.abs() <= 0.0 {
        return (-1.0, w, m);
    }
    let mut mind = -1.0f32;
    let t0 = (dl0y * dl1z - dl0z * dl1y) * d.x;
    if 0.0 < ((dl0z * dl1x - dl0x * dl1z) * d.y + t0 + (dl1y * dl0x - dl1x * dl0y) * d.z) * vl {
        let (subd, sw, sm) = project_origin3(a, b, d);
        mind = subd;
        m = (sm & 2) + (sm & 4) * 2 + (sm & 1);
        w = [sw[0], sw[1], 0.0, sw[2]];
    }
    if 0.0 < ((dl1y * dl2z - dl2y * dl1z) * d.x + (dl2x * dl1z - dl1x * dl2z) * d.y + (dl2y * dl1x - dl1y * dl2x) * d.z) * vl {
        let (subd, sw, sm) = project_origin3(b, c, d);
        if mind < 0.0 || subd < mind {
            m = ((sm & 4) + (sm & 2) + (sm & 1)) * 2;
            w = [0.0, sw[0], sw[1], sw[2]];
            mind = subd;
        }
    }
    if 0.0 < ((dl0x * dl2z - dl0z * dl2x) * d.y + (dl2y * dl0z - dl2z * dl0y) * d.x + (dl2x * dl0y - dl2y * dl0x) * d.z) * vl {
        let (subd, sw, sm) = project_origin3(c, a, d);
        if mind < 0.0 || subd < mind {
            m = (sm >> 1 & 1) + ((sm & 4) + (sm & 1) * 2) * 2;
            w = [sw[1], 0.0, sw[0], sw[2]];
            mind = subd;
        }
    }
    if mind < 0.0 {
        mind = 0.0;
        m = 15;
        let w0 = (((((b.x * c.z * d.y + b.z * c.y * d.x) - b.z * c.x * d.y) - c.y * b.x * d.z) + c.x * b.y * d.z) - b.y * c.z * d.x) / vl;
        let w1 = (((((a.z * c.x * d.y + a.y * c.z * d.x) - a.x * c.z * d.y) - a.y * c.x * d.z) + a.x * c.y * d.z) - a.z * c.y * d.x) / vl;
        let w2 = (((((a.x * b.z * d.y + a.z * b.y * d.x) - a.z * b.x * d.y) - a.x * b.y * d.z) + a.y * b.x * d.z) - a.y * b.z * d.x) / vl;
        w = [w0, w1, w2, 1.0 - (w1 + w0 + w2)];
    }
    (mind, w, m)
}

pub struct Gjk<'a> {
    pub shape: MinkowskiDiff<'a>,
    pub ray: Vec3,
    pub distance: f32,
    pub simplices: [SSimplex; 2],
    pub store: [Sv; 4],
    pub free: [usize; 4],
    pub nfree: u32,
    pub current: usize,
    pub simplex: usize,
    pub status: i32,
}

impl<'a> Gjk<'a> {
    pub fn new(shape: MinkowskiDiff<'a>) -> Gjk<'a> {
        Gjk { shape, ray: Vec3::ZERO, distance: 0.0, simplices: [SSimplex::default(); 2], store: [Sv::default(); 4], free: [0; 4], nfree: 0, current: 0, simplex: 0, status: GJK_FAILED }
    }

    fn append_vertice(&mut self, si: usize, v: Vec3) {
        let rank = self.simplices[si].rank as usize;
        self.simplices[si].p[rank] = 0.0;
        self.nfree -= 1;
        let c = self.free[self.nfree as usize];
        self.simplices[si].c[rank] = c;
        self.simplices[si].rank += 1;
        self.store[c] = self.shape.get_support(v);
    }

    fn remove_vertice(&mut self, si: usize) {
        self.simplices[si].rank -= 1;
        let rank = self.simplices[si].rank as usize;
        self.free[self.nfree as usize] = self.simplices[si].c[rank];
        self.nfree += 1;
    }

    pub fn evaluate(&mut self, guess: Vec3) -> i32 {
        let mut alpha = 0.0f32;
        self.free = [0, 1, 2, 3];
        self.nfree = 4;
        self.status = GJK_VALID;
        self.distance = 0.0;
        self.simplices[0].rank = 0;
        self.ray = guess;
        let r = self.ray;
        let sqrl = r.x * r.x + r.y * r.y + r.z * r.z;
        let first = if sqrl <= 0.0 { v3(1.0, 0.0, 0.0) } else { v3(-r.x, -r.y, -r.z) };
        self.append_vertice(0, first);
        self.simplices[0].p[0] = 1.0;
        self.ray = self.store[self.simplices[0].c[0]].w;
        let mut lastw = [self.ray; 4];
        let mut clastw = 0usize;
        let mut iterations = 0u32;
        let mut sqdist = 0.0f32;
        loop {
            let cur = self.current;
            let next = 1 - cur;
            let ray = self.ray;
            let rl = (ray.x * ray.x + ray.y * ray.y + ray.z * ray.z).sqrt();
            if rl < EPS_0001 {
                self.status = GJK_INSIDE;
                break;
            }
            self.append_vertice(cur, v3(-ray.x, -ray.y, -ray.z));
            let rank = self.simplices[cur].rank as usize;
            let w = self.store[self.simplices[cur].c[rank - 1]].w;
            let mut found = false;
            for l in &lastw {
                let (dy, dx, dz) = (w.y - l.y, w.x - l.x, w.z - l.z);
                if dy * dy + dx * dx + dz * dz < EPS_0001 {
                    found = true;
                    break;
                }
            }
            if found {
                self.remove_vertice(self.current);
                break;
            }
            clastw = (clastw + 1) & 3;
            lastw[clastw] = w;
            let omega = (w.y * ray.y + w.x * ray.x + w.z * ray.z) / rl;
            alpha = if omega <= alpha { alpha } else { omega };
            if (rl - alpha) - rl * EPS_0001 <= 0.0 {
                self.remove_vertice(self.current);
                break;
            }
            let mut weights = [0.0f32; 4];
            let mut mask = 0u32;
            let s = self.simplices[cur];
            match s.rank {
                2 => {
                    let (a, b) = (self.store[s.c[0]].w, self.store[s.c[1]].w);
                    let (dx, dy, dz) = (b.x - a.x, b.y - a.y, b.z - a.z);
                    let l = dy * dy + dx * dx + dz * dz;
                    if l <= 0.0 {
                        sqdist = -1.0;
                    } else {
                        let t = -(a.y * dy + a.x * dx + a.z * dz) / l;
                        if t < 1.0 {
                            if 0.0 < t {
                                mask = 3;
                                let px = dx * t + a.x;
                                let pz = dz * t + a.z;
                                weights[0] = 1.0 - t;
                                weights[1] = t;
                                let py = dy * t + a.y;
                                sqdist = py * py + px * px + pz * pz;
                            } else {
                                mask = 1;
                                weights[0] = 1.0;
                                sqdist = a.y * a.y + a.x * a.x + a.z * a.z;
                            }
                        } else {
                            mask = 2;
                            weights[1] = 1.0;
                            sqdist = b.y * b.y + b.x * b.x + b.z * b.z;
                        }
                    }
                }
                3 => {
                    let (d, sw, m) = project_origin3(self.store[s.c[0]].w, self.store[s.c[1]].w, self.store[s.c[2]].w);
                    sqdist = d;
                    weights = [sw[0], sw[1], sw[2], 0.0];
                    mask = m;
                }
                4 => {
                    let (d, sw, m) = project_origin4(self.store[s.c[0]].w, self.store[s.c[1]].w, self.store[s.c[2]].w, self.store[s.c[3]].w);
                    sqdist = d;
                    weights = sw;
                    mask = m;
                }
                _ => {}
            }
            if sqdist < 0.0 {
                self.remove_vertice(self.current);
                break;
            }
            self.simplices[next].rank = 0;
            self.ray = Vec3::ZERO;
            self.current = next;
            for i in 0..s.rank as usize {
                if mask & (1 << i) != 0 {
                    let wt = weights[i];
                    let nr = self.simplices[next].rank as usize;
                    self.simplices[next].c[nr] = s.c[i];
                    self.simplices[next].p[nr] = wt;
                    self.simplices[next].rank += 1;
                    let v = self.store[s.c[i]].w;
                    self.ray.y = wt * v.y + self.ray.y;
                    self.ray.z = wt * v.z + self.ray.z;
                    self.ray.x = wt * v.x + self.ray.x;
                } else {
                    self.free[self.nfree as usize] = s.c[i];
                    self.nfree += 1;
                }
            }
            if mask == 0xf {
                self.status = GJK_INSIDE;
            }
            iterations += 1;
            if 127 < iterations {
                self.status = GJK_FAILED;
                break;
            }
            if self.status != GJK_VALID {
                break;
            }
        }
        self.simplex = self.current;
        if self.status == GJK_VALID {
            let r = self.ray;
            self.distance = (r.x * r.x + r.y * r.y + r.z * r.z).sqrt();
        } else if self.status == GJK_INSIDE {
            self.distance = 0.0;
        }
        self.status
    }

    pub fn enclose_origin(&mut self) -> bool {
        let si = self.simplex;
        let s = self.simplices[si];
        match s.rank {
            1 => {
                for i in 0..3 {
                    let mut axis = Vec3::ZERO;
                    axis[i] = 1.0;
                    self.append_vertice(si, axis);
                    if self.enclose_origin() {
                        return true;
                    }
                    self.remove_vertice(si);
                    self.append_vertice(si, v3(-axis.x, -axis.y, -axis.z));
                    if self.enclose_origin() {
                        return true;
                    }
                    self.remove_vertice(si);
                }
                false
            }
            2 => {
                let (c0, c1) = (self.store[s.c[0]].w, self.store[s.c[1]].w);
                let (dz, dy, dx) = (c1.z - c0.z, c1.y - c0.y, c1.x - c0.x);
                for i in 0..3 {
                    let mut axis = Vec3::ZERO;
                    axis[i] = 1.0;
                    let px = axis.z * dy - axis.y * dz;
                    let pz = axis.y * dx - axis.x * dy;
                    let py = axis.x * dz - axis.z * dx;
                    if 0.0 < py * py + px * px + pz * pz {
                        self.append_vertice(si, v3(px, py, pz));
                        if self.enclose_origin() {
                            return true;
                        }
                        self.remove_vertice(si);
                        self.append_vertice(si, v3(-px, -py, -pz));
                        if self.enclose_origin() {
                            return true;
                        }
                        self.remove_vertice(si);
                    }
                }
                false
            }
            3 => {
                let (c0, c1, c2) = (self.store[s.c[0]].w, self.store[s.c[1]].w, self.store[s.c[2]].w);
                let (e2z, e2y, e2x) = (c2.z - c0.z, c2.y - c0.y, c2.x - c0.x);
                let (e1z, e1x, e1y) = (c1.z - c0.z, c1.x - c0.x, c1.y - c0.y);
                let nx = e1y * e2z - e1z * e2y;
                let ny = e1z * e2x - e1x * e2z;
                let nz = e1x * e2y - e1y * e2x;
                if 0.0 < ny * ny + nx * nx + nz * nz {
                    self.append_vertice(si, v3(nx, ny, nz));
                    if self.enclose_origin() {
                        return true;
                    }
                    self.remove_vertice(si);
                    self.append_vertice(si, v3(-nx, -ny, -nz));
                    if self.enclose_origin() {
                        return true;
                    }
                    self.remove_vertice(si);
                }
                false
            }
            4 => {
                let (c0, c1, c2, c3) = (self.store[s.c[0]].w, self.store[s.c[1]].w, self.store[s.c[2]].w, self.store[s.c[3]].w);
                let a = v3(c0.x - c3.x, c0.y - c3.y, c0.z - c3.z);
                let b = v3(c1.x - c3.x, c1.y - c3.y, c1.z - c3.z);
                let c = v3(c2.x - c3.x, c2.y - c3.y, c2.z - c3.z);
                0.0 < det(a, b, c).abs()
            }
            _ => false,
        }
    }
}

pub const EPA_VALID: i32 = 0;
pub const EPA_DEGENERATED: i32 = 2;
pub const EPA_NONCONVEX: i32 = 3;
pub const EPA_INVALID_HULL: i32 = 4;
pub const EPA_OUT_OF_FACES: i32 = 5;
pub const EPA_OUT_OF_VERTICES: i32 = 6;
pub const EPA_ACCURACY_REACHED: i32 = 7;
pub const EPA_FALLBACK: i32 = 8;
pub const EPA_FAILED: i32 = 9;

#[derive(Clone, Copy, Debug)]
struct Face {
    n: Vec3,
    d: f32,
    c: [usize; 3],
    f: [u16; 3],
    l: [u16; 2],
    e: [u8; 3],
    pass: u8,
}

const EMPTY_FACE: Face = Face { n: Vec3::ZERO, d: 0.0, c: [0; 3], f: [NONE; 3], l: [NONE; 2], e: [0; 3], pass: 0 };

#[derive(Clone, Copy)]
struct List {
    root: u16,
    count: u32,
}

struct Horizon {
    cf: u16,
    ff: u16,
    nf: u32,
}

pub struct Epa {
    pub status: i32,
    pub result: SSimplex,
    pub normal: Vec3,
    pub depth: f32,
    sv_store: [Sv; MAX_VERTICES],
    fc_store: [Face; MAX_FACES],
    nextsv: u32,
    hull: List,
    stock: List,
}

impl Default for Epa {
    fn default() -> Epa {
        Epa::new()
    }
}

impl Epa {

    pub fn new() -> Epa {
        let mut e = Epa {
            status: EPA_FAILED,
            result: SSimplex::default(),
            normal: Vec3::ZERO,
            depth: 0.0,
            sv_store: [Sv::default(); MAX_VERTICES],
            fc_store: [EMPTY_FACE; MAX_FACES],
            nextsv: 0,
            hull: List { root: NONE, count: 0 },
            stock: List { root: NONE, count: 0 },
        };
        for i in 0..MAX_FACES {
            let idx = (MAX_FACES - 1 - i) as u16;
            e.append_stock(idx);
        }
        e
    }

    fn remove_hull(&mut self, f: u16) {
        let (l0, l1) = (self.fc_store[f as usize].l[0], self.fc_store[f as usize].l[1]);
        if l1 != NONE {
            self.fc_store[l1 as usize].l[0] = l0;
        }
        if l0 != NONE {
            self.fc_store[l0 as usize].l[1] = l1;
        }
        if f == self.hull.root {
            self.hull.root = l1;
        }
        self.hull.count -= 1;
    }

    fn remove_stock(&mut self, f: u16) {
        let (l0, l1) = (self.fc_store[f as usize].l[0], self.fc_store[f as usize].l[1]);
        if l1 != NONE {
            self.fc_store[l1 as usize].l[0] = l0;
        }
        if l0 != NONE {
            self.fc_store[l0 as usize].l[1] = l1;
        }
        if f == self.stock.root {
            self.stock.root = l1;
        }
        self.stock.count -= 1;
    }

    fn append_hull(&mut self, f: u16) {
        self.fc_store[f as usize].l[0] = NONE;
        self.fc_store[f as usize].l[1] = self.hull.root;
        if self.hull.root != NONE {
            self.fc_store[self.hull.root as usize].l[0] = f;
        }
        self.hull.root = f;
        self.hull.count += 1;
    }

    fn append_stock(&mut self, f: u16) {
        self.fc_store[f as usize].l[0] = NONE;
        self.fc_store[f as usize].l[1] = self.stock.root;
        if self.stock.root != NONE {
            self.fc_store[self.stock.root as usize].l[0] = f;
        }
        self.stock.root = f;
        self.stock.count += 1;
    }

    fn bind(&mut self, fa: u16, ea: u8, fb: u16, eb: u8) {
        self.fc_store[fa as usize].e[ea as usize] = eb;
        self.fc_store[fa as usize].f[ea as usize] = fb;
        self.fc_store[fb as usize].e[eb as usize] = ea;
        self.fc_store[fb as usize].f[eb as usize] = fa;
    }

    #[inline]
    fn sv(&self, gjk: &Gjk, i: usize) -> Sv {
        if i < 4 { gjk.store[i] } else { self.sv_store[i - 4] }
    }

    fn edge_dist(&self, n: Vec3, a: Vec3, b: Vec3) -> Option<f32> {
        let bax = b.x - a.x;
        let bay = b.y - a.y;
        let baz = b.z - a.z;
        if (baz * n.x - bax * n.z) * a.y + (bay * n.z - baz * n.y) * a.x + (bax * n.y - bay * n.x) * a.z < 0.0 {
            let d = if a.x * bax + a.y * bay + a.z * baz <= 0.0 {
                let b2 = b.y * b.y + b.x * b.x + b.z * b.z;
                if b.x * bax + b.y * bay + b.z * baz < 0.0 {
                    return Some(b2.sqrt());
                }
                let ab = a.y * b.y + a.x * b.x + a.z * b.z;
                let d2 = ((a.y * a.y + a.x * a.x + a.z * a.z) * b2 - ab * ab) / (bax * bax + bay * bay + baz * baz);
                if d2 <= 0.0 { 0.0 } else { d2 }
            } else {
                a.y * a.y + a.x * a.x + a.z * a.z
            };
            return Some(d.sqrt());
        }
        None
    }

    fn new_face(&mut self, gjk: &Gjk, a: usize, b: usize, c: usize, forced: bool) -> u16 {
        let fi = self.stock.root;
        if fi == NONE {
            self.status = EPA_OUT_OF_FACES;
            return NONE;
        }
        self.remove_stock(fi);
        self.append_hull(fi);
        let (wa, wb, wc) = (self.sv(gjk, a).w, self.sv(gjk, b).w, self.sv(gjk, c).w);
        {
            let f = &mut self.fc_store[fi as usize];
            f.pass = 0;
            f.c = [a, b, c];
        }
        let (e1x, e1z, e1y) = (wb.x - wa.x, wb.z - wa.z, wb.y - wa.y);
        let (e2y, e2z, e2x) = (wc.y - wa.y, wc.z - wa.z, wc.x - wa.x);
        let nx = e1y * e2z - e1z * e2y;
        let nz = e1x * e2y - e1y * e2x;
        let ny = e1z * e2x - e1x * e2z;
        self.fc_store[fi as usize].n = v3(nx, ny, nz);
        let l = (ny * ny + nx * nx + nz * nz).sqrt();
        if l <= EPS_0001 {
            self.status = EPA_DEGENERATED;
        } else {
            let n = v3(nx, ny, nz);
            let d = match self.edge_dist(n, wa, wb).or_else(|| self.edge_dist(n, wb, wc)).or_else(|| self.edge_dist(n, wc, wa)) {
                Some(d) => d,
                None => (ny * wa.y + wa.x * nx + nz * wa.z) / l,
            };
            self.fc_store[fi as usize].d = d;
            let r = 1.0 / l;
            self.fc_store[fi as usize].n = v3(r * nx, r * ny, r * nz);
            if forced || -PLANE_EPS <= d {
                return fi;
            }
            self.status = EPA_NONCONVEX;
        }
        self.remove_hull(fi);
        self.append_stock(fi);
        NONE
    }

    fn find_best(&self) -> u16 {
        let mut best = self.hull.root;
        let mut mind = self.fc_store[best as usize].d * self.fc_store[best as usize].d;
        let mut f = self.fc_store[best as usize].l[1];
        while f != NONE {
            let d = self.fc_store[f as usize].d;
            let sq = d * d;
            if !(mind <= sq) {
                best = f;
            }
            if sq < mind {
                mind = sq;
            }
            f = self.fc_store[f as usize].l[1];
        }
        best
    }

    fn expand(&mut self, gjk: &Gjk, pass: u32, w: usize, f: u16, e: u8, horizon: &mut Horizon) -> bool {
        const I1M3: [usize; 3] = [1, 2, 0];
        const I2M3: [usize; 3] = [2, 0, 1];
        let face = self.fc_store[f as usize];
        if face.pass as u32 != pass {
            let e1 = I1M3[e as usize];
            let ww = self.sv(gjk, w).w;
            if -PLANE_EPS <= (ww.y * face.n.y + face.n.x * ww.x + ww.z * face.n.z) - face.d {
                let e2 = I2M3[e as usize];
                self.fc_store[f as usize].pass = pass as u8;
                if self.expand(gjk, pass, w, face.f[e1], face.e[e1], horizon) && self.expand(gjk, pass, w, face.f[e2], face.e[e2], horizon) {
                    self.remove_hull(f);
                    self.append_stock(f);
                    return true;
                }
            } else {
                let nf = self.new_face(gjk, face.c[e1], face.c[e as usize], w, false);
                if nf != NONE {
                    self.bind(nf, 0, f, e);
                    if horizon.cf != NONE {
                        self.bind(horizon.cf, 1, nf, 2);
                    } else {
                        horizon.ff = nf;
                    }
                    horizon.cf = nf;
                    horizon.nf += 1;
                    return true;
                }
            }
        }
        false
    }

    pub fn evaluate(&mut self, gjk: &mut Gjk, guess: Vec3) -> i32 {
        if gjk.simplices[gjk.simplex].rank > 1 && gjk.enclose_origin() {
            while self.hull.root != NONE {
                let f = self.hull.root;
                self.remove_hull(f);
                self.append_stock(f);
            }
            self.status = EPA_VALID;
            self.nextsv = 0;
            let si = gjk.simplex;
            let (w0, w1, w2, w3) = (
                gjk.store[gjk.simplices[si].c[0]].w,
                gjk.store[gjk.simplices[si].c[1]].w,
                gjk.store[gjk.simplices[si].c[2]].w,
                gjk.store[gjk.simplices[si].c[3]].w,
            );
            let a = v3(w0.x - w3.x, w0.y - w3.y, w0.z - w3.z);
            let b = v3(w1.x - w3.x, w1.y - w3.y, w1.z - w3.z);
            let c = v3(w2.x - w3.x, w2.y - w3.y, w2.z - w3.z);
            if det(a, b, c) < 0.0 {
                gjk.simplices[si].c.swap(0, 1);
                gjk.simplices[si].p.swap(0, 1);
            }
            let sc = gjk.simplices[si].c;
            let t0 = self.new_face(gjk, sc[0], sc[1], sc[2], true);
            let t1 = self.new_face(gjk, sc[1], sc[0], sc[3], true);
            let t2 = self.new_face(gjk, sc[2], sc[1], sc[3], true);
            let t3 = self.new_face(gjk, sc[0], sc[2], sc[3], true);
            if self.hull.count == 4 {
                let mut best = self.find_best();
                let mut outer = self.fc_store[best as usize];
                let mut pass = 0u32;
                let mut iterations = 0u32;
                self.bind(t0, 0, t1, 0);
                self.bind(t0, 1, t2, 0);
                self.bind(t0, 2, t3, 0);
                self.bind(t1, 1, t3, 2);
                self.bind(t1, 2, t2, 1);
                self.bind(t2, 2, t3, 1);
                self.status = EPA_VALID;
                loop {
                    if self.nextsv as usize >= MAX_VERTICES {
                        self.status = EPA_OUT_OF_VERTICES;
                        break;
                    }
                    pass += 1;
                    let wi = self.nextsv as usize;
                    self.nextsv += 1;
                    let mut horizon = Horizon { cf: NONE, ff: NONE, nf: 0 };
                    self.fc_store[best as usize].pass = pass as u8;
                    let bf = self.fc_store[best as usize];
                    self.sv_store[wi] = gjk.shape.get_support(bf.n);
                    let w = self.sv_store[wi].w;
                    if (w.y * bf.n.y + w.x * bf.n.x + w.z * bf.n.z) - bf.d <= EPS_0001 {
                        self.status = EPA_ACCURACY_REACHED;
                        break;
                    }
                    let mut valid = true;
                    for j in 0..3 {
                        if !valid {
                            break;
                        }
                        let (ff, fe) = (self.fc_store[best as usize].f[j], self.fc_store[best as usize].e[j]);
                        valid = self.expand(gjk, pass, wi + 4, ff, fe, &mut horizon);
                    }
                    if !valid || horizon.nf < 3 {
                        self.status = EPA_INVALID_HULL;
                        break;
                    }
                    self.bind(horizon.cf, 1, horizon.ff, 2);
                    self.remove_hull(best);
                    self.append_stock(best);
                    best = self.find_best();
                    outer = self.fc_store[best as usize];
                    iterations += 1;
                    if iterations >= 255 {
                        break;
                    }
                }
                self.normal = outer.n;
                self.depth = outer.d;
                self.result.rank = 3;
                self.result.c[0] = outer.c[0];
                self.result.c[1] = outer.c[1];
                self.result.c[2] = outer.c[2];
                let dd = outer.d;
                let py = outer.n.y * dd;
                let pz = outer.n.z * dd;
                let px = outer.n.x * dd;
                let (c0, c1, c2) = (self.sv(gjk, outer.c[0]).w, self.sv(gjk, outer.c[1]).w, self.sv(gjk, outer.c[2]).w);
                let area = |u: Vec3, v: Vec3| {
                    let (ay, az, ax) = (u.y - py, u.z - pz, u.x - px);
                    let (by, bz, bx) = (v.y - py, v.z - pz, v.x - px);
                    let cx = ay * bz - az * by;
                    let cy = az * bx - ax * bz;
                    let cz = ax * by - ay * bx;
                    (cy * cy + cx * cx + cz * cz).sqrt()
                };
                let s0 = area(c1, c2);
                let s1 = area(c2, c0);
                let s2 = area(c0, c1);
                let sum = s1 + s0 + s2;
                self.result.p[0] = s0 / sum;
                self.result.p[1] = s1 / sum;
                self.result.p[2] = s2 / sum;
                return self.status;
            }
        }
        self.status = EPA_FALLBACK;
        let (nx, ny, nz) = (-guess.x, -guess.y, -guess.z);
        let nl = (nx * nx + ny * ny + nz * nz).sqrt();
        self.normal = if 0.0 < nl {
            let k = 1.0 / nl;
            v3(k * nx, ny * k, k * nz)
        } else {
            v3(1.0, 0.0, 0.0)
        };
        self.depth = 0.0;
        self.result.rank = 1;
        self.result.c[0] = gjk.simplices[gjk.simplex].c[0];
        self.result.p[0] = 1.0;
        self.status
    }
}

pub fn penetration(a: &Shape, wa: &Transform, b: &Shape, wb: &Transform, guess: Vec3, with_margins: bool) -> (bool, EpaResults) {
    let shape = MinkowskiDiff::new(a, wa, b, wb, with_margins);
    let mut res = EpaResults::default();
    let mut gjk = Gjk::new(shape);
    let neg = v3(-guess.x, -guess.y, -guess.z);
    let st = gjk.evaluate(neg);
    if st == GJK_INSIDE {
        let mut epa = Box::new(Epa::new());
        let es = epa.evaluate(&mut gjk, neg);
        if es != EPA_FAILED {
            let (mut wx, mut wy, mut wz) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..epa.result.rank as usize {
                let ci = epa.result.c[i];
                let d = epa.sv(&gjk, ci).d;
                let s = gjk.shape.ls(gjk.shape.a, d);
                let p = epa.result.p[i];
                wx = wx + s.x * p;
                wy = wy + s.y * p;
                wz = wz + s.z * p;
            }
            res.status = 1;
            let r = &wa.basis.r;
            let o = wa.origin;
            res.witness[0] = v3(wy * r[0].y + wx * r[0].x + wz * r[0].z + o.x, wx * r[1].x + wy * r[1].y + wz * r[1].z + o.y, wy * r[2].y + wx * r[2].x + wz * r[2].z + o.z);
            let (ux, uy, uz) = (wx - epa.normal.x * epa.depth, wy - epa.normal.y * epa.depth, wz - epa.normal.z * epa.depth);
            res.witness[1] = v3(uy * r[0].y + ux * r[0].x + uz * r[0].z + o.x, ux * r[1].x + uy * r[1].y + uz * r[1].z + o.y, uy * r[2].y + ux * r[2].x + uz * r[2].z + o.z);
            res.normal = v3(-epa.normal.x, -epa.normal.y, -epa.normal.z);
            res.distance = -epa.depth;
            return (true, res);
        }
        res.status = 3;
    } else if st == GJK_FAILED {
        res.status = 2;
    }
    (false, res)
}

pub fn distance(a: &Shape, wa: &Transform, b: &Shape, wb: &Transform, guess: Vec3) -> (bool, EpaResults) {
    let shape = MinkowskiDiff::new(a, wa, b, wb, false);
    let mut res = EpaResults::default();
    let mut gjk = Gjk::new(shape);
    let st = gjk.evaluate(guess);
    if st != GJK_VALID {
        res.status = if st != GJK_INSIDE { 2 } else { 1 };
        return (false, res);
    }
    let (mut w0x, mut w0y, mut w0z, mut w1x, mut w1y, mut w1z) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let s = gjk.simplices[gjk.simplex];
    for i in 0..s.rank as usize {
        let p = s.p[i];
        let d = gjk.store[s.c[i]].d;
        let sa = gjk.shape.ls(gjk.shape.a, d);
        w0x = w0x + sa.x * p;
        w0y = w0y + sa.y * p;
        w0z = w0z + sa.z * p;
        let sb = gjk.shape.support1(v3(-d.x, -d.y, -d.z));
        w1x = w1x + sb.x * p;
        w1y = w1y + sb.y * p;
        w1z = w1z + sb.z * p;
    }
    let r = &wa.basis.r;
    let o = wa.origin;
    res.witness[0] = v3(w0x * r[0].x + w0y * r[0].y + w0z * r[0].z + o.x, w0x * r[1].x + w0y * r[1].y + w0z * r[1].z + o.y, w0x * r[2].x + w0y * r[2].y + w0z * r[2].z + o.z);
    res.witness[1] = v3(w1x * r[0].x + w1y * r[0].y + w1z * r[0].z + o.x, w1x * r[1].x + w1y * r[1].y + w1z * r[1].z + o.y, w1x * r[2].x + w1y * r[2].y + w1z * r[2].z + o.z);
    let (nx, ny, nz) = (w0x - w1x, w0y - w1y, w0z - w1z);
    let l = (nx * nx + ny * ny + nz * nz).sqrt();
    res.distance = l;
    let k = 1.0 / (if l <= EPS_0001 { 1.0 } else { l });
    res.normal = v3(nx * k, k * ny, k * nz);
    (true, res)
}
