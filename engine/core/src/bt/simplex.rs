use crate::math::{Vec3, v3};

const LARGE: f32 = f32::from_bits(0x5d5e0b6b);

#[derive(Clone, Copy, Debug)]
pub struct Simplex {
    pub n: i32,
    pub w: [Vec3; 5],
    pub p: [Vec3; 5],
    pub q: [Vec3; 5],
    pub cached_p1: Vec3,
    pub cached_p2: Vec3,
    pub cached_v: Vec3,
    pub last_w: Vec3,
    pub equal_threshold: f32,
    pub cached_valid: bool,
    pub closest: Vec3,
    pub used: u8,
    pub bary: [f32; 4],
    pub degenerate: bool,
    pub needs_update: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SubResult {
    pub closest: Vec3,
    pub used: u8,
    pub bary: [f32; 3],
}

impl Default for Simplex {
    fn default() -> Simplex {
        Simplex::new()
    }
}

impl Simplex {

    pub fn new() -> Simplex {
        Simplex {
            n: 0,
            w: [Vec3::ZERO; 5],
            p: [Vec3::ZERO; 5],
            q: [Vec3::ZERO; 5],
            cached_p1: Vec3::ZERO,
            cached_p2: Vec3::ZERO,
            cached_v: Vec3::ZERO,
            last_w: Vec3::ZERO,
            equal_threshold: f32::from_bits(0x38d1b717),
            cached_valid: false,
            closest: Vec3::ZERO,
            used: 0,
            bary: [0.0; 4],
            degenerate: false,
            needs_update: false,
        }
    }

    pub fn reset(&mut self) {
        self.cached_valid = false;
        self.n = 0;
        self.needs_update = true;
        self.last_w = v3(LARGE, LARGE, LARGE);
        self.degenerate = false;
        self.bary = [0.0; 4];
        self.used = 0;
    }

    pub fn add_vertex(&mut self, w: Vec3, p: Vec3, q: Vec3) {
        self.last_w = w;
        self.needs_update = true;
        let i = self.n as usize;
        self.w[i] = w;
        self.p[i] = p;
        self.q[i] = q;
        self.n += 1;
    }

    pub fn in_simplex(&self, w: Vec3) -> bool {
        let mut found = false;
        for i in 0..self.n as usize {
            let s = self.w[i];
            let (dy, dx, dz) = (w.y - s.y, w.x - s.x, w.z - s.z);
            if dy * dy + dx * dx + dz * dz <= self.equal_threshold {
                found = true;
            }
        }
        if w.z == self.last_w.z && w.y == self.last_w.y && w.x == self.last_w.x {
            return true;
        }
        found
    }

    fn reduce(&mut self, used: u8) {
        let mut n = self.n;
        for (bit, i) in [(8u8, 3usize), (4, 2), (2, 1), (1, 0)] {
            if n > i as i32 && used & bit == 0 {
                let last = (n - 1) as usize;
                n -= 1;
                self.w[i] = self.w[last];
                self.p[i] = self.p[last];
                self.q[i] = self.q[last];
            }
        }
        self.n = n;
    }

    fn is_valid(&self) -> bool {
        0.0 <= self.bary[0] && 0.0 <= self.bary[1] && 0.0 <= self.bary[2] && 0.0 <= self.bary[3]
    }

    pub fn update(&mut self) -> bool {
        if !self.needs_update {
            return self.cached_valid;
        }
        self.degenerate = false;
        self.bary = [0.0; 4];
        self.used = 0;
        self.needs_update = false;
        match self.n {
            1 => {
                self.cached_p1 = self.p[0];
                self.cached_p2 = self.q[0];
                let (a, b) = (self.cached_p1, self.cached_p2);
                self.cached_v = v3(a.x - b.x, a.y - b.y, a.z - b.z);
                self.used = 0;
                self.degenerate = false;
                self.bary = [1.0, 0.0, 0.0, 0.0];
            }
            2 => {
                let (w0, w1) = (self.w[0], self.w[1]);
                let (ux, uy, uz) = (w1.x - w0.x, w1.y - w0.y, w1.z - w0.z);
                let mut t = (0.0 - w0.y) * uy + (0.0 - w0.x) * ux + (0.0 - w0.z) * uz;
                let used;
                if t <= 0.0 {
                    t = 0.0;
                    used = 1;
                } else {
                    let dd = ux * ux + uy * uy + uz * uz;
                    if dd <= t {
                        t = 1.0;
                        used = 2;
                    } else {
                        used = 3;
                        t /= dd;
                    }
                }
                self.used = used;
                self.bary = [1.0 - t, t, 0.0, 0.0];
                let (p0, p1, q0, q1) = (self.p[0], self.p[1], self.q[0], self.q[1]);
                self.cached_p1 = v3((p1.x - p0.x) * t + p0.x, (p1.y - p0.y) * t + p0.y, (p1.z - p0.z) * t + p0.z);
                self.cached_p2 = v3((q1.x - q0.x) * t + q0.x, (q1.y - q0.y) * t + q0.y, (q1.z - q0.z) * t + q0.z);
                let (a, b) = (self.cached_p1, self.cached_p2);
                self.cached_v = v3(a.x - b.x, a.y - b.y, a.z - b.z);
                self.reduce(used);
            }
            3 => {
                let r = closest_triangle(Vec3::ZERO, self.w[0], self.w[1], self.w[2]);
                self.closest = r.closest;
                self.used = r.used;
                self.bary = [r.bary[0], r.bary[1], r.bary[2], 0.0];
                let [b0, b1, b2] = r.bary;
                let (p0, p1, p2, q0, q1, q2) = (self.p[0], self.p[1], self.p[2], self.q[0], self.q[1], self.q[2]);
                self.cached_p1 = v3(b1 * p1.x + b0 * p0.x + b2 * p2.x, b1 * p1.y + b0 * p0.y + b2 * p2.y, b1 * p1.z + b0 * p0.z + b2 * p2.z);
                self.cached_p2 = v3(b0 * q0.x + b1 * q1.x + b2 * q2.x, b1 * q1.y + b0 * q0.y + b2 * q2.y, b1 * q1.z + b0 * q0.z + b2 * q2.z);
                let (a, b) = (self.cached_p1, self.cached_p2);
                self.cached_v = v3(a.x - b.x, a.y - b.y, a.z - b.z);
                self.reduce(r.used);
            }
            4 => {
                if !self.closest_tetrahedron() {
                    if self.degenerate {
                        self.cached_valid = false;
                        return false;
                    }
                    self.cached_valid = true;
                    self.cached_v = Vec3::ZERO;
                    return true;
                }
                let [b0, b1, b2, b3] = self.bary;
                let (p, q) = (self.p, self.q);
                self.cached_p1 = v3(
                    b1 * p[1].x + b0 * p[0].x + b2 * p[2].x + b3 * p[3].x,
                    b1 * p[1].y + b0 * p[0].y + b2 * p[2].y + b3 * p[3].y,
                    b1 * p[1].z + b0 * p[0].z + b2 * p[2].z + b3 * p[3].z,
                );
                self.cached_p2 = v3(
                    b0 * q[0].x + b1 * q[1].x + b2 * q[2].x + b3 * q[3].x,
                    b1 * q[1].y + b0 * q[0].y + b2 * q[2].y + b3 * q[3].y,
                    b1 * q[1].z + b0 * q[0].z + b2 * q[2].z + b3 * q[3].z,
                );
                let (a, b) = (self.cached_p1, self.cached_p2);
                self.cached_v = v3(a.x - b.x, a.y - b.y, a.z - b.z);
                self.reduce(self.used);
            }
            _ => {
                self.cached_valid = false;
                return false;
            }
        }
        self.cached_valid = self.is_valid();
        self.cached_valid
    }

    fn closest_tetrahedron(&mut self) -> bool {
        let p = Vec3::ZERO;
        let (a, b, c, d) = (self.w[0], self.w[1], self.w[2], self.w[3]);
        self.closest = p;
        self.used |= 0xf;
        let abc = outside_plane(p, a, b, c, d);
        let acd = outside_plane(p, a, c, d, b);
        let adb = outside_plane(p, a, d, b, c);
        let bdc = outside_plane(p, b, d, c, a);
        if abc < 0 || acd < 0 || adb < 0 || bdc < 0 {
            self.degenerate = true;
            return false;
        }
        if abc == 0 && acd == 0 && adb == 0 && bdc == 0 {
            return false;
        }
        let mut best = f32::MAX;
        let dist = |t: Vec3| {
            let (dy, dx, dz) = (t.y - p.y, t.x - p.x, t.z - p.z);
            dy * dy + dx * dx + dz * dz
        };
        if abc != 0 {
            let t = closest_triangle(p, a, b, c);
            let d2 = dist(t.closest);
            if d2 < f32::MAX {
                self.closest = t.closest;
                self.used = t.used & 7;
                self.bary = [t.bary[0], t.bary[1], t.bary[2], 0.0];
                best = d2;
            }
        }
        if acd != 0 {
            let t = closest_triangle(p, a, c, d);
            let d2 = dist(t.closest);
            if d2 < best {
                self.closest = t.closest;
                self.used = (t.used & 6) * 2 | t.used & 1;
                self.bary = [t.bary[0], 0.0, t.bary[1], t.bary[2]];
                best = d2;
            }
        }
        if adb != 0 {
            let t = closest_triangle(p, a, d, b);
            let d2 = dist(t.closest);
            if d2 < best {
                self.closest = t.closest;
                self.used = t.used >> 1 & 2 | (t.used & 2) << 2 | t.used & 1;
                self.bary = [t.bary[0], t.bary[2], 0.0, t.bary[1]];
                best = d2;
            }
        }
        if bdc != 0 {
            let t = closest_triangle(p, b, d, c);
            let d2 = dist(t.closest);
            if d2 < best {
                self.closest = t.closest;
                self.bary = [0.0, t.bary[0], t.bary[2], t.bary[1]];
                self.used = ((t.used & 2) * 2 | t.used & 1) * 2 | t.used & 4;
            }
        }
        true
    }
}

pub fn outside_plane(p: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> i32 {
    let (bax, bay, baz) = (b.x - a.x, b.y - a.y, b.z - a.z);
    let (cax, cay, caz) = (c.x - a.x, c.y - a.y, c.z - a.z);
    let nx = caz * bay - cay * baz;
    let nz = cay * bax - cax * bay;
    let ny = cax * baz - caz * bax;
    let signd = (d.y - a.y) * ny + (d.x - a.x) * nx + (d.z - a.z) * nz;
    if f32::from_bits(0x322bcc76) <= signd * signd {
        (((p.y - a.y) * ny + (p.x - a.x) * nx + (p.z - a.z) * nz) * signd < 0.0) as i32
    } else {
        -1
    }
}

pub fn closest_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> SubResult {
    let abz = b.z - a.z;
    let acy = c.y - a.y;
    let abx = b.x - a.x;
    let acx = c.x - a.x;
    let acz = c.z - a.z;
    let aby = b.y - a.y;
    let d2 = (p.y - a.y) * acy + (p.x - a.x) * acx + (p.z - a.z) * acz;
    let d1 = (p.y - a.y) * aby + (p.x - a.x) * abx + (p.z - a.z) * abz;
    if d1 <= 0.0 && d2 <= 0.0 {
        return SubResult { closest: a, used: 1, bary: [1.0, 0.0, 0.0] };
    }
    let d4 = (p.y - b.y) * acy + (p.x - b.x) * acx + (p.z - b.z) * acz;
    let d3 = (p.y - b.y) * aby + (p.x - b.x) * abx + (p.z - b.z) * abz;
    if d3 < 0.0 || d3 < d4 {
        let vc = d4 * d1 - d3 * d2;
        if vc <= 0.0 && 0.0 <= d1 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return SubResult { closest: v3(v * abx + a.x, v * aby + a.y, v * abz + a.z), used: 3, bary: [1.0 - v, v, 0.0] };
        }
        let d5 = (p.y - c.y) * aby + (p.x - c.x) * abx + (p.z - c.z) * abz;
        let d6 = (p.y - c.y) * acy + (p.x - c.x) * acx + (p.z - c.z) * acz;
        if 0.0 <= d6 && d5 <= d6 {
            return SubResult { closest: c, used: 4, bary: [0.0, 0.0, 1.0] };
        }
        let vb = d5 * d2 - d6 * d1;
        if 0.0 < vb || d2 < 0.0 || 0.0 < d6 {
            let va = d6 * d3 - d5 * d4;
            let d4m3 = d4 - d3;
            if va <= 0.0 && 0.0 <= d4m3 && 0.0 <= d5 - d6 {
                let w = d4m3 / ((d5 - d6) + d4m3);
                return SubResult { closest: v3((c.x - b.x) * w + b.x, (c.y - b.y) * w + b.y, (c.z - b.z) * w + b.z), used: 6, bary: [0.0, 1.0 - w, w] };
            }
            let denom = 1.0 / (va + vb + vc);
            let w = denom * vc;
            let v = denom * vb;
            return SubResult { closest: v3(v * abx + a.x + w * acx, v * aby + a.y + w * acy, v * abz + a.z + w * acz), used: 7, bary: [(1.0 - v) - w, v, w] };
        }
        let w = d2 / (d2 - d6);
        return SubResult { closest: v3(w * acx + a.x, w * acy + a.y, w * acz + a.z), used: 5, bary: [1.0 - w, 0.0, w] };
    }
    SubResult { closest: b, used: 2, bary: [0.0, 1.0, 0.0] }
}
