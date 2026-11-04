use crate::math::{Transform, Vec3, v3};

pub const SHAPE_BOX: i32 = 0;
pub const SHAPE_TRIANGLE: i32 = 1;
pub const SHAPE_SPHERE: i32 = 8;
pub const SHAPE_MESH: i32 = 21;
pub const SHAPE_PLANE: i32 = 28;
pub const SHAPE_COMPOUND: i32 = 31;

pub const MARGIN_DEFAULT: f32 = f32::from_bits(0x3d23d70a);
const EPS2: f32 = f32::from_bits(0x28800000);
const LARGE: f32 = f32::from_bits(0x5d5e0b6b);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereShape {
    pub scale: Vec3,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxShape {
    pub scale: Vec3,
    pub half: Vec3,
    pub margin: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneShape {
    pub half_x: f32,
    pub half_y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleShape {
    pub v: [Vec3; 3],
    pub margin: f32,
}

#[derive(Clone, Debug)]
pub enum Shape {
    Sphere(SphereShape),
    Box(BoxShape),
    Plane(PlaneShape),
    Triangle(TriangleShape),
    #[cfg(feature = "collide")]
    Compound(super::algo_compound::CompoundShape),
    #[cfg(feature = "collide")]
    Mesh(super::bvh::MeshShape),
}

impl SphereShape {

    pub fn new(radius: f32) -> SphereShape {
        SphereShape { scale: v3(1.0, 1.0, 1.0), radius }
    }

    #[inline]
    pub fn margin(&self) -> f32 {
        self.radius * self.scale.x
    }

    pub fn aabb(&self, t: &Transform) -> (Vec3, Vec3) {
        let e = self.radius * self.scale.x + MARGIN_DEFAULT + MARGIN_DEFAULT;
        (v3(t.origin.x - e, t.origin.y - e, t.origin.z - e), v3(e + t.origin.x, e + t.origin.y, e + t.origin.z))
    }

    pub fn local_support(&self, dir: Vec3) -> Vec3 {
        let (mut x, mut y, mut z) = (dir.x, dir.y, dir.z);
        if x * x + y * y + z * z < EPS2 {
            x = -1.0;
            y = -1.0;
            z = -1.0;
        }
        let k = 1.0 / (x * x + y * y + z * z).sqrt();
        let m = self.margin();
        v3(k * x * m + 0.0, k * y * m + 0.0, k * z * m + 0.0)
    }

    pub fn local_inertia(&self, mass: f32) -> Vec3 {
        let m = self.margin();
        let e = m * (m * (mass * 0.4));
        v3(e, e, e)
    }

    pub fn bounding_sphere(&self) -> (Vec3, f32) {
        (Vec3::ZERO, self.radius * self.scale.x + MARGIN_DEFAULT + MARGIN_DEFAULT)
    }
}

impl BoxShape {

    pub fn new(half: Vec3) -> BoxShape {
        let mut b = BoxShape { scale: v3(1.0, 1.0, 1.0), half: Vec3::ZERO, margin: MARGIN_DEFAULT };
        let i = if half.y <= half.x {
            (half.z <= half.y) as usize + 1
        } else if half.x < half.z {
            0
        } else {
            2
        };
        let tenth = half[i] * 0.1;
        if tenth < b.margin {
            b.set_margin(tenth);
        }
        let m = b.margin;
        b.half = v3(half.x * b.scale.x - m, b.scale.y * half.y - m, b.scale.z * half.z - m);
        b
    }

    pub fn set_margin(&mut self, new_margin: f32) {
        let old = self.margin;
        let (hx, hy, hz) = (self.half.x, self.half.y, self.half.z);
        self.margin = new_margin;
        self.half = v3(old + hx - new_margin, old + hy - new_margin, old + hz - new_margin);
    }

    pub fn aabb(&self, t: &Transform) -> (Vec3, Vec3) {
        let m = self.margin;
        let ez = m + self.half.z;
        let ex = m + self.half.x;
        let ey = m + self.half.y;
        let r = &t.basis.r;
        let dx = r[0].y.abs() * ey + r[0].x.abs() * ex + r[0].z.abs() * ez;
        let dy = r[1].y.abs() * ey + r[1].x.abs() * ex + r[1].z.abs() * ez;
        let dz = r[2].y.abs() * ey + r[2].x.abs() * ex + r[2].z.abs() * ez;
        let o = t.origin;
        (v3(o.x - dx, o.y - dy, o.z - dz), v3(dx + o.x, dy + o.y, dz + o.z))
    }

    pub fn local_support(&self, dir: Vec3) -> Vec3 {
        let m = self.margin;
        let mut x = m + self.half.x;
        let mut z = self.half.z + m;
        let mut y = self.half.y + m;
        if dir.z < 0.0 {
            z = -z;
        }
        if dir.y < 0.0 {
            y = -y;
        }
        if dir.x < 0.0 {
            x = -x;
        }
        v3(x, y, z)
    }

    pub fn local_inertia(&self, mass: f32) -> Vec3 {
        let m = self.margin;
        let k = mass / 12.0;
        let (hx, hy, hz) = (m + self.half.x, self.half.y + m, m + self.half.z);
        let (x, y, z) = (hx + hx, hy + hy, hz + hz);
        let (x2, y2, z2) = (x * x, y * y, z * z);
        v3((z2 + y2) * k, (x2 + z2) * k, (x2 + y2) * k)
    }
}

impl PlaneShape {

    pub fn new(half_x: f32, half_y: f32) -> PlaneShape {
        PlaneShape { half_x, half_y }
    }

    pub fn aabb(&self) -> (Vec3, Vec3) {
        (v3(-LARGE, -LARGE, -LARGE), v3(LARGE, LARGE, LARGE))
    }

    pub fn broadphase_aabb(&self, t: &Transform) -> (Vec3, Vec3) {
        let r = &t.basis.r;
        let (r00, r01, r02) = (r[0].x, r[0].y, r[0].z);
        let (r10, r11, r12) = (r[1].x, r[1].y, r[1].z);
        let (r20, r21, r22) = (r[2].x, r[2].y, r[2].z);
        let (hx, hy) = (self.half_x, self.half_y);
        let hxz = hx * 0.0;
        let ax = hxz * r01 + r00 * hx + hxz * r02;
        let ay = hx * r10 + hxz * r11 + hxz * r12;
        let az = hx * r20 + hxz * r21 + hxz * r22;
        let hyz = hy * 0.0;
        let bx = r00 * hyz + hy * r01 + r02 * hyz;
        let by = hy * r11 + hyz * r10 + hyz * r12;
        let bz = hy * r21 + hyz * r20 + hyz * r22;
        let dx = (r00 * -0.0 + r01 * -0.0) - r02 * LARGE;
        let dy = (r10 * -0.0 + r11 * -0.0) - r12 * LARGE;
        let dz = (r20 * -0.0 + r21 * -0.0) - r22 * LARGE;
        let o = t.origin;
        let (x_lo, x_hi) = (o.x - ax, o.x + ax);
        let (y_lo, y_hi) = (o.y - ay, o.y + ay);
        let (z_hi, z_lo) = (o.z + az, o.z - az);

        let c0 = v3(x_lo - bx, y_lo - by, z_lo - bz);
        let c1 = v3(x_hi + bx, y_hi + by, z_hi + bz);
        let c2 = v3(x_lo + bx, y_lo + by, z_lo + bz);
        let c3 = v3(x_hi - bx, y_hi - by, z_hi - bz);
        let (mut lo, mut hi) = (c0, c0);
        let d = v3(dx, dy, dz);
        for c in [c1, c2, c3, c0 + d, c1 + d, c2 + d, c3 + d] {
            lo = lo.min(c);
            hi = hi.max(c);
        }
        (lo, hi)
    }
}

impl TriangleShape {

    pub fn local_support_no_margin(&self, d: Vec3) -> Vec3 {
        let dot = |v: Vec3| d.y * v.y + d.x * v.x + d.z * v.z;
        let (d0, d1, d2) = (dot(self.v[0]), dot(self.v[1]), dot(self.v[2]));
        let i = if d1 <= d0 {
            if d0 < d2 { 2 } else { 0 }
        } else {
            (d1 < d2) as usize + 1
        };
        self.v[i]
    }

    pub fn local_support(&self, dir: Vec3) -> Vec3 {
        let mut out = self.local_support_no_margin(dir);
        if self.margin != 0.0 {
            let (mut x, mut y, mut z) = (dir.x, dir.y, dir.z);
            if x * x + y * y + z * z < EPS2 {
                x = -1.0;
                y = -1.0;
                z = -1.0;
            }
            let k = 1.0 / (x * x + y * y + z * z).sqrt();
            let m = self.margin;
            out = v3(k * x * m + out.x, k * y * m + out.y, k * z * m + out.z);
        }
        out
    }

    pub fn aabb(&self, t: &Transform) -> (Vec3, Vec3) {
        let m = self.margin;
        let r = &t.basis.r;
        let o = t.origin;
        let mut lo = Vec3::ZERO;
        let mut hi = Vec3::ZERO;
        for i in 0..3 {
            let mut vec = Vec3::ZERO;
            vec[i] = 1.0;
            let d = v3(vec.y * r[1].x + vec.x * r[0].x + vec.z * r[2].x, vec.x * r[0].y + vec.y * r[1].y + vec.z * r[2].y, vec.y * r[1].z + vec.x * r[0].z + vec.z * r[2].z);
            let s = self.local_support(d);
            let w = v3(s.y * r[0].y + s.x * r[0].x + s.z * r[0].z + o.x, s.y * r[1].y + s.x * r[1].x + s.z * r[1].z + o.y, s.y * r[2].y + s.x * r[2].x + s.z * r[2].z + o.z);
            hi[i] = m + w[i];
            vec[i] = -1.0;
            let d = v3(vec.y * r[1].x + vec.x * r[0].x + vec.z * r[2].x, vec.x * r[0].y + vec.y * r[1].y + vec.z * r[2].y, vec.y * r[1].z + vec.x * r[0].z + vec.z * r[2].z);
            let s = self.local_support(d);
            let w = v3(s.x * r[0].x + s.y * r[0].y + s.z * r[0].z + o.x, s.x * r[1].x + s.y * r[1].y + s.z * r[1].z + o.y, s.x * r[2].x + s.y * r[2].y + s.z * r[2].z + o.z);
            lo[i] = w[i] - m;
        }
        (lo, hi)
    }

    pub fn normal(&self) -> Vec3 {
        let v = &self.v;
        let e2z = v[2].z - v[0].z;
        let e2x = v[2].x - v[0].x;
        let e1y = v[1].y - v[0].y;
        let e1z = v[1].z - v[0].z;
        let e1x = v[1].x - v[0].x;
        let e2y = v[2].y - v[0].y;
        let nx = e2z * e1y - e2y * e1z;
        let ny = e2x * e1z - e2z * e1x;
        let nz = e2y * e1x - e2x * e1y;
        let k = 1.0 / (ny * ny + nx * nx + nz * nz).sqrt();
        v3(k * nx, k * ny, k * nz)
    }
}

impl Shape {

    #[inline]
    pub fn type_id(&self) -> i32 {
        match self {
            Shape::Sphere(_) => SHAPE_SPHERE,
            Shape::Box(_) => SHAPE_BOX,
            Shape::Plane(_) => SHAPE_PLANE,
            Shape::Triangle(_) => SHAPE_TRIANGLE,
            #[cfg(feature = "collide")]
            Shape::Compound(_) => SHAPE_COMPOUND,
            #[cfg(feature = "collide")]
            Shape::Mesh(_) => SHAPE_MESH,
        }
    }

    pub fn margin(&self) -> f32 {
        match self {
            Shape::Sphere(s) => s.margin(),
            Shape::Box(b) => b.margin,
            Shape::Plane(_) => 0.0,
            Shape::Triangle(t) => t.margin,
            #[cfg(feature = "collide")]
            Shape::Compound(c) => c.margin,
            #[cfg(feature = "collide")]
            Shape::Mesh(m) => m.margin,
        }
    }

    pub fn aabb(&self, t: &Transform) -> (Vec3, Vec3) {
        match self {
            Shape::Sphere(s) => s.aabb(t),
            Shape::Box(b) => b.aabb(t),
            Shape::Plane(p) => p.aabb(),
            Shape::Triangle(tr) => tr.aabb(t),
            #[cfg(feature = "collide")]
            Shape::Compound(c) => c.aabb(t),
            #[cfg(feature = "collide")]
            Shape::Mesh(m) => m.aabb(t),
        }
    }

    pub fn broadphase_aabb(&self, t: &Transform) -> (Vec3, Vec3) {
        match self {
            Shape::Plane(p) => p.broadphase_aabb(t),
            _ => self.aabb(t),
        }
    }

    pub fn local_support(&self, dir: Vec3) -> Vec3 {
        match self {
            Shape::Sphere(s) => s.local_support(dir),
            Shape::Box(b) => b.local_support(dir),
            Shape::Triangle(t) => t.local_support(dir),
            _ => unreachable!("not a convex shape"),
        }
    }

    pub fn support_no_margin(&self, d: Vec3) -> Vec3 {
        match self {
            Shape::Sphere(_) => Vec3::ZERO,
            Shape::Box(b) => {
                let z = if d.z < 0.0 { -b.half.z } else { b.half.z };
                let y = if d.y < 0.0 { -b.half.y } else { b.half.y };
                let x = if d.x < 0.0 { -b.half.x } else { b.half.x };
                v3(x, y, z)
            }
            Shape::Triangle(t) => t.local_support_no_margin(d),
            _ => unreachable!("not a convex shape"),
        }
    }

    pub fn support_with_margin(&self, dir: Vec3) -> Vec3 {
        let (mut x, mut y, mut z) = (dir.x, dir.y, dir.z);
        if x * x + y * y + z * z < EPS2 {
            x = -1.0;
            y = -1.0;
            z = -1.0;
        }
        let k = 1.0 / (x * x + y * y + z * z).sqrt();
        let (x, y, z) = (x * k, y * k, z * k);
        let m = self.margin();
        let (ox, oy, oz) = (x * m, y * m, z * m);
        let s = self.support_no_margin(v3(x, y, z));
        v3(ox + s.x, oy + s.y, oz + s.z)
    }

    pub fn bounding_sphere(&self) -> (Vec3, f32) {
        if let Shape::Sphere(s) = self {
            return s.bounding_sphere();
        }
        let (lo, hi) = self.aabb(&Transform::IDENTITY);
        let (dx, dy, dz) = (hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
        let r = (dy * dy + dx * dx + dz * dz).sqrt() * 0.5;
        (v3((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, (lo.z + hi.z) * 0.5), r)
    }

    pub fn angular_motion_disc(&self) -> f32 {
        let (c, r) = self.bounding_sphere();
        (c.x * c.x + c.y * c.y + c.z * c.z).sqrt() + r
    }

    pub fn contact_breaking_threshold(&self, factor: f32) -> f32 {
        self.angular_motion_disc() * factor
    }

    pub fn local_inertia(&self, mass: f32) -> Vec3 {
        match self {
            Shape::Sphere(s) => s.local_inertia(mass),
            Shape::Box(b) => b.local_inertia(mass),
            #[cfg(feature = "collide")]
            Shape::Compound(c) => c.local_inertia(mass),
            _ => Vec3::ZERO,
        }
    }
}
