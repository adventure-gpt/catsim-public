pub mod crt;
pub mod crt_pow;
pub mod rot;

use core::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[inline(always)]
pub const fn v3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub const ZERO: Vec3 = v3(0.0, 0.0, 0.0);

    #[inline(always)]
    pub fn dot_xy(self, b: Vec3) -> f32 {
        self.x * b.x + self.y * b.y + self.z * b.z
    }

    #[inline(always)]
    pub fn dot_xz(self, b: Vec3) -> f32 {
        self.x * b.x + self.z * b.z + self.y * b.y
    }

    #[inline(always)]
    pub fn dot_yz(self, b: Vec3) -> f32 {
        self.y * b.y + self.z * b.z + self.x * b.x
    }
    #[inline(always)]
    pub fn len2_xy(self) -> f32 {
        self.dot_xy(self)
    }
    #[inline(always)]
    pub fn len2_xz(self) -> f32 {
        self.dot_xz(self)
    }
    #[inline(always)]
    pub fn len2_yz(self) -> f32 {
        self.dot_yz(self)
    }

    #[inline(always)]
    pub fn cross(self, b: Vec3) -> Vec3 {
        v3(self.y * b.z - self.z * b.y, self.z * b.x - self.x * b.z, self.x * b.y - self.y * b.x)
    }

    #[inline(always)]
    pub fn min(self, b: Vec3) -> Vec3 {
        v3(if b.x < self.x { b.x } else { self.x }, if b.y < self.y { b.y } else { self.y }, if b.z < self.z { b.z } else { self.z })
    }

    #[inline(always)]
    pub fn max(self, b: Vec3) -> Vec3 {
        v3(if self.x < b.x { b.x } else { self.x }, if self.y < b.y { b.y } else { self.y }, if self.z < b.z { b.z } else { self.z })
    }
    #[inline(always)]
    pub fn comp_mul(self, b: Vec3) -> Vec3 {
        v3(self.x * b.x, self.y * b.y, self.z * b.z)
    }
    #[inline(always)]
    pub fn abs(self) -> Vec3 {
        v3(self.x.abs(), self.y.abs(), self.z.abs())
    }

    #[inline(always)]
    pub fn bits(self) -> [u32; 3] {
        [self.x.to_bits(), self.y.to_bits(), self.z.to_bits()]
    }
    #[inline(always)]
    pub fn from_bits(b: [u32; 3]) -> Vec3 {
        v3(f32::from_bits(b[0]), f32::from_bits(b[1]), f32::from_bits(b[2]))
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn add(self, b: Vec3) -> Vec3 {
        v3(self.x + b.x, self.y + b.y, self.z + b.z)
    }
}
impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, b: Vec3) {
        *self = *self + b;
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn sub(self, b: Vec3) -> Vec3 {
        v3(self.x - b.x, self.y - b.y, self.z - b.z)
    }
}
impl SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, b: Vec3) {
        *self = *self - b;
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn neg(self) -> Vec3 {
        v3(-self.x, -self.y, -self.z)
    }
}
impl Mul<f32> for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, s: f32) -> Vec3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}
impl Index<usize> for Vec3 {
    type Output = f32;
    #[inline(always)]
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}
impl IndexMut<usize> for Vec3 {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.z,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
pub struct Mat3 {
    pub r: [Vec3; 3],
}

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3 { r: [v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)] };
    pub const ZERO: Mat3 = Mat3 { r: [Vec3::ZERO; 3] };

    #[inline(always)]
    pub fn col(&self, c: usize) -> Vec3 {
        v3(self.r[0][c], self.r[1][c], self.r[2][c])
    }

    #[inline(always)]
    pub fn mul_xy(&self, v: Vec3) -> Vec3 {
        v3(self.r[0].dot_xy(v), self.r[1].dot_xy(v), self.r[2].dot_xy(v))
    }
    #[inline(always)]
    pub fn mul_xz(&self, v: Vec3) -> Vec3 {
        v3(self.r[0].dot_xz(v), self.r[1].dot_xz(v), self.r[2].dot_xz(v))
    }
    #[inline(always)]
    pub fn mul_yz(&self, v: Vec3) -> Vec3 {
        v3(self.r[0].dot_yz(v), self.r[1].dot_yz(v), self.r[2].dot_yz(v))
    }

    #[inline(always)]
    pub fn tmul_xy(&self, v: Vec3) -> Vec3 {
        v3(self.col(0).dot_xy(v), self.col(1).dot_xy(v), self.col(2).dot_xy(v))
    }
    #[inline(always)]
    pub fn tmul_xz(&self, v: Vec3) -> Vec3 {
        v3(self.col(0).dot_xz(v), self.col(1).dot_xz(v), self.col(2).dot_xz(v))
    }
    #[inline(always)]
    pub fn tmul_yz(&self, v: Vec3) -> Vec3 {
        v3(self.col(0).dot_yz(v), self.col(1).dot_yz(v), self.col(2).dot_yz(v))
    }
    #[inline(always)]
    pub fn transpose(&self) -> Mat3 {
        Mat3 { r: [self.col(0), self.col(1), self.col(2)] }
    }
    pub fn bits(&self) -> [u32; 9] {
        let (a, b, c) = (self.r[0].bits(), self.r[1].bits(), self.r[2].bits());
        [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
pub struct Transform {
    pub basis: Mat3,
    pub origin: Vec3,
}

impl Transform {
    pub const IDENTITY: Transform = Transform { basis: Mat3::IDENTITY, origin: Vec3::ZERO };
}

#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn u32(&mut self) -> u32 {
        (self.u64() >> 32) as u32
    }

    pub fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        let t = (self.u32() >> 8) as f32 / 16777215.0;
        lo + (hi - lo) * t
    }

    pub fn f32_tricky(&mut self, scale: f32) -> f32 {
        match self.u32() % 16 {
            0 => 0.0,
            1 => 1.0,
            2 => -1.0,
            3 => self.f32_in(-1e-4, 1e-4),
            4 => (self.u32() % 2048) as f32 - 1024.0,
            _ => self.f32_in(-scale, scale),
        }
    }
    pub fn vec3(&mut self, scale: f32) -> Vec3 {
        v3(self.f32_in(-scale, scale), self.f32_in(-scale, scale), self.f32_in(-scale, scale))
    }

    pub fn rotation(&mut self) -> Mat3 {
        let (mut x, mut y, mut z, mut w) = (self.f32_in(-1.0, 1.0), self.f32_in(-1.0, 1.0), self.f32_in(-1.0, 1.0), self.f32_in(-1.0, 1.0));
        let n = (x * x + y * y + z * z + w * w).sqrt().max(1e-3);
        x /= n;
        y /= n;
        z /= n;
        w /= n;
        Mat3 {
            r: [
                v3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
                v3(2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
                v3(2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
            ],
        }
    }
}
