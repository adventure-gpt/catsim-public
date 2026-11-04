#![allow(clippy::missing_safety_doc)]

use core::arch::x86_64::*;
use core::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy)]
pub struct F8(pub __m256);

#[derive(Clone, Copy)]
pub struct M8(pub __m256);
#[derive(Clone, Copy)]
pub struct D4(pub __m256d);
#[derive(Clone, Copy)]
pub struct I8(pub __m256i);

pub fn available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

macro_rules! binop {
    ($tr:ident, $f:ident, $op:ident) => {
        impl $tr for F8 {
            type Output = F8;
            #[inline(always)]
            fn $f(self, o: F8) -> F8 {
                unsafe { F8($op(self.0, o.0)) }
            }
        }
        impl $tr<f32> for F8 {
            type Output = F8;
            #[inline(always)]
            fn $f(self, o: f32) -> F8 {
                unsafe { F8($op(self.0, _mm256_set1_ps(o))) }
            }
        }
        impl $tr<F8> for f32 {
            type Output = F8;
            #[inline(always)]
            fn $f(self, o: F8) -> F8 {
                unsafe { F8($op(_mm256_set1_ps(self), o.0)) }
            }
        }
    };
}
binop!(Add, add, _mm256_add_ps);
binop!(Sub, sub, _mm256_sub_ps);
binop!(Mul, mul, _mm256_mul_ps);
binop!(Div, div, _mm256_div_ps);

impl Neg for F8 {
    type Output = F8;
    #[inline(always)]
    fn neg(self) -> F8 {
        unsafe { F8(_mm256_xor_ps(self.0, _mm256_set1_ps(-0.0))) }
    }
}

impl F8 {
    #[inline(always)]
    pub fn splat(x: f32) -> F8 {
        unsafe { F8(_mm256_set1_ps(x)) }
    }
    #[inline(always)]
    pub fn load(a: &[f32; 8]) -> F8 {
        unsafe { F8(_mm256_loadu_ps(a.as_ptr())) }
    }
    #[inline(always)]
    pub fn store(self, a: &mut [f32; 8]) {
        unsafe { _mm256_storeu_ps(a.as_mut_ptr(), self.0) }
    }
    #[inline(always)]
    pub fn sqrt(self) -> F8 {
        unsafe { F8(_mm256_sqrt_ps(self.0)) }
    }
    #[inline(always)]
    pub fn lt(self, o: F8) -> M8 {
        unsafe { M8(_mm256_cmp_ps::<_CMP_LT_OQ>(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn le(self, o: F8) -> M8 {
        unsafe { M8(_mm256_cmp_ps::<_CMP_LE_OQ>(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn eq(self, o: F8) -> M8 {
        unsafe { M8(_mm256_cmp_ps::<_CMP_EQ_OQ>(self.0, o.0)) }
    }

    #[inline(always)]
    pub fn abs_bits(self) -> I8 {
        unsafe { I8(_mm256_and_si256(_mm256_castps_si256(self.0), _mm256_set1_epi32(0x7fff_ffff))) }
    }
    #[inline(always)]
    pub fn abs(self) -> F8 {
        unsafe { F8(_mm256_and_ps(self.0, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff)))) }
    }

    #[inline(always)]
    pub fn finite(self) -> M8 {
        unsafe { M8(_mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f80_0000), self.abs_bits().0))) }
    }

    #[inline(always)]
    pub fn cvt(self) -> I8 {
        unsafe { I8(_mm256_cvtps_epi32(self.0)) }
    }

    #[inline(always)]
    pub fn cvtt(self) -> I8 {
        unsafe { I8(_mm256_cvttps_epi32(self.0)) }
    }
    #[inline(always)]
    pub fn lo(self) -> D4 {
        unsafe { D4(_mm256_cvtps_pd(_mm256_castps256_ps128(self.0))) }
    }
    #[inline(always)]
    pub fn hi(self) -> D4 {
        unsafe { D4(_mm256_cvtps_pd(_mm256_extractf128_ps::<1>(self.0))) }
    }

    #[inline(always)]
    pub fn from_d(lo: D4, hi: D4) -> F8 {
        unsafe { F8(_mm256_set_m128(_mm256_cvtpd_ps(hi.0), _mm256_cvtpd_ps(lo.0))) }
    }
}

impl M8 {
    #[inline(always)]
    pub fn and(self, o: M8) -> M8 {
        unsafe { M8(_mm256_and_ps(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn or(self, o: M8) -> M8 {
        unsafe { M8(_mm256_or_ps(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn not(self) -> M8 {
        unsafe { M8(_mm256_xor_ps(self.0, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))) }
    }

    #[inline(always)]
    pub fn bits(self) -> u32 {
        unsafe { _mm256_movemask_ps(self.0) as u32 }
    }
}

impl I8 {
    #[inline(always)]
    pub fn splat(x: i32) -> I8 {
        unsafe { I8(_mm256_set1_epi32(x)) }
    }

    #[inline(always)]
    pub fn lt(self, o: I8) -> M8 {
        unsafe { M8(_mm256_castsi256_ps(_mm256_cmpgt_epi32(o.0, self.0))) }
    }
    #[inline(always)]
    pub fn le(self, o: I8) -> M8 {
        unsafe { M8(_mm256_castsi256_ps(_mm256_xor_si256(_mm256_cmpgt_epi32(self.0, o.0), _mm256_set1_epi32(-1)))) }
    }

    #[inline(always)]
    pub fn sra1(self) -> I8 {
        unsafe { I8(_mm256_srai_epi32::<1>(self.0)) }
    }
    #[inline(always)]
    pub fn clamp(self, lo: i32, hi: i32) -> I8 {
        unsafe { I8(_mm256_min_epi32(_mm256_max_epi32(self.0, _mm256_set1_epi32(lo)), _mm256_set1_epi32(hi))) }
    }

    #[inline(always)]
    pub fn to_f(self) -> F8 {
        unsafe { F8(_mm256_cvtepi32_ps(self.0)) }
    }
}

#[inline(always)]
pub fn sel(m: M8, a: F8, b: F8) -> F8 {
    unsafe { F8(_mm256_blendv_ps(b.0, a.0, m.0)) }
}

impl D4 {
    #[inline(always)]
    pub fn splat(x: f64) -> D4 {
        unsafe { D4(_mm256_set1_pd(x)) }
    }
    #[inline(always)]
    pub fn mul(self, o: D4) -> D4 {
        unsafe { D4(_mm256_mul_pd(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn sub(self, o: D4) -> D4 {
        unsafe { D4(_mm256_sub_pd(self.0, o.0)) }
    }
    #[inline(always)]
    pub fn neg(self) -> D4 {
        unsafe { D4(_mm256_xor_pd(self.0, _mm256_set1_pd(-0.0))) }
    }
    #[inline(always)]
    pub fn sqrt(self) -> D4 {
        unsafe { D4(_mm256_sqrt_pd(self.0)) }
    }

    #[inline(always)]
    pub fn mul_add(self, b: D4, c: D4) -> D4 {
        unsafe { D4(_mm256_fmadd_pd(self.0, b.0, c.0)) }
    }
}

const S1: f64 = f64::from_bits(0xbfc5555555555555);
const S2: f64 = f64::from_bits(0x3f81111111111111);
const S3: f64 = f64::from_bits(0xbf2a01a01a01a01a);
const S4: f64 = f64::from_bits(0x3ec71de3a556c734);
const C1: f64 = f64::from_bits(0x3fa5555555555555);
const C2: f64 = f64::from_bits(0xbf56c16c16c16c16);
const C3: f64 = f64::from_bits(0x3efa01a01a01a019);
const C4: f64 = f64::from_bits(0xbe927e4fb7789f5c);
const ONE_SIXTH: f64 = f64::from_bits(0x3fc5555555555555);

#[inline(always)]
fn sin_kernel(r: D4) -> D4 {
    let r2 = r.mul(r);
    let p = r2.mul_add(D4::splat(S4), D4::splat(S3));
    let p = p.mul_add(r2, D4::splat(S2));
    let p = p.mul_add(r2, D4::splat(S1));
    let r3 = r.mul(r2);
    p.mul_add(r3, r)
}

#[inline(always)]
fn cos_poly(r2: D4, head: D4) -> D4 {
    let p = r2.mul_add(D4::splat(C4), D4::splat(C3));
    let p = p.mul_add(r2, D4::splat(C2));
    let p = p.mul_add(r2, D4::splat(C1));
    let r4 = r2.mul(r2);
    p.mul_add(r4, head)
}

#[inline(always)]
fn sin_half(d: D4) -> (D4, D4) {
    let x3 = d.mul(d).mul(d);
    (x3.neg().mul_add(D4::splat(ONE_SIXTH), d), sin_kernel(d))
}

#[inline(always)]
fn cos_half(d: D4) -> (D4, D4) {
    let h = d.mul(D4::splat(0.5));
    let a = h.neg().mul_add(d, D4::splat(1.0));
    let r2 = d.mul(d);
    let b = cos_poly(r2, D4::splat(1.0).sub(r2.mul(D4::splat(0.5))));
    (a, b)
}

#[inline(always)]
pub fn trig_small_ok(x: F8) -> M8 {
    x.abs_bits().le(I8::splat(0x3f49_0fdb))
}

#[inline(always)]
pub fn sinf_small(x: F8) -> F8 {
    let bits = x.abs_bits();
    let tiny = bits.lt(I8::splat(0x3900_0000));
    let cubic = bits.lt(I8::splat(0x3c00_0000));
    let (c_lo, k_lo) = sin_half(x.lo());
    let (c_hi, k_hi) = sin_half(x.hi());
    let c = F8::from_d(c_lo, c_hi);
    let k = F8::from_d(k_lo, k_hi);
    sel(tiny, x, sel(cubic, c, k))
}

#[inline(always)]
pub fn cosf_small(x: F8) -> F8 {
    let bits = x.abs_bits();
    let tiny = bits.lt(I8::splat(0x3900_0000));
    let small = bits.lt(I8::splat(0x3c00_0000));
    let (a_lo, b_lo) = cos_half(x.lo());
    let (a_hi, b_hi) = cos_half(x.hi());
    let a = F8::from_d(a_lo, a_hi);
    let b = F8::from_d(b_lo, b_hi);
    sel(tiny, F8::splat(1.0), sel(small, a, b))
}

#[inline(always)]
pub fn mat_to_quat(m: &[F8; 9]) -> [F8; 4] {
    let (m00, m01, m02, m10, m11, m12, m20, m21, m22) = (m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
    let trace = m00 + m11 + m22;
    let neg = trace.le(F8::splat(0.0));

    let a = m11.le(m00);
    let i2 = sel(a, F8(m00.lt(m22).0), F8(m11.lt(m22).0));
    let i2 = M8(i2.0);
    let i0 = a.and(i2.not());
    let i1 = a.not().and(i2.not());
    let arg_neg = sel(i0, m00 - m11 - m22, sel(i1, m11 - m22 - m00, m22 - m00 - m11));
    let s_arg = sel(neg, arg_neg + 1.0, trace + 1.0);
    let s = s_arg.sqrt();
    let t = 0.5 / s;
    let half = s * 0.5;
    let d1 = (m21 - m12) * t;
    let d2 = (m02 - m20) * t;
    let d3 = (m10 - m01) * t;
    let qa = (m10 + m01) * t;
    let qb = (m21 + m12) * t;
    let qc = (m02 + m20) * t;
    let x = sel(neg, sel(i0, half, sel(i1, qa, qc)), d1);
    let y = sel(neg, sel(i0, qa, sel(i1, half, qb)), d2);
    let z = sel(neg, sel(i0, qc, sel(i1, qb, half)), d3);
    let w = sel(neg, sel(i0, d1, sel(i1, d2, d3)), half);
    [x, y, z, w]
}

#[inline(always)]
pub fn quat_to_mat(x: F8, y: F8, z: F8, w: F8) -> [F8; 9] {
    let s = 2.0 / (y * y + x * x + z * z + w * w);
    let sz = s * z;
    let sy = s * y;
    let sxx = s * x * x;
    let sxw = s * x * w;
    [
        1.0 - (sz * z + sy * y),
        sy * x - sz * w,
        sz * x + sy * w,
        sy * x + sz * w,
        1.0 - (sz * z + sxx),
        sz * y - sxw,
        sz * x - sy * w,
        sz * y + sxw,
        1.0 - (sy * y + sxx),
    ]
}

#[inline(always)]
pub fn integrate_transform(o: &[F8; 3], m: &[F8; 9], lin: &[F8; 3], ang: &[F8; 3], dt: F8) -> ([F8; 3], [F8; 9], M8) {
    let origin = [dt * lin[0] + o[0], dt * lin[1] + o[1], dt * lin[2] + o[2]];
    let (wx, wy, wz) = (ang[0], ang[1], ang[2]);
    let w = (wx * wx + wy * wy + wz * wz).sqrt();
    let w = sel(F8::splat(0.7853982).lt(w * dt), 0.7853982 / dt, w);
    let big = F8::splat(0.001).le(w);
    let sarg = w * 0.5 * dt;
    let s = sel(big, sinf_small(sarg) / w, dt * 0.5 - dt * dt * dt * 0.020833334 * w * w);
    let (ax, az, ay) = (wx * s, wz * s, wy * s);
    let carg = w * dt * 0.5;
    let c = cosf_small(carg);
    let ok = trig_small_ok(carg).and(big.not().or(trig_small_ok(sarg)));
    let [qx, qy, qz, qw] = mat_to_quat(m);
    let x = (ax * qw + c * qx + qz * ay) - az * qy;
    let y = (c * qy + qw * ay + az * qx) - ax * qz;
    let z = (c * qz + az * qw + ax * qy) - ay * qx;
    let w2 = ((c * qw - ax * qx) - ay * qy) - az * qz;
    let n = y * y + x * x + z * z + w2 * w2;
    let k = 1.0 / n.sqrt();
    (origin, quat_to_mat(x * k, y * k, z * k, w2 * k), ok)
}

#[inline(always)]
fn capped(v: &[F8; 3], m: F8) -> ([F8; 3], M8) {
    let n2 = v[1] * v[1] + v[0] * v[0] + v[2] * v[2];
    let over = F8::splat(0.0).lt(m).and((m * m).lt(n2));
    let one = n2.eq(F8::splat(1.0));
    let big = F8::splat(1e-08).le(n2);
    let k = 1.0 / n2.sqrt();
    let mut out = *v;
    for c in 0..3 {
        let d = sel(one, v[c], sel(big, k * v[c], F8::splat(0.0)));
        out[c] = sel(over, m * d, v[c]);
    }
    (out, over)
}

#[inline(always)]
fn grid_location(v: F8) -> F8 {
    let x = 100.0f32 * v;
    let k = (x + x + 0.5).cvt().sra1();
    (1.0f32 / 100.0f32) * k.to_f()
}

#[inline(always)]
fn grid_velocity(v: F8, scale: f32) -> F8 {
    let k = (scale * v).cvtt();
    let inv = 1.0f32 / scale;
    let f = inv * k.to_f();
    let zero = F8::splat(0.0);
    let sign = sel(f.le(zero), sel(zero.le(f), zero, F8::splat(-1.0)), F8::splat(1.0));
    sign * (inv * 0.1f32) + f
}

#[inline(always)]
fn quat_one(v: F8) -> I8 {
    const MAX: i32 = (1 << 18) - 1;
    let f = (v * f32::from_bits(0x3fb504f3) * 0.5 + 0.5) * MAX as f32;
    (f + f + 0.5).cvt().sra1().clamp(0, MAX)
}

#[inline(always)]
fn quat_back(k: I8) -> F8 {
    const MAX: f32 = ((1 << 18) - 1) as f32;
    let f = k.to_f() / MAX - 0.5;
    (f + f) * f32::from_bits(0x3f3504f3)
}

#[inline(always)]
fn quat_codec(q: [F8; 4]) -> [F8; 4] {
    let [qx, qy, qz, qw] = q;
    let n2 = qy * qy + qx * qx + qz * qz + qw * qw;
    let norm = F8::splat(1e-08).lt(n2);
    let k = 1.0 / n2.sqrt();
    let w = sel(norm, k * qw, F8::splat(1.0));
    let x = sel(norm, k * qx, F8::splat(0.0));
    let y = sel(norm, k * qy, F8::splat(0.0));
    let z = sel(norm, k * qz, F8::splat(0.0));
    let (ax, ay, az, aw) = (x.abs(), y.abs(), z.abs(), w.abs());

    let p1 = ax.lt(ay);
    let a = sel(p1, ay, ax);
    let p3 = az.lt(aw);
    let b = sel(p3, aw, az);
    let first = b.le(a);
    let w0 = first.and(p1.not());
    let w1 = first.and(p1);
    let w2 = first.not().and(p3.not());

    let c0 = sel(w0, y, x);
    let c1 = sel(w0.or(w1), z, y);
    let c2 = sel(first.or(w2), w, z);
    let big = sel(w0, x, sel(w1, y, sel(w2, z, w)));
    let flip = big.lt(F8::splat(0.0));
    let (k0, k1, k2) = (quat_one(sel(flip, c0 * -1.0, c0)), quat_one(sel(flip, c1 * -1.0, c1)), quat_one(sel(flip, c2 * -1.0, c2)));
    let (a, b, c) = (quat_back(k0), quat_back(k1), quat_back(k2));
    let e = 1.0 - a * a - b * b - c * c;
    let big = F8::from_d(e.lo().sqrt(), e.hi().sqrt());

    let x = sel(w0, big, a);
    let y = sel(w0, a, sel(w1, big, b));
    let z = sel(first, b, sel(w2, big, c));
    let w = sel(first.or(w2), c, big);
    let n2 = x * x + y * y + z * z + w * w;
    let id = n2.le(F8::splat(1e-08));
    let k = 1.0 / n2.sqrt();
    [sel(id, F8::splat(0.0), x * k), sel(id, F8::splat(0.0), y * k), sel(id, F8::splat(0.0), z * k), sel(id, F8::splat(1.0), w * k)]
}

#[inline(always)]
pub fn read_awake(o: &[F8; 3], m: &[F8; 9], lin: &[F8; 3], ang: &[F8; 3], max_lin: F8, max_ang: F8, quantize: bool) -> ([F8; 3], [F8; 4], [F8; 3], [F8; 3]) {
    let mut loc = [o[0] * 50.0, o[1] * 50.0, o[2] * 50.0];
    let mut quat = mat_to_quat(m);
    let vel = [lin[0] * 50.0, lin[1] * 50.0, lin[2] * 50.0];
    let (mut vel, _) = capped(&vel, max_lin);
    let (mut spin, _) = capped(ang, max_ang);
    if quantize {
        loc = [grid_location(loc[0]), grid_location(loc[1]), grid_location(loc[2])];
        quat = quat_codec(quat);
        vel = [grid_velocity(vel[0], 100.0), grid_velocity(vel[1], 100.0), grid_velocity(vel[2], 100.0)];
        spin = [grid_velocity(spin[0], 10000.0), grid_velocity(spin[1], 10000.0), grid_velocity(spin[2], 10000.0)];
    }
    (loc, quat, vel, spin)
}

#[inline(always)]
pub fn damp(lin: &[F8; 3], ang: &[F8; 3], max_lin: F8, max_ang: F8, d_lin: F8, d_ang: F8) -> ([F8; 3], [F8; 3]) {
    let l = speed_limit(lin, max_lin);
    let a = speed_limit(ang, max_ang);
    ([d_lin * l[0], d_lin * l[1], d_lin * l[2]], [d_ang * a[0], d_ang * a[1], d_ang * a[2]])
}

#[inline(always)]
fn speed_limit(v: &[F8; 3], m: F8) -> [F8; 3] {
    let l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let over = F8::splat(0.0).lt(m).and((m * m).lt(l2));
    let k = 1.0 / l2.sqrt();
    [sel(over, v[0] * k * m, v[0]), sel(over, v[1] * k * m, v[1]), sel(over, v[2] * k * m, v[2])]
}
