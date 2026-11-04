use super::crt::{atan2f, sinf};
use super::{Mat3, v3};
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
pub struct Rotator {
    pub pitch: i32,
    pub yaw: i32,
    pub roll: i32,
}

static TABLE: OnceLock<Box<[f32; 16384]>> = OnceLock::new();

pub fn sine_table() -> &'static [f32; 16384] {
    TABLE.get_or_init(|| {
        let mut t = Box::new([0.0f32; 16384]);
        for i in 0..16384i32 {
            let f = i as f32;
            t[i as usize] = sinf((f + f) * f32::from_bits(0x40490fdb) * f32::from_bits(0x38800000));
        }
        t
    })
}

#[inline]
pub fn tsin(t: &[f32; 16384], units: i32) -> f32 {
    t[((units as i64 >> 2) as u32 & 0x3fff) as usize]
}

#[inline]
pub fn tcos(t: &[f32; 16384], units: i32) -> f32 {
    t[(((units as i64 + 0x4000) >> 2) as u32 & 0x3fff) as usize]
}

pub fn rotator_to_matrix(r: Rotator) -> Mat3 {
    let t = sine_table();
    let sr = tsin(t, r.roll);
    let sp = tsin(t, r.pitch);
    let sy = tsin(t, r.yaw);
    let cr = tcos(t, r.roll);
    let cp = tcos(t, r.pitch);
    let cy = tcos(t, r.yaw);
    Mat3 {
        r: [
            v3(cy * cp, cp * sy, sp),
            v3(sp * sr * cy - cr * sy, cy * cr + sp * sr * sy, -sr * cp),
            v3(-(cr * sp * cy + sy * sr), cy * sr - cr * sp * sy, cp * cr),
        ],
    }
}

#[inline]
pub fn cvtss2si(x: f32) -> i32 {
    let r = x.round_ties_even();
    if (-2147483648.0..2147483648.0).contains(&r) { r as i32 } else { i32::MIN }
}

#[inline]
pub fn cvttss2si(x: f32) -> i32 {
    if x > -2147483904.0 && x < 2147483648.0 { x as i32 } else { i32::MIN }
}

#[inline]
pub fn app_round(x: f32) -> i32 {
    cvtss2si(x + x + 0.5) >> 1
}

pub fn matrix_to_rotator(m: &Mat3) -> Rotator {
    const PI: f32 = f32::from_bits(0x40490fdb);
    let (fx, fy, fz) = (m.r[0].x, m.r[0].y, m.r[0].z);
    let yaw_u = (atan2f(fy, fx) * 32768.0) / PI;
    let len = f64::from(fy * fy + fx * fx).sqrt() as f32;
    let pitch_u = (atan2f(fz, len) * 32768.0) / PI;
    let mut out = Rotator { pitch: app_round(pitch_u), yaw: app_round(yaw_u), roll: 0 };
    let a = rotator_to_matrix(out).r[1];
    let top = m.r[2].y * a.y + m.r[2].x * a.x + m.r[2].z * a.z;
    let bot = m.r[1].y * a.y + m.r[1].x * a.x + m.r[1].z * a.z;
    out.roll = app_round((atan2f(top, bot) * 32768.0) / PI);
    out
}
