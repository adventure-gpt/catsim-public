use super::{Mat3, Vec3, v3};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
pub fn rows_yx(b: &Mat3, v: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let x = _mm_mul_ps(
            _mm_set1_ps(v.x),
            _mm_set_ps(0.0, b.r[2].x, b.r[1].x, b.r[0].x),
        );
        let y = _mm_mul_ps(
            _mm_set1_ps(v.y),
            _mm_set_ps(0.0, b.r[2].y, b.r[1].y, b.r[0].y),
        );
        let z = _mm_mul_ps(
            _mm_set1_ps(v.z),
            _mm_set_ps(0.0, b.r[2].z, b.r[1].z, b.r[0].z),
        );
        let sum = _mm_add_ps(_mm_add_ps(y, x), z);
        let mut out = [0.0; 4];
        _mm_storeu_ps(out.as_mut_ptr(), sum);
        v3(out[0], out[1], out[2])
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let f = |r: Vec3| v.y * r.y + v.x * r.x + v.z * r.z;
        v3(f(b.r[0]), f(b.r[1]), f(b.r[2]))
    }
}

#[inline(always)]
pub fn solver_impulse(lin: &mut Vec3, ang: &mut Vec3, mass: Vec3, normal: Vec3,
    lin_factor: Vec3, ang_factor: Vec3, axis: Vec3, impulse: f32, x_normal_first: bool) -> bool {
    let old_lin=lin.bits();let old_ang=ang.bits();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let pack = |v:Vec3| _mm_set_ps(0.0,v.z,v.y,v.x);
        let (left,right)=if x_normal_first {
            (v3(normal.x,mass.y,mass.z),v3(mass.x,normal.y,normal.z))
        } else { (mass,normal) };
        let d=_mm_set1_ps(impulse);
        let l=_mm_add_ps(_mm_mul_ps(_mm_mul_ps(_mm_mul_ps(pack(left),pack(right)),d),pack(lin_factor)),pack(*lin));
        let a=_mm_add_ps(_mm_mul_ps(_mm_mul_ps(d,pack(ang_factor)),pack(axis)),pack(*ang));
        let mut out=[0.0f32;4];_mm_storeu_ps(out.as_mut_ptr(),l);*lin=v3(out[0],out[1],out[2]);
        _mm_storeu_ps(out.as_mut_ptr(),a);*ang=v3(out[0],out[1],out[2]);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        lin.x=(if x_normal_first {normal.x*mass.x}else{mass.x*normal.x})*impulse*lin_factor.x+lin.x;
        lin.y=mass.y*normal.y*impulse*lin_factor.y+lin.y;
        lin.z=mass.z*normal.z*impulse*lin_factor.z+lin.z;
        ang.x=impulse*ang_factor.x*axis.x+ang.x;
        ang.y=impulse*ang_factor.y*axis.y+ang.y;
        ang.z=impulse*ang_factor.z*axis.z+ang.z;
    }
    lin.bits()!=old_lin || ang.bits()!=old_ang
}
