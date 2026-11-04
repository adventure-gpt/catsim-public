use super::collision_world::CollisionWorld;
use super::shapes::Shape;
use super::vehicle::{RayCandidates, RayHit, VehicleWorld, WheelInfo};
use crate::math::{Transform,Vec3,v3};
use core::arch::x86_64::*;

pub fn plane(t:&Transform,from:[Vec3;4],to:[Vec3;4])->[Option<RayHit>;4] {
    unsafe {
        let pack=|a:[f32;4]|_mm_loadu_ps(a.as_ptr());
        let r=&t.basis.r;
        let normal=v3(r[0].y*0.0+r[0].x*0.0+r[0].z,
            r[1].y*0.0+r[1].x*0.0+r[1].z,r[2].y*0.0+r[2].x*0.0+r[2].z);
        let f=[pack(from.map(|p|p.x)),pack(from.map(|p|p.y)),pack(from.map(|p|p.z))];
        let g=[pack(to.map(|p|p.x)),pack(to.map(|p|p.y)),pack(to.map(|p|p.z))];
        let n=[_mm_set1_ps(normal.x),_mm_set1_ps(normal.y),_mm_set1_ps(normal.z)];
        let o=[_mm_set1_ps(t.origin.x),_mm_set1_ps(t.origin.y),_mm_set1_ps(t.origin.z)];
        let dot=|end:[__m128;3]|_mm_add_ps(_mm_add_ps(
            _mm_mul_ps(_mm_sub_ps(end[1],f[1]),n[1]),
            _mm_mul_ps(_mm_sub_ps(end[0],f[0]),n[0])),
            _mm_mul_ps(_mm_sub_ps(end[2],f[2]),n[2]));
        let span=_mm_xor_ps(dot(g),_mm_set1_ps(-0.0));
        let reach=_mm_xor_ps(dot(o),_mm_set1_ps(-0.0));
        let eps=_mm_set1_ps(f32::from_bits(0x3400_0000));
        let fraction=_mm_div_ps(reach,span);
        let mask=_mm_and_ps(_mm_and_ps(_mm_cmplt_ps(eps,span),_mm_cmplt_ps(eps,reach)),
            _mm_and_ps(_mm_cmple_ps(reach,span),_mm_cmplt_ps(fraction,_mm_set1_ps(1.0))));
        let valid=_mm_movemask_ps(mask)as u32;
        let s=_mm_sub_ps(_mm_set1_ps(1.0),fraction);
        let mut points=[[0.;4];3];
        for k in 0..3 {_mm_storeu_ps(points[k].as_mut_ptr(),_mm_add_ps(_mm_mul_ps(s,f[k]),_mm_mul_ps(fraction,g[k])));}
        let mut fractions=[0.;4];_mm_storeu_ps(fractions.as_mut_ptr(),fraction);
        std::array::from_fn(|i|if valid&(1<<i)!=0 {
            Some(RayHit{fraction:fractions[i],normal,point:v3(points[0][i],points[1][i],points[2][i])})
        }else{None})
    }
}

pub fn cast(world:&CollisionWorld,wheels:&mut[WheelInfo;4],cands:&RayCandidates) {
    let from=std::array::from_fn(|i|wheels[i].hard_point);
    let to=std::array::from_fn(|i|wheels[i].contact);
    let mut best=[1.0;4];let mut hits=[None;4];
    for &id in &cands.objects[..cands.len] {
        let body=&world.bodies[id.0 as usize];
        let rays=match &world.shapes[body.shape.0 as usize] {
            Shape::Plane(_)=>plane(&body.transform,from,to),
            _=>std::array::from_fn(|i|VehicleWorld::ray_test(world,from[i],to[i],id)),
        };
        for i in 0..4 {if let Some(h)=rays[i] {
            if h.fraction<best[i] {
                let(dx,dy,dz)=(h.point.x-from[i].x,h.point.y-from[i].y,h.point.z-from[i].z);
                if wheels[i].min_hit_dist2<=dx*dx+dy*dy+dz*dz {
                    best[i]=h.fraction;hits[i]=Some((id,h));
                }
            }
        }}
    }
    for i in 0..4 {match hits[i] {
        Some((id,h))=>{wheels[i].contact=h.point;wheels[i].normal=h.normal;wheels[i].in_contact=true;wheels[i].ground=Some(id);}
        None=>{wheels[i].in_contact=false;wheels[i].ground=None;}
    }}
}
