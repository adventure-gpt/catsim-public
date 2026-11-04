use super::{algo_convex::transform_inverse, bvh::{Bvh, QNode}};
use crate::math::{Transform, Vec3};

#[derive(Clone,Debug)]
pub(super) struct FilterLanes {lower:[[f32;64];3],upper:[[f32;64];3]}
impl Default for FilterLanes {
    fn default()->Self {Self{lower:[[0.;64];3],upper:[[0.;64];3]}}
}
impl FilterLanes {
    pub fn set(&mut self,id:usize,filter:&MeshFilter) {
        if id<64 {for a in 0..3 {self.lower[a][id]=filter.root.lower[a];self.upper[a][id]=filter.root.upper[a];}}
    }
    #[inline]
    pub fn reject(&self,from:Vec3,to:Vec3,valid:u64)->u64 {
        #[cfg(target_arch="x86_64")]
        unsafe {
            use core::arch::x86_64::*;
            let mut result=0;let mut remaining=valid;
            while remaining!=0 {
                let offset=(remaining.trailing_zeros() as usize)&!3;
                let mut mask=_mm_setzero_ps();
                for a in 0..3 {
                    let f=_mm_set1_ps(from[a]);let t=_mm_set1_ps(to[a]);
                    let lower=_mm_loadu_ps(self.lower[a].as_ptr().add(offset));
                    let upper=_mm_loadu_ps(self.upper[a].as_ptr().add(offset));
                    mask=_mm_or_ps(mask,_mm_or_ps(
                        _mm_and_ps(_mm_cmplt_ps(f,lower),_mm_cmplt_ps(t,lower)),
                        _mm_and_ps(_mm_cmpgt_ps(f,upper),_mm_cmpgt_ps(t,upper))));
                }
                result|=(_mm_movemask_ps(mask)as u64)<<offset;
                remaining&=!(15u64<<offset);
            }
            result&valid
        }
        #[cfg(not(target_arch="x86_64"))]
        {let mut result=0;for i in 0..64 {if valid&(1u64<<i)!=0 {
            if (0..3).any(|a|(from[a]<self.lower[a][i]&&to[a]<self.lower[a][i])
                ||(from[a]>self.upper[a][i]&&to[a]>self.upper[a][i])) {result|=1u64<<i;}
        }}result}
    }
}

#[derive(Clone, Debug)]
pub(super) struct MeshFilter {
    mesh: std::sync::Arc<super::bvh::MeshShape>,
    root: RootFilter,
}
impl MeshFilter {
    pub fn new(mesh:&std::sync::Arc<super::bvh::MeshShape>,t:&Transform)->Option<Self> {
        Some(Self{mesh:mesh.clone(),root:RootFilter::new(&mesh.bvh,t)?})
    }
    #[inline]
    pub fn matches(&self,mesh:&std::sync::Arc<super::bvh::MeshShape>,t:&Transform)->bool {
        std::sync::Arc::ptr_eq(&self.mesh,mesh) && self.root.transform==bits(t)
    }
    #[inline(always)]
    pub fn rejects_finite(&self,from:Vec3,to:Vec3)->bool {self.root.rejects_finite(from,to)}
    #[inline]
    pub fn rejects(&self,from:Vec3,to:Vec3,mesh:&std::sync::Arc<super::bvh::MeshShape>,t:&Transform)->bool {
        self.rejects_finite(from,to) && from.x.is_finite() && from.y.is_finite() && from.z.is_finite()
            && to.x.is_finite() && to.y.is_finite() && to.z.is_finite() && self.matches(mesh,t)
    }
}

#[derive(Clone, Debug)]
pub struct RootFilter {
    transform: [u32; 12],
    min: [u32; 3],
    max: [u32; 3],
    quant: [u32; 3],
    root: QNode,
    count: i32,
    lower: Vec3,
    upper: Vec3,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::v3;
    #[test]
    fn root_lanes_preserve_all_predicates_and_masks() {
        let mut seed=3401u32;let mut next=||{seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;seed};
        let mut lanes=FilterLanes::default();
        for a in 0..3 {for i in 0..64 {lanes.lower[a][i]=f32::from_bits(next());lanes.upper[a][i]=f32::from_bits(next());}}
        for case in 0..100000 {
            let mut f=v3(f32::from_bits(next()),f32::from_bits(next()),f32::from_bits(next()));
            let mut t=v3(f32::from_bits(next()),f32::from_bits(next()),f32::from_bits(next()));
            if case%4==0 {let i=(case/4)%64;f=v3(lanes.lower[0][i],lanes.lower[1][i],lanes.lower[2][i]);t=f;}
            let valid=((next()as u64)<<32)|next()as u64;
            let mut expected=0;
            for i in 0..64 {if valid&(1u64<<i)!=0 && (0..3).any(|a|
                (f[a]<lanes.lower[a][i]&&t[a]<lanes.lower[a][i])||(f[a]>lanes.upper[a][i]&&t[a]>lanes.upper[a][i])) {expected|=1u64<<i;}}
            assert_eq!(lanes.reject(f,t,valid),expected,"case={case}");
        }
    }
    fn root_hit(b:&Bvh,t:&Transform,f:Vec3,g:Vec3)->bool {
        let inv=transform_inverse(t);let r=&inv.basis.r;
        let local=|p:Vec3| v3(p.x*r[0].x+p.y*r[0].y+p.z*r[0].z+inv.origin.x,
            p.x*r[1].x+p.y*r[1].y+p.z*r[1].z+inv.origin.y,
            p.x*r[2].x+p.y*r[2].y+p.z*r[2].z+inv.origin.z);
        let (f,g)=(local(f),local(g));
        let lo=v3(if g.x<f.x {g.x}else{f.x},if g.y<f.y {g.y}else{f.y},if g.z<f.z {g.z}else{f.z});
        let hi=v3(if f.x<g.x {g.x}else{f.x},if f.y<g.y {g.y}else{f.y},if f.z<g.z {g.z}else{f.z});
        let (lo,hi)=b.quantize_query(lo+Vec3::ZERO,hi+Vec3::ZERO);let n=&b.nodes[0];
        n.min[1]<=hi[1] && lo[2]<=n.max[2] && lo[1]<=n.max[1]
            && n.min[0]<=hi[0] && n.min[2]<=hi[2] && lo[0]<=n.max[0]
    }
    #[test]
    fn exact_root_filter_boundaries_and_mutation() {
        let mut seed=913u32;
        let mut rand=||{seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;seed};
        let mut checked=0;
        for case in 0..128 {
            let mut t=Transform::IDENTITY;t.origin=v3(case as f32-64.0,102.4,-0.0);
            t.basis.r[0].y=-0.0;t.basis.r[2].x=-0.0;
            let mut b=Bvh{aabb_min:v3(-130.0,-140.0,-20.0),aabb_max:v3(130.0,140.0,80.0),
                quantization:v3(250.0,230.0,650.0),cur_node_index:1,nodes:vec![QNode::default()]};
            for a in 0..3 { let x=(rand()%60000) as u16;let y=(rand()%60000) as u16;
                b.nodes[0].min[a]=x.min(y)&0xfffe;b.nodes[0].max[a]=x.max(y)|1; }
            let filter=RootFilter::new(&b,&t).unwrap();
            let mut check=|f,g| {let rejected=filter.rejects(f,g,&b,&t);
                assert!(!rejected || !root_hit(&b,&t,f,g),"false rejection {f:?} {g:?}");checked+=1;};
            for _ in 0..512 { let f=v3(f32::from_bits(rand()),f32::from_bits(rand()),f32::from_bits(rand()));
                let g=v3(f32::from_bits(rand()),f32::from_bits(rand()),f32::from_bits(rand()));check(f,g); }
            let center=(filter.lower+filter.upper)*0.5;
            for a in 0..3 {for edge in [filter.lower[a],filter.upper[a]] {for p in [edge.next_down(),edge,edge.next_up()] {
                let mut f=center;f[a]=p;check(f,f);check(f,center);
            }}}
            let far=v3(f32::MAX,f32::MAX,f32::MAX);
            let mut changed=t;changed.origin.x=changed.origin.x.next_up();
            assert!(!filter.rejects(far,far,&b,&changed));
            b.nodes[0].max[0]^=2;assert!(!filter.rejects(far,far,&b,&t));
        }
        eprintln!("root filter checked {checked} random/boundary segments and 256 stale-cache cases");
    }

    #[test]
    fn mesh_identity_rejects_copy_on_write_and_transform_changes() {
        use std::sync::Arc;
        let mut mesh=Arc::new(super::super::bvh::MeshShape::new(
            vec![v3(-1.,-1.,0.),v3(1.,-1.,0.),v3(0.,1.,0.)],vec![[0,1,2]]));
        let t=Transform::IDENTITY;
        let filter=MeshFilter::new(&mesh,&t).unwrap();
        assert!(filter.matches(&mesh,&t));
        let mut changed=t;changed.origin.x=f32::from_bits(1);
        assert!(!filter.matches(&mesh,&changed));
        let original=mesh.clone();
        Arc::make_mut(&mut mesh).bvh.nodes[0].min[0]^=2;
        assert!(!filter.matches(&mesh,&t));
        assert!(filter.matches(&original,&t));
    }
}

fn bits(t: &Transform) -> [u32; 12] {
    let r = &t.basis.r;
    [r[0].x.to_bits(),r[0].y.to_bits(),r[0].z.to_bits(),
     r[1].x.to_bits(),r[1].y.to_bits(),r[1].z.to_bits(),
     r[2].x.to_bits(),r[2].y.to_bits(),r[2].z.to_bits(),
     t.origin.x.to_bits(),t.origin.y.to_bits(),t.origin.z.to_bits()]
}

fn from_key(k: u32) -> f32 {
    f32::from_bits(if k & 0x8000_0000 != 0 { k ^ 0x8000_0000 } else { !k })
}
fn first_true(mut f: impl FnMut(f32) -> bool) -> Option<f32> {
    let (mut l, mut h) = (0x0080_0000u32, 0xff7f_ffffu32);
    if !f(from_key(h)) { return None; }
    while l < h {
        let m = l + (h-l)/2;
        if f(from_key(m)) { h=m; } else { l=m+1; }
    }
    Some(from_key(l))
}

impl RootFilter {
    pub fn new(b: &Bvh, t: &Transform) -> Option<Self> {
        if b.cur_node_index <= 0 { return None; }
        for a in 0..3 {
            for c in 0..3 {
                if t.basis.r[a][c] != if a==c { 1.0 } else { 0.0 } { return None; }
            }
            let (lo,hi,q)=(b.aabb_min[a],b.aabb_max[a],b.quantization[a]);

            let top=(hi-lo)*q;
            if !lo.is_finite() || !hi.is_finite() || !q.is_finite() || lo>hi || q<=0.0
                || !top.is_finite() || top<0.0 || top+1.0>=65536.0 { return None; }
        }
        let inv=transform_inverse(t);
        if !inv.origin.x.is_finite() || !inv.origin.y.is_finite() || !inv.origin.z.is_finite() { return None; }
        let root=b.nodes[0];
        let mut lower=Vec3::ZERO;
        let mut upper=Vec3::ZERO;
        for a in 0..3 {
            let quant = |p:f32, high:bool| {

                let local=(p+0.0)+0.0+inv.origin[a];
                let local=local+0.0;
                let c=if local<b.aabb_min[a] { b.aabb_min[a] } else { local };
                let c=if b.aabb_max[a]<c { b.aabb_max[a] } else { c };
                let q=(c-b.aabb_min[a])*b.quantization[a];
                if high { ((q+1.0) as i32 as u16)|1 } else { (q as i32 as u16)&0xfffe }
            };
            lower[a]=first_true(|p| root.min[a]<=quant(p,true))?;

            upper[a]=-first_true(|p| quant(-p,false)<=root.max[a])?;
        }
        Some(Self { transform:bits(t),min:b.aabb_min.bits(),max:b.aabb_max.bits(),
            quant:b.quantization.bits(),root,count:b.cur_node_index,lower,upper })
    }

    #[inline]
    pub fn rejects(&self, from:Vec3, to:Vec3, b:&Bvh, t:&Transform) -> bool {
        self.rejects_finite(from,to) && from.x.is_finite() && from.y.is_finite() && from.z.is_finite()
            && to.x.is_finite() && to.y.is_finite() && to.z.is_finite() && self.matches(b,t)
    }

    #[inline(always)]
    pub(super) fn rejects_finite(&self, from:Vec3,to:Vec3)->bool {
        (from.x<self.lower.x && to.x<self.lower.x)
            || (from.x>self.upper.x && to.x>self.upper.x)
            || (from.y<self.lower.y && to.y<self.lower.y)
            || (from.y>self.upper.y && to.y>self.upper.y)
            || (from.z<self.lower.z && to.z<self.lower.z)
            || (from.z>self.upper.z && to.z>self.upper.z)
    }

    pub(super) fn matches(&self,b:&Bvh,t:&Transform)->bool {
        self.transform==bits(t) && self.min==b.aabb_min.bits()
            && self.max==b.aabb_max.bits() && self.quant==b.quantization.bits()
            && self.count==b.cur_node_index && b.nodes.first()==Some(&self.root)
    }
}
