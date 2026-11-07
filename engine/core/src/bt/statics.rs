use super::collision_world::CollisionWorld;
use super::types::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct StaticEntry {
    pub id: u32,
    pub handle: u16,
    pub group: u16,
    pub mask: u16,

    pub plane: Option<[u32; 6]>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StaticTable {

    pub(super) epoch: u64,

    pub(super) ok: bool,

    pub(super) order: Vec<StaticEntry>,

    pub(super) handle_bits: [u64; 4],

    pub(super) body_bits: [u64; 4],

    pub(super) sleeping_bits: [u64; 4],

    pub(super) prefix: bool,

    pub(super) islands_settled: bool,

    pub(super) ss_prefix: usize,

    pub(super) valid_filters: u64,

    snapshot: Vec<(u32, [u32; 12], u32, u32, u32, i32)>,

    pub(super) aabbs_settled: bool,

    pub(super) boxes: Vec<(crate::math::Vec3, crate::math::Vec3)>,

    pub(super) boxes_soa: [Vec<f32>; 6],

    pub clear: super::clear_grid::ClearRef,

    pub clear_nf: super::clear_grid::ClearRef,

    pub floor: Option<u64>,
}

impl StaticTable {
    #[inline(always)]
    fn bit(bits: &[u64; 4], i: usize) -> bool {
        i < 256 && bits[(i >> 6) & 3] >> (i & 63) & 1 != 0
    }
    #[inline(always)]
    pub(super) fn is_static_handle(&self, h: u16) -> bool {
        Self::bit(&self.handle_bits, h as usize)
    }
    #[inline(always)]
    pub(super) fn is_sleeping_static(&self, h: u16) -> bool {
        Self::bit(&self.sleeping_bits, h as usize)
    }
    #[inline(always)]
    pub(super) fn is_static_body(&self, id: u32) -> bool {
        Self::bit(&self.body_bits, id as usize)
    }
}

fn transform_bits(t: &crate::math::Transform) -> [u32; 12] {
    let r = &t.basis.r;
    [
        r[0].x.to_bits(), r[0].y.to_bits(), r[0].z.to_bits(),
        r[1].x.to_bits(), r[1].y.to_bits(), r[1].z.to_bits(),
        r[2].x.to_bits(), r[2].y.to_bits(), r[2].z.to_bits(),
        t.origin.x.to_bits(), t.origin.y.to_bits(), t.origin.z.to_bits(),
    ]
}

impl CollisionWorld {

    fn build_statics(&self) -> StaticTable {
        let mut t = StaticTable { epoch: self.static_epoch, ok: true, ..StaticTable::default() };
        let bp = &self.broadphase;
        if bp.handles.len() > 256 || self.bodies.len() > 256 {
            t.ok = false;
            return t;
        }
        for id in &self.objects {
            let b = &self.bodies[id.0 as usize];
            if b.flags & CF_KINEMATIC != 0 {
                t.ok = false;
            }
            if b.flags & CF_STATIC != 0 {
                if b.proxy == 0 {
                    t.ok = false;
                    continue;
                }
                t.handle_bits[b.proxy as usize >> 6] |= 1 << (b.proxy & 63);
                t.body_bits[id.0 as usize >> 6] |= 1 << (id.0 & 63);
                if b.activation_state == ISLAND_SLEEPING || b.activation_state == DISABLE_SIMULATION {
                    t.sleeping_bits[b.proxy as usize >> 6] |= 1 << (b.proxy & 63);
                }
                t.snapshot.push((id.0, transform_bits(&b.transform), b.shape.0, b.flags, b.proxy, b.activation_state));
            }
        }
        for i in 1..(bp.num_handles as usize * 2 + 1) {
            let e = bp.edges[0][i];
            if e.pos & 1 == 0 || !t.is_static_handle(e.handle) {
                continue;
            }
            let h = &bp.handles[e.handle as usize];
            let b = &self.bodies[h.client as usize];

            let plane = matches!(self.shapes[b.shape.0 as usize], super::shapes::Shape::Plane(_)).then(|| {
                let r = &b.transform.basis.r;
                let nx = r[0].y * 0.0 + r[0].x * 0.0 + r[0].z;
                let ny = r[1].y * 0.0 + r[1].x * 0.0 + r[1].z;
                let nz = r[2].y * 0.0 + r[2].x * 0.0 + r[2].z;
                let o = b.transform.origin;
                [nx.to_bits(), ny.to_bits(), nz.to_bits(), o.x.to_bits(), o.y.to_bits(), o.z.to_bits()]
            });
            t.order.push(StaticEntry { id: h.client, handle: e.handle, group: h.group, mask: h.mask, plane });
        }
        if t.order.len() != t.snapshot.len() {
            t.ok = false;
        }
        let pairs = &bp.pairs.pairs;
        let k = pairs.iter().take_while(|p| t.is_sleeping_static(p.proxy0) && t.is_sleeping_static(p.proxy1)).count();
        t.ss_prefix = if pairs[k..].iter().any(|p| t.is_static_handle(p.proxy0) && t.is_static_handle(p.proxy1)) { 0 } else { k };
        let s = t.order.len();
        t.prefix = self.objects.len() >= s && self.objects.iter().enumerate().all(|(k, id)| (k < s) == (self.bodies[id.0 as usize].flags & CF_STATIC != 0));
        t.valid_filters = self.verified_ray_filters();
        if t.ok {
            use super::clear_grid::StaticGeom;
            let geoms: Option<Vec<StaticGeom>> = t
                .order
                .iter()
                .map(|e| {
                    let b = &self.bodies[e.id as usize];
                    match (&e.plane, &self.shapes[b.shape.0 as usize]) {
                        (Some(p), _) => Some(StaticGeom::Plane {
                            n: [f32::from_bits(p[0]), f32::from_bits(p[1]), f32::from_bits(p[2])],
                            o: [f32::from_bits(p[3]), f32::from_bits(p[4]), f32::from_bits(p[5])],
                        }),
                        (None, super::shapes::Shape::Mesh(m)) => Some(StaticGeom::Mesh { mesh: m, t: b.transform }),
                        _ => None,
                    }
                })
                .collect();

            #[cfg(feature = "profile")]
            let is_floor = |e: &StaticEntry| {
                e.plane.map_or(false, |p| p[0] == 0f32.to_bits() && p[1] == 0f32.to_bits() && p[2] == 1f32.to_bits() && p[5] == 0f32.to_bits())
            };
            #[cfg(feature = "profile")]
            if let Some(g) = geoms.as_ref() {
                let nf: Vec<StaticGeom> = g
                    .iter()
                    .zip(t.order.iter())
                    .filter(|(_, e)| !is_floor(e))
                    .map(|(x, _)| match x {
                        StaticGeom::Plane { n, o } => StaticGeom::Plane { n: *n, o: *o },
                        StaticGeom::Mesh { mesh, t } => StaticGeom::Mesh { mesh, t: *t },
                    })
                    .collect();
                if nf.len() < g.len() {
                    t.clear_nf = super::clear_grid::ClearRef(super::clear_grid::shared(&nf));
                    t.floor = Some(t.order.iter().filter(|e| is_floor(e)).fold(0u64, |m, e| m | (1u64 << (e.id & 63))));
                }
            }
            t.clear = super::clear_grid::ClearRef(geoms.and_then(|g| super::clear_grid::shared(&g)));
        }
        t
    }

    /// After the pair cache was replaced (bt::contacts): where the static-static pairs end, as the table's build finds it.
    pub(super) fn refresh_static_pair_prefix(&mut self) {
        if self.statics.epoch != self.static_epoch || !self.statics.ok {
            return;
        }
        let t = &mut self.statics;
        let pairs = &self.broadphase.pairs.pairs;
        let k = pairs.iter().take_while(|p| t.is_sleeping_static(p.proxy0) && t.is_sleeping_static(p.proxy1)).count();
        t.ss_prefix = if pairs[k..].iter().any(|p| t.is_static_handle(p.proxy0) && t.is_static_handle(p.proxy1)) { 0 } else { k };
    }

    pub fn ensure_statics(&mut self) {
        if self.statics.epoch != self.static_epoch {
            self.statics = self.build_statics();
        }
    }

    #[inline(always)]
    pub(super) fn statics_ready(&self) -> Option<&StaticTable> {
        let t = &self.statics;
        (t.epoch == self.static_epoch && t.ok).then_some(t)
    }

    fn static_box(&self, id: u32) -> (crate::math::Vec3, crate::math::Vec3) {
        let b = &self.bodies[id as usize];
        let (lo, hi) = self.shapes[b.shape.0 as usize].aabb(&b.transform);
        let th = super::dispatch::CONTACT_BREAKING_THRESHOLD;
        (crate::math::v3(lo.x - th, lo.y - th, lo.z - th), crate::math::v3(hi.x + th, hi.y + th, hi.z + th))
    }

    pub(super) fn settle_statics(&mut self) {
        let hs = &self.broadphase.handles;
        self.statics.boxes = self.statics.order.iter().map(|e| (hs[e.handle as usize].aabb_min, hs[e.handle as usize].aabb_max)).collect();
        let n = self.statics.boxes.len().div_ceil(4) * 4;
        let mut soa: [Vec<f32>; 6] = core::array::from_fn(|_| vec![f32::NAN; n]);
        for (i, (lo, hi)) in self.statics.boxes.iter().enumerate() {
            for (k, v) in [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z].into_iter().enumerate() {
                soa[k][i] = v;
            }
        }
        self.statics.boxes_soa = soa;
        self.statics.aabbs_settled = true;
    }

    #[inline]
    pub(super) fn aabb_test_statics(&self, aabb_min: crate::math::Vec3, aabb_max: crate::math::Vec3, visit: &mut dyn FnMut(BodyId, u16, u16)) -> Option<()> {
        let t = self.statics_ready()?;
        if !t.aabbs_settled || t.order.len() > 48 {
            return None;
        }
        let bp = &self.broadphase;
        let mut hits = [(0u16, 0u16); 64];
        let mut n = 0usize;
        let mut dynamic = 0usize;
        let overlap = |lo: &crate::math::Vec3, hi: &crate::math::Vec3| {
            let mut overlap = true;
            if hi.x < aabb_min.x || aabb_max.x < lo.x {
                overlap = false;
            }
            if hi.z < aabb_min.z || aabb_max.z < lo.z {
                overlap = false;
            }
            aabb_min.y <= hi.y && lo.y <= aabb_max.y && overlap
        };
        #[cfg(target_arch = "x86_64")]
        unsafe {

            use core::arch::x86_64::*;
            let s = &t.boxes_soa;
            let (mnx, mny, mnz) = (_mm_set1_ps(aabb_min.x), _mm_set1_ps(aabb_min.y), _mm_set1_ps(aabb_min.z));
            let (mxx, mxy, mxz) = (_mm_set1_ps(aabb_max.x), _mm_set1_ps(aabb_max.y), _mm_set1_ps(aabb_max.z));
            let mut g = 0;
            while g < s[0].len() {
                let ld = |k: usize| _mm_loadu_ps(s[k].as_ptr().add(g));
                let (lx, ly, lz, hx, hy, hz) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
                let xfail = _mm_or_ps(_mm_cmplt_ps(hx, mnx), _mm_cmplt_ps(mxx, lx));
                let zfail = _mm_or_ps(_mm_cmplt_ps(hz, mnz), _mm_cmplt_ps(mxz, lz));
                let yok = _mm_and_ps(_mm_cmple_ps(mny, hy), _mm_cmple_ps(ly, mxy));
                let mut bits = _mm_movemask_ps(_mm_andnot_ps(_mm_or_ps(xfail, zfail), yok)) as u32;
                while bits != 0 {
                    let e = &t.order[g + bits.trailing_zeros() as usize];
                    hits[n] = (bp.handles[e.handle as usize].max_edges[0], e.handle);
                    n += 1;
                    bits &= bits - 1;
                }
                g += 4;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        for (e, (lo, hi)) in t.order.iter().zip(&t.boxes) {
            if overlap(lo, hi) {
                hits[n] = (bp.handles[e.handle as usize].max_edges[0], e.handle);
                n += 1;
            }
        }
        let rest = if t.prefix { &self.objects[t.order.len()..] } else { &self.objects[..] };
        for id in rest {
            if !t.prefix && t.is_static_body(id.0) {
                continue;
            }
            let p = self.bodies[id.0 as usize].proxy;
            if p == 0 {
                continue;
            }
            dynamic += 1;
            let h = &bp.handles[p as usize];
            if overlap(&h.aabb_min, &h.aabb_max) {
                if n == hits.len() {
                    return None;
                }
                hits[n] = (h.max_edges[0], p as u16);
                n += 1;
            }
        }

        if t.order.len() + dynamic != bp.num_handles as usize {
            return None;
        }
        let hits = &mut hits[..n];
        hits.sort_unstable_by_key(|x| x.0);
        for &(_, hi) in hits.iter() {
            let h = &bp.handles[hi as usize];
            visit(BodyId(h.client), h.group, h.mask);
        }
        Some(())
    }

    pub fn clear_grid(&self) -> Option<std::sync::Arc<super::clear_grid::ClearGrid>> {
        self.statics_ready().and_then(|t| t.clear.0.clone())
    }

    #[cfg(feature = "profile")]
    pub fn floor_grids(&self) -> Option<(u64, std::sync::Arc<super::clear_grid::ClearGrid>, std::sync::Arc<super::clear_grid::ClearGrid>)> {
        let t = self.statics_ready()?;
        Some((t.floor?, t.clear.0.clone()?, t.clear_nf.0.clone()?))
    }

    pub fn plant_statics_bug(&mut self) {
        if self.statics.order.len() > 1 {
            self.statics.order.swap(0, 1);
        }
    }

    pub fn verify_statics(&self) -> Result<(), String> {
        let t = &self.statics;
        if t.epoch != self.static_epoch {
            return Ok(());
        }
        let fresh = self.build_statics();
        if fresh.ok != t.ok || fresh.order != t.order || fresh.handle_bits != t.handle_bits || fresh.body_bits != t.body_bits || fresh.sleeping_bits != t.sleeping_bits || fresh.prefix != t.prefix || fresh.ss_prefix != t.ss_prefix || fresh.valid_filters != t.valid_filters || fresh.snapshot != t.snapshot || fresh.clear != t.clear {
            return Err(format!("static table changed: ok {}/{} order {}/{} filters {:x}/{:x}", t.ok, fresh.ok, t.order.len(), fresh.order.len(), t.valid_filters, fresh.valid_filters));
        }
        if t.islands_settled {
            for e in &t.order {
                let b = &self.bodies[e.id as usize];
                if b.island_tag != -1 || b.companion_id != -2 || b.hit_fraction.to_bits() != 1.0f32.to_bits() {
                    return Err(format!("static {} island fields {} {} {}", e.id, b.island_tag, b.companion_id, b.hit_fraction));
                }
            }
        }
        if t.aabbs_settled {
            for (e, (lo, hi)) in t.order.iter().zip(&t.boxes) {
                let h = &self.broadphase.handles[e.handle as usize];
                if lo.bits() != h.aabb_min.bits() || hi.bits() != h.aabb_max.bits() {
                    return Err(format!("static {} table box differs from its handle", e.id));
                }
                let k = t.order.iter().position(|o| o == e).unwrap();
                let soa = [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z];
                if (0..6).any(|j| t.boxes_soa[j][k].to_bits() != soa[j].to_bits()) {
                    return Err(format!("static {} table box lanes differ", e.id));
                }
            }
            for e in &t.order {
                let b = &self.bodies[e.id as usize];
                if b.activation_state == DISABLE_SIMULATION && !self.force_update_all_aabbs {
                    continue;
                }
                let (lo, hi) = self.static_box(e.id);
                let h = &self.broadphase.handles[e.handle as usize];
                if lo.bits() != h.aabb_min.bits() || hi.bits() != h.aabb_max.bits() {
                    return Err(format!("static {} box moved", e.id));
                }
                let (qlo, qhi) = (self.broadphase.quantize(lo, 0), self.broadphase.quantize(hi, 1));
                for a in 0..3 {
                    if self.broadphase.edges[a][h.min_edges[a] as usize].pos != qlo[a] || self.broadphase.edges[a][h.max_edges[a] as usize].pos != qhi[a] {
                        return Err(format!("static {} edges moved on axis {a}", e.id));
                    }
                }
            }
        }
        Ok(())
    }
}
