use super::dispatch::Dispatcher;
use super::pairs::PairCache;
use crate::math::{Vec3, v3};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Edge {
    pub pos: u16,
    pub handle: u16,
}

impl Edge {
    #[inline(always)]
    fn is_max(self) -> bool {
        self.pos & 1 != 0
    }
}

#[derive(Clone, Debug)]
pub struct Handle {

    pub client: u32,
    pub group: u16,
    pub mask: u16,
    pub unique_id: i32,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,

    pub min_edges: [u16; 3],
    pub max_edges: [u16; 3],
}

impl Default for Handle {
    fn default() -> Handle {
        Handle { client: 0, group: 0, mask: 0, unique_id: 0, aabb_min: Vec3::ZERO, aabb_max: Vec3::ZERO, min_edges: [0; 3], max_edges: [0; 3] }
    }
}

#[derive(Clone, Debug)]
pub struct AxisSweep {
    pub handle_mask: u16,
    pub handle_sentinel: u16,
    pub world_min: Vec3,
    pub world_max: Vec3,
    pub quantize_scale: Vec3,
    pub num_handles: u16,
    pub max_handles: u16,
    pub handles: Vec<Handle>,
    pub first_free: u16,
    pub edges: [Vec<Edge>; 3],
    pub pairs: PairCache,
}

pub const WORLD_LIMIT: f32 = f32::from_bits(0x4623_d6f6);

impl AxisSweep {

    pub fn new_game() -> AxisSweep {
        AxisSweep::new(v3(-WORLD_LIMIT, -WORLD_LIMIT, -WORLD_LIMIT), v3(WORLD_LIMIT, WORLD_LIMIT, WORLD_LIMIT), 0x4000)
    }

    pub fn new(world_min: Vec3, world_max: Vec3, user_max_handles: u16) -> AxisSweep {
        let handle_mask = 0xfffeu16;
        let handle_sentinel = 0xffffu16;
        let max_handles = user_max_handles + 1;
        let sentinel_f = handle_sentinel as f32;
        let quantize_scale = v3(
            sentinel_f / (world_max.x - world_min.x),
            sentinel_f / (world_max.y - world_min.y),
            sentinel_f / (world_max.z - world_min.z),
        );
        let n = max_handles as usize;
        let mut handles = vec![Handle::default(); n];

        for i in 1..n {
            handles[i].min_edges[0] = (i + 1) as u16;
        }
        handles[n - 1].min_edges[0] = 0;

        handles[0].min_edges = [0; 3];
        handles[0].max_edges = [1; 3];
        let make_edges = || {
            let mut e = vec![Edge::default(); n * 2];
            e[0] = Edge { pos: 0, handle: 0 };
            e[1] = Edge { pos: handle_sentinel, handle: 0 };
            e
        };
        AxisSweep {
            handle_mask,
            handle_sentinel,
            world_min,
            world_max,
            quantize_scale,
            num_handles: 0,
            max_handles,
            handles,
            first_free: 1,
            edges: [make_edges(), make_edges(), make_edges()],
            pairs: PairCache::new(),
        }
    }

    pub fn quantize(&self, point: Vec3, is_max: u16) -> [u16; 3] {
        let v = (point - self.world_min).comp_mul(self.quantize_scale);
        let one = |c: f32| -> u16 {
            if 0.0 < c {
                let q = if c < self.handle_sentinel as f32 { c as i32 as u16 } else { self.handle_sentinel };
                (self.handle_mask & q) | is_max
            } else {
                is_max
            }
        };
        [one(v.x), one(v.y), one(v.z)]
    }

    #[inline]
    fn alloc_handle(&mut self) -> u16 {
        let h = self.first_free;
        self.first_free = self.handles[h as usize].min_edges[0];
        self.num_handles += 1;
        h
    }

    #[inline(always)]
    fn test_overlap_2d(a: &Handle, b: &Handle, axis1: usize, axis2: usize) -> bool {
        b.min_edges[axis1] <= a.max_edges[axis1]
            && a.min_edges[axis1] <= b.max_edges[axis1]
            && b.min_edges[axis2] <= a.max_edges[axis2]
            && a.min_edges[axis2] <= b.max_edges[axis2]
    }

    fn sort_min_down(&mut self, axis: usize, edge: u16, update_overlaps: bool) {
        let mut e = edge as usize;
        let h0 = self.edges[axis][e].handle as usize;
        let axis1 = (1usize << axis) & 3;
        let axis2 = (1usize << axis1) & 3;
        while self.edges[axis][e].pos < self.edges[axis][e - 1].pos {
            let prev = self.edges[axis][e - 1];
            let h1 = prev.handle as usize;
            if prev.is_max() {
                if update_overlaps && Self::test_overlap_2d(&self.handles[h0], &self.handles[h1], axis1, axis2) {
                    self.pairs.add_overlapping_pair(&self.handles, h0 as u16, h1 as u16);
                }
                self.handles[h1].max_edges[axis] += 1;
            } else {
                self.handles[h1].min_edges[axis] += 1;
            }
            self.handles[h0].min_edges[axis] -= 1;
            self.edges[axis].swap(e, e - 1);
            e -= 1;
        }
    }

    fn sort_min_up(&mut self, axis: usize, edge: u16, disp: &mut Dispatcher, update_overlaps: bool) {
        let mut e = edge as usize;
        let h0 = self.edges[axis][e].handle as usize;
        let axis1 = (1usize << axis) & 3;
        let axis2 = (1usize << axis1) & 3;
        while self.edges[axis][e + 1].handle != 0 && self.edges[axis][e].pos >= self.edges[axis][e + 1].pos {
            let next = self.edges[axis][e + 1];
            let h1 = next.handle as usize;
            if next.is_max() {
                if update_overlaps && Self::test_overlap_2d(&self.handles[h0], &self.handles[h1], axis1, axis2) {
                    self.pairs.remove_overlapping_pair(h0 as u16, h1 as u16, disp);
                }
                self.handles[h1].max_edges[axis] -= 1;
            } else {
                self.handles[h1].min_edges[axis] -= 1;
            }
            self.handles[h0].min_edges[axis] += 1;
            self.edges[axis].swap(e, e + 1);
            e += 1;
        }
    }

    fn sort_max_down(&mut self, axis: usize, edge: u16, disp: &mut Dispatcher, update_overlaps: bool) {
        let mut e = edge as usize;
        let h0 = self.edges[axis][e].handle as usize;
        let axis1 = (1usize << axis) & 3;
        let axis2 = (1usize << axis1) & 3;
        while self.edges[axis][e].pos < self.edges[axis][e - 1].pos {
            let prev = self.edges[axis][e - 1];
            let h1 = prev.handle as usize;
            if !prev.is_max() {
                if update_overlaps && Self::test_overlap_2d(&self.handles[h0], &self.handles[h1], axis1, axis2) {
                    self.pairs.remove_overlapping_pair(h0 as u16, h1 as u16, disp);
                }
                self.handles[h1].min_edges[axis] += 1;
            } else {
                self.handles[h1].max_edges[axis] += 1;
            }
            self.handles[h0].max_edges[axis] -= 1;
            self.edges[axis].swap(e, e - 1);
            e -= 1;
        }
    }

    fn sort_max_up(&mut self, axis: usize, edge: u16, update_overlaps: bool) {
        let mut e = edge as usize;
        let h0 = self.edges[axis][e].handle as usize;
        let axis1 = (1usize << axis) & 3;
        let axis2 = (1usize << axis1) & 3;
        while self.edges[axis][e + 1].handle != 0 && self.edges[axis][e].pos >= self.edges[axis][e + 1].pos {
            let next = self.edges[axis][e + 1];
            let h1 = next.handle as usize;
            if !next.is_max() {
                if update_overlaps && Self::test_overlap_2d(&self.handles[h0], &self.handles[h1], axis1, axis2) {
                    self.pairs.add_overlapping_pair(&self.handles, h0 as u16, h1 as u16);
                }
                self.handles[h1].min_edges[axis] -= 1;
            } else {
                self.handles[h1].max_edges[axis] -= 1;
            }
            self.handles[h0].max_edges[axis] += 1;
            self.edges[axis].swap(e, e + 1);
            e += 1;
        }
    }

    pub fn add_handle(&mut self, aabb_min: Vec3, aabb_max: Vec3, client: u32, group: u16, mask: u16, disp: &mut Dispatcher) -> u16 {
        let qmin = self.quantize(aabb_min, 0);
        let qmax = self.quantize(aabb_max, 1);
        let h = self.alloc_handle();
        {
            let hd = &mut self.handles[h as usize];
            hd.unique_id = h as i32;
            hd.client = client;
            hd.group = group;
            hd.mask = mask;
        }
        let limit = (self.num_handles as usize) * 2;
        for axis in 0..3 {
            self.handles[0].max_edges[axis] += 2;
            self.edges[axis][limit + 1] = self.edges[axis][limit - 1];
            self.edges[axis][limit - 1] = Edge { pos: qmin[axis], handle: h };
            self.edges[axis][limit] = Edge { pos: qmax[axis], handle: h };
            self.handles[h as usize].min_edges[axis] = (limit - 1) as u16;
            self.handles[h as usize].max_edges[axis] = limit as u16;
        }
        let e = self.handles[h as usize].min_edges[0];
        self.sort_min_down(0, e, false);
        let e = self.handles[h as usize].max_edges[0];
        self.sort_max_down(0, e, disp, false);
        let e = self.handles[h as usize].min_edges[1];
        self.sort_min_down(1, e, false);
        let e = self.handles[h as usize].max_edges[1];
        self.sort_max_down(1, e, disp, false);
        let e = self.handles[h as usize].min_edges[2];
        self.sort_min_down(2, e, true);
        let e = self.handles[h as usize].max_edges[2];
        self.sort_max_down(2, e, disp, true);
        h
    }

    pub fn create_proxy(&mut self, aabb_min: Vec3, aabb_max: Vec3, client: u32, group: u16, mask: u16, disp: &mut Dispatcher) -> u16 {
        let h = self.add_handle(aabb_min, aabb_max, client, group, mask, disp);
        self.set_aabb(h, aabb_min, aabb_max, disp);
        h
    }

    pub fn set_aabb(&mut self, h: u16, aabb_min: Vec3, aabb_max: Vec3, disp: &mut Dispatcher) {
        self.handles[h as usize].aabb_min = aabb_min;
        self.handles[h as usize].aabb_max = aabb_max;
        self.update_handle(h, aabb_min, aabb_max, disp);
    }

    pub fn update_handle(&mut self, h: u16, aabb_min: Vec3, aabb_max: Vec3, disp: &mut Dispatcher) {
        let qmin = self.quantize(aabb_min, 0);
        let qmax = self.quantize(aabb_max, 1);
        for axis in 0..3 {
            let emin = self.handles[h as usize].min_edges[axis];
            let emax = self.handles[h as usize].max_edges[axis];
            let dmin = qmin[axis] as i32 - self.edges[axis][emin as usize].pos as i32;
            let dmax = qmax[axis] as i32 - self.edges[axis][emax as usize].pos as i32;
            self.edges[axis][emin as usize].pos = qmin[axis];
            self.edges[axis][emax as usize].pos = qmax[axis];
            if dmin < 0 {
                self.sort_min_down(axis, emin, true);
            }
            if dmax > 0 {
                self.sort_max_up(axis, emax, true);
            }
            if dmin > 0 {
                self.sort_min_up(axis, emin, disp, true);
            }
            if dmax < 0 {
                self.sort_max_down(axis, emax, disp, true);
            }
        }
    }

    pub fn destroy_proxy(&mut self, h: u16, disp: &mut Dispatcher) {
        self.pairs.remove_overlapping_pairs_containing_proxy(h, disp);
        let limit = self.num_handles as usize * 2;
        for axis in 0..3 {
            self.handles[0].max_edges[axis] -= 2;
        }
        for axis in 0..3 {
            let emax = self.handles[h as usize].max_edges[axis];
            self.edges[axis][emax as usize].pos = self.handle_sentinel;
            self.sort_max_up(axis, emax, false);
            let emin = self.handles[h as usize].min_edges[axis];
            self.edges[axis][emin as usize].pos = self.handle_sentinel;
            self.sort_min_up(axis, emin, disp, false);
            self.edges[axis][limit - 1] = Edge { pos: self.handle_sentinel, handle: 0 };
        }
        self.handles[h as usize].min_edges[0] = self.first_free;
        self.first_free = h;
        self.num_handles -= 1;
    }

    pub fn aabb_test(&self, aabb_min: Vec3, aabb_max: Vec3, mut callback: impl FnMut(&Handle)) {
        let n = self.num_handles as usize * 2 + 1;
        for i in 1..n {
            let edge = self.edges[0][i];
            if edge.is_max() {
                let h = &self.handles[edge.handle as usize];
                let mut overlap = true;
                if h.aabb_max.x < aabb_min.x || aabb_max.x < h.aabb_min.x {
                    overlap = false;
                }
                if h.aabb_max.z < aabb_min.z || aabb_max.z < h.aabb_min.z {
                    overlap = false;
                }
                if aabb_min.y <= h.aabb_max.y && h.aabb_min.y <= aabb_max.y && overlap {
                    callback(h);
                }
            }
        }
    }
}
