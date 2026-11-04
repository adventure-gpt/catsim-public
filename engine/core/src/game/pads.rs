use crate::math::rot::{Rotator, rotator_to_matrix};
use crate::math::{Vec3, v3};
use super::round::FrameTimer;

pub const NUM_PADS: usize = 34;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadDef {
    pub loc: Vec3,
    pub big: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum PadLayout {
    #[default]
    Stadium,
    UtopiaNight,
}

impl PadLayout {
    pub fn from_map(name: &str) -> Option<Self> {
        match name {
            "Stadium_P" => Some(Self::Stadium),
            "UF_Night_P" => Some(Self::UtopiaNight),
            _ => None,
        }
    }

    pub fn definition(self, i: usize) -> PadDef {
        let mut d = PADS[i];
        if self == Self::UtopiaNight {

            let z = crate::data::table("pad_z_UF_Night_P")[0].fnn::<2>(0);
            d.loc.z = if d.big { z[1] } else { z[0] };
        }
        d
    }
}



pub static PADS: std::sync::LazyLock<[PadDef; NUM_PADS]> = std::sync::LazyLock::new(|| {
    let rows = crate::data::table("pads");
    assert_eq!(rows.len(), NUM_PADS, "the pads table must have one row per pad");
    std::array::from_fn(|i| PadDef { loc: Vec3::from_bits(rows[i].wn(0)), big: rows[i].w(3) != 0 })
});

pub const SMALL_RESPAWN: f32 = 4.0;
pub const BIG_RESPAWN: f32 = 10.0;
pub const SMALL_AMOUNT: f32 = f32::from_bits(0x3df5_c28f);
pub const BIG_AMOUNT: f32 = 9999.0;

pub const PAD_RADIUS_SMALL: f32 = 128.0;
pub const PAD_RADIUS_BIG: f32 = 160.0;
pub const PAD_HEIGHT: f32 = 64.0;

pub const CAR_EXTENT: Vec3 = v3(48.0, 48.0, 32.0);

#[inline]
fn dot3(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
pub static OCTANE_PHYS_BOX: std::sync::LazyLock<[f32; 6]> = std::sync::LazyLock::new(|| crate::data::table("default_phys_box")[0].fnn(0));

static PHYS_BOXES: std::sync::LazyLock<Vec<(&'static str, [u32; 5])>> = std::sync::LazyLock::new(|| crate::data::table("phys_boxes").iter().map(|r| (r.s(0), r.wn(0))).collect());

pub fn phys_box(preset: &str) -> [f32; 6] {
    match PHYS_BOXES.iter().find(|(n, _)| *n == preset) {
        Some((_, w)) => {
            let f = |b: u32| f32::from_bits(b);
            [f(w[0]) * 0.5, f(w[1]) * 0.5, f(w[2]) * 0.5, f(w[3]), 0.0, f(w[4])]
        }
        None => *OCTANE_PHYS_BOX,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pad {
    pub picked_up: bool,
    pub respawn_left: f32,
    pub timer: FrameTimer,

    pub skip: u8,

    pub respawned: bool,

    pub tick_enabled: bool,
    pub ticked_this_frame: bool,
}

impl Pad {
    pub fn cooldown(seconds: f32) -> Self {
        Self { picked_up: seconds > 0.0, respawn_left: seconds.max(0.0), timer: FrameTimer::new(seconds), skip: 0, respawned: false, tick_enabled: seconds > 0.0, ticked_this_frame: false }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cylinder_line_check(origin: Vec3, radius: f32, height: f32, end: Vec3, start: Vec3, extent: Vec3, flags: u32) -> Option<f32> {
    let (ox, oy, oz) = (origin.x, origin.y, origin.z);
    let (rx, ry, h) = (radius + extent.x, radius + extent.y, height + extent.z);

    let hi = ox + rx;
    let lo = ox - rx;
    if !((start.x <= hi || end.x <= hi) && (lo <= start.x || lo <= end.x)) {
        return None;
    }
    let hi = oy + ry;
    let lo = oy - ry;
    if !((start.y <= hi || end.y <= hi) && (lo <= start.y || lo <= end.y)) {
        return None;
    }
    let (zhi, zlo) = (oz + h, oz - h);
    if !((start.z <= zhi || end.z <= zhi) && (zlo <= start.z || zlo <= end.z)) {
        return None;
    }
    let (mut near, mut far) = (0.0f32, 1.0f32);
    if start.z <= zhi || zhi <= end.z {
        if start.z < zhi && zhi < end.z {
            let t = (zhi - start.z) / (end.z - start.z);
            far = if t <= 1.0 { t } else { 1.0 };
        }
    } else {
        let t = (zhi - start.z) / (end.z - start.z);
        if 0.0 < t {
            near = t;
        }
    }
    if zlo <= start.z || end.z <= zlo {
        if zlo < start.z && end.z < zlo {
            let t = (zlo - start.z) / (end.z - start.z);
            if t <= far {
                far = t;
            }
        }
    } else {
        let t = (zlo - start.z) / (end.z - start.z);
        if near < t {
            near = t;
        }
    }
    if !(near < far) {
        return None;
    }
    let dy0 = start.y - oy;
    let (dx, dy) = (end.x - start.x, end.y - start.y);
    let dx0 = start.x - ox;
    let a = dy * dy + dx * dx;
    let half = dy0 * dy + dx0 * dx;
    let b = half + half;
    let c = (dy0 * dy0 + dx0 * dx0) - rx * rx;
    let disc = b * b - a * 4.0 * c;
    if 1.0 <= c || start.z <= zlo || zhi <= start.z {
        if !(0.0 <= disc) {
            return None;
        }
        if f32::from_bits(0x322b_cc76) <= a {
            let root = f64::from(disc).sqrt() as f32;
            let k = 0.5 / a;
            let t1 = -(root + b) * k;
            let t2 = (root - b) * k;
            if t2 <= far {
                far = t2;
            }
            if near < t1 {
                near = t1;
            }
            if !(near < far) {
                return None;
            }
        } else if !(c <= 0.0) {
            return None;
        }
        if flags & 0x80000 == 0 {
            near -= 0.001;
        }
        let mut t = 0.0;
        if 0.0 <= near {
            t = if 1.0 <= near { 1.0 } else { near };
        }
        Some(t)
    } else if half < -0.1 {

        Some(0.0)
    } else {
        None
    }
}

pub fn cylinder_point_check(origin: Vec3, radius: f32, height: f32, point: Vec3, extent: Vec3) -> bool {
    let reach_z = extent.z + height;
    let dz = origin.z - point.z;
    if reach_z * reach_z <= dz * dz {
        return false;
    }
    let (low_x, high_x) = (point.x - extent.x, point.x + extent.x);
    let (low_y, high_y) = (point.y - extent.y, point.y + extent.y);
    let dx = if origin.x < low_x { origin.x - low_x } else if origin.x > high_x { origin.x - high_x } else { 0.0 };
    let dy = if origin.y < low_y { origin.y - low_y } else if origin.y > high_y { origin.y - high_y } else { 0.0 };
    let distance = f64::from(dy * dy + dx * dx).sqrt() as f32;
    radius >= distance
}

#[derive(Clone, Debug)]
pub struct Pads {
    pub pads: [Pad; NUM_PADS],
    pub layout: PadLayout,
}

impl Default for Pads {
    fn default() -> Pads {
        Pads { pads: [Pad::default(); NUM_PADS], layout: PadLayout::default() }
    }
}

impl Pads {

    pub fn reset(&mut self) {
        self.pads = [Pad::default(); NUM_PADS];
    }

    pub fn frame_tick(&mut self, dt: f32) {
        for p in self.pads.iter_mut() {
            p.ticked_this_frame = p.tick_enabled;
            if p.picked_up {
                if p.skip > 0 {
                    p.skip -= 1;
                    continue;
                }
                if p.timer.advance(dt, 1.0) {
                    p.picked_up = false;
                    p.respawned = true;
                }
                p.respawn_left = p.timer.remaining();
            }
        }
    }

    #[inline]

    pub fn sweep_candidates(from: Vec3, to: Vec3) -> u64 {
        Self::sweep_candidates_with_layout(PadLayout::Stadium, from, to)
    }

    pub fn sweep_candidates_with_layout(layout: PadLayout, from: Vec3, to: Vec3) -> u64 {

        const N: usize = NUM_PADS.div_ceil(4) * 4;
        static SLABS: [std::sync::OnceLock<[[f32; N]; 6]>; 2] = [std::sync::OnceLock::new(), std::sync::OnceLock::new()];
        let slabs = SLABS[layout as usize].get_or_init(|| {
            let mut t = [[f32::NAN; N]; 6];
            for i in 0..NUM_PADS {
                let d = layout.definition(i);
                let r = if d.big { PAD_RADIUS_BIG } else { PAD_RADIUS_SMALL };
                let (o, e) = (d.loc, CAR_EXTENT);
                let (rx, ry, h) = (r + e.x, r + e.y, PAD_HEIGHT + e.z);
                let v = [o.x + rx, o.x - rx, o.y + ry, o.y - ry, o.z + h, o.z - h];
                for k in 0..6 {
                    t[k][i] = v[k];
                }
            }
            t
        });
        let (start, end) = (from, to);
        #[cfg(target_arch = "x86_64")]
        unsafe {

            use core::arch::x86_64::*;
            let mut m = 0u64;
            let (sx, sy, sz) = (_mm_set1_ps(start.x), _mm_set1_ps(start.y), _mm_set1_ps(start.z));
            let (ex, ey, ez) = (_mm_set1_ps(end.x), _mm_set1_ps(end.y), _mm_set1_ps(end.z));
            let axis = |s: __m128, e: __m128, hi: __m128, lo: __m128| {
                _mm_and_ps(_mm_or_ps(_mm_cmple_ps(s, hi), _mm_cmple_ps(e, hi)), _mm_or_ps(_mm_cmple_ps(lo, s), _mm_cmple_ps(lo, e)))
            };
            for g in (0..N).step_by(4) {
                let ld = |k: usize| _mm_loadu_ps(slabs[k].as_ptr().add(g));
                let ok = _mm_and_ps(_mm_and_ps(axis(sx, ex, ld(0), ld(1)), axis(sy, ey, ld(2), ld(3))), axis(sz, ez, ld(4), ld(5)));
                m |= (_mm_movemask_ps(ok) as u64) << g;
            }
            m & ((1u64 << NUM_PADS) - 1)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut m = 0u64;
            for i in 0..NUM_PADS {
                let s = |k: usize| slabs[k][i];
                let x = (start.x <= s(0) || end.x <= s(0)) && (s(1) <= start.x || s(1) <= end.x);
                let y = (start.y <= s(2) || end.y <= s(2)) && (s(3) <= start.y || s(3) <= end.y);
                let z = (start.z <= s(4) || end.z <= s(4)) && (s(5) <= start.z || s(5) <= end.z);
                m |= ((x & y & z) as u64) << i;
            }
            m
        }
    }

    pub fn sweep(i: usize, from: Vec3, to: Vec3) -> bool {
        Self::sweep_with_layout(PadLayout::Stadium, i, from, to)
    }

    pub fn sweep_with_layout(layout: PadLayout, i: usize, from: Vec3, to: Vec3) -> bool {
        let d = layout.definition(i);
        let r = if d.big { PAD_RADIUS_BIG } else { PAD_RADIUS_SMALL };
        cylinder_line_check(d.loc, r, PAD_HEIGHT, to, from, CAR_EXTENT, 0).is_some()
    }

    #[inline]
    pub fn overlaps(i: usize, car: Vec3) -> bool {
        Self::overlaps_with_layout(PadLayout::Stadium, i, car)
    }

    pub fn overlaps_with_layout(layout: PadLayout, i: usize, car: Vec3) -> bool {
        let d = layout.definition(i);
        let r = if d.big { PAD_RADIUS_BIG } else { PAD_RADIUS_SMALL };
        let e = v3(r, r, PAD_HEIGHT);
        let low = d.loc - e;
        let high = d.loc + e;
        let extent = (high - low) * 0.5;
        cylinder_point_check(car, CAR_EXTENT.x, CAR_EXTENT.z, extent + low, extent)
    }

    pub fn mesh_overlaps(i: usize, loc: Vec3, rot: Rotator) -> bool {
        Self::mesh_overlaps_box(i, loc, rot, *OCTANE_PHYS_BOX)
    }

    pub fn mesh_overlaps_box(i: usize, loc: Vec3, rot: Rotator, bx: [f32; 6]) -> bool {
        Self::mesh_overlaps_box_with_layout(PadLayout::Stadium, i, loc, rot, bx)
    }

    pub fn mesh_overlaps_box_with_layout(layout: PadLayout, i: usize, loc: Vec3, rot: Rotator, bx: [f32; 6]) -> bool {
        let d = layout.definition(i);
        let r = if d.big { PAD_RADIUS_BIG } else { PAD_RADIUS_SMALL };
        let ha = [r, r, PAD_HEIGHT];
        let m = rotator_to_matrix(rot).r;
        let c = loc + m[0] * bx[3] + m[1] * bx[4] + m[2] * bx[5];
        let hb = [bx[0], bx[1], bx[2]];
        let t = c - d.loc;
        let world = [v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)];
        let sep = |ax: Vec3| -> bool {
            let n = dot3(ax, ax);
            if n < 1e-8 {
                return false;
            }
            let ra = ax.x.abs() * ha[0] + ax.y.abs() * ha[1] + ax.z.abs() * ha[2];
            let rb = (0..3).map(|j| dot3(ax, m[j]).abs() * hb[j]).sum::<f32>();
            dot3(t, ax).abs() > ra + rb
        };
        for a in world.iter().chain(m.iter()) {
            if sep(*a) {
                return false;
            }
        }
        for a in world.iter() {
            for b in m.iter() {
                if sep(a.cross(*b)) {
                    return false;
                }
            }
        }
        true
    }

    #[inline]
    pub fn can_pick_up(&self, i: usize) -> bool {
        !self.pads[i].picked_up
    }

    pub fn pick_up(&mut self, i: usize) -> f32 {
        let big = PADS[i].big;
        let ticked = self.pads[i].ticked_this_frame;
        self.pads[i] = Pad::cooldown(if big { BIG_RESPAWN } else { SMALL_RESPAWN });
        self.pads[i].ticked_this_frame = ticked;
        if big { BIG_AMOUNT } else { SMALL_AMOUNT }
    }
}
