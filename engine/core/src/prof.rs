pub const NAMES: [&str; 56] = [
    "gravity",
    "predict",
    "vehicles",
    "ccd",
    "collision",
    "islands",
    "solver",
    "integrate",
    "activation",
    "arena_tick",
    "ray_plane",
    "ray_mesh",
    "ray_mesh_walk",
    "ray_compound",
    "ray_convex",
    "ray_triangle",
    "pair_cvx_cvx",
    "pair_sph_sph",
    "pair_sph_tri",
    "pair_box_box",
    "pair_cvx_plane",
    "pair_cvx_mesh",
    "pair_cmp_plane",
    "pair_cmp_mesh",
    "pair_cmp_other",
    "pair_cmpcmp",
    "ccd_ray",
    "ccd_hit",
    "pair_empty_out",
    "tri_far",
    "ccd_clear",
    "wheel_cast",
    "cc_box_call",
    "cc_box_walk",
    "cc_box_tri",
    "cc_box_tri_pass",
    "cc_box_grid_empty",
    "cc_tri_pts",
    "ray_walk_full",
    "ccd_car_ray",
    "k_car_tick",
    "k_floor4_iso",
    "k_air_iso",
    "k_floor4_contact",
    "k_floor4_dynpair",
    "k_floor4_near",
    "k_air_contact",
    "k_air_dynpair",
    "k_air_near",
    "k_wheels_nonfloor",
    "k_wheels_1to3",
    "k_ball_tick",
    "k_ball_dynpair",
    "k_ball_free_iso",
    "k_ball_floor_only",
    "k_ball_other",
];
#[cfg(feature = "profile")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "profile")]
static NS: [AtomicU64; 56] = [const { AtomicU64::new(0) }; 56];
#[cfg(feature = "profile")]
static CALLS: [AtomicU64; 56] = [const { AtomicU64::new(0) }; 56];
pub struct Span {
    #[cfg(feature = "profile")]
    id: usize,
    #[cfg(feature = "profile")]
    start: std::time::Instant,
}
#[inline(always)]
pub fn span(_id: usize) -> Span {
    Span {
        #[cfg(feature = "profile")]
        id: _id,
        #[cfg(feature = "profile")]
        start: std::time::Instant::now(),
    }
}
impl Drop for Span {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(feature = "profile")]
        {
            NS[self.id].fetch_add(self.start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            CALLS[self.id].fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[inline(always)]
pub fn count(_id: usize) {
    #[cfg(feature = "profile")]
    CALLS[_id].fetch_add(1, Ordering::Relaxed);
}
pub fn stats() -> Vec<(String, u64, u64)> {
    #[cfg(feature = "profile")]
    {
        NAMES
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (
                    n.to_string(),
                    CALLS[i].load(Ordering::Relaxed),
                    NS[i].load(Ordering::Relaxed),
                )
            })
            .collect()
    }
    #[cfg(not(feature = "profile"))]
    {
        Vec::new()
    }
}
