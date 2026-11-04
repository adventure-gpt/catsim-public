use super::round::CrtRand;

#[derive(Clone, Debug)]
pub struct LockChain {
    pub release: f32,
    pub time: f32,
    pub random_time: f32,

    pub running: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshTimer {

    pub owner: i32,
    pub kind: u8,
    pub last: f32,

    pub spawn_gen: u32,
}

pub const MESH_BODY: u8 = 0;
pub const MESH_WHEEL: u8 = 1;
pub const MESH_EXTRA: u8 = 2;
pub const MESH_GOAL: u8 = 3;
pub const MESH_MAP: u8 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshConfig {
    pub wheels: u32,
    pub extras: u32,
    pub map: u32,

    pub goal_mesh: bool,

    pub goal: (u32, u32),
    pub demolish: (u32, u32),

    pub reset: (u32, u32),

    pub respawn: (u32, u32),
}

#[derive(Clone, Debug)]
pub struct FrameRng {
    pub world_time: f32,
    pub pad_before: [f32; 34],
    pub steps: i32,
    pub locks: Vec<LockChain>,
    pub pending_pickup: u32,
    pub pending_motion: u32,
    pub motion_fx: bool,

    pub countdown_entry: bool,
    pub frame_event_draws: u32,

    pub goal_fx: u32,
    pub frames: u64,

    pub draws: [u64; 5],
    pub last: [u32; 5],

    pub mesh: Option<MeshConfig>,
    pub timers: Vec<MeshTimer>,

    pub pending_parts: Vec<(u32, usize, u8)>,

    pub pending_events: Vec<(u32, u32, bool)>,

    pub goal_mesh_wait: u32,
    pub goal_mesh_ref: Option<f32>,

    pub car_gen: Vec<u32>,
    pub next_gen: u32,

    pub motion_by: Vec<u32>,
}

impl FrameRng {
    pub fn new(world_time: f32, locks: Vec<LockChain>, motion_fx: bool) -> Self {
        Self { world_time, pad_before: [0.0; 34], steps: 0, locks, pending_pickup: 0, pending_motion: 0,
            motion_fx, countdown_entry: false, frame_event_draws: 0, goal_fx: 0,
            frames: 0, draws: [0; 5], last: [0; 5], mesh: None, timers: Vec::new(), pending_parts: Vec::new(),
            pending_events: Vec::new(), goal_mesh_wait: 0, goal_mesh_ref: None,
            car_gen: Vec::new(), next_gen: 1, motion_by: Vec::new() }
    }

    pub fn start_mesh(&mut self, cfg: MeshConfig, cars: &[usize]) {
        self.mesh = Some(cfg);
        self.timers.clear();
        for _ in 0..cfg.map { self.timers.push(MeshTimer { owner: -1, kind: MESH_MAP, last: 0.0, spawn_gen: 0 }); }
        let g = self.new_gen(cars);
        for &i in cars { self.push_car_parts(i, true, true, g); }
    }

    fn new_gen(&mut self, cars: &[usize]) -> u32 {
        let g = self.next_gen;
        self.next_gen += 1;
        for &i in cars {
            if self.car_gen.len() <= i { self.car_gen.resize(i + 1, 0); }
            self.car_gen[i] = g;
        }
        g
    }

    fn push_car_parts(&mut self, car: usize, body: bool, rest: bool, spawn_gen: u32) {
        let Some(cfg) = self.mesh else { return };

        let mut at = match self.timers.iter().rposition(|t| t.owner == car as i32) {
            Some(j) => j + 1,
            None => self.timers.iter().position(|t| t.kind == MESH_GOAL).unwrap_or(self.timers.len()),
        };
        let mut kinds = Vec::new();
        if body { kinds.push((MESH_BODY, 1)); }
        if rest { kinds.push((MESH_WHEEL, cfg.wheels)); kinds.push((MESH_EXTRA, cfg.extras)); }
        for (kind, n) in kinds {
            for _ in 0..n {
                self.timers.insert(at, MeshTimer { owner: car as i32, kind, last: 0.0, spawn_gen });
                at += 1;
            }
        }
    }

    pub fn car_removed(&mut self, car: usize) {
        if self.mesh.is_some() {
            self.timers.retain(|t| t.owner != car as i32);
        }
    }

    pub fn cars_restarted(&mut self, cars: &[usize]) {
        if self.mesh.is_some() {
            self.timers.retain(|t| t.kind != MESH_GOAL);
            self.goal_mesh_wait = 0;
            self.goal_mesh_ref = None;
            self.new_gen(cars);
            for &i in cars {
                self.pending_parts.push((2, i, 0));
                self.pending_parts.push((3, i, 1));
            }
        }
    }

    pub fn car_respawned(&mut self, car: usize) {
        if let Some(cfg) = self.mesh {
            self.new_gen(&[car]);
            if cfg.respawn.0 > 0 { self.pending_parts.push((cfg.respawn.0, car, 2)); }
            if cfg.respawn.1 > 0 { self.pending_parts.push((cfg.respawn.1, car, 3)); }
        }
    }

    pub fn schedule(&mut self, delay: u32, draws: u32, early: bool) {
        if draws > 0 { self.pending_events.push((delay, draws, early)); }
    }

    pub fn enter_countdown(&mut self, state: &mut u32) {
        if self.countdown_entry {
            let mut rng = CrtRand { state: *state };
            rng.next_15(); rng.next_15();
            *state = rng.state;
            self.frame_event_draws += 2;
        }
    }

    pub fn goal(&mut self) {
        match self.mesh {
            Some(cfg) => {
                self.schedule(cfg.goal.0, cfg.goal.1, false);
                if cfg.goal_mesh { self.goal_mesh_wait = 3; }
            }
            None => self.goal_fx = 3,
        }
    }

    pub fn demolished(&mut self) {
        if let Some(cfg) = self.mesh { self.schedule(cfg.demolish.0, cfg.demolish.1, true); }
    }

    pub fn kickoff_reset(&mut self, state: &mut u32, cars: u32) {
        let mut rng = CrtRand { state: *state };
        let n = match self.mesh { Some(cfg) => cfg.reset.0 + cfg.reset.1 * cars, None => 2 + 8 * cars };
        for _ in 0..n {
            rng.next_15();
        }
        *state = rng.state;
        self.frame_event_draws += n;
    }

    pub fn finish(&mut self, state: &mut u32, dt: f32, steps: i32, pickups: u32, motion: u32, pickups_now: u32) {
        self.world_time += dt;
        let mut rng = CrtRand { state: *state };
        let mut locks = 0;

        let zero_now = steps == 0 && self.mesh.is_none();
        let pickup = self.pending_pickup + pickups_now * 5 + if zero_now { pickups * 5 } else { 0 };
        self.pending_pickup = if zero_now { 0 } else { pickups * 5 };

        let motion_now = if self.motion_fx { self.pending_motion + motion } else { 0 };
        self.pending_motion = 0;

        let mut early = 0;
        let pickup_slot = if self.mesh.is_some() { early = pickup; pickup } else { 0 };

        let (mut goal, mut early_events) = (0, 0);
        for e in &mut self.pending_events {
            e.0 -= 1;
            if e.0 == 0 { if e.2 { early_events += e.1; } else { goal += e.1; } }
        }
        self.pending_events.retain(|e| e.0 > 0);
        if let Some(cfg) = self.mesh {
            let mut made = Vec::new();
            for p in &mut self.pending_parts {
                p.0 -= 1;
                if p.0 == 0 { made.push((p.1, p.2)); }
            }
            self.pending_parts.retain(|p| p.0 > 0);
            for (car, part) in made {
                let g = self.car_gen.get(car).copied().unwrap_or(0);
                self.push_car_parts(car, part % 2 == 0, part % 2 == 1, g);
            }
            if cfg.goal_mesh {
                if let Some(t0) = self.goal_mesh_ref {
                    if self.world_time - t0 >= 1.0f32 {
                        self.timers.push(MeshTimer { owner: -2, kind: MESH_GOAL, last: 0.0, spawn_gen: u32::MAX });
                        self.goal_mesh_ref = None;
                    }
                }
            }

            let mut fx_left = pickup_slot + early_events;
            let mut motion: Vec<(u32, u32)> = self.motion_by.iter().enumerate().filter(|(_, n)| **n > 0)
                .map(|(i, &n)| (self.car_gen.get(i).copied().unwrap_or(0), n)).collect();
            let mut motion_left = motion_now.saturating_sub(motion.iter().map(|m| m.1).sum::<u32>());
            for t in self.timers.iter_mut() {
                if fx_left > 0 && t.kind != MESH_MAP {
                    for _ in 0..fx_left { rng.next_15(); }
                    fx_left = 0;
                }
                if t.kind != MESH_MAP {
                    for m in motion.iter_mut() {
                        if m.1 > 0 && t.spawn_gen > m.0 {
                            for _ in 0..m.1 { rng.next_15(); }
                            m.1 = 0;
                        }
                    }
                }
                if 1.0f32 < self.world_time - t.last {
                    locks += 1;
                    t.last = ((rng.next_15() as f32 / 32767.0f32) * 0.5f32 + self.world_time) - 0.25f32;
                }
            }
            motion_left += motion.iter().map(|m| m.1).sum::<u32>();
            for _ in 0..fx_left + motion_left { rng.next_15(); }
            early += motion_now;
        }
        for lock in &mut self.locks {
            if lock.running && self.world_time >= lock.release {
                let duration = if lock.random_time > 0.0 {
                    locks += 1;
                    lock.time + lock.random_time * (rng.next_15() as f32 / 32767.0)
                } else { lock.time };
                lock.release = self.world_time + duration;
            }
        }
        if self.goal_fx > 0 {
            self.goal_fx -= 1;
            if self.goal_fx == 0 {
                goal = 7;
            }
        }

        if self.goal_mesh_wait > 0 {
            self.goal_mesh_wait -= 1;
            if self.goal_mesh_wait == 0 { self.goal_mesh_ref = Some(self.world_time); }
        }
        self.last = [2, pickup + goal + early_events, motion_now, locks, core::mem::take(&mut self.frame_event_draws)];
        for _ in 0..pickup + goal + motion_now + 2 - early { rng.next_15(); }
        *state = rng.state;
        for i in 0..5 { self.draws[i] += self.last[i] as u64; }
        self.frames += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_step_and_deferred_effects() {
        let mut m = FrameRng::new(0.0, vec![], false);
        let mut s = 123;
        let mut r = CrtRand { state: s };
        m.finish(&mut s, 0.004, 1, 1, 0, 0);
        for _ in 0..2 { r.next_15(); }
        assert_eq!(s, r.state);
        m.finish(&mut s, 0.004, 0, 2, 0, 0);
        for _ in 0..17 { r.next_15(); }
        assert_eq!(s, r.state);
        assert_eq!(m.draws, [4, 15, 0, 0, 0]);
    }
    #[test]
    fn locks_use_the_predicted_draw_and_float32_clock() {
        let mut m = FrameRng::new(32.0, vec![LockChain {
            release: 32.0, time: 0.75, random_time: 0.5, running: true,
        }], false);
        let mut s = 1;
        let mut r = CrtRand { state: s };
        let now = 32.0f32 + 0.004f32;
        let expiry = now + (0.75f32 + 0.5f32 * (r.next_15() as f32 / 32767.0));
        r.next_15(); r.next_15();
        m.finish(&mut s, 0.004, 0, 0, 0, 0);
        assert_eq!(s, r.state);
        assert_eq!(m.locks[0].release.to_bits(), expiry.to_bits());
        assert_eq!(m.last, [2, 0, 0, 1, 0]);
    }

    fn mesh_model(map: u32, cars: &[usize]) -> FrameRng {
        let mut m = FrameRng::new(1.5, vec![], true);
        m.start_mesh(MeshConfig { wheels: 1, extras: 0, map, goal_mesh: false, goal: (3, 7), demolish: (4, 6), reset: (2, 7),
            respawn: (2, 3) }, cars);
        m
    }
    fn lod(r: &mut CrtRand, t: f32) -> f32 {
        ((r.next_15() as f32 / 32767.0f32) * 0.5f32 + t) - 0.25f32
    }
    #[test]
    fn mesh_order_map_pickup_fx_cars_then_frame_end() {
        let mut m = mesh_model(1, &[0, 1]);
        let mut s = 7;
        let mut r = CrtRand { state: s };
        let t = 1.5f32 + 0.01f32;
        m.pending_pickup = 5;
        m.finish(&mut s, 0.01, 1, 0, 0, 0);
        let map = lod(&mut r, t);
        for _ in 0..5 { r.next_15(); }
        let cars: Vec<f32> = (0..4).map(|_| lod(&mut r, t)).collect();
        r.next_15(); r.next_15();
        assert_eq!(s, r.state);
        assert_eq!(m.timers[0].last.to_bits(), map.to_bits());
        for (k, v) in cars.iter().enumerate() { assert_eq!(m.timers[1 + k].last.to_bits(), v.to_bits()); }
        assert_eq!(m.last, [2, 5, 0, 5, 0]);
    }
    #[test]
    fn zero_step_pickup_fx_wait_for_the_next_frame_in_the_mesh_model() {
        let mut m = mesh_model(0, &[]);
        let mut s = 11;
        let mut r = CrtRand { state: s };
        m.finish(&mut s, 0.01, 0, 1, 0, 0);
        r.next_15(); r.next_15();
        assert_eq!(s, r.state);
        m.finish(&mut s, 0.01, 1, 0, 0, 0);
        for _ in 0..7 { r.next_15(); }
        assert_eq!(s, r.state);
    }
    #[test]
    fn respawned_car_joins_the_end_and_motion_fx_follow_their_generation() {
        let mut m = mesh_model(0, &[0, 1]);
        let mut s = 3;
        m.finish(&mut s, 0.01, 1, 0, 0, 0);
        m.car_removed(0);
        m.car_respawned(0);
        assert_eq!(m.timers.iter().map(|t| t.owner).collect::<Vec<_>>(), vec![1, 1]);
        m.finish(&mut s, 0.01, 1, 0, 0, 0);
        m.finish(&mut s, 0.01, 1, 0, 0, 0);
        assert_eq!(m.timers.iter().map(|t| (t.owner, t.kind)).collect::<Vec<_>>(), vec![(1, MESH_BODY), (1, MESH_WHEEL), (0, MESH_BODY)]);
        m.finish(&mut s, 0.01, 1, 0, 0, 0);
        assert_eq!(m.timers.iter().map(|t| (t.owner, t.kind)).collect::<Vec<_>>(),
                   vec![(1, MESH_BODY), (1, MESH_WHEEL), (0, MESH_BODY), (0, MESH_WHEEL)]);

        let t = m.world_time + 2.0;
        m.world_time = t - 0.01;
        let mut r = CrtRand { state: s };
        m.motion_by = vec![6, 6];
        m.finish(&mut s, 0.01, 1, 0, 12, 0);
        let a = lod(&mut r, t); let b = lod(&mut r, t);
        for _ in 0..6 { r.next_15(); }
        let c = lod(&mut r, t); let d = lod(&mut r, t);
        for _ in 0..6 { r.next_15(); }
        r.next_15(); r.next_15();
        assert_eq!(s, r.state);
        let got: Vec<u32> = m.timers.iter().map(|t| t.last.to_bits()).collect();
        assert_eq!(got, vec![a.to_bits(), b.to_bits(), c.to_bits(), d.to_bits()]);
    }
    #[test]
    fn demolition_fx_draw_before_the_cars() {
        let mut m = mesh_model(1, &[0]);
        let mut s = 5;
        m.finish(&mut s, 0.01, 1, 0, 0, 0);
        m.demolished();
        for _ in 0..3 { m.finish(&mut s, 0.01, 1, 0, 0, 0); }
        let t = m.world_time + 2.0;
        m.world_time = t - 0.01;
        let mut r = CrtRand { state: s };
        m.finish(&mut s, 0.01, 1, 0, 0, 0);
        let map = lod(&mut r, t);
        for _ in 0..6 { r.next_15(); }
        let body = lod(&mut r, t); let wheel = lod(&mut r, t);
        r.next_15(); r.next_15();
        assert_eq!(s, r.state);
        assert_eq!([m.timers[0].last.to_bits(), m.timers[1].last.to_bits(), m.timers[2].last.to_bits()], [map.to_bits(), body.to_bits(), wheel.to_bits()]);
        assert_eq!(m.last[1], 6);
    }
}
