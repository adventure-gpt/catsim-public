use crate::math::Vec3;
use crate::math::rot::Rotator;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameTimer {
    pub rate: f32,
    pub elapsed: f32,
    pub active: bool,
}

impl FrameTimer {
    pub fn new(rate: f32) -> Self {
        Self { rate, elapsed: 0.0, active: rate > 0.0 }
    }

    pub fn advance(&mut self, dt: f32, dilation: f32) -> bool {
        if !self.active {
            return false;
        }
        self.elapsed += dt * dilation;
        if self.elapsed > self.rate {
            self.active = false;
            true
        } else {
            false
        }
    }

    pub fn remaining(&self) -> f32 {
        if self.active { (self.rate - self.elapsed).max(0.0) } else { 0.0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CountdownClock {
    pub countdown_time: i32,
    pub default_countdown_time: i32,
    pub remaining: i32,
    pub replicated_remaining: i32,
    pub start_timer: FrameTimer,
    pub update_timer: FrameTimer,
    pub next_spawn_index: i32,
    pub warmup: bool,
}

impl CountdownClock {
    pub fn new(countdown_time: i32, default_countdown_time: i32) -> Self {
        let mut out = Self { countdown_time, default_countdown_time, remaining: 0,
            replicated_remaining: 0, start_timer: FrameTimer::default(),
            update_timer: FrameTimer::default(), next_spawn_index: 0, warmup: false };
        out.begin();
        out
    }

    pub fn with_warmup(countdown_time: i32, default_countdown_time: i32, warmup_time: i32) -> Self {
        let mut out = Self::new(countdown_time, default_countdown_time);
        if warmup_time > 0 {
            out.start_timer = FrameTimer::default();
            out.warmup = true;
            out.set_remaining(warmup_time, false);
            out.replicated_remaining = out.remaining;
        }
        out
    }

    pub fn begin(&mut self) {
        self.warmup = false;
        self.start_timer = FrameTimer::new(if self.countdown_time > 0 { 1.0 } else { 0.01 });
    }

    pub fn set_remaining(&mut self, value: i32, from_replication: bool) {
        let value = value.max(0);
        if value == self.remaining { return; }
        self.remaining = value;
        if value == 0 {
            self.update_timer = FrameTimer::default();
        } else if from_replication || !self.update_timer.active {
            self.update_timer = FrameTimer::new(1.0);
        }
        if value % 5 == 0 { self.replicated_remaining = value; }
    }

    pub fn advance(&mut self, dt: f32) -> bool {
        let start_fires = self.start_timer.advance(dt, 1.0);
        if self.update_timer.active { self.update_timer.elapsed += dt; }
        if start_fires {
            if self.countdown_time <= 0 { return true; }
            self.set_remaining(self.countdown_time, false);
            self.replicated_remaining = self.remaining;
        }
        if self.update_timer.active && self.update_timer.elapsed > self.update_timer.rate {
            let calls = (self.update_timer.elapsed / self.update_timer.rate) as i32;
            self.update_timer.elapsed -= calls as f32 * self.update_timer.rate;
            for _ in 0..calls {
                if !self.update_timer.active { break; }
                self.set_remaining(self.remaining - 1, false);
                if self.remaining == 0 {
                    if self.warmup {
                        self.begin();
                        return false;
                    }
                    return true;
                }
            }
        }
        false
    }

    pub fn end(&mut self) {

        self.update_timer = FrameTimer::default();
        self.remaining = 0;
        self.replicated_remaining = 0;
        self.start_timer = FrameTimer::default();
        self.next_spawn_index = self.next_spawn_index.wrapping_add(1);
        if self.countdown_time > 0 { self.countdown_time = self.default_countdown_time; }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrtRand {
    pub state: u32,
}

impl CrtRand {

    pub fn next_15(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(214013).wrapping_add(2531011);
        (self.state >> 16) & 0x7fff
    }

    pub fn index(&mut self, count: i32) -> i32 {
        if count <= 0 {
            return 0;
        }
        let fraction = self.next_15() as f32 * (1.0 / 32768.0);
        (fraction * count as f32) as i32
    }

    pub fn shuffle<T>(&mut self, mut remaining: Vec<T>) -> Vec<T> {
        let mut result = Vec::with_capacity(remaining.len());
        while !remaining.is_empty() {
            let i = self.index(remaining.len() as i32) as usize;
            result.push(remaining.remove(i));
        }
        result
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpawnSpot {
    pub location: Vec3,
    pub rotation: Rotator,
}

#[derive(Clone, Debug)]
pub struct SpawnSelector {
    pub next_spawn_index: i32,
    pub primary: [Vec<SpawnSpot>; 2],
    pub secondary: [Vec<SpawnSpot>; 2],
    pub rng: CrtRand,
}

impl SpawnSelector {

    pub fn end_countdown(&mut self) {
        self.next_spawn_index = self.next_spawn_index.wrapping_add(1);
    }

    pub fn choose(
        &mut self,
        team: usize,
        active: bool,
        mut encroached: impl FnMut(&SpawnSpot) -> bool,
        mut valid_offset: impl FnMut(&SpawnSpot) -> Option<SpawnSpot>,
    ) -> Option<SpawnSpot> {
        assert!(team < 2 && self.next_spawn_index >= 0);
        let spots = if active && !self.secondary[team].is_empty() { &self.secondary[team] } else { &self.primary[team] };
        if spots.is_empty() {
            return None;
        }
        let start = if active {
            self.rng.index(spots.len() as i32) as usize
        } else {
            self.next_spawn_index %= spots.len() as i32;
            self.next_spawn_index as usize
        };
        for offset in 0..spots.len() {
            let spot = spots[(start + offset) % spots.len()];
            if !encroached(&spot) {
                return Some(spot);
            }
        }
        for offset in 0..spots.len() {
            if let Some(spot) = valid_offset(&spots[(start + offset) % spots.len()]) {
                return Some(spot);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::v3;

    #[test]
    fn actor_timer_strict_boundary_and_reset() {
        let mut t = FrameTimer::new(1.0);
        assert!(!t.advance(0.5, 2.0));
        assert_eq!(t.remaining(), 0.0);
        assert!(t.active);
        assert!(t.advance(0.125, 1.0));
        assert!(!t.advance(5.0, 1.0));
        t = FrameTimer::new(0.5);
        assert!(!t.advance(0.125, 1.0));
        assert_eq!(t.remaining(), 0.375);
    }

    #[test]
    fn countdown_callback_frames() {
        let mut c = CountdownClock::new(2, 3);
        assert!(!c.advance(1.0));
        assert!(!c.advance(0.25));
        assert_eq!((c.remaining, c.update_timer.elapsed), (2, 0.0));
        assert!(!c.advance(1.0));
        assert!(!c.advance(0.25));
        assert_eq!((c.remaining, c.update_timer.elapsed), (1, 0.25));
        assert!(!c.advance(0.75));
        assert!(c.advance(0.25));
        assert!(!c.update_timer.active);
        c.end();
        assert_eq!((c.countdown_time, c.next_spawn_index), (3, 1));
        c.begin();
        assert!(!c.advance(1.25));
        assert!(!c.advance(2.25));
        assert_eq!((c.remaining, c.update_timer.elapsed), (1, 0.25));
        assert!(c.advance(3.0));
        assert_eq!(c.remaining, 0);
    }

    #[test]
    fn countdown_preserves_active_loop() {
        let mut c = CountdownClock::new(3, 3);
        c.advance(1.25);
        c.advance(0.5);
        c.set_remaining(6, false);
        assert_eq!(c.update_timer.elapsed, 0.5);
        c.set_remaining(6, true);
        assert_eq!(c.update_timer.elapsed, 0.5);
        c.set_remaining(5, true);
        assert_eq!(c.update_timer.elapsed, 0.0);
        assert_eq!(c.replicated_remaining, 5);
        c.set_remaining(-3, false);
        assert_eq!(c.remaining, 0);
        assert!(!c.update_timer.active);
        let mut instant = CountdownClock::new(0, 3);
        assert!(!instant.advance(0.01));
        assert!(instant.advance(0.001));
        instant.end();
        assert_eq!(instant.countdown_time, 0);
    }

    #[test]
    fn warmup_enters_a_separate_countdown_without_advancing_selector() {
        let mut c = CountdownClock::with_warmup(1, 3, 2);
        assert!(!c.advance(1.25));
        assert_eq!(c.remaining, 1);
        assert!(!c.advance(1.0));
        assert!(!c.warmup);
        assert_eq!((c.start_timer.elapsed, c.remaining, c.next_spawn_index), (0.0, 0, 0));
        assert!(!c.advance(1.0));
        assert!(!c.advance(0.25));
        assert_eq!((c.remaining, c.update_timer.elapsed), (1, 0.0));
        assert!(!c.advance(1.0));
        assert!(c.advance(0.25));
        c.end();
        assert_eq!(c.next_spawn_index, 1);
    }

    #[test]
    fn crt_rand_scaling_and_consumption() {
        let mut r = CrtRand { state: 0 };
        assert_eq!(r.index(0), 0);
        assert_eq!(r.index(-1), 0);
        assert_eq!(r.state, 0);
        assert_eq!(r.next_15(), 38);
        assert_eq!(r.state, 2531011);
        assert_eq!(r.next_15(), 7719);
        assert_eq!(r.state, 505908858);
        r.state = 0;
        assert_eq!(r.index(1), 0);
        assert_eq!(r.state, 2531011);
        assert_eq!(r.index(100), 23);
    }

    #[cfg(windows)]
    #[test]
    fn crt_matches_local_runtime() {
        #[link(name = "ucrt")]
        unsafe extern "C" {
            fn srand(seed: u32);
            fn rand() -> i32;
        }

        for seed in [0, 1, 12345, 0x12345678, 0x80000000, 0xffffffff] {
            let mut r = CrtRand { state: seed };
            unsafe { srand(seed) };
            for _ in 0..1024 {
                assert_eq!(r.next_15() as i32, unsafe { rand() });
            }
        }
    }

    #[test]
    fn shuffle_consumes_last_draw() {
        let mut r = CrtRand { state: 0 };
        assert_eq!(r.shuffle(vec![0, 1, 2]), vec![0, 1, 2]);
        let mut expected = CrtRand { state: 0 };
        for _ in 0..3 { expected.next_15(); }
        assert_eq!(r.state, expected.state);
    }

    #[test]
    fn kickoff_counter_and_demo_two_pass_search() {
        let spots: Vec<_> = (0..3).map(|i| SpawnSpot { location: v3(i as f32, 0.0, 83.0), rotation: Rotator::default() }).collect();
        let mut s = SpawnSelector { next_spawn_index: 1, primary: [spots.clone(), spots.clone()], secondary: [vec![], vec![]], rng: CrtRand { state: 0 } };
        assert_eq!(s.choose(0, false, |p| p.location.x == 1.0, |_| None), Some(spots[2]));
        assert_eq!(s.next_spawn_index, 1);
        assert_eq!(s.rng.state, 0);
        s.end_countdown();
        assert_eq!(s.choose(1, false, |_| false, |_| None), Some(spots[2]));
        let mut visited = Vec::new();
        assert_eq!(s.choose(0, true, |_| true, |p| { visited.push(p.location.x); if p.location.x == 2.0 { Some(*p) } else { None } }), Some(spots[2]));
        assert_eq!(visited, vec![0.0, 1.0, 2.0]);
        assert_eq!(s.rng.state, 2531011);
    }
}
