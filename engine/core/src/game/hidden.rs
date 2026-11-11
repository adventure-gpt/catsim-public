//! Hidden state: what the next ticks read that a packet doesn't show, and the bodies exactly as the physics holds them.
//!
//! For restoring a recorded or saved state (benchmarks, replays, search from a mid-game state). Every field here is
//! read back exactly as it is set; what can't be set (the contact manifolds' cached points, the broadphase's pair
//! order) is listed in docs/API.md. Field meanings follow the game, not RocketSim: docs/API.md maps RocketSim v3's
//! `CarState` fields onto these.

use crate::bt::{ACTIVE, ISLAND_SLEEPING};
use crate::game::arena::{Arena, Phase};
use crate::game::car::VehicleInputs;
use crate::game::components::Component;
use crate::game::rb_state;
use crate::game::round::{CountdownClock, FrameTimer};
use crate::math::{Mat3, Vec3, v3};

/// A game component (jump, dodge, boost, ...): on or off, and the seconds since it last changed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Clock {
    pub active: bool,
    pub time: f32,
}

impl Clock {
    fn of(c: &Component) -> Clock {
        Clock { active: c.active, time: c.time }
    }
    fn put(self, c: &mut Component) {
        c.active = self.active;
        c.time = self.time;
    }
}

/// A rigid body exactly as the physics holds it: Bullet units (1 = 50 uu), the rotation matrix by rows.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyState {
    pub origin: Vec3,
    pub basis: Mat3,
    pub lin_vel: Vec3,
    pub ang_vel: Vec3,
    /// Bullet's sleep: a sleeping body doesn't move until something wakes it.
    pub asleep: bool,
    /// Seconds the body has been below Bullet's sleep thresholds (it falls asleep after 2 s).
    pub deactivation_time: f32,
}

/// What a wheel stands on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WheelGround {
    /// The arena (its surfaces hold the wheels: the sticky force).
    #[default]
    Arena,
    /// Another car.
    Car,
    /// The ball.
    Ball,
}

/// One wheel's contact as the last tick left it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WheelHidden {
    pub contact: bool,
    pub normal: Vec3,
    /// What the wheel touches (read only when `contact`).
    pub ground: WheelGround,
}

/// A car's hidden state. Times are seconds; `*_frame` fields are the arena's physics frame numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarHidden {
    /// The car takes input (off during a kickoff countdown; on through the post-goal time, as in the game).
    pub driving: bool,
    pub on_ground: bool,
    /// The game's own jump flags (cleared on landing).
    pub jumped: bool,
    pub double_jumped: bool,
    pub supersonic: bool,
    /// Seconds below supersonic speed while still counted supersonic.
    pub supersonic_time: f32,
    pub jump: Clock,
    /// The jump button was let go during the jump (the jump's extra force stops).
    pub jump_released: bool,
    pub double_jump: Clock,
    pub dodge: Clock,
    /// The dodge's torque (whole numbers: x side, y forward) and its direction.
    pub dodge_torque: Vec3,
    pub dodge_direction: Vec3,
    /// The automatic flip back onto the wheels (car on its roof).
    pub auto_flip: Clock,
    pub auto_flip_right: bool,
    /// How long the automatic flip lasts (set from the car's roll when it starts).
    pub auto_flip_duration: f32,
    pub boost: Clock,
    /// Boost 0 to 1, exactly as the game holds it (a packet shows it times 100, rounded).
    pub boost_amount: f32,
    pub air_control: Clock,
    /// The smoothed handbrake, 0 to 1.
    pub handbrake: f32,
    /// The input the car holds from its last tick: throttle, steer, pitch, yaw, roll, jump, boost, handbrake
    /// (axes as the game stores them, 8 bits each).
    pub held_input: [f32; 8],
    pub wheels: [WheelHidden; 4],
    pub world_contact: bool,
    pub world_contact_normal: Vec3,
    /// The last car this car bumped and when (s of physics time); the same car can be bumped again 0.25 s later.
    pub bump_victim: Option<usize>,
    pub bump_time: f32,
    /// The frame of the last ball hit that gave the ball its extra impulse (the next can come 2 frames later), and of
    /// the last touch.
    pub ball_impact_frame: i32,
    pub ball_touch_frame: i32,
}

/// The arena's hidden state: the physics frame, the round's phase and timers, the countdown clock.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundHidden {
    pub physics_frame: i32,
    /// 0 countdown, 1 active, 2 after a goal, 3 ended.
    pub phase: i32,
    /// Countdown ticks left, or seconds left after a goal.
    pub phase_left: f32,
    /// The game's own countdown clock, when it runs: (seconds remaining shown, start timer elapsed or -1 when done,
    /// update timer elapsed or -1).
    pub countdown_clock: Option<(i32, f32, f32)>,
    pub ball_has_been_hit: bool,
    pub ball_held_2d: bool,
    pub game_time: f32,
    pub score: [i32; 2],
}

fn read_body(b: &crate::bt::Body) -> BodyState {
    BodyState {
        origin: b.transform.origin,
        basis: b.transform.basis,
        lin_vel: b.lin_vel,
        ang_vel: b.ang_vel,
        asleep: b.activation_state == ISLAND_SLEEPING,
        deactivation_time: b.deactivation_time,
    }
}

fn write_body(b: &mut crate::bt::Body, s: &BodyState) {
    b.transform.origin = s.origin;
    b.transform.basis = s.basis;
    b.interp_transform = b.transform;
    b.lin_vel = s.lin_vel;
    b.ang_vel = s.ang_vel;
    b.interp_lin_vel = s.lin_vel;
    b.interp_ang_vel = s.ang_vel;
    b.total_force = Vec3::ZERO;
    b.total_torque = Vec3::ZERO;
    b.update_inertia_tensor();
    if s.asleep {
        rb_state::set_activation_state(b, ISLAND_SLEEPING);
    } else {
        rb_state::set_activation_state(b, ACTIVE);
    }
    b.deactivation_time = s.deactivation_time;
    b.update_revision = b.update_revision.wrapping_add(2);
}

fn input_row(i: &VehicleInputs) -> [f32; 8] {
    let b = |x: bool| if x { 1.0 } else { 0.0 };
    [i.throttle, i.steer, i.pitch, i.yaw, i.roll, b(i.jump), b(i.holding_boost), b(i.handbrake)]
}

impl Arena {
    /// A car's body exactly as the physics holds it (None while demolished).
    pub fn car_body(&self, i: usize) -> Option<BodyState> {
        let c = self.players.get(i)?.car.as_ref()?;
        Some(read_body(&self.world.cw.bodies[c.body.0 as usize]))
    }

    /// Write a car's body exactly (no game-style rounding of the rotation). The car's kept state (what the game read
    /// last tick) becomes this body's read too.
    pub fn set_car_body(&mut self, i: usize, s: &BodyState) -> bool {
        let Some(c) = self.players.get_mut(i).and_then(|p| p.car.as_mut()) else { return false };
        let b = &mut self.world.cw.bodies[c.body.0 as usize];
        write_body(b, s);
        let st = rb_state::read_body(b, &c.limits);
        c.state = st;
        c.old_state = st;
        let p = &mut self.players[i];
        p.fresh_body = false;
        p.actor_loc = st.location;
        p.actor_rot_src = Some((st.quat, st.quat, 1.0));
        true
    }

    /// The ball's body exactly (None when there is no ball, e.g. just after a goal's explosion).
    pub fn ball_body(&self) -> Option<BodyState> {
        let b = self.ball.as_ref()?;
        Some(read_body(&self.world.cw.bodies[b.body.0 as usize]))
    }

    /// Write the ball's body exactly. Makes a ball first if there is none.
    pub fn set_ball_body(&mut self, s: &BodyState) {
        if self.ball.is_none() {
            self.reset_ball();
        }
        let bl = self.ball.as_mut().unwrap();
        let b = &mut self.world.cw.bodies[bl.body.0 as usize];
        write_body(b, s);
        let st = rb_state::read_body(b, &bl.limits);
        bl.state = st;
        bl.old_state = st;
        bl.first_frame = false;
        bl.round_active = true;
        bl.age = bl.age.max(2);
        bl.actor_loc = st.location;
    }

    /// A car's hidden state (None while demolished).
    pub fn car_hidden(&self, i: usize) -> Option<CarHidden> {
        let p = self.players.get(i)?;
        let c = p.car.as_ref()?;
        let mut wheels = [WheelHidden::default(); 4];
        for (w, s) in wheels.iter_mut().zip(c.wheels.iter()) {
            let owner = s.contact.actor.map(|b| self.world.cw.bodies[b.0 as usize].owner);
            let ground = match owner {
                Some(crate::bt::Owner::Car(_)) => WheelGround::Car,
                Some(crate::bt::Owner::Ball) => WheelGround::Ball,
                _ => WheelGround::Arena,
            };
            *w = WheelHidden { contact: s.contact.has_contact, normal: s.contact.normal, ground };
        }
        Some(CarHidden {
            driving: c.driving,
            on_ground: c.on_ground,
            jumped: c.jumped,
            double_jumped: c.double_jumped,
            supersonic: c.supersonic,
            supersonic_time: c.time_below_supersonic,
            jump: Clock::of(&c.jump.c),
            jump_released: c.jump.deactivate,
            double_jump: Clock::of(&c.double_jump.c),
            dodge: Clock::of(&c.dodge.c),
            dodge_torque: c.dodge.torque,
            dodge_direction: c.dodge.direction,
            auto_flip: Clock::of(&c.flip.c),
            auto_flip_right: c.flip.flip_right,
            auto_flip_duration: c.flip.time,
            boost: Clock::of(&c.boost.c),
            boost_amount: c.boost.amount,
            air_control: Clock::of(&c.air.c),
            handbrake: c.sim.out_handbrake,
            held_input: input_row(&c.input),
            wheels,
            world_contact: c.world_contact.has_contact,
            world_contact_normal: c.world_contact.normal,
            bump_victim: p.last_hit_car,
            bump_time: p.last_hit_time,
            ball_impact_frame: p.last_ball_impact_frame,
            ball_touch_frame: p.last_ball_touch_frame,
        })
    }

    /// Set a car's hidden state. The held input goes through the game's input store (8-bit axes) without a jump press
    /// or release event. Returns false while the car is demolished.
    pub fn set_car_hidden(&mut self, i: usize, h: &CarHidden) -> bool {
        let ball = self.ball.as_ref().map(|b| b.body);
        let Some(p) = self.players.get_mut(i) else { return false };
        let Some(c) = p.car.as_mut() else { return false };
        c.driving = h.driving;
        c.game_active = true;
        c.on_ground = h.on_ground;
        c.jumped = h.jumped;
        c.double_jumped = h.double_jumped;
        c.supersonic = h.supersonic;
        c.time_below_supersonic = h.supersonic_time;
        h.jump.put(&mut c.jump.c);
        c.jump.deactivate = h.jump_released;
        h.double_jump.put(&mut c.double_jump.c);
        h.dodge.put(&mut c.dodge.c);
        c.dodge.torque = h.dodge_torque;
        c.dodge.direction = h.dodge_direction;
        h.auto_flip.put(&mut c.flip.c);
        c.flip.flip_right = h.auto_flip_right;
        c.flip.time = h.auto_flip_duration;
        h.boost.put(&mut c.boost.c);
        c.boost.amount = h.boost_amount;
        h.air_control.put(&mut c.air.c);
        c.sim.out_handbrake = h.handbrake;
        let mut input = crate::adapter::input_from_row(&h.held_input);
        input.activate_boost = input.holding_boost && 0.0 < c.boost.amount;
        c.store_input(input);
        p.input = input;
        for (s, w) in c.wheels.iter_mut().zip(h.wheels.iter()) {
            let arena = w.contact && w.ground == WheelGround::Arena;
            s.contact.has_contact = w.contact;
            // between ticks the wheel's touch event has always been sent for the contact it has
            s.contact_notified = w.contact;
            s.contact.world = arena;
            s.contact.sticky = arena;
            s.contact.normal = w.normal;
            // Read only by the wheels' ball-touch event, which asks whether the wheel is on the ball.
            s.contact.actor = if w.contact && w.ground == WheelGround::Ball { ball } else { None };
        }
        c.world_contact.has_contact = h.world_contact;
        c.world_contact.normal = h.world_contact_normal;
        p.last_hit_car = h.bump_victim;
        p.last_hit_time = h.bump_time;
        p.last_ball_impact_frame = h.ball_impact_frame;
        p.last_ball_touch_frame = h.ball_touch_frame;
        p.demolished = false;
        p.demolish_pending = false;
        true
    }

    /// The dodge torque and direction the game would give car `i` for a dodge input (forward, right), as its dodge
    /// start computes them (normalised, then the torque rounded to whole numbers).
    pub fn dodge_from_input(&self, i: usize, forward: f32, right: f32) -> Option<(Vec3, Vec3)> {
        let c = self.players.get(i)?.car.as_ref()?;
        let d = &c.dodge;
        let n2 = forward * forward + right * right;
        let (dx, dy) = if n2 != 1.0 {
            if 1e-08 <= n2 {
                let k = 1.0 / n2.sqrt();
                (k * forward, k * right)
            } else {
                (0.0, 0.0)
            }
        } else {
            (forward, right)
        };
        let ty = d.forward_torque * dx * d.torque_scale;
        let tx = d.side_torque * -dy * d.torque_scale;
        let sign = if c.gravity_z > 0.0 { -1.0 } else { 1.0 };
        let round = crate::math::rot::app_round;
        Some((v3(round(tx) as f32, round(ty) as f32, 0.0), v3(dx, dy * sign, 0.0)))
    }

    /// Demolish a car now, as a demolition would (its body goes; it respawns after `respawn_left` seconds).
    pub fn set_car_demolished(&mut self, i: usize, respawn_left: f32) {
        if self.players[i].car.is_some() {
            if let Some(m) = &mut self.frame_rng { m.demolished(); }
            self.remove_car_body(i);
        }
        let p = &mut self.players[i];
        p.demolished = true;
        p.demolish_pending = false;
        p.respawn_left = respawn_left;
        self.seq += 1;
        self.players[i].ticket_seq = self.seq;
    }

    /// Where the next respawns of car `i` go: (location uu, rotation), used in turn (otherwise the game's choice).
    pub fn queue_respawn_spot(&mut self, i: usize, loc: Vec3, rot: crate::math::rot::Rotator) {
        if self.respawn_queue.len() <= i {
            self.respawn_queue.resize(self.players.len().max(i + 1), Vec::new());
        }
        self.respawn_queue[i].push((loc, rot));
    }

    /// Where the next kickoff's cars go, one (location uu, rotation) per car in order (otherwise the game's choice).
    pub fn queue_kickoff_spots(&mut self, spots: Vec<(Vec3, crate::math::rot::Rotator)>) {
        self.spawn_queue.push(spots);
    }

    /// The arena's hidden state.
    pub fn round_hidden(&self) -> RoundHidden {
        let (phase, left) = match self.phase {
            Phase::Countdown { left } => (0, left as f32),
            Phase::Active => (1, 0.0),
            Phase::PostGoal { left } => (2, left),
            Phase::Ended => (3, 0.0),
        };
        let t = |f: &FrameTimer| if f.active { f.elapsed } else { -1.0 };
        RoundHidden {
            physics_frame: self.physics_frame,
            phase,
            phase_left: left,
            countdown_clock: self.countdown_clock.as_ref().map(|c| (c.remaining, t(&c.start_timer), t(&c.update_timer))),
            ball_has_been_hit: self.clock.ball_has_been_hit,
            ball_held_2d: self.ball_held_2d,
            game_time: self.clock.game_time,
            score: self.clock.score,
        }
    }

    /// Set the arena's hidden state. A countdown clock given here replaces the arena's (and turns it on).
    pub fn set_round_hidden(&mut self, r: &RoundHidden) {
        self.physics_frame = r.physics_frame;
        self.phase = match r.phase {
            0 => Phase::Countdown { left: r.phase_left as i32 },
            2 => Phase::PostGoal { left: r.phase_left },
            3 => Phase::Ended,
            _ => Phase::Active,
        };
        if let Some((remaining, start, update)) = r.countdown_clock {
            let mut c = self.countdown_clock.clone().unwrap_or_else(|| CountdownClock::new(3, 3));
            c.warmup = false;
            c.remaining = remaining;
            c.replicated_remaining = remaining;
            c.start_timer = if start >= 0.0 { FrameTimer { rate: 1.0, elapsed: start, active: true } } else { FrameTimer::default() };
            c.update_timer = if update >= 0.0 { FrameTimer { rate: 1.0, elapsed: update, active: true } } else { FrameTimer::default() };
            self.countdown_clock = Some(c);
        }
        self.clock.ball_has_been_hit = r.ball_has_been_hit;
        self.ball_held_2d = r.ball_held_2d;
        self.clock.game_time = r.game_time;
        self.clock.score = r.score;
    }

    /// Turn on the game's own 3-second kickoff countdown (a 1 s start timer, then 3, 2, 1), as the real game runs it,
    /// for every later kickoff. Off by default (a kickoff's countdown then lasts `RoundConfig::countdown_ticks`).
    pub fn use_game_countdown(&mut self, on: bool) {
        self.countdown_clock = if on { Some(CountdownClock::new(3, 3)) } else { None };
        self.rule_countdown = on;
    }
}

/// Field names of `CarHidden::to_row`, in order.
pub const CAR_HIDDEN_FIELDS: [&str; 65] = [
    "driving", "on_ground", "jumped", "double_jumped", "supersonic", "supersonic_time",
    "jump_active", "jump_time", "jump_released", "double_jump_active", "double_jump_time",
    "dodge_active", "dodge_time", "dodge_torque_x", "dodge_torque_y", "dodge_torque_z",
    "dodge_direction_x", "dodge_direction_y", "dodge_direction_z",
    "auto_flip_active", "auto_flip_time", "auto_flip_right", "auto_flip_duration", "boost_active", "boost_time", "boost_amount",
    "air_control_active", "air_control_time", "handbrake",
    "held_throttle", "held_steer", "held_pitch", "held_yaw", "held_roll", "held_jump", "held_boost", "held_handbrake",
    "wheel0_contact", "wheel1_contact", "wheel2_contact", "wheel3_contact",
    "wheel0_normal_x", "wheel0_normal_y", "wheel0_normal_z", "wheel1_normal_x", "wheel1_normal_y", "wheel1_normal_z",
    "wheel2_normal_x", "wheel2_normal_y", "wheel2_normal_z", "wheel3_normal_x", "wheel3_normal_y", "wheel3_normal_z",
    "world_contact", "world_contact_normal_x", "world_contact_normal_y", "world_contact_normal_z",
    "bump_victim", "bump_time", "ball_impact_frame", "ball_touch_frame",
    "wheel0_ground", "wheel1_ground", "wheel2_ground", "wheel3_ground",
];

impl CarHidden {
    /// One number per field (`CAR_HIDDEN_FIELDS`), as f64 so every f32 and frame number survives exactly.
    pub fn to_row(&self) -> Vec<f64> {
        let b = |x: bool| if x { 1.0 } else { 0.0 };
        let f = |x: f32| x as f64;
        let mut r = vec![
            b(self.driving), b(self.on_ground), b(self.jumped), b(self.double_jumped), b(self.supersonic), f(self.supersonic_time),
            b(self.jump.active), f(self.jump.time), b(self.jump_released), b(self.double_jump.active), f(self.double_jump.time),
            b(self.dodge.active), f(self.dodge.time), f(self.dodge_torque.x), f(self.dodge_torque.y), f(self.dodge_torque.z),
            f(self.dodge_direction.x), f(self.dodge_direction.y), f(self.dodge_direction.z),
            b(self.auto_flip.active), f(self.auto_flip.time), b(self.auto_flip_right), f(self.auto_flip_duration), b(self.boost.active), f(self.boost.time),
            f(self.boost_amount), b(self.air_control.active), f(self.air_control.time), f(self.handbrake),
        ];
        r.extend(self.held_input.iter().map(|&x| x as f64));
        r.extend(self.wheels.iter().map(|w| b(w.contact)));
        for w in &self.wheels {
            r.extend([f(w.normal.x), f(w.normal.y), f(w.normal.z)]);
        }
        r.extend([b(self.world_contact), f(self.world_contact_normal.x), f(self.world_contact_normal.y), f(self.world_contact_normal.z)]);
        r.extend([self.bump_victim.map_or(-1.0, |v| v as f64), f(self.bump_time), self.ball_impact_frame as f64, self.ball_touch_frame as f64]);
        r.extend(self.wheels.iter().map(|w| w.ground as u8 as f64));
        debug_assert_eq!(r.len(), CAR_HIDDEN_FIELDS.len());
        r
    }

    pub fn from_row(r: &[f64]) -> Option<CarHidden> {
        if r.len() != CAR_HIDDEN_FIELDS.len() {
            return None;
        }
        let b = |k: usize| r[k] != 0.0;
        let f = |k: usize| r[k] as f32;
        let v = |k: usize| v3(r[k] as f32, r[k + 1] as f32, r[k + 2] as f32);
        let mut held = [0.0f32; 8];
        for (k, h) in held.iter_mut().enumerate() {
            *h = f(29 + k);
        }
        let mut wheels = [WheelHidden::default(); 4];
        for (w, wh) in wheels.iter_mut().enumerate() {
            let ground = match r[61 + w] as i32 {
                1 => WheelGround::Car,
                2 => WheelGround::Ball,
                _ => WheelGround::Arena,
            };
            *wh = WheelHidden { contact: b(37 + w), normal: v(41 + 3 * w), ground };
        }
        Some(CarHidden {
            driving: b(0), on_ground: b(1), jumped: b(2), double_jumped: b(3), supersonic: b(4), supersonic_time: f(5),
            jump: Clock { active: b(6), time: f(7) }, jump_released: b(8), double_jump: Clock { active: b(9), time: f(10) },
            dodge: Clock { active: b(11), time: f(12) }, dodge_torque: v(13), dodge_direction: v(16),
            auto_flip: Clock { active: b(19), time: f(20) }, auto_flip_right: b(21), auto_flip_duration: f(22),
            boost: Clock { active: b(23), time: f(24) }, boost_amount: f(25), air_control: Clock { active: b(26), time: f(27) },
            handbrake: f(28), held_input: held, wheels,
            world_contact: b(53), world_contact_normal: v(54),
            bump_victim: if r[57] < 0.0 { None } else { Some(r[57] as usize) }, bump_time: f(58),
            ball_impact_frame: r[59] as i32, ball_touch_frame: r[60] as i32,
        })
    }
}

/// Field names of `BodyState::to_row`, in order.
pub const BODY_FIELDS: [&str; 20] = [
    "origin_x", "origin_y", "origin_z", "basis_00", "basis_01", "basis_02", "basis_10", "basis_11", "basis_12",
    "basis_20", "basis_21", "basis_22", "lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "asleep", "deactivation_time",
];

impl BodyState {
    pub fn to_row(&self) -> Vec<f64> {
        let mut r = vec![self.origin.x as f64, self.origin.y as f64, self.origin.z as f64];
        for row in &self.basis.r {
            r.extend([row.x as f64, row.y as f64, row.z as f64]);
        }
        r.extend([self.lin_vel.x as f64, self.lin_vel.y as f64, self.lin_vel.z as f64]);
        r.extend([self.ang_vel.x as f64, self.ang_vel.y as f64, self.ang_vel.z as f64]);
        r.extend([if self.asleep { 1.0 } else { 0.0 }, self.deactivation_time as f64]);
        r
    }

    pub fn from_row(r: &[f64]) -> Option<BodyState> {
        if r.len() != BODY_FIELDS.len() {
            return None;
        }
        let v = |k: usize| v3(r[k] as f32, r[k + 1] as f32, r[k + 2] as f32);
        Some(BodyState {
            origin: v(0),
            basis: Mat3 { r: [v(3), v(6), v(9)] },
            lin_vel: v(12),
            ang_vel: v(15),
            asleep: r[18] != 0.0,
            deactivation_time: r[19] as f32,
        })
    }
}
