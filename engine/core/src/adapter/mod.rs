use crate::game::arena::{Arena, Phase, RoundConfig};
use crate::game::car::VehicleInputs;
use crate::game::config::{Preset, preset};
use crate::game::pads::{NUM_PADS, PADS};
use crate::game::rb_state::RbState;
use crate::math::rot::{Rotator, matrix_to_rotator};
use crate::math::{Vec3, v3};

pub const MAX_CARS: usize = 6;

pub const A_BALL_POS: usize = 0;
pub const A_BALL_VEL: usize = 3;
pub const A_BALL_ANGVEL: usize = 6;
pub const A_PAD_ACTIVE: usize = 9;
pub const A_PAD_TIME_LEFT: usize = 9 + NUM_PADS;
pub const A_TIME_SINCE_TOUCH: usize = 9 + 2 * NUM_PADS;
pub const A_EPISODE_TIME: usize = A_TIME_SINCE_TOUCH + 1;
pub const A_DIM: usize = A_EPISODE_TIME + 1;

pub const C_PRESENT: usize = 0;
pub const C_TEAM: usize = 1;
pub const C_POS: usize = 2;
pub const C_VEL: usize = C_POS + 3;
pub const C_ANGVEL: usize = C_VEL + 3;
pub const C_FORWARD: usize = C_ANGVEL + 3;
pub const C_UP: usize = C_FORWARD + 3;
pub const C_BOOST: usize = C_UP + 3;
pub const C_ON_GROUND: usize = C_BOOST + 1;
pub const C_HAS_JUMPED: usize = C_ON_GROUND + 1;
pub const C_HAS_DOUBLE_JUMPED: usize = C_HAS_JUMPED + 1;
pub const C_HAS_FLIPPED: usize = C_HAS_DOUBLE_JUMPED + 1;
pub const C_IS_JUMPING: usize = C_HAS_FLIPPED + 1;
pub const C_IS_FLIPPING: usize = C_IS_JUMPING + 1;
pub const C_CAN_DODGE: usize = C_IS_FLIPPING + 1;
pub const C_DODGE_TIME_LEFT: usize = C_CAN_DODGE + 1;
pub const C_IS_SUPERSONIC: usize = C_DODGE_TIME_LEFT + 1;
pub const C_IS_DEMOED: usize = C_IS_SUPERSONIC + 1;
pub const C_DEMO_TIME_LEFT: usize = C_IS_DEMOED + 1;
pub const C_AIR_TIME_SINCE_JUMP: usize = C_DEMO_TIME_LEFT + 1;
pub const C_HANDBRAKE_VAL: usize = C_AIR_TIME_SINCE_JUMP + 1;
pub const C_IS_AUTO_FLIPPING: usize = C_HANDBRAKE_VAL + 1;
pub const C_WHEELS_CONTACT: usize = C_IS_AUTO_FLIPPING + 1;
pub const C_DIM: usize = C_WHEELS_CONTACT + 1;

pub const E_TOUCHED: usize = 0;
pub const E_DEMO_INFLICTED: usize = 1;
pub const E_DEMOED: usize = 2;
pub const E_BUMP_INFLICTED: usize = 3;
pub const E_DIM: usize = 4;
pub const AE_GOAL: usize = 0;
pub const AE_DIM: usize = 1;

pub const H_ROT: usize = 0;
pub const H_FLIP_TIME: usize = 9;
pub const H_JUMP_TICKS: usize = 10;
pub const H_AIR_TIME: usize = 11;
pub const H_AIR_TIME_SINCE_JUMP: usize = 12;
pub const H_BOOSTING_TIME: usize = 13;
pub const H_TIME_SINCE_BOOSTED: usize = 14;
pub const H_SS_GRACE: usize = 15;
pub const H_HANDBRAKE: usize = 16;
pub const H_BUMP_COOLDOWN: usize = 17;
pub const H_AUTO_FLIP_TIMER: usize = 18;
pub const H_IS_BOOSTING: usize = 19;
pub const H_WHEEL: usize = 20;
pub const H_ROTATOR: usize = 24;
pub const H_DIM: usize = 27;

pub fn arena_field_names() -> Vec<String> {
    let mut n = vec![String::new(); A_DIM];
    for i in 0..3 {
        n[A_BALL_POS + i] = format!("ball_pos{i}");
        n[A_BALL_VEL + i] = format!("ball_vel{i}");
        n[A_BALL_ANGVEL + i] = format!("ball_angvel{i}");
    }
    for i in 0..NUM_PADS {
        n[A_PAD_ACTIVE + i] = format!("pad_active{i}");
        n[A_PAD_TIME_LEFT + i] = format!("pad_time_left{i}");
    }
    n[A_TIME_SINCE_TOUCH] = "time_since_touch".into();
    n[A_EPISODE_TIME] = "episode_time".into();
    n
}

pub fn car_field_names() -> Vec<String> {
    let mut n = vec![String::new(); C_DIM];
    let mut set = |i: usize, s: &str| n[i] = s.to_string();
    set(C_PRESENT, "present");
    set(C_TEAM, "team");
    for (k, a) in ["pos", "vel", "angvel", "forward", "up"].iter().enumerate() {
        for i in 0..3 {
            set(C_POS + k * 3 + i, &format!("{a}{i}"));
        }
    }
    for (i, s) in [
        (C_BOOST, "boost"),
        (C_ON_GROUND, "on_ground"),
        (C_HAS_JUMPED, "has_jumped"),
        (C_HAS_DOUBLE_JUMPED, "has_double_jumped"),
        (C_HAS_FLIPPED, "has_flipped"),
        (C_IS_JUMPING, "is_jumping"),
        (C_IS_FLIPPING, "is_flipping"),
        (C_CAN_DODGE, "can_dodge"),
        (C_DODGE_TIME_LEFT, "dodge_time_left"),
        (C_IS_SUPERSONIC, "is_supersonic"),
        (C_IS_DEMOED, "is_demoed"),
        (C_DEMO_TIME_LEFT, "demo_time_left"),
        (C_AIR_TIME_SINCE_JUMP, "air_time_since_jump"),
        (C_HANDBRAKE_VAL, "handbrake_val"),
        (C_IS_AUTO_FLIPPING, "is_auto_flipping"),
        (C_WHEELS_CONTACT, "wheels_contact"),
    ] {
        set(i, s);
    }
    n
}

pub fn event_field_names() -> Vec<String> {
    vec!["touched".into(), "demo_inflicted".into(), "demoed".into(), "bump_inflicted".into()]
}

pub fn hidden_field_names() -> Vec<String> {
    let mut n = vec![String::new(); H_DIM];
    for i in 0..9 {
        n[H_ROT + i] = format!("rot{i}");
    }
    for (i, name) in ["flip_time", "jump_ticks", "air_time", "air_time_since_jump", "boosting_time", "time_since_boosted", "supersonic_grace", "handbrake", "bump_cooldown", "auto_flip_timer", "is_boosting"].iter().enumerate() {
        n[H_FLIP_TIME + i] = (*name).to_string();
    }
    for i in 0..4 {
        n[H_WHEEL + i] = format!("wheel{i}");
    }
    n[H_ROTATOR] = "rot_pitch".into();
    n[H_ROTATOR + 1] = "rot_yaw".into();
    n[H_ROTATOR + 2] = "rot_roll".into();
    n
}

pub fn pad_positions() -> Vec<[f32; 3]> {
    PADS.iter().map(|p| [p.loc.x, p.loc.y, p.loc.z]).collect()
}

pub fn pad_is_big() -> Vec<bool> {
    PADS.iter().map(|p| p.big).collect()
}

pub fn input_from_row(r: &[f32]) -> VehicleInputs {
    let c = |v: f32| v.clamp(-1.0, 1.0);
    VehicleInputs {
        throttle: c(r[0]),
        steer: c(r[1]),
        pitch: c(r[2]),
        yaw: c(r[3]),
        roll: c(r[4]),
        dodge_forward: -c(r[2]),
        dodge_right: c(r[3]),
        handbrake: r[7] > 0.5,
        jump: r[5] > 0.5,
        holding_boost: r[6] > 0.5,
        ..VehicleInputs::default()
    }
}

pub fn state_rotator(s: &RbState) -> Rotator {
    matrix_to_rotator(&crate::game::rb_state::quat_to_matrix(s.quat))
}

#[derive(Clone)]
pub struct ArenaSlot {
    pub arena: Arena,
    pub num_cars: usize,
    pub time_since_touch: f32,
    pub episode_time: f32,

    pub applied: Vec<VehicleInputs>,
    pub delay: Vec<usize>,
    pub queue: Vec<Vec<VehicleInputs>>,

    pub events: Vec<[f32; E_DIM]>,
    pub goal: f32,
}

impl ArenaSlot {

    pub fn new(team_size: usize, body: &str, meshes: &std::path::Path, cfg: RoundConfig) -> Result<ArenaSlot, String> {
        Self::new_with_layout(team_size, body, meshes, cfg, crate::game::pads::PadLayout::Stadium)
    }

    pub fn new_with_layout(team_size: usize, body: &str, meshes: &std::path::Path, cfg: RoundConfig, layout: crate::game::pads::PadLayout) -> Result<ArenaSlot, String> {
        crate::data::load(meshes).map_err(|e| e.to_string())?;
        let names: Vec<&str> = body.split(',').map(str::trim).collect();

        let pick = |i: usize| -> Result<&'static Preset, String> {
            let n = names[i.min(names.len() - 1)];
            if let Some(b) = crate::game::config::body_spawn(n) {
                return preset(b.preset).ok_or_else(|| format!("no preset {} for body {n}", b.preset));
            }
            preset(n).or_else(|| preset(&capitalize(n))).ok_or_else(|| format!("no preset {n}"))
        };
        let mut list = Vec::new();
        for t in 0..2u8 {
            for i in 0..team_size {
                list.push((pick(list.len() + i * 0)?, t));
            }
        }
        let mut arena = Arena::new_with_layout(&list, meshes, cfg, layout).map_err(|e| e.to_string())?;
        let n = list.len();
        for i in 0..n {
            if let Some(b) = crate::game::config::body_spawn(names[i.min(names.len() - 1)]) {
                arena.players[i].body = b;
            }
        }
        Ok(ArenaSlot { arena, num_cars: n, time_since_touch: 0.0, episode_time: 0.0, applied: vec![VehicleInputs::default(); n], delay: vec![0; n], queue: vec![Vec::new(); n], events: vec![[0.0; E_DIM]; n], goal: -1.0 })
    }

    pub fn step(&mut self, controls: &[VehicleInputs]) {
        self.step_frame(controls, 1);
    }

    pub fn step_frame(&mut self, controls: &[VehicleInputs], steps: i32) {
        self.step_frame_with_time(controls, steps, self.arena.dt * steps as f32, self.arena.render_alpha);
    }

    pub fn step_frame_with_time(&mut self, controls: &[VehicleInputs], steps: i32, frame_time: f32, render_alpha: f32) {
        for i in 0..self.num_cars {
            self.applied[i] = if self.delay[i] == 0 {
                controls[i]
            } else {
                self.queue[i].push(controls[i]);
                if self.queue[i].len() > self.delay[i] { self.queue[i].remove(0) } else { self.applied[i] }
            };
        }
        for e in self.events.iter_mut() {
            *e = [0.0; E_DIM];
        }
        self.arena.frame_with_time(&self.applied, steps, frame_time, render_alpha);
        self.goal = self.arena.goal_scored_this_tick.map_or(-1.0, |t| t as f32);
        let mut touched = false;
        for (i, p) in self.arena.players.iter().enumerate() {
            let ev = p.events;
            let row = &mut self.events[i];
            row[E_TOUCHED] = ev.touched_ball as i32 as f32;
            row[E_DEMO_INFLICTED] = ev.demo_inflicted as i32 as f32;
            row[E_DEMOED] = ev.demoed as i32 as f32;
            row[E_BUMP_INFLICTED] = ev.bump_inflicted as i32 as f32;
            touched |= ev.touched_ball;
        }
        self.time_since_touch = if touched { 0.0 } else { self.time_since_touch + frame_time };
        self.episode_time += frame_time;
    }

    pub fn write_arena(&self, row: &mut [f32]) {
        if let Some(b) = self.arena.ball.as_ref() {

            let s = &if b.first_frame { b.birth_state } else { b.read_memo.peek(&self.arena.world.cw.bodies[b.body.0 as usize], &b.limits) };
            put3(row, A_BALL_POS, s.location);
            put3(row, A_BALL_VEL, s.velocity);
            put3(row, A_BALL_ANGVEL, s.spin);
        } else {
            for i in 0..9 {
                row[i] = f32::NAN;
            }
        }
        for i in 0..NUM_PADS {
            let p = self.arena.pads.pads[i];
            row[A_PAD_ACTIVE + i] = if p.picked_up { 0.0 } else { 1.0 };
            row[A_PAD_TIME_LEFT + i] = p.respawn_left;
        }
        row[A_TIME_SINCE_TOUCH] = self.time_since_touch;
        row[A_EPISODE_TIME] = self.episode_time;
    }

    pub fn write_car(&self, i: usize, row: &mut [f32]) {
        let p = &self.arena.players[i];
        row[C_PRESENT] = 1.0;
        row[C_TEAM] = p.team as f32;
        let Some(c) = p.car.as_ref() else {

            if let Some((loc, rot)) = p.pending_spawn {
                put3(row, C_POS, loc);
                let m = crate::math::rot::rotator_to_matrix(rot);
                put3(row, C_FORWARD, m.r[0]);
                put3(row, C_UP, m.r[2]);
            }
            row[C_IS_DEMOED] = p.demolished as i32 as f32;
            row[C_DEMO_TIME_LEFT] = p.respawn_left;
            return;
        };
        let s = &c.read_memo.peek(&self.arena.world.cw.bodies[c.body.0 as usize], &c.limits);
        put3(row, C_POS, s.location);
        if p.fresh_body {

            put3(row, C_POS, v3(p.spawn_loc.x, p.spawn_loc.y, crate::game::arena::SPAWN_ACTOR_Z));
        }
        put3(row, C_VEL, s.velocity);
        put3(row, C_ANGVEL, s.spin);
        let b = &self.arena.world.cw.bodies[c.body.0 as usize];
        put3(row, C_FORWARD, b.transform.basis.col(0));
        put3(row, C_UP, b.transform.basis.col(2));
        row[C_BOOST] = c.boost.amount * 100.0;
        row[C_ON_GROUND] = c.on_ground as i32 as f32;
        row[C_HAS_JUMPED] = c.jumped as i32 as f32;
        row[C_HAS_DOUBLE_JUMPED] = c.double_jumped as i32 as f32;
        row[C_HAS_FLIPPED] = c.dodge.c.active as i32 as f32;
        row[C_IS_JUMPING] = c.jump.c.active as i32 as f32;
        row[C_IS_FLIPPING] = c.dodge.c.active as i32 as f32;
        row[C_IS_SUPERSONIC] = c.supersonic as i32 as f32;
        row[C_IS_DEMOED] = 0.0;
        row[C_DEMO_TIME_LEFT] = 0.0;
        row[C_HANDBRAKE_VAL] = c.sim.out_handbrake;
        row[C_WHEELS_CONTACT] = crate::game::drive::num_wheel_contacts(&c.wheels) as f32;
    }

    pub fn write_hidden(&self, i: usize, row: &mut [f32]) {
        let p = &self.arena.players[i];
        let Some(c) = p.car.as_ref() else { return };
        let b = &self.arena.world.cw.bodies[c.body.0 as usize];
        for (k, r) in (0..3).map(|k| (k, b.transform.basis.col(k))) {
            row[H_ROT + k * 3] = r.x;
            row[H_ROT + k * 3 + 1] = r.y;
            row[H_ROT + k * 3 + 2] = r.z;
        }
        row[H_FLIP_TIME] = c.dodge.c.active_time();
        row[H_JUMP_TICKS] = c.jump.c.active_time() * 120.0;
        row[H_AIR_TIME] = c.jump.c.inactive_time();
        row[H_AIR_TIME_SINCE_JUMP] = c.jump.c.inactive_time();
        row[H_BOOSTING_TIME] = c.boost.c.active_time();
        row[H_TIME_SINCE_BOOSTED] = c.boost.c.inactive_time();
        row[H_SS_GRACE] = c.time_below_supersonic;
        row[H_HANDBRAKE] = c.sim.out_handbrake;
        row[H_BUMP_COOLDOWN] = 0.0;
        row[H_AUTO_FLIP_TIMER] = c.flip.c.active_time();
        row[H_IS_BOOSTING] = c.boost.c.active as i32 as f32;
        for w in 0..4 {
            row[H_WHEEL + w] = c.wheels[w].contact.has_contact as i32 as f32;
        }
        let r = state_rotator(&c.state);
        row[H_ROTATOR] = r.pitch as f32;
        row[H_ROTATOR + 1] = r.yaw as f32;
        row[H_ROTATOR + 2] = r.roll as f32;
    }

    pub fn set_car_row(&mut self, i: usize, r: &[f32]) {
        let quat = crate::game::arena::state_set_rotation(r[10], r[9], r[11]);
        let s = RbState { quat, location: v3(r[0], r[1], r[2]), velocity: v3(r[3], r[4], r[5]), spin: v3(r[6], r[7], r[8]), sleeping: false };
        self.arena.set_car_state(i, &s, Some(r[12]));
    }

    pub fn set_ball_row(&mut self, b: &[f32], sleep_at_rest: bool) {
        let s = RbState { location: v3(b[0], b[1], b[2]), velocity: v3(b[3], b[4], b[5]), spin: v3(b[6], b[7], b[8]), ..RbState::default() };
        self.arena.set_ball_state(&s, sleep_at_rest);
    }

    pub fn active(&self) -> bool {
        self.arena.phase == Phase::Active
    }
}

pub fn rot_from_ypr(yaw: f32, pitch: f32, roll: f32) -> Rotator {
    let u = |a: f32| ((a * 32768.0 / core::f32::consts::PI) as f64).round() as i32;
    Rotator { pitch: u(pitch), yaw: u(yaw), roll: u(roll) }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

#[inline]
fn put3(row: &mut [f32], at: usize, v: Vec3) {
    row[at] = v.x;
    row[at + 1] = v.y;
    row[at + 2] = v.z;
}
