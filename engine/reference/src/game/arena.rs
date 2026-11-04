use super::ball::{self, Ball};
use super::car::{Car, VehicleInputs};
use super::config::{self, Preset, new_car};
use super::explosion::{Explosion, ExplosionHit, apply_hit, orientation_for_goal};
use super::pads::{NUM_PADS, Pads};
use super::rb_state::{self, ForceMode, RbState};
use super::rules::{self, Held, Impact, NotifyPoint, RbCollision};
use super::round::{CountdownClock, CrtRand, FrameTimer, SpawnSelector};
use crate::bt::algo_compound::CompoundShape;
use crate::bt::shapes::{BoxShape, Shape};
use crate::bt::world::World;
use crate::bt::{BodyId, Owner, ShapeId};
use crate::math::rot::{Rotator, rotator_to_matrix};
use crate::math::{Mat3, Quat, Transform, Vec3, v3};

pub const PHYSICS_RATE: f32 = 120.0;
fn pad_tick_post() -> bool {
    static L: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    *L.get_or_init(|| std::env::var("CATSIM_PAD_TICK_POST").map_or(false, |v| v == "1"))
}
fn pad_respawn_mesh() -> bool {
    static L: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    *L.get_or_init(|| std::env::var("CATSIM_PAD_RESPAWN_MESH").map_or(true, |v| v != "0"))
}
fn explosion_debug() -> bool {
    static L: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *L.get_or_init(|| std::env::var("CATSIM_EXPLOSION_DEBUG").map_or(false, |v| v == "1"))
}
fn legacy_explosion() -> bool {
    static L: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *L.get_or_init(|| std::env::var("CATSIM_LEGACY_EXPLOSION").map_or(false, |v| v == "1"))
}

pub const GROUND_TOLERANCE_Z: f32 = f32::from_bits(0x3f75_c28f);

pub const GRAVITY_UU: f32 = -650.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Actor {
    Ball,
    Car(usize),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CarEvents {
    pub touched_ball: bool,
    pub bump_inflicted: bool,
    pub bumped: bool,
    pub demo_inflicted: bool,
    pub demoed: bool,
    pub picked_pad: i32,
    pub goal_for_team: bool,
}

#[derive(Clone, Debug)]
pub struct Player {
    pub team: u8,
    pub preset: &'static Preset,
    pub car: Option<Car>,
    pub contacts: Vec<RbCollision>,
    pub last_ball_impact_frame: i32,
    pub last_ball_touch_frame: i32,

    pub last_wheels_hit_ball_time: f32,
    pub last_hit_car: Option<usize>,
    pub last_hit_time: f32,
    pub local_offset: Vec3,

    pub respawn_left: f32,

    pub ticket_seq: u64,

    pub spawn_seq: u64,
    pub demolished: bool,
    pub demolish_pending: bool,

    pub pending_spawn: Option<(Vec3, Rotator)>,

    pub fresh_body: bool,

    pub spawn_dz: f32,

    pub body: &'static config::BodySpawn,

    pub spawn_spot: bool,

    pub birth_delay: u32,

    pub spawn_loc: Vec3,

    pub actor_loc: Vec3,

    pub actor_rot: Rotator,

    pub touching: u64,
    pub events: CarEvents,
    pub input: VehicleInputs,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Phase {

    Countdown { left: i32 },
    Active,

    PostGoal { left: f32 },
    Ended,
}

#[derive(Clone, Debug)]
pub struct RoundConfig {

    pub countdown_ticks: i32,

    pub post_goal_time: f32,

    pub match_seconds: Option<f32>,
}

impl Default for RoundConfig {
    fn default() -> RoundConfig {
        RoundConfig { countdown_ticks: 480, post_goal_time: 3.0, match_seconds: Some(300.0) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Clock {
    pub game_time: f32,
    pub overtime: bool,
    pub ball_has_been_hit: bool,
    pub hit_delay_left: f32,
    pub score: [i32; 2],
    pub round_num: i32,
    pub waiting_end_round: bool,
}

#[derive(Clone)]
pub struct Arena {
    pub world: World,
    pub ball: Option<Ball>,

    pub ball_shape: ShapeId,
    pub players: Vec<Player>,

    pub notifies: Vec<Actor>,

    pub vehicles: Vec<usize>,
    pub pads: Pads,
    pub phase: Phase,
    pub clock: Clock,
    pub config: RoundConfig,
    pub physics_frame: i32,
    pub explosion: Option<Explosion>,
    pub dt: f32,

    pub spawn_queue: Vec<Vec<(Vec3, Rotator)>>,

    pub respawn_queue: Vec<Vec<(Vec3, Rotator)>>,

    pub selector: Option<SpawnSelector>,

    pub crt_state: Option<u32>,

    pub frame_rng: Option<super::frame_rng::FrameRng>,
    frame_rng_motion: u32,

    frame_rng_motion_by: Vec<u32>,

    frame_rng_pickups_now: u32,

    pub selector_failures: u32,
    pub goal_scored_this_tick: Option<u8>,

    pub pad_rule_on: bool,
    pub ball_held_2d: bool,

    pub ball_on_floor: bool,

    pub ball_world_nz: f32,

    pub scripted: bool,

    pub rule_countdown: bool,
    pub rule_restarts: bool,
    pub rule_respawns: bool,
    pub rng: u64,
    scratch_hits: Vec<ExplosionHit>,

    pub players_order: Vec<usize>,

    pub seq: u64,

    next_handles: Vec<u16>,
    forced_restart: bool,
    forced_round_start: bool,
    countdown_timer: FrameTimer,

    pub countdown_clock: Option<CountdownClock>,
    post_goal_timer: FrameTimer,
    demolish_queue: Vec<usize>,

    pub debug_capture: bool,
    pub debug_last: Vec<String>,

    pub render_alpha: f32,
}

pub const KICKOFF_SPOTS: [(f32, f32, i32); 5] = [(-2048.0, -2560.0, 8192), (2048.0, -2560.0, 24576), (-256.0, -3840.0, 16384), (256.0, -3840.0, 16384), (0.0, -4608.0, 16384)];

pub const RESPAWN_SPOTS: [(f32, f32, i32); 4] = [(-2304.0, -4608.0, 16384), (-2688.0, -4608.0, 16384), (2304.0, -4608.0, 16384), (2688.0, -4608.0, 16384)];

pub const SPAWN_ACTOR_Z: f32 = 83.0;

pub const SPAWN_BODY_Z_BT: f32 = f32::from_bits(0x3f38_6bdd);
pub const DEMO_RESPAWN_TIME: f32 = 3.0;
pub const START_BOOST: f32 = f32::from_bits(0x3eaa_7efa);

impl Arena {

    pub fn new(presets: &[(&'static Preset, u8)], meshes_dir: &std::path::Path, config: RoundConfig) -> std::io::Result<Arena> {
        Self::new_with_layout(presets, meshes_dir, config, super::pads::PadLayout::Stadium)
    }

    pub fn new_with_layout(presets: &[(&'static Preset, u8)], meshes_dir: &std::path::Path, config: RoundConfig, layout: super::pads::PadLayout) -> std::io::Result<Arena> {
        crate::data::load(meshes_dir)?;
        let mut world = World::new();
        world.cw.add_soccar_statics_for_map(meshes_dir, layout == super::pads::PadLayout::Stadium)?;

        for k in 0..world.cw.objects.len() {
            let id = world.cw.objects[k];
            world.cw.bodies[id.0 as usize].sticky_wheels = true;
        }
        world.set_gravity(v3(0.0, 0.0, GRAVITY_UU * 0.02));
        world.set_use_continuous(true);
        let ball_shape = world.cw.add_shape(ball::shape());
        let players = presets
            .iter()
            .map(|(p, team)| Player {
                team: *team,
                preset: p,
                car: None,
                contacts: Vec::new(),
                last_ball_impact_frame: 0,
                last_ball_touch_frame: 0,
                last_wheels_hit_ball_time: 0.0,
                last_hit_car: None,
                last_hit_time: 0.0,
                local_offset: v3(p.offset().x * 50.0, 0.0, p.offset().z * 50.0),
                respawn_left: 0.0,
                ticket_seq: 0,
                spawn_seq: 0,
                demolished: false,
                demolish_pending: false,
                pending_spawn: None,
                fresh_body: false,
                spawn_dz: 0.0,
                body: &config::BODIES[0],
                spawn_spot: false,
                birth_delay: 0,
                spawn_loc: v3(0.0, 0.0, 0.0),
                actor_loc: Vec3::ZERO,
                actor_rot: Rotator::default(),
                touching: 0,
                events: CarEvents { picked_pad: -1, ..CarEvents::default() },
                input: VehicleInputs::default(),
            })
            .collect();
        Ok(Arena {
            world,
            ball: None,
            ball_shape,
            players,
            notifies: Vec::new(),
            vehicles: Vec::new(),
            pads: Pads { layout, ..Pads::default() },
            phase: Phase::Countdown { left: config.countdown_ticks },
            countdown_timer: FrameTimer::new(if config.countdown_ticks <= 2 { 0.01 } else { config.countdown_ticks as f32 / PHYSICS_RATE }),
            countdown_clock: None,
            post_goal_timer: FrameTimer::default(),
            clock: Clock { game_time: config.match_seconds.unwrap_or(0.0), ..Clock::default() },
            config,
            physics_frame: 0,
            explosion: None,
            dt: 1.0 / PHYSICS_RATE,
            spawn_queue: Vec::new(),
            respawn_queue: Vec::new(),
            selector: None,
            crt_state: None,
            frame_rng: None,
            frame_rng_motion: 0,
            frame_rng_motion_by: Vec::new(),
            frame_rng_pickups_now: 0,
            selector_failures: 0,
            goal_scored_this_tick: None,
            pad_rule_on: true,
            ball_held_2d: false,
            ball_on_floor: false,
            ball_world_nz: -2.0,
            scripted: false,
            rule_countdown: false,
            rule_restarts: false,
            rule_respawns: false,
            rng: 0x9e37_79b9_7f4a_7c15,
            scratch_hits: Vec::new(),
            players_order: Vec::new(),
            seq: 0,
            next_handles: Vec::new(),
            forced_restart: false,
            forced_round_start: false,
            demolish_queue: Vec::new(),
            debug_capture: false,
            debug_last: Vec::new(),
            render_alpha: 0.5,
        })
    }

    pub fn set_next_handles(&mut self, order: &[u16]) {
        self.next_handles = order.to_vec();
    }

    fn apply_next_handles(&mut self) {
        let order = core::mem::take(&mut self.next_handles);
        if order.is_empty() {
            return;
        }
        let br = &mut self.world.cw.broadphase;
        let mut h = br.first_free;
        for _ in 0..order.len() {
            h = br.handles[h as usize].min_edges[0];
        }
        for w in order.windows(2) {
            br.handles[w[0] as usize].min_edges[0] = w[1];
        }
        br.handles[order[order.len() - 1] as usize].min_edges[0] = h;
        br.first_free = order[0];
    }

    pub fn start_at_drop(&mut self, spots: Option<Vec<(Vec3, Rotator)>>) {
        self.explosion = None;
        self.destroy_ball();
        if let Some(q) = spots {
            self.spawn_queue.insert(0, q);
        }
        self.reset_ball();
        self.reset_players();
        for i in 0..self.players.len() {
            self.init_car_body(i);
        }
        self.pads.reset();
        self.start_round();
    }

    pub fn start_like_restart(&mut self, spots: Option<Vec<(Vec3, Rotator)>>, steps: i32) {
        self.explosion = None;
        self.destroy_ball();
        for i in 0..self.players.len() {
            self.destroy_car(i);
        }
        self.apply_next_handles();
        if let Some(q) = spots {
            self.spawn_queue.insert(0, q);
        }
        self.phase = Phase::Countdown { left: 2 };
        self.countdown_timer = FrameTimer::new(0.01);
        if let Some(c) = self.countdown_clock.as_mut() { c.begin(); }
        self.reset_ball();
        self.reset_players();
        self.pads.reset();
        self.clock.ball_has_been_hit = false;
        let none = vec![VehicleInputs::default(); self.players.len()];
        for _ in 0..steps {
            self.tick(&none);
        }
        self.script_tick(self.dt * steps as f32);
    }

    pub fn post_goal_elapsed(&self) -> f32 {
        if self.post_goal_timer.active { self.post_goal_timer.elapsed } else { -1.0 }
    }

    #[inline]
    pub fn physics_time(&self) -> f32 {
        (1.0 / PHYSICS_RATE) * self.physics_frame as f32
    }

    pub fn frame(&mut self, inputs: &[VehicleInputs], steps: i32) {
        self.frame_with_time(inputs, steps, self.dt * steps as f32, self.render_alpha);
    }

    pub fn frame_with_time(&mut self, inputs: &[VehicleInputs], steps: i32, frame_time: f32, render_alpha: f32) {
        if explosion_debug() { self.debug_last.clear(); }
        assert!(steps >= 0 && frame_time.is_finite() && frame_time >= 0.0);
        assert!(render_alpha.is_finite() && (0.0..=1.0).contains(&render_alpha));
        self.frame_rng_motion = 0;
        self.frame_rng_motion_by.clear();
        self.frame_rng_pickups_now = 0;
        if let Some(m) = &mut self.frame_rng {
            m.pad_before = self.pads.pads.map(|p| p.respawn_left);
            m.steps = steps;
        }
        self.render_alpha = render_alpha;
        self.goal_scored_this_tick = None;
        for p in self.players.iter_mut() {
            p.events = CarEvents { picked_pad: -1, ..CarEvents::default() };
        }

        if steps == 0 {
            for (i, p) in self.players.iter_mut().enumerate() {
                p.input = inputs.get(i).copied().unwrap_or_default();
            }
        }

        self.tick_spawn_tickets(frame_time);
        if let Some(b) = self.ball.as_mut() {

            b.first_frame = false;
        }
        if !pad_tick_post() {
            self.pads.frame_tick(frame_time);
            if self.pad_rule_on {
                self.pads_touching();
            }
        }
        let was_warmup = self.countdown_clock.as_ref().map_or(false, |c| c.warmup);
        self.game_event_tick(frame_time, steps);
        if was_warmup && self.countdown_clock.as_ref().map_or(false, |c| !c.warmup) {
            if let (Some(model), Some(state)) = (&mut self.frame_rng, &mut self.crt_state) {
                model.enter_countdown(state);
            }
        }
        for _ in 0..steps {
            self.tick(inputs);
        }
        self.script_tick(frame_time);
        self.finish_frame_rng(frame_time);
    }

    fn finish_frame_rng(&mut self, frame_time: f32) {
        if let (Some(model), Some(state)) = (&mut self.frame_rng, &mut self.crt_state) {
            let pickups = self.pads.pads.iter().zip(model.pad_before).filter(|(p, old)| p.respawn_left > old + 0.5).count() as u32;
            let now = self.frame_rng_pickups_now.min(pickups);
            model.motion_by.clear();
            model.motion_by.extend_from_slice(&self.frame_rng_motion_by);
            model.finish(state, frame_time, model.steps, pickups - now, self.frame_rng_motion, now);
        }
    }

    fn game_event_tick(&mut self, frame_time: f32, steps: i32) {
        if core::mem::take(&mut self.forced_restart) {
            self.start_replay();
            return;
        }
        match self.phase {
            Phase::Countdown { left } => {
                let expired = if self.scripted && !self.rule_countdown {
                    left <= steps
                } else if let Some(c) = self.countdown_clock.as_mut() {
                    c.advance(frame_time)
                } else {
                    self.countdown_timer.advance(frame_time, 1.0)
                };
                if core::mem::take(&mut self.forced_round_start) || expired {
                    self.start_round();
                } else {
                    self.phase = Phase::Countdown { left: left - steps };
                }
            }
            Phase::Active => self.active_tick(frame_time),
            Phase::PostGoal { .. } => {
                let expired = self.post_goal_timer.advance(frame_time, 1.0);
                let left = self.post_goal_timer.remaining();
                if expired && (!self.scripted || self.rule_restarts) {
                    if self.clock.overtime && self.clock.score[0] != self.clock.score[1] && self.config.match_seconds.is_some() {

                        self.explosion = None;
                        self.phase = Phase::Ended;
                    } else {
                        self.start_replay();
                    }
                } else {
                    self.phase = Phase::PostGoal { left };
                }
            }
            Phase::Ended => {}
        }
    }

    pub fn tick(&mut self, inputs: &[VehicleInputs]) {

        self.flush_demolished();
        self.physics_frame += 1;
        for p in self.players.iter_mut() {
            if p.pending_spawn.is_some() && p.birth_delay > 0 {
                p.birth_delay -= 1;
            }
        }

        for i in 0..self.players.len() {
            let mut input = inputs.get(i).copied().unwrap_or_default();
            let p = &mut self.players[i];

            p.input = input;
            if let Some(car) = p.car.as_mut() {
                let old = (car.jump.c.active, car.double_jump.c.active, car.dodge.c.active);
                car.push_input(&mut input);
                if self.frame_rng.is_some() {
                    let n = 6 * (!old.0 && car.jump.c.active) as u32
                        + 6 * (!old.1 && car.double_jump.c.active) as u32
                        + 7 * (!old.2 && car.dodge.c.active) as u32;
                    self.frame_rng_motion += n;
                    if self.frame_rng_motion_by.len() <= i { self.frame_rng_motion_by.resize(i + 1, 0); }
                    self.frame_rng_motion_by[i] += n;
                }
            }
        }
        self.physics_tick();
    }

    pub fn tick_held(&mut self) {
        self.flush_demolished();
        self.physics_frame += 1;
        self.physics_tick();
    }

    fn physics_tick(&mut self) {
        self.ball_world_nz = -2.0;
        for p in self.players.iter_mut() {
            p.fresh_body = false;
        }

        let dt = self.dt;
        for k in 0..self.notifies.len() {
            match self.notifies[k] {
                Actor::Ball => {
                    if let Some(bl) = self.ball.as_mut() {
                        bl.pre_tick(&mut self.world.cw.bodies[bl.body.0 as usize]);
                    }
                }
                Actor::Car(i) => {
                    let p = &mut self.players[i];
                    p.contacts.clear();
                    if let Some(car) = p.car.as_mut() {
                        car.pre_tick(&mut self.world.cw.bodies[car.body.0 as usize], dt);
                    }
                }
            }
        }

        let physics_time = self.physics_time();
        let held = self.ball_held_2d;
        let Arena { world, players, vehicles, .. } = self;
        let frame_time = physics_time;
        let mut hook = |cp: &mut crate::bt::types::ManifoldPoint, a: &crate::bt::manifold::Wrap, b: &crate::bt::manifold::Wrap| rules::contact_added(cp, a, b, held);
        world.step_hooked(
            dt,
            &mut |w: &mut World, dt: f32| {
                for &i in vehicles.iter() {
                    if let Some(car) = players[i].car.as_mut() {
                        car.vehicle_update(&mut w.cw, dt, frame_time);
                    }
                }
            },
            Some(&mut hook),
        );
        if self.debug_capture {
            self.capture_debug();
        }

        self.notify_walk();

        let mut k = 0;
        while k < self.notifies.len() {
            let a = self.notifies[k];
            match a {
                Actor::Ball => self.ball_post_tick(),
                Actor::Car(i) => self.car_post_tick(i),
            }

            if self.notifies.get(k) == Some(&a) {
                k += 1;
            }
        }
    }

    fn flush_demolished(&mut self) {
        for i in core::mem::take(&mut self.demolish_queue) {
            self.players[i].demolish_pending = false;
            self.destroy_car(i);
        }
    }

    fn capture_debug(&mut self) {
        let cw = &self.world.cw;
        self.debug_last.clear();
        for id in cw.manifolds() {
            let m = cw.manifold(*id);
            let (o0, o1) = (cw.bodies[m.body0.0 as usize].owner, cw.bodies[m.body1.0 as usize].owner);
            if o0 == Owner::World || o1 == Owner::World || m.cached == 0 {
                continue;
            }
            let mut line = format!("{o0:?}-{o1:?} n{}", m.cached);
            for k in 0..m.cached as usize {
                let q = &m.points[k];
                line += &format!(" [d {:.5} life {} imp {:.4} lat {:.4} {:.4} n {:.4} {:.4} {:.4} fric {:.3} rest {:.3} a {:.3} {:.3} {:.3}]", q.distance, q.life_time, q.applied_impulse, q.applied_impulse_lat1, q.applied_impulse_lat2, q.normal_b.x, q.normal_b.y, q.normal_b.z, q.combined_friction, q.combined_restitution, q.local_a.x, q.local_a.y, q.local_a.z);
            }
            self.debug_last.push(line);
        }
        for (i, p) in self.players.iter().enumerate() {
            if let Some(c) = p.car.as_ref() {
                let g: Vec<String> = c.vehicle.wheels.iter().map(|w| w.ground.map_or("-".to_string(), |b| format!("{:?}", cw.bodies[b.0 as usize].owner))).collect();
                self.debug_last.push(format!("car {i} wheels {g:?}"));
            }
        }
    }

    fn notify_walk(&mut self) {
        let Arena { world, ball, players, ball_on_floor, .. } = self;
        let cw = &world.cw;
        *ball_on_floor = false;
        let mut pts: Vec<NotifyPoint> = Vec::with_capacity(4);
        for id in cw.manifolds() {
            let m = cw.manifold(*id);
            let (b0, b1) = (&cw.bodies[m.body0.0 as usize], &cw.bodies[m.body1.0 as usize]);
            let Some((force, fric)) = rules::notify_points(m, b0, b1, &mut pts) else { continue };
            let (o0, o1) = (b0.owner, b1.owner);

            for (me, other) in [(b0, b1), (b1, b0)] {
                if me.owner == Owner::Ball && other.owner == Owner::World
                    && matches!(cw.shapes[other.shape.0 as usize], Shape::Plane(_)) && other.transform.basis.r[2].z > 0.99 {
                    *ball_on_floor = true;
                }
            }
            let (k0, k1) = (actor_key(o0), actor_key(o1));
            append_to(ball, players, o0, o1, k1, &pts, force, fric);
            for p in pts.iter_mut() {
                core::mem::swap(&mut p.vel0, &mut p.vel1);
                p.normal = -p.normal;
            }
            append_to(ball, players, o1, o0, k0, &pts, force, fric);
        }
    }

    fn ball_post_tick(&mut self) {
        let Some(bl) = self.ball.as_mut() else { return };
        let b = &self.world.cw.bodies[bl.body.0 as usize];
        let was = bl.post_read(b);
        bl.world_contact = rules::contact_average(&mut bl.contacts, |o| o == Owner::World);
        let n = bl.contacts.len();
        for r in 0..n {
            let rec = self.ball.as_ref().unwrap().contacts[r];

            if rec.other == Owner::World {
                self.ball_world_nz = self.ball_world_nz.max(rec.normal.z);
            }
            if rec.other == Owner::World && rec.normal.z >= GROUND_TOLERANCE_Z && self.on_ball_ground(rec) {
                return;
            }
        }
        let dt = self.dt;
        if let Some(bl) = self.ball.as_mut() {
            let loc = bl.state.location;
            let hit = ball::goals().iter().find(|g| g.fully_inside(loc, ball::SCRIPT_RADIUS)).copied();
            if let Some(g) = hit {
                self.on_goal(g.team_scoring, loc);
            } else {
                bl.grounded_clock(was, dt);
            }
        }
    }

    fn car_post_tick(&mut self, i: usize) {
        let Some(car) = self.players[i].car.as_mut() else { return };
        car.post_tick(&mut self.world.cw.bodies[car.body.0 as usize]);
        let wc = rules::contact_average(&mut self.players[i].contacts, |o| o == Owner::World);
        if let Some(car) = self.players[i].car.as_mut() {
            car.world_contact = super::car::WorldContact { has_contact: wc.has, location: wc.loc, velocity: wc.vel, normal: wc.normal };
        }
        let n = self.players[i].contacts.len();
        for r in 0..n {
            let rec = self.players[i].contacts[r];
            if self.players[i].car.is_none() {
                break;
            }
            match rec.other {
                Owner::Ball => self.car_on_hit_ball(i, &rec),
                Owner::Car(j) => self.car_impact(i, j as usize, &rec),
                Owner::World => {}
            }
        }
        self.wheel_contacts_changed(i);
    }

    fn wheel_contacts_changed(&mut self, i: usize) {
        let now = self.physics_time();
        for w in 0..4 {
            let Some(car) = self.players[i].car.as_mut() else { return };
            let c = car.wheels[w].contact;
            if car.wheels[w].contact_notified == c.has_contact {
                continue;
            }
            car.wheels[w].contact_notified = c.has_contact;
            let p = &mut self.players[i];
            if now - p.last_wheels_hit_ball_time > 0.15 && c.actor.is_some_and(|b| self.world.cw.bodies[b.0 as usize].owner == Owner::Ball) {

                if self.phase == Phase::Active {
                    p.events.touched_ball = true;
                    self.clock.ball_has_been_hit = true;
                }
                p.last_wheels_hit_ball_time = now;
            }
        }
    }

    fn held(car: &Car) -> Held {
        let s = car.old_state;
        Held { quat: s.quat, loc: s.location, vel: s.velocity, ang: s.spin }
    }

    fn car_on_hit_ball(&mut self, i: usize, _rec: &RbCollision) {
        let frame = self.physics_frame;
        let Some(bl) = self.ball.as_ref() else { return };
        let (ball_old, ball_body) = (bl.old_state, bl.body);
        let ball_held = Held { quat: ball_old.quat, loc: ball_old.location, vel: ball_old.velocity, ang: ball_old.spin };
        let p = &mut self.players[i];
        let car_held = Self::held(p.car.as_ref().unwrap());
        if frame - p.last_ball_impact_frame > 1 {
            p.last_ball_impact_frame = frame;
            if let Some(imp) = rules::ball_impact(&car_held, &ball_held) {
                rules::rb_add_force(&mut self.world.cw.bodies[ball_body.0 as usize], imp, ForceMode::VelocityChange);
            }
        }
        let p = &mut self.players[i];
        if frame - p.last_ball_touch_frame > 1 {
            p.events.touched_ball = true;

            if self.phase == Phase::Active {
                self.clock.ball_has_been_hit = true;
            }
        }
        self.players[i].last_ball_touch_frame = frame;
    }

    fn car_impact(&mut self, i: usize, j: usize, rec: &RbCollision) {
        if self.players[j].car.is_none() {
            return;
        }
        let (att, vic) = (Self::held(self.players[i].car.as_ref().unwrap()), Self::held(self.players[j].car.as_ref().unwrap()));
        let rules::CarImpactGate::Go { speed, .. } = rules::car_impact_gate(&att, &vic) else { return };
        if self.should_demolish(i, j, rec) {
            self.demolish(i, j);
            return;
        }
        let now = self.physics_time();
        let p = &self.players[i];
        if p.last_hit_car != Some(j) || now > p.last_hit_time + rules::BUMP_INTERVAL {

            let im = rules::impact_from_old(&att, &vic, rec.loc, rec.normal);
            let in_cone = rules::car_within_cone(p.local_offset, self.players[j].local_offset, &im, rules::BUMP_YAW, rules::BUMP_PITCH, false);
            let on_ground = self.players[j].car.as_ref().unwrap().is_on_ground();
            let up = rotator_to_matrix(rules::quat_rotator(vic.quat)).r[2];
            let imp = rules::bump_impulse(speed, att.vel, in_cone, on_ground, up);
            let vb = self.players[j].car.as_ref().unwrap().body;
            rules::add_impulse(&mut self.world.cw.bodies[vb.0 as usize], imp, Vec3::ZERO, false);
            let p = &mut self.players[i];
            p.last_hit_car = Some(j);
            p.last_hit_time = now;
            p.events.bump_inflicted = true;
            self.players[j].events.bumped = true;
        }
    }

    fn should_demolish(&mut self, i: usize, j: usize, rec: &RbCollision) -> bool {
        if self.players[i].team == self.players[j].team {

            return false;
        }
        let car = self.players[i].car.as_ref().unwrap();
        if !car.supersonic {
            return false;
        }
        let att = Self::held(car);
        let vic = Self::held(self.players[j].car.as_ref().unwrap());
        let forward = rotator_to_matrix(rules::quat_rotator(att.quat)).r[0];
        let fs = rules::dot(att.vel, forward);
        if fs.abs() < rules::DEMO_SPEED_REQUIRED {
            return false;
        }
        let mut im = self.time_of_impact(i, j);
        if !(im.fraction < 1.0) {
            im = rules::impact_from_old(&att, &vic, rec.loc, rec.normal);
        }
        let reverse = fs < 0.0;
        let (oi, oj) = (self.players[i].local_offset, self.players[j].local_offset);
        rules::car_within_cone(oi, oj, &im, rules::DEMO_YAW, rules::DEMO_PITCH, reverse)
    }

    fn time_of_impact(&self, i: usize, j: usize) -> Impact {
        let (ci, cj) = (self.players[i].car.as_ref().unwrap(), self.players[j].car.as_ref().unwrap());
        let (si, ei) = sweep_ends(&ci.old_state, self.dt);
        let (sj, ej) = sweep_ends(&cj.old_state, self.dt);
        let shapes = &self.world.cw.shapes;
        let (bi, bj) = (&self.world.cw.bodies[ci.body.0 as usize], &self.world.cw.bodies[cj.body.0 as usize]);
        let mut best = None;
        rules::cast_shapes(&shapes[bi.shape.0 as usize], &shapes[bj.shape.0 as usize], &si, &ei, &sj, &ej, rules::TOI_ALLOWED_PENETRATION, &mut best);
        let mut im = Impact { fraction: 1.0, ..Impact::default() };
        if let Some((f, n, point)) = best {
            if f < 1.0 {
                let (qsi, qei) = (rb_state::matrix_to_quat(&si.basis), rb_state::matrix_to_quat(&ei.basis));
                let (qsj, qej) = (rb_state::matrix_to_quat(&sj.basis), rb_state::matrix_to_quat(&ej.basis));
                let (li, ri) = rules::toi_blend(si.origin, ei.origin, qsi, qei, f);
                let (lj, rj) = rules::toi_blend(sj.origin, ej.origin, qsj, qej, f);
                im = Impact { fraction: f, location: li, rotation: ri, other_location: lj, other_rotation: rj, impact_location: point * 50.0, impact_normal: rules::normal(n) };
            }
        }
        im
    }

    fn demolish(&mut self, attacker: usize, victim: usize) {
        if self.players[victim].demolished {
            return;
        }
        self.players[attacker].events.demo_inflicted = true;
        self.players[victim].events.demoed = true;

        let p = &mut self.players[victim];
        p.demolished = true;
        p.demolish_pending = true;
        p.respawn_left = DEMO_RESPAWN_TIME;
        self.seq += 1;
        self.players[victim].ticket_seq = self.seq;
        self.demolish_queue.push(victim);
        if let Some(m) = &mut self.frame_rng { m.demolished(); }
    }

    fn pad_box(&self, i: usize) -> [f32; 6] {
        let mut b = super::pads::phys_box(self.players[i].preset.name);
        let body = self.players[i].body;
        b[3] += body.mesh_x;
        b[5] += f32::from_bits(body.mesh_z);
        b
    }

    fn destroy_car(&mut self, i: usize) {
        let Some(car) = self.players[i].car.take() else { return };
        if let Some(m) = &mut self.frame_rng { m.car_removed(i); }
        self.world.remove_rigid_body(car.body);
        self.world.cw.free_body(car.body);
        if let Some(at) = self.vehicles.iter().position(|v| *v == i) {
            self.vehicles.swap_remove(at);
        }
        if let Some(at) = self.notifies.iter().position(|a| *a == Actor::Car(i)) {
            self.notifies.remove(at);
        }
        self.players[i].contacts.clear();
        self.players[i].touching = 0;
    }

    fn destroy_ball(&mut self) {
        let Some(bl) = self.ball.take() else { return };
        self.world.remove_rigid_body(bl.body);
        self.world.cw.free_body(bl.body);
        if let Some(at) = self.notifies.iter().position(|a| *a == Actor::Ball) {
            self.notifies.remove(at);
        }
    }

    fn on_ball_ground(&mut self, _rec: RbCollision) -> bool {
        if self.clock.waiting_end_round && self.phase == Phase::Active {
            self.start_new_round();
            return true;
        }
        false
    }

    fn on_goal(&mut self, team: u8, loc: Vec3) {
        if self.phase != Phase::Active {
            return;
        }
        self.clock.score[team as usize] += 1;
        self.goal_scored_this_tick = Some(team);
        for p in self.players.iter_mut() {
            if p.team == team {
                p.events.goal_for_team = true;
            }
        }
        let origin = self.ball.as_ref().map_or(loc, |b| b.actor_loc);
        if let Some(m) = &mut self.frame_rng {
            m.goal();
        }
        let mut ex = Explosion::new(origin, orientation_for_goal(loc.y));
        ex.ticks = -1;
        self.explosion = Some(ex);
        self.destroy_ball_body_for_explosion();
        self.phase = Phase::PostGoal { left: self.config.post_goal_time };
        self.post_goal_timer = FrameTimer::new(self.config.post_goal_time);
    }

    fn destroy_ball_body_for_explosion(&mut self) {
        self.destroy_ball();
    }

    pub fn script_tick(&mut self, dt: f32) {

        self.flush_demolished();

        let mut order: Vec<usize> = (0..self.players.len()).map(|k| self.players_order.get(k).copied().unwrap_or(k)).collect();
        order.sort_by_key(|&i| self.players[i].spawn_seq);
        for i in order {
            if self.players[i].pending_spawn.is_some() {
                if self.players[i].birth_delay > 0 {
                    continue;
                }
                self.init_car_body(i);
            }
        }
        for p in self.players.iter_mut() {
            if let Some(car) = p.car.as_mut() {
                car.frame_tick();
            }
        }

        if pad_tick_post() {
            if self.pad_rule_on {
                self.pads_touching();
            }
            self.pads.frame_tick(dt);
        }

        if let Some(mut ex) = self.explosion.take() {
            self.scratch_hits.clear();
            if ex.ticks < 0 {
                ex.ticks = 0;
            } else if legacy_explosion() {
                let cars: Vec<(u32, Vec3)> = self.players.iter().enumerate().filter_map(|(i, p)| p.car.as_ref().map(|_| (i as u32, p.actor_loc))).collect();
                ex.frame_tick(dt, &cars, &mut self.scratch_hits);
            } else {
                use super::explosion::{car_mesh_bounds, ExplosionPrimitive};
                let mut prims = Vec::with_capacity(self.players.len() * 2);
                for (i, p) in self.players.iter().enumerate() {
                    if p.car.is_none() || p.demolished { continue; }

                    let off = [p.body.forward, 0.0, f32::from_bits(p.body.mesh_z)];
                    let m = car_mesh_bounds(p.body.mesh_bounds(), off, p.actor_loc, p.actor_rot, 1.0);
                    for (o, e) in [(p.actor_loc, v3(48.0, 48.0, 32.0)), (v3(m[0], m[1], m[2]), v3(m[3], m[4], m[5]))] {
                        prims.push(ExplosionPrimitive { actor: i as u32, actor_location: p.actor_loc, bounds_origin: o, bounds_extent: e,
                            eligible: true, processable: true, physics_is_rigid_body: true });
                    }
                }
                if explosion_debug() {
                    let r = ex.life * super::explosion::SPEED;
                    for q in prims.iter() {
                        let (o, e) = (q.bounds_origin, q.bounds_extent);
                        let mut d2 = 0.0f32;
                        for (c, lo, hi) in [(ex.location.x, o.x - e.x, o.x + e.x), (ex.location.y, o.y - e.y, o.y + e.y), (ex.location.z, o.z - e.z, o.z + e.z)] {
                            if c < lo { d2 += (c - lo) * (c - lo); } else if c > hi { d2 += (c - hi) * (c - hi); }
                        }
                        self.debug_last.push(format!("ex t{} r {:.4} car {} dist {:.4} hit {}", ex.ticks, r, q.actor, d2.sqrt(), ex.hit.contains(&q.actor)));
                    }
                }
                ex.frame_tick_bounds(dt, &prims, &mut self.scratch_hits);
            }
            for h in self.scratch_hits.clone() {
                if let Some(c) = self.players[h.car as usize].car.as_ref() {
                    apply_hit(&mut self.world.cw.bodies[c.body.0 as usize], &h);
                }
            }
            self.explosion = Some(ex);
        }

        if dt >= 0.0003 {
            self.move_actors();
        }
    }

    fn tick_spawn_tickets(&mut self, dt: f32) {
        let mut order: Vec<usize> = (0..self.players.len()).collect();
        order.sort_by_key(|&i| self.players[i].ticket_seq);
        for i in order {
            if self.players[i].demolished {

                self.players[i].respawn_left -= dt;
                if self.players[i].respawn_left <= 0.0 && self.phase == Phase::Active && (!self.scripted || self.rule_respawns) {
                    self.respawn_demolished(i);
                }
            }
        }
    }

    fn active_tick(&mut self, dt: f32) {
        if !self.clock.ball_has_been_hit {
            self.clock.hit_delay_left -= dt;
            if self.clock.hit_delay_left <= 0.0 {
                self.clock.ball_has_been_hit = true;
            }
            return;
        }
        let up = self.config.match_seconds.is_none() || self.clock.overtime;
        if up || 0.0 < self.clock.game_time {
            let rate = if up { 1.0 } else { -1.0 };
            self.clock.game_time += dt * rate;
            if !up && self.clock.game_time <= 0.0 {
                self.clock.game_time = 0.0;
                self.clock.waiting_end_round = true;
            }
        }
    }

    fn start_replay(&mut self) {
        self.explosion = None;
        self.destroy_ball();
        for i in 0..self.players.len() {
            self.destroy_car(i);
        }
        self.apply_next_handles();
        self.start_new_round();
        let cars = self.players.len() as u32;
        if let (Some(m), Some(state)) = (&mut self.frame_rng, &mut self.crt_state) {
            m.kickoff_reset(state, cars);
            let order: Vec<usize> = (0..self.players.len()).collect();
            m.cars_restarted(&order);
        }
    }

    pub fn start_new_round(&mut self) {
        let has_winner = self.clock.score[0] != self.clock.score[1];
        if self.clock.waiting_end_round || (!self.clock.overtime && self.config.match_seconds.is_some() && self.clock.game_time <= 0.0) {
            if has_winner {
                self.phase = Phase::Ended;
                return;
            }
            self.clock.overtime = true;
            self.clock.waiting_end_round = false;
            self.clock.game_time = 0.0;
        }

        self.phase = Phase::Countdown { left: self.config.countdown_ticks };
        self.countdown_timer = FrameTimer::new(if self.config.countdown_ticks <= 2 { 0.01 } else { self.config.countdown_ticks as f32 / PHYSICS_RATE });
        if let Some(c) = self.countdown_clock.as_mut() { c.begin(); }
        for p in self.players.iter_mut() {
            if let Some(c) = p.car.as_mut() {
                c.set_driving(false);
            }
        }
        self.reset_ball();
        self.reset_players();
        self.pads.reset();
        self.clock.ball_has_been_hit = false;
    }

    pub fn reset_ball(&mut self) {
        self.destroy_ball();
        let shape = self.ball_shape;
        let inertia = self.world.cw.shapes[shape.0 as usize].local_inertia(ball::MASS_PROP);
        let b = ball::body(shape, inertia, v3(0.0, 0.0, ball::SPAWN_Z_BT), Mat3::IDENTITY);
        let id = self.world.cw.insert_body(b);
        self.world.add_rigid_body(id, config::BALL_GROUP, config::BALL_MASK);
        let mut bl = Ball::new(id, ball::read_limits(config::PHYSICS_CONFIG.quantize_physics));
        bl.init_state(&self.world.cw.bodies[id.0 as usize]);
        self.ball = Some(bl);
        self.add_notify(Actor::Ball);
        self.ball_held_2d = true;
    }

    pub fn reset_players(&mut self) {
        for i in 0..self.players.len() {
            self.destroy_car(i);
        }
        let queued = if self.spawn_queue.is_empty() { None } else { Some(self.spawn_queue.remove(0)) };
        if queued.is_none() && self.selector.is_some() {

            let mut taken: Vec<(Vec3, Rotator, &'static config::BodySpawn)> = Vec::new();
            for k in 0..self.players.len() {
                let i = self.players_order.get(k).copied().unwrap_or(k);
                let team = self.players[i].team as usize;
                let sel = self.selector.as_mut().unwrap();
                let pick = sel.choose(team, false, |sp| spot_blocked(sp.location, &taken), |_| None);
                let Some(sp) = pick else { self.selector_failures += 1; continue };
                taken.push((sp.location, sp.rotation, self.players[i].body));
                self.spawn_car_actor(i, sp.location, sp.rotation);
            }
            return;
        }
        let n_blue = self.players.iter().filter(|p| p.team == 0).count();
        let slots = self.random_slots(n_blue.max(self.players.len() - n_blue));
        let mut seen = [0usize; 2];
        for i in 0..self.players.len() {
            let team = self.players[i].team as usize;
            let spot = match &queued {
                Some(q) if i < q.len() => q[i],
                _ => {
                    let s = KICKOFF_SPOTS[slots[seen[team]]];
                    seen[team] += 1;
                    mirror_spot(s, team)
                }
            };
            self.spawn_car_actor(i, spot.0, spot.1);
        }
    }

    fn random_slots(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..KICKOFF_SPOTS.len()).collect();
        for k in (1..v.len()).rev() {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            v.swap(k, (self.rng % (k as u64 + 1)) as usize);
        }
        v.truncate(n);
        v
    }

    fn spawn_car_actor(&mut self, i: usize, loc: Vec3, rot: Rotator) {
        self.players[i].pending_spawn = Some((loc, rot));
        self.seq += 1;
        self.players[i].spawn_seq = self.seq;
        self.players[i].spawn_spot = true;
        self.players[i].demolished = false;
        self.players[i].actor_loc = loc;
    }

    fn init_car_body(&mut self, i: usize) {
        let Some((loc, rot)) = self.players[i].pending_spawn.take() else { return };
        self.players[i].spawn_loc = loc;

        self.destroy_car(i);
        let preset = self.players[i].preset;

        let half = preset.half_extents();
        let off = preset.offset();
        let mut comp = CompoundShape::new();
        comp.add_child(Transform { basis: Mat3::IDENTITY, origin: off }, Shape::Box(BoxShape::new(half)));
        let shape = self.world.cw.add_shape(Shape::Compound(comp));
        let inertia = self.world.cw.shapes[shape.0 as usize].local_inertia(config::CAR_MASS);
        let basis = spawn_basis(rot);

        let origin = if self.players[i].spawn_spot {
            let o = config::spawn_body_origin(loc, rot.yaw, self.players[i].team, self.players[i].body);
            v3(o.x, o.y, o.z + self.players[i].spawn_dz * 0.02)
        } else {
            v3(loc.x * 0.02, loc.y * 0.02, SPAWN_BODY_Z_BT + self.players[i].spawn_dz * 0.02)
        };
        let info = config::car_rigid_info(Transform { basis, origin }, shape, inertia);
        let mut b = crate::bt::Body::new_rigid(&info);
        config::finish_car_body(&mut b, i as u32);
        let inv_mass = b.inv_mass;
        let id = self.world.cw.insert_body(b);
        self.world.add_rigid_body(id, config::CAR_GROUP, config::CAR_MASK);
        let mut car = new_car(preset, id, inv_mass);
        let s = rb_state::read_body(&self.world.cw.bodies[id.0 as usize], &car.limits);
        car.state = s;
        car.old_state = s;
        car.boost.set_amount(car.boost.start_amount);
        car.input.jump = self.players[i].input.jump;

        if self.phase == Phase::Active && car.input.jump {
            car.jumped = true;
            car.double_jumped = true;
        }

        car.driving = self.phase == Phase::Active;
        car.game_active = true;
        car.gravity_z = GRAVITY_UU;
        let p = &mut self.players[i];
        p.car = Some(car);
        p.contacts.clear();
        p.last_ball_impact_frame = 0;
        p.last_ball_touch_frame = 0;
        p.last_hit_car = None;
        p.last_hit_time = 0.0;
        p.fresh_body = true;
        self.vehicles.push(i);
        self.add_notify(Actor::Car(i));
    }

    fn add_notify(&mut self, a: Actor) {
        if let Some(at) = self.notifies.iter().position(|x| *x == a) {
            self.notifies.remove(at);
        }
        self.notifies.push(a);
    }

    fn start_round(&mut self) {
        if matches!(self.phase, Phase::Countdown { .. }) {
            if let Some(c) = self.countdown_clock.as_mut() { c.end(); }

            if let Some(sel) = self.selector.as_mut() { sel.end_countdown(); }
        }
        self.clock.round_num += 1;
        self.clock.ball_has_been_hit = false;
        self.clock.hit_delay_left = 5.0;
        self.ball_held_2d = false;
        self.phase = Phase::Active;
        for p in self.players.iter_mut() {
            if let Some(c) = p.car.as_mut() {
                c.set_driving(true);
            }
        }
    }

    fn respawn_demolished(&mut self, i: usize) {
        let team = self.players[i].team as usize;
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let mut queued = self.respawn_queue.get_mut(i).filter(|q| !q.is_empty()).map(|q| q.remove(0));
        if queued.is_none() && self.selector.is_some() {
            if let Some(state) = self.crt_state {

                let others: Vec<(Vec3, Rotator, &'static config::BodySpawn)> = self.players.iter().enumerate().filter(|(j, p)| *j != i && !p.demolished)
                    .filter_map(|(_, p)| {
                        let at = if let Some(c) = p.car.as_ref() { Some((p.actor_loc, rules::quat_rotator(c.state.quat))) } else { p.pending_spawn };
                        at.map(|(l, r)| (l, r, p.body))
                    }).collect();
                let sel = self.selector.as_mut().unwrap();
                sel.rng = CrtRand { state };
                let pick = sel.choose(team, true, |sp| spot_blocked(sp.location, &others), |_| None);
                self.crt_state = Some(sel.rng.state);
                queued = pick.map(|sp| (sp.location, sp.rotation));
            }
            if queued.is_none() { self.selector_failures += 1; }
        }
        let spot = queued.unwrap_or_else(|| mirror_spot(RESPAWN_SPOTS[(self.rng % 4) as usize], team));
        self.spawn_car_actor(i, spot.0, spot.1);
        self.players[i].respawn_left = 0.0;
        if let Some(m) = &mut self.frame_rng { m.car_respawned(i); }
    }

    pub fn round_gate(&mut self, started: bool, steps: i32) {
        self.forced_round_start = false;
        if let Phase::Countdown { left } = self.phase {
            if started {
                self.forced_round_start = true;
            } else if left <= steps {
                self.phase = Phase::Countdown { left: steps + 1 };
            }
        }
    }

    pub fn send_input(&mut self, i: usize, mut input: VehicleInputs) {
        if let Some(c) = self.players[i].car.as_mut() {
            c.push_input(&mut input);
        }
    }

    pub fn force_start_replay(&mut self) {
        self.forced_restart = true;
    }

    pub fn force_respawn(&mut self, i: usize, loc: Vec3, rot: Rotator) {
        self.spawn_car_actor(i, loc, rot);
        self.players[i].respawn_left = 0.0;
        if let Some(m) = &mut self.frame_rng { m.car_respawned(i); }
    }

    pub fn force_pickup(&mut self, pad: usize, player: usize) {
        let amount = self.pads.pick_up(pad);
        if let Some(c) = self.players[player].car.as_mut() {
            let v = c.boost.amount + amount;
            c.boost.set_amount(v);
        }
        self.players[player].events.picked_pad = pad as i32;
    }

    fn pads_touching(&mut self) {
        for pad in 0..NUM_PADS {
            let fresh = core::mem::take(&mut self.pads.pads[pad].respawned) && pad_respawn_mesh();
            if !self.pads.can_pick_up(pad) {

                for p in self.players.iter_mut() {
                    p.touching &= !(1u64 << pad);
                }
                continue;
            }

            let n_notifies = self.notifies.len();
            for kk in 0..n_notifies {
                let k = if fresh { n_notifies - 1 - kk } else { kk };
                let Actor::Car(i) = self.notifies[k] else { continue };

                if fresh && self.players[i].car.is_some()
                    && Pads::mesh_overlaps_box_with_layout(self.pads.layout, pad, self.players[i].actor_loc, self.players[i].actor_rot, self.pad_box(i))
                {
                    self.players[i].touching |= 1u64 << pad;
                }
                if self.players[i].touching >> pad & 1 != 0 && self.car_can_pick_up(i) {
                    self.force_pickup(pad, i);
                    if self.frame_rng.is_some() {
                        self.frame_rng_pickups_now += 1;
                    }
                    break;
                }
            }
        }
        self.update_pad_ticks();
    }

    fn update_pad_ticks(&mut self) {
        let touching = self.players.iter().fold(0u64, |bits, p| bits | p.touching);
        for (i, pad) in self.pads.pads.iter_mut().enumerate() {
            pad.tick_enabled = pad.picked_up || touching >> i & 1 != 0;
        }
    }

    fn car_can_pick_up(&self, i: usize) -> bool {
        let Some(c) = self.players[i].car.as_ref() else { return false };
        c.driving && !c.boost.no_boost && c.boost.amount < c.boost.max_amount
    }

    fn move_actors(&mut self) {
        let a = self.render_alpha;
        let lerp = |o: Vec3, s: Vec3| v3((s.x - o.x) * a + o.x, (s.y - o.y) * a + o.y, (s.z - o.z) * a + o.z);
        if let Some(bl) = self.ball.as_mut() {
            bl.actor_loc = lerp(bl.old_state.location, bl.state.location);
        }
        for k in 0..self.notifies.len() {
            let Actor::Car(i) = self.notifies[k] else { continue };
            let Some(c) = self.players[i].car.as_ref() else { continue };
            let new = lerp(c.old_state.location, c.state.location);
            let rot = rules::quat_rotator(rules::slerp(c.old_state.quat, c.state.quat, a));
            let prev = self.players[i].actor_loc;
            self.players[i].actor_loc = new;
            self.players[i].actor_rot = rot;
            if !self.pad_rule_on {
                continue;
            }
            for pad in 0..NUM_PADS {
                let bit = 1u64 << pad;
                if self.players[i].touching & bit == 0 && Pads::sweep_with_layout(self.pads.layout, pad, prev, new) {
                    self.players[i].touching |= bit;
                    self.pads.pads[pad].tick_enabled = true;
                    if self.pads.can_pick_up(pad) && self.car_can_pick_up(i) {
                        self.force_pickup(pad, i);
                        if !pad_tick_post() {

                            self.pads.pads[pad].skip = (!self.pads.pads[pad].ticked_this_frame) as u8;
                        }
                    }
                }

                if self.players[i].touching & bit != 0 && !Pads::mesh_overlaps_box_with_layout(
                    self.pads.layout, pad, new, rot, self.pad_box(i)) {
                    self.players[i].touching &= !bit;
                }
            }
        }
    }

    pub fn set_car_state(&mut self, i: usize, s: &RbState, boost: Option<f32>) {
        if let Some(c) = self.players[i].car.as_mut() {
            let st = with_sleep_flag(s);
            rb_state::write_body(&mut self.world.cw.bodies[c.body.0 as usize], &st);
            c.state = st;
            if let Some(bv) = boost {
                if 0.0 <= bv {
                    let whole = (bv as i32).min(255) as u8;
                    c.boost.set_amount((whole as f32 / 100.0) * c.boost.max_amount);
                }
            }
        }
    }

    pub fn set_ball_state(&mut self, s: &RbState, sleep_at_rest: bool) {
        if let Some(bl) = self.ball.as_mut() {
            let mut st = with_sleep_flag(s);
            if !sleep_at_rest {
                st.sleeping = false;
            }
            rb_state::write_body(&mut self.world.cw.bodies[bl.body.0 as usize], &st);
            bl.state = st;
        }
    }

    pub fn set_ball_at_rest(&mut self, asleep: bool) {
        let s = RbState { quat: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }, location: v3(0.0, 0.0, 92.75), velocity: Vec3::ZERO, spin: Vec3::ZERO, sleeping: asleep };
        if let Some(bl) = self.ball.as_mut() {
            rb_state::write_body(&mut self.world.cw.bodies[bl.body.0 as usize], &s);
            bl.state = s;
            bl.old_state = s;
            bl.first_frame = false;
        }
    }

    pub fn set_car_fresh(&mut self, i: usize, s: &RbState, boost: f32) {
        let team_rot = Rotator::default();
        self.players[i].pending_spawn = Some((s.location, team_rot));
        self.players[i].spawn_spot = false;
        self.players[i].demolished = false;
        self.players[i].demolish_pending = false;
        self.init_car_body(i);
        self.players[i].fresh_body = false;
        if let Some(c) = self.players[i].car.as_mut() {
            let st = with_sleep_flag(s);
            rb_state::write_body(&mut self.world.cw.bodies[c.body.0 as usize], &st);
            c.state = st;
            c.old_state = st;
            c.boost.set_amount(boost);
            c.driving = true;
        }
    }
}

fn with_sleep_flag(s: &RbState) -> RbState {
    let mut st = *s;
    st.sleeping = st.velocity.x == 0.0 && st.velocity.y == 0.0 && st.velocity.z == 0.0 && st.spin.x == 0.0 && st.spin.y == 0.0 && st.spin.z == 0.0;
    st
}

pub fn state_set_rotation(pitch: f32, yaw: f32, roll: f32) -> Quat {
    let (p, y, r) = (pitch * 57.295776, yaw * 57.295776, roll * 57.295776);
    let rot = Rotator { pitch: (p * 182.04445) as i32, yaw: (y * 182.04445) as i32, roll: (r * 182.04445) as i32 };
    fmatrix_to_quat(&rotator_to_matrix(rot))
}

fn actor_key(o: Owner) -> u32 {
    match o {
        Owner::World => 0,
        Owner::Ball => 1,
        Owner::Car(i) => 2 + i,
    }
}

fn append_to(ball: &mut Option<Ball>, players: &mut [Player], me: Owner, other: Owner, key: u32, pts: &[NotifyPoint], force: Vec3, fric: Vec3) {
    match me {
        Owner::Ball => {
            if let Some(b) = ball.as_mut() {
                rules::contact_append(&mut b.contacts, other, key, pts, force, fric);
            }
        }
        Owner::Car(i) => rules::contact_append(&mut players[i as usize].contacts, other, key, pts, force, fric),
        Owner::World => {}
    }
}

fn spot_blocked(spot: Vec3, others: &[(Vec3, Rotator, &'static config::BodySpawn)]) -> bool {
    use super::explosion::{car_mesh_bounds, sphere_overlaps_bounds};
    others.iter().any(|(loc, rot, b)| {
        let m = car_mesh_bounds(b.mesh_bounds(), [b.forward, 0.0, f32::from_bits(b.mesh_z)], *loc, *rot, 1.0);
        sphere_overlaps_bounds(spot, 48.0 * 48.0, *loc, v3(48.0, 48.0, 32.0))
            || sphere_overlaps_bounds(spot, 48.0 * 48.0, v3(m[0], m[1], m[2]), v3(m[3], m[4], m[5]))
    })
}

fn mirror_spot(s: (f32, f32, i32), team: usize) -> (Vec3, Rotator) {
    if team == 0 {
        (v3(s.0, s.1, SPAWN_ACTOR_Z), Rotator { pitch: 0, yaw: s.2, roll: 0 })
    } else {
        (v3(-s.0, -s.1, SPAWN_ACTOR_Z), Rotator { pitch: 0, yaw: wrap16(s.2 - 32768), roll: 0 })
    }
}

fn wrap16(a: i32) -> i32 {
    let mut a = a % 65536;
    if a > 32767 {
        a -= 65536;
    }
    if a < -32768 {
        a += 65536;
    }
    a
}

pub fn fmatrix_to_quat(m: &Mat3) -> Quat {
    let (m0, m1, m2) = (m.r[0], m.r[1], m.r[2]);
    let small = |r: Vec3| r.x.abs() < 0.0001 && r.y.abs() < 0.0001 && r.z.abs() < 0.0001;
    if small(m0) && small(m1) && small(m2) {
        return Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
    }
    let t = m1.y + m0.x + m2.z;
    if t <= 0.0 {
        let d = [m0.x, m1.y, m2.z];
        let mut i = (m0.x < m1.y) as usize;
        if d[i] <= m2.z && m2.z != d[i] {
            i = 2;
        }
        const NEXT: [usize; 3] = [1, 2, 0];
        let (j, k) = (NEXT[i], NEXT[NEXT[i]]);
        let at = |r: usize, c: usize| m.r[r][c];
        let f = 1.0 / (((d[i] - d[j]) - d[k]) + 1.0).sqrt();
        let h = f * 0.5;
        let mut q = [0.0f32; 4];
        q[i] = 0.5 / f;
        q[3] = (at(j, k) - at(k, j)) * h;
        q[j] = (at(i, j) + at(j, i)) * h;
        q[k] = (at(i, k) + at(k, i)) * h;
        Quat { x: q[0], y: q[1], z: q[2], w: q[3] }
    } else {
        let f = 1.0 / (t + 1.0).sqrt();
        let h = f * 0.5;
        Quat { x: (m1.z - m2.y) * h, y: (m2.x - m0.z) * h, z: (m0.y - m1.x) * h, w: 0.5 / f }
    }
}


/// A car spawned facing one of the six kickoff and respawn yaws starts with these axes (float bits, row by row), as
/// the running game leaves them. They differ from the rotator formula in `spawn_basis` in the last bit and in the
/// signs of zeros, and the first ticks' spin depends on that. Measured in the game, not read from its files: a
/// documented rule (docs/PROVENANCE.md). Any other yaw uses the formula.
const SPAWN_BASE_BITS: [(i32, [[u32; 3]; 3]); 6] = [
    (16384, [[0x00000000, 0xbf7fffff, 0x00000000], [0x3f7fffff, 0x00000000, 0x00000000], [0x00000000, 0x00000000, 0x3f800000]]),
    (-16384, [[0x34000000, 0x3f7fffff, 0x00000000], [0xbf7fffff, 0x34000000, 0x80000000], [0x80000000, 0x00000000, 0x3f800000]]),
    (8192, [[0x3f3504f2, 0xbf3504f4, 0x00000000], [0x3f3504f4, 0x3f3504f2, 0x00000000], [0x00000000, 0x00000000, 0x3f800000]]),
    (24576, [[0xbf3504f6, 0xbf3504f4, 0x00000000], [0x3f3504f4, 0xbf3504f6, 0x00000000], [0x00000000, 0x00000000, 0x3f800000]]),
    (-8192, [[0x3f3504f5, 0x3f3504f2, 0x00000000], [0xbf3504f2, 0x3f3504f5, 0x80000000], [0x80000000, 0x00000000, 0x3f800000]]),
    (-24576, [[0xbf3504f2, 0x3f3504f4, 0x00000000], [0xbf3504f4, 0xbf3504f2, 0x00000000], [0x00000000, 0x00000000, 0x3f800000]]),
];

static SPAWN_BASES: std::sync::LazyLock<Vec<(i32, [Vec3; 3])>> =
    std::sync::LazyLock::new(|| SPAWN_BASE_BITS.iter().map(|(y, b)| (*y, b.map(Vec3::from_bits))).collect());

fn spawn_basis(rot: Rotator) -> Mat3 {
    if rot.pitch == 0 && rot.roll == 0 {
        for (y, b) in SPAWN_BASES.iter() {
            if *y == rot.yaw {
                return Mat3 { r: *b };
            }
        }
    }
    rb_state::quat_to_btmatrix(fmatrix_to_quat(&rotator_to_matrix(rot)))
}

pub fn sweep_ends(s: &RbState, dt: f32) -> (Transform, Transform) {
    let basis = rb_state::quat_to_btmatrix(s.quat);
    let cur = Transform { basis, origin: v3(s.location.x * 0.02, s.location.y * 0.02, s.location.z * 0.02) };
    let back_lin = v3(-s.velocity.x * 0.02, -s.velocity.y * 0.02, -s.velocity.z * 0.02);
    let back = crate::bt::rigid_body::integrate_transform(&cur, back_lin, -s.spin, dt);
    let fwd_lin = v3(s.velocity.x * 0.02, s.velocity.y * 0.02, s.velocity.z * 0.02);
    let fwd = crate::bt::rigid_body::integrate_transform(&cur, fwd_lin, s.spin, dt);
    (back, fwd)
}

#[allow(unused)]
fn quat_identity() -> Quat {
    Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
}
