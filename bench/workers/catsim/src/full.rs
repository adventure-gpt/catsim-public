//! Level A for catSim: the full-state suite's start state (its `s.*` arrays, RocketGPU's field layout) mapped into
//! catSim's arena. docs/BENCHMARKS.md lists every field: what it becomes here, and what can't be set (the contact
//! manifolds' cached points and the broadphase's pair order; catSim makes those at the first tick).

use catsim::bt::{ACTIVE, ISLAND_SLEEPING};
use catsim::game::arena::{Actor, Arena, Phase, SPAWN_ACTOR_Z};
use catsim::game::hidden::{BodyState, CarHidden};
use catsim::game::round::{CountdownClock, FrameTimer};
use catsim::math::rot::Rotator;
use catsim::math::{Mat3, Vec3, v3};
use serde_json::Value;

const SLOTS: usize = 8;
/// catSim's wheel w is the suite's (RocketSim order) wheel SUITE_WHEEL[w]: left and right swapped.
const SUITE_WHEEL: [usize; 4] = [1, 0, 3, 2];

pub struct Full<'a> {
    index: &'a Value,
    data: &'a [u8],
}

impl<'a> Full<'a> {
    pub fn new(index: &'a Value, data: &'a [u8]) -> Self {
        Full { index, data }
    }

    fn raw(&self, name: &str) -> &'a [u8] {
        match self.index.get(name).and_then(Value::as_array) {
            Some(r) => {
                let (o, n) = (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize);
                &self.data[o..o + n]
            }
            None => &[],
        }
    }

    fn slot(&self, name: &str, k: usize) -> &'a [u8] {
        let r = self.raw(name);
        let per = r.len() / SLOTS;
        &r[k * per..(k + 1) * per]
    }

    fn f(&self, name: &str, k: usize) -> Vec<f32> {
        self.slot(name, k).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn f1(&self, name: &str, k: usize) -> f32 {
        self.f(name, k).first().copied().unwrap_or(0.0)
    }

    fn u8(&self, name: &str, k: usize) -> u8 {
        self.slot(name, k).first().copied().unwrap_or(0)
    }

    fn u32(&self, name: &str, k: usize) -> u32 {
        let s = self.slot(name, k);
        if s.len() < 4 { 0 } else { u32::from_le_bytes([s[0], s[1], s[2], s[3]]) }
    }

    fn u64(&self, name: &str, k: usize) -> u64 {
        let s = self.slot(name, k);
        if s.len() < 8 { u64::MAX } else { u64::from_le_bytes(s[..8].try_into().unwrap()) }
    }

    fn whole_f(&self, name: &str) -> Vec<f32> {
        self.raw(name).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn whole_u(&self, name: &str) -> u64 {
        let r = self.raw(name);
        let mut b = [0u8; 8];
        b[..r.len().min(8)].copy_from_slice(&r[..r.len().min(8)]);
        u64::from_le_bytes(b)
    }

    fn vec(v: &[f32], at: usize) -> Vec3 {
        v3(v[at], v[at + 1], v[at + 2])
    }

    /// The Bullet basis by rows from its three columns.
    fn basis(fwd: &[f32], right: &[f32], up: &[f32]) -> Mat3 {
        Mat3 { r: [v3(fwd[0], right[0], up[0]), v3(fwd[1], right[1], up[1]), v3(fwd[2], right[2], up[2])] }
    }

    /// The broadphase handles the start state's bodies hold (ball, then the cars), when the state has them.
    pub fn handles(&self, nc: usize) -> Option<Vec<u16>> {
        let r = self.raw("solver_scratch.bp.sweep.sap_body_handle");
        if r.len() < nc + 1 {
            return None;
        }
        let h: Vec<u16> = r[..nc + 1].iter().map(|&x| x as u16).collect();
        let distinct = h.iter().collect::<std::collections::HashSet<_>>().len() == h.len();
        (h.iter().all(|&x| x != 0) && distinct).then_some(h)
    }

    /// Map the whole start state in. `dt0`: the start tick's frame time (the suite's frame timers already hold it; catSim's
    /// frame adds it again, so the timers go back one frame).
    pub fn apply(&self, a: &mut Arena, req: &Value, dt0: f32) -> Result<(), String> {
        let nc = a.players.len();
        let phys = self.whole_u("env_states.tick") as i64 - 1;
        a.physics_frame = phys as i32;
        let ids: Vec<u64> = req.get("ids").and_then(Value::as_array).map(|v| v.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
        let demoed: Vec<bool> = (0..nc).map(|k| self.u8("car_states.is_demoed", k) != 0).collect();
        for k in 0..nc {
            if demoed[k] {
                continue;
            }
            let body = BodyState {
                origin: Self::vec(&self.f("car_states.pos_bt", k), 0),
                basis: Self::basis(&self.f("car_states.basis_forward", k), &self.f("car_states.basis_right", k), &self.f("car_states.basis_up", k)),
                lin_vel: Self::vec(&self.f("car_states.vel_bt", k), 0),
                ang_vel: Self::vec(&self.f("car_states.internal_ang_vel", k), 0),
                asleep: false,
                deactivation_time: 0.0,
            };
            if a.players[k].car.is_none() {
                let s = catsim::game::rb_state::RbState { location: v3(body.origin.x * 50.0, body.origin.y * 50.0, body.origin.z * 50.0), ..Default::default() };
                a.set_car_fresh(k, &s, 0.0);
            }
            let act = a.players[k].car.as_ref().map(|c| a.world.cw.bodies[c.body.0 as usize].activation_state).unwrap_or(ACTIVE);
            let body = BodyState { asleep: act == ISLAND_SLEEPING, ..body };
            a.set_car_body(k, &body);
            if self.u8("car_states.restored_inv_inertia_valid", k) != 0 {
                let m = self.f("car_states.restored_inv_inertia_bt", k);
                if m.len() == 9 {
                    let c = a.players[k].car.as_ref().unwrap();
                    a.world.cw.bodies[c.body.0 as usize].inv_inertia_world = Mat3 { r: [Self::vec(&m, 0), Self::vec(&m, 3), Self::vec(&m, 6)] };
                }
            }
            let mut h = a.car_hidden(k).ok_or("car has no body")?;
            self.car_hidden(a, k, &mut h, phys, &ids);
            a.set_car_hidden(k, &h);
            a.players[k].last_wheels_hit_ball_time = -1000.0;
        }
        // the ball
        let bp = self.whole_f("ball_states.pos_bt");
        let bv = self.whole_f("ball_states.vel_bt");
        let asleep = self.whole_u("ball_states.sleeping") as u32 != 0;
        let dtime = self.whole_f("ball_states.deactivation_time").first().copied().unwrap_or(0.0);
        // a ball resting on the floor (or moving) keeps its deactivation clock; one parked still off the floor by a state set
        // sleeps whatever its clock says
        let resting = (bp[2] * 50.0 - 92.75).abs() < 0.01;
        let still = bv.iter().all(|&x| x == 0.0) && self.whole_f("ball_states.ang_vel").iter().all(|&x| x == 0.0);
        let dt_used = if resting || !still { dtime } else { 0.0 };
        let sleeping = asleep && dt_used <= 0.0;
        let ball = BodyState {
            origin: Self::vec(&bp, 0),
            basis: Self::basis(&self.whole_f("ball_states.basis_forward"), &self.whole_f("ball_states.basis_right"), &self.whole_f("ball_states.basis_up")),
            lin_vel: Self::vec(&bv, 0),
            ang_vel: Self::vec(&self.whole_f("ball_states.ang_vel"), 0),
            asleep: sleeping,
            deactivation_time: dtime,
        };
        a.set_ball_body(&ball);
        if self.whole_u("ball_states.restored_inv_inertia_valid") as u32 != 0 {
            let m = self.whole_f("ball_states.restored_inv_inertia_bt");
            if m.len() == 9 {
                let b = a.ball.as_ref().unwrap();
                a.world.cw.bodies[b.body.0 as usize].inv_inertia_world = Mat3 { r: [Self::vec(&m, 0), Self::vec(&m, 3), Self::vec(&m, 6)] };
            }
        }
        a.ball_held_2d = (self.whole_u("ball_states.constrained_2d") & 0xffff) != 0;
        // the broadphase boxes as the game's last bounds update left them (its predictive contacts read them first)
        let boxes = self.whole_f("solver_scratch.bp.sweep.sap_body_aabb");
        let valid = self.whole_u("solver_scratch.bp.sweep.sap_body_aabb_valid") & 0xffff;
        for obj in 0..=nc {
            if valid >> obj & 1 == 0 || boxes.len() < (obj + 1) * 6 {
                continue;
            }
            let id = if obj == 0 { a.ball.as_ref().map(|b| b.body) } else if demoed[obj - 1] { None } else { a.players[obj - 1].car.as_ref().map(|c| c.body) };
            if let Some(id) = id {
                let cw = &mut a.world.cw;
                let h = cw.bodies[id.0 as usize].proxy as u16;
                cw.broadphase.set_aabb(h, Self::vec(&boxes, obj * 6), Self::vec(&boxes, obj * 6 + 3), &mut cw.dispatcher);
            }
        }
        for k in 0..nc {
            if demoed[k] {
                a.set_car_demolished(k, self.f1("car_states.demo_respawn_timer", k) - dt0);
            }
        }
        // the visit order: the ball's notify first, then the cars in the game's order
        for call in req.get("calls").and_then(Value::as_array).into_iter().flatten() {
            if call[0].as_str() == Some("set_car_visit_order") {
                let order: Vec<usize> = call[1][0].as_array().into_iter().flatten().filter_map(Value::as_u64)
                    .filter_map(|x| ids.iter().position(|&i| i == x)).filter(|&k| k < nc && !demoed[k]).collect();
                if order.len() == a.vehicles.len() {
                    a.notifies = std::iter::once(Actor::Ball).chain(order.iter().map(|&k| Actor::Car(k))).collect();
                    a.vehicles = order;
                }
            }
        }
        // a countdown in progress: stage 1 the 1 s start timer, stage 2 the update loop ("3 - fires" left)
        let stage = self.whole_u("env_states.countdown_phase") as u8;
        if stage == 1 || stage == 2 {
            let cnt = self.whole_f("env_states.countdown_count").first().copied().unwrap_or(0.0) - dt0;
            let fires = self.whole_u("env_states.countdown_fires") as u8 as i32;
            let mut c = CountdownClock::new(3, 3);
            if stage == 1 {
                c.start_timer = FrameTimer { rate: 1.0, elapsed: cnt, active: true };
            } else {
                c.start_timer = FrameTimer::default();
                c.remaining = 3 - fires;
                c.replicated_remaining = c.remaining;
                c.update_timer = FrameTimer { rate: 1.0, elapsed: cnt, active: true };
            }
            a.countdown_clock = Some(c);
            a.rule_countdown = true;
            a.phase = Phase::Countdown { left: 1 << 20 };
            a.ball_held_2d = true;
            for p in a.players.iter_mut() {
                if let Some(c) = p.car.as_mut() {
                    c.set_driving(false);
                }
            }
        }
        // respawn spots for the demolished cars, from the recording (the suite hands its simulators these choices)
        if let Some(sp) = req["start"].get("respawn_spots").and_then(Value::as_array) {
            for (k, s) in sp.iter().enumerate() {
                if let Some(s) = s.as_array() {
                    let g = |i: usize| s[i].as_f64().unwrap_or(0.0) as f32;
                    a.queue_respawn_spot(k, v3(g(0), g(1), SPAWN_ACTOR_Z), Rotator { pitch: 0, yaw: g(2) as i32, roll: 0 });
                }
            }
        }
        Ok(())
    }

    fn car_hidden(&self, a: &Arena, k: usize, h: &mut CarHidden, phys: i64, ids: &[u64]) {
        let u = |n: &str| self.u8(n, k) != 0;
        let f = |n: &str| self.f1(n, k);
        // the game's own flags as the suite calibrates them
        h.jumped = self.u8("wheel_runtime.cal_game_jumped", k) != 0;
        h.double_jumped = self.u8("wheel_runtime.cal_game_double_jumped", k) != 0;
        h.on_ground = u("car_states.is_on_ground");
        h.supersonic = u("car_states.internal_is_supersonic") || u("car_states.is_supersonic");
        h.supersonic_time = f("car_states.supersonic_time");
        h.boost_amount = f("car_states.boost") / 100.0;
        h.boost.active = u("car_states.is_boosting");
        h.boost.time = if h.boost.active { f("car_states.boosting_time") } else { f("car_states.time_since_boosted") };
        h.jump.active = u("car_states.is_jumping");
        h.jump.time = f("car_states.jump_time");
        h.jump_released = u("car_states.jump_hold_broken");
        // the double jump's clock is not in the state; only the dodge window reads it, as the least of three clocks
        h.double_jump.time = 10.0;
        h.dodge.active = u("car_states.is_flipping");
        h.dodge.time = f("car_states.flip_time");
        let rel = self.f("car_states.flip_rel_torque", k);
        if rel.len() >= 2 {
            let c = a.players[k].car.as_ref().unwrap();
            let d = &c.dodge;
            let round = catsim::math::rot::app_round;
            h.dodge_torque = v3(round(rel[0] * d.side_torque * d.torque_scale) as f32, round(rel[1] * d.forward_torque * d.torque_scale) as f32, 0.0);
            h.dodge_direction = v3(rel[1], -rel[0], 0.0);
        }
        h.auto_flip.active = u("car_states.is_auto_flipping");
        if h.auto_flip.active {
            h.auto_flip_right = f("car_states.auto_flip_torque_scale") > 0.0;
        }
        h.handbrake = f("car_states.handbrake_val");
        let lc = self.slot("car_states.last_controls", k);
        if lc.len() >= 23 {
            for i in 0..5 {
                h.held_input[i] = f32::from_le_bytes(lc[4 * i..4 * i + 4].try_into().unwrap());
            }
            h.held_input[5] = lc[20] as f32;
            h.held_input[6] = lc[21] as f32;
            h.held_input[7] = lc[22] as f32;
        }
        let wc = self.slot("car_states.wheels_with_contact", k);
        let nrm = self.f("car_states.set_wheel_normals", k);
        for w in 0..4 {
            let q = SUITE_WHEEL[w];
            h.wheels[w].contact = wc.get(q).copied().unwrap_or(0) != 0;
            if nrm.len() >= 12 {
                h.wheels[w].normal = Self::vec(&nrm, 3 * q);
            }
        }
        h.world_contact = u("car_states.world_contact.has_contact");
        h.world_contact_normal = Self::vec(&self.f("car_states.world_contact.contact_normal", k), 0);
        let t_imp = self.u64("car_states.ball_hit_info.tick_count_when_extra_impulse_applied", k);
        h.ball_impact_frame = if t_imp < (1 << 31) { t_imp as i32 } else { -1_000_000 };
        let t_touch = self.u64("car_states.latest_touch_tick", k);
        h.ball_touch_frame = if t_touch < (1 << 31) { t_touch as i32 } else { -1_000_000 };
        let other = self.u32("car_states.car_contact.other_car_id", k) as u64;
        let cd = f("car_states.car_contact.cooldown_timer");
        if cd > 0.0 {
            if let Some(v) = ids.iter().position(|&i| i == other) {
                h.bump_victim = Some(v);
                h.bump_time = (1.0 / 120.0) * phys as f32 - (0.25 - cd);
            }
        }
    }
}
