use engine::bt::types::*;
use engine::bt::vehicle::{WheelInfo,Vehicle};
use engine::math::{Vec3,Mat3,Transform};
fn vec3(x:&Vec3,o:&mut Vec<u32>){o.extend(x.bits());}
fn mat3(x:&Mat3,o:&mut Vec<u32>){for r in &x.r{vec3(r,o);}}
fn transform(x:&Transform,o:&mut Vec<u32>){vec3(&x.origin,o);mat3(&x.basis,o);}
fn point(x:&ManifoldPoint,o:&mut Vec<u32>){
vec3(&x.local_a,o);
vec3(&x.local_b,o);
vec3(&x.world_b,o);
vec3(&x.world_a,o);
vec3(&x.normal_b,o);
vec3(&x.offset_a,o);
vec3(&x.offset_b,o);
o.push(x.distance.to_bits());
o.push(x.combined_friction.to_bits());
o.push(x.combined_rolling.to_bits());
o.push(x.combined_restitution.to_bits());
o.push(x.part0 as u32);
o.push(x.part1 as u32);
o.push(x.index0 as u32);
o.push(x.index1 as u32);
o.extend([x.user_data as u32,(x.user_data>>32) as u32]);
o.push(x.lateral_initialized as u32);
o.push(x.applied_impulse.to_bits());
o.push(x.applied_impulse_lat1.to_bits());
o.push(x.applied_impulse_lat2.to_bits());
o.push(x.contact_motion1.to_bits());
o.push(x.contact_motion2.to_bits());
o.push(x.contact_cfm1.to_bits());
o.push(x.contact_cfm2.to_bits());
o.push(x.life_time as u32);
vec3(&x.lat_dir1,o);
vec3(&x.lat_dir2,o);
vec3(&x.vel_a,o);
vec3(&x.vel_b,o);
o.push(x.rest_speed.to_bits());
o.push(x.row_scale1.to_bits());
o.push(x.row_scale2.to_bits());
o.push(x.sphere_on_static as u32);
}
fn body(x:&Body,o:&mut Vec<u32>){
transform(&x.transform,o);
transform(&x.interp_transform,o);
vec3(&x.interp_lin_vel,o);
vec3(&x.interp_ang_vel,o);
vec3(&x.aniso_friction,o);
o.push(x.has_aniso_friction as u32);
o.push(x.contact_processing_threshold.to_bits());
o.push(x.proxy as u32);
o.push(x.shape.0 as u32);
o.push(x.flags as u32);
o.push(x.island_tag as u32);
o.push(x.companion_id as u32);
o.push(x.activation_state as u32);
o.push(x.deactivation_time.to_bits());
o.push(x.friction.to_bits());
o.push(x.restitution.to_bits());
o.push(x.rolling_friction.to_bits());
o.push(x.internal_type as u32);
o.push(x.hit_fraction.to_bits());
o.push(x.ccd_swept_radius.to_bits());
o.push(x.ccd_motion_threshold.to_bits());
o.push(x.check_collide_with as u32);
o.push(x.update_revision as u32);
mat3(&x.inv_inertia_world,o);
vec3(&x.lin_vel,o);
vec3(&x.ang_vel,o);
o.push(x.inv_mass.to_bits());
vec3(&x.lin_factor,o);
vec3(&x.gravity,o);
vec3(&x.gravity_accel,o);
vec3(&x.inv_inertia_local,o);
vec3(&x.total_force,o);
vec3(&x.total_torque,o);
o.push(x.lin_damping.to_bits());
o.push(x.ang_damping.to_bits());
o.push(x.additional_damping as u32);
o.push(x.additional_damping_factor.to_bits());
o.push(x.additional_lin_threshold_sqr.to_bits());
o.push(x.additional_ang_threshold_sqr.to_bits());
o.push(x.additional_ang_factor.to_bits());
o.push(x.lin_sleep_threshold.to_bits());
o.push(x.ang_sleep_threshold.to_bits());
o.push(x.rb_flags as u32);
o.push(x.debug_body_id as u32);
point(&x.merged,o);
o.push(x.merged_dist.to_bits());
o.push(x.merged_count as u32);
o.push(x.max_lin_speed.to_bits());
o.push(x.max_ang_speed.to_bits());
vec3(&x.ang_factor,o);
vec3(&x.inv_mass_factor,o);
o.extend(match x.owner {Owner::World=>[0,0],Owner::Ball=>[1,0],Owner::Car(i)=>[2,i]});
o.push(x.sticky_wheels as u32);
}
fn manifold(x:&Manifold,o:&mut Vec<u32>){
for p in &x.points{point(p,o);}
o.push(x.body0.0 as u32);
o.push(x.body1.0 as u32);
o.push(x.cached as u32);
o.push(x.breaking_threshold.to_bits());
o.push(x.processing_threshold.to_bits());
o.push(x.companion_a as u32);
o.push(x.companion_b as u32);
o.push(x.index1a as u32);
}
fn wheel(x:&WheelInfo,o:&mut Vec<u32>){
vec3(&x.normal,o);
vec3(&x.contact,o);
vec3(&x.contact_velocity,o);
vec3(&x.hard_point,o);
o.push(x.in_contact as u32);
o.extend(match x.ground{Some(id)=>[1,id.0],None=>[0,0]});
o.push(x.suspension_length.to_bits());
o.push(x.min_hit_dist2.to_bits());
mat3(&x.basis,o);
vec3(&x.connection,o);
o.push(x.radius.to_bits());
o.push(x.rest.to_bits());
o.push(x.travel.to_bits());
o.push(x.max_raise.to_bits());
o.push(x.stiffness.to_bits());
o.push(x.damping_compression.to_bits());
o.push(x.damping_relaxation.to_bits());
o.push(x.steer.to_bits());
o.push(x.engine_force.to_bits());
o.push(x.brake.to_bits());
o.push(x.force_scale.to_bits());
o.push(x.up_offset.to_bits());
o.push(x.push_force.to_bits());
o.push(x.clipped_inv_dot.to_bits());
o.push(x.relative_velocity.to_bits());
o.push(x.suspension_force.to_bits());
o.push(x.lat_friction.to_bits());
o.push(x.long_friction.to_bits());
o.push(x.extra_pushback.to_bits());
}
fn vehicle(x:&Vehicle,o:&mut Vec<u32>){
for p in &x.friction_impulse{vec3(p,o);}
o.push(x.contact_mask as u32);
o.push(x.ray_group as u32);
o.push(x.chassis.0 as u32);
o.push(x.keep_contacts as u32);
for p in &x.wheels{wheel(p,o);}
}
pub fn capture(s:&engine::adapter::ArenaSlot,o:&mut Vec<u32>){
    use engine::adapter::*;
    o.clear();
    capture_arena(&s.arena,o);
    let mut ar=[0.0;A_DIM];s.write_arena(&mut ar);o.extend(ar.map(f32::to_bits));
    for i in 0..s.num_cars{
        let mut c=[0.0;C_DIM];s.write_car(i,&mut c);o.extend(c.map(f32::to_bits));
        let mut h=[0.0;H_DIM];s.write_hidden(i,&mut h);o.extend(h.map(f32::to_bits));
        o.extend(s.events[i].map(f32::to_bits));
    }
    o.push(s.goal.to_bits());
}
fn rb(s:&engine::game::rb_state::RbState,o:&mut Vec<u32>){o.extend([s.quat.x.to_bits(),s.quat.y.to_bits(),s.quat.z.to_bits(),s.quat.w.to_bits(),s.sleeping as u32]);vec3(&s.location,o);vec3(&s.velocity,o);vec3(&s.spin,o);}

pub fn capture_raw(a:&engine::game::arena::Arena,o:&mut Vec<u32>){
    o.clear();
    capture_arena(a,o);
    o.extend([a.ball_world_nz.to_bits(),a.ball_on_floor as u32,a.ball_held_2d as u32]);
    if let Some(b)=a.ball.as_ref(){
        rb(&b.state,o);rb(&b.old_state,o);
        o.extend([b.age,b.first_frame as u32,b.seconds_since_grounded.to_bits(),b.last_hit_world_time.to_bits(),b.round_active as u32,b.contacts.len() as u32]);
        vec3(&b.actor_loc,o);
    }
}
fn capture_arena(a:&engine::game::arena::Arena,o:&mut Vec<u32>){
    o.extend([a.physics_frame as u32,a.clock.score[0] as u32,a.clock.score[1] as u32]);
    o.extend([a.rng as u32,(a.rng>>32) as u32,a.render_alpha.to_bits(),a.dt.to_bits()]);
    o.extend([a.crt_state.is_some() as u32, a.crt_state.unwrap_or(0)]);
    if let Some(m) = &a.frame_rng {
        o.extend([1, m.world_time.to_bits(), m.pending_pickup, m.pending_motion,
                  m.motion_fx as u32, m.frames as u32, (m.frames >> 32) as u32, m.steps as u32,
                  m.countdown_entry as u32, m.frame_event_draws]);
        for x in m.draws { o.extend([x as u32, (x >> 32) as u32]); }
        o.extend(m.last); o.extend(m.pad_before.map(f32::to_bits));
        o.push(m.locks.len() as u32);
        for l in &m.locks { o.extend([l.release.to_bits(), l.time.to_bits(), l.random_time.to_bits(), l.running as u32]); }

        o.push(m.mesh.is_some() as u32);
        if let Some(c) = m.mesh {
            o.extend([c.wheels, c.extras, c.map, c.goal_mesh as u32, c.goal.0, c.goal.1, c.demolish.0, c.demolish.1,
                      c.reset.0, c.reset.1, c.respawn.0, c.respawn.1]);
        }
        o.push(m.timers.len() as u32);
        for t in &m.timers { o.extend([t.owner as u32, t.kind as u32, t.last.to_bits(), t.spawn_gen]); }
        o.push(m.pending_parts.len() as u32);
        for p in &m.pending_parts { o.extend([p.0, p.1 as u32, p.2 as u32]); }
        o.push(m.pending_events.len() as u32);
        for e in &m.pending_events { o.extend([e.0, e.1, e.2 as u32]); }
        o.extend([m.goal_fx, m.goal_mesh_wait, m.goal_mesh_ref.is_some() as u32, m.goal_mesh_ref.unwrap_or(0.0).to_bits(), m.next_gen]);
        o.push(m.car_gen.len() as u32); o.extend(m.car_gen.iter().copied());
        o.push(m.motion_by.len() as u32); o.extend(m.motion_by.iter().copied());
    } else { o.push(0); }
    use engine::game::arena::{Phase,Actor};
    o.extend(match a.phase {Phase::Countdown{left}=>[0,left as u32],Phase::Active=>[1,0],Phase::PostGoal{left}=>[2,left.to_bits()],Phase::Ended=>[3,0]});
    o.extend([a.clock.game_time.to_bits(),a.clock.overtime as u32,a.clock.ball_has_been_hit as u32,a.clock.hit_delay_left.to_bits(),a.clock.round_num as u32,a.clock.waiting_end_round as u32]);
    o.push(a.pads.layout as u32);
    for p in &a.pads.pads{o.extend([p.picked_up as u32,p.respawn_left.to_bits(),p.timer.rate.to_bits(),p.timer.elapsed.to_bits(),p.timer.active as u32]);}
    o.push(a.notifies.len() as u32);
    for n in &a.notifies{o.extend(match n {Actor::Ball=>[0,0],Actor::Car(i)=>[1,*i as u32]});}
    o.push(a.vehicles.len() as u32);o.extend(a.vehicles.iter().map(|v|*v as u32));
    for p in &a.players{o.extend([p.respawn_left.to_bits(),p.demolished as u32,p.demolish_pending as u32,p.fresh_body as u32,p.last_ball_impact_frame as u32,p.last_ball_touch_frame as u32,p.last_hit_time.to_bits()]);}
    for p in &a.pads.pads{o.extend([p.respawned as u32,p.skip as u32,p.tick_enabled as u32,p.ticked_this_frame as u32]);}
    for p in &a.players{o.extend([p.touching as u32,(p.touching>>32) as u32,p.body.product]);}
    for p in &a.players{match &p.car{Some(c)=>{o.push(1);vehicle(&c.vehicle,o);o.extend(c.wheels.iter().map(|w|w.contact_notified as u32));},None=>o.push(0)}}
    if let Some(b)=a.ball.as_ref(){let s=&b.birth_state;o.extend([s.quat.x.to_bits(),s.quat.y.to_bits(),s.quat.z.to_bits(),s.quat.w.to_bits(),s.sleeping as u32]);vec3(&s.location,o);vec3(&s.velocity,o);vec3(&s.spin,o);}
    o.push(a.world.cw.bodies.len() as u32);
    for b in &a.world.cw.bodies{body(b,o);}
    o.push(a.world.cw.objects.len() as u32);
    o.extend(a.world.cw.objects.iter().map(|id|id.0));
    o.push(a.world.rigid_bodies.len() as u32);
    o.extend(a.world.rigid_bodies.iter().map(|id|id.0));
    let bp=&a.world.cw.broadphase;
    o.extend([bp.num_handles as u32,bp.first_free as u32,bp.handle_mask as u32,bp.handle_sentinel as u32]);
    vec3(&bp.world_min,o);vec3(&bp.world_max,o);vec3(&bp.quantize_scale,o);
    for axis in 0..3 {for e in &bp.edges[axis][..bp.num_handles as usize*2+2] {o.extend([e.pos as u32,e.handle as u32]);}}
    for id in &a.world.cw.objects {
        let proxy=a.world.cw.bodies[id.0 as usize].proxy as usize;
        let h=&bp.handles[proxy];
        o.extend([proxy as u32,h.client,h.group as u32,h.mask as u32,h.unique_id as u32]);
        vec3(&h.aabb_min,o);vec3(&h.aabb_max,o);
        o.extend(h.min_edges.map(u32::from));o.extend(h.max_edges.map(u32::from));
    }
    o.push(bp.pairs.capacity as u32);o.push(bp.pairs.pairs.len() as u32);
    for p in &bp.pairs.pairs{o.extend([p.proxy0 as u32,p.proxy1 as u32,p.algorithm.is_some() as u32]);}
    o.extend(bp.pairs.hash_table.iter().map(|x|*x as u32));o.extend(bp.pairs.next.iter().map(|x|*x as u32));
    let m=a.world.cw.manifolds();o.push(m.len() as u32);
    for &id in m{o.push(id.0);manifold(a.world.cw.manifold(id),o);}
}
