use super::rigid_body::integrate_transform;
use super::types::*;
use crate::math::{Transform, Vec3, v3};

#[derive(Clone, Copy, Debug)]
pub struct SolverInfo {
    pub tau: f32,
    pub damping: f32,
    pub friction: f32,
    pub time_step: f32,
    pub restitution: f32,
    pub num_iterations: i32,
    pub max_error_reduction: f32,
    pub sor: f32,
    pub erp: f32,
    pub erp2: f32,
    pub global_cfm: f32,
    pub split_impulse: i32,
    pub split_threshold: f32,
    pub split_turn_erp: f32,
    pub linear_slop: f32,
    pub warmstarting: f32,
    pub solver_mode: i32,
    pub resting_restitution: i32,
    pub min_batch: i32,
    pub max_gyroscopic: f32,
    pub single_axis_rolling: f32,

    pub sub_separation: bool,
}

impl SolverInfo {

    pub fn game() -> SolverInfo {
        SolverInfo {
            tau: f32::from_bits(0x3f19999a),
            damping: 1.0,
            friction: f32::from_bits(0x3e99999a),
            time_step: f32::from_bits(0x3c888889),
            restitution: 0.0,
            num_iterations: 10,
            max_error_reduction: 20.0,
            sor: 1.0,
            erp: f32::from_bits(0x3e4ccccd),
            erp2: f32::from_bits(0x3f4ccccd),
            global_cfm: 0.0,
            split_impulse: 1,
            split_threshold: f32::from_bits(0x7149f2ca),
            split_turn_erp: f32::from_bits(0x3dcccccd),
            linear_slop: 0.0,
            warmstarting: f32::from_bits(0x3f59999a),
            solver_mode: 0,
            resting_restitution: 2,
            min_batch: 128,
            max_gyroscopic: 100.0,
            single_axis_rolling: f32::from_bits(0x7149f2ca),
            sub_separation: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SolverBody {
    pub transform: Transform,
    pub delta_lin: Vec3,
    pub delta_ang: Vec3,
    pub ang_factor: Vec3,
    pub lin_factor: Vec3,
    pub inv_mass: Vec3,
    pub push: Vec3,
    pub turn: Vec3,
    pub lin_vel: Vec3,
    pub ang_vel: Vec3,
    pub ext_force: Vec3,
    pub ext_torque: Vec3,
    pub orig: Option<BodyId>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Row {
    pub cross1: Vec3,
    pub n1: Vec3,
    pub cross2: Vec3,
    pub n2: Vec3,
    pub ang_a: Vec3,
    pub ang_b: Vec3,
    pub push: f32,
    pub applied: f32,
    pub friction: f32,
    pub jac_inv: f32,
    pub rhs: f32,
    pub cfm: f32,
    pub lower: f32,
    pub upper: f32,
    pub rhs_pen: f32,

    pub other: i32,
    pub a: i32,
    pub b: i32,
    pub scale1: f32,
    pub scale2: f32,
    pub skip: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Solver {
    pub bodies: Vec<SolverBody>,
    pub contacts: Vec<Row>,
    pub frictions: Vec<Row>,
    pub fixed: i32,
}

pub fn init_solver_body(body: Option<(BodyId, &Body)>, dt: f32) -> SolverBody {
    match body {
        None => SolverBody {
            transform: Transform::IDENTITY,
            delta_lin: Vec3::ZERO,
            delta_ang: Vec3::ZERO,
            ang_factor: v3(1.0, 1.0, 1.0),
            lin_factor: v3(1.0, 1.0, 1.0),
            inv_mass: Vec3::ZERO,
            push: Vec3::ZERO,
            turn: Vec3::ZERO,
            lin_vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            ext_force: Vec3::ZERO,
            ext_torque: Vec3::ZERO,
            orig: None,
        },
        Some((id, b)) => {
            let im = b.inv_mass;
            let i = &b.inv_inertia_world;
            let t = b.total_torque;
            let f = b.total_force;
            SolverBody {
                transform: b.transform,
                delta_lin: Vec3::ZERO,
                delta_ang: Vec3::ZERO,
                ang_factor: b.ang_factor,
                lin_factor: b.lin_factor,
                inv_mass: v3(im * b.lin_factor.x, im * b.lin_factor.y, im * b.lin_factor.z),
                push: Vec3::ZERO,
                turn: Vec3::ZERO,
                lin_vel: b.lin_vel,
                ang_vel: b.ang_vel,
                ext_force: v3(im * f.x * dt, im * f.y * dt, im * f.z * dt),
                ext_torque: v3(
                    (t.y * i.r[1].x + t.x * i.r[0].x + t.z * i.r[2].x) * dt,
                    (t.y * i.r[1].y + t.x * i.r[0].y + t.z * i.r[2].y) * dt,
                    (t.y * i.r[1].z + t.x * i.r[0].z + t.z * i.r[2].z) * dt,
                ),
                orig: Some(id),
            }
        }
    }
}

pub fn apply_anisotropic(col: Option<&Body>, d: &mut Vec3) {
    if let Some(b) = col {
        if b.has_aniso_friction & 1 != 0 {
            let m = &b.transform.basis;
            let a = b.aniso_friction;
            let x = (m.r[1].x * d.y + m.r[0].x * d.x + m.r[2].x * d.z) * a.x;
            let y = (d.y * m.r[1].y + m.r[0].y * d.x + d.z * m.r[2].y) * a.y;
            let z = (d.y * m.r[1].z + m.r[0].z * d.x + d.z * m.r[2].z) * a.z;
            *d = v3(
                y * m.r[0].y + x * m.r[0].x + z * m.r[0].z,
                y * m.r[1].y + x * m.r[1].x + z * m.r[1].z,
                y * m.r[2].y + x * m.r[2].x + z * m.r[2].z,
            );
        }
    }
}

impl Solver {
    pub fn new() -> Solver {
        Solver { bodies: Vec::new(), contacts: Vec::new(), frictions: Vec::new(), fixed: -1 }
    }

    fn fixed_body(&mut self, dt: f32) -> i32 {
        if self.fixed < 0 {
            self.fixed = self.bodies.len() as i32;
            self.bodies.push(init_solver_body(None, dt));
        }
        self.fixed
    }

    pub fn solver_body_of(&mut self, bodies: &mut [Body], id: BodyId, dt: f32) -> i32 {
        let c = bodies[id.0 as usize].companion_id;
        if c >= 0 {
            return c;
        }
        let b = &bodies[id.0 as usize];
        if !b.is_rigid() || (b.inv_mass == 0.0 && b.flags & CF_KINEMATIC == 0) {
            return self.fixed_body(dt);
        }
        let n = self.bodies.len() as i32;
        self.bodies.push(init_solver_body(Some((id, b)), dt));
        bodies[id.0 as usize].companion_id = n;
        n
    }

    pub fn contact_row(&self, bodies: &[Body], ia: i32, ib: i32, pt: &mut ManifoldPoint, info: &SolverInfo, r1: Vec3, r2: Vec3) -> Row {
        let sa = &self.bodies[ia as usize];
        let sb = &self.bodies[ib as usize];
        let rb_a = sa.orig.map(|i| &bodies[i.0 as usize]);
        let rb_b = sb.orig.map(|i| &bodies[i.0 as usize]);
        let relax = 1.0f32;
        let n = pt.normal_b;
        let c1 = v3(r1.y * n.z - r1.z * n.y, n.x * r1.z - r1.x * n.z, r1.x * n.y - n.x * r1.y);
        let ang_a = match rb_a {
            None => Vec3::ZERO,
            Some(b) => {
                let i = &b.inv_inertia_world;
                v3(
                    (c1.y * i.r[0].y + c1.x * i.r[0].x + c1.z * i.r[0].z) * b.ang_factor.x,
                    (c1.y * i.r[1].y + c1.x * i.r[1].x + c1.z * i.r[1].z) * b.ang_factor.y,
                    (c1.y * i.r[2].y + c1.x * i.r[2].x + c1.z * i.r[2].z) * b.ang_factor.z,
                )
            }
        };
        let c2 = v3(-(r2.y * n.z - r2.z * n.y), -(n.x * r2.z - r2.x * n.z), -(r2.x * n.y - n.x * r2.y));
        let ang_b = match rb_b {
            None => Vec3::ZERO,
            Some(b) => {
                let i = &b.inv_inertia_world;
                v3(
                    (c2.y * i.r[0].y + c2.x * i.r[0].x + c2.z * i.r[0].z) * b.ang_factor.x,
                    (c2.y * i.r[1].y + c2.x * i.r[1].x + c2.z * i.r[1].z) * b.ang_factor.y,
                    (c2.y * i.r[2].y + c2.x * i.r[2].x + c2.z * i.r[2].z) * b.ang_factor.z,
                )
            }
        };
        let denom_a = match rb_a {
            None => 0.0,
            Some(b) => {
                let a = ang_a;
                (r1.x * a.z - a.x * r1.z) * n.y + (r1.z * a.y - r1.y * a.z) * n.x + (a.x * r1.y - r1.x * a.y) * n.z + b.inv_mass
            }
        };
        let denom_b = match rb_b {
            None => 0.0,
            Some(b) => {
                let a = ang_b;
                (r2.x * -a.z - r2.z * -a.x) * n.y + (r2.z * -a.y - r2.y * -a.z) * n.x + (r2.y * -a.x - r2.x * -a.y) * n.z + b.inv_mass
            }
        };
        let mut row = Row { a: ia, b: ib, ang_a, ang_b, jac_inv: relax / (denom_b + denom_a), ..Row::default() };
        if rb_a.is_some() {
            row.n1 = n;
            row.cross1 = c1;
        }
        if rb_b.is_some() {
            row.n2 = v3(-n.x, -n.y, -n.z);
            row.cross2 = c2;
        }
        let d = pt.distance + info.linear_slop;
        let v1 = match rb_a {
            None => Vec3::ZERO,
            Some(b) => {
                let (w, v) = (b.ang_vel, b.lin_vel);
                v3((r1.z * w.y - r1.y * w.z) + v.x, (r1.x * w.z - w.x * r1.z) + v.y, (w.x * r1.y - r1.x * w.y) + v.z)
            }
        };
        let v2 = match rb_b {
            None => Vec3::ZERO,
            Some(b) => {
                let (w, v) = (b.ang_vel, b.lin_vel);
                v3((r2.z * w.y - r2.y * w.z) + v.x, (r2.x * w.z - w.x * r2.z) + v.y, (w.x * r2.y - r2.x * w.y) + v.z)
            }
        };
        row.friction = pt.combined_friction;
        pt.vel_a = v1;
        pt.vel_b = v2;
        let s = -((v1.y - v2.y) * n.y + (v1.x - v2.x) * n.x + (v1.z - v2.z) * n.z) * pt.combined_restitution;
        let rest = if 0.0 <= s { s } else { 0.0 };
        pt.rest_speed = rest;
        let (ef_a, et_a) = if sa.orig.is_none() { (Vec3::ZERO, Vec3::ZERO) } else { (sa.ext_force, sa.ext_torque) };
        let (ef_b, et_b) = if sb.orig.is_none() { (Vec3::ZERO, Vec3::ZERO) } else { (sb.ext_force, sb.ext_torque) };

        let spin_a = (et_a.y + sa.ang_vel.y) * row.cross1.y + (et_a.x + sa.ang_vel.x) * row.cross1.x + (et_a.z + sa.ang_vel.z) * row.cross1.z;
        let lin_a = (ef_a.y + sa.lin_vel.y) * row.n1.y + (ef_a.x + sa.lin_vel.x) * row.n1.x + (ef_a.z + sa.lin_vel.z) * row.n1.z;
        let lin_b = (ef_b.y + sb.lin_vel.y) * row.n2.y + (ef_b.x + sb.lin_vel.x) * row.n2.x + (ef_b.z + sb.lin_vel.z) * row.n2.z;
        let spin_b = (et_b.y + sb.ang_vel.y) * row.cross2.y + (et_b.x + sb.ang_vel.x) * row.cross2.x + (et_b.z + sb.ang_vel.z) * row.cross2.z;
        let mut vel = rest - ((spin_a + lin_a) + (lin_b + spin_b));
        let no_split = info.split_impulse == 0 || (info.split_threshold <= d && d != info.split_threshold);
        let erp = if no_split { info.erp } else { info.erp2 };
        let mut pen = 0.0;
        if d <= 0.0 {
            pen = (-d * erp) / info.time_step;
        } else if info.sub_separation {
            vel -= d / info.time_step;
        }
        let mut vel = row.jac_inv * vel;
        let mut pen = row.jac_inv * pen;
        if no_split {
            vel += pen;
            pen = 0.0;
        }
        row.rhs = vel;
        row.rhs_pen = pen;
        row.cfm = 0.0;
        row.lower = 0.0;
        row.upper = 1e10;
        row.scale1 = pt.row_scale1;
        row.scale2 = pt.row_scale2;
        row.skip = pt.sphere_on_static;
        row
    }

    pub fn friction_row(&self, bodies: &[Body], dir: Vec3, ia: i32, ib: i32, friction: f32, r1: Vec3, r2: Vec3, relax: f32, motion: f32, cfm: f32, contact: i32) -> Row {
        let sa = &self.bodies[ia as usize];
        let sb = &self.bodies[ib as usize];
        let rb_a = sa.orig.map(|i| &bodies[i.0 as usize]);
        let rb_b = sb.orig.map(|i| &bodies[i.0 as usize]);
        let mut row = Row { a: ia, b: ib, friction, other: contact, ..Row::default() };
        if let Some(b) = rb_a {
            row.n1 = dir;
            let c = v3(r1.y * dir.z - r1.z * dir.y, dir.x * r1.z - r1.x * dir.z, r1.x * dir.y - dir.x * r1.y);
            row.cross1 = c;
            let i = &b.inv_inertia_world;
            row.ang_a = v3(
                (c.y * i.r[0].y + c.x * i.r[0].x + c.z * i.r[0].z) * b.ang_factor.x,
                (c.y * i.r[1].y + c.x * i.r[1].x + c.z * i.r[1].z) * b.ang_factor.y,
                (c.y * i.r[2].y + c.x * i.r[2].x + c.z * i.r[2].z) * b.ang_factor.z,
            );
        }
        if let Some(b) = rb_b {
            let d = v3(-dir.x, -dir.y, -dir.z);
            row.n2 = d;
            let c = v3(r2.y * d.z - r2.z * d.y, d.x * r2.z - r2.x * d.z, r2.x * d.y - d.x * r2.y);
            row.cross2 = c;
            let i = &b.inv_inertia_world;
            row.ang_b = v3(
                (c.y * i.r[0].y + c.x * i.r[0].x + c.z * i.r[0].z) * b.ang_factor.x,
                (c.y * i.r[1].y + c.x * i.r[1].x + c.z * i.r[1].z) * b.ang_factor.y,
                (c.y * i.r[2].y + c.x * i.r[2].x + c.z * i.r[2].z) * b.ang_factor.z,
            );
        }
        let denom_a = match rb_a {
            None => 0.0,
            Some(b) => {
                let a = row.ang_a;
                (r1.z * a.y - r1.y * a.z) * dir.x + (r1.x * a.z - r1.z * a.x) * dir.y + (r1.y * a.x - r1.x * a.y) * dir.z + b.inv_mass
            }
        };
        let denom_b = match rb_b {
            None => 0.0,
            Some(b) => {
                let a = row.ang_b;
                (r2.z * -a.y - r2.y * -a.z) * dir.x + (r2.x * -a.z - r2.z * -a.x) * dir.y + (r2.y * -a.x - r2.x * -a.y) * dir.z + b.inv_mass
            }
        };
        let jac = relax / (denom_b + denom_a);
        row.jac_inv = jac;
        let (va, wa) = match rb_a {
            None => (Vec3::ZERO, Vec3::ZERO),
            Some(_) => (sa.lin_vel + sa.ext_force, sa.ang_vel),
        };
        let (vb, wb) = match rb_b {
            None => (Vec3::ZERO, Vec3::ZERO),
            Some(_) => (sb.lin_vel + sb.ext_force, sb.ang_vel),
        };
        row.upper = row.friction;
        row.lower = -row.friction;

        let lin_a = va.y * row.n1.y + va.x * row.n1.x + va.z * row.n1.z;
        let spin_a = wa.y * row.cross1.y + wa.x * row.cross1.x + wa.z * row.cross1.z;
        let lin_b = vb.y * row.n2.y + vb.x * row.n2.x + vb.z * row.n2.z;
        let spin_b = wb.y * row.cross2.y + wb.x * row.cross2.x + wb.z * row.cross2.z;
        row.rhs = (motion - ((spin_b + lin_b) + (spin_a + lin_a))) * jac;
        row.cfm = cfm;
        row
    }

    pub fn friction_for_contact(&mut self, bodies: &[Body], col0: Option<BodyId>, col1: Option<BodyId>, ia: i32, ib: i32, pt: &mut ManifoldPoint, contact: i32, r1: Vec3, r2: Vec3, relax: f32) {
        let sa = &self.bodies[ia as usize];
        let sb = &self.bodies[ib as usize];
        let vel_at = |s: &SolverBody, r: Vec3| -> Vec3 {
            if s.orig.is_none() {
                return Vec3::ZERO;
            }
            let wy = s.ang_vel.y + s.ext_torque.y;
            let wz = s.ang_vel.z + s.ext_torque.z;
            let wx = s.ang_vel.x + s.ext_torque.x;
            v3(
                (r.z * wy - r.y * wz) + (s.lin_vel.x + s.ext_force.x),
                (r.x * wz - r.z * wx) + (s.lin_vel.y + s.ext_force.y),
                (r.y * wx - r.x * wy) + (s.lin_vel.z + s.ext_force.z),
            )
        };
        let v1 = vel_at(sa, r1);
        let v2 = vel_at(sb, r2);
        let n = pt.normal_b;
        let dv = v3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
        let rel = n.x * dv.x + n.y * dv.y + n.z * dv.z;
        let lat = v3(dv.x - n.x * rel, dv.y - n.y * rel, dv.z - n.z * rel);
        pt.lat_dir1 = lat;
        let l2 = lat.y * lat.y + lat.x * lat.x + lat.z * lat.z;
        if l2 <= f32::from_bits(0x34000000) {

            if n.z.abs() <= f32::from_bits(0x3f3504f3) {
                let a = n.x * n.x + n.y * n.y;
                let k = if f32::from_bits(0x25800000) < a { 1.0 / a.sqrt() } else { 0.0 };
                pt.lat_dir1 = v3(-n.y * k, k * n.x, 0.0);
                pt.lat_dir2 = v3(-n.z * pt.lat_dir1.y, pt.lat_dir1.x * n.z, k * a);
            } else {
                let a = n.z * n.z + n.y * n.y;
                let k = if f32::from_bits(0x25800000) < a { 1.0 / a.sqrt() } else { 0.0 };
                pt.lat_dir1 = v3(0.0, -n.z * k, k * n.y);
                pt.lat_dir2 = v3(k * a, -n.x * pt.lat_dir1.z, n.x * pt.lat_dir1.y);
            }
        } else {
            let k = 1.0 / l2.sqrt();
            pt.lat_dir1 = v3(k * lat.x, k * lat.y, k * lat.z);
        }
        let mut d = pt.lat_dir1;
        apply_anisotropic(col0.map(|i| &bodies[i.0 as usize]), &mut d);
        apply_anisotropic(col1.map(|i| &bodies[i.0 as usize]), &mut d);
        pt.lat_dir1 = d;
        let row = self.friction_row(bodies, d, ia, ib, pt.combined_friction, r1, r2, relax, 0.0, 0.0, contact);
        self.frictions.push(row);
    }

    pub fn convert_contact(&mut self, bodies: &mut [Body], m: &mut Manifold, info: &SolverInfo) {
        let (id0, id1) = (m.body0, m.body1);
        let rigid0 = bodies[id0.0 as usize].is_rigid();
        let rigid1 = bodies[id1.0 as usize].is_rigid();
        let ia = self.solver_body_of(bodies, id0, info.time_step);
        let ib = self.solver_body_of(bodies, id1, info.time_step);
        {
            let (sa, sb) = (&self.bodies[ia as usize], &self.bodies[ib as usize]);
            let z = |v: Vec3| v.x == 0.0 && v.y == 0.0 && v.z == 0.0;
            if z(sa.inv_mass) && z(sb.inv_mass) {
                return;
            }
        }
        let threshold = m.processing_threshold;
        for j in 0..m.cached.max(0) as usize {
            let pt = &mut m.points[j];
            if pt.distance <= threshold {
                let contact = self.contacts.len() as i32;
                let o0 = bodies[id0.0 as usize].transform.origin;
                let o1 = bodies[id1.0 as usize].transform.origin;
                let r1 = v3((pt.world_a.x + pt.offset_a.x) - o0.x, (pt.world_a.y + pt.offset_a.y) - o0.y, (pt.world_a.z + pt.offset_a.z) - o0.z);
                let r2 = v3((pt.world_b.x + pt.offset_b.x) - o1.x, (pt.world_b.y + pt.offset_b.y) - o1.y, (pt.world_b.z + pt.offset_b.z) - o1.z);
                let row = self.contact_row(bodies, ia, ib, pt, info, r1, r2);
                self.contacts.push(row);
                if !pt.sphere_on_static {
                    self.friction_for_contact(bodies, Some(id0), Some(id1), ia, ib, pt, contact, r1, r2, 1.0);
                    self.contacts[contact as usize].other = contact_friction_index(&self.frictions);
                } else {
                    if rigid0 {
                        let b = &mut bodies[id0.0 as usize];
                        b.merged_count += 1;
                        b.merged.combined_friction = pt.combined_friction + b.merged.combined_friction;
                        b.merged.combined_restitution = pt.combined_restitution + b.merged.combined_restitution;
                        let d2 = r1.y * r1.y + r1.x * r1.x + r1.z * r1.z;
                        b.merged.normal_b.x = pt.normal_b.x + b.merged.normal_b.x;
                        b.merged.normal_b.y = pt.normal_b.y + b.merged.normal_b.y;
                        b.merged.normal_b.z = pt.normal_b.z + b.merged.normal_b.z;
                        b.merged_dist = d2.sqrt() + b.merged_dist;
                    }
                    if rigid1 {
                        let b = &mut bodies[id1.0 as usize];
                        b.merged_count += 1;
                        b.merged.combined_friction = pt.combined_friction + b.merged.combined_friction;
                        b.merged.combined_restitution = pt.combined_restitution + b.merged.combined_restitution;
                        let d2 = r2.y * r2.y + r2.x * r2.x + r2.z * r2.z;
                        b.merged.normal_b.x = pt.normal_b.x + b.merged.normal_b.x;
                        b.merged.normal_b.y = pt.normal_b.y + b.merged.normal_b.y;
                        b.merged.normal_b.z = pt.normal_b.z + b.merged.normal_b.z;
                        b.merged_dist = d2.sqrt() + b.merged_dist;
                    }
                }
            }
        }
    }

    fn merged_row(&mut self, bodies: &mut [Body], id: BodyId, info: &SolverInfo) {
        let ia = self.solver_body_of(bodies, id, info.time_step);
        let s = &self.bodies[ia as usize];
        if s.inv_mass.x == 0.0 && s.inv_mass.y == 0.0 && s.inv_mass.z == 0.0 {
            return;
        }
        let ib = self.fixed_body(info.time_step);
        let contact = self.contacts.len() as i32;
        let mut pt = bodies[id.0 as usize].merged;
        let d = bodies[id.0 as usize].merged_dist;
        let r1 = v3(-pt.normal_b.x * d, -pt.normal_b.y * d, -pt.normal_b.z * d);
        let r2 = Vec3::ZERO;
        let row = self.contact_row(bodies, ia, ib, &mut pt, info, r1, r2);
        self.contacts.push(row);
        self.friction_for_contact(bodies, Some(id), None, ia, ib, &mut pt, contact, r1, r2, 1.0);
        self.contacts[contact as usize].other = contact_friction_index(&self.frictions);
        bodies[id.0 as usize].merged = pt;
    }

    pub fn setup(&mut self, bodies: &mut [Body], group: &[BodyId], manifolds: &mut [Manifold], list: &[ManifoldId], info: &SolverInfo) {
        self.fixed = -1;
        for id in group {
            bodies[id.0 as usize].companion_id = -1;
        }
        self.bodies.clear();
        for &id in group {
            self.solver_body_of(bodies, id, info.time_step);
            let b = &mut bodies[id.0 as usize];
            if b.is_rigid() && b.inv_mass != 0.0 {
                b.merged = ManifoldPoint::default();
                b.merged_dist = 0.0;
                b.merged_count = 0;
            }
        }
        for m in list {
            self.convert_contact(bodies, &mut manifolds[m.0 as usize], info);
        }
        for &id in group {
            let b = &mut bodies[id.0 as usize];
            if b.is_rigid() && b.merged_count > 0 {
                if b.merged_count > 1 {
                    let k = 1.0 / (b.merged_count as f32);
                    b.merged_count = 1;
                    b.merged.combined_friction = k * b.merged.combined_friction;
                    b.merged.combined_restitution = k * b.merged.combined_restitution;
                    let n = b.merged.normal_b;
                    let l = 1.0 / (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
                    b.merged.normal_b = v3(l * n.x, l * n.y, l * n.z);
                    b.merged_dist = k * b.merged_dist;
                }
                self.merged_row(bodies, id, info);
            }
        }
    }

    pub fn split_impulse_iterations(&mut self, info: &SolverInfo) {
        if info.split_impulse != 0 {
            for _ in 0..info.num_iterations {
                for i in 0..self.contacts.len() {
                    let mut r = self.contacts[i];
                    resolve_split(&mut self.bodies, &mut r);
                    self.contacts[i] = r;
                }
            }
        }
    }

    pub fn iteration(&mut self) {
        for i in 0..self.contacts.len() {
            if !self.contacts[i].skip {
                let mut r = self.contacts[i];
                resolve_lower(&mut self.bodies, &mut r);
                self.contacts[i] = r;
            }
        }
        for i in 0..self.frictions.len() {
            let imp = self.contacts[self.frictions[i].other as usize].applied;
            if 0.0 < imp {
                let mut r = self.frictions[i];
                let limit = imp * r.friction;
                r.upper = limit;
                r.lower = -limit;
                resolve_generic(&mut self.bodies, &mut r);
                self.frictions[i] = r;
            }
        }
    }

    pub fn finish(&mut self, bodies: &mut [Body], info: &SolverInfo) {
        for s in self.bodies.iter_mut() {
            let Some(id) = s.orig else { continue };
            s.lin_vel = v3(s.lin_vel.x + s.delta_lin.x, s.delta_lin.y + s.lin_vel.y, s.delta_lin.z + s.lin_vel.z);
            s.ang_vel = v3(s.ang_vel.x + s.delta_ang.x, s.delta_ang.y + s.ang_vel.y, s.delta_ang.z + s.ang_vel.z);
            let split = info.split_impulse != 0;
            if split {
                let (p, t) = (s.push, s.turn);
                if p.x != 0.0 || p.y != 0.0 || p.z != 0.0 || t.x != 0.0 || t.y != 0.0 || t.z != 0.0 {
                    let e = info.split_turn_erp;
                    let scaled = v3(e * t.x, e * t.y, e * t.z);
                    s.transform = integrate_transform(&s.transform, p, scaled, info.time_step);
                }
            }
            let b = &mut bodies[id.0 as usize];
            b.update_revision += 1;
            b.lin_vel = v3(s.lin_vel.x + s.ext_force.x, s.ext_force.y + s.lin_vel.y, s.ext_force.z + s.lin_vel.z);
            b.update_revision += 1;
            b.ang_vel = v3(s.ang_vel.x + s.ext_torque.x, s.ext_torque.y + s.ang_vel.y, s.ext_torque.z + s.ang_vel.z);
            if split {
                b.update_revision += 1;
                b.transform = s.transform;
            }
            b.companion_id = -1;
            if b.activation_state == ISLAND_SLEEPING
                && (b.lin_vel.x != 0.0 || b.lin_vel.y != 0.0 || b.lin_vel.z != 0.0 || b.ang_vel.x != 0.0 || b.ang_vel.y != 0.0 || b.ang_vel.z != 0.0)
            {
                b.set_activation_state(WANTS_DEACTIVATION);
                b.deactivation_time = 0.0;
            }
        }
        self.contacts.clear();
        self.frictions.clear();
        self.bodies.clear();
    }

    pub fn solve_group(&mut self, bodies: &mut [Body], group: &[BodyId], manifolds: &mut [Manifold], list: &[ManifoldId], info: &SolverInfo) {
        self.setup(bodies, group, manifolds, list, info);
        self.split_impulse_iterations(info);
        for _ in 0..info.num_iterations {
            self.iteration();
        }
        self.finish(bodies, info);
    }
}

#[inline]
fn contact_friction_index(frictions: &[Row]) -> i32 {
    frictions.len() as i32 - 1
}

#[inline(always)]
fn chain(c: Vec3, n: Vec3, w: Vec3, v: Vec3) -> f32 {
    (c.y * w.y + c.x * w.x + c.z * w.z) + (n.x * v.x + n.y * v.y + n.z * v.z)
}

pub fn resolve_generic(bodies: &mut [SolverBody], r: &mut Row) {
    let (ia, ib) = (r.a as usize, r.b as usize);
    let mut dl = ((r.rhs - r.applied * r.cfm) - chain(r.cross1, r.n1, bodies[ia].delta_ang, bodies[ia].delta_lin) * r.jac_inv) - chain(r.cross2, r.n2, bodies[ib].delta_ang, bodies[ib].delta_lin) * r.jac_inv;
    let mut sum = dl + r.applied;
    if r.lower <= sum {
        if r.upper < sum {
            dl = r.upper - r.applied;
            sum = r.upper;
        }
    } else {
        dl = r.lower - r.applied;
        sum = r.lower;
    }
    r.applied = sum;
    let a = &mut bodies[ia];
    if a.orig.is_some() {
        a.delta_lin.y = a.inv_mass.y * r.n1.y * dl * a.lin_factor.y + a.delta_lin.y;
        a.delta_lin.z = a.inv_mass.z * r.n1.z * dl * a.lin_factor.z + a.delta_lin.z;
        a.delta_lin.x = r.n1.x * a.inv_mass.x * dl * a.lin_factor.x + a.delta_lin.x;
        a.delta_ang.x = dl * a.ang_factor.x * r.ang_a.x + a.delta_ang.x;
        a.delta_ang.z = dl * a.ang_factor.z * r.ang_a.z + a.delta_ang.z;
        a.delta_ang.y = dl * a.ang_factor.y * r.ang_a.y + a.delta_ang.y;
    }
    let b = &mut bodies[ib];
    if b.orig.is_some() {
        b.delta_lin.y = b.inv_mass.y * r.n2.y * dl * b.lin_factor.y + b.delta_lin.y;
        b.delta_lin.x = b.inv_mass.x * r.n2.x * dl * b.lin_factor.x + b.delta_lin.x;
        b.delta_lin.z = b.inv_mass.z * r.n2.z * dl * b.lin_factor.z + b.delta_lin.z;
        b.delta_ang.y = dl * b.ang_factor.y * r.ang_b.y + b.delta_ang.y;
        b.delta_ang.z = dl * b.ang_factor.z * r.ang_b.z + b.delta_ang.z;
        b.delta_ang.x = dl * b.ang_factor.x * r.ang_b.x + b.delta_ang.x;
    }
}

pub fn resolve_lower(bodies: &mut [SolverBody], r: &mut Row) {
    let (ia, ib) = (r.a as usize, r.b as usize);
    let mut dl = ((r.rhs - r.applied * r.cfm) - chain(r.cross1, r.n1, bodies[ia].delta_ang, bodies[ia].delta_lin) * r.jac_inv) - chain(r.cross2, r.n2, bodies[ib].delta_ang, bodies[ib].delta_lin) * r.jac_inv;
    let mut sum = dl + r.applied;
    if sum < r.lower {
        dl = r.lower - r.applied;
        sum = r.lower;
    }
    r.applied = sum;
    let d = dl * r.scale1;
    let a = &mut bodies[ia];
    if a.orig.is_some() {
        a.delta_lin.y = a.inv_mass.y * r.n1.y * d * a.lin_factor.y + a.delta_lin.y;
        a.delta_lin.z = a.inv_mass.z * r.n1.z * d * a.lin_factor.z + a.delta_lin.z;
        a.delta_lin.x = r.n1.x * a.inv_mass.x * d * a.lin_factor.x + a.delta_lin.x;
        a.delta_ang.y = d * a.ang_factor.y * r.ang_a.y + a.delta_ang.y;
        a.delta_ang.z = d * a.ang_factor.z * r.ang_a.z + a.delta_ang.z;
        a.delta_ang.x = d * a.ang_factor.x * r.ang_a.x + a.delta_ang.x;
    }
    let d = dl * r.scale2;
    let b = &mut bodies[ib];
    if b.orig.is_some() {
        b.delta_lin.y = b.inv_mass.y * r.n2.y * d * b.lin_factor.y + b.delta_lin.y;
        b.delta_lin.x = r.n2.x * b.inv_mass.x * d * b.lin_factor.x + b.delta_lin.x;
        b.delta_lin.z = b.inv_mass.z * r.n2.z * d * b.lin_factor.z + b.delta_lin.z;
        b.delta_ang.y = d * b.ang_factor.y * r.ang_b.y + b.delta_ang.y;
        b.delta_ang.z = d * b.ang_factor.z * r.ang_b.z + b.delta_ang.z;
        b.delta_ang.x = d * b.ang_factor.x * r.ang_b.x + b.delta_ang.x;
    }
}

pub fn resolve_split(bodies: &mut [SolverBody], r: &mut Row) {
    if r.rhs_pen == 0.0 {
        return;
    }
    let (ia, ib) = (r.a as usize, r.b as usize);
    let mut dl = ((r.rhs_pen - r.push * r.cfm) - chain(r.cross1, r.n1, bodies[ia].turn, bodies[ia].push) * r.jac_inv) - chain(r.cross2, r.n2, bodies[ib].turn, bodies[ib].push) * r.jac_inv;
    let mut sum = dl + r.push;
    if sum < r.lower {
        dl = r.lower - r.push;
        sum = r.lower;
    }
    r.push = sum;
    let d = dl * r.scale1;
    let a = &mut bodies[ia];
    if a.orig.is_some() {
        a.push.y = a.inv_mass.y * r.n1.y * d * a.lin_factor.y + a.push.y;
        a.push.z = a.inv_mass.z * r.n1.z * d * a.lin_factor.z + a.push.z;
        a.push.x = r.n1.x * a.inv_mass.x * d * a.lin_factor.x + a.push.x;
        a.turn.y = d * a.ang_factor.y * r.ang_a.y + a.turn.y;
        a.turn.z = d * a.ang_factor.z * r.ang_a.z + a.turn.z;
        a.turn.x = d * a.ang_factor.x * r.ang_a.x + a.turn.x;
    }
    let d = dl * r.scale2;
    let b = &mut bodies[ib];
    if b.orig.is_some() {
        b.push.y = b.inv_mass.y * r.n2.y * d * b.lin_factor.y + b.push.y;
        b.push.x = r.n2.x * b.inv_mass.x * d * b.lin_factor.x + b.push.x;
        b.push.z = b.inv_mass.z * r.n2.z * d * b.lin_factor.z + b.push.z;
        b.turn.y = d * b.ang_factor.y * r.ang_b.y + b.turn.y;
        b.turn.z = d * b.ang_factor.z * r.ang_b.z + b.turn.z;
        b.turn.x = d * b.ang_factor.x * r.ang_b.x + b.turn.x;
    }
}
