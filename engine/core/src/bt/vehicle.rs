use super::{Body, BodyId};
use crate::game::rb_state::quat_to_btmatrix;
use crate::math::crt::{cosf, sinf};
use crate::math::{Mat3, Quat, Vec3, v3};

#[derive(Clone, Debug, Default)]
pub struct WheelInfo {
    pub normal: Vec3,
    pub contact: Vec3,
    pub contact_velocity: Vec3,
    pub hard_point: Vec3,
    pub in_contact: bool,
    pub ground: Option<BodyId>,
    pub suspension_length: f32,
    pub min_hit_dist2: f32,
    pub basis: Mat3,
    pub connection: Vec3,
    pub radius: f32,
    pub rest: f32,
    pub travel: f32,
    pub max_raise: f32,
    pub stiffness: f32,
    pub damping_compression: f32,
    pub damping_relaxation: f32,
    pub steer: f32,
    pub engine_force: f32,
    pub brake: f32,
    pub force_scale: f32,
    pub up_offset: f32,
    pub push_force: f32,
    pub clipped_inv_dot: f32,
    pub relative_velocity: f32,
    pub suspension_force: f32,
    pub lat_friction: f32,
    pub long_friction: f32,
    pub extra_pushback: f32,
}

#[derive(Clone, Debug)]
pub struct Vehicle {
    pub friction_impulse: [Vec3; 4],
    pub contact_mask: u16,
    pub ray_group: u16,
    pub chassis: BodyId,

    pub keep_contacts: bool,
    pub wheels: [WheelInfo; 4],
}

pub trait VehicleWorld {
    fn body(&self, id: BodyId) -> &Body;
    fn body_mut(&mut self, id: BodyId) -> &mut Body;

    fn aabb_test(&self, min: Vec3, max: Vec3, visit: &mut dyn FnMut(BodyId, u16, u16));

    fn proxy_group(&self, id: BodyId) -> u16;

    fn ray_test(&self, from: Vec3, to: Vec3, object: BodyId) -> Option<RayHit>;
    fn cast_wheels(&self,wheels:&mut[WheelInfo;4],cands:&RayCandidates) {
        for w in wheels {wheel_cast(w,cands,self);}
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    pub fraction: f32,
    pub normal: Vec3,
    pub point: Vec3,
}

#[derive(Clone, Debug)]
pub struct RayCandidates {
    pub objects: [BodyId;32],
    pub len: usize,
}
impl Default for RayCandidates {
    fn default()->Self {Self{objects:[BodyId(0);32],len:0}}
}

#[inline]
pub fn wheel_ray(w: &mut WheelInfo, c: &Body) {
    let m = &c.transform.basis;
    let o = c.transform.origin;
    let (cx, cy, cz) = (w.connection.x, w.connection.y, w.connection.z);
    let up = m.col(2);
    let hy = cy * m.r[1].y + cx * m.r[1].x + cz * m.r[1].z + o.y;
    let len = w.rest + w.radius + w.travel;
    let hz = cy * m.r[2].y + cx * m.r[2].x + cz * m.r[2].z + o.z;
    let hx = cy * m.r[0].y + cx * m.r[0].x + up.x * cz + o.x;
    w.hard_point = v3(hx, hy, hz);
    w.contact = v3(hx - up.x * 1.0 * len, len * up.y * -1.0 + hy, len * up.z * -1.0 + hz);
    w.suspension_length = len;
}

pub fn ray_bounds(wheels: &[WheelInfo]) -> (Vec3, Vec3) {
    let (mut mn, mut mx) = (Vec3::ZERO, Vec3::ZERO);
    for (i, w) in wheels.iter().enumerate() {
        let (h, e) = (w.hard_point, w.contact);
        let lo = v3(if h.x < e.x { h.x } else { e.x }, if h.y < e.y { h.y } else { e.y }, if h.z < e.z { h.z } else { e.z });
        let hi = v3(if e.x < h.x { h.x } else { e.x }, if e.y < h.y { h.y } else { e.y }, if e.z < h.z { h.z } else { e.z });
        if i == 0 {
            mn = lo;
            mx = hi;
        } else {
            mn = v3(if lo.x <= mn.x { lo.x } else { mn.x }, if lo.y <= mn.y { lo.y } else { mn.y }, if lo.z <= mn.z { lo.z } else { mn.z });
            mx = v3(if mx.x <= hi.x { hi.x } else { mx.x }, if mx.y <= hi.y { hi.y } else { mx.y }, if mx.z <= hi.z { hi.z } else { mx.z });
        }
    }
    (mn, mx)
}

pub fn ray_candidates<W: VehicleWorld + ?Sized>(v: &Vehicle, world: &W) -> RayCandidates {
    let (min, max) = ray_bounds(&v.wheels);
    let chassis_group = world.proxy_group(v.chassis);
    let mut out = RayCandidates::default();
    let (ray_group, chassis) = (v.ray_group, v.chassis);
    world.aabb_test(min, max, &mut |id, group, mask| {
        if out.len < 32 && id != chassis && group & ray_group != 0 && mask & chassis_group != 0 {
            out.objects[out.len]=id;
            out.len+=1;
        }
    });
    out
}

pub fn wheel_cast<W: VehicleWorld + ?Sized>(w: &mut WheelInfo, cands: &RayCandidates, world: &W) {
    let (from, to) = (w.hard_point, w.contact);
    let mut best = 1.0f32;
    let mut hit: Option<(BodyId, Vec3, Vec3)> = None;
    for &obj in &cands.objects[..cands.len] {
        if let Some(h) = world.ray_test(from, to, obj) {
            if h.fraction < best {
                let (dx, dy, dz) = (h.point.x - from.x, h.point.y - from.y, h.point.z - from.z);
                if w.min_hit_dist2 <= dx * dx + dy * dy + dz * dz {
                    best = h.fraction;
                    hit = Some((obj, h.normal, h.point));
                }
            }
        }
    }
    match hit {
        Some((obj, n, p)) => {
            w.contact = p;
            w.normal = n;
            w.in_contact = true;
            w.ground = Some(obj);
        }
        None => {
            w.in_contact = false;
            w.ground = None;
        }
    }
}

pub fn wheel_keep(w: &mut WheelInfo, c: &Body) {
    let m = &c.transform.basis;
    let o = c.transform.origin;
    let (cx, cy, cz) = (w.connection.x, w.connection.y, w.connection.z);
    let hy = cy * m.r[1].y + cx * m.r[1].x + cz * m.r[1].z + o.y;
    let hz = cy * m.r[2].y + cx * m.r[2].x + cz * m.r[2].z + o.z;
    let hx = cy * m.r[0].y + cx * m.r[0].x + cz * m.r[0].z + o.x;
    w.hard_point = v3(hx, hy, hz);
    if w.in_contact {
        let len = w.rest + w.radius + w.travel;
        let d = (hy - w.contact.y) * w.normal.y + (hx - w.contact.x) * w.normal.x + (hz - w.contact.z) * w.normal.z;
        if 0.0 < d {
            let up = m.col(2);
            let (ax, ay, az) = ((hx - len * up.x) - hx, (hy - len * up.y) - hy, (hz - len * up.z) - hz);
            let k = -(w.normal.y * ay + w.normal.x * ax + w.normal.z * az);
            if d < k {
                let t = d / k;
                w.contact = v3(t * ax + hx, t * ay + hy, t * az + hz);
                return;
            }
        }
        w.in_contact = false;
        w.ground = None;
    }
}

pub fn contact_impulse(a: &Body, b: Option<&Body>, p: Vec3, n: Vec3, depth: f32) -> f32 {
    const ERP: f32 = f32::from_bits(0x3e4ccccd);
    const DT: f32 = f32::from_bits(0x3c888889);
    let rz = p.z - a.transform.origin.z;
    let rx = p.x - a.transform.origin.x;
    let ry = p.y - a.transform.origin.y;
    let (vbx, vby, vbz) = match b {
        Some(b) => {
            let (sx, sy, sz) = (p.x - b.transform.origin.x, p.y - b.transform.origin.y, p.z - b.transform.origin.z);
            (
                (b.ang_vel.y * sz - b.ang_vel.z * sy) + b.lin_vel.x,
                (b.ang_vel.z * sx - b.ang_vel.x * sz) + b.lin_vel.y,
                (b.ang_vel.x * sy - b.ang_vel.y * sx) + b.lin_vel.z,
            )
        }
        None => (0.0, 0.0, 0.0),
    };
    let (av, aw) = (a.lin_vel, a.ang_vel);
    let vrel = (((aw.z * rx - aw.x * rz) + av.y) - vby) * n.y + (((aw.y * rz - aw.z * ry) + av.x) - vbx) * n.x + (((aw.x * ry - aw.y * rx) + av.z) - vbz) * n.z;
    let cx = n.z * ry - n.y * rz;
    let cy = n.x * rz - n.z * rx;
    let cz = n.y * rx - n.x * ry;
    let i = &a.inv_inertia_world;
    let ix = cy * i.r[1].x + cx * i.r[0].x + cz * i.r[2].x;
    let iz = cy * i.r[1].z + cx * i.r[0].z + cz * i.r[2].z;
    let iy = cy * i.r[1].y + cx * i.r[0].y + cz * i.r[2].y;
    let kb = match b {
        None => 0.0,
        Some(b) => {
            let (sy, sz, sx) = (p.y - b.transform.origin.y, p.z - b.transform.origin.z, p.x - b.transform.origin.x);
            let dx = sy * n.z - sz * n.y;
            let dy = sz * n.x - sx * n.z;
            let dz = sx * n.y - sy * n.x;
            let j = &b.inv_inertia_world;
            let jx = dy * j.r[1].x + dx * j.r[0].x + dz * j.r[2].x;
            let jz = dy * j.r[1].z + dx * j.r[0].z + dz * j.r[2].z;
            let jy = dy * j.r[1].y + dx * j.r[0].y + dz * j.r[2].y;
            (jy * sz - jz * sy) * n.x + (jz * sx - jx * sz) * n.y + (jx * sy - jy * sx) * n.z + b.inv_mass
        }
    };
    let ka = (rz * iy - ry * iz) * n.x + (rx * iz - rz * ix) * n.y + (ry * ix - rx * iy) * n.z + a.inv_mass;
    let k = 1.0 / (ka + kb);
    let imp = k * -(-vrel * 0.0 + 1.0) * vrel + k * ((-depth * ERP) / DT);
    if 0.0 <= imp { imp } else { 0.0 }
}

pub fn wheel_result<W: VehicleWorld + ?Sized>(w: &mut WheelInfo, chassis: BodyId, contact_mask: u16, world: &mut W) {
    w.extra_pushback = 0.0;
    if !w.in_contact {
        let m = &world.body(chassis).transform.basis;
        w.normal = m.col(2);
        w.suspension_length = w.travel + w.rest;
        w.relative_velocity = 0.0;
        w.clipped_inv_dot = 1.0;
        return;
    }
    let dyn_ground = match w.ground {
        Some(g) if world.body(g).internal_type & 2 != 0 => Some(g),
        _ => None,
    };
    let limit = (w.radius + w.rest) - w.max_raise;
    let up = world.body(chassis).transform.basis.col(2);
    let dist = ((w.contact.x - w.hard_point.x) * up.x * -1.0 - (w.contact.y - w.hard_point.y) * up.y * 1.0) - (w.contact.z - w.hard_point.z) * up.z * 1.0;
    w.suspension_length = dist - w.radius;
    if dist < limit {
        let allowed = match dyn_ground {
            None => true,
            Some(g) => contact_mask & world.proxy_group(g) != 0,
        };
        if allowed {
            let imp = {
                let a = world.body(chassis);
                let b = dyn_ground.map(|g| world.body(g));
                contact_impulse(a, b, w.contact, w.normal, dist - limit)
            };
            w.extra_pushback = imp * 0.25;
            if let Some(g) = dyn_ground {
                let b = world.body_mut(g);
                let o = b.transform.origin;
                let rel = v3(w.contact.x - o.x, w.contact.y - o.y, w.contact.z - o.z);
                let imp = -imp;
                crate::game::rb_state::apply_impulse(b, v3(imp * w.normal.x, imp * w.normal.y, imp * w.normal.z), rel);
            }
        }
    }
    if let Some(g) = dyn_ground {
        let b = world.body_mut(g);
        crate::game::rb_state::activate(b);
        if 0.0 < w.push_force {
            let o = b.transform.origin;
            let rel = v3(w.contact.x - o.x, w.contact.y - o.y, w.contact.z - o.z);
            let p = -w.push_force;
            crate::game::rb_state::apply_impulse(b, v3(p * w.normal.x, p * w.normal.y, p * w.normal.z), rel);
        }
    }
    let a = world.body(chassis);
    let o = a.transform.origin;
    let (rz, ry, rx) = (w.contact.z - o.z, w.contact.y - o.y, w.contact.x - o.x);
    let up = a.transform.basis.col(2);
    let d = (up.x * -1.0 * w.normal.x - up.y * 1.0 * w.normal.y) - up.z * 1.0 * w.normal.z;
    let vy = (a.ang_vel.z * rx - a.ang_vel.x * rz) + a.lin_vel.y;
    let vx = (a.ang_vel.y * rz - a.ang_vel.z * ry) + a.lin_vel.x;
    let vz = (a.ang_vel.x * ry - a.ang_vel.y * rx) + a.lin_vel.z;
    w.contact_velocity = v3(vx, vy, vz);
    if d < -0.1 {
        let inv = -1.0 / d;
        w.relative_velocity = (w.normal.y * vy + w.normal.x * vx + vz * w.normal.z) * inv;
        w.clipped_inv_dot = inv;
    } else {
        w.relative_velocity = 0.0;
        w.clipped_inv_dot = 10.0;
    }
}

pub fn suspension(wheels: &mut [WheelInfo], c: &mut Body, dt: f32) {
    for w in wheels.iter_mut() {
        if !w.in_contact {
            w.suspension_force = 0.0;
        } else {
            let d = if w.relative_velocity < 0.0 { w.damping_compression } else { w.damping_relaxation };
            let f = (((w.rest - w.suspension_length) * w.stiffness * w.clipped_inv_dot + 0.0) - w.relative_velocity * d) * w.force_scale;
            w.suspension_force = if f < 0.0 { 0.0 } else { f };
        }
    }
    for w in wheels.iter() {
        if w.suspension_force != 0.0 {
            let o = c.transform.origin;
            let (rx, ry, rz) = (w.contact.x - o.x, w.contact.y - o.y, w.contact.z - o.z);
            let imp = w.suspension_force * dt + w.extra_pushback;
            let n = w.normal;
            let im = c.inv_mass;
            if im != 0.0 {
                let lf = c.lin_factor;
                c.lin_vel.y = imp * n.y * lf.y * im + c.lin_vel.y;
                c.lin_vel.z = imp * n.z * lf.z * im + c.lin_vel.z;
                c.lin_vel.x = imp * n.x * lf.x * im + c.lin_vel.x;
                let fz = imp * n.z * lf.z;
                let fy = imp * n.y * lf.y;
                let fx = imp * n.x * lf.x;
                let ty = fx * rz - fz * rx;
                let tx = fz * ry - fy * rz;
                let tz = fy * rx - fx * ry;
                let i = &c.inv_inertia_world;
                let af = c.ang_factor;
                c.ang_vel.y = (ty * i.r[1].y + tx * i.r[1].x + tz * i.r[1].z) * af.y + c.ang_vel.y;
                c.ang_vel.z = (ty * i.r[2].y + tx * i.r[2].x + tz * i.r[2].z) * af.z + c.ang_vel.z;
                c.ang_vel.x = (ty * i.r[0].y + tx * i.r[0].x + tz * i.r[0].z) * af.x + c.ang_vel.x;
            }
        }
    }
}

pub fn side_impulse(c: &Body, p: Vec3, axis: Vec3) -> f32 {
    if axis.x * axis.x + axis.y * axis.y + axis.z * axis.z <= 1.1 {
        let o = c.transform.origin;
        let (rx, ry, rz) = (p.x - o.x, p.y - o.y, p.z - o.z);
        let m = &c.transform.basis;

        let ax = axis.z * ry - axis.y * rz;
        let ay = axis.x * rz - rx * axis.z;
        let az = rx * axis.y - axis.x * ry;
        let jx = ay * m.r[1].x + ax * m.r[0].x + az * m.r[2].x;
        let jy = ay * m.r[1].y + ax * m.r[0].y + az * m.r[2].y;
        let jz = ay * m.r[1].z + ax * m.r[0].z + az * m.r[2].z;
        let i = c.inv_inertia_local;
        let dx = jx * i.x;
        let dy = i.y * jy;
        let dz = i.z * jz;
        let diag = jy * dy + jx * dx + jz * dz + c.inv_mass;
        let (w, v) = (c.ang_vel, c.lin_vel);
        let rel = ((w.z * rx - w.x * rz) + v.y) * axis.y + ((w.y * rz - w.z * ry) + v.x) * axis.x + ((w.x * ry - w.y * rx) + v.z) * axis.z;
        rel * -0.2 * (1.0 / diag)
    } else {
        0.0
    }
}

pub fn friction(impulses: &mut [Vec3; 4], wheels: &[WheelInfo], c: &mut Body, dt: f32) {
    for (imp, w) in impulses.iter_mut().zip(wheels.iter()) {
        if w.ground.is_none() {
            *imp = Vec3::ZERO;
            continue;
        }
        let axle = w.basis.col(1);
        let side = side_impulse(c, w.contact, axle);
        let n = w.normal;
        let s = side * (dt * 60.0) * w.lat_friction;
        let (sx, sy, sz) = (s * axle.x, s * axle.y, s * axle.z);
        let lx = n.y * axle.z - n.z * axle.y;
        let ly = n.z * axle.x - n.x * axle.z;
        let lz = n.x * axle.y - n.y * axle.x;
        let k = 1.0 / (ly * ly + lx * lx + lz * lz).sqrt();
        let mut roll = 0.0f32;
        if w.engine_force == 0.0 {
            if w.brake != 0.0 {
                let v = w.contact_velocity;
                let cur = -(-(k * ly) * v.y + -(k * lx) * v.x + -(k * lz) * v.z) * f32::from_bits(0x42e37ab1);
                let b = w.brake;
                let t = if b < cur { b } else { cur };
                let lo = -b;
                roll = if lo > t { lo } else { t };
                roll = roll * (dt * 60.0);
            }
        } else {
            roll = w.engine_force * dt;
        }
        roll = roll * w.long_friction;
        *imp = v3(roll * -(k * lx) + sx, roll * -(k * ly) + sy, roll * -(k * lz) + sz);
    }
    for (imp, w) in impulses.iter().zip(wheels.iter()) {
        if w.ground.is_none() {
            continue;
        }
        let o = c.transform.origin;
        let (px, py, pz) = (w.contact.x - o.x, w.contact.y - o.y, w.contact.z - o.z);
        let up = c.transform.basis.col(2);
        let proj = up.y * py + up.x * px + up.z * pz;
        let off = w.up_offset;
        let rx = (px - up.x * proj) - up.x * off;
        let ry = (py - up.y * proj) - up.y * off;
        let rz = (pz - up.z * proj) - up.z * off;
        let im = c.inv_mass;
        if im != 0.0 {
            let lf = c.lin_factor;
            c.lin_vel.z = imp.z * lf.z * im + c.lin_vel.z;
            c.lin_vel.y = lf.y * imp.y * im + c.lin_vel.y;
            c.lin_vel.x = imp.x * lf.x * im + c.lin_vel.x;
            let fy = imp.y * lf.y;
            let fz = lf.z * imp.z;
            let fx = imp.x * lf.x;
            let tx = ry * fz - rz * fy;
            let tz = rx * fy - ry * fx;
            let ty = rz * fx - rx * fz;
            let i = &c.inv_inertia_world;
            let af = c.ang_factor;
            c.ang_vel.y = (ty * i.r[1].y + tx * i.r[1].x + tz * i.r[1].z) * af.y + c.ang_vel.y;
            c.ang_vel.x = (ty * i.r[0].y + tx * i.r[0].x + tz * i.r[0].z) * af.x + c.ang_vel.x;
            c.ang_vel.z = (ty * i.r[2].y + tx * i.r[2].x + tz * i.r[2].z) * af.z + c.ang_vel.z;
        }
    }
}

pub fn wheel_basis(steer: f32, c: &Body) -> Mat3 {
    if steer == 0.0 {
        return c.transform.basis;
    }
    let m = &c.transform.basis;
    let half = steer * 0.5;
    let up = m.col(2);
    let s = sinf(half);
    let len = (up.y * up.y + up.x * up.x + up.z * up.z).sqrt();
    let k = s / len;
    let q = Quat { x: up.x * k, y: up.y * k, z: up.z * k, w: cosf(half) };
    let t = quat_to_btmatrix(q);
    let (r0, r1, r2) = (m.r[0], m.r[1], m.r[2]);
    Mat3 {
        r: [
            v3(r1.x * t.r[0].y + r0.x * t.r[0].x + r2.x * t.r[0].z, r1.y * t.r[0].y + r0.y * t.r[0].x + r2.y * t.r[0].z, r1.z * t.r[0].y + r0.z * t.r[0].x + r2.z * t.r[0].z),
            v3(r0.x * t.r[1].x + r1.x * t.r[1].y + r2.x * t.r[1].z, r1.y * t.r[1].y + r0.y * t.r[1].x + r2.y * t.r[1].z, r1.z * t.r[1].y + r0.z * t.r[1].x + r2.z * t.r[1].z),
            v3(r1.x * t.r[2].y + r0.x * t.r[2].x + r2.x * t.r[2].z, r1.y * t.r[2].y + r0.y * t.r[2].x + r2.y * t.r[2].z, r1.z * t.r[2].y + r0.z * t.r[2].x + r2.z * t.r[2].z),
        ],
    }
}

pub fn update_wheel_bases(wheels: &mut [WheelInfo; 4], c: &Body) {
    let m0 = wheel_basis(wheels[0].steer, c);
    wheels[0].basis = m0;
    wheels[1].basis = m0;
    wheels[3].basis = c.transform.basis;
    wheels[2].basis = c.transform.basis;
}

pub fn set_force_scales(wheels: &mut [WheelInfo; 4], inv_mass: f32) {
    let m = 1.0 / inv_mass;
    let x0 = wheels[0].connection.x;
    let x2 = wheels[2].connection.x;
    let (back, front);
    if x0 < 0.0 || -x2 < 0.0 {
        back = m * 0.5;
        front = back;
    } else {
        back = (x0 / (-x2 + x0)) * m;
        front = m - back;
    }
    wheels[0].force_scale = front * 0.5;
    wheels[1].force_scale = front * 0.5;
    wheels[2].force_scale = back * 0.5;
    wheels[3].force_scale = back * 0.5;
}

pub fn update_rays_and_contacts<W: VehicleWorld + ?Sized>(v: &mut Vehicle, world: &mut W) {
    if !v.keep_contacts {
        {
            let c = world.body(v.chassis);
            for w in v.wheels.iter_mut() {
                wheel_ray(w, c);
            }
        }
        let cands = ray_candidates(v, world);
        world.cast_wheels(&mut v.wheels,&cands);
    } else {
        let c = world.body(v.chassis);
        for w in v.wheels.iter_mut() {
            wheel_keep(w, c);
        }
    }
    let (chassis, mask) = (v.chassis, v.contact_mask);
    for w in v.wheels.iter_mut() {
        wheel_result(w, chassis, mask, world);
    }
}

pub fn update_suspension_and_friction<W: VehicleWorld + ?Sized>(v: &mut Vehicle, world: &mut W, dt: f32) {
    let c = world.body_mut(v.chassis);
    suspension(&mut v.wheels, c, dt);
    friction(&mut v.friction_impulse, &v.wheels, c, dt);
}

pub fn update_vehicle<W: VehicleWorld + ?Sized>(v: &mut Vehicle, world: &mut W, dt: f32, callback: impl FnOnce(&mut Vehicle, &mut W, f32)) {
    update_rays_and_contacts(v, world);
    callback(v, world, dt);
    update_suspension_and_friction(v, world, dt);
}
