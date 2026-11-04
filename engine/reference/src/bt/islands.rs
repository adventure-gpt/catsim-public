use super::types::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UfElement {
    pub id: i32,
    pub sz: i32,
}

#[derive(Clone, Debug, Default)]
pub struct IslandManager {
    pub uf: Vec<UfElement>,
    pub island_manifolds: Vec<ManifoldId>,
    pub island_bodies: Vec<BodyId>,
}

impl IslandManager {
    pub fn new() -> IslandManager {
        IslandManager::default()
    }

    #[inline]
    fn find(&mut self, mut x: i32) -> i32 {
        while x != self.uf[x as usize].id {
            let p = self.uf[x as usize].id;
            self.uf[x as usize].id = self.uf[p as usize].id;
            x = self.uf[p as usize].id;
        }
        x
    }

    pub fn unite(&mut self, a: i32, b: i32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.uf[ra as usize].id = rb;
            self.uf[rb as usize].sz += self.uf[ra as usize].sz;
        }
    }

    pub fn update_activation_state(&mut self, bodies: &mut [Body], objects: &[BodyId], pairs: impl Iterator<Item = (BodyId, BodyId)>) {
        let mut n = 0i32;
        for id in objects {
            let b = &mut bodies[id.0 as usize];
            if b.flags & (CF_STATIC | CF_KINEMATIC) == 0 {
                b.island_tag = n;
                n += 1;
            }
            b.companion_id = -1;
            b.hit_fraction = 1.0;
        }
        self.uf.clear();
        for i in 0..n {
            self.uf.push(UfElement { id: i, sz: 1 });
        }
        for (a, b) in pairs {
            let (ba, bb) = (&bodies[a.0 as usize], &bodies[b.0 as usize]);
            if ba.flags & (CF_STATIC | CF_KINEMATIC | CF_NO_RESPONSE) == 0 && bb.flags & (CF_STATIC | CF_KINEMATIC | CF_NO_RESPONSE) == 0 {
                let (ta, tb) = (ba.island_tag, bb.island_tag);
                self.unite(ta, tb);
            }
        }
    }

    pub fn store_island_activation_state(&mut self, bodies: &mut [Body], objects: &[BodyId]) {
        let mut idx = 0i32;
        for (k, id) in objects.iter().enumerate() {
            if bodies[id.0 as usize].flags & (CF_STATIC | CF_KINEMATIC) == 0 {
                let root = self.find(idx);
                let b = &mut bodies[id.0 as usize];
                b.island_tag = root;
                self.uf[idx as usize].sz = k as i32;
                idx += 1;
                b.companion_id = -1;
            } else {
                let b = &mut bodies[id.0 as usize];
                b.island_tag = -1;
                b.companion_id = -2;
            }
        }
    }

    pub fn build_islands(&mut self, bodies: &mut [Body], objects: &[BodyId], manifolds: &[Manifold], array: &[ManifoldId]) {
        self.island_manifolds.clear();
        for i in 0..self.uf.len() {
            let r = self.find(i as i32);
            self.uf[i].id = r;
        }
        let n = self.uf.len();
        if n > 1 {
            sort_elements(&mut self.uf, 0, n as i32 - 1);
        }
        let mut start = 0usize;
        while start < n {
            let id = self.uf[start].id;
            let mut end = start + 1;
            while end < n && self.uf[end].id == id {
                end += 1;
            }
            let mut all_sleeping = true;
            for e in &self.uf[start..end] {
                let b = &bodies[objects[e.sz as usize].0 as usize];
                if b.island_tag == id && (b.activation_state == ACTIVE || b.activation_state == DISABLE_DEACTIVATION) {
                    all_sleeping = false;
                }
            }
            if all_sleeping {
                for k in start..end {
                    let oi = objects[self.uf[k].sz as usize].0 as usize;
                    if bodies[oi].island_tag == id {
                        bodies[oi].set_activation_state(ISLAND_SLEEPING);
                    }
                }
            }
            start = end;
        }
        for &mid in array {
            let m = &manifolds[mid.0 as usize];
            let (i0, i1) = (m.body0.0 as usize, m.body1.0 as usize);
            if bodies[i0].activation_state != ISLAND_SLEEPING || bodies[i1].activation_state != ISLAND_SLEEPING {
                if bodies[i0].flags & CF_KINEMATIC != 0 && bodies[i0].activation_state != ISLAND_SLEEPING && bodies[i0].flags & CF_NO_RESPONSE == 0 {
                    bodies[i1].activate(false);
                }
                if bodies[i1].flags & CF_KINEMATIC != 0 && bodies[i1].activation_state != ISLAND_SLEEPING && bodies[i1].flags & CF_NO_RESPONSE == 0 {
                    bodies[i0].activate(false);
                }
                if needs_response(&bodies[i0], &bodies[i1]) {
                    self.island_manifolds.push(mid);
                }
            }
        }
    }

    pub fn build_and_process_islands(&mut self, bodies: &mut [Body], objects: &[BodyId], manifolds: &mut [Manifold], array: &[ManifoldId], mut island: impl FnMut(&mut [Body], &mut [Manifold], &[BodyId], &[ManifoldId], i32)) {
        self.build_islands(bodies, objects, manifolds, array);
        let nm = self.island_manifolds.len();
        if nm > 1 {
            sort_manifolds(&mut self.island_manifolds, bodies, manifolds, 0, nm as i32 - 1);
        }
        let n = self.uf.len();
        let mut start = 0usize;
        let mut mstart = 0usize;
        while start < n {
            let id = self.uf[start].id;
            let mut sleeping = true;
            self.island_bodies.clear();
            let mut end = start;
            while end < n && self.uf[end].id == id {
                let oid = objects[self.uf[end].sz as usize];
                self.island_bodies.push(oid);
                let s = bodies[oid.0 as usize].activation_state;
                if s != ISLAND_SLEEPING && s != DISABLE_SIMULATION {
                    sleeping = false;
                }
                end += 1;
            }
            let mut mend = mstart;
            if mstart < nm && island_id(&manifolds[self.island_manifolds[mstart].0 as usize], bodies) == id {
                mend = mstart + 1;
                while mend < nm && island_id(&manifolds[self.island_manifolds[mend].0 as usize], bodies) == id {
                    mend += 1;
                }
            }
            if !sleeping {
                island(bodies, manifolds, &self.island_bodies, &self.island_manifolds[mstart..mend], id);
            }
            if mend != mstart {
                mstart = mend;
            }
            start = end;
        }
    }
}

#[inline]
pub fn needs_response(a: &Body, b: &Body) -> bool {
    if a.flags & CF_NO_RESPONSE != 0 || b.flags & CF_NO_RESPONSE != 0 {
        return false;
    }
    !(a.flags & (CF_STATIC | CF_KINEMATIC) != 0 && b.flags & (CF_STATIC | CF_KINEMATIC) != 0)
}

#[inline]
pub fn island_id(m: &Manifold, bodies: &[Body]) -> i32 {
    let t = bodies[m.body0.0 as usize].island_tag;
    if t < 0 { bodies[m.body1.0 as usize].island_tag } else { t }
}

pub fn sort_elements(a: &mut [UfElement], lo: i32, hi: i32) {
    let mut lo = lo;
    loop {
        let x = a[((lo + hi) / 2) as usize].id;
        let (mut i, mut j) = (lo, hi);
        loop {
            while a[i as usize].id < x {
                i += 1;
            }
            while x < a[j as usize].id {
                j -= 1;
            }
            if j < i {
                break;
            }
            a.swap(i as usize, j as usize);
            i += 1;
            j -= 1;
            if j < i {
                break;
            }
        }
        if lo < j {
            sort_elements(a, lo, j);
        }
        if hi <= i {
            return;
        }
        lo = i;
    }
}

pub fn sort_manifolds(a: &mut [ManifoldId], bodies: &[Body], manifolds: &[Manifold], lo: i32, hi: i32) {
    let key = |m: ManifoldId| island_id(&manifolds[m.0 as usize], bodies);
    let mut lo = lo;
    loop {
        let x = key(a[((lo + hi) / 2) as usize]);
        let (mut i, mut j) = (lo, hi);
        loop {
            while x > key(a[i as usize]) {
                i += 1;
            }
            while key(a[j as usize]) > x {
                j -= 1;
            }
            if j < i {
                break;
            }
            a.swap(i as usize, j as usize);
            i += 1;
            j -= 1;
            if j < i {
                break;
            }
        }
        if lo < j {
            sort_manifolds(a, bodies, manifolds, lo, j);
        }
        if hi <= i {
            return;
        }
        lo = i;
    }
}
