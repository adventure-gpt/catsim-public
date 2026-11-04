use super::dispatch::{Algorithm, Dispatcher};
use super::sweep::Handle;

#[derive(Clone, Debug)]
pub struct Pair {
    pub proxy0: u16,
    pub proxy1: u16,
    pub algorithm: Option<Box<Algorithm>>,
}

#[derive(Clone, Debug)]
pub struct PairCache {
    pub pairs: Vec<Pair>,

    pub capacity: i32,
    pub hash_table: Vec<i32>,
    pub next: Vec<i32>,
}

impl Default for PairCache {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
pub(crate) fn pair_hash(id0: i32, id1: i32) -> i32 {
    let mut k: i32 = (id1 << 16) | id0;
    k = k.wrapping_add(!(k << 15));
    k = (k >> 10) ^ k;
    k = k.wrapping_mul(9);
    k = (k >> 6) ^ k;
    k = k.wrapping_add(!(k << 11));
    (k >> 16) ^ k
}

impl PairCache {

    pub fn new() -> PairCache {
        let mut c = PairCache { pairs: Vec::new(), capacity: 2, hash_table: Vec::new(), next: Vec::new() };
        c.grow_tables();
        c
    }

    fn grow_tables(&mut self) {
        let old = self.hash_table.len();
        let cap = self.capacity as usize;
        if old < cap {
            self.hash_table.clear();
            self.hash_table.resize(cap, -1);
            self.next.clear();
            self.next.resize(cap, -1);
            let mask = self.capacity - 1;
            for i in 0..old {
                let p = &self.pairs[i];
                let h = (pair_hash(p.proxy0 as i32, p.proxy1 as i32) & mask) as usize;
                self.next[i] = self.hash_table[h];
                self.hash_table[h] = i as i32;
            }
        }
    }

    #[inline(always)]
    fn filter(a: &Handle, b: &Handle) -> bool {
        (b.mask & a.group) != 0 && (b.group & a.mask) != 0
    }

    pub fn add_overlapping_pair(&mut self, handles: &[Handle], a: u16, b: u16) {
        if !Self::filter(&handles[a as usize], &handles[b as usize]) {
            return;
        }
        let (p0, p1) = if a <= b { (a, b) } else { (b, a) };
        let old_cap = self.capacity;
        let key = pair_hash(p0 as i32, p1 as i32);
        let mut hash = key & (old_cap - 1);
        let mut i = self.hash_table[hash as usize];
        while i != -1 {
            let p = &self.pairs[i as usize];
            if p.proxy0 == p0 && p.proxy1 == p1 {
                return;
            }
            i = self.next[i as usize];
        }
        let index = self.pairs.len() as i32;
        if index == self.capacity {
            let want = if index == 0 { 1 } else { index * 2 };
            if self.capacity < want {
                self.capacity = want;
            }
        }
        self.pairs.push(Pair { proxy0: p0, proxy1: p1, algorithm: None });
        if old_cap < self.capacity {
            self.grow_tables();
            hash = key & (self.capacity - 1);
        }
        self.next[index as usize] = self.hash_table[hash as usize];
        self.hash_table[hash as usize] = index;
    }

    pub fn find_pair(&self, a: u16, b: u16) -> Option<usize> {
        let (p0, p1) = if a <= b { (a, b) } else { (b, a) };
        let hash = pair_hash(p0 as i32, p1 as i32) & (self.capacity - 1);
        if hash as usize >= self.hash_table.len() {
            return None;
        }
        let mut i = self.hash_table[hash as usize];
        while i != -1 {
            let p = &self.pairs[i as usize];
            if p.proxy0 == p0 && p.proxy1 == p1 {
                return Some(i as usize);
            }
            i = self.next[i as usize];
        }
        None
    }

    pub fn remove_overlapping_pair(&mut self, a: u16, b: u16, disp: &mut Dispatcher) {
        let (p0, p1) = if a <= b { (a, b) } else { (b, a) };
        let hash = pair_hash(p0 as i32, p1 as i32) & (self.capacity - 1);
        let mut i = self.hash_table[hash as usize];
        while i != -1 {
            let p = &self.pairs[i as usize];
            if p.proxy0 == p0 && p.proxy1 == p1 {
                break;
            }
            i = self.next[i as usize];
        }
        if i == -1 {
            return;
        }
        let pair_index = i;
        self.clean_overlapping_pair(pair_index as usize, disp);

        let mut index = self.hash_table[hash as usize];
        let mut previous = -1;
        while index != pair_index {
            previous = index;
            index = self.next[index as usize];
        }
        if previous != -1 {
            self.next[previous as usize] = self.next[pair_index as usize];
        } else {
            self.hash_table[hash as usize] = self.next[pair_index as usize];
        }
        let last = self.pairs.len() as i32 - 1;
        if last != pair_index {

            let l = &self.pairs[last as usize];
            let last_hash = pair_hash(l.proxy0 as i32, l.proxy1 as i32) & (self.capacity - 1);
            let mut index = self.hash_table[last_hash as usize];
            let mut previous = -1;
            while index != last {
                previous = index;
                index = self.next[index as usize];
            }
            if previous != -1 {
                self.next[previous as usize] = self.next[last as usize];
            } else {
                self.hash_table[last_hash as usize] = self.next[last as usize];
            }
            self.pairs.swap(pair_index as usize, last as usize);
            self.next[pair_index as usize] = self.hash_table[last_hash as usize];
            self.hash_table[last_hash as usize] = pair_index;
        }
        self.pairs.pop();
    }

    pub fn clean_overlapping_pair(&mut self, index: usize, disp: &mut Dispatcher) {
        if let Some(a) = self.pairs[index].algorithm.take() {
            disp.free_algorithm(*a);
        }
    }

    pub fn remove_overlapping_pairs_containing_proxy(&mut self, proxy: u16, disp: &mut Dispatcher) {
        let mut i = 0;
        while i < self.pairs.len() {
            let (p0, p1) = (self.pairs[i].proxy0, self.pairs[i].proxy1);
            if p0 == proxy || p1 == proxy {
                self.remove_overlapping_pair(p0, p1, disp);
            } else {
                i += 1;
            }
        }
    }

    pub fn clean_proxy_from_pairs(&mut self, proxy: u16, disp: &mut Dispatcher) {
        for i in 0..self.pairs.len() {
            if self.pairs[i].proxy0 == proxy || self.pairs[i].proxy1 == proxy {
                self.clean_overlapping_pair(i, disp);
            }
        }
    }
}
