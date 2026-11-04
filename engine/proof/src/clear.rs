fn next(s: &mut u64) -> u32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 32) as u32
}
fn unit(s: &mut u64) -> f32 {
    next(s) as f32 / 4294967296.0
}
fn sym(s: &mut u64, w: f32) -> f32 {
    (unit(s) * 2.0 - 1.0) * w
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = std::path::Path::new(&a[1]);
    let count: u64 = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(1_000_000);
    let mut seed: u64 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);
    let mutate = a.get(4).is_some_and(|s| s == "mutate");
    let t0 = std::time::Instant::now();
    let mut f = fast::bt::collision_world::CollisionWorld::new();
    f.add_soccar_statics(path).unwrap();
    f.ensure_statics();
    let g = f.clear_grid().expect("no clear grid built");
    let built = t0.elapsed().as_secs_f64();
    let (lo, n) = g.extent();
    println!("grid lo={lo:?} n={n:?} marked={} of {} triangles={} max_tau={:.4} build_s={built:.2}", g.stats.0, n[0] * n[1] * n[2], g.stats.1, g.stats.2);
    let g = if mutate { std::sync::Arc::new(g.with_bug()) } else { g };
    let mut s = scalar::bt::collision_world::CollisionWorld::new();
    s.add_soccar_statics(path).unwrap();
    let meshes: Vec<(usize, std::sync::Arc<fast::bt::bvh::MeshShape>)> = (0..f.bodies.len())
        .filter_map(|i| match &f.shapes[f.bodies[i].shape.0 as usize] {
            fast::bt::shapes::Shape::Mesh(m) => Some((i, m.clone())),
            _ => None,
        })
        .collect();
    let world = |i: usize, p: fast::math::Vec3| {
        let t = &f.bodies[i].transform;
        let r = &t.basis.r;
        fast::math::v3(r[0].x * p.x + r[0].y * p.y + r[0].z * p.z + t.origin.x, r[1].x * p.x + r[1].y * p.y + r[1].z * p.z + t.origin.y, r[2].x * p.x + r[2].y * p.y + r[2].z * p.z + t.origin.z)
    };
    let (mut clear, mut hits_near) = (0u64, 0u64);
    for case in 0..count {
        let mode = case % 5;
        let p = match mode {
            0 => fast::math::v3(lo[0] + unit(&mut seed) * n[0] as f32 * 0.5, lo[1] + unit(&mut seed) * n[1] as f32 * 0.5, lo[2] + unit(&mut seed) * n[2] as f32 * 0.5),
            1 | 2 => {
                let (i, m) = &meshes[next(&mut seed) as usize % meshes.len()];
                let tri = m.triangle((next(&mut seed) as usize % m.triangles.len()) as i32);
                let (u, v) = (unit(&mut seed), unit(&mut seed));
                let (u, v) = if u + v > 1.0 { (1.0 - u, 1.0 - v) } else { (u, v) };
                let q = fast::math::v3(
                    tri[0].x + u * (tri[1].x - tri[0].x) + v * (tri[2].x - tri[0].x),
                    tri[0].y + u * (tri[1].y - tri[0].y) + v * (tri[2].y - tri[0].y),
                    tri[0].z + u * (tri[1].z - tri[0].z) + v * (tri[2].z - tri[0].z),
                );
                let w = world(*i, if mode == 1 { tri[(next(&mut seed) % 3) as usize] } else { q });
                let k = if mode == 1 { 1.5 } else { 0.8 };
                fast::math::v3(w.x + sym(&mut seed, k), w.y + sym(&mut seed, k), w.z + sym(&mut seed, k))
            }
            3 => fast::math::v3(sym(&mut seed, 85.0), sym(&mut seed, 125.0), if next(&mut seed) % 2 == 0 { sym(&mut seed, 1.0) } else { 40.96 + sym(&mut seed, 1.0) }),
            _ => fast::math::v3(if next(&mut seed) % 2 == 0 { 81.92 + sym(&mut seed, 1.0) } else { -81.92 + sym(&mut seed, 1.0) }, sym(&mut seed, 125.0), sym(&mut seed, 42.0) + 20.0),
        };
        let l = unit(&mut seed) * if case % 7 == 0 { 3.0 } else { 1.2 };
        let d = fast::math::v3(sym(&mut seed, 1.0), sym(&mut seed, 1.0), sym(&mut seed, 1.0));
        let dl = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt().max(1e-6);
        let q = fast::math::v3(p.x + d.x / dl * l, p.y + d.y / dl * l, p.z + d.z / dl * l);
        let (pa, qa) = if case % 2 == 0 { (p, q) } else { (q, p) };
        if !g.segment_clear(pa, qa) {
            continue;
        }
        clear += 1;
        for id in 0..s.bodies.len() {
            let h = s.ray_test_object(scalar::math::v3(pa.x, pa.y, pa.z), scalar::math::v3(qa.x, qa.y, qa.z), scalar::bt::BodyId(id as u32));
            if h.body.is_some() {
                eprintln!("DIFF case={case} mode={mode} static={id} from={pa:?} to={qa:?} fraction={}", h.closest);
                std::process::exit(1);
            }
        }
        if mode == 2 {
            hits_near += 1;
        }
    }
    println!("CLEAR_OK cases={count} clear={clear} clear_near_triangles={hits_near}");
}
