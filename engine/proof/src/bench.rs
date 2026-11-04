use std::time::Instant;
fn next(s: &mut u64) -> u32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 32) as u32
}
fn range(s: &mut u64, n: u32) -> f32 {
    (next(s) % n) as f32
}
macro_rules! bench_group {
    (fast, $balls:expr, $grp:expr, $rs:expr) => {{
        for chunk in $balls.chunks_mut($grp) {
            let mut refs: Vec<&mut fast::game::arena::Arena> = chunk.iter_mut().collect();
            fast::group::frames(&mut refs, $rs as i32);
        }
    }};
    (scalar, $balls:expr, $grp:expr, $rs:expr) => {{
        for a in $balls.iter_mut() {
            a.frame(&[], $rs as i32);
        }
    }};
}
macro_rules! bench {
    ($e:ident, $meshes:expr, $arenas:expr, $ticks:expr, $ring:expr, $ring_ticks:expr, $seed:expr, $rsteps:expr, $group:expr) => {{
        use $e as engine;
        let meshes: &std::path::Path = $meshes;
        let mut rng: u64 = $seed;
        let cfg = || engine::game::arena::RoundConfig { countdown_ticks: 2, post_goal_time: 0.2, match_seconds: None };

        let modes: Vec<usize> = (0..$arenas).map(|i| [1, 1, 1, 2, 3][i % 5]).collect();
        let mut slots: Vec<_> = modes
            .iter()
            .map(|&m| {
                let mut a = engine::adapter::ArenaSlot::new(m, "Octane", meshes, cfg()).unwrap();
                a.arena.rng = rng;
                a.arena.start_at_drop(None);
                a
            })
            .collect();
        let mut ctl = vec![[[0.0f32; 8]; 6]; $arenas];
        let mut car_ticks = 0u64;
        let start = Instant::now();
        for t in 0..$ticks {
            for (k, a) in slots.iter_mut().enumerate() {
                let mode = modes[k];
                if (t + k) % 8 == 0 {
                    for row in &mut ctl[k][..mode * 2] {
                        for (j, v) in row.iter_mut().enumerate() {
                            *v = if j < 5 { (range(&mut rng, 2001) - 1000.0) * 0.001 } else { (next(&mut rng) % 5 == 0) as u8 as f32 };
                        }
                    }
                }
                if (t + 37 * k) % 600 == 3 {
                    use engine::game::rb_state::RbState;
                    use engine::math::{v3, Quat};
                    let mut r = |n: u32, o: f32| range(&mut rng, n) - o;
                    let ball = RbState { location: v3(r(7201, 3600.0), r(9601, 4800.0), 100.0 + r(1700, 0.0)), velocity: v3(r(5001, 2500.0), r(5001, 2500.0), r(3001, 1500.0)), spin: v3(0.0, 0.0, 0.0), quat: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }, sleeping: false };
                    a.arena.set_ball_state(&ball, false);
                    for i in 0..mode * 2 {
                        let st = RbState { location: v3(r(7201, 3600.0), r(9601, 4800.0), 17.0), velocity: v3(r(3001, 1500.0), r(3001, 1500.0), 0.0), spin: v3(0.0, 0.0, 0.0), quat: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }, sleeping: false };
                        if let Some(c) = a.arena.players[i].car.as_mut() {
                            engine::game::rb_state::write_body(&mut a.arena.world.cw.bodies[c.body.0 as usize], &st);
                            c.state = st;
                            c.old_state = st;
                        }
                    }
                }
                let c: Vec<_> = ctl[k][..mode * 2].iter().map(|x| engine::adapter::input_from_row(x)).collect();
                a.step(&c);
                car_ticks += 1;
            }
        }
        let car_s = start.elapsed().as_secs_f64();

        let mut balls: Vec<_> = (0..$ring)
            .map(|_| {
                let mut a = engine::game::arena::Arena::new(&[], meshes, engine::game::arena::RoundConfig { countdown_ticks: 0, post_goal_time: f32::MAX, match_seconds: None }).unwrap();
                a.start_at_drop(None);
                a.phase = engine::game::arena::Phase::Ended;
                a.ball_held_2d = false;
                a.scripted = true;
                a
            })
            .collect();
        let start = Instant::now();
        let mut ball_ticks = 0u64;
        let (rs, grp) = ($rsteps as usize, $group as usize);
        for t in (0..$ring_ticks).step_by(rs) {
            for (k, a) in balls.iter_mut().enumerate() {
                if (t / rs + 13 * k) % (120 / rs).max(1) == 0 {
                    use engine::game::rb_state::RbState;
                    use engine::math::{v3, Quat};
                    let mut r = |n: u32, o: f32| range(&mut rng, n) - o;
                    let ball = RbState { location: v3(r(7201, 3600.0), r(9601, 4800.0), 93.0 + r(1900, 0.0)), velocity: v3(r(5001, 2500.0), r(5001, 2500.0), r(3001, 1500.0)), spin: v3(r(11, 5.0), r(11, 5.0), r(11, 5.0)), quat: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }, sleeping: false };
                    a.phase = engine::game::arena::Phase::Ended;
                    a.explosion = None;
                    a.set_ball_state(&ball, false);
                    if let Some(b) = a.ball.as_mut() {
                        b.first_frame = false;
                        b.old_state = ball;
                    }
                }
                a.phase = engine::game::arena::Phase::Ended;
                a.explosion = None;
                if grp <= 1 {
                    a.frame(&[], rs as i32);
                }
                ball_ticks += rs as u64;
            }
            if grp > 1 {
                bench_group!($e, &mut balls, grp, rs);
            }
        }
        let ball_s = start.elapsed().as_secs_f64();
        let mut sum = 0u64;
        for a in &slots {
            for b in &a.arena.world.cw.bodies {
                sum = sum.wrapping_mul(31).wrapping_add(b.transform.origin.x.to_bits() as u64);
            }
        }
        for a in &balls {
            for b in &a.world.cw.bodies {
                sum = sum.wrapping_mul(31).wrapping_add(b.transform.origin.x.to_bits() as u64);
            }
        }
        (car_ticks, car_s, ball_ticks, ball_s, sum)
    }};
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("catsim_bench MESHES [ARENAS=96] [TICKS=2400] [RING=96] [RING_TICKS=4800] [SEED=1] [ENGINE=fast|scalar]");
        std::process::exit(2);
    }
    let meshes = std::path::PathBuf::from(&a[1]);
    let g = |i: usize, d: usize| a.get(i).map(|s| s.parse().unwrap()).unwrap_or(d);
    let (arenas, ticks, ring, ring_ticks) = (g(2, 96), g(3, 2400), g(4, 96), g(5, 4800));
    let seed = g(6, 1) as u64;
    let engine = a.get(7).map(|s| s.as_str()).unwrap_or("fast");
    let (rsteps, group) = (g(8, 1), g(9, 1));
    let (ct, cs, bt, bs, sum) = if engine == "scalar" {
        bench!(scalar, &meshes, arenas, ticks, ring, ring_ticks, seed, rsteps, group)
    } else {
        bench!(fast, &meshes, arenas, ticks, ring, ring_ticks, seed, rsteps, group)
    };
    println!(
        "engine={engine} car_arena_ticks={ct} car_ns_per_arena_tick={:.0} ball_ticks={bt} ball_ns_per_tick={:.0} state_sum={sum:016x}",
        cs * 1e9 / ct.max(1) as f64,
        bs * 1e9 / bt.max(1) as f64
    );
    for (name, calls, ns) in fast::prof::stats() {
        if calls > 0 {
            println!("  stage {name:14} calls {calls:>10} ms {:>10.1} ns/call {:>8.1}", ns as f64 / 1e6, ns as f64 / calls as f64);
        }
    }
}
