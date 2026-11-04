mod f {
    use fast as engine;
    include!("fields.rs");
}
mod r {
    use scalar as engine;
    include!("fields.rs");
}
use std::time::Instant;
fn next(s: &mut u64) -> u32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 32) as u32
}
fn unit(s: &mut u64) -> f32 {
    (next(s) >> 8) as f32 / 16777216.0
}
fn span(s: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(s)
}

fn ball_row(rng: &mut u64) -> [f32; 13] {
    let k = next(rng) % 8;
    let mut r = [0.0f32; 13];
    let (x, y, z);
    match k {

        0 | 1 => {
            x = span(rng, -4000.0, 4000.0);
            y = span(rng, -5000.0, 5000.0);
            z = span(rng, 100.0, 1950.0);
        }

        2 | 3 => {
            x = span(rng, -3900.0, 3900.0);
            y = span(rng, -4900.0, 4900.0);
            z = span(rng, 91.0, 96.0);
        }

        4 => {
            x = if next(rng) % 2 == 0 { span(rng, 3900.0, 4000.0) } else { span(rng, -4000.0, -3900.0) };
            y = span(rng, -5000.0, 5000.0);
            z = span(rng, 92.0, 1900.0);
        }
        5 => {
            x = span(rng, -3500.0, 3500.0);
            y = if next(rng) % 2 == 0 { span(rng, 4980.0, 5030.0) } else { span(rng, -5030.0, -4980.0) };
            z = span(rng, 92.0, 1900.0);
        }
        6 => {
            x = span(rng, -3000.0, 3000.0);
            y = span(rng, -4000.0, 4000.0);
            z = span(rng, 1900.0, 1952.0);
        }

        _ => {
            x = if next(rng) % 2 == 0 { span(rng, 2900.0, 4000.0) } else { span(rng, -4000.0, -2900.0) };
            y = if next(rng) % 2 == 0 { span(rng, 3900.0, 5600.0) } else { span(rng, -5600.0, -3900.0) };
            z = span(rng, 92.0, 900.0);
        }
    }
    r[0] = x;
    r[1] = y;
    r[2] = z;
    let still = next(rng) % 6 == 0;
    let roll = matches!(k, 2 | 3) && next(rng) % 2 == 0;
    let sp = [500.0, 2500.0, 6000.0][next(rng) as usize % 3];
    for i in 3..6 {
        r[i] = if still { 0.0 } else { span(rng, -sp, sp) };
    }
    if roll {
        r[5] = span(rng, -5.0, 5.0);
    }
    for i in 6..9 {
        r[i] = if still { 0.0 } else { span(rng, -7.0, 7.0) };
    }
    if next(rng) % 3 == 0 {
        let q = [span(rng, -1.0, 1.0), span(rng, -1.0, 1.0), span(rng, -1.0, 1.0), span(rng, -1.0, 1.0)];
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().max(1e-3);
        for i in 0..4 {
            r[9 + i] = q[i] / n;
        }
    } else {
        r[12] = 1.0;
    }
    r
}
macro_rules! arena {
    ($e:ident, $meshes:expr) => {{
        let mut a = $e::game::arena::Arena::new(&[], $meshes, $e::game::arena::RoundConfig { countdown_ticks: 0, post_goal_time: f32::MAX, match_seconds: None }).unwrap();
        a.start_at_drop(None);
        a.phase = $e::game::arena::Phase::Ended;
        a.ball_held_2d = false;
        a.scripted = true;
        a
    }};
}
macro_rules! seed {
    ($e:ident, $a:expr, $r:expr, $clear:expr) => {{
        let r = $r;
        let st = $e::game::rb_state::RbState {
            location: $e::math::v3(r[0], r[1], r[2]),
            velocity: $e::math::v3(r[3], r[4], r[5]),
            spin: $e::math::v3(r[6], r[7], r[8]),
            quat: $e::math::Quat { x: r[9], y: r[10], z: r[11], w: r[12] },
            sleeping: false,
        };
        let a = &mut $a;
        a.phase = $e::game::arena::Phase::Ended;
        a.ball_held_2d = false;
        a.explosion = None;
        a.set_ball_state(&st, false);
        if let Some(b) = a.ball.as_mut() {
            b.first_frame = false;
            b.old_state = st;
        }
        if $clear {
            let ids = a.world.cw.manifolds().to_vec();
            for id in ids {
                a.world.cw.dispatcher.clear_manifold(id);
            }
        }
    }};
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("catsim_ring MESHES FRAMES [SEED=1] [MUTATE=none|lane|lane_predict|lane_read|nolanes]");
        std::process::exit(2);
    }
    let meshes = std::path::PathBuf::from(&args[1]);
    let frames: u64 = args[2].parse().unwrap();
    let seed: u64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);
    let mutate = args.get(4).map(|s| s.as_str()).unwrap_or("none");
    assert!(["none", "lane", "lane_predict", "lane_read", "nolanes"].contains(&mutate));
    if mutate == "nolanes" {
        fast::group::set_lanes(false);
    } else if !fast::group::lanes_on() {
        eprintln!("NOTE lanes unavailable on this CPU: the group runs without them");
    }
    let mut rng = seed;
    const N: usize = 48;
    let mut fa: Vec<_> = (0..N).map(|_| arena!(fast, &meshes)).collect();
    let mut ra: Vec<_> = (0..N).map(|_| arena!(scalar, &meshes)).collect();
    for k in 0..N {
        let row = ball_row(&mut rng);
        seed!(fast, fa[k], row, false);
        seed!(scalar, ra[k], row, false);
    }
    let (mut cf, mut cr) = (Vec::new(), Vec::new());
    let start = Instant::now();
    let mut ticks = 0u64;
    let mut done = 0u64;
    while done < frames {

        let mut order: Vec<usize> = (0..N).collect();
        for i in (1..N).rev() {
            let j = next(&mut rng) as usize % (i + 1);
            order.swap(i, j);
        }
        let mut at = 0;
        while at < N && done < frames {
            let g = (1 + next(&mut rng) as usize % 8).min(N - at);
            let ids = &order[at..at + g];
            at += g;
            let steps = 1 + (next(&mut rng) % 8) as i32;
            for &k in ids {
                if next(&mut rng) % 40 == 0 {
                    let row = ball_row(&mut rng);
                    let clear = next(&mut rng) % 2 == 0;
                    seed!(fast, fa[k], row, clear);
                    seed!(scalar, ra[k], row, clear);
                }
                let dbg = next(&mut rng) % 50 == 0;
                fa[k].debug_capture = dbg;
                ra[k].debug_capture = dbg;
            }
            if done == 300 {
                match mutate {
                    "lane" => fast::group::plant_lane_bug(2),
                    "lane_predict" => fast::group::plant_lane_bug(1),
                    "lane_read" => fast::group::plant_lane_bug(3),
                    _ => {}
                }
            }
            {
                let mut refs: Vec<&mut fast::game::arena::Arena> = Vec::with_capacity(g);
                let base = fa.as_mut_ptr();
                for &k in ids {

                    refs.push(unsafe { &mut *base.add(k) });
                }
                fast::group::frames(&mut refs, steps);
            }
            for &k in ids {
                ra[k].frame(&[], steps);
            }
            for &k in ids {
                f::capture_raw(&fa[k], &mut cf);
                r::capture_raw(&ra[k], &mut cr);
                if cf != cr {
                    let i = cf.iter().zip(&cr).position(|(x, y)| x != y).unwrap_or(cf.len().min(cr.len()));
                    eprintln!(
                        "DIFF seed={seed} frame={done} arena={k} group={g} steps={steps} word={i} fast={:?} scalar={:?} lengths={}/{}",
                        cf.get(i),
                        cr.get(i),
                        cf.len(),
                        cr.len()
                    );
                    std::process::exit(1);
                }
            }
            ticks += steps as u64 * g as u64;
            done += 1;
            if done % 200000 == 0 {
                println!("progress frames={done} arena_ticks={ticks} seconds={:.1}", start.elapsed().as_secs_f64());
            }
        }
    }
    println!(
        "PASS ring_group_frames={done} arena_ticks={ticks} seed={seed} lanes={} lane_stats(tried,stored: predict,integrate,read)={:?} mutation={mutate} seconds={:.3}",
        fast::group::lanes_on(),
        fast::group::lane_stats(),
        start.elapsed().as_secs_f64()
    );
}
