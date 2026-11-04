use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
mod f {
    use fast as engine;
    include!("fields.rs");
}
mod r {
    use scalar as engine;
    include!("fields.rs");
}
fn next(s: &mut u64) -> u32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 32) as u32
}
fn range(s: &mut u64, n: u32) -> f32 {
    (next(s) % n) as f32
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("catsim_fuzz MESHES TICKS [THREADS=4] [SEED=1] [MUTATE=none|velocity|event|statics|farbox]");
        std::process::exit(2);
    }
    let meshes = std::path::PathBuf::from(&args[1]);
    fast::data::load(&meshes).unwrap();
    scalar::data::load(&meshes).unwrap();
    let ticks: u64 = args[2].parse().unwrap();
    let threads: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(4);
    assert!((1..=4).contains(&threads));
    let seed: u64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1);
    let mutate = args.get(5).map(|s| s.as_str()).unwrap_or("none");
    assert!(["none", "velocity", "event", "statics", "farbox", "rng", "rng-fault", "rng-mesh", "pad-map"].contains(&mutate));
    if mutate == "farbox" {
        fast::bt::algo_concave::PLANT_FAR_BUG.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let count = AtomicU64::new(0);
    let start = Instant::now();
    std::thread::scope(|scope| {
        for worker in 0..threads {
            let count = &count;
            let meshes = &meshes;
            scope.spawn(move||{
                let quota=ticks/threads as u64+u64::from((worker as u64)<ticks%threads as u64);
                let mut done=0;let mut rng=seed.wrapping_add(worker as u64*1000003);let mut ep=0;
                let(mut fa,mut ra)=(Vec::new(),Vec::new());
                while done<quota{
                    let mode=1+(next(&mut rng)%3) as usize;

                    let presets=&fast::game::config::PRESETS;let bodies=&fast::game::config::BODIES;
                    let mut pick=|rng:&mut u64|{let k=next(rng) as usize%(presets.len()+bodies.len());if k<presets.len(){presets[k].name}else{bodies[k-presets.len()].name}};
                    let name:String=if ep%2==0{pick(&mut rng).to_string()}else{(0..mode*2).map(|_|pick(&mut rng)).collect::<Vec<_>>().join(",")};
                    let name=name.as_str();
                    let fl = if mutate == "pad-map" { fast::game::pads::PadLayout::UtopiaNight } else { fast::game::pads::PadLayout::Stadium };
                    let sl = if mutate == "pad-map" { scalar::game::pads::PadLayout::UtopiaNight } else { scalar::game::pads::PadLayout::Stadium };
                    let mut a=fast::adapter::ArenaSlot::new_with_layout(mode,name,meshes,fast::game::arena::RoundConfig{countdown_ticks:2,post_goal_time:0.2,match_seconds:None},fl).unwrap();
                    let mut b=scalar::adapter::ArenaSlot::new_with_layout(mode,name,meshes,scalar::game::arena::RoundConfig{countdown_ticks:2,post_goal_time:0.2,match_seconds:None},sl).unwrap();
                    a.arena.rng=rng;b.arena.rng=rng;
                    a.arena.start_at_drop(None);b.arena.start_at_drop(None);
                    if mutate.starts_with("rng") {
                        macro_rules! seed_rng {($s:ident,$e:ident)=>{{
                            $s.arena.crt_state = Some(rng as u32);
                            $s.arena.frame_rng = Some($e::game::frame_rng::FrameRng::new(32.0,
                                (0..mode*2).filter(|_| mutate != "rng-mesh").map(|i| $e::game::frame_rng::LockChain {
                                    release: 32.0+i as f32*0.1, time: 0.75, random_time: 0.5, running: true,
                                }).collect(), true));

                            if mutate == "rng-mesh" {
                                let cars: Vec<usize> = (0..mode*2).collect();
                                $s.arena.frame_rng.as_mut().unwrap().start_mesh($e::game::frame_rng::MeshConfig {
                                    wheels: 4, extras: 1, map: 5, goal_mesh: true, goal: (3, 7), demolish: (4, 6),
                                    reset: (2, 7), respawn: (2, 3) }, &cars);
                            }
                        }}}
                        seed_rng!(a,fast);seed_rng!(b,scalar);

                        macro_rules! seed_countdown {($s:ident,$e:ident)=>{{
                            $s.arena.frame_rng.as_mut().unwrap().countdown_entry = true;
                            $s.arena.countdown_clock = Some($e::game::round::CountdownClock::with_warmup(3, 3, 3));
                            $s.arena.rule_countdown = true;
                        }}}
                        seed_countdown!(a,fast);seed_countdown!(b,scalar);
                    }
                    let episode=if ep%17==0{36000}else{1200};
                    let mut ctl=[[0.0;8];6];
                    let mut seed_rows=[[0.0;12];7];
                    for row in &mut seed_rows{
                        row[0]=range(&mut rng,7201)-3600.0;row[1]=range(&mut rng,9601)-4800.0;
                        row[2]=100.0+range(&mut rng,1700);row[3]=range(&mut rng,5001)-2500.0;
                        row[4]=range(&mut rng,5001)-2500.0;row[5]=range(&mut rng,3001)-1500.0;
                        for v in &mut row[6..9]{*v=(range(&mut rng,1001)-500.0)*0.01;}
                    }
                    for t in 0..episode{
                        if done>=quota{break;}
                        if t%8==0{for row in &mut ctl[..mode*2]{for (j,v) in row.iter_mut().enumerate(){*v=if j<5{(range(&mut rng,2001)-1000.0)*0.001}else{(next(&mut rng)%5==0) as u8 as f32};}}}

                        if t==3 && ep%4!=0{
                            macro_rules! set {($s:ident,$e:ident)=>{{
                                let row=&seed_rows[6];
                                $s.arena.set_ball_state(&$e::game::rb_state::RbState{location:$e::math::v3(row[0],row[1],row[2]),velocity:$e::math::v3(row[3],row[4],row[5]),spin:$e::math::v3(row[6],row[7],row[8]),quat:$e::math::Quat{x:0.0,y:0.0,z:0.0,w:1.0},sleeping:false},false);
                                for i in 0..mode*2{
                                    let row=&seed_rows[i];let c=$s.arena.players[i].car.as_mut().unwrap();
                                    let st=$e::game::rb_state::RbState{location:$e::math::v3(if ep%4==1{(i as f32)*90.0}else{row[0]},if ep%4==1{0.0}else{row[1]},if ep%4==2{row[2]}else{17.0}),velocity:$e::math::v3(row[3],row[4],row[5]),spin:$e::math::v3(row[6],row[7],row[8]),quat:$e::math::Quat{x:0.0,y:0.0,z:0.0,w:1.0},sleeping:false};
                                    $e::game::rb_state::write_body(&mut $s.arena.world.cw.bodies[c.body.0 as usize],&st);c.state=st;c.old_state=st;
                                }
                            }}}
                            set!(a,fast);set!(b,scalar);
                        }
                        let ac:Vec<_>=ctl[..mode*2].iter().map(|x|fast::adapter::input_from_row(x)).collect();
                        let bc:Vec<_>=ctl[..mode*2].iter().map(|x|scalar::adapter::input_from_row(x)).collect();
                        if mutate.starts_with("rng") {
                            let steps = (t % 5) as i32;
                            let dt = 1.0f32 / 240.0 + steps as f32 / 120.0;
                            let alpha = (t % 17) as f32 / 17.0;
                            a.step_frame_with_time(&ac,steps,dt,alpha);b.step_frame_with_time(&bc,steps,dt,alpha);
                        } else { a.step(&ac);b.step(&bc); }
                        if done==64 && mutate=="rng-fault"{a.arena.crt_state=a.arena.crt_state.map(|s|s^1);}
                        if done==64 && mutate=="velocity"{let id=a.arena.world.rigid_bodies[0];let x=&mut a.arena.world.cw.bodies[id.0 as usize].lin_vel.x;*x=f32::from_bits(x.to_bits()^1);}
                        if done==64 && mutate=="event"{a.events[0][0]=1.0-a.events[0][0];}
                        if done==64 && mutate=="statics"{a.arena.world.cw.plant_statics_bug();}
                        if let Err(e)=a.arena.world.cw.verify_statics(){eprintln!("DIFF static invariant worker={worker} seed={seed} episode={ep} tick={t}: {e}");std::process::exit(1);}
                        f::capture(&a,&mut fa);r::capture(&b,&mut ra);
                        if fa!=ra{
                            let i=fa.iter().zip(&ra).position(|(x,y)|x!=y).unwrap_or(fa.len().min(ra.len()));
                            eprintln!("DIFF worker={worker} seed={seed} episode={ep} tick={t} mode={mode} preset={name} word={i} fast={:?} scalar={:?} lengths={}/{}",fa.get(i),ra.get(i),fa.len(),ra.len());std::process::exit(1);
                        }
                        done+=1;
                        if done%10000==0{count.fetch_add(10000,Ordering::Relaxed);}
                    }
                    ep+=1;
                    if ep%25==0{let n=count.load(Ordering::Relaxed);println!("progress compared_ticks={n} seconds={:.1} ticks_per_second={:.0}",start.elapsed().as_secs_f64(),n as f64/start.elapsed().as_secs_f64());}
                }
                count.fetch_add(done%10000,Ordering::Relaxed);
            });
        }
    });
    let n = count.load(Ordering::Relaxed);
    assert_eq!(n, ticks);
    println!(
        "PASS compared_arena_ticks={n} threads={threads} seed={seed} seconds={:.3} mutation={mutate}",
        start.elapsed().as_secs_f64()
    );
}
