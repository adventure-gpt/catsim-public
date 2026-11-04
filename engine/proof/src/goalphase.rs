mod f { use fast as engine; include!("fields.rs"); }
mod r { use scalar as engine; include!("fields.rs"); }

macro_rules! goal_words {
    ($a:expr, $out:expr, $rot:expr) => {{
        let a = $a;
        let o = $out;
        o.push(a.post_goal_elapsed().to_bits());
        match &a.explosion {
            None => o.push(0),
            Some(e) => {
                o.extend([1, e.life.to_bits(), e.ticks as u32]);
                o.extend([e.location.x.to_bits(), e.location.y.to_bits(), e.location.z.to_bits()]);
                o.extend([e.rotation.pitch as u32, e.rotation.yaw as u32, e.rotation.roll as u32]);
                o.push(e.hit.len() as u32); o.extend(&e.hit);
            }
        }
        for p in &a.players {
            o.extend([p.actor_loc.x.to_bits(),p.actor_loc.y.to_bits(),p.actor_loc.z.to_bits()]);
            let q = ($rot)(p);
            o.extend([q.pitch as u32,q.yaw as u32,q.roll as u32]);
        }
    }};
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let meshes = std::path::Path::new(args.get(1).expect("catsim_goalphase MESHES [mutate]"));
    fast::data::load(meshes).unwrap();
    scalar::data::load(meshes).unwrap();
    let mutate = args.get(2).is_some_and(|v| v == "mutate");
    let mut frames = 0u64;
    let mut steps_total = 0u64;
    let mut goals = 0u64;
    let mut hits = 0u64;
    let mut restarts = 0u64;
    for body in fast::game::config::BODIES.iter() {
        for mode in 1..=3 {
            for side in [-1.0f32, 1.0] {
                let name = body.name;
                let mut a = fast::adapter::ArenaSlot::new(mode,name,meshes,fast::game::arena::RoundConfig {
                    countdown_ticks:2, post_goal_time:3.0, match_seconds:None }).unwrap();
                let mut b = scalar::adapter::ArenaSlot::new(mode,name,meshes,scalar::game::arena::RoundConfig {
                    countdown_ticks:2, post_goal_time:3.0, match_seconds:None }).unwrap();
                macro_rules! setup { ($s:ident,$e:ident) => {{
                    $s.arena.start_at_drop(None);
                    $s.arena.phase = $e::game::arena::Phase::Active;
                    for i in 0..mode*2 {
                        let mut state = $s.arena.players[i].car.as_ref().unwrap().state;
                        state.location = $e::math::v3((i as f32-2.5)*190.0,side*(4900.0-i as f32*220.0),350.0+i as f32*60.0);
                        state.velocity = $e::math::v3(600.0-i as f32*130.0,side*500.0,120.0);
                        $s.arena.set_car_state(i,&state,Some(50.0));
                    }
                    let mut state = $s.arena.ball.as_ref().unwrap().state;
                    state.location = $e::math::v3(100.0,side*5190.0,200.0);
                    state.velocity = $e::math::v3(0.0,side*1800.0,0.0);
                    $s.arena.set_ball_state(&state,false);

                    $s.arena.frame_with_time(&[],0,0.001,1.0);
                }}; }
                setup!(a,fast); setup!(b,scalar);
                let mut fa=Vec::new(); let mut ra=Vec::new();
                let mut saw_goal=false; let mut saw_restart=false; let mut last_hits=0;
                let mut accumulator=0.0f32;
                for n in 0..1500 {
                    let dt=match n%19 {0=>0.018,1..=7=>0.001,8=>0.0002,_=>0.0041667};
                    accumulator += dt*120.0;
                    let steps=accumulator.floor() as i32;
                    accumulator -= steps as f32;
                    let mut input=[[0.0;8];6];
                    for (i,row) in input[..mode*2].iter_mut().enumerate() {
                        row[0]=1.0; row[1]=if i%2==0 {0.25} else {-0.5};
                        row[2]=0.2; row[3]=-0.3; row[4]=0.1;
                        row[5]=((n+i)%97<5) as u8 as f32; row[6]=(n%131<20) as u8 as f32;
                    }
                    let ac:Vec<_>=input[..mode*2].iter().map(|x|fast::adapter::input_from_row(x)).collect();
                    let bc:Vec<_>=input[..mode*2].iter().map(|x|scalar::adapter::input_from_row(x)).collect();
                    a.step_frame_with_time(&ac,steps,dt,accumulator);
                    b.step_frame_with_time(&bc,steps,dt,accumulator);
                    if let Some(e)=&a.arena.explosion {
                        if !saw_goal {goals+=1;saw_goal=true;}
                        hits+=(e.hit.len()-last_hits) as u64; last_hits=e.hit.len();
                    } else if saw_goal && !saw_restart {saw_restart=true;restarts+=1;}
                    if mutate && saw_goal {
                        if let Some(e)=a.arena.explosion.as_mut() { e.life=f32::from_bits(e.life.to_bits()^1); }
                    }
                    f::capture(&a,&mut fa); r::capture(&b,&mut ra);
                    goal_words!(&a.arena,&mut fa,|p:&fast::game::arena::Player|p.actor_rot());
                    goal_words!(&b.arena,&mut ra,|p:&scalar::game::arena::Player|p.actor_rot);
                    if fa!=ra {
                        let word=fa.iter().zip(&ra).position(|(x,y)|x!=y).unwrap_or(fa.len().min(ra.len()));
                        eprintln!("DIFF body={name} mode={mode} side={side} frame={n} word={word} fast={:?} reference={:?}",fa.get(word),ra.get(word));
                        std::process::exit(1);
                    }
                    frames+=1;steps_total+=steps as u64;
                    if saw_restart && n>900 {break;}
                }
                assert!(saw_goal && saw_restart,"scenario did not cover goal/reset: {name} {mode} {side}");
            }
        }
    }
    assert!(hits>0);
    println!("PASS frames={frames} physics_steps={steps_total} goals={goals} hit_actors={hits} restarts={restarts} mutation={mutate}");
}
