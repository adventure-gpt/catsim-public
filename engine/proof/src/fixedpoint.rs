fn next(s:&mut u64)->u32{*s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);(*s>>32)as u32}
fn value(s:&mut u64)->f32 {
    let u=next(s);
    match u%16 {0=>0.0,1=>-0.0,_=>((u%20001)as f32-10000.0)*0.001}
}
macro_rules! make {
    ($engine:ident,$seed:expr,$rows:expr)=>{{
        use $engine::bt::{solver::*,BodyId};
        let mut rng=$seed;
        let mut s=Solver::new();
        let mut vec=||$engine::math::v3(value(&mut rng),value(&mut rng),value(&mut rng));
        for i in 0..5 {
            let mut b=init_solver_body(None,1.0/120.0);
            if i<4 {b.orig=Some(BodyId(i));}
            b.delta_lin=vec();b.delta_ang=vec();b.push=vec();b.turn=vec();
            b.inv_mass=vec();b.ang_factor=vec();b.lin_factor=vec();s.bodies.push(b);
        }
        for _ in 0..$rows {
            let mut r=Row::default();
            r.a=(next(&mut rng)%5)as i32;r.b=(next(&mut rng)%5)as i32;
            let mut vec=||$engine::math::v3(value(&mut rng),value(&mut rng),value(&mut rng));
            r.cross1=vec();r.cross2=vec();r.n1=vec();r.n2=vec();r.ang_a=vec();r.ang_b=vec();
            r.rhs=value(&mut rng);r.rhs_pen=value(&mut rng);r.cfm=value(&mut rng)*0.001;
            r.jac_inv=value(&mut rng).abs()*0.001;r.scale1=value(&mut rng)*0.1;r.scale2=value(&mut rng)*0.1;
            r.lower=0.0;r.upper=100.0;r.friction=value(&mut rng).abs()*0.1;r.skip=next(&mut rng)%3==0;
            r.other=s.contacts.len()as i32;s.contacts.push(r);s.frictions.push(r);
        }
        s
    }}
}
macro_rules! capture {
    ($s:expr)=>{{
        let s=&$s;let mut o=Vec::new();
        for b in &s.bodies{for v in [b.delta_lin,b.delta_ang,b.push,b.turn]{o.extend(v.bits());}}
        for r in s.contacts.iter().chain(&s.frictions){o.extend([r.applied,r.push,r.lower,r.upper].map(f32::to_bits));}
        o
    }}
}
fn main(){
    let args:Vec<_>=std::env::args().collect();let cases:u64=args.get(1).map(|x|x.parse().unwrap()).unwrap_or(100000);
    let mutation=args.get(2).is_some_and(|x|x=="early");let mut seed=2601;
    for case in 0..cases{
        let rows=1+next(&mut seed)as usize%12;
        let mut a=make!(fast,seed,rows);let mut b=make!(scalar,seed,rows);
        let iterations=[1,2,10,30][case as usize%4];
        let mut ia=fast::bt::solver::SolverInfo::game();let mut ib=scalar::bt::solver::SolverInfo::game();
        ia.num_iterations=iterations;ib.num_iterations=iterations;
        a.split_impulse_iterations(&ia);b.split_impulse_iterations(&ib);
        for _ in 0..iterations{if !a.iteration()||mutation{break;}}
        for _ in 0..iterations{b.iteration();}
        let fa=capture!(a);let fb=capture!(b);
        if fa!=fb {let i=fa.iter().zip(&fb).position(|(x,y)|x!=y).unwrap();eprintln!("DIFF case={case} word={i} fast={:08x} scalar={:08x} mutation={mutation}",fa[i],fb[i]);std::process::exit(1);}
    }
    println!("PASS cases={cases} full_solver_state=true mutation={mutation}");
}
