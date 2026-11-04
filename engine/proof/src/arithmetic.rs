fn next(s: &mut u64) -> u32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 32) as u32
}
fn sample(s: &mut u64) -> f32 {
    let x = next(s);
    if x & 15 == 0 {
        f32::from_bits([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                        0x3f800000, 0xbf800000][(x >> 4) as usize & 7])
    } else {
        f32::from_bits(x & 0xdeffffff)
    }
}
fn main() {
    use fast::math::{Mat3, v3};
    let args: Vec<_> = std::env::args().collect();
    let n: u64 = args.get(1).map(|x| x.parse().unwrap()).unwrap_or(2_000_000);
    let mutant = args.get(2).is_some_and(|x| x == "reassociate");
    let solver_mutant = args.get(2).is_some_and(|x| x == "solver-reassociate");
    let mut rng = 941u64;
    for case in 0..n {
        let mut vector = || v3(sample(&mut rng),sample(&mut rng),sample(&mut rng));
        let b = Mat3 { r: [vector(), vector(), vector()] };
        let v = vector();
        let mut actual = fast::math::lanes::rows_yx(&b, v);
        if mutant {
            let f = |r: fast::math::Vec3| v.y*r.y + (v.x*r.x + v.z*r.z);
            actual = v3(f(b.r[0]),f(b.r[1]),f(b.r[2]));
        }
        let f = |r: fast::math::Vec3| (v.y*r.y + v.x*r.x) + v.z*r.z;
        let expected = v3(f(b.r[0]),f(b.r[1]),f(b.r[2]));
        if actual.bits() != expected.bits() {
            eprintln!("DIFF arithmetic case={case} mutation={mutant} actual={:?} expected={:?}",actual.bits(),expected.bits());
            std::process::exit(1);
        }
        let (mass,normal,lf,af,axis,base_l,base_a)=(vector(),vector(),vector(),vector(),vector(),vector(),vector());
        let impulse=sample(&mut rng);
        for xfirst in [false,true] {
            let (mut lin,mut ang)=(base_l,base_a);
            fast::math::lanes::solver_impulse(&mut lin,&mut ang,mass,normal,lf,af,axis,impulse,xfirst);
            if solver_mutant { lin.x=(mass.x*normal.x)*(impulse*lf.x)+base_l.x; }
            let expected_l=v3((if xfirst {normal.x*mass.x}else{mass.x*normal.x})*impulse*lf.x+base_l.x,
                mass.y*normal.y*impulse*lf.y+base_l.y,mass.z*normal.z*impulse*lf.z+base_l.z);
            let expected_a=v3(impulse*af.x*axis.x+base_a.x,impulse*af.y*axis.y+base_a.y,impulse*af.z*axis.z+base_a.z);
            if lin.bits()!=expected_l.bits() || ang.bits()!=expected_a.bits() {
                eprintln!("DIFF solver case={case} mutation={solver_mutant} xfirst={xfirst} actual={:?}/{:?} expected={:?}/{:?}",lin.bits(),ang.bits(),expected_l.bits(),expected_a.bits());
                std::process::exit(1);
            }
        }
    }
    println!("PASS lane_matrices={n} solver_updates={} mutation={mutant}/{solver_mutant}",n*2);
}
