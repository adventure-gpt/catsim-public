fn check(name: &str, x: f32, a: f32, b: f32) {
    if a.to_bits() != b.to_bits() {
        eprintln!(
            "DIFF {name} x={:08x} fast={:08x} scalar={:08x}",
            x.to_bits(),
            a.to_bits(),
            b.to_bits()
        );
        std::process::exit(1);
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let limit: u64 = args.get(1).map(|x| x.parse().unwrap()).unwrap_or(2_000_000);
    let all = args.get(2).is_some_and(|x| x == "all");
    let start = std::time::Instant::now();
    let mut rng = 1u64;
    for i in 0..limit {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = if all { i as u32 } else { (rng >> 32) as u32 };
        let x = f32::from_bits(bits);
        if !x.is_nan() {
            check(
                "sin",
                x,
                fast::math::crt::sinf(x),
                scalar::math::crt::sinf(x),
            );
            check(
                "cos",
                x,
                fast::math::crt::cosf(x),
                scalar::math::crt::cosf(x),
            );
            check(
                "acos",
                x,
                fast::math::crt::acosf(x),
                scalar::math::crt::acosf(x),
            );
            let y = f32::from_bits((rng as u32) & 0x7f7fffff);
            check(
                "atan2",
                x,
                fast::math::crt::atan2f(y, x),
                scalar::math::crt::atan2f(y, x),
            );
        }
    }
    println!(
        "PASS arguments={limit} exhaustive={all} seconds={:.3}",
        start.elapsed().as_secs_f64()
    );
}
