fn next(s:&mut u64)->u32{*s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);(*s>>32)as u32}
fn value(s:&mut u64)->f32{let x=next(s);if x%16==0{-0.0}else{((x%20001)as f32-10000.)*0.001}}
fn main(){
    let args:Vec<_>=std::env::args().collect();let cases:u64=args.get(1).map(|x|x.parse().unwrap()).unwrap_or(200000);
    let mutate=args.get(2).is_some_and(|x|x=="mutate");let mut seed=3001;
    let mut hits=0;
    for case in 0..cases{
        let mut t=fast::math::Transform::IDENTITY;
        for row in &mut t.basis.r {*row=fast::math::v3(value(&mut seed),value(&mut seed),value(&mut seed));}
        t.origin=fast::math::v3(value(&mut seed),value(&mut seed),value(&mut seed));
        let mut from:[fast::math::Vec3;4]=std::array::from_fn(|_|fast::math::v3(value(&mut seed)*100.,value(&mut seed)*100.,value(&mut seed)*100.));
        let mut to:[fast::math::Vec3;4]=std::array::from_fn(|_|fast::math::v3(value(&mut seed)*100.,value(&mut seed)*100.,value(&mut seed)*100.));
        if case%4==0 {
            t=fast::math::Transform::IDENTITY;t.basis.r[0].y=-0.0;t.basis.r[1].x=-0.0;
            let eps=f32::from_bits(0x3400_0000);
            for i in 0..4 {from[i].z=[0.,eps.next_down(),eps,eps.next_up()][i];to[i].z=-from[i].z;}
        }
        let actual=fast::bt::wheel_lanes::plane(&t,from,to);
        let vec=|v:fast::math::Vec3|scalar::math::v3(v.x,v.y,v.z);
        let rt=scalar::math::Transform{basis:scalar::math::Mat3{r:t.basis.r.map(vec)},origin:vec(t.origin)};
        for lane in 0..4 {
            let f=scalar::math::Transform{basis:scalar::math::Mat3::IDENTITY,origin:vec(from[lane])};
            let g=scalar::math::Transform{basis:scalar::math::Mat3::IDENTITY,origin:vec(to[lane])};
            let mut cb=scalar::bt::collision_world::ClosestRay::new(f.origin,g.origin);
            scalar::bt::collision_world::ray_test_single(&f,&g,scalar::bt::BodyId(7),
                &scalar::bt::shapes::Shape::Plane(scalar::bt::shapes::PlaneShape::new(1.,1.)),&rt,&mut cb);
            let expected=cb.body.map(|_|[cb.closest.to_bits(),cb.normal.x.to_bits(),cb.normal.y.to_bits(),cb.normal.z.to_bits(),cb.point.x.to_bits(),cb.point.y.to_bits(),cb.point.z.to_bits()]);
            let mut got=actual[lane].map(|h|[h.fraction.to_bits(),h.normal.x.to_bits(),h.normal.y.to_bits(),h.normal.z.to_bits(),h.point.x.to_bits(),h.point.y.to_bits(),h.point.z.to_bits()]);
            if let Some(ref mut a)=got {hits+=1;if mutate {a[4]^=1;}}
            if got!=expected {eprintln!("DIFF case={case} lane={lane} fast={got:?} scalar={expected:?} mutation={mutate}");std::process::exit(1);}
        }
    }
    println!("PASS rays={} hits={hits} mutation={mutate}",cases*4);
}
