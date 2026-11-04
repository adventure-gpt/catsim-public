fn next(s:&mut u64)->u32{*s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);(*s>>32)as u32}
fn value(s:&mut u64)->f32{((next(s)%320001)as f32-160000.)*0.001}
fn main(){
    let args:Vec<_>=std::env::args().collect();let path=std::path::Path::new(&args[1]);
    let count:u64=args.get(2).map(|s|s.parse().unwrap()).unwrap_or(200000);
    let mutate=args.get(3).is_some_and(|s|s=="mutate");
    let mut f=fast::bt::collision_world::CollisionWorld::new();f.add_soccar_statics(path).unwrap();
    let mut s=scalar::bt::collision_world::CollisionWorld::new();s.add_soccar_statics(path).unwrap();
    let grid_mutant=args.get(3).is_some_and(|s|s=="grid");
    if grid_mutant {
        for sh in f.shapes.iter_mut() { if let fast::bt::shapes::Shape::Mesh(m)=sh { std::sync::Arc::make_mut(m).bvh.grid.plant_bug(); } }
    }
    let mut seed=3301;let mut hits=0;let mut boxes=0u64;let(mut fl,mut sl)=(Vec::new(),Vec::new());
    for case in 0..count {
        let id=6+(next(&mut seed)as usize%(s.bodies.len()-6));
        let mut a=fast::math::v3(value(&mut seed),value(&mut seed),value(&mut seed));
        let mut b=fast::math::v3(value(&mut seed),value(&mut seed),value(&mut seed));

        let near=next(&mut seed)%2==0;
        let local={
            let fast::bt::shapes::Shape::Mesh(m)=&f.shapes[f.bodies[id].shape.0 as usize] else{panic!()};
            let v=m.vertices[next(&mut seed) as usize%m.vertices.len()];
            let mut d=||((next(&mut seed)%2001) as f32-1000.0)*0.004;
            fast::math::v3(v.x+d(),v.y+d(),v.z+d())
        };
        if near {
            let o=f.bodies[id].transform.origin;
            a=fast::math::v3(local.x+o.x,local.y+o.y,local.z+o.z);
            let mut d=||((next(&mut seed)%2001) as f32-1000.0)*0.006;
            b=fast::math::v3(a.x+d(),a.y+d(),a.z+d());
        }
        if case%4096==4095 {
            let n=f.bodies[id].shape.0 as usize;
            let fast::bt::shapes::Shape::Mesh(m)=&mut f.shapes[n] else{panic!()};
            let scalar::bt::shapes::Shape::Mesh(r)=&mut s.shapes[n] else{panic!()};
            std::sync::Arc::make_mut(m).bvh.quantization.x=m.bvh.quantization.x.next_up();
            r.bvh.quantization.x=r.bvh.quantization.x.next_up();
        }
        if case%4096==2048 {
            f.bodies[id].transform.origin.x=f.bodies[id].transform.origin.x.next_up();
            s.bodies[id].transform.origin.x=s.bodies[id].transform.origin.x.next_up();
        }
        let r=f.ray_test_object(a,b,fast::bt::BodyId(id as u32));
        let q=s.ray_test_object(scalar::math::v3(a.x,a.y,a.z),scalar::math::v3(b.x,b.y,b.z),scalar::bt::BodyId(id as u32));
        let mut actual=[r.closest.to_bits(),r.body.map_or(u32::MAX,|b|b.0),r.normal.x.to_bits(),r.normal.y.to_bits(),r.normal.z.to_bits(),r.point.x.to_bits(),r.point.y.to_bits(),r.point.z.to_bits()];
        let expected=[q.closest.to_bits(),q.body.map_or(u32::MAX,|b|b.0),q.normal.x.to_bits(),q.normal.y.to_bits(),q.normal.z.to_bits(),q.point.x.to_bits(),q.point.y.to_bits(),q.point.z.to_bits()];
        if r.body.is_some(){hits+=1;if mutate{actual[5]^=1;}}
        if actual!=expected {eprintln!("DIFF case={case} id={id} fast={actual:?} scalar={expected:?}");std::process::exit(1);}

        let n=f.bodies[id].shape.0 as usize;
        let fast::bt::shapes::Shape::Mesh(m)=&f.shapes[n] else{panic!()};
        let scalar::bt::shapes::Shape::Mesh(r)=&s.shapes[n] else{panic!()};
        let h=(next(&mut seed)%1201) as f32*0.01;let h2=if next(&mut seed)%8==0 {-h} else {h};
        let c=if near {local} else {a};
        let lo=fast::math::v3(c.x-h,c.y-h2,c.z-h);let hi=fast::math::v3(c.x+h,c.y+h2,c.z+h*0.5);
        fl.clear();sl.clear();
        m.bvh.report_aabb_overlapping_nodes(lo,hi,|p,i|fl.push((p,i)));
        r.bvh.report_aabb_overlapping_nodes(scalar::math::v3(lo.x,lo.y,lo.z),scalar::math::v3(hi.x,hi.y,hi.z),|p,i|sl.push((p,i)));
        if !fl.is_empty(){boxes+=1;}
        if fl!=sl {eprintln!("DIFF box case={case} id={id} fast={} scalar={}",fl.len(),sl.len());std::process::exit(1);}
    }
    println!("PASS mesh_rays={count} hits={hits} boxes_with_leaves={boxes} mutation={mutate}/{grid_mutant}");
}
