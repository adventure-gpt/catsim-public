fn next(s:&mut u64)->u32{*s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);(*s>>32)as u32}
macro_rules! capture {
    ($s:expr)=>{{
        let s=&$s;let mut out=Vec::new();
        out.extend([s.max_handles as u32,s.num_handles as u32,s.first_free as u32]);
        for axis in 0..3{for e in &s.edges[axis][..s.num_handles as usize*2+2]{out.extend([e.pos as u32,e.handle as u32]);}}
        for i in 0..s.max_handles as usize{
            if let Some(h)=s.handles.get(i){
                out.extend([h.client,h.group as u32,h.mask as u32,h.unique_id as u32]);
                out.extend(h.aabb_min.bits());out.extend(h.aabb_max.bits());
                out.extend(h.min_edges.map(u32::from));out.extend(h.max_edges.map(u32::from));
            }else{
                out.extend([0;10]);out.extend([if i+1==s.max_handles as usize{0}else{(i+1)as u32},0,0,0,0,0]);
            }
        }
        out.push(s.pairs.capacity as u32);out.push(s.pairs.pairs.len()as u32);
        for p in &s.pairs.pairs{out.extend([p.proxy0 as u32,p.proxy1 as u32,p.algorithm.is_some()as u32]);}
        out.extend(s.pairs.hash_table.iter().map(|x|*x as u32));out.extend(s.pairs.next.iter().map(|x|*x as u32));out
    }}
}
fn main(){
    let args:Vec<_>=std::env::args().collect();let cases:usize=args.get(1).map(|s|s.parse().unwrap()).unwrap_or(5000);
    let mut a=fast::bt::sweep::AxisSweep::new(fast::math::v3(-500.,-500.,-500.),fast::math::v3(500.,500.,500.),512);
    let mut b=scalar::bt::sweep::AxisSweep::new(scalar::math::v3(-500.,-500.,-500.),scalar::math::v3(500.,500.,500.),512);
    let mut ad=fast::bt::dispatch::Dispatcher::new();let mut bd=scalar::bt::dispatch::Dispatcher::new();
    let mut active=Vec::new();let mut rng=2801;let mut max_prefix=0;
    for case in 0..cases{
        let action=if case<400{0}else{next(&mut rng)%3};
        let x=next(&mut rng)%900;let y=next(&mut rng)%900;let z=next(&mut rng)%900;
        let lo=[x as f32-450.,y as f32-450.,z as f32-450.];let hi=lo.map(|v|v+20.);
        macro_rules! v{($e:ident,$x:ident)=>{$e::math::v3($x[0],$x[1],$x[2])}}
        if (action==0&&active.len()<500)||active.is_empty(){
            let ah=a.create_proxy(v!(fast,lo),v!(fast,hi),case as u32,1,0xffff,&mut ad);
            let bh=b.create_proxy(v!(scalar,lo),v!(scalar,hi),case as u32,1,0xffff,&mut bd);
            assert_eq!(ah,bh);active.push(ah);
        }else{
            let index=next(&mut rng)as usize%active.len();let id=active[index];
            if action==1{a.destroy_proxy(id,&mut ad);b.destroy_proxy(id,&mut bd);active.swap_remove(index);}
            else{a.set_aabb(id,v!(fast,lo),v!(fast,hi),&mut ad);b.set_aabb(id,v!(scalar,lo),v!(scalar,hi),&mut bd);}
        }
        max_prefix=max_prefix.max(a.handles.len());
        let af=capture!(a);let bf=capture!(b);
        if af!=bf{let i=af.iter().zip(&bf).position(|(x,y)|x!=y).unwrap();eprintln!("DIFF case={case} word={i} fast={:?} scalar={:?}",af[i],bf[i]);std::process::exit(1);}
    }
    println!("PASS operations={cases} logical_capacity={} maximum_materialized_prefix={max_prefix} all_logical_handles_compared=true",a.max_handles);
}
