fn main() {
    let meshes = std::path::PathBuf::from(std::env::args().nth(1).expect("MESHES/soccar"));
    for mode in 1..=3usize {
        let mut a = fast::adapter::ArenaSlot::new(mode, "Octane", &meshes, fast::game::arena::RoundConfig { countdown_ticks: 2, post_goal_time: 0.2, match_seconds: None }).unwrap();
        a.arena.start_at_drop(None);
        let ctl: Vec<_> = (0..mode * 2).map(|_| fast::adapter::input_from_row(&[1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])).collect();
        let mut tot = [0usize; 6];
        let n = 2000;
        for t in 0..n {
            a.step(&ctl);
            let cw = &a.arena.world.cw;
            let pairs = &cw.broadphase.pairs.pairs;
            let h = &cw.broadphase.handles;
            let mut ss = 0;
            for p in pairs {
                let (b0, b1) = (&cw.bodies[h[p.proxy0 as usize].client as usize], &cw.bodies[h[p.proxy1 as usize].client as usize]);
                if b0.flags & fast::bt::CF_STATIC != 0 && b1.flags & fast::bt::CF_STATIC != 0 { ss += 1; }
            }
            let live = cw.manifolds().iter().filter(|&&id| cw.manifold(id).cached > 0).count();
            tot[0] += pairs.len(); tot[1] += ss; tot[2] += cw.manifolds().len(); tot[3] += live; tot[4] += cw.objects.len(); tot[5] += a.arena.world.rigid_bodies.len();
            if t == 1999 { println!("mode {mode} sample: pairs {} static-static {} manifolds {} live {} objects {} rigid {}", pairs.len(), ss, cw.manifolds().len(), live, cw.objects.len(), a.arena.world.rigid_bodies.len()); }
        }
        let f = |x: usize| x as f64 / n as f64;
        println!("mode {mode} mean: pairs {:.2} static-static {:.2} manifolds {:.2} live {:.2} objects {:.1} rigid {:.1}", f(tot[0]), f(tot[1]), f(tot[2]), f(tot[3]), f(tot[4]), f(tot[5]));
    }
}
