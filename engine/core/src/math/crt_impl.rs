const S1: f64 = f64::from_bits(0xbfc5555555555555);
const S2: f64 = f64::from_bits(0x3f81111111111111);
const S3: f64 = f64::from_bits(0xbf2a01a01a01a01a);
const S4: f64 = f64::from_bits(0x3ec71de3a556c734);
const C1: f64 = f64::from_bits(0x3fa5555555555555);
const C2: f64 = f64::from_bits(0xbf56c16c16c16c16);
const C3: f64 = f64::from_bits(0x3efa01a01a01a019);
const C4: f64 = f64::from_bits(0xbe927e4fb7789f5c);
const TWO_OVER_PI: f64 = f64::from_bits(0x3fe45f306dc9c883);
const PIO2_1: f64 = f64::from_bits(0x3ff921fb54400000);
const PIO2_1T: f64 = f64::from_bits(0x3dd0b4611a626331);
const ONE_SIXTH: f64 = f64::from_bits(0x3fc5555555555555);

#[inline(always)]
fn sin_kernel(r: f64) -> f64 {
    let r2 = r * r;
    let p = r2.mul_add(S4, S3);
    let p = p.mul_add(r2, S2);
    let p = p.mul_add(r2, S1);
    let r3 = r * r2;
    p.mul_add(r3, r)
}

#[inline(always)]
fn cos_poly(r2: f64, head: f64) -> f64 {
    let p = r2.mul_add(C4, C3);
    let p = p.mul_add(r2, C2);
    let p = p.mul_add(r2, C1);
    let r4 = r2 * r2;
    p.mul_add(r4, head)
}

#[inline(always)]
fn reduce(abs_x: f64) -> (u32, f64) {
    let n = TWO_OVER_PI.mul_add(abs_x, 0.5) as i32;
    let nd = f64::from(n);
    let a = (-PIO2_1).mul_add(nd, abs_x);
    let b = nd * PIO2_1T;
    (n as u32 & 3, a - b)
}

#[inline(always)]
pub fn sinf(x: f32) -> f32 {
    let bits = x.to_bits() & 0x7fff_ffff;
    if bits >= 0x7f80_0000 {
        return f32::NAN;
    }
    let d = f64::from(x);
    if bits <= 0x3f49_0fdb {
        if bits < 0x3900_0000 {
            return x;
        }
        if bits < 0x3c00_0000 {
            let x3 = d * d * d;
            return ((-x3).mul_add(ONE_SIXTH, d)) as f32;
        }
        return sin_kernel(d) as f32;
    }
    let abs_x = if bits < 0x4b80_0456 { f64::from_bits(d.to_bits() & 0x7fff_ffff_ffff_ffff) } else { f64::from_bits(d.to_bits() & 0x7fff_ffff_ffff_ffff) % core::f64::consts::TAU };
    let (q, r) = reduce(abs_x);
    let k = if q & 1 == 0 {
        sin_kernel(r)
    } else {
        let r2 = r * r;
        cos_poly(r2, r2.mul_add(-0.5, 1.0))
    };
    let mut bits64 = k.to_bits();
    if q & 2 != 0 {
        bits64 ^= 1 << 63;
    }
    if x < 0.0 {
        bits64 ^= 1 << 63;
    }
    f64::from_bits(bits64) as f32
}

#[inline(always)]
pub fn cosf(x: f32) -> f32 {
    let bits = x.to_bits() & 0x7fff_ffff;
    if bits >= 0x7f80_0000 {
        return f32::NAN;
    }
    let d = f64::from(x);
    if bits <= 0x3f49_0fdb {
        if bits < 0x3900_0000 {
            return 1.0;
        }
        if bits < 0x3c00_0000 {
            let half = d * 0.5;
            return ((-half).mul_add(d, 1.0)) as f32;
        }
        let r2 = d * d;
        return cos_poly(r2, 1.0 - r2 * 0.5) as f32;
    }
    let abs_x = if bits < 0x4f49_0fdb { f64::from_bits(d.to_bits() & 0x7fff_ffff_ffff_ffff) } else { f64::from_bits(d.to_bits() & 0x7fff_ffff_ffff_ffff) % core::f64::consts::TAU };
    let (q, r) = reduce(abs_x);
    let k = if q & 1 == 1 {
        let s = sin_kernel(r);
        if q & 2 == 0 { -s } else { s }
    } else {
        let r2 = r * r;
        let c = cos_poly(r2, 1.0 - r2 * 0.5);
        if q & 2 != 0 { -c } else { c }
    };
    k as f32
}

#[inline(always)]
pub fn acosf(x: f32) -> f32 {
    let u = x.to_bits();
    let exp = (u >> 23) & 0xff;
    if u & 0x7fff_ffff > 0x7f80_0000 {
        return f32::from_bits(u | 0x0040_0000);
    }
    if exp < 0x65 {
        return f32::from_bits(0x3fc9_0fdb);
    }
    if exp >= 0x7f {
        return if x == 1.0 {
            0.0
        } else if x == -1.0 {
            f32::from_bits(0x4049_0fdb)
        } else {
            f32::from_bits(0xffc0_0000)
        };
    }
    let neg = (u as i32) < 0;
    let xa = if neg { -x } else { x };
    let big = exp >= 0x7e;
    let (r, s) = if big {
        let r = (1.0 - xa) * 0.5;
        (r, r.sqrt())
    } else {
        (xa * xa, 0.0)
    };
    let c40 = f32::from_bits(0x3b81_ce6b);
    let c44 = f32::from_bits(0x3d67_8bdd);
    let c48 = f32::from_bits(0x3e3c_94dc);
    let c4c = f32::from_bits(0x3f56_1f0d);
    let c50 = f32::from_bits(0x3f8d_6fa5);
    let c54 = f32::from_bits(0xbc5b_3fe1);
    let p = (-c40).mul_add(r, c54);
    let p = p.mul_add(r, -c44);
    let p = p.mul_add(r, c48);
    let num = p * r;
    let den = (-c4c).mul_add(r, c50);
    let v = num / den;
    const TINY: f64 = f64::from_bits(0x3c91_a626_3314_5c07);
    const PIO2: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18);
    const PI: f64 = f64::from_bits(0x4009_21fb_5444_2d18);
    if !big {
        let a = TINY - (v * x) as f64;
        return (PIO2 - (x as f64 - a)) as f32;
    }
    if neg {
        let a = (s * v) as f64 - TINY + s as f64;
        return (PI - (a + a)) as f32;
    }
    let c = f32::from_bits(s.to_bits() & 0xffff_0000);
    let f = (-c).mul_add(c, r) / (c + s);
    let t = v.mul_add(s + s, f + f);
    t + (c + c)
}

const ATAN_TAB: [u64; 241] = [
    0x3faff55bb72cfde9, 0x3fb0f99ea71d52a6, 0x3fb1f86dbf082d58, 0x3fb2f719318a4a9a, 0x3fb3f59f0e7c559d, 0x3fb4f3fd677292fb,
    0x3fb5f2324fd2d7b2, 0x3fb6f03bdcea4b0c, 0x3fb7ee182602f10e, 0x3fb8ebc54478fb28, 0x3fb9e94153cfdcf1, 0x3fbae68a71c722b8,
    0x3fbbe39ebe6f07c3, 0x3fbce07c5c3cca32, 0x3fbddd21701eba6e, 0x3fbed98c2190043a, 0x3fbfd5ba9aac2f6d, 0x3fc068d584212b3d,
    0x3fc0e6adccf40881, 0x3fc1646541060850, 0x3fc1e1fafb043726, 0x3fc25f6e171a535c, 0x3fc2dcbdb2fba1ff, 0x3fc359e8edeb99a3,
    0x3fc3d6eee8c6626c, 0x3fc453cec6092a9e, 0x3fc4d087a9da4f17, 0x3fc54d18ba11570a, 0x3fc5c9811e3ec269, 0x3fc645bfffb3aa73,
    0x3fc6c1d4898933d8, 0x3fc73dbde8a7d201, 0x3fc7b97b4bce5b02, 0x3fc8350be398ebc7, 0x3fc8b06ee2879c28, 0x3fc92ba37d050271,
    0x3fc9a6a8e96c8626, 0x3fca217e601081a5, 0x3fca9c231b403279, 0x3fcb1696574d780b, 0x3fcb90d7529260a2, 0x3fcc0ae54d768466,
    0x3fcc84bf8a742e6d, 0x3fccfe654e1d5395, 0x3fcd77d5df205736, 0x3fcdf110864c9d9d, 0x3fce6a148e96ec4d, 0x3fcee2e1451d980c,
    0x3fcf5b75f92c80dd, 0x3fcfd3d1fc40dbe4, 0x3fd025fa510665b5, 0x3fd061eea03d6290, 0x3fd09dc597d86362, 0x3fd0d97ee509acb3,
    0x3fd1151a362431c9, 0x3fd150973a9ce546, 0x3fd18bf5a30bf178, 0x3fd1c735212dd883, 0x3fd2025567e47c95, 0x3fd23d562b381041,
    0x3fd278372057ef45, 0x3fd2b2f7fd9b5fe2, 0x3fd2ed987a823cfe, 0x3fd328184fb58951, 0x3fd362773707ebcb, 0x3fd39cb4eb76157b,
    0x3fd3d6d129271134, 0x3fd410cbad6c7d32, 0x3fd44aa436c2af09, 0x3fd4845a84d0c21b, 0x3fd4bdee586890e6, 0x3fd4f75f73869978,
    0x3fd530ad9951cd49, 0x3fd569d88e1b4cd7, 0x3fd5a2e0175e0f4e, 0x3fd5dbc3fbbe768d, 0x3fd614840309cfe1, 0x3fd64d1ff635c1c5,
    0x3fd685979f5fa6fd, 0x3fd6bdeac9cbd76c, 0x3fd6f61941e4def0, 0x3fd72e22d53aa2a9, 0x3fd7660752817501, 0x3fd79dc6899118d1,
    0x3fd7d5604b63b3f7, 0x3fd80cd46a14b1d0, 0x3fd84422b8df95d7, 0x3fd87b4b0c1ebedb, 0x3fd8b24d394a1b25, 0x3fd8e92916f5cde8,
    0x3fd91fde7cd0c662, 0x3fd9566d43a34907, 0x3fd98cd5454d6b18, 0x3fd9c3165cc58107, 0x3fd9f93066168001, 0x3fda2f233e5e530b,
    0x3fda64eec3cc23fc, 0x3fda9a92d59e98cf, 0x3fdad00f5422058b, 0x3fdb056420ae9343, 0x3fdb3a911da65c6c, 0x3fdb6f962e737efb,
    0x3fdba473378624a5, 0x3fdbd9281e528191, 0x3fdc0db4c94ec9ef, 0x3fdc42191ff11eb6, 0x3fdc76550aad71f8, 0x3fdcaa6872f3631b,
    0x3fdcde53432c1350, 0x3fdd121566b7f2ad, 0x3fdd45aec9ec862b, 0x3fdd791f5a1226f4, 0x3fddac670561bb4f, 0x3fdddf85bb026974,
    0x3fde127b6b0744af, 0x3fde4548066cf51a, 0x3fde77eb7f175a34, 0x3fdeaa65c7cf28c4, 0x3fdedcb6d43f8434, 0x3fdf0ede98f393cf,
    0x3fdf40dd0b541417, 0x3fdf72b221a4e495, 0x3fdfa45dd3029258, 0x3fdfd5e0175fdf83, 0x3fe0039c73c1a40b, 0x3fe01c341e82422d,
    0x3fe034b709250488, 0x3fe04d25314342e5, 0x3fe0657e94db30cf, 0x3fe07dc3324e9b38, 0x3fe095f30861a58f, 0x3fe0ae0e1639866c,
    0x3fe0c6145b5b43da, 0x3fe0de05d7aa6f7c, 0x3fe0f5e28b67e295, 0x3fe10daa77307a0d, 0x3fe1255d9bfbd2a8, 0x3fe13cfbfb1b056e,
    0x3fe1548596376469, 0x3fe16bfa6f5137e1, 0x3fe1835a88be7c13, 0x3fe19aa5e5299f99, 0x3fe1b1dc87904284, 0x3fe1c8fe7341f64f,
    0x3fe1e00babdefeb3, 0x3fe1f7043557138a, 0x3fe20de813e823b1, 0x3fe224b74c1d192a, 0x3fe23b71e2cc9e6a, 0x3fe25217dd17e501,
    0x3fe268a940696da6, 0x3fe27f261273d1b3, 0x3fe2958e59308e30, 0x3fe2abe21aded073, 0x3fe2c2215e024465, 0x3fe2d84c2961e48b,
    0x3fe2ee628406cbca, 0x3fe30464753b090a, 0x3fe31a52048874be, 0x3fe3302b39b78856, 0x3fe345f01cce37bb, 0x3fe35ba0b60eccce,
    0x3fe3713d0df6c503, 0x3fe386c52d3db11e, 0x3fe39c391cd41719, 0x3fe3b198e5e2564a, 0x3fe3c6e491c78dc4, 0x3fe3dc1c2a188504,
    0x3fe3f13fb89e96f4, 0x3fe4064f47569f48, 0x3fe41b4ae06fea41, 0x3fe430328e4b26d5, 0x3fe445065b795b55, 0x3fe459c652badc7f,
    0x3fe46e727efe4715, 0x3fe4830aeb5f7bfd, 0x3fe4978fa3269ee1, 0x3fe4ac00b1c71762, 0x3fe4c05e22de94e4, 0x3fe4d4a8023414e8,
    0x3fe4e8de5bb6ec04, 0x3fe4fd013b7dd17e, 0x3fe51110adc5ed81, 0x3fe5250cbef1e9fa, 0x3fe538f57b89061e, 0x3fe54ccaf0362c8f,
    0x3fe5608d29c70c34, 0x3fe5743c352b33b9, 0x3fe587d81f732fba, 0x3fe59b60f5cfab9d, 0x3fe5aed6c5909517, 0x3fe5c2399c244260,
    0x3fe5d58987169b18, 0x3fe5e8c6941043cf, 0x3fe5fbf0d0d5cc49, 0x3fe60f084b46e05e, 0x3fe6220d115d7b8d, 0x3fe634ff312d1f3b,
    0x3fe647deb8e20b8f, 0x3fe65aabb6c07b02, 0x3fe66d663923e086, 0x3fe6800e4e7e2857, 0x3fe692a40556fb6a, 0x3fe6a5276c4b0575,
    0x3fe6b798920b3d98, 0x3fe6c9f7855c3198, 0x3fe6dc44551553ae, 0x3fe6ee7f10204aef, 0x3fe700a7c5784633, 0x3fe712be84295198,
    0x3fe724c35b4fae7b, 0x3fe736b65a172dff, 0x3fe748978fba8e0f, 0x3fe75a670b82d8d8, 0x3fe76c24dcc6c6c0, 0x3fe77dd112ea22c7,
    0x3fe78f6bbd5d315e, 0x3fe7a0f4eb9c19a2, 0x3fe7b26cad2e50fd, 0x3fe7c3d311a6092b, 0x3fe7d528289fa093, 0x3fe7e66c01c114fd,
    0x3fe7f79eacb97898, 0x3fe808c03940694a, 0x3fe819d0b7158a4c, 0x3fe82ad036000005, 0x3fe83bbec5cdee22, 0x3fe84c9c7653f7ea,
    0x3fe85d69576cc2c5, 0x3fe86e2578f87ae5, 0x3fe87ed0eadc5a2a, 0x3fe88f6bbd023118, 0x3fe89ff5ff57f1f7, 0x3fe8b06fc1cf3dfe,
    0x3fe8c0d9145cf49d, 0x3fe8d13206f8c4ca, 0x3fe8e17aa99cc05d, 0x3fe8f1b30c44f167, 0x3fe901db3eeef187, 0x3fe911f35199833b,
    0x3fe921fb54442d18,
];

#[inline(always)]
pub fn atan2f(y: f32, x: f32) -> f32 {
    let (dy, dx) = (y as f64, x as f64);
    let (ub, xb) = (dy.to_bits(), dx.to_bits());
    let neg_y = ub >> 63 != 0;
    let neg_x = xb >> 63 != 0;
    let (ay, ax) = (ub & 0x7fff_ffff_ffff_ffff, xb & 0x7fff_ffff_ffff_ffff);
    const PIO2_F: u32 = 0x3fc9_0fdb;
    const PI_F: u32 = 0x4049_0fdb;
    if ay == 0 {
        return if neg_x {
            f32::from_bits(if neg_y { PI_F | 0x8000_0000 } else { PI_F })
        } else {
            y
        };
    }
    let d = ((ub >> 52) & 0x7ff) as i32 - ((xb >> 52) & 0x7ff) as i32;
    if ax == 0 || d > 26 {
        return f32::from_bits(if neg_y { PIO2_F | 0x8000_0000 } else { PIO2_F });
    }
    if d < -13 && !neg_x {
        if d < -150 {
            return if neg_y { -0.0 } else { 0.0 };
        }
        return (dy / dx) as f32;
    }
    if d < -26 && neg_x {
        return f32::from_bits(if neg_y { PI_F | 0x8000_0000 } else { PI_F });
    }
    let (mut u, mut v) = (dx.abs(), dy.abs());
    let swap = v > u;
    if swap {
        core::mem::swap(&mut u, &mut v);
    }
    let q = v / u;
    const K_1_3: f64 = f64::from_bits(0x3fd5_5555_5555_0877);
    let mut a = if q > 0.0625 {
        let k = q.mul_add(256.0, 0.5) as i32;
        let kd = k as f64;
        let n = v.mul_add(256.0, -(kd * u));
        let m = v.mul_add(kd, u * 256.0);
        let t = n / m;
        let t2 = t * t;
        let base = t + f64::from_bits(ATAN_TAB[(k - 16) as usize]);
        (-(t2 * t)).mul_add(K_1_3, base)
    } else if q < f64::from_bits(0x3f1a_36e2_eb1c_432d) {
        q
    } else {
        let q2 = q * q;
        let p = (-q2).mul_add(f64::from_bits(0x3fc2_4924_82bd_6be1), f64::from_bits(0x3fc9_9999_9996_43a3));
        let p = (-q2).mul_add(p, f64::from_bits(0x3fd5_5555_5555_5538));
        (-(q2 * q)).mul_add(p, q)
    };
    if swap {
        a = f64::from_bits(0x3ff9_21fb_5444_2d18) - a;
    }
    if neg_x {
        a = f64::from_bits(0x4009_21fb_5444_2d18) - a;
    }
    if neg_y {
        a = -a;
    }
    a as f32
}
