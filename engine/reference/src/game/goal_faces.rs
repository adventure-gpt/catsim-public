static GOALS: std::sync::LazyLock<[Goal; 2]> = std::sync::LazyLock::new(|| {
    let rows = crate::data::table("goals");
    assert_eq!(rows.len(), 2, "the goals table must have two rows");
    std::array::from_fn(|g| {
        let r = &rows[g];
        Goal {
            location: Vec3::from_bits(r.wn(0)),
            team_scoring: r.w(3) as u8,
            faces: std::array::from_fn(|k| GoalFace { normal: Vec3::from_bits(r.wn(4 + 6 * k)), vertex: Vec3::from_bits(r.wn(7 + 6 * k)) }),
        }
    })
});
