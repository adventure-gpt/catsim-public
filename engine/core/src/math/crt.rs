#[path="crt_impl.rs"] mod exact;
#[inline]
pub fn sinf(x:f32)->f32 {
    #[cfg(target_arch="x86_64")]
    if std::is_x86_feature_detected!("fma") { return unsafe{sinf_fma(x)}; }
    exact::sinf(x)
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="fma")]
unsafe fn sinf_fma(x:f32)->f32 {exact::sinf(x)}
#[inline]
pub fn cosf(x:f32)->f32 {
    #[cfg(target_arch="x86_64")]
    if std::is_x86_feature_detected!("fma") { return unsafe{cosf_fma(x)}; }
    exact::cosf(x)
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="fma")]
unsafe fn cosf_fma(x:f32)->f32 {exact::cosf(x)}
#[inline]
pub fn acosf(x:f32)->f32 {
    #[cfg(target_arch="x86_64")]
    if std::is_x86_feature_detected!("fma") { return unsafe{acosf_fma(x)}; }
    exact::acosf(x)
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="fma")]
unsafe fn acosf_fma(x:f32)->f32 {exact::acosf(x)}
#[inline]
pub fn atan2f(y:f32,x:f32)->f32 {
    #[cfg(target_arch="x86_64")]
    if std::is_x86_feature_detected!("fma") { return unsafe{atan2f_fma(y,x)}; }
    exact::atan2f(y,x)
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="fma")]
unsafe fn atan2f_fma(y:f32,x:f32)->f32 {exact::atan2f(y,x)}
