#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::excessive_precision)]
pub mod data;

pub mod bt;
#[cfg(feature = "car_core")]
pub mod game;
pub mod math;
pub mod prof;
#[cfg(feature = "flow")]
pub mod adapter;
#[cfg(feature = "flow")]
pub mod group;
