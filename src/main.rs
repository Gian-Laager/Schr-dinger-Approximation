mod airy;
mod airy_wave_func;
mod energy;
mod integrals;
mod newtons_method;
mod plot;
mod potentials;
mod tui;
mod turning_points;
mod utils;
mod wave_function_builder;
mod wkb_wave_func;

use crate::airy::airy_ai;
use crate::airy_wave_func::AiryWaveFunction;
use crate::integrals::*;
use crate::newtons_method::derivative;
use crate::utils::Func;
use crate::utils::*;
use crate::wave_function_builder::*;
use crate::wkb_wave_func::WkbWaveFunction;
use num::complex::Complex64;
use num::pow::Pow;
use rayon::iter::*;
use std::collections::HashMap;
use std::f64;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

const INTEG_STEPS: usize = 64000;
const TRAPEZE_PER_THREAD: usize = 1000;
const NUMBER_OF_POINTS: usize = 100000;

const AIRY_TRANSITION_FRACTION: f64 = 0.5;
const WKB_TRANSITION_FRACTION: f64 = 0.05;

const ENABLE_WKB_JOINTS: bool = false;
const ENABLE_AIRY_JOINTS: bool = false;

const COMPLEX_AIRY: bool = false;
const COMPLEX_EXP_WKB: bool = false;
const COMPLEX_OSZ_WKB: bool = false;

const APPROX_INF: (f64, f64) = (-200.0, 200.0);

fn main() {
    // let wave_function = wave_function_builder::SuperPosition::new(
    //     &potentials::square,
    //     1.0,
    //     &[
    //         (4, 1.0.into()),
    //         (5, complex(0.0, 1.0)),
    //         (6, complex(1.0, 1.0)),
    //     ],
    //     APPROX_INF,
    //     1.5,
    //     ScalingType::Renormalize(complex(1.0, 0.0)),
    // );

    let wave_function = wave_function_builder::WaveFunction::new(
        &potentials::square,
        1.0,
        2,
        APPROX_INF,
        1.5,
        ScalingType::Renormalize(1.0.into()),
    );

    let output_dir = Path::new("output");
    plot::plot_wavefunction_parts(&wave_function, output_dir, "data.txt");
}
