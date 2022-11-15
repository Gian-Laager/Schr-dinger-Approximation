#![allow(dead_code)]

mod airy;
mod airy_wave_func;
mod check;
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
use std::f64;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

const INTEG_STEPS: usize = 64000;
const TRAPEZE_PER_THREAD: usize = 1000;
const NUMBER_OF_POINTS: usize = 100000;

const AIRY_TRANSITION_FRACTION: f64 = 0.5;
const ENABLE_AIRY_JOINTS: bool = true;

const VALIDITY_LL_FACTOR: f64 = 3.5;

const APPROX_INF: (f64, f64) = (-200.0, 200.0);
const VIEW_FACTOR: f64 = 0.2;

fn main() {
    // let wave_function = wave_function_builder::WaveFunction::new(
    //     &potentials::mexican_hat,
    //     1.0, // mass
    //     54,  // nth energy
    //     APPROX_INF,
    //     VIEW_FACTOR,
    //     ScalingType::Renormalize(complex(0.0, f64::consts::PI / 4.0).exp()),
    // );

    let wave_function = wave_function_builder::SuperPosition::new(
        &potentials::square,
        1.0, // mass
        &[
            (1, complex(1.0, 0.0)), // (nth energy, phase)
            (2, complex(0.0, 1.0)), // (nth energy, phase)
            // (10, complex(0.0, 1.0)), // (nth energy, phase)
        ],
        APPROX_INF,
        VIEW_FACTOR,
        ScalingType::Renormalize(complex(1.0, 0.0)),
    );

    let output_dir = Path::new("output");

    // For WaveFunction
    // plot::plot_wavefunction(&wave_function, output_dir, "data.txt");
    // plot::plot_wavefunction_parts(&wave_function, output_dir, "data.txt");
    // plot::plot_probability(&wave_function, output_dir, "data.txt");

    // For SuperPosition
    plot::plot_superposition(&wave_function, output_dir, "data.txt");
    // plot::plot_probability_super_pos(&wave_function, output_dir, "data.txt");
}
