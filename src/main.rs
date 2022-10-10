#![feature(unboxed_closures)]

mod airy;
mod airy_wave_func;
mod energy;
mod integrals;
mod newtons_method;
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

const PLOT_POTENTIAL: bool = false;
const SEPARATE_FUNCTIONS: bool = false;
const RENORMALIZE: bool = false;
const NORMALIZE_POTENTIAL: bool = false;

const MASS: f64 = 2.0;
const AIRY_EXTRA: f64 = 0.5;
const N_ENERGY: usize = 12;

const APPROX_INF: (f64, f64) = (-200.0, 200.0);
const ENERGY_INF: f64 = 1e6;
const VIEW_FACTOR: f64 = 1.5;

fn main() {
    let output_dir = Path::new("output");
    std::env::set_current_dir(&output_dir).unwrap();

    let wave_function = wave_function_builder::SuperPosition::new(
        &potentials::mexican_hat,
        MASS,
        &[
            (37, complex(1.0, 0.0)),
            (38, complex(1.0, 0.0)),
            (39, complex(1.0, 0.0)),
        ],
        APPROX_INF,
        VIEW_FACTOR,
        ScalingType::Renormalize(complex(1.0, 0.0)),
    );

    let all_values = evaluate_function_between(
        &wave_function,
        wave_function.get_view().0,
        wave_function.get_view().1,
        NUMBER_OF_POINTS,
    );

    let all_values_str = all_values
        .par_iter()
        .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
        .reduce(|| String::new(), |s: String, current: String| s + &*current);

    let mut data_full = File::create("data.txt").unwrap();
    data_full.write_all(all_values_str.as_ref()).unwrap();

    let mut plot_3d_file = File::create("plot_3d.gnuplot").unwrap();
    let plot_3d_cmd: String = "splot \"data.txt\" u 1:2:3 t \"Psi\" w l".to_string();
    plot_3d_file.write_all(plot_3d_cmd.as_ref()).unwrap();

    let mut plot_file = File::create("plot.gnuplot").unwrap();

    let plot_cmd = "plot \"data.txt\" u 1:2 t \"Psi\" w l".to_string();

    plot_file.write_all(plot_cmd.as_ref()).unwrap();

    let mut plot_imag_file = File::create("plot_im.gnuplot").unwrap();

    let plot_imag_cmd = "plot \"data.txt\" u 1:3 t \"Psi\" w l".to_string();

    plot_imag_file.write_all(plot_imag_cmd.as_ref()).unwrap();
}
