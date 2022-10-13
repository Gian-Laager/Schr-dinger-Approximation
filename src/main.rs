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
use std::collections::HashMap;
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
const WKB_TRANSITION_FRACTION: f64 = 0.05;
const N_ENERGY: usize = 12;

const APPROX_INF: (f64, f64) = (-200.0, 200.0);
const ENERGY_INF: f64 = 1e6;
const VIEW_FACTOR: f64 = 1.5;

fn main() {
    let output_dir = Path::new("output");
    std::env::set_current_dir(&output_dir).unwrap();

    // let wave_function = wave_function_builder::SuperPosition::new(
    //     &potentials::mexican_hat,
    //     MASS,
    //     &[
    //         // (5, 1.0.into()),
    //         (12, 1.0.into()),
    //         // (21, 1.0.into()),
    //     ],
    //     APPROX_INF,
    //     VIEW_FACTOR,
    //     ScalingType::Renormalize(complex(1.0, 0.0)),
    // );

    let wave_function = wave_function_builder::WaveFunction::new(
        &potentials::mexican_hat,
        MASS,
        12,
        APPROX_INF,
        1.5,
        ScalingType::Renormalize(1.0.into()),
    );
    println!("energy: {}", wave_function.get_energy());

    let wkb_values = wave_function
        .get_wkb_ranges_in_view()
        .iter()
        .map(|range| {
            evaluate_function_between(
                &wave_function,
                range.0,
                range.1,
                NUMBER_OF_POINTS,
            )
        })
        .collect::<Vec<Vec<Point<f64, Complex64>>>>();

    let airy_values = wave_function
        .get_airy_ranges()
        .iter()
        .map(|range| {
            evaluate_function_between(
                &wave_function,
                f64::max(wave_function.get_view().0, range.0),
                f64::min(wave_function.get_view().1, range.1),
                NUMBER_OF_POINTS,
            )
        })
        .collect::<Vec<Vec<Point<f64, Complex64>>>>();

    let wkb_values_str = wkb_values
        .par_iter()
        .map(|values| {
            values
                .par_iter()
                .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
                .reduce(|| String::new(), |s: String, current: String| s + &*current)
        })
        .reduce(
            || String::new(),
            |s: String, current: String| s + "\n\n" + &*current,
        );

    let airy_values_str = airy_values
        .par_iter()
        .map(|values| {
            values
                .par_iter()
                .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
                .reduce(|| String::new(), |s: String, current: String| s + &*current)
        })
        .reduce(
            || String::new(),
            |s: String, current: String| s + "\n\n" + &*current,
        );

    let mut data_full = File::create("data.txt").unwrap();
    data_full.write_all(wkb_values_str.as_ref()).unwrap();
    data_full.write_all("\n\n".as_bytes()).unwrap();
    data_full.write_all(airy_values_str.as_ref()).unwrap();

    let mut plot_3d_file = File::create("plot_3d.gnuplot").unwrap();

    let wkb_3d_cmd = (1..=wkb_values.len())
        .into_iter()
        .map(|n| format!("\"data.txt\" u 1:2:3 i {} t \"WKB {}\" w l", n - 1, n))
        .collect::<Vec<String>>()
        .join(", ");

    let airy_3d_cmd = (1..=airy_values.len())
        .into_iter()
        .map(|n| format!("\"data.txt\" u 1:2:3 i {} t \"Airy {}\" w l", n + wkb_values.len() - 1, n))
        .collect::<Vec<String>>()
        .join(", ");
    let plot_3d_cmd: String = "splot ".to_string() + &wkb_3d_cmd + ", " + &airy_3d_cmd;
    plot_3d_file.write_all(plot_3d_cmd.as_ref()).unwrap();

    let mut plot_file = File::create("plot.gnuplot").unwrap();
    let wkb_cmd = (1..=wkb_values.len())
        .into_iter()
        .map(|n| format!("\"data.txt\" u 1:2 i {} t \"WKB {}\" w l", n - 1, n))
        .collect::<Vec<String>>()
        .join(", ");

    let airy_cmd = (1..=airy_values.len())
        .into_iter()
        .map(|n| format!("\"data.txt\" u 1:2 i {} t \"Airy {}\" w l", n + wkb_values.len() - 1, n))
        .collect::<Vec<String>>()
        .join(", ");
    let plot_cmd: String = "plot ".to_string() + &wkb_cmd + ", " + &airy_cmd;

    plot_file.write_all(plot_cmd.as_ref()).unwrap();

    let mut plot_imag_file = File::create("plot_im.gnuplot").unwrap();

    let wkb_im_cmd = (1..=wkb_values.len())
        .into_iter()
        .map(|n| format!("\"data.txt\" u 1:3 i {} t \"WKB {}\" w l", n - 1, n))
        .collect::<Vec<String>>()
        .join(", ");

    let airy_im_cmd = (1..=airy_values.len())
        .into_iter()
        .map(|n| format!("\"data.txt\" u 1:3 i {} t \"Airy {}\" w l", n + wkb_values.len() - 1, n))
        .collect::<Vec<String>>()
        .join(", ");
    let plot_imag_cmd: String = "plot ".to_string() + &wkb_im_cmd + ", " + &airy_im_cmd;

    plot_imag_file.write_all(plot_imag_cmd.as_ref()).unwrap();
}
