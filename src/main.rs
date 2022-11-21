#![allow(dead_code)]
#![feature(test)]

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
const VIEW_FACTOR: f64 = 0.5;

fn pot(x: f64) -> f64 {
    const SIGMA: f64 = 0.01;
    const EPSILON: f64 = 1.0;
    4.0 * EPSILON * ((SIGMA / x).powi(12) - (SIGMA / x).powi(6))
}

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
            (2, complex(2.0, 0.0)), // (nth energy, phase)
            (3, complex(3.0, 0.0)), // (nth energy, phase)
            (4, complex(4.0, 0.0)), // (nth energy, phase)
            (5, complex(5.0, 0.0)), // (nth energy, phase)
            (6, complex(6.0, 0.0)), // (nth energy, phase)
            (7, complex(7.0, 0.0)), // (nth energy, phase)
            (8, complex(8.0, 0.0)), // (nth energy, phase)
            (9, complex(9.0, 0.0)), // (nth energy, phase)
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
    // plot::plot_probability_superposition(&wave_function, output_dir, "data.txt");
}

#[cfg(test)]
mod test {
    use super::*;

    extern crate test;
    use test::Bencher;

    macro_rules! evaluate_bench_helper {
        ($i:ident, $n:expr) => {
            #[bench]
            pub fn $i(b: &mut Bencher) {
                let wave_function = wave_function_builder::WaveFunction::new(
                    &potentials::square,
                    1.0, // mass
                    $n,  // nth energy
                    APPROX_INF,
                    VIEW_FACTOR,
                    ScalingType::Renormalize(complex(0.0, f64::consts::PI / 4.0).exp()),
                );

                b.iter(|| {
                    let end = test::black_box(10.0);
                    evaluate_function_between(&wave_function, -10.0, end, 10);
                })
            }
        };
    }

    macro_rules! evaluate_bench {
        ($(($i:ident, $n:expr), )*) => {
            $(evaluate_bench_helper!($i, $n);)*
        }

    }

    macro_rules! energy_bench_helper {
        ($i:ident, $n:expr) => {
            #[bench]
            pub fn $i(b: &mut Bencher) {
                b.iter(|| {
                    let n = test::black_box($n);
                    let wave_function = wave_function_builder::WaveFunction::new(
                        &potentials::square,
                        1.0, // mass
                        n,   // nth energy
                        APPROX_INF,
                        VIEW_FACTOR,
                        ScalingType::Renormalize(complex(0.0, f64::consts::PI / 4.0).exp()),
                    );
                    let _ = test::black_box(&wave_function);
                })
            }
        };
    }

    macro_rules! energy_bench {
        ($(($i:ident, $n:expr), )*) => {
            $(energy_bench_helper!($i, $n);)*
        }

    }

    evaluate_bench!(
        (evaluate_bench1, 1),
        (evaluate_bench2, 2),
        (evaluate_bench3, 3),
        (evaluate_bench4, 4),
        (evaluate_bench5, 5),
        (evaluate_bench6, 6),
        (evaluate_bench7, 7),
        (evaluate_bench8, 8),
        (evaluate_bench9, 9),
    );

    energy_bench!(
        (energy_bench1, 1),
        (energy_bench2, 2),
        (energy_bench3, 3),
        (energy_bench4, 4),
        (energy_bench5, 5),
        (energy_bench6, 6),
        (energy_bench7, 7),
        (energy_bench8, 8),
        (energy_bench9, 9),
    );
}
