mod airy;
mod utils;
mod airy_wave_func;
mod newtons_method;
mod integrals;
mod wave_function_builder;
mod wkb_wave_func;

use crate::airy::airy_ai;
use crate::airy_wave_func::AiryWaveFunction;
use crate::integrals::*;
use crate::utils::*;
use num::complex::Complex64;
use num::pow::Pow;
use rayon::iter::*;
use std::f64;
use std::fs::File;
use std::io::Write;
use std::process::Command;
use tokio;
use crate::wkb_wave_func::WkbWaveFunction;

const TRAPEZE_PER_THREAD: usize = 1000;
const INTEG_STEPS: usize = 10000;
const NUMBER_OF_POINTS: usize = 10000;
const H_BAR: f64 = 1.5;
const X_0: f64 = 6.0;
const ENERGY: f64 = 20.0;
const MASS: f64 = 3.0;
const C_0: f64 = 1.0;
const THETA: f64 = 0.0;

const VIEW: (f64, f64) = (-4.5, 4.5);

#[derive(Copy, Clone)]
pub struct Phase {
    energy: f64,
    mass: f64,
    potential: fn(&f64) -> f64,
    x_0: f64,
}

impl Phase {
    fn default() -> Phase {
        Phase {
            energy: 0.0,
            mass: 0.0,
            potential: |x| 0.0,
            x_0: 0.0,
        }
    }

    const fn new(energy: f64, mass: f64, potential: fn(&f64) -> f64, x_0: f64) -> Phase {
        return Phase {
            energy,
            mass,
            potential,
            x_0,
        };
    }
}

impl ReToC for Phase {
    fn eval(&self, x: &f64) -> Complex64 {
        return (complex(2.0, 0.0)
            * complex(self.mass, 0.0)
            * complex((self.potential)(x) - self.energy, 0.0))
            .sqrt()
            / complex(H_BAR, 0.0);
    }
}

fn square(x: &f64) -> f64 {
    // 5.0 * (x + 1.0) * (x - 1.0) * (x + 2.0) * (x - 2.0) - 1.0
    x * x
}

fn order_ts((t1, t2): &(f64, f64)) -> (f64, f64) {
    return if t1 > t2 { (*t2, *t1) } else { (*t1, *t2) };
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let phase1: Phase = Phase::new(ENERGY, MASS, square, -X_0);
    let wave_func1 = WkbWaveFunction::new(&phase1, C_0, THETA, INTEG_STEPS);
    let values1 = evaluate_function_between(&wave_func1, VIEW.0, 0.0, NUMBER_OF_POINTS);
    let phase2: Phase = Phase::new(ENERGY, MASS, square, X_0);
    let wave_func2 = WkbWaveFunction::new(&phase2, C_0, THETA, INTEG_STEPS);
    let values2 = evaluate_function_between(&wave_func2, 0.0, VIEW.1, NUMBER_OF_POINTS);
    let turning_point_boundaries = &AiryWaveFunction::calc_ts(&phase1, VIEW).ts;

    let mut data_file = File::create("data.txt").unwrap();

    let mut data_str: String = values1
        .par_iter()
        .filter(|p| {
            (turning_point_boundaries.clone()).iter()
                .map(order_ts)
                .map(|(t1, t2)| p.x < t1 || p.x > t2)
                .fold(true, |a, b| a && b)
        })
        .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
        .reduce(|| String::new(), |s: String, current: String| s + &*current);

    data_str.push_str(&*values2
        .par_iter()
        .filter(|p| {
            (turning_point_boundaries.clone()).iter()
                .map(order_ts)
                .map(|(t1, t2)| p.x < t1 || p.x > t2)
                .fold(true, |a, b| a && b)
        })
        .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
        .reduce(|| String::new(), |s: String, current: String| s + &*current));

    let airy_wave_funcs = AiryWaveFunction::new(&wave_func1, (VIEW.0, 0.0));

    let airy_data_str = airy_wave_funcs.iter().map(|airy_wave_func| {
        let airy_values = evaluate_function_between(airy_wave_func, airy_wave_func.ts.0, airy_wave_func.ts.1, NUMBER_OF_POINTS);

        airy_values
            .par_iter()
            .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
            .reduce(|| String::new(), |s: String, current: String| s + &*current)
    })
        .fold(String::new(), |s: String, current: String| s + "\n\n" + &*current);
    data_str.push_str(&*airy_data_str);

    let airy_wave_funcs = AiryWaveFunction::new(&wave_func2, (0.0, VIEW.1));

    let airy_data_str = airy_wave_funcs.iter().map(|airy_wave_func| {
        let airy_values = evaluate_function_between(airy_wave_func, airy_wave_func.ts.0, airy_wave_func.ts.1, NUMBER_OF_POINTS);

        airy_values
            .par_iter()
            .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
            .reduce(|| String::new(), |s: String, current: String| s + &*current)
    })
        .fold(String::new(), |s: String, current: String| s + "\n\n" + &*current);
    data_str.push_str(&*airy_data_str);

    data_file
        .write_all((data_str).as_ref()).unwrap();

    let mut plot_3d_file = File::create("plot_3d.gnuplot").unwrap();

    let plot_3d_cmd = "splot \"data.txt\" i 0 u 1:2:3 t \"WKB\" w l, \"data.txt\" i 1 u 1:2:3 t \"Airy 1\" w l, \"data.txt\" i 2 u 1:2:3 t \"Ariy 2\" w l";
    plot_3d_file.write_all(plot_3d_cmd.as_ref()).unwrap();

    let mut plot_3d_file = File::create("plot.gnuplot").unwrap();

    let plot_3d_cmd = "plot \"data.txt\" i 0 u 1:2 t \"WKB\", \"data.txt\" i 1 u 1:2 t \"Airy 1\", \"data.txt\" i 2 u 1:2 t \"Ariy 2\"";
    plot_3d_file.write_all(plot_3d_cmd.as_ref()).unwrap();
}