#![feature(unboxed_closures)]

mod airy;
mod airy_wave_func;
mod integrals;
mod newtons_method;
mod turning_points;
mod utils;
mod wave_function_builder;
mod wkb_wave_func;

use crate::airy::airy_ai;
use crate::airy_wave_func::AiryWaveFunction;
use crate::integrals::*;
use crate::newtons_method::derivative;
use crate::utils::*;
use crate::wkb_wave_func::WkbWaveFunction;
use num::complex::Complex64;
use num::pow::Pow;
use rayon::iter::*;
use std::f64;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::process::Command;
use tokio;

fn nth_energy_square(n: usize) -> f64 {
    (2.0 * n as f64 + 1.0) * 2.0_f64.sqrt() / (2.0 * MASS.sqrt())
}

const TRAPEZE_PER_THREAD: usize = 1000;
const INTEG_STEPS: usize = 10000;
const NUMBER_OF_POINTS: usize = 100000;

fn ENERGY() -> f64 {
    nth_energy_square(12)
}

const MASS: f64 = 1.0;
const C_0: f64 = 1.0;
const _THETA: f64 = 0.0;
const AIRY_EXTRA: f64 = 1.0;

const VIEW: (f64, f64) = (-6.0, 6.0);

#[derive(Copy, Clone)]
pub struct Phase {
    energy: f64,
    mass: f64,
    potential: fn(f64) -> f64,
    phase_off: f64,
}

impl Phase {
    fn default() -> Phase {
        Phase {
            energy: 0.0,
            mass: 0.0,
            potential: |_x| 0.0,
            phase_off: f64::consts::PI / 4.0,
        }
    }

    fn new(energy: f64, mass: f64, potential: fn(f64) -> f64, phase_off: f64) -> Phase {
        return Phase {
            energy,
            mass,
            potential,
            phase_off,
        };
    }

    fn momentum(self, x: f64) -> f64 {
        self.eval(x).abs().sqrt()
    }
}

impl Func<f64, f64> for Phase {
    fn eval(&self, x: f64) -> f64 {
       (2.0 * self.mass * ((self.potential)(x) - self.energy)).abs().sqrt()
    }
}

fn square(x: f64) -> f64 {
    // 5.0 * (x + 1.0) * (x - 1.0) * (x + 2.0) * (x - 2.0) - 1.0

    x * x

    // (x-4.5)*(x-1.5)*(x+1.5)*(x+4.5)/4.0 + 20.0

    // let l = 3.0;
    // -1.0 / x + l*(l+1.0) / (2.0*MASS * x * x)
}

fn order_ts(((t1, t2), _): &((f64, f64), f64)) -> (f64, f64) {
    return if t1 > t2 { (*t2, *t1) } else { (*t1, *t2) };
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    // {
    //     let mut phase_off = -f64::consts::PI;
    //     const RANGE: f64 = 0.75;
    //
    //     struct WaveDeriv<'a, 'b> {
    //         wave: &'a WkbWaveFunction<'b>,
    //     }
    //
    //     impl Func<f64, f64> for WaveDeriv<'_, '_> {
    //         fn eval(&self, x: f64) -> f64 {
    //             return derivative(&|x| self.wave.eval(x), x);
    //         }
    //     }
    //
    //     let mut min_diff = f64::NAN;
    //     let mut optimal_phase = 0.0;
    //
    //     while phase_off < f64::consts::PI {
    //         let phase: Phase = Phase::new(ENERGY(), MASS, square, f64::consts::PI / 4.0);
    //         let (_, boundaries1) = AiryWaveFunction::new(&phase, (VIEW.0, 0.0));
    //         let wave_func1 =
    //             WkbWaveFunction::new(&phase, C_0, INTEG_STEPS, boundaries1.ts.first().unwrap().1);
    //
    //         let (_, boundaries2) = AiryWaveFunction::new(&phase, (0.0, VIEW.1));
    //         let wave_func2 =
    //             WkbWaveFunction::new(&phase, C_0, INTEG_STEPS, boundaries2.ts.last().unwrap().1);
    //         let diff = (wave_func2.eval(0.0) - wave_func1.eval(0.0)).abs();
    //
    //         if min_diff.is_nan() {
    //             min_diff = diff;
    //             optimal_phase = phase_off;
    //         } else if diff > min_diff {
    //             min_diff = diff;
    //             optimal_phase = phase_off;
    //         }
    //         phase_off += 0.1;
    //     }
    //
    //     println!("min_diff: {}, phase: {}", min_diff, optimal_phase);
    // }
    println!("Energy: {}", ENERGY());
    let phase: Phase = Phase::new(ENERGY(), MASS, square, f64::consts::PI / 4.0);
    let (airy_wave_func1, mut boundaries1) = AiryWaveFunction::new(&phase, (VIEW.0, 0.0));
    let wave_func1 =
        WkbWaveFunction::new(&phase, C_0, INTEG_STEPS, boundaries1.ts.first().unwrap().1);
    let values1 = evaluate_function_between(&wave_func1, VIEW.0, 0.0, NUMBER_OF_POINTS);

    let (airy_wave_func2, mut boundaries2) = AiryWaveFunction::new(&phase, (0.0, VIEW.1));
    let wave_func2 =
        WkbWaveFunction::new(&phase, C_0, INTEG_STEPS, boundaries2.ts.last().unwrap().1);
    let values2 = evaluate_function_between(&wave_func2, 0.0, VIEW.1, NUMBER_OF_POINTS);
    let mut turning_point_boundaries = vec![];
    turning_point_boundaries.append(&mut boundaries1.ts);
    turning_point_boundaries.append(&mut boundaries2.ts);

    let mut data_file = File::create("data.txt").unwrap();

    let mut data_str: String = values1
        .par_iter()
        .filter(|p| {
            (turning_point_boundaries.clone())
                .iter()
                .map(order_ts)
                .map(|(t1, t2)| p.x < t1 || p.x > t2)
                .fold(true, |a, b| a && b)
        })
        .map(|p| -> String { format!("{} {}\n", p.x, p.y) })
        .reduce(|| String::new(), |s: String, current: String| s + &*current);

    data_str.push_str(
        &*values2
            .par_iter()
            .filter(|p| {
                (turning_point_boundaries.clone())
                    .iter()
                    .map(order_ts)
                    .map(|(t1, t2)| p.x < t1 || p.x > t2)
                    .fold(true, |a, b| a && b)
            })
            .map(|p| -> String { format!("{} {}\n", p.x, p.y) })
            .reduce(|| String::new(), |s: String, current: String| s + &*current),
    );

    let airy_wave_funcs = airy_wave_func1;
    let airy_data_str = airy_wave_funcs
        .iter()
        .map(|airy_wave_func| {
            let airy_values = evaluate_function_between(
                airy_wave_func,
                airy_wave_func.ts.0 - AIRY_EXTRA,
                airy_wave_func.ts.1 + AIRY_EXTRA,
                NUMBER_OF_POINTS,
            );

            airy_values
                .par_iter()
                .map(|p| -> String { format!("{} {}\n", p.x, p.y) })
                .reduce(|| String::new(), |s: String, current: String| s + &*current)
        })
        .fold(String::new(), |s: String, current: String| {
            s + "\n\n" + &*current
        });
    data_str.push_str(&*airy_data_str);

    let airy_wave_funcs = airy_wave_func2;

    let airy_data_str = airy_wave_funcs
        .iter()
        .map(|airy_wave_func| {
            let airy_values = evaluate_function_between(
                airy_wave_func,
                airy_wave_func.ts.0 - AIRY_EXTRA,
                airy_wave_func.ts.1 + AIRY_EXTRA,
                NUMBER_OF_POINTS,
            );

            airy_values
                .par_iter()
                .map(|p| -> String { format!("{} {}\n", p.x, p.y) })
                .reduce(|| String::new(), |s: String, current: String| s + &*current)
        })
        .fold(String::new(), |s: String, current: String| {
            s + "\n\n" + &*current
        });
    data_str.push_str(&*airy_data_str);

    let pot_re_to_c = Function {
        f: |x| complex(square(x), 0.0),
    };
    let potential = evaluate_function_between(&pot_re_to_c, VIEW.0, VIEW.1, NUMBER_OF_POINTS);

    let pot_str = potential
        .par_iter()
        .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
        .reduce(|| String::new(), |s: String, current: String| s + &*current);
    data_str.push_str("\n\n");
    data_str.push_str(&*pot_str);
    data_file.write_all((data_str).as_ref()).unwrap();

    let mut plot_3d_file = File::create("plot_3d.gnuplot").unwrap();

    let plot_3d_cmd = "splot \"data.txt\" i 0 u 1:2:3 t \"WKB\" w l, \"data.txt\" i 1 u 1:2:3 t \"Airy 1\" w l, \"data.txt\" i 2 u 1:2:3 t \"Ariy 2\" w l";
    plot_3d_file.write_all(plot_3d_cmd.as_ref()).unwrap();

    let mut plot_3d_file = File::create("plot.gnuplot").unwrap();

    let plot_3d_cmd = "plot \"data.txt\" i 0 u 1:2 t \"WKB\", \"data.txt\" i 1 u 1:2 t \"Airy 1\", \"data.txt\" i 2 u 1:2 t \"Ariy 2\"";
    plot_3d_file.write_all(plot_3d_cmd.as_ref()).unwrap();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn phase_off() {
    }
}
