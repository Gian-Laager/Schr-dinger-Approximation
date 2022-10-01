#![feature(unboxed_closures)]

mod airy;
mod airy_wave_func;
mod energy;
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
use std::io;
use std::io::Write;
use std::process::Command;
use tokio;
use std::sync::Arc;

const INTEG_STEPS: usize = 64000;
const TRAPEZE_PER_THREAD: usize = 1000;
const NUMBER_OF_POINTS: usize = 100000;

const MASS: f64 = 1.0;
const C_0: f64 = 1.0;
const AIRY_EXTRA: f64 = 0.1;
const N_ENERGY: usize = 20;

const APPROX_INF: (f64, f64) = (-200.0, 200.0);
const VIEW_FACTOR: f64 = 1.5;

#[derive(Clone)]
pub struct Phase {
    energy: f64,
    mass: f64,
    potential: Arc<dyn Fn(f64) -> f64 + Send + Sync>,
    phase_off: f64,
}

impl Phase {
    fn default() -> Phase {
        Phase {
            energy: 0.0,
            mass: 0.0,
            potential: Arc::new(|_x| 0.0),
            phase_off: f64::consts::PI / 4.0,
        }
    }

    fn new<F: Fn(f64) -> f64 + Sync + Send>(energy: f64, mass: f64, potential: &'static F, phase_off: f64) -> Phase {
        return Phase {
            energy,
            mass,
            potential: Arc::new(potential),
            phase_off,
        };
    }

    fn momentum(&self, x: f64) -> f64 {
        self.eval(x).abs().sqrt()
    }
}

impl Func<f64, f64> for Phase {
    fn eval(&self, x: f64) -> f64 {
        (2.0 * self.mass * ((self.potential)(x) - self.energy))
            .abs()
            .sqrt()
    }
}

fn potential(x: f64) -> f64 {
    // (x - 2.0).powi(4) * (x + 2.0).powi(4)
    x * x
}

fn order_ts((t1, t2): &(f64, f64)) -> (f64, f64) {
    return if t1 > t2 { (*t2, *t1) } else { (*t1, *t2) };
}

fn get_float_from_user(message: &str) -> f64 {
    loop {
        println!("{}", message);
        let mut input = String::new();

        // io::stdout().lock().write(message.as_ref()).unwrap();
        io::stdin()
            .read_line(&mut input)
            .expect("Not a valid string");
        println!("");
        let num = input.trim().parse();
        if num.is_ok() {
            return num.unwrap();
        }
    }
}

fn get_user_bounds() -> (f64, f64) {
    let user_bound_lower: f64 = get_float_from_user("Lower Bound: ");

    let user_bound_upper: f64 = get_float_from_user("Upper_bound: ");
    return (user_bound_lower, user_bound_upper);
}
fn ask_user_for_view(lower_bound: Option<f64>, upper_bound: Option<f64>) -> (f64, f64) {
    println!("Failed to determine boundary of the graph automatically.");
    println!("Pleas enter values manualy.");
    lower_bound.map(|b| println!("(Suggestion for lower bound: {})", b));
    upper_bound.map(|b| println!("(Suggestion for upper bound: {})", b));

    return get_user_bounds();
}

fn main() {
    let energy = energy::nth_energy(N_ENERGY, 1.0, &potential, APPROX_INF);
    println!("Energy: {}", energy);

    let lower_bound = newtons_method::newtons_method_max_iters(
        &|x| potential(x) - energy,
        APPROX_INF.0,
        1e-7,
        100000,
    );
    let upper_bound = newtons_method::newtons_method_max_iters(
        &|x| potential(x) - energy,
        APPROX_INF.1,
        1e-7,
        100000,
    );

    let view = if lower_bound.is_none() || upper_bound.is_none() {
        ask_user_for_view(lower_bound, upper_bound)
    } else {
        (
            lower_bound.unwrap() * VIEW_FACTOR,
            upper_bound.unwrap() * VIEW_FACTOR,
        )
    };

    println!("View: {:?}", view);
    let phase = Arc::new(Phase::new(energy, MASS, &potential, f64::consts::PI / 4.0));
    let (_, t_boundaries) = AiryWaveFunction::new(phase.clone(), (view.0, view.1));
    println!(
        "{:?}",
        t_boundaries.ts.iter().map(|p| p.1).collect::<Vec<f64>>()
    );

    let (airy_wave_func, boundaries) = AiryWaveFunction::new(phase.clone(), (view.0, view.1));

    if boundaries.ts.len() == 0 {
        panic!("No turning points found!");
    }

    let turning_point_boundaries: Vec<(f64, f64)> = boundaries.ts.iter().map(|p| p.0).collect();

    let turning_points: Vec<f64> = [
        vec![2.0 * view.0 - boundaries.ts.first().unwrap().1],
        boundaries.ts.iter().map(|p| p.1).collect(),
        vec![2.0 * view.1 - boundaries.ts.last().unwrap().1],
    ]
    .concat();

    let wave_funcs = turning_points
        .iter()
        .zip(turning_points.iter().skip(1))
        .zip(turning_points.iter().skip(2))
        .map(
            |((previous, boundary), next)| -> (WkbWaveFunction, (f64, f64)) {
                println!(
                    "WKB between: ({:.12}, {:.12}), prev: {:.12}, current: {:.12}, next: {:.12}",
                    (boundary + previous) / 2.0,
                    (next + boundary) / 2.0,
                    previous,
                    boundary,
                    next
                );
                (
                    WkbWaveFunction::new(phase.clone(), C_0, INTEG_STEPS, *boundary),
                    ((boundary + previous) / 2.0, (next + boundary) / 2.0),
                )
            },
        )
        .collect::<Vec<(WkbWaveFunction, (f64, f64))>>();

    let values = wave_funcs
        .par_iter()
        .map(|(w, (a, b))| {
            println!("eval between: ({:.12}, {:.12})", a, b);
            evaluate_function_between(w, *a, *b, NUMBER_OF_POINTS)
        })
        .flatten()
        .collect::<Vec<Point<f64, Complex64>>>();

    let mut data_file = File::create("data.txt").unwrap();

    let mut data_str: String = values
        .par_iter()
        .filter(|p| {
            (turning_point_boundaries.clone())
                .iter()
                .map(order_ts)
                .map(|(t1, t2)| p.x < t1 || p.x > t2)
                .fold(true, |a, b| a && b)
        })
        .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
        .reduce(|| String::new(), |s: String, current: String| s + &*current);

    let airy_data_str = airy_wave_func
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
                .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
                .reduce(|| String::new(), |s: String, current: String| s + &*current)
        })
        .fold(String::new(), |s: String, current: String| {
            s + "\n\n" + &*current
        });
    data_str.push_str(&*airy_data_str);

    let pot_re_to_c = Function {
        f: |x| complex(potential(x), 0.0),
    };
    let potential = evaluate_function_between(&pot_re_to_c, view.0, view.1, NUMBER_OF_POINTS);

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

    let mut plot_file = File::create("plot.gnuplot").unwrap();

    let plot_cmd = "plot \"data.txt\" i 0 u 1:2 t \"WKB\" w l, \"data.txt\" i 1 u 1:2 t \"Airy 1\" w l, \"data.txt\" i 2 u 1:2 t \"Ariy 2\" w l";
    plot_file.write_all(plot_cmd.as_ref()).unwrap();
}

#[cfg(test)]
mod test {
    use super::*;
    use std::cmp::Ordering;

    fn pot(x: f64) -> f64 {
        1.0 / (x * x)
    }

    fn pot_in(x: f64) -> f64 {
        1.0 / x.sqrt()
    }

    #[test]
    fn phase_off() {
        let energy_cond = |e: f64| -> f64 { (0.5 * (e - 0.5)) % 1.0 };

        let integ = Function::<f64, f64>::new(energy_cond);
        let mut values = evaluate_function_between(&integ, 0.0, 5.0, NUMBER_OF_POINTS);
        let sort_func =
            |p1: &Point<f64, f64>, p2: &Point<f64, f64>| -> Ordering { cmp_f64(&p1.x, &p2.x) };
        values.sort_by(sort_func);

        let mut data_file = File::create("energy.txt").unwrap();

        let data_str: String = values
            .par_iter()
            .map(|p| -> String { format!("{} {}\n", p.x, p.y) })
            .reduce(|| String::new(), |s: String, current: String| s + &*current);

        data_file.write_all((data_str).as_ref()).unwrap()
    }
}
