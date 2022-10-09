#![feature(unboxed_closures)]

mod airy;
mod airy_wave_func;
mod energy;
mod integrals;
mod newtons_method;
mod potentials;
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
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use tokio;

const INTEG_STEPS: usize = 64000;
const TRAPEZE_PER_THREAD: usize = 1000;
const NUMBER_OF_POINTS: usize = 100000;

const PLOT_POTENTIAL: bool = false;
const SEPARATE_FUNCTIONS: bool = false;
const RENORMALIZE: bool = false;
const NORMALIZE_POTENTIAL: bool = false;

const MASS: f64 = 2.0;
const C_0: f64 = 0.18 / 0.46;
const AIRY_EXTRA: f64 = 0.5;
const N_ENERGY: usize = 12;

const APPROX_INF: (f64, f64) = (-200.0, 200.0);
const ENERGY_INF: f64 = 1e6;
const VIEW_FACTOR: f64 = 1.5;

pub struct Derivative<'a> {
    f: &'a dyn Func<f64, Complex64>,
}

impl Func<f64, Complex64> for Derivative<'_> {
    fn eval(&self, x: f64) -> Complex64 {
        derivative(&|x| self.f.eval(x), x)
    }
}

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

    fn new<F: Fn(f64) -> f64 + Sync + Send>(
        energy: f64,
        mass: f64,
        potential: &'static F,
        phase_off: f64,
    ) -> Phase {
        return Phase {
            energy,
            mass,
            potential: Arc::new(potential),
            phase_off,
        };
    }

    fn sqrt_momentum(&self, x: f64) -> f64 {
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
    potentials::mexican_hat(x)
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

fn identiy(c: Complex64) -> Complex64 {
    c
}

fn conjugate(c: Complex64) -> Complex64 {
    c.conj()
}

fn negative(c: Complex64) -> Complex64 {
    -c
}

fn negative_conj(c: Complex64) -> Complex64 {
    -c.conj()
}

fn sign_match(mut f1: f64, mut f2: f64) -> bool {
    return f1.signum() == f2.signum();
}

fn sign_match_complex(mut c1: Complex64, mut c2: Complex64) -> bool {
    if c1.re.abs() < c1.im.abs() {
        c1.re = 0.0;
    }

    if c1.im.abs() < c1.re.abs() {
        c1.im = 0.0;
    }

    if c2.re.abs() < c2.im.abs() {
        c2.re = 0.0;
    }

    if c2.im.abs() < c2.re.abs() {
        c2.im = 0.0;
    }

    return sign_match(c1.re, c2.re) && sign_match(c1.im, c2.im);
}

fn find_best_op(
    phase: Arc<Phase>,
    wkb: &WkbWaveFunction,
    boundary: f64,
    previous: f64,
    previous_op: fn(Complex64) -> Complex64,
) -> fn(Complex64) -> Complex64 {
    let wkb_prev = WkbWaveFunction::new(phase.clone(), C_0, INTEG_STEPS, previous);
    let deriv_prev = derivative(
        &|x| previous_op(wkb_prev.eval(x)),
        (boundary + previous) / 2.0,
    );
    let val_prev = previous_op(wkb_prev.eval((boundary + previous) / 2.0));
    let deriv = derivative(&|x| wkb.eval(x), (boundary + previous) / 2.0);
    let val = wkb.eval((boundary + previous) / 2.0);

    println!(
        "deriv: {:.17}, deriv_prev: {:.17}, val: {:.17}, val_prev: {:.17}",
        deriv, deriv_prev, val, val_prev
    );
    if (phase.potential)((boundary + previous) / 2.0) >= phase.energy {
        previous_op
    } else if sign_match_complex(conjugate(deriv), deriv_prev)
        && sign_match_complex(conjugate(val), val_prev)
    {
        println!("conjugate");
        conjugate
    } else if sign_match_complex(negative_conj(deriv), deriv_prev)
        && sign_match_complex(negative_conj(val), val_prev)
    {
        println!("negative_conj");
        negative_conj
    } else if sign_match_complex(negative(deriv), deriv_prev)
        && sign_match_complex(negative(val), val_prev)
    {
        println!("negative");
        negative
    } else {
        println!("identiy");
        identiy
    }
}

fn main() {


    let output_dir = Path::new("output");
    std::env::set_current_dir(&output_dir).unwrap();
    let energy = energy::nth_energy(N_ENERGY, 1.0, &potential, APPROX_INF);
    println!("{} Energy: {}", N_ENERGY, energy);

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
    let mut phase = Phase::new(energy, MASS, &potential, 0.0);
    let (_, t_boundaries) = AiryWaveFunction::new(Arc::new(phase.clone()), (view.0, view.1));
    println!(
        "{:?}",
        t_boundaries.ts.iter().map(|p| p.1).collect::<Vec<f64>>()
    );

    if t_boundaries.ts.len() != 0 {
        // conjecture based on observations in all the plots
        phase.phase_off = f64::consts::PI / (t_boundaries.ts.len() as f64) - f64::consts::PI / 2.0;
    }
    println!("phase offset: {}", phase.phase_off);

    let phase = Arc::new(phase);
    let mut all_values: Vec<Point<f64, Complex64>> = vec![];

    let (airy_wave_func, boundaries) = AiryWaveFunction::new(phase.clone(), (view.0, view.1));

    let (wave_funcs, turning_point_boundaries) = if boundaries.ts.len() == 0 {
        println!("No turning points found in view! Results might be in accurate");
        let wkb1 = WkbWaveFunction::new(phase.clone(), C_0, INTEG_STEPS, APPROX_INF.0);
        let wkb2 = WkbWaveFunction::new(phase.clone(), C_0, INTEG_STEPS, APPROX_INF.1);
        let op = find_best_op(phase.clone(), &wkb2, APPROX_INF.1, APPROX_INF.0, identiy);
        let center = (view.0 + view.1) / 2.0;
        (
            vec![
                (
                    wkb1,
                    (view.0, center),
                    None,
                    identiy as fn(Complex64) -> Complex64,
                ),
                (wkb2, (center, view.1), None, op),
            ],
            vec![],
        )
    } else {
        let turning_point_boundaries: Vec<(f64, f64)> = boundaries.ts.iter().map(|p| p.0).collect();

        let turning_points: Vec<f64> = [
            vec![2.0 * view.0 - boundaries.ts.first().unwrap().1],
            boundaries.ts.iter().map(|p| p.1).collect(),
            vec![2.0 * view.1 - boundaries.ts.last().unwrap().1],
        ]
        .concat();

        let mut is_first = true;
        let mut previous_op: fn(Complex64) -> Complex64 = identiy;
        let wave_funcs = turning_points
            .iter()
            .zip(turning_points.iter().skip(1).zip(airy_wave_func.clone()))
            .zip(turning_points.iter().skip(2))
            .map(
                |((previous, (boundary, airy_wave_func)), next)| -> (
                    WkbWaveFunction,
                    (f64, f64),
                    Option<AiryWaveFunction>,
                    fn(Complex64) -> Complex64,
                ) {
                    let wkb = WkbWaveFunction::new(phase.clone(), C_0, INTEG_STEPS, *boundary);
                    let op: fn(Complex64) -> Complex64 = if is_first {
                        is_first = false;
                        identiy
                    } else {
                        find_best_op(phase.clone(), &wkb, *boundary, *previous, previous_op)
                    };
                    previous_op = op;
                    (
                        wkb,
                        ((boundary + previous) / 2.0, (next + boundary) / 2.0),
                        Some(airy_wave_func),
                        op,
                    )
                },
            )
            .collect::<Vec<(
                WkbWaveFunction,
                (f64, f64),
                Option<AiryWaveFunction>,
                fn(Complex64) -> Complex64,
            )>>();

        (wave_funcs, turning_point_boundaries)
    };

    let values = wave_funcs
        .par_iter()
        .map(|(w, (a, b), _, op)| {
            evaluate_function_between(w, *a, *b, NUMBER_OF_POINTS)
                .iter()
                .map(|p| Point { x: p.x, y: op(p.y) })
                .collect::<Vec<Point<f64, Complex64>>>()
        })
        .flatten()
        .filter(|p| {
            (turning_point_boundaries.clone())
                .iter()
                .map(order_ts)
                .map(|(t1, t2)| {
                    let distance = t2 - t1;
                    !wave_function_builder::is_in_range(
                        (t1 - AIRY_EXTRA * distance, t2 + AIRY_EXTRA * distance),
                        p.x,
                    )
                })
                .fold(true, |a, b| a && b)
        })
        .collect::<Vec<Point<f64, Complex64>>>();

    let airy_values = wave_funcs
        .iter()
        .filter(|(_, _, airy_wave_func, _)| airy_wave_func.is_some())
        .map(|(wkb, _, airy_wave_func, op)| {
            let distance =
                airy_wave_func.as_ref().unwrap().ts.1 - airy_wave_func.as_ref().unwrap().ts.0;

            let airy_values = evaluate_function_between(
                airy_wave_func.as_ref().unwrap(),
                airy_wave_func.as_ref().unwrap().ts.0,
                airy_wave_func.as_ref().unwrap().ts.1,
                NUMBER_OF_POINTS,
            );

            let joint_left = wave_function_builder::Joint {
                left: Arc::new(airy_wave_func.as_ref().unwrap().clone()),
                right: Arc::new(wkb.clone()),
                cut: airy_wave_func.as_ref().unwrap().ts.0,
                delta: -distance * AIRY_EXTRA,
            };

            let joint_right = wave_function_builder::Joint {
                left: Arc::new(wkb.clone()),
                right: Arc::new(airy_wave_func.as_ref().unwrap().clone()),
                cut: airy_wave_func.as_ref().unwrap().ts.1,
                delta: distance * AIRY_EXTRA,
            };

            let joint_vals_l = evaluate_function_between(
                &joint_left,
                joint_left.range().0,
                joint_left.range().1,
                NUMBER_OF_POINTS,
            );

            let joint_vals_r = evaluate_function_between(
                &joint_right,
                joint_right.range().0,
                joint_right.range().1,
                NUMBER_OF_POINTS,
            );

            let airy_values: Vec<Point<f64, Complex64>> =
                [joint_vals_l, airy_values, joint_vals_r].concat();

            airy_values
                .par_iter()
                .map(|p| Point { x: p.x, y: op(p.y) })
                .collect::<Vec<Point<f64, Complex64>>>()
        })
        .collect::<Vec<Vec<Point<f64, Complex64>>>>();

    if SEPARATE_FUNCTIONS {
        let mut data_file = File::create("data.txt").unwrap();

        let mut data_str: String = values
            .par_iter()
            .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
            .reduce(|| String::new(), |s: String, current: String| s + &*current);

        let airy_data_str = airy_values
            .iter()
            .map(|vals| -> String {
                vals.par_iter()
                    .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
                    .reduce(|| String::new(), |s: String, current: String| s + &*current)
            })
            .fold(String::new(), |s: String, current: String| {
                s + "\n\n" + &*current
            });

        data_str.push_str(&*airy_data_str);

        if PLOT_POTENTIAL {
            let pot_re_to_c = Function {
                f: |x| complex(potential(x), 0.0),
            };
            let potential =
                evaluate_function_between(&pot_re_to_c, view.0, view.1, NUMBER_OF_POINTS);

            let potential = if NORMALIZE_POTENTIAL {
                let airy_max = airy_values
                    .par_iter()
                    .flatten()
                    .map(|p| p.y.re.max(p.y.im))
                    .max_by(cmp_f64)
                    .unwrap();

                let wkb_max = values
                    .par_iter()
                    .map(|p| p.y.re.max(p.y.im))
                    .max_by(cmp_f64)
                    .unwrap();

                let psi_max = airy_max.max(wkb_max);
                let pot_max = potential
                    .par_iter()
                    .map(|p| p.y.re.max(p.y.im))
                    .max_by(cmp_f64)
                    .unwrap();

                potential
                    .par_iter()
                    .map(|p| Point {
                        x: p.x,
                        y: p.y * psi_max / pot_max,
                    })
                    .collect()
            } else {
                potential
            };

            let pot_str = potential
                .par_iter()
                .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
                .reduce(|| String::new(), |s: String, current: String| s + &*current);
            data_str.push_str("\n\n");
            data_str.push_str(&*pot_str);
        }
        data_file.write_all((data_str).as_ref()).unwrap();

        let mut plot_3d_file = File::create("plot_3d.gnuplot").unwrap();

        // \"data.txt\" i 1 u 1:2:3 t \"Airy 1\" w l,
        let mut plot_3d_cmd: String = "splot \"data.txt\" i 0 u 1:2:3 t \"WKB\" w l".to_string();

        for i in 1..=airy_wave_func.len() {
            plot_3d_cmd += &format!(", \"data.txt\" i {} u 1:2:3 t \"Airy {}\" w l", i, i);
        }

        if PLOT_POTENTIAL {
            plot_3d_cmd += &format!(", \"data.txt\" i {} u 1:2:3 t \"Potential\" w l", airy_wave_func.len() + 1);
        }

        plot_3d_file.write_all(plot_3d_cmd.as_ref()).unwrap();

        let mut plot_file = File::create("plot.gnuplot").unwrap();

        let mut plot_cmd = "plot \"data.txt\" i 0 u 1:2 t \"WKB\" w l".to_string();

        for i in 1..=airy_wave_func.len() {
            plot_cmd += &format!(", \"data.txt\" i {} u 1:2 t \"Airy {}\" w l", i, i);
        }

        if PLOT_POTENTIAL {
            plot_cmd += &format!(", \"data.txt\" i {} u 1:2 t \"Potential\" w l", airy_wave_func.len() + 1);
        }

        plot_file.write_all(plot_cmd.as_ref()).unwrap();

        let mut plot_imag_file = File::create("plot_im.gnuplot").unwrap();

        let mut plot_imag_cmd = "plot \"data.txt\" i 0 u 1:3 t \"WKB\" w l".to_string();

        for i in 1..=airy_wave_func.len() {
            plot_imag_cmd += &format!(", \"data.txt\" i {} u 1:3 t \"Airy {}\" w l", i, i);
        }

        if PLOT_POTENTIAL {
            plot_imag_cmd += &format!(", \"data.txt\" i {} u 1:2 t \"Potential\" w l", airy_wave_func.len() + 1);
        }

        plot_imag_file.write_all(plot_imag_cmd.as_ref()).unwrap();
    } else {
        all_values = [
            all_values,
            values
                .par_iter()
                .filter(|p| {
                    (turning_point_boundaries.clone())
                        .iter()
                        .map(order_ts)
                        .map(|(t1, t2)| {
                            let distance = t2 - t1;
                            !wave_function_builder::is_in_range(
                                (t1 - AIRY_EXTRA * distance, t2 + AIRY_EXTRA * distance),
                                p.x,
                            )
                        })
                        .fold(true, |a, b| a && b)
                })
                .map(|p| p.clone())
                .collect::<Vec<Point<f64, Complex64>>>(),
        ]
        .concat();

        all_values = [
            all_values,
            airy_values
                .par_iter()
                .flatten()
                .map(|p| p.clone())
                .collect(),
        ]
        .concat();

        all_values.sort_by(|p1, p2| cmp_f64(&p1.x, &p2.x));

        if RENORMALIZE {
            let total_probability: f64 = all_values.par_iter().map(|p| p.y.norm_sqr()).sum();
            println!("total probability: {}", total_probability);
            all_values = all_values
                .par_iter()
                .map(|p| Point {
                    x: p.x,
                    y: p.y / total_probability,
                })
                .collect::<Vec<Point<f64, Complex64>>>();
        }

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

    #[test]
    fn sign_check_complex_test() {
        let range = (-50.0, 50.0);
        let n = 100000;
        for ri1 in 0..n {
            for ii1 in 0..n {
                for ri2 in 0..n {
                    for ii2 in 0..n {
                        let re1 = index_to_range(ri1 as f64, 0.0, n as f64, range.0, range.1);
                        let im1 = index_to_range(ii1 as f64, 0.0, n as f64, range.0, range.1);
                        let re2 = index_to_range(ri2 as f64, 0.0, n as f64, range.0, range.1);
                        let im2 = index_to_range(ii2 as f64, 0.0, n as f64, range.0, range.1);

                        assert_eq!(
                            sign_match_complex(complex(re1, im1), complex(re2, im2)),
                            sign_match_complex(complex(re2, im2), complex(re1, im1))
                        );
                    }
                }
            }
        }
    }
}
