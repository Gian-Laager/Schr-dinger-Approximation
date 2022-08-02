use crate::newtons_method::newtons_method;
use crate::newtons_method::newtons_method_max_iters;
use crate::newtons_method::{
    derivative, make_guess, newtons_method_find_new_zero, regula_falsi_bisection,
    regula_falsi_method, NewtonsMethodFindNewZero,
};
use crate::utils::cmp_f64;
use crate::*;
use num::integer::sqrt;
use num::{signum, zero};
use std::cmp::{min, Ordering};
use std::sync::Arc;
use scilib::constant::H;

const MAX_TURNING_POINTS: usize = 256;
const ACCURACY: f64 = 1e-9;

fn Ai(x: Complex64) -> Complex64 {
    let go_return;
    unsafe {
        go_return = airy_ai(x.re, x.im);
    }
    return complex(go_return.r0, go_return.r1);
}

fn Bi(x: Complex64) -> Complex64 {
    return -complex(0.0, 1.0) * Ai(x)
        + 2.0 * Ai(x * complex(-0.5, 3.0_f64.sqrt() / 2.0)) * complex(3_f64.sqrt() / 2.0, 0.5);
}

#[derive(Debug)]
pub struct TGroup {
    // pub t0: Option<f64>,
    pub ts: Vec<(f64, f64)>,
    // pub tn: Option<f64>,
}

impl TGroup {
    pub fn new() -> TGroup {
        TGroup { ts: vec![] }
    }

    pub fn add_ts(&mut self, new_t: (f64, f64)) {
        self.ts.push(new_t);
    }
}

pub struct AiryWaveFunction {
    pub c_a: Complex64,
    pub c_b: Complex64,
    u_1: f64,
    x_1: f64,
    pub ts: (f64, f64),
}

fn validity_func(phase: Phase) -> Box<dyn Fn(f64) -> f64> {
    Box::new(move |x: f64| {
        let momentum = |x| if (phase.potential)(&x) > phase.energy {
            (2.0 * phase.mass * ((phase.potential)(&x) - phase.energy)).sqrt()
        } else {
            (2.0 * phase.mass * (phase.energy - (phase.potential)(&x))).sqrt()
        };
        // H_BAR / (2.0 * phase.mass).sqrt() * derivative(&|t| (phase.potential)(&t), x).abs()
        //     - ((phase.potential)(&x) - phase.energy).pow(2)

        // derivative(&|x| H_BAR / momentum(x), x) - 1.0

        let v1 = -H_BAR * 2.0_f64.sqrt() * derivative(&|x| (phase.potential)(&x), x).abs() - 4.0 * ((phase.potential)(&x) - phase.energy) * ((-phase.mass * (phase.energy - (phase.potential)(&x))).abs()).sqrt();
        let v2 = -H_BAR * 2.0_f64.sqrt() * derivative(&|x| (phase.potential)(&x), x).abs() + 4.0 * ((phase.potential)(&x) - phase.energy) * ((-phase.mass * (phase.energy - (phase.potential)(&x))).abs()).sqrt();

        return if v1.abs() < v2.abs() {
            -v1
        } else {
            -v2
        };
    })
}

impl AiryWaveFunction {
    fn get_u_1_cube_root(u_1: f64) -> Complex64 {
        complex(u_1, 0.0).pow(1.0 / 3.0)
    }

    fn calc_c_a_and_c_b(
        phase: &Phase,
        t: (f64, f64),
        c_wkb: (f64, f64),
        u_1: f64,
        x_1: f64,
    ) -> (Complex64, Complex64) {
        let wkb_plus = |x| integrate(
            evaluate_function_between(phase, phase.x_0, x, INTEG_STEPS),
            TRAPEZE_PER_THREAD,
        )
            .exp()
            / phase.eval(&x).sqrt();

        let wkb_minus = |x| (-integrate(
            evaluate_function_between(phase, phase.x_0, x, INTEG_STEPS),
            TRAPEZE_PER_THREAD,
        ))
            .exp()
            / phase.eval(&x).sqrt();

        let u_1_cube_root = Self::get_u_1_cube_root(u_1);
        let wkb_plus_1 = wkb_plus(t.0);
        let wkb_minus_1 = wkb_minus(t.0);
        let wkb_plus_2 = wkb_plus(t.1);
        let wkb_minus_2 = wkb_minus(t.1);

        let airy_ai_1 = Ai(u_1_cube_root * (t.0 - x_1));
        let airy_bi_1 = Bi(u_1_cube_root * (t.0 - x_1));
        let airy_ai_2 = Ai(u_1_cube_root * (t.1 - x_1));
        let airy_bi_2 = Bi(u_1_cube_root * (t.1 - x_1));

        let c_a = ((-c_wkb.1 * (airy_bi_1 * wkb_minus_2 - airy_bi_2 * wkb_minus_1))
            / (airy_ai_1 * airy_bi_2 - airy_ai_2 * airy_bi_1))
            - ((c_wkb.0 * (airy_bi_1 * wkb_plus_2 - airy_bi_2 * wkb_plus_1))
            / (airy_ai_1 * airy_bi_2 - airy_ai_2 * airy_bi_1));
        let c_b = ((c_wkb.1 * (airy_ai_1 * wkb_minus_2 - airy_ai_2 * wkb_minus_1))
            / (airy_ai_1 * airy_bi_2 - airy_ai_2 * airy_bi_1))
            + ((c_wkb.0 * (airy_ai_1 * wkb_plus_2 - airy_ai_2 * wkb_plus_1))
            / (airy_ai_1 * airy_bi_2 - airy_ai_2 * airy_bi_1));

        // return (c_a, c_b);
        let c_airy1 = (wkb_plus_1 * c_wkb.0 +  c_wkb.1 * wkb_minus_1) / airy_ai_1;
        let c_airy2 = (wkb_plus_2 * c_wkb.0 +  c_wkb.1 * wkb_minus_2) / airy_ai_2;

        println!("wkb_a: {}, wkb_b: {},  c_a: {}, c_b: {}", c_wkb.0, c_wkb.1, (c_airy1 + c_airy2) / 2.0, c_b);
        println!("airy_ai_1: {}, airy_bi_1: {},  airy_ai_2: {}, airy_bi_2: {}", airy_ai_1, airy_bi_1, airy_ai_2, airy_bi_2);

        return ((c_airy1 + c_airy2) / 2.0, complex(0.0, 0.0))
    }

    fn group_ts(zeros: &Vec<f64>, phase: &Phase) -> TGroup {
        let mut zeros = zeros.clone();
        let valid = validity_func(phase.clone());

        zeros.sort_by(cmp_f64);
        let mut derivatives = zeros
            .iter()
            .map(|x| derivative(&valid, *x))
            .map(signum)
            .zip(zeros.clone())
            .collect::<Vec<(f64, f64)>>();

        let mut groups = TGroup { ts: vec![] };

        if let Some((deriv, z)) = derivatives.first() {
            if *deriv < 0.0 {
                let mut guess = z - ACCURACY.sqrt();
                let mut new_deriv = *deriv;
                let mut missing_t = *z;

                while new_deriv < 0.0 {
                    missing_t = regula_falsi_bisection(&valid, guess, -ACCURACY.sqrt(), ACCURACY);
                    new_deriv = signum(derivative(&valid, missing_t));
                    guess -= ACCURACY.sqrt();
                }

                derivatives.insert(0, (signum(derivative(&valid, missing_t)), missing_t));
            }
        }

        if let Some((deriv, z)) = derivatives.last() {
            if *deriv > 0.0 {
                let mut guess = z + ACCURACY.sqrt();
                let mut new_deriv = *deriv;
                let mut missing_t = *z;

                while new_deriv > 0.0 {
                    missing_t = regula_falsi_bisection(&valid, guess, ACCURACY.sqrt(), ACCURACY);
                    new_deriv = signum(derivative(&valid, missing_t));
                    guess += ACCURACY.sqrt();
                }

                derivatives.push((signum(derivative(&valid, missing_t)), missing_t));
            }
        }

        assert_eq!(derivatives.len() % 2, 0);

        for i in (0..derivatives.len()).step_by(2) {
            let (t1_deriv, t1) = derivatives[i];
            let (t2_deriv, t2) = derivatives[i + 1];
            assert!(t1_deriv > 0.0);
            assert!(t2_deriv < 0.0);

            groups.add_ts((t1, t2));
        }

        println!("{:?}", groups.ts);
        return groups;
    }

    fn find_zeros(phase: &Phase, view: (f64, f64)) -> Vec<f64> {
        let mut zeros =
            NewtonsMethodFindNewZero::new(validity_func(phase.clone()), ACCURACY, 1e4 as usize);


        (0..MAX_TURNING_POINTS).into_iter().for_each(|_| {
            let modified_func = |x| zeros.modified_func(x);

            let guess = make_guess(&modified_func, view, 1000);
            guess.map(|g| zeros.next_zero(g));
        });

        let view = if view.0 < view.1 {
            view
        } else {
            (view.1, view.0)
        };
        let unique_zeros = zeros
            .get_previous_zeros()
            .iter()
            .filter(|x| **x > view.0 && **x < view.1)
            .map(|x| *x)
            .collect::<Vec<f64>>();
        return unique_zeros;
    }

    pub fn calc_ts(phase: &Phase, view: (f64, f64)) -> TGroup {
        // return TGroup{ts:vec![(-4.692, -4.255), (4.255, 4.692)]};
        let zeros = Self::find_zeros(phase, view);
        println!("zeros: {:?}", zeros);
        return Self::group_ts(&zeros, phase);
    }

    pub fn new(wave_func: &WkbWaveFunction, view: (f64, f64)) -> (Vec<AiryWaveFunction>, TGroup) {
        let phase = wave_func.phase;
        let turning_point_boundaries = AiryWaveFunction::calc_ts(phase, view);

        let funcs = turning_point_boundaries
            .ts
            .iter()
            .map(|(t1, t2)| {
                let x_1 = newtons_method(
                    &|x| (phase.potential)(&x) - phase.energy,
                    (*t1 + *t2) / 2.0,
                    1e-7,
                );
                let u_1 = 2.0 * phase.mass / (H_BAR * H_BAR)
                    * derivative(&|x| (phase.potential)(&x), x_1);
                // let u_1 = |x| -2.0 * phase.mass * ((phase.potential)(&x) - phase.energy) / (H_BAR * H_BAR * (x - x_1));
                println!("u_1 = {}, x_1 = {}", u_1, x_1);
                let (c_a, c_b) = AiryWaveFunction::calc_c_a_and_c_b(
                    phase,
                    (*t1, *t2),
                    (wave_func.c_plus, wave_func.c_minus),
                    u_1,
                    x_1,
                );

                AiryWaveFunction {
                    c_a,
                    c_b,
                    u_1,
                    x_1,
                    ts: (*t1, *t2),
                }
            })
            .collect();
        return (funcs, turning_point_boundaries);
    }
}

impl ReToC for AiryWaveFunction {
    fn eval(&self, x: &f64) -> Complex64 {
        let u_1_cube_root = Self::get_u_1_cube_root(self.u_1);
        let ai = self.c_a * Ai(u_1_cube_root * (x - self.x_1));
        let bi = self.c_b * Bi(u_1_cube_root * (x - self.x_1));
        return ai + bi;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn airy_func_plot() {
        let airy_ai = Function::new(|x| Ai(complex(x, 0.0)));
        let airy_bi = Function::new(|x| Bi(complex(x, 0.0)));
        let values = evaluate_function_between(&airy_ai, -10.0, 5.0, NUMBER_OF_POINTS);

        let mut data_file = File::create("airy.txt").unwrap();

        let data_str_ai: String = values
            .par_iter()
            .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
            .reduce(|| String::new(), |s: String, current: String| s + &*current);

        let values_bi = evaluate_function_between(&airy_bi, -5.0, 2.0, NUMBER_OF_POINTS);

        let data_str_bi: String = values_bi
            .par_iter()
            .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
            .reduce(|| String::new(), |s: String, current: String| s + &*current);

        data_file
            .write_all((data_str_ai + "\n\n" + &*data_str_bi).as_ref())
            .unwrap()
    }
}
