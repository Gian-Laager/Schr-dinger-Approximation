use crate::newtons_method::derivative;
use crate::newtons_method::newtons_method;
use crate::newtons_method::newtons_method_max_iters;
use crate::*;
use num::signum;
use std::cmp::Ordering;

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

struct TGroup {
    t0: Option<f64>,
    ts: Option<Vec<(f64, f64)>>,
    tn: Option<f64>,
}

impl TGroup {
    pub fn new() -> TGroup {
        TGroup {
            t0: None,
            ts: None,
            tn: None,
        }
    }

    pub fn add_ts(&mut self, new_t: (f64, f64)) {
        if let Some(existing_ts) = &mut self.ts {
            existing_ts.push(new_t);
        } else {
            self.ts = Some(vec![new_t]);
        }
    }
}

pub struct AiryWaveFunction {
    pub c_a: Complex64,
    pub c_b: Complex64,
    u_1: f64,
    x_1: f64,
    ts: TGroup,
}

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

fn validity_func(phase: Phase) -> Box<dyn Fn(f64) -> f64> {
    Box::new(move |x: f64| {
        H_BAR / (2.0 * phase.mass).sqrt() * derivative(&|t| (phase.potential)(&t), x).abs()
            - ((phase.potential)(&x) - phase.energy).pow(2)
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
        let u_1_cube_root = Self::get_u_1_cube_root(u_1);
        let wkb_plus_1 = integrate(
            evaluate_function_between(phase, X_0, t.0, INTEG_STEPS),
            TRAPEZE_PER_THREAD,
        )
        .exp()
            / phase.eval(&t.0).sqrt();
        let wkb_minus_1 = (-integrate(
            evaluate_function_between(phase, X_0, t.0, INTEG_STEPS),
            TRAPEZE_PER_THREAD,
        ))
        .exp()
            / phase.eval(&t.0).sqrt();
        let wkb_plus_2 = integrate(
            evaluate_function_between(phase, X_0, t.1, INTEG_STEPS),
            TRAPEZE_PER_THREAD,
        )
        .exp()
            / phase.eval(&t.1).sqrt();
        let wkb_minus_2 = (-integrate(
            evaluate_function_between(phase, X_0, t.1, INTEG_STEPS),
            TRAPEZE_PER_THREAD,
        ))
        .exp()
            / phase.eval(&t.1).sqrt();

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

        return (c_a, c_b);
    }

    fn group_ts(zeros: &Vec<f64>, phase: &Phase) -> TGroup {
        let mut zeros = zeros.clone();
        zeros.sort_by(cmp_f64);
        let derivatives = zeros
            .iter()
            .map(|x| validity_func(phase.clone())(*x))
            .map(signum)
            .collect::<Vec<f64>>();

        if let Some(first) = derivatives.get(0) {
            if let Some(second) = derivatives.get(1) {
                todo!();
            } else {
                return TGroup {
                    t0: Some(*first),
                    ts: None,
                    tn: None,
                };
            }
        }

        todo!();
    }

    fn get_guess_for_ts(phase: &Phase, view: (f64, f64)) -> (f64, f64) {
        const MAX_TURNING_POINTS: i32 = 64;
        const ACCURACY: f64 = 1e-15;
        let zeros = (0..MAX_TURNING_POINTS)
            .into_par_iter()
            .map(|i| index_to_range(i as f64, 0.0, MAX_TURNING_POINTS as f64, view.0, view.1))
            .map(|x| newtons_method_max_iters(&validity_func(phase.clone()), x, ACCURACY, 10000))
            .collect::<Vec<Option<f64>>>();

        let mut unique_zeros = vec![];

        let mut result = TGroup::new();

        for z in zeros.iter().flatten() {
            if !unique_zeros.iter().any(|w: &f64| (w - *z).abs() < ACCURACY) {
                unique_zeros.push(*z);
            }
        }

        return (unique_zeros[0], unique_zeros[1]);
    }

    pub fn calc_ts(phase: &Phase, view: (f64, f64)) -> (f64, f64) {
        // let t1 = newtons_method(&validity_func, 2.5, 1e-5);
        // let t2 = newtons_method(&validity_func, 3.0, 1e-5);

        let (t1, t2) = Self::get_guess_for_ts(phase, view);

        // let t1 = signum(X_0) * f64::sqrt(phase.energy + H_BAR * H_BAR / phase.mass + f64::sqrt(H_BAR * H_BAR * (H_BAR * H_BAR + 2.0 * phase.mass * phase.energy)) / phase.mass);
        // let t2 = signum(X_0) * f64::sqrt(phase.energy + H_BAR * H_BAR / phase.mass - f64::sqrt(H_BAR * H_BAR * (H_BAR * H_BAR + 2.0 * phase.mass * phase.energy)) / phase.mass);

        println!("zeros = ({}, {})", t1, t2);

        return (t1, t2);
    }

    pub fn new(wave_func: &WaveFunction, view: (f64, f64)) -> AiryWaveFunction {
        let phase = wave_func.phase;
        let (t1, t2) = AiryWaveFunction::calc_ts(phase, view);
        let x_1 = newtons_method(&|x| (phase.potential)(&x) - phase.energy, t1, 1e-7);
        let u_1 = 2.0 * phase.mass / (H_BAR * H_BAR) * derivative(&|x| (phase.potential)(&x), x_1);

        let (c_a, c_b) = AiryWaveFunction::calc_c_a_and_c_b(
            phase,
            (t1, t2),
            (wave_func.c_plus, wave_func.c_minus),
            u_1,
            x_1,
        );

        AiryWaveFunction {
            c_a,
            c_b,
            u_1,
            x_1,
            ts: TGroup {
                t0: None,
                ts: Some(vec![(t1, t2)]),
                tn: None,
            },
        }
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
