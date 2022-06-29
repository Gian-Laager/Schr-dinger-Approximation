use std::ops::{Add, Mul, Sub};
use std::process::Output;
use num::abs;
use num::traits::MulAdd;
use crate::*;

pub struct AiryWaveFunction {
    pub c_a: Complex64,
    pub c_b: Complex64,
    u_1: f64,
    x_1: f64,
    ts: Vec<(f64, f64)>,
}

impl AiryWaveFunction {
    fn calc_c_a_and_c_b(phase: &Phase, t: (f64, f64), c_wkb: (f64, f64), u_1: f64, x_1: f64) -> (Complex64, Complex64) {
        let wkb_plus_1 = integrate(evaluate_function_between(phase, X_0, t.0, INTEG_STEPS), TRAPEZE_PER_THREAD).exp() / phase.eval(&t.0).sqrt();
        let wkb_minus_1 = (-integrate(evaluate_function_between(phase, X_0, t.0, INTEG_STEPS), TRAPEZE_PER_THREAD)).exp() / phase.eval(&t.0).sqrt();
        let wkb_plus_2 = integrate(evaluate_function_between(phase, X_0, t.1, INTEG_STEPS), TRAPEZE_PER_THREAD).exp() / phase.eval(&t.1).sqrt();
        let wkb_minus_2 = (-integrate(evaluate_function_between(phase, X_0, t.1, INTEG_STEPS), TRAPEZE_PER_THREAD)).exp() / phase.eval(&t.1).sqrt();

        let airy_ai_1 = Ai(complex(u_1, 0.0).pow(1.0 / 3.0) * (t.0 - x_1));
        let airy_bi_1 = Bi(complex(u_1, 0.0).pow(1.0 / 3.0) * (t.0 - x_1));
        let airy_ai_2 = Ai(complex(u_1, 0.0).pow(1.0 / 3.0) * (t.1 - x_1));
        let airy_bi_2 = Bi(complex(u_1, 0.0).pow(1.0 / 3.0) * (t.1 - x_1));

        let c_a = ((-c_wkb.1 * (airy_bi_1 * wkb_minus_2 - airy_bi_2 * wkb_minus_1)) / (airy_ai_1 * airy_bi_2 - airy_ai_2 * airy_bi_1)) - ((c_wkb.0 * (airy_bi_1 * wkb_plus_2 - airy_bi_2 * wkb_plus_1)) / (airy_ai_1 * airy_bi_2 - airy_ai_2 * airy_bi_1));
        let c_b = ((c_wkb.1 * (airy_ai_1 * wkb_minus_2 - airy_ai_2 * wkb_minus_1)) / (airy_ai_1 * airy_bi_2 - airy_ai_2 * airy_bi_1)) + ((c_wkb.0 * (airy_ai_1 * wkb_plus_2 - airy_ai_2 * wkb_plus_1)) / (airy_ai_1 * airy_bi_2 - airy_ai_2 * airy_bi_1));

        return (c_a, c_b);
    }

    pub fn calc_ts(phase: &Phase) -> (f64, f64) {
        let function = |x: &f64| -> f64 {
            H_BAR / (2.0 * phase.mass).sqrt() * abs(derivative(&phase.potential, *x)) - ((phase.potential)(&x) - phase.energy).pow(2)
        };

        let mut zeros: Vec<f64> = vec![];

        const INITIAL_NUMBER_OF_GUESSES: i64 = 100;
        let guesses_result = (0..INITIAL_NUMBER_OF_GUESSES).into_par_iter()
            .map(|i| index_to_range(i as f64, 0.0, INITIAL_NUMBER_OF_GUESSES as f64, VIEW.0, VIEW.1))
            .map(|x: f64| (x, function(&x)))
            .map(|(x, y)| (x, signum(y)))
            .collect::<Vec<(f64, f64)>>();

        let guessed_zeros_indices = guesses_result.iter()
            .filter(|(x_, y)| *y == 0.0)
            .map(|(x, y_)| zeros.push(*x));


        let mut zeros_between: Vec<(f64, f64)> = vec![];

        for i in 0..(guesses_result.len() - 1) {
            if guesses_result[i].1 != guesses_result[i + 1].1
            {
                zeros_between.push((guesses_result[i].0, guesses_result[i + 1].0))
            }
        }

        let zeros = zeros_between.iter().map(|(x1, x2)| (x1 + x2) / 2.0).map(|guess| newtons_method(&function, guess)).collect::<Vec<f64>>();

        zeros_between.iter().for_each(|(x1, x2)| println!("between: {}, {}", x1, x2));

        let t1 = signum(X_0) * f64::sqrt(phase.energy + H_BAR * H_BAR / phase.mass + f64::sqrt(H_BAR * H_BAR * (H_BAR * H_BAR + 2.0 * phase.mass * phase.energy)) / phase.mass);
        let t2 = signum(X_0) * f64::sqrt(phase.energy + H_BAR * H_BAR / phase.mass - f64::sqrt(H_BAR * H_BAR * (H_BAR * H_BAR + 2.0 * phase.mass * phase.energy)) / phase.mass);
        return (t1, t2);
    }

    pub fn new(wave_func: &WaveFunction) -> AiryWaveFunction {
        let phase = wave_func.phase;
        let (t1, t2) = AiryWaveFunction::calc_ts(phase);
        let x_1 = (t1 + t2) / 2.0;
        let u_1 = 2.0 * phase.mass / (H_BAR * H_BAR) * ((phase.potential)(&t1) - (phase.potential)(&t2)) / (t1 - t2);

        let (c_a, c_b) = AiryWaveFunction::calc_c_a_and_c_b(phase, (t1, t2), (wave_func.c_plus, wave_func.c_minus), u_1, x_1);

        AiryWaveFunction { c_a, c_b, u_1, x_1, ts: vec![(t1, t2)] }
    }
}

impl ReToC for AiryWaveFunction {
    fn eval(&self, x: &f64) -> Complex64 {
        let u_1_cube_root = complex(self.u_1, 0.0).pow(1.0 / 3.0);
        let ai = self.c_a * Ai(u_1_cube_root * (x - self.x_1));
        let bi = self.c_b * Bi(u_1_cube_root * (x - self.x_1));
        return ai + bi;
    }
}

fn derivative<F>(f: &F, x: f64) -> f64
    where F: Fn(&f64) -> f64 {
    let epsilon = f64::EPSILON.sqrt() * x.abs().log2().round().exp2();
    (f(&(x + epsilon / 2.0)) - f(&(x - epsilon / 2.0))) / epsilon
}

fn newtons_method<F>(f: &F, guess: f64) -> f64 where F: Fn(&f64) -> f64 {
    let tolerance = f64::EPSILON.sqrt() * guess.abs().log2().round().exp2();
    let shift = derivative(f, guess) / f(&guess);
    if abs(shift) < tolerance {
        return guess;
    }
    return newtons_method(f, guess - shift);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn derivative_test() {
        let actual = |x| 2.0 * x;

        for i in 0..100 {
            let x = index_to_range(i as f64, 0.0, 100.0, -20.0, 20.0);
            assert_eq!()
        }
    }
}