use crate::newtons_method::newtons_method;
use crate::newtons_method::newtons_method_max_iters;
use crate::newtons_method::{
    derivative, make_guess, newtons_method_find_new_zero, regula_falsi_bisection,
    regula_falsi_method, NewtonsMethodFindNewZero,
};
use crate::turning_points::*;
use crate::utils::cmp_f64;
use crate::*;
use num::integer::sqrt;
use num::{signum, zero};
use scilib::constant::H;
use std::cmp::{min, Ordering};
use std::sync::Arc;

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

#[derive(Clone)]
pub struct AiryWaveFunction {
    u_1: f64,
    pub turning_point: f64,
    phase: Arc<Phase>,
    pub ts: (f64, f64),
}

impl AiryWaveFunction {
    fn get_u_1_cube_root(u_1: f64) -> f64 {
        signum(u_1) * u_1.abs().pow(1.0 / 3.0)
    }

    pub fn new<'a>(phase: Arc<Phase>, view: (f64, f64)) -> (Vec<AiryWaveFunction>, TGroup) {
        let phase = phase;
        let turning_point_boundaries = turning_points::calc_ts(phase.as_ref(), view);

        let funcs: Vec<AiryWaveFunction> = turning_point_boundaries
            .ts
            .iter()
            .map(|((t1, t2), _)| {
                let x_1 = newtons_method(
                    &|x| (phase.potential)(x) - phase.energy,
                    (*t1 + *t2) / 2.0,
                    1e-7,
                );
                let u_1 = 2.0 * phase.mass * -derivative(phase.potential.as_ref(), x_1);
                // let u_1 = |x| -2.0 * phase.mass * ((phase.potential)(&x) - phase.energy) / (H_BAR * H_BAR * (x - x_1));

                AiryWaveFunction {
                    u_1,
                    turning_point: x_1,
                    phase: phase.clone(),
                    ts: (*t1, *t2),
                }
            })
            .collect::<Vec<AiryWaveFunction>>();
        return (funcs, turning_point_boundaries);
    }
}

impl Func<f64, Complex64> for AiryWaveFunction {
    fn eval(&self, x: f64) -> Complex64 {
        let u_1_cube_root = Self::get_u_1_cube_root(self.u_1);

        return ((std::f64::consts::PI.sqrt() / (self.u_1).abs().pow(1.0 / 6.0))
            * Ai(complex(u_1_cube_root * (self.turning_point - x), 0.0))) as Complex64
            * complex((self.u_1.signum() * (self.turning_point - x) + self.phase.phase_off).cos(), (self.u_1.signum() * (self.turning_point - x) + self.phase.phase_off).sin());
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
