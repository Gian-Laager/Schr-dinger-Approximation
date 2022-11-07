use crate::*;
use std::fmt::Display;
use std::sync::Arc;

#[derive(Clone)]
pub struct Phase {
    pub energy: f64,
    pub mass: f64,
    pub potential: Arc<dyn Fn(f64) -> f64 + Send + Sync>,
}

impl Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Phase {{energy: {}, mass: {}, potential: [func]}}",
            self.energy, self.mass
        )
    }
}

impl Phase {
    fn default() -> Phase {
        Phase {
            energy: 0.0,
            mass: 0.0,
            potential: Arc::new(|_x| 0.0),
        }
    }

    pub fn new<F: Fn(f64) -> f64 + Sync + Send>(
        energy: f64,
        mass: f64,
        potential: &'static F,
    ) -> Phase {
        return Phase {
            energy,
            mass,
            potential: Arc::new(potential),
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

#[derive(Clone)]
pub struct WkbWaveFunction {
    pub c: Complex64,
    pub turning_point_exp: f64,
    pub turning_point_osz: f64,
    pub phase: Arc<Phase>,
    integration_steps: usize,
    op: fn(Complex64) -> Complex64,
    pub phase_off: f64,
}

impl WkbWaveFunction {
    pub fn get_c(&self) -> Complex64 {
        self.c
    }

    pub fn with_c(&self, c: Complex64) -> WkbWaveFunction {
        WkbWaveFunction {
            c,
            turning_point_exp: self.turning_point_exp,
            turning_point_osz: self.turning_point_osz,
            phase: self.phase.clone(),
            integration_steps: self.integration_steps,
            op: self.op,
            phase_off: self.phase_off,
        }
    }

    pub fn new(
        phase: Arc<Phase>,
        c: Complex64,
        integration_steps: usize,
        turning_point_exp: f64,
        turning_point_osz: f64,
        phase_off: f64,
    ) -> WkbWaveFunction {
        println!("WKB phase_off: {}", phase_off);
        return WkbWaveFunction {
            c,
            turning_point_exp,
            turning_point_osz,
            phase: phase.clone(),
            integration_steps,
            op: identity,
            phase_off,
        };
    }

    pub fn with_op(&self, op: fn(Complex64) -> Complex64) -> WkbWaveFunction {
        return WkbWaveFunction {
            c: self.c,
            turning_point_exp: self.turning_point_exp,
            turning_point_osz: self.turning_point_osz,
            phase: self.phase.clone(),
            integration_steps: self.integration_steps,
            op,
            phase_off: self.phase_off,
        };
    }

    pub fn get_op(&self) -> Box<fn(Complex64) -> Complex64> {
        Box::new(self.op)
    }

    pub fn get_exp_sign(&self) -> f64 {
        let limit_sign = if self.turning_point_exp == self.turning_point_osz {
            1.0
        } else {
            -1.0
        };

        self.psi_osz(self.turning_point_exp + limit_sign * f64::EPSILON.sqrt())
            .re
            .signum()
    }

    fn psi_osz(&self, x: f64) -> Complex64 {
            let integral = integrate(
                evaluate_function_between(
                    self.phase.as_ref(),
                    x,
                    self.turning_point_osz,
                    self.integration_steps,
                ),
                TRAPEZE_PER_THREAD,
            );
            -self.c * complex((integral - self.phase_off).cos(), 0.0) / self.phase.sqrt_momentum(x)
    }

    fn psi_exp(&self, x: f64) -> Complex64 {
            let integral = integrate(
                evaluate_function_between(
                    self.phase.as_ref(),
                    x,
                    self.turning_point_exp,
                    self.integration_steps,
                ),
                TRAPEZE_PER_THREAD,
            );
            let exp_sign = self.get_exp_sign();

            if x < self.turning_point_exp {
                exp_sign
                    * (self.c * 0.5 * (-integral.abs()).exp())
                    * if COMPLEX_EXP_WKB {
                        complex((self.phase_off).cos(), -(self.phase_off).sin())
                            / self.phase.sqrt_momentum(x)
                    } else {
                        1.0.into()
                    }
            } else {
                exp_sign
                    * (self.c * 0.5 * (-integral.abs()).exp())
                    * if COMPLEX_EXP_WKB {
                        complex((self.phase_off).cos(), (self.phase_off).sin())
                            / self.phase.sqrt_momentum(x)
                    } else {
                        1.0.into()
                    }
            }
    }
}

impl Func<f64, Complex64> for WkbWaveFunction {
    fn eval(&self, x: f64) -> Complex64 {
        let val = if self.phase.energy < (self.phase.potential)(x) {
            self.psi_exp(x)
        } else {
            self.psi_osz(x)
        };

        return (self.op)(val);
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
}
