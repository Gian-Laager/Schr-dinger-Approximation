use crate::*;

pub struct WkbWaveFunction<'a> {
    pub c_plus: f64,
    pub c_minus: f64,
    pub phase: &'a Phase,
    integration_steps: usize,
}

impl WkbWaveFunction<'_> {
    pub fn new(phase: &Phase, c0: f64, theta: f64, integration_steps: usize) -> WkbWaveFunction {
        let c_plus = 0.5 * c0 * f64::cos(theta - std::f64::consts::PI / 4.0);
        let c_minus = -0.5 * c0 * f64::sin(theta - std::f64::consts::PI / 4.0);
        return WkbWaveFunction {
            c_plus,
            c_minus,
            phase,
            integration_steps,
        };
    }
}

impl ReToC for WkbWaveFunction<'_> {
    fn eval(&self, x: &f64) -> Complex64 {
        let integral = integrate(
            evaluate_function_between(self.phase, self.phase.x_0, *x, self.integration_steps),
            TRAPEZE_PER_THREAD,
        );

        return (complex(self.c_plus, 0.0) * integral.exp()
            + complex(self.c_minus, 0.0) * (-integral).exp())
            / (self.phase.eval(&x)).sqrt();
    }
}
