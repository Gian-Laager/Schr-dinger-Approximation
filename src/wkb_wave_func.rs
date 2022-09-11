use crate::*;

pub struct WkbWaveFunction<'a> {
    pub c: f64,
    pub turning_point: f64,
    pub phase: &'a Phase,
    integration_steps: usize,
}

impl WkbWaveFunction<'_> {
    pub fn new(phase: &Phase, c: f64, theta: f64, integration_steps: usize, turning_point: f64) -> WkbWaveFunction {
        return WkbWaveFunction {
            c,
            turning_point,
            phase,
            integration_steps,
        };
    }
}

impl Func<f64, f64> for WkbWaveFunction<'_> {
    fn eval(&self, x: f64) -> f64 {
        let integral = integrate(
            evaluate_function_between(self.phase, self.turning_point, x, self.integration_steps),
            TRAPEZE_PER_THREAD,
        );



        return (self.c * integral.exp()) / self.phase.momentum(x);
    }
}
