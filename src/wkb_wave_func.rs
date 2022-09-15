use crate::*;

pub struct WkbWaveFunction<'a> {
    pub c: f64,
    pub turning_point: f64,
    pub phase: &'a Phase,
    integration_steps: usize,
}

impl WkbWaveFunction<'_> {
    pub fn new(
        phase: &Phase,
        c: f64,
        integration_steps: usize,
        turning_point: f64,
    ) -> WkbWaveFunction {
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
            evaluate_function_between(self.phase, x, self.turning_point, self.integration_steps),
            TRAPEZE_PER_THREAD,
        );

        if self.phase.energy < (self.phase.potential)(x) {
            return -f64::signum(x) * (self.c / 2.0 * (-integral.abs()).exp()) / self.phase.momentum(x);
        } else {
            return (self.c * (integral.abs() + f64::consts::PI / 4.0).cos()) / self.phase.momentum(x);
        }
    }
}
