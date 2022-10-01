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

impl Func<f64, Complex64> for WkbWaveFunction<'_> {
    fn eval(&self, x: f64) -> Complex64 {
        let integral = integrate(
            evaluate_function_between(self.phase, x, self.turning_point, self.integration_steps),
            TRAPEZE_PER_THREAD,
        );

        if self.phase.energy < (self.phase.potential)(x) {
            return complex(
                (self.c * 0.5 * (-integral.abs()).exp()) / self.phase.momentum(x),
                0.0,
            );
        } else {
            return complex(
                self.c * (integral.abs() - self.phase.phase_off).cos(),
                0.0
                // self.c * (-integral.abs() - self.phase.phase_off).sin(),
            ) / self.phase.momentum(x);
        }
    }
}
