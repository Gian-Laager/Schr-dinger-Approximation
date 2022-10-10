use crate::*;
use std::sync::Arc;

#[derive(Clone)]
pub struct WkbWaveFunction {
    pub c: f64,
    pub turning_point: f64,
    pub phase: Arc<Phase>,
    integration_steps: usize,
    op: fn(Complex64) -> Complex64
}

impl WkbWaveFunction {
    pub fn new(
        phase: Arc<Phase>,
        c: f64,
        integration_steps: usize,
        turning_point: f64,
    ) -> WkbWaveFunction {
        return WkbWaveFunction {
            c,
            turning_point,
            phase: phase.clone(),
            integration_steps,
            op: identiy
        };
    }

    pub fn with_op(&self, op: fn(Complex64) -> Complex64) -> WkbWaveFunction{
        return WkbWaveFunction {
            c: self. c,
            turning_point: self.turning_point,
            phase: self.phase.clone(),
            integration_steps: self.integration_steps,
            op
        };
    }

    pub fn get_op(&self) -> Box<fn(Complex64) -> Complex64> {
        Box::new(self.op)
    }
}

impl Func<f64, Complex64> for WkbWaveFunction {
    fn eval(&self, x: f64) -> Complex64 {
        let integral = integrate(
            evaluate_function_between(
                self.phase.as_ref(),
                x,
                self.turning_point,
                self.integration_steps,
            ),
            TRAPEZE_PER_THREAD,
        );

        let val = if self.phase.energy < (self.phase.potential)(x) {
            (self.c * 0.5 * (-integral.abs()).exp())
                * complex(
                    (self.phase.phase_off).cos(),
                    (self.phase.phase_off).sin(),
                )
                / self.phase.sqrt_momentum(x)
        } else {
            complex(
                self.c * (-integral.abs() + self.phase.phase_off).cos(),
                self.c * (-integral.abs() + self.phase.phase_off).sin(),
            ) / self.phase.sqrt_momentum(x)
        };

        return (self.op)(val);
    }
}
