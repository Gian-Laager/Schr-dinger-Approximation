use crate::*;

pub struct SchroedingerError<'a> {
    pub wave_func: &'a WaveFunction,
}

impl Func<f64, Complex64> for SchroedingerError<'_> {
    fn eval(&self, x: f64) -> Complex64 {
        complex(-1.0 / (2.0 * self.wave_func.get_phase().mass), 0.0)
            * Derivative {
                f: &Derivative { f: self.wave_func },
            }
            .eval(x)
            + ((self.wave_func.get_phase().potential)(x) - self.wave_func.get_phase().energy)
                * self.wave_func.eval(x)
    }
}
