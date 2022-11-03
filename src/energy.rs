use crate::*;

struct Integrand<'a, F: Fn(f64) -> f64 + Sync> {
    mass: f64,
    pot: &'a F,
    view: (f64, f64),
    energy: f64,
}

impl<F: Fn(f64) -> f64 + Sync> Func<f64, f64> for Integrand<'_, F> {
    fn eval(&self, x: f64) -> f64 {
        let pot = (self.pot)(x);

        if !pot.is_finite() {
            return 0.0;
        }

        if pot < self.energy {
            return (2.0 * self.mass * (self.energy - pot)).sqrt();
        } else {
            return 0.0;
        }
    }
}

struct SommerfeldCond<'a, F: Fn(f64) -> f64 + Sync> {
    mass: f64,
    pot: &'a F,
    view: (f64, f64),
}

impl<F: Fn(f64) -> f64 + Sync> Func<f64, f64> for SommerfeldCond<'_, F> {
    fn eval(&self, energy: f64) -> f64 {
        let integrand = Integrand {
            mass: self.mass,
            pot: self.pot,
            view: self.view,
            energy,
        };
        let integral = integrate(
            evaluate_function_between(&integrand, self.view.0, self.view.1, INTEG_STEPS),
            TRAPEZE_PER_THREAD,
        );
        return ((integral - f64::consts::PI) / f64::consts::PI) % 1.0;
    }
}

pub fn nth_energy<F: Fn(f64) -> f64 + Sync>(n: usize, mass: f64, pot: &F, view: (f64, f64)) -> f64 {
    const ENERGY_STEP: f64 = 10.0;
    const CHECKS_PER_ENERGY_STEP: usize = INTEG_STEPS;
    let sommerfeld_cond = SommerfeldCond { mass, pot, view };

    let mut energy = 0.0; // newtons_method_non_smooth(&|e| sommerfeld_cond.eval(e), 1e-7, 1e-7);
    let mut i = 0;

    loop {
        let vals = evaluate_function_between(
            &sommerfeld_cond,
            energy,
            energy + ENERGY_STEP,
            CHECKS_PER_ENERGY_STEP,
        );
        let mut int_solutions = vals
            .iter()
            .zip(vals.iter().skip(1))
            .collect::<Vec<(&Point<f64, f64>, &Point<f64, f64>)>>()
            .par_iter()
            .filter(|(p1, p2)| (p1.y - p2.y).abs() > 0.5 || p1.y.signum() != p2.y.signum())
            .map(|ps| ps.1)
            .collect::<Vec<&Point<f64, f64>>>();
        int_solutions.sort_by(|p1, p2| cmp_f64(&p1.x, &p2.x));
        if i + int_solutions.len() > n {
            return int_solutions[n - i].x;
        }
        energy += ENERGY_STEP - (ENERGY_STEP / (CHECKS_PER_ENERGY_STEP as f64 + 1.0));
        i += int_solutions.len();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // #[test]
    // fn square() {
    //     let pot = |x| x * x;
    //     assert!((nth_energy(0, 1.0, &pot, (-100.0, 100.0)) - 0.707107).abs() < 1e-7);
    // }
}
