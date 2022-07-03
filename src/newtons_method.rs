use num::Float;
use std::ops::*;

pub fn derivative<F, R>(f: &F, x: f64) -> R
where
    F: Fn(f64) -> R,
    R: Sub<R, Output = R> + Div<f64, Output = R>,
{
    let epsilon = f64::epsilon().sqrt();
    (f(x + epsilon / 2.0) - f(x - epsilon / 2.0)) / epsilon
}

pub fn newtons_method<F>(f: &F, mut guess: f64, precision: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    loop {
        let step = f(guess) / derivative(f, guess);
        if step.abs() < precision {
            return guess;
        } else {
            guess -= step;
        }
    }
}

pub fn newtons_method_max_iters<F>(
    f: &F,
    mut guess: f64,
    precision: f64,
    max_iters: usize,
) -> Option<f64>
where
    F: Fn(f64) -> f64,
{
    for _ in 0..max_iters {
        let step = f(guess) / derivative(f, guess);
        if step.abs() < precision {
            return Some(guess);
        } else {
            guess -= step;
        }
    }
    None
}
#[cfg(test)]
mod test {
    use super::*;
    use crate::index_to_range;

    fn float_compare(expect: f64, actual: f64, epsilon: f64) -> bool {
        let average = (expect.abs() + actual.abs()) / 2.0;
        if average != 0.0 {
            (expect - actual).abs() / average < epsilon
        } else {
            (expect - actual).abs() < epsilon
        }
    }

    #[test]
    fn derivative_square_test() {
        let square = |x| x * x;
        let actual = |x| 2.0 * x;

        for i in 0..100 {
            let x = index_to_range(i as f64, 0.0, 100.0, -20.0, 20.0);
            assert!(float_compare(derivative(&square, x), actual(x), 1e-4));
        }
    }

    #[test]
    fn derivative_exp_test() {
        let exp = |x: f64| x.exp();

        for i in 0..100 {
            let x = index_to_range(i as f64, 0.0, 100.0, -20.0, 20.0);
            assert!(float_compare(derivative(&exp, x), exp(x), 1e-4));
        }
    }

    #[test]
    fn newtons_method_square() {
        for i in 0..100 {
            let zero = index_to_range(i as f64, 0.0, 100.0, 0.1, 10.0);
            let func = |x| x * x - zero * zero;
            assert!(float_compare(
                newtons_method(&func, 100.0, 1e-7),
                zero,
                1e-4
            ));
            assert!(float_compare(
                newtons_method(&func, -100.0, 1e-7),
                -zero,
                1e-4
            ));
        }
    }

    #[test]
    fn newtons_method_cube() {
        for i in 0..100 {
            let zero = index_to_range(i as f64, 0.0, 100.0, 0.1, 10.0);
            let func = |x| (x - zero) * (x + zero) * (x - zero / 2.0);
            assert!(float_compare(
                newtons_method(&func, 100.0, 1e-7),
                zero,
                1e-4
            ));
            assert!(float_compare(
                newtons_method(&func, -100.0, 1e-7),
                -zero,
                1e-4
            ));
            assert!(float_compare(
                newtons_method(&func, 0.0, 1e-7),
                zero / 2.0,
                1e-4
            ));
        }
    }
}
