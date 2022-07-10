use std::cmp::Ordering;
use num::Float;
use std::ops::*;
use std::rc::Rc;
use std::sync::Arc;
use rayon::prelude::*;
use crate::{index_to_range, utils};
use crate::utils::cmp_f64;

pub fn derivative<F, R>(f: &F, x: f64) -> R
    where
        F: Fn(f64) -> R,
        R: Sub<R, Output=R> + Div<f64, Output=R>,
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

#[derive(Clone)]
pub struct NewtonsMethodFindNewZero<F>
    where
        F: Fn(f64) -> f64,
{
    f: F,
    precision: f64,
    max_iters: usize,
    previous_zeros: Vec<(i32, f64)>,
}

impl<F: Fn(f64) -> f64> NewtonsMethodFindNewZero<F> {
    pub(crate) fn new(f: F, precision: f64, max_iters: usize) -> NewtonsMethodFindNewZero<F> {
        NewtonsMethodFindNewZero {
            f,
            precision,
            max_iters,
            previous_zeros: vec![],
        }
    }

    pub(crate) fn modified_func(&self, x: f64) -> f64 {
        let divisor = self.previous_zeros.iter().fold(1.0, |acc, (n, z)| acc * (x - z).powi(*n));
        let divisor = if divisor == 0.0 {
            divisor + self.precision
        } else {
            divisor
        };
        (self.f)(x) / divisor
    }

    pub(crate) fn next_zero(&mut self, guess: f64) -> Option<f64> {
        let zero = newtons_method_max_iters(&|x| self.modified_func(x), guess, self.precision, self.max_iters);

        if let Some(z) = zero {
            // to avoid hitting maxima and minima twice
            if derivative(&|x| self.modified_func(x), z).abs() < self.precision {
                self.previous_zeros.push((2, z));
            } else {
                self.previous_zeros.push((1, z));
            }
        }

        return zero;
    }

    pub(crate) fn get_previous_zeros(&self) -> Vec<f64> {
        self.previous_zeros.iter().map(|(_, z)| *z).collect::<Vec<f64>>()
    }
}

pub fn make_guess<F>(f: &F, (start, end): (f64, f64), n: usize) -> Option<f64>
    where
        F: Fn(f64) -> f64,
{
    let sort_func = |(_, y1): &(f64, f64), (_, y2): &(f64, f64)| -> Ordering { cmp_f64(&y1, &y2) };
    let mut points: Vec<(f64, f64)> = (0..n).into_iter()
        .map(|i| index_to_range(i as f64, 0.0, n as f64, start, end))
        .map(|x| {
            let der = derivative(f, x);
            (x, f(x) / (-(-der * der).exp() + 1.0))
        })
        .map(|(x, y)| (x, y.abs()))
        .collect();
    points.sort_by(sort_func);
    points.get(0).map(|point| point.0)
}

pub fn newtons_method_find_new_zero<F>(
    f: &F,
    mut guess: f64,
    precision: f64,
    max_iters: usize,
    known_zeros: &Vec<f64>,
) -> Option<f64> where
    F: Fn(f64) -> f64,
{
    let f_modified = |x| f(x) / known_zeros.iter().fold(0.0, |acc, &z| acc * (x - z));
    newtons_method_max_iters(&f_modified, guess, precision, max_iters)
}

#[cfg(test)]
mod test {
    use num::zero;
    use super::*;
    use crate::index_to_range;
    use crate::utils::cmp_f64;

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
                1e-4,
            ));
            assert!(float_compare(
                newtons_method(&func, -100.0, 1e-7),
                -zero,
                1e-4,
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
                1e-4,
            ));
            assert!(float_compare(
                newtons_method(&func, -100.0, 1e-7),
                -zero,
                1e-4,
            ));
            assert!(float_compare(
                newtons_method(&func, 0.0, 1e-7),
                zero / 2.0,
                1e-4,
            ));
        }
    }

    #[test]
    fn newtons_method_find_next_polynomial() {
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let a = index_to_range(i as f64, 0.0, 10.0, -10.0, 10.0);
                    let b = index_to_range(j as f64, 0.0, 10.0, -100.0, 0.0);
                    let c = index_to_range(k as f64, 0.0, 10.0, -1.0, 20.0);
                    let test_func = |x: f64| (x - a) * (x - b) * (x - c);

                    for guess in [a, b, c] {
                        let mut finder = NewtonsMethodFindNewZero::new(
                            &test_func,
                            1e-15,
                            10000000,
                        );

                        finder.next_zero(1.0);
                        finder.next_zero(1.0);
                        finder.next_zero(1.0);

                        let mut zeros_expected = [a, b, c];
                        let mut zeros_actual = finder.get_previous_zeros().clone();

                        zeros_expected.sort_by(cmp_f64);
                        zeros_actual.sort_by(cmp_f64);

                        assert_eq!(zeros_actual.len(), 3);

                        for (expected, actual) in zeros_expected.iter().zip(zeros_actual.iter()) {
                            assert!((*expected - *actual).abs() < 1e-10);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn newtons_method_find_next_sin() {
        use std::f64::consts;
        let interval = (-10.0, 10.0);

        let test_func = |x: f64| x.sin() * (x - 2.0 * consts::PI);

        let mut finder = NewtonsMethodFindNewZero::new(
            &test_func,
            1e-15,
            100000000,
        );

        for i in 0..7 {
            let guess = make_guess(&|x| finder.modified_func(x), interval, 1000);
            println!("guess: {}", guess.unwrap());
            finder.next_zero(guess.unwrap());
        }

        let zeros = finder.get_previous_zeros().clone();
        println!("zeros: {:#?}", zeros);

        for i in 0..zeros.len() {
            for j in 0..zeros.len() {
                if i != j {
                    assert!((zeros[i] - zeros[j]).abs() > 1.0);
                }
            }
        }

        zeros.iter().for_each(|z| {
            println!("{}, {}", z, (z.abs() / consts::PI) % 1.0);
            assert!((z.abs() / consts::PI) % 1.0 < 1e-3 || 1.0 - (z.abs() / consts::PI) % 1.0 < 1e-10);
        });
    }
}
