use std::cmp::Ordering;
use std::fmt::Debug;
use num::{Float, signum};
use std::ops::*;
use std::rc::Rc;
use std::sync::Arc;
use num::traits::FloatConst;
use rayon::prelude::*;
use crate::integrals::*;
use crate::utils::cmp_f64;

#[derive(Default, Debug)]
pub struct Vec2 {
    x: f64,
    y: f64,
}

impl Vec2 {
    pub fn dot(&self, other: &Vec2) -> f64 {
        return self.x * other.x + self.y + other.y;
    }

    pub fn mag(&self) -> f64 {
        return (self.x.powi(2) * self.y.powi(2)).sqrt();
    }

    pub fn pseudo_inverse(&self) -> CoVec2 {
        CoVec2(self.x, self.y) * (1.0 / (self.x.powi(2) + self.y.powi(2)))
    }
}

impl Add for Vec2 {
    type Output = Vec2;

    fn add(self, other: Self) -> Self::Output {
        Vec2 { x: self.x + other.x, y: self.y + other.y }
    }
}

impl Sub for Vec2 {
    type Output = Vec2;

    fn sub(self, other: Self) -> Self::Output {
        Vec2 { x: self.x - other.x, y: self.x - other.y }
    }
}

impl Mul<f64> for Vec2 {
    type Output = Vec2;

    fn mul(self, s: f64) -> Self::Output {
        Vec2 { x: self.x * s, y: self.y * s }
    }
}

#[derive(Debug)]
pub struct CoVec2(f64, f64);


impl Add for CoVec2 {
    type Output = CoVec2;

    fn add(self, other: Self) -> Self::Output {
        CoVec2(self.0 + other.0, self.1 + other.1)
    }
}

impl Sub for CoVec2 {
    type Output = CoVec2;

    fn sub(self, other: Self) -> Self::Output {
        CoVec2(self.0 - other.0, self.1 - other.1)
    }
}

impl Mul<Vec2> for CoVec2 {
    type Output = f64;

    fn mul(self, vec: Vec2) -> Self::Output {
        return self.0 * vec.x + self.1 * vec.y;
    }
}

impl Mul<f64> for CoVec2 {
    type Output = CoVec2;

    fn mul(self, s: f64) -> Self::Output {
        CoVec2(self.0 * s, self.1 * s)
    }
}

fn gradient<F>(f: F, x: f64) -> Vec2 where F: Fn(f64) -> Vec2 {
    let x_component = |x| f(x).x;
    let y_component = |x| f(x).y;
    return Vec2 {
        x: derivative(&x_component, x),
        y: derivative(&y_component, x),
    };
}

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

pub fn newtons_method_2d<F>(f: &F, mut guess: f64, precision: f64) -> f64
    where
        F: Fn(f64) -> Vec2, F::Output: Debug
{
    loop {
        let jacobian = gradient(f, guess);
        println!("guess: {:?}", guess);
        println!("f(guess): {:?}", f(guess));
        println!("inverse jacobian: {:?}", jacobian.pseudo_inverse());
        let step: f64 = jacobian.pseudo_inverse() * f(guess);
        if step.abs() < precision {
            return guess;
        } else {
            guess -= step;
            ;
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

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn smooth_sgn(x: f64) -> f64 {
    if x > 0.0 {
        (x + 3.0).exp() - 3.0.exp()
    } else {
        0.0
    }
}

fn check_sign(initial: f64, new: f64) -> bool {
    if initial == new {
        return false;
    }
    return (initial <= -0.0 && new >= 0.0) || (initial >= 0.0 && new <= 0.0);
}

pub fn bisection_search_sign_change<F>(f: &F, initial_guess: f64, step: f64) -> (f64, f64) where F: Fn(f64) -> f64 {
    let mut result = initial_guess;
    while !check_sign(f(initial_guess), f(result)) {
        result += step
    }
    return (result-step, result);
}


fn regula_falsi_c<F>(f: &F, a: f64, b: f64) -> f64 where F: Fn(f64) -> f64 {
    return (a * f(b) - b * f(a)) / (f(b) - f(a));
}

pub fn regula_falsi_method<F>(f: &F, mut a: f64, mut b: f64, precision: f64) -> f64 where F: Fn(f64) -> f64 {
    if a > b {
        let temp = a;
        a = b;
        b = temp;
    }

    let mut c = regula_falsi_c(f, a, b);
    while f64::abs(f(c)) > precision {
        b = regula_falsi_c(f, a, b);
        a = regula_falsi_c(f, a, b);
        c = regula_falsi_c(f, a, b);
    }
    return c;
}

pub fn regula_falsi_bisection<F>(f: &F, guess: f64, bisection_step: f64, precision: f64) -> f64 where F: Fn(f64) -> f64 {
    let (a, b) = bisection_search_sign_change(f, guess, bisection_step);
    return regula_falsi_method(f, a, b, precision);
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
        .map(move |x| {
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

pub fn inverse<F, A, R>(f: &F) -> Box<dyn Fn(R) -> Vec<A>> where F: Fn(A) -> R {
    todo!();  
}

#[cfg(test)]
mod test {
    use num::zero;
    use super::*;
    use crate::utils::cmp_f64;
    use crate::integrals::*;

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
    fn newtons_method_find_next_test() {
        use std::f64::consts;
        let interval = (-10.0, 10.0);

        let test_func = |x: f64| 5.0 * (3.0 * x + 1.0).abs() - (1.5 * x.powi(2) + x - 50.0).powi(2);

        let mut finder = NewtonsMethodFindNewZero::new(
            &test_func,
            1e-11,
            100000000,
        );

        for i in 0..4 {
            let guess = make_guess(&|x| finder.modified_func(x), interval, 1000);
            finder.next_zero(guess.unwrap());
        }

        let mut zeros = finder.get_previous_zeros().clone();
        zeros.sort_by(cmp_f64);
        let expected = [-6.65276132415, -5.58024707627, 4.91358040961, 5.98609465748];

        println!("zeros: {:#?}", zeros);

        assert_eq!(zeros.len(), expected.len());

        for (expected, actual) in expected.iter().zip(zeros.iter()) {
            assert!((*expected - *actual).abs() < 1e-10);
        }
    }

    #[test]
    fn regula_falsi_bisection_test() {
        let func = |x: f64| x * (x - 2.0) * (x + 2.0);

        let actual = regula_falsi_bisection(&func, -1e-3, -1e-3, 1e-5);
        let expected = -2.0;

        println!("expected: {}, actual {}", expected, actual);
        assert!(float_compare(expected, actual, 1e-3));
    }
}


