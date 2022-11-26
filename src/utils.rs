use crate::newtons_method::derivative;
use crate::Complex64;
use std::cmp::Ordering;

pub fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

pub fn complex(re: f64, im: f64) -> Complex64 {
    return Complex64 { re, im };
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn identity(c: Complex64) -> Complex64 {
    c
}

pub fn conjugate(c: Complex64) -> Complex64 {
    c.conj()
}

pub fn negative(c: Complex64) -> Complex64 {
    -c
}

pub fn negative_conj(c: Complex64) -> Complex64 {
    -c.conj()
}

pub fn complex_compare(expect: Complex64, actual: Complex64, epsilon: f64) -> bool {
    let average = (expect.norm() + actual.norm()) / 2.0;
    return (expect - actual).norm() / average < epsilon;
}

pub fn float_compare(expect: f64, actual: f64, epsilon: f64) -> bool {
    let average = (expect + actual) / 2.0;

    if average < epsilon {
        return expect == actual;
    }

    return (expect - actual) / average < epsilon;
}

pub trait Func<A, R>: Sync + Send {
    fn eval(&self, x: A) -> R;
}

pub trait ReToC: Sync + Func<f64, Complex64> {}

pub trait ReToRe: Sync + Func<f64, f64> {}

pub struct Function<A, R> {
    pub(crate) f: fn(A) -> R,
}

impl<A, R> Function<A, R> {
    pub const fn new(f: fn(A) -> R) -> Function<A, R> {
        return Function { f };
    }
}

impl<A, R> Func<A, R> for Function<A, R> {
    fn eval(&self, x: A) -> R {
        (self.f)(x)
    }
}

pub struct NormSquare<'a> {
    pub f: &'a dyn Func<f64, Complex64>,
}

impl Func<f64, f64> for NormSquare<'_> {
    fn eval(&self, x: f64) -> f64 {
        self.f.eval(x).norm_sqr()
    }
}

pub struct Derivative<'a> {
    pub f: &'a dyn Func<f64, Complex64>,
}

impl Func<f64, Complex64> for Derivative<'_> {
    fn eval(&self, x: f64) -> Complex64 {
        derivative(&|x| self.f.eval(x), x)
    }
}
