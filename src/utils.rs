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

pub fn identiy(c: Complex64) -> Complex64 {
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
    return (expect - actual) / average < epsilon;
}
