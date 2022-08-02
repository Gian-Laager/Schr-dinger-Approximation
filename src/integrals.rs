use rayon::prelude::*;
use crate::Complex64;
use crate::utils::*;
use crate::*;

pub trait ReToC: Sync {
    fn eval(&self, x: &f64) -> Complex64;
}

pub struct Function {
    pub(crate) f: fn(f64) -> Complex64,
}

impl Function {
    pub const fn new(f: fn(f64) -> Complex64) -> Function {
        return Function { f };
    }
}

impl ReToC for Function {
    fn eval(&self, x: &f64) -> Complex64 {
        (self.f)(*x)
    }
}
pub struct Point {
    pub x: f64,
    pub y: Complex64,
}

pub fn trapezoidal_approx(start: &Point, end: &Point) -> Complex64 {
    return complex(end.x - start.x, 0.0) * (start.y + end.y) / complex(2.0, 0.0);
}

pub fn index_to_range(x: f64, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> f64 {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

pub fn integrate(points: Vec<Point>, batch_size: usize) -> Complex64 {
    if points.len() < 2 {
        return complex(0.0, 0.0);
    }

    let batches: Vec<&[Point]> = points.chunks(batch_size).collect();

    let parallel: Complex64 = batches
        .par_iter()
        .map(|batch| {
            let mut sum = complex(0.0, 0.0);
            for i in 0..(batch.len() - 1) {
                sum += trapezoidal_approx(&batch[i], &batch[i + 1]);
            }
            return sum;
        })
        .sum();

    let mut rest = complex(0.0, 0.0);

    for i in 0..batches.len() - 1 {
        rest += trapezoidal_approx(&batches[i][batches[i].len() - 1], &batches[i + 1][0]);
    }

    return parallel + rest;
}

pub fn evaluate_function_between(f: &dyn ReToC, a: f64, b: f64, n: usize) -> Vec<Point> {
    if a == b {
        return vec![];
    }

    (0..n)
        .into_par_iter()
        .map(|i| index_to_range(i as f64, 0.0, n as f64 - 1.0, a, b))
        .map(|x| Point { x, y: f.eval(&x) })
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;

    fn square(x: f64) -> Complex64 {
        return complex(x * x, 0.0);
    }

    fn square_itegral(a: f64, b: f64) -> Complex64 {
        return complex(b * b * b / 3.0 - a * a * a / 3.0, 0.0);
    }

    fn float_compare(expect: Complex64, actual: Complex64, epsilon: f64) -> bool {
        let average = (expect.norm() + actual.norm()) / 2.0;
        return (expect - actual).norm() / average < epsilon;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn integral_of_square() {
        static SQUARE_FUNC: Function = Function::new(square);
        for i in 0..100 {
            for j in 0..10 {
                let a = f64::from(i - 50) / 12.3;
                let b = f64::from(j - 50) / 12.3;

                if i == j {
                    assert_eq!(
                        integrate(
                            evaluate_function_between(&SQUARE_FUNC, a, b, INTEG_STEPS),
                            TRAPEZE_PER_THREAD,
                        ),
                        complex(0.0, 0.0)
                    );
                    continue;
                }

                let epsilon = 0.00001;
                assert!(float_compare(
                    integrate(
                        evaluate_function_between(&SQUARE_FUNC, a, b, INTEG_STEPS),
                        TRAPEZE_PER_THREAD,
                    ),
                    square_itegral(a, b),
                    epsilon,
                ));
            }
        }
    }

    #[test]
    fn evaluate_square_func_between() {
        static SQUARE_FUNC: Function = Function::new(square);
        let actual = evaluate_function_between(&SQUARE_FUNC, -2.0, 2.0, 5);
        let expected = vec![
            Point {
                x: -2.0,
                y: complex(4.0, 0.0),
            },
            Point {
                x: -1.0,
                y: complex(1.0, 0.0),
            },
            Point {
                x: 0.0,
                y: complex(0.0, 0.0),
            },
            Point {
                x: 1.0,
                y: complex(1.0, 0.0),
            },
            Point {
                x: 2.0,
                y: complex(4.0, 0.0),
            },
        ];

        for (a, e) in actual.iter().zip(expected) {
            assert_eq!(a.x, e.x);
            assert_eq!(a.y, e.y);
        }
    }

    fn sinusoidal_exp_complex(x: f64) -> Complex64 {
        return complex(x, x).exp();
    }

    fn sinusoidal_exp_complex_integral(a: f64, b: f64) -> Complex64 {
        // (-1/2 + i/2) (e^((1 + i) a) - e^((1 + i) b))
        return complex(-0.5, 0.5) * (complex(a, a).exp() - complex(b, b).exp());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn integral_of_sinusoidal_exp() {
        static SINUSOIDAL_EXP_COMPLEX: Function = Function::new(sinusoidal_exp_complex);
        for i in 0..10 {
            for j in 0..10 {
                let a = f64::from(i - 50) / 12.3;
                let b = f64::from(j - 50) / 12.3;

                if i == j {
                    assert_eq!(
                        integrate(
                            evaluate_function_between(&SINUSOIDAL_EXP_COMPLEX, a, b, INTEG_STEPS),
                            TRAPEZE_PER_THREAD,
                        ),
                        complex(0.0, 0.0)
                    );
                    continue;
                }
                let epsilon = 0.0001;
                assert!(float_compare(
                    integrate(
                        evaluate_function_between(&SINUSOIDAL_EXP_COMPLEX, a, b, INTEG_STEPS),
                        TRAPEZE_PER_THREAD,
                    ),
                    sinusoidal_exp_complex_integral(a, b),
                    epsilon,
                ));
            }
        }
    }
}