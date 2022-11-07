use crate::*;
use rayon::prelude::*;

#[allow(non_camel_case_types)]
#[derive(Clone)]
pub struct Point<T_X, T_Y> {
    pub x: T_X,
    pub y: T_Y,
}

pub fn trapezoidal_approx<X, Y>(start: &Point<X, Y>, end: &Point<X, Y>) -> Y
where
    X: std::ops::Sub<Output = X> + Copy,
    Y: std::ops::Add<Output = Y>
        + std::ops::Mul<Output = Y>
        + std::ops::Div<f64, Output = Y>
        + Copy
        + From<X>,
{
    return Y::from(end.x - start.x) * (start.y + end.y) / 2.0_f64;
}

pub fn index_to_range<T>(x: T, in_min: T, in_max: T, out_min: T, out_max: T) -> T
where
    T: Copy
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>,
{
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

pub fn integrate<
    X: Sync + std::ops::Add<Output = X> + std::ops::Sub<Output = X> + Copy,
    Y: Default
        + Sync
        + std::ops::AddAssign
        + std::ops::Div<f64, Output = Y>
        + std::ops::Mul<Output = Y>
        + std::ops::Add<Output = Y>
        + Send
        + std::iter::Sum<Y>
        + Copy
        + From<X>,
>(
    points: Vec<Point<X, Y>>,
    batch_size: usize,
) -> Y {
    if points.len() < 2 {
        return Y::default();
    }

    let batches: Vec<&[Point<X, Y>]> = points.chunks(batch_size).collect();

    let parallel: Y = batches
        .par_iter()
        .map(|batch| {
            let mut sum = Y::default();
            for i in 0..(batch.len() - 1) {
                sum += trapezoidal_approx(&batch[i], &batch[i + 1]);
            }
            return sum;
        })
        .sum();

    let mut rest = Y::default();

    for i in 0..batches.len() - 1 {
        rest += trapezoidal_approx(&batches[i][batches[i].len() - 1], &batches[i + 1][0]);
    }

    return parallel + rest;
}

pub fn evaluate_function_between<X, Y>(f: &dyn Func<X, Y>, a: X, b: X, n: usize) -> Vec<Point<X, Y>>
where
    X: Copy
        + Send
        + Sync
        + std::cmp::PartialEq
        + From<f64>
        + std::ops::Add<Output = X>
        + std::ops::Sub<Output = X>
        + std::ops::Mul<Output = X>
        + std::ops::Div<Output = X>,
    Y: Send + Sync,
{
    if a == b {
        return vec![];
    }

    (0..n)
        .into_par_iter()
        .map(|i| {
            index_to_range(
                X::from(i as f64),
                X::from(0.0_f64),
                X::from((n - 1) as f64),
                a,
                b,
            )
        })
        .map(|x: X| Point { x, y: f.eval(x) })
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;

    fn square(x: f64) -> Complex64 {
        return complex(x * x, 0.0);
    }

    fn square_integral(a: f64, b: f64) -> Complex64 {
        return complex(b * b * b / 3.0 - a * a * a / 3.0, 0.0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn integral_of_square() {
        let square_func: Function<f64, Complex64> = Function::new(square);
        for i in 0..100 {
            for j in 0..10 {
                let a = f64::from(i - 50) / 12.3;
                let b = f64::from(j - 50) / 12.3;

                if i == j {
                    assert_eq!(
                        integrate(
                            evaluate_function_between(&square_func, a, b, INTEG_STEPS),
                            TRAPEZE_PER_THREAD,
                        ),
                        complex(0.0, 0.0)
                    );
                    continue;
                }

                let epsilon = 0.00001;
                assert!(complex_compare(
                    integrate(
                        evaluate_function_between(&square_func, a, b, INTEG_STEPS),
                        TRAPEZE_PER_THREAD,
                    ),
                    square_integral(a, b),
                    epsilon,
                ));
            }
        }
    }

    #[test]
    fn evaluate_square_func_between() {
        let square_func: Function<f64, Complex64> = Function::new(square);
        let actual = evaluate_function_between(&square_func, -2.0, 2.0, 5);
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
        let sinusoidal_exp_complex: Function<f64, Complex64> =
            Function::new(sinusoidal_exp_complex);
        for i in 0..10 {
            for j in 0..10 {
                let a = f64::from(i - 50) / 12.3;
                let b = f64::from(j - 50) / 12.3;

                if i == j {
                    assert_eq!(
                        integrate(
                            evaluate_function_between(&sinusoidal_exp_complex, a, b, INTEG_STEPS),
                            TRAPEZE_PER_THREAD,
                        ),
                        complex(0.0, 0.0)
                    );
                    continue;
                }
                let epsilon = 0.0001;
                assert!(complex_compare(
                    integrate(
                        evaluate_function_between(&sinusoidal_exp_complex, a, b, INTEG_STEPS),
                        TRAPEZE_PER_THREAD,
                    ),
                    sinusoidal_exp_complex_integral(a, b),
                    epsilon,
                ));
            }
        }
    }
}
