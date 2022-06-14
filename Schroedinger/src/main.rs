use std::cmp::min;
use std::f32::consts::PI;
use std::future::Future;
use std::ops::Deref;
use std::os::unix::process::parent_id;
use std::process::Output;
use num::CheckedMul;
use num::complex::{Complex, Complex64};
use tokio;
use tokio::runtime;
use tokio::runtime::Runtime;

const MAX_PARTIAL_SUM_PER_THREAD: usize = 32;
const INTEG_STEPS: usize = 1000;
const H_BAR: f64 = 1.0;
const X_0: f64 = 0.0;

trait ReToC: Sync {
    fn eval(&self, args: f64) -> Complex64;
}

struct Function {
    f: fn(f64) -> Complex64,
}

impl Function {
    const fn new(f: fn(f64) -> Complex64) -> Function {
        return Function { f };
    }
}

impl ReToC for Function {
    fn eval(&self, x: f64) -> Complex64 {
        (self.f)(x)
    }
}

struct Phase {
    energy: f64,
    mass: f64,
    potential: fn(f64) -> f64,
}

impl Phase {
    fn new(energy: f64, mass: f64, potential: fn(f64) -> f64) -> Phase {
        return Phase { energy, mass, potential };
    }
}

impl ReToC for Phase {
    fn eval(&self, x: f64) -> Complex64 {
        return (complex(2.0, 0.0) * complex(self.mass, 0.0) * complex((self.potential)(x) - self.energy, 0.0)).sqrt() / complex(H_BAR, 0.0);
    }
}


fn complex(re: f64, im: f64) -> Complex64 {
    return Complex64 { re, im };
}

fn trapezoidal_approx(f: &dyn ReToC, i: usize, a: f64, b: f64, n: usize) -> Complex64 {
    let a_ = a + i as f64 * (b - a) / n as f64;
    let b_ = a + (i as f64 + 1.0) * (b - a) / n as f64;

    return complex(b_ - a_, 0.0) * (f.eval(a_) + f.eval(b_)) / complex(2.0, 0.0);
}

async fn integrate(f: &'static dyn ReToC, a: f64, b: f64, n: usize) -> Complex64 {
    let mut sum = complex(0.0, 0.0);

    return if n < MAX_PARTIAL_SUM_PER_THREAD {
        for i in 0..n {
            sum += trapezoidal_approx(f, i, a, b, n);
        }
        sum
    } else {
        let mut futures = vec![];
        for i in (0..n).step_by(MAX_PARTIAL_SUM_PER_THREAD) {
            futures.push(tokio::spawn(async move {
                let mut partial_sum = complex(0.0, 0.0);

                let end = min(n, i + MAX_PARTIAL_SUM_PER_THREAD);

                for j in i..end {
                    partial_sum += trapezoidal_approx(f, j, a, b, n);
                }
                return partial_sum;
            }));
        }

        for f in futures {
            sum += f.await.unwrap();
        }

        // rest due to rounding errors
        sum
    };
}

struct WaveFunction {
    c_plus: Complex64,
    c_minus: Complex64,
    phase: &'static Phase,
    integration_steps: usize,
}

impl WaveFunction {
    fn new(phase: &'static Phase, c0: f64, theta: f64, integration_steps: usize) {
        let c_plus = complex(0.5 * c0 * f64::cos(theta - std::f64::consts::PI / 4.0), 0.0);
        let c_minus = complex(-0.5 * c0 * f64::cos(theta - std::f64::consts::PI / 4.0), 0.0);
        WaveFunction { c_plus, c_minus, phase, integration_steps };
    }

    fn eval(&self, x: f64) -> Box<dyn Future<Output=Complex64> + '_> {
        Box::new(async move {
            let integral = integrate(self.phase, X_0, x, self.integration_steps).await;

            return (self.c_plus * integral.exp() + self.c_minus * (-integral).exp()) / (self.phase.eval(x)).sqrt();
        })
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod test {
    use std::future::Future;
    use num::abs;
    use tokio::runtime;
    use tokio::task::JoinHandle;
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
        static square_func: Function = Function::new(square);
        for i in 0..1000 {
            for j in 0..1000 {
                let a = f64::from(i - 50) / 123.0;
                let b = f64::from(j - 50) / 123.0;

                if i == j {
                    assert_eq!(integrate(&square_func, a, b, INTEG_STEPS).await, complex(0.0, 0.0));
                    continue;
                }
                let epsilon = 0.00001;
                assert!(float_compare(integrate(&square_func, a, b, INTEG_STEPS).await, square_itegral(a, b), epsilon));
            }
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
        for i in 0..1000 {
            for j in 0..1000 {
                let a = f64::from(i - 50) / 123.0;
                let b = f64::from(j - 50) / 123.0;

                if i == j {
                    assert_eq!(integrate(&SINUSOIDAL_EXP_COMPLEX, a, b, INTEG_STEPS).await, complex(0.0, 0.0));
                    continue;
                }
                let epsilon = 0.0001;
                assert!(float_compare(integrate(&SINUSOIDAL_EXP_COMPLEX, a, b, INTEG_STEPS).await, sinusoidal_exp_complex_integral(a, b), epsilon));
            }
        }
    }
}
