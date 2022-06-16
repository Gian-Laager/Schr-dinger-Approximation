use std::f64;
use std::fs::File;
use std::io::Write;
use num::complex::{Complex64};
use num::pow::Pow;
use num::{Complex, signum};
use tokio;
use rayon::iter::*;
use rgsl::{complex, Mode};
use rgsl::gamma_beta::gamma;

const TRAPEZE_PER_THREAD: usize = 1000;
const INTEG_STEPS: usize = 10000;
const NUMBER_OF_POINTS: usize = 10000;
const H_BAR: f64 = 1.0;
const X_0: f64 = -10.0;
const ENERGY: f64 = 20.0;
const MASS: f64 = 3.0;
const C_0: f64 = 1.0;
const THETA: f64 = 1.0;

const VIEW: (f64, f64) = (-10.0, 0.0);


trait ReToC: Sync {
    fn eval(&self, x: &f64) -> Complex64;
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
    fn eval(&self, x: &f64) -> Complex64 {
        (self.f)(*x)
    }
}

struct Phase {
    energy: f64,
    mass: f64,
    potential: fn(&f64) -> f64,
}

impl Phase {
    const fn new(energy: f64, mass: f64, potential: fn(&f64) -> f64) -> Phase {
        return Phase { energy, mass, potential };
    }
}

impl ReToC for Phase {
    fn eval(&self, x: &f64) -> Complex64 {
        return (complex(2.0, 0.0) * complex(self.mass, 0.0) * complex((self.potential)(x) - self.energy, 0.0)).sqrt() / complex(H_BAR, 0.0);
    }
}


fn complex(re: f64, im: f64) -> Complex64 {
    return Complex64 { re, im };
}

fn trapezoidal_approx(start: &Point, end: &Point) -> Complex64 {
    return complex(end.x - start.x, 0.0) * (start.y + end.y) / complex(2.0, 0.0);
}


fn index_to_range(x: f64, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> f64 {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

fn Ai(x: Complex64) -> Complex64 {
    const COEFFICIENTS: [f64; 8] = [
        0.355028053887817239260,
        -0.2588194037928067984051,
        0.0591713423146362065433,
        -0.0215682836494005665338,
        0.00197237807715454021811,
        -0.000513530563080965869852,
        0.000281768296736362888302,
        -5.70589514534406522057e-6];
    let mut result = complex(0.0, 0.0);
    for i in (0..COEFFICIENTS.len()).step_by(2) {
        result += x.pow(3.0 * i as f64) * COEFFICIENTS[i] + x.pow(3.0 * i as f64 + 1.0) * COEFFICIENTS[i + 1];
    }
    return result;
}

fn Bi(x: Complex64) -> Complex64 {
    return -complex(0.0, 1.0) * Ai(x) + 2.0 * Ai(x * complex(-0.5, 3.0_f64.sqrt() / 2.0)) * complex(3_f64.sqrt() / 2.0, 0.5);
}

struct Point {
    x: f64,
    y: Complex64,
}

fn integrate(points: Vec<Point>, batch_size: usize) -> Complex64 {
    if points.len() < 2 {
        return complex(0.0, 0.0);
    }

    let batches: Vec<&[Point]> = points.chunks(batch_size).collect();

    let parallel: Complex64 = batches.par_iter()
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

fn evaluate_function_between(f: &dyn ReToC, a: f64, b: f64, n: usize) -> Vec<Point> {
    if a == b {
        return vec![];
    }

    (0..n).into_par_iter()
        .map(|i| index_to_range(i as f64, 0.0, n as f64 - 1.0, a, b))
        .map(|x| Point { x, y: f.eval(&x) })
        .collect()
}

struct WaveFunction<'a> {
    c_plus: f64,
    c_minus: f64,
    phase: &'a Phase,
    integration_steps: usize,
}

impl WaveFunction<'_> {
    fn new(phase: &Phase, c0: f64, theta: f64, integration_steps: usize) -> WaveFunction {
        let c_plus = 0.5 * c0 * f64::cos(theta - std::f64::consts::PI / 4.0);
        let c_minus = -0.5 * c0 * f64::cos(theta - std::f64::consts::PI / 4.0);
        return WaveFunction { c_plus, c_minus, phase, integration_steps };
    }
}

impl ReToC for WaveFunction<'_> {
    fn eval(&self, x: &f64) -> Complex64 {
        let integral = integrate(evaluate_function_between(self.phase, X_0, *x, self.integration_steps), TRAPEZE_PER_THREAD);

        return (complex(self.c_plus, 0.0) * integral.exp() + complex(self.c_minus, 0.0) * (-integral).exp()) / (self.phase.eval(&x)).sqrt();
    }
}

struct AiryWaveFunction {
    c_a: Complex64,
    c_b: Complex64,
    u_1: f64,
    x_1: f64,
}

impl AiryWaveFunction {
    fn calc_c_a_and_c_b(phase: &Phase, t: (f64, f64), c_wkb: (f64, f64), u_1: f64, x_1: f64) -> (Complex64, Complex64) {
        let wkb_plus_1 = integrate(evaluate_function_between(phase, X_0, t.0, INTEG_STEPS), TRAPEZE_PER_THREAD).exp() / phase.eval(&t.0).sqrt();
        let wkb_minus_1 = -integrate(evaluate_function_between(phase, X_0, t.0, INTEG_STEPS), TRAPEZE_PER_THREAD).exp() / phase.eval(&t.0).sqrt();

        println!("wkb_plus_1 = {}", wkb_plus_1);
        println!("wkb_minus_1 = {}", wkb_minus_1);

        let wkb_plus_2 = integrate(evaluate_function_between(phase, -X_0, t.1, INTEG_STEPS), TRAPEZE_PER_THREAD).exp() / phase.eval(&t.1).sqrt();
        let wkb_minus_2 = -integrate(evaluate_function_between(phase, -X_0, t.1, INTEG_STEPS), TRAPEZE_PER_THREAD).exp() / phase.eval(&t.1).sqrt();

        println!("wkb_plus_2 = {}", wkb_plus_2);
        println!("wkb_minus_2 = {}", wkb_minus_2);

        let airy_ai_1 = Ai(complex(u_1, 0.0).pow(1.0 / 3.0) * (t.0 - x_1));
        let airy_bi_1 = Bi(complex(u_1, 0.0).pow(1.0 / 3.0) * (t.0 - x_1));
        let airy_ai_2 = Ai(complex(u_1, 0.0).pow(1.0 / 3.0) * (t.1 - x_1));
        let airy_bi_2 = Bi(complex(u_1, 0.0).pow(1.0 / 3.0) * (t.1 - x_1));
        println!("u_1.pow(1.0 / 3.0) * (t.0 - x_1) = {}", u_1.pow(1.0 / 3.0) * (t.0 - x_1));
        println!("u_1.pow(1.0 / 3.0) * (t.1 - x_1) = {}", u_1.pow(1.0 / 3.0) * (t.1 - x_1));
        println!("airy_ai_1 = {}", airy_ai_1);
        println!("airy_bi_1 = {}", airy_bi_1);
        println!("airy_ai_2 = {}", airy_ai_2);
        println!("airy_bi_2 = {}", airy_bi_2);
        let f1 = airy_ai_1;
        let g1 = airy_bi_1;
        let h1 = wkb_plus_1;
        let k1 = wkb_minus_1;

        let f2 = airy_ai_2;
        let g2 = airy_bi_2;
        let h2 = wkb_plus_2;
        let k2 = wkb_minus_2;


        let c_a = ((-c_wkb.1 * (g1 * k2 - g2 * k1)) / (f1 * g2 - f2 * g1)) - ((c_wkb.0 * (g1 * h2 - g2 * h1)) / (f1 * g2 - f2 * g1));
        let c_b = ((c_wkb.1 * (f1 * k2 - f2 * k1)) / (f1 * g2 - f2 * g1)) + ((c_wkb.0 * (f1 * h2 - f2 * h1)) / (f1 * g2 - f2 * g1));

        return (c_a, c_b);
    }

    fn calc_ts(phase: &Phase) -> (f64, f64) {
        let t1 = signum(X_0) * f64::sqrt(phase.energy + H_BAR * H_BAR / phase.mass - f64::sqrt(H_BAR * H_BAR * (H_BAR * H_BAR + 2.0 * phase.mass * phase.energy)) / phase.mass) / 1.1;
        let t2 = signum(X_0) * f64::sqrt(phase.energy + H_BAR * H_BAR / phase.mass + f64::sqrt(H_BAR * H_BAR * (H_BAR * H_BAR + 2.0 * phase.mass * phase.energy)) / phase.mass) * 1.1;
        return (t1, t2);
    }

    fn new(wave_func: &WaveFunction) -> AiryWaveFunction {
        let phase = wave_func.phase;
        let (t1, t2) = AiryWaveFunction::calc_ts(phase);
        let x_1 = (t1 + &t2) / 2.0;
        let u_1 = 2.0 * phase.mass / (H_BAR * H_BAR) * ((phase.potential)(&t2) - (phase.potential)(&t1)) / (t2 - t1);

        println!("x_1 = {}, u_1 = {}", x_1, u_1);
        println!("t_1 = {}, t_2 = {}", t1, t2);

        let (c_a, c_b) = AiryWaveFunction::calc_c_a_and_c_b(phase, (t1, t2), (wave_func.c_plus, wave_func.c_minus), u_1, x_1);
        println!("c_a = {}, c_b = {}", c_a, c_b);

        AiryWaveFunction { c_a, c_b, u_1, x_1 }
    }
}

impl ReToC for AiryWaveFunction {
    fn eval(&self, x: &f64) -> Complex64 {
        let u_1_cube_root = complex(self.u_1, 0.0).pow(1.0 / 3.0);
        let ai = self.c_a * Ai(u_1_cube_root);
        let bi = self.c_b * Bi(u_1_cube_root);
        (x - self.x_1) * (ai + bi)
    }
}

fn square(x: &f64) -> f64 {
    return x * x;
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let phase: Phase = Phase::new(ENERGY, MASS, square);
    let wave_func = WaveFunction::new(&phase, C_0, THETA, INTEG_STEPS);
    let values = evaluate_function_between(&wave_func, VIEW.0, VIEW.1, NUMBER_OF_POINTS);

    let mut data_file = File::create("data.txt").unwrap();

    let data_str: String = values.par_iter().map(|p| -> String {
        format!("{} {} {}\n", p.x, p.y.re, p.y.im)
    }).reduce(|| String::new(), |s: String, current: String| {
        s + &*current
    });

    let (t1, t2) = AiryWaveFunction::calc_ts(&phase);
    let airy_wave_func = AiryWaveFunction::new(&wave_func);

    let airy_values = evaluate_function_between(&airy_wave_func, t1, t2, NUMBER_OF_POINTS);

    let airy_data_str: String = airy_values.par_iter().map(|p| -> String {
        format!("{} {} {}\n", p.x, p.y.re, p.y.im)
    }).reduce(|| String::new(), |s: String, current: String| {
        s + &*current
    });
    data_file.write_all((data_str + "\n\n" + &*airy_data_str).as_ref()).unwrap()
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
            for j in 0..100 {
                let a = f64::from(i - 50) / 12.3;
                let b = f64::from(j - 50) / 12.3;

                if i == j {
                    assert_eq!(integrate(evaluate_function_between(&SQUARE_FUNC, a, b, INTEG_STEPS), TRAPEZE_PER_THREAD), complex(0.0, 0.0));
                    continue;
                }

                let epsilon = 0.00001;
                assert!(float_compare(integrate(evaluate_function_between(&SQUARE_FUNC, a, b, INTEG_STEPS), TRAPEZE_PER_THREAD), square_itegral(a, b), epsilon));
            }
        }
    }


    #[test]
    fn evaluate_square_func_between() {
        static SQUARE_FUNC: Function = Function::new(square);
        let actual = evaluate_function_between(&SQUARE_FUNC, -2.0, 2.0, 5);
        let expected = vec![Point { x: -2.0, y: complex(4.0, 0.0) },
                            Point { x: -1.0, y: complex(1.0, 0.0) },
                            Point { x: 0.0, y: complex(0.0, 0.0) },
                            Point { x: 1.0, y: complex(1.0, 0.0) },
                            Point { x: 2.0, y: complex(4.0, 0.0) }];

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
        for i in 0..100 {
            for j in 0..100 {
                let a = f64::from(i - 50) / 12.3;
                let b = f64::from(j - 50) / 12.3;

                if i == j {
                    assert_eq!(integrate(evaluate_function_between(&SINUSOIDAL_EXP_COMPLEX, a, b, INTEG_STEPS), TRAPEZE_PER_THREAD), complex(0.0, 0.0));
                    continue;
                }
                let epsilon = 0.0001;
                assert!(float_compare(integrate(evaluate_function_between(&SINUSOIDAL_EXP_COMPLEX, a, b, INTEG_STEPS), TRAPEZE_PER_THREAD), sinusoidal_exp_complex_integral(a, b), epsilon));
            }
        }
    }
}
