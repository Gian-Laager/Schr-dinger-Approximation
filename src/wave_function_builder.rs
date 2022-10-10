use crate::*;
use std::rc::*;
use std::sync::*;

const AIRY_JOINT_WIDTH_PERCENT: f64 = 0.2;

pub trait WaveFunctionPart: Func<f64, Complex64> + Sync + Send {
    fn range(&self) -> (f64, f64);
}

pub trait WaveFunctionPartWithOp: WaveFunctionPart {
    fn get_op(&self) -> Box<fn(Complex64) -> Complex64>;
    fn with_op(&self, op: fn(Complex64) -> Complex64) -> Box<dyn WaveFunctionPartWithOp>;
}

pub fn is_in_range(range: (f64, f64), x: f64) -> bool {
    return range.0 <= x && range.1 > x;
}

#[derive(Clone)]
pub struct Joint {
    pub left: Arc<dyn Func<f64, Complex64>>,
    pub right: Arc<dyn Func<f64, Complex64>>,
    pub cut: f64,
    pub delta: f64,
}

impl WaveFunctionPart for Joint {
    fn range(&self) -> (f64, f64) {
        if self.delta > 0.0 {
            (self.cut, self.cut + self.delta)
        } else {
            (self.cut + self.delta, self.cut)
        }
    }
}

impl Func<f64, Complex64> for Joint {
    fn eval(&self, x: f64) -> Complex64 {
        // self.left.eval(x)
        //     + (self.right.eval(x) - self.left.eval(x)) * sigmoid((x - self.cut) / self.delta)
        let (left, right) = if self.delta > 0.0 {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        };

        left.eval(x) * f64::sin((x - self.cut) / (-self.delta) * f64::consts::PI / 2.0).powi(2)
            + right.eval(x)
                * f64::cos((x - self.cut) / (-self.delta) * f64::consts::PI / 2.0).powi(2)
    }
}

struct PureWkb {
    wkb: Arc<WkbWaveFunction>,
    range: (f64, f64),
}

impl WaveFunctionPart for PureWkb {
    fn range(&self) -> (f64, f64) {
        self.range
    }
}

impl WaveFunctionPartWithOp for PureWkb {

    fn get_op(&self) -> Box<fn(Complex64) -> Complex64> {
        self.wkb.get_op()
    }

    fn with_op(&self, op: fn(Complex64) -> Complex64) -> Box<dyn WaveFunctionPartWithOp> {
        Box::new(PureWkb {
            wkb: Arc::new(self.wkb.with_op(op)),
            range: self.range,
        })
    }
}

impl Func<f64, Complex64> for PureWkb {
    fn eval(&self, x: f64) -> Complex64 {
        self.wkb.eval(x)
    }
}

struct ApproxPart {
    airy: Arc<AiryWaveFunction>,
    wkb: Arc<WkbWaveFunction>,
    airy_join_l: Joint,
    airy_join_r: Joint,
    range: (f64, f64),
}

impl WaveFunctionPart for ApproxPart {
    fn range(&self) -> (f64, f64) {
        self.range
    }
}

impl WaveFunctionPartWithOp for ApproxPart {
    fn get_op(&self) -> Box<fn(Complex64) -> Complex64> {
        self.wkb.get_op()
    }

    fn with_op(&self, op: fn(Complex64) -> Complex64) -> Box<dyn WaveFunctionPartWithOp> {
        Box::new(ApproxPart {
            airy: Arc::new(self.airy.with_op(op)),
            wkb: Arc::new(self.wkb.with_op(op)),
            airy_join_l: self.airy_join_l.clone(),
            airy_join_r: self.airy_join_r.clone(),
            range: self.range,
        })
    }
}

impl ApproxPart {
    fn new(airy: AiryWaveFunction, wkb: WkbWaveFunction, range: (f64, f64)) -> ApproxPart {
        let airy_rc = Arc::new(airy);
        let wkb_rc = Arc::new(wkb);
        let delta = (airy_rc.ts.1 - airy_rc.ts.0) * AIRY_JOINT_WIDTH_PERCENT;
        ApproxPart {
            airy: airy_rc.clone(),
            wkb: wkb_rc.clone(),
            airy_join_l: Joint {
                left: wkb_rc.clone(),
                right: airy_rc.clone(),
                cut: airy_rc.ts.0,
                delta,
            },
            airy_join_r: Joint {
                left: airy_rc.clone(),
                right: wkb_rc.clone(),
                cut: airy_rc.ts.1,
                delta,
            },
            range,
        }
    }
}

impl Func<f64, Complex64> for ApproxPart {
    fn eval(&self, x: f64) -> Complex64 {
        if is_in_range(self.airy_join_l.range(), x) {
            return self.airy_join_l.eval(x);
        } else if is_in_range(self.airy_join_r.range(), x) {
            return self.airy_join_r.eval(x);
        } else if is_in_range(self.airy.ts, x) {
            return self.airy.eval(x);
        } else if is_in_range(self.range, x) {
            return self.wkb.eval(x);
        }
        panic!("invalid value for x: {}", x);
    }
}
struct WaveFunction {
    phase: Arc<Phase>,
    view: (f64, f64),
    parts: Vec<Arc<dyn WaveFunctionPart>>,
}

pub fn find_best_op(
    phase: Arc<Phase>,
    previous: &dyn WaveFunctionPartWithOp,
    current: &dyn WaveFunctionPartWithOp,
) -> fn(Complex64) -> Complex64 {
    assert!(float_compare(current.range().0, previous.range().1, 1e-7));
    let boundary = current.range().0;

    let deriv_prev = derivative(&|x| previous.eval(x), current.range().0);
    let val_prev = previous.eval(current.range().0);
    let deriv = derivative(&|x| current.eval(x), current.range().0);
    let val = current.eval(boundary);

    println!(
        "deriv: {:.17}, deriv_prev: {:.17}, val: {:.17}, val_prev: {:.17}",
        deriv, deriv_prev, val, val_prev
    );
    if (phase.potential)(boundary) >= phase.energy {
        *previous.get_op()
    } else if sign_match_complex(conjugate(deriv), deriv_prev)
        && sign_match_complex(conjugate(val), val_prev)
    {
        println!("conjugate");
        conjugate
    } else if sign_match_complex(negative_conj(deriv), deriv_prev)
        && sign_match_complex(negative_conj(val), val_prev)
    {
        println!("negative_conj");
        negative_conj
    } else if sign_match_complex(negative(deriv), deriv_prev)
        && sign_match_complex(negative(val), val_prev)
    {
        println!("negative");
        negative
    } else {
        println!("identiy");
        identiy
    }
}

impl WaveFunction {
    fn new<F: Fn(f64) -> f64 + Sync + Send>(
        potential: &'static F,
        mass: f64,
        n_energy: usize,
        approx_inf: (f64, f64),
        view_factor: f64,
    ) -> WaveFunction {
        let energy = energy::nth_energy(n_energy, mass, &potential, approx_inf);

        let lower_bound = newtons_method::newtons_method_max_iters(
            &|x| potential(x) - energy,
            approx_inf.0,
            1e-7,
            100000,
        );
        let upper_bound = newtons_method::newtons_method_max_iters(
            &|x| potential(x) - energy,
            approx_inf.1,
            1e-7,
            100000,
        );

        let view = if lower_bound.is_some() && upper_bound.is_some() {
            (
                lower_bound.unwrap() * view_factor,
                upper_bound.unwrap() * view_factor,
            )
        } else {
            println!("Failed to determine view automaticaly, using APPROX_INF as view");
            approx_inf.clone()
        };

        let mut phase = Phase::new(energy, MASS, potential, f64::consts::PI / 4.0);
        let (_, t_boundaries) = AiryWaveFunction::new(Arc::new(phase.clone()), (view.0, view.1));
        if t_boundaries.ts.len() != 0 {
            // conjecture based on observations in all the plots
            phase.phase_off =
                f64::consts::PI / (t_boundaries.ts.len() as f64) - f64::consts::PI / 2.0;
        }
        let phase = Arc::new(phase);

        let (airy_wave_funcs, boundaries) = AiryWaveFunction::new(phase.clone(), (view.0, view.1));
        let parts: Vec<Arc<dyn WaveFunctionPartWithOp>> = if boundaries.ts.len() == 0 {
            println!("No turning points found in view! Results might be in accurate");
            let wkb1 = WkbWaveFunction::new(phase.clone(), C_0, INTEG_STEPS, APPROX_INF.0);
            let wkb2 = WkbWaveFunction::new(phase.clone(), C_0, INTEG_STEPS, APPROX_INF.1);

            let center = (view.0 + view.1) / 2.0;
            let wkb1 = Box::new(PureWkb {
                wkb: Arc::new(wkb1),
                range: (view.0, center),
            });

            let wkb2 = Box::new(PureWkb {
                wkb: Arc::new(wkb2),
                range: (center, view.1),
            });

            let op =
                wave_function_builder::find_best_op(phase.clone(), wkb1.as_ref(), wkb2.as_ref());

            vec![Arc::from(wkb1 as Box<dyn WaveFunctionPartWithOp>), Arc::from(wkb2.with_op(op))]
        } else {
            let turning_points: Vec<f64> = [
                vec![2.0 * view.0 - boundaries.ts.first().unwrap().1],
                boundaries.ts.iter().map(|p| p.1).collect(),
                vec![2.0 * view.1 - boundaries.ts.last().unwrap().1],
            ]
            .concat();

            let wave_funcs = turning_points
                .iter()
                .zip(turning_points.iter().skip(1))
                .zip(turning_points.iter().skip(2))
                .map(
                    |((previous, boundary), next)| -> (WkbWaveFunction, (f64, f64)) {
                        println!(
                    "WKB between: ({:.12}, {:.12}), prev: {:.12}, current: {:.12}, next: {:.12}",
                    (boundary + previous) / 2.0,
                    (next + boundary) / 2.0,
                    previous,
                    boundary,
                    next
                );
                        (
                            WkbWaveFunction::new(phase.clone(), C_0, INTEG_STEPS, *boundary),
                            ((boundary + previous) / 2.0, (next + boundary) / 2.0),
                        )
                    },
                )
                .collect::<Vec<(WkbWaveFunction, (f64, f64))>>();

            let wkb_airy_pair: Vec<(&(WkbWaveFunction, (f64, f64)), AiryWaveFunction)> = wave_funcs
                .iter()
                .zip(airy_wave_funcs.iter())
                .map(|(w, a)| (w, a.clone()))
                .collect();

            let approx_parts: Vec<Arc<dyn WaveFunctionPartWithOp>> = wkb_airy_pair
                .iter()
                .map(|((wkb, range), airy)| -> Arc<dyn WaveFunctionPartWithOp> {
                    Arc::new(ApproxPart::new(airy.clone(), wkb.clone(), *range))
                })
                .collect();

            let mut approx_parts_with_op = vec![Arc::from(approx_parts.first().unwrap().with_op(identiy))];
            approx_parts_with_op.reserve(approx_parts.len() - 1);

            for i in 0..(approx_parts.len() - 1) {
                let part1 = &approx_parts[i];
                let part2 = &approx_parts[i + 1];
                let p2_with_op =
                    part2.with_op(find_best_op(phase.clone(), part1.as_ref(), part2.as_ref()));
                approx_parts_with_op.push(Arc::from(p2_with_op));
            }

            approx_parts_with_op
        };

        let parts = parts.iter().map(|p| p.clone() as Arc<dyn WaveFunctionPart>).collect();

        return WaveFunction {
            phase,
            view,
            parts
        };
    }

    pub fn calc_psi(&self, x: f64) -> Complex64 {
        for part in self.parts.as_slice() {
            if is_in_range(part.range(), x) {
                return part.eval(x);
            }
        }
        panic!("[WkbWaveFunction::calc_psi] x out of range");
    }
}
