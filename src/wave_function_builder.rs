use crate::wkb_wave_func::Phase;
use crate::*;
use std::sync::*;

pub enum ScalingType {
    Mul(Complex64),
    Renormalize(Complex64),
    None,
}

pub trait WaveFunctionPart: Func<f64, Complex64> + Sync + Send {
    fn range(&self) -> (f64, f64);
    fn as_func(&self) -> Box<dyn Func<f64, Complex64>>;
}

pub trait WaveFunctionPartWithOp: WaveFunctionPart {
    fn get_op(&self) -> Box<fn(Complex64) -> Complex64>;
    fn with_op(&self, op: fn(Complex64) -> Complex64) -> Box<dyn WaveFunctionPartWithOp>;
    fn as_wave_function_part(&self) -> Box<dyn WaveFunctionPart>;
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
    fn as_func(&self) -> Box<dyn Func<f64, Complex64>> {
        return Box::new(self.clone());
    }
}

impl Func<f64, Complex64> for Joint {
    fn eval(&self, x: f64) -> Complex64 {
        let (left, right) = if self.delta < 0.0 {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        };

        left.eval(x) * f64::sin((x - self.cut) / (self.delta) * f64::consts::PI / 2.0).powi(2)
            + right.eval(x)
                * f64::cos((x - self.cut) / (self.delta) * f64::consts::PI / 2.0).powi(2)
    }
}

#[derive(Clone)]
struct PureWkb {
    wkb: Arc<WkbWaveFunction>,
    range: (f64, f64),
}

impl WaveFunctionPart for PureWkb {
    fn range(&self) -> (f64, f64) {
        self.range
    }
    fn as_func(&self) -> Box<dyn Func<f64, Complex64>> {
        Box::new(self.clone())
    }
}

impl WaveFunctionPartWithOp for PureWkb {
    fn as_wave_function_part(&self) -> Box<dyn WaveFunctionPart> {
        Box::new(self.clone())
    }

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

#[derive(Clone)]
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
    fn as_func(&self) -> Box<dyn Func<f64, Complex64>> {
        Box::new(self.clone())
    }
}

impl WaveFunctionPartWithOp for ApproxPart {
    fn as_wave_function_part(&self) -> Box<dyn WaveFunctionPart> {
        Box::new(self.clone())
    }

    fn get_op(&self) -> Box<fn(Complex64) -> Complex64> {
        self.wkb.get_op()
    }

    fn with_op(&self, op: fn(Complex64) -> Complex64) -> Box<dyn WaveFunctionPartWithOp> {
        Box::new(ApproxPart::new(
            self.airy.with_op(op),
            self.wkb.with_op(op),
            self.range,
        ))
    }
}

impl ApproxPart {
    fn new(airy: AiryWaveFunction, wkb: WkbWaveFunction, range: (f64, f64)) -> ApproxPart {
        let airy_rc = Arc::new(airy);
        let wkb_rc = Arc::new(wkb);
        let delta = (airy_rc.ts.1 - airy_rc.ts.0) * AIRY_EXTRA;
        ApproxPart {
            airy: airy_rc.clone(),
            wkb: wkb_rc.clone(),
            airy_join_l: Joint {
                left: wkb_rc.clone(),
                right: airy_rc.clone(),
                cut: airy_rc.ts.0 + delta / 2.0,
                delta: -delta,
            },
            airy_join_r: Joint {
                left: airy_rc.clone(),
                right: wkb_rc.clone(),
                cut: airy_rc.ts.1 - delta / 2.0,
                delta,
            },
            range,
        }
    }

    pub fn is_airy(&self, x: f64) -> bool {
        if is_in_range(self.airy_join_l.range(), x) {
            return true;
        } else if is_in_range(self.airy_join_r.range(), x) {
            return true;
        } else if is_in_range(self.airy.ts, x) {
            return true;
        }
        return false;
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
        } else {
            return self.wkb.eval(x);
        }
    }
}

#[derive(Clone)]
pub struct WaveFunction {
    phase: Arc<Phase>,
    view: (f64, f64),
    parts: Vec<Arc<dyn WaveFunctionPart>>,
    airy_ranges: Vec<(f64, f64)>,
    wkb_ranges: Vec<(f64, f64)>,
    scaling: Complex64,
}

fn sign_match(f1: f64, f2: f64) -> bool {
    return f1.signum() == f2.signum();
}

fn sign_match_complex(mut c1: Complex64, mut c2: Complex64) -> bool {
    if c1.re.abs() < c1.im.abs() {
        c1.re = 0.0;
    }

    if c1.im.abs() < c1.re.abs() {
        c1.im = 0.0;
    }

    if c2.re.abs() < c2.im.abs() {
        c2.re = 0.0;
    }

    if c2.im.abs() < c2.re.abs() {
        c2.im = 0.0;
    }

    return sign_match(c1.re, c2.re) && sign_match(c1.im, c2.im);
}

fn calc_phase_offset(phase: Arc<Phase>, (turn_left, turn_right): (f64, f64)) -> Option<f64> {
    let critical_x = (turn_left + turn_right) / 2.0;
    if (phase.potential)(critical_x) > phase.energy {
        return None;
    }

    let int_left = integrate(
        evaluate_function_between(phase.as_ref(), turn_left, critical_x, INTEG_STEPS),
        TRAPEZE_PER_THREAD,
    );
    let int_right = -integrate(
        evaluate_function_between(phase.as_ref(), critical_x, turn_right, INTEG_STEPS),
        TRAPEZE_PER_THREAD,
    );

    let phase_off = (int_left - int_right) / 2.0;

    println!("phase_off: {:.12}, turn: ({:.6}, {:.6}), cut: {}", phase_off, turn_left, turn_right, (turn_right + turn_left) / 2.0);

    Some(phase_off)
}

pub fn find_best_op_wave_func_part(
    phase: Arc<Phase>,
    previous: &dyn WaveFunctionPartWithOp,
    current: &dyn WaveFunctionPartWithOp,
) -> fn(Complex64) -> Complex64 {
    if !float_compare(current.range().0, previous.range().1, 1e-3) {
        println!("current: ({}, {})", current.range().0, current.range().1);
        println!("previous: ({}, {})", previous.range().0, previous.range().1);
    }
    assert!(float_compare(current.range().0, previous.range().1, 1e-3));
    let boundary = current.range().0;

    let deriv_prev = derivative(&|x| previous.eval(x), current.range().0);
    let val_prev = previous.eval(current.range().0);
    let deriv = derivative(&|x| current.eval(x), current.range().0);
    let val = current.eval(boundary);

    return if (phase.potential)(boundary) >= phase.energy {
        *previous.get_op()
    } else {
        let conj_deriv = conjugate(deriv);
        let conj_val = conjugate(val);
        let neg_conj_deriv = negative_conj(deriv);
        let neg_conj_val = negative_conj(val);
        let neg_deriv = negative(deriv);
        let neg_val = negative(val);

        let conj_mse = (conj_deriv - deriv_prev).norm_sqr() + (conj_val - val_prev).norm_sqr();
        let neg_conj_mse =
            (neg_conj_deriv - deriv_prev).norm_sqr() + (neg_conj_val - val_prev).norm_sqr();
        let neg_mse = (neg_deriv - deriv_prev).norm_sqr() + (neg_val - val_prev).norm_sqr();
        let id_mse = (deriv - deriv_prev).norm_sqr() + (val - val_prev).norm_sqr();

        if conj_mse < neg_conj_mse && conj_mse < neg_mse && conj_mse < id_mse {
            println!(
                "conjugate mse, conj: {}, neg conj: {}, neg: {}, id: {}",
                conj_mse, neg_conj_mse, neg_mse, id_mse
            );
            conjugate
        } else if neg_conj_mse < conj_mse && neg_conj_mse < neg_mse && neg_conj_mse < id_mse {
            println!(
                "negative conj mse, conj: {}, neg conj: {}, neg: {}, id: {}",
                conj_mse, neg_conj_mse, neg_mse, id_mse
            );
            negative_conj
        } else if neg_mse < conj_mse && neg_mse < neg_conj_mse && neg_mse < id_mse {
            println!(
                "negative mse, conj: {}, neg conj: {}, neg: {}, id: {}",
                conj_mse, neg_conj_mse, neg_mse, id_mse
            );
            negative
        } else {
            println!(
                "identity mse, conj: {}, neg conj: {}, neg: {}, id: {}",
                conj_mse, neg_conj_mse, neg_mse, id_mse
            );
            identity
        }
    };
}

impl WaveFunction {
    pub fn get_energy(&self) -> f64 {
        self.phase.energy
    }

    pub fn new<F: Fn(f64) -> f64 + Sync + Send>(
        potential: &'static F,
        mass: f64,
        n_energy: usize,
        approx_inf: (f64, f64),
        view_factor: f64,
        scaling: ScalingType,
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
            println!("Failed to determine view automatically, using APPROX_INF as view");
            approx_inf.clone()
        };

        let phase = Arc::new(Phase::new(energy, mass, potential));

        let (airy_wave_funcs, boundaries) = AiryWaveFunction::new(phase.clone(), (view.0, view.1));
        let (parts, airy_ranges, wkb_ranges): (
            Vec<Arc<dyn WaveFunctionPart>>,
            Vec<(f64, f64)>,
            Vec<(f64, f64)>,
        ) = if boundaries.ts.len() == 0 {
            println!("No turning points found in view! Results might be in accurate");
            let wkb1 = WkbWaveFunction::new(
                phase.clone(),
                1.0.into(),
                INTEG_STEPS,
                approx_inf.0,
                calc_phase_offset(phase.clone(), approx_inf).unwrap_or(f64::consts::PI / 4.0),
            );
            let wkb2 = WkbWaveFunction::new(
                phase.clone(),
                1.0.into(),
                INTEG_STEPS,
                approx_inf.1,
                calc_phase_offset(phase.clone(), approx_inf).unwrap_or(f64::consts::PI / 4.0),
            );

            let center = (view.0 + view.1) / 2.0;
            let wkb1 = Box::new(PureWkb {
                wkb: Arc::new(wkb1),
                range: (approx_inf.0, center),
            });

            let wkb2 = Box::new(PureWkb {
                wkb: Arc::new(wkb2),
                range: (center, approx_inf.1),
            });

            let op = find_best_op_wave_func_part(phase.clone(), wkb1.as_ref(), wkb2.as_ref());

            let wkb1_range = wkb1.range();
            let wkb2 = wkb2.with_op(op);
            let delta = (view.1 - view.0) * WKB_TRANSITION_FRACTION;
            (
                vec![
                    Arc::new(Joint {
                        left: Arc::from(wkb1.as_func()),
                        right: Arc::from(wkb2.as_func()),
                        cut: (view.0 + view.1) / 2.0 - delta / 2.0,
                        delta: delta,
                    }),
                    Arc::from(wkb1.as_wave_function_part()),
                    Arc::from(wkb2.as_wave_function_part()),
                ],
                vec![],
                vec![wkb1_range, wkb2.range()],
            )
        } else {
            let turning_points: Vec<f64> = [
                vec![2.0 * approx_inf.0 - boundaries.ts.first().unwrap().1],
                boundaries.ts.iter().map(|p| p.1).collect(),
                vec![2.0 * approx_inf.1 - boundaries.ts.last().unwrap().1],
            ]
            .concat();

            let wave_funcs = turning_points
                .iter()
                .zip(turning_points.iter().skip(1))
                .zip(turning_points.iter().skip(2))
                .map(
                    |((previous, boundary), next)| -> (WkbWaveFunction, (f64, f64)) {
                        (
                            WkbWaveFunction::new(
                                phase.clone(),
                                1.0.into(),
                                INTEG_STEPS,
                                *boundary,
                                calc_phase_offset(phase.clone(), (*previous, *boundary))
                                    .unwrap_or(f64::consts::PI / 4.0),
                            ),
                            ((boundary + previous) / 2.0, (next + boundary) / 2.0),
                        )
                    },
                )
                .collect::<Vec<(WkbWaveFunction, (f64, f64))>>();

            let wkb_airy_pair: Vec<(&(WkbWaveFunction, (f64, f64)), AiryWaveFunction)> = wave_funcs
                .iter()
                .zip(airy_wave_funcs.iter())
                .map(|(w, a)| (w, a.with_phase_off(w.0.phase_off)))
                .collect();

            let wkb_ranges = wkb_airy_pair
                .iter()
                .map(|((_, wkb_range), _)| *wkb_range)
                .collect();
            let airy_ranges = wkb_airy_pair.iter().map(|(_, airy)| airy.ts).collect();

            let approx_parts: Vec<Arc<dyn WaveFunctionPartWithOp>> = wkb_airy_pair
                .iter()
                .map(|((wkb, range), airy)| -> Arc<dyn WaveFunctionPartWithOp> {
                    Arc::new(ApproxPart::new(airy.clone(), wkb.clone(), *range))
                })
                .collect();

            let mut approx_parts_with_op: Vec<Arc<dyn WaveFunctionPartWithOp>> =
                vec![Arc::from(approx_parts.first().unwrap().with_op(identity))];
            approx_parts_with_op.reserve(approx_parts.len() - 1);

            for i in 0..(approx_parts.len() - 1) {
                let part1 = &approx_parts[i];
                let part2 = &approx_parts[i + 1];
                let p2_with_op = part2.with_op(find_best_op_wave_func_part(
                    phase.clone(),
                    part1.as_ref(),
                    part2.as_ref(),
                ));
                approx_parts_with_op.push(Arc::from(p2_with_op));
            }

            let mut approx_parts_with_joints: Vec<Arc<dyn WaveFunctionPart>> = vec![];

            for (prev, curr) in approx_parts_with_op
                .iter()
                .zip(approx_parts_with_op.iter().skip(1))
            {
                assert!(float_compare(prev.range().1, curr.range().0, 1e-4));

                let distance = (f64::min(prev.range().1, view.1)
                    - f64::max(prev.range().0, view.0))
                    + (f64::min(curr.range().1, view.1) - f64::max(curr.range().0, view.0));
                let delta = distance * WKB_TRANSITION_FRACTION;
                let joint = Joint {
                    left: Arc::from(prev.as_func()),
                    right: Arc::from(curr.as_func()),
                    cut: f64::min(prev.range().1, view.1) - delta / 2.0,
                    delta,
                };

                println!("Joint in range: {:#?}, delta: {}", joint.range(), delta);

                approx_parts_with_joints.push(Arc::new(joint));
            }

            approx_parts_with_joints = vec![
                approx_parts_with_joints,
                approx_parts_with_op
                    .iter()
                    .map(|p| Arc::from(p.as_wave_function_part()))
                    .collect(),
            ]
            .concat();

            (approx_parts_with_joints, airy_ranges, wkb_ranges)
        };

        match scaling {
            ScalingType::Mul(s) => WaveFunction {
                phase,
                view,
                parts,
                airy_ranges,
                wkb_ranges,
                scaling: s,
            },
            ScalingType::None => WaveFunction {
                phase,
                view,
                parts,
                airy_ranges,
                wkb_ranges,
                scaling: complex(1.0, 0.0),
            },
            ScalingType::Renormalize(s) => {
                let unscaled = WaveFunction {
                    phase: phase.clone(),
                    view,
                    parts: parts.clone(),
                    airy_ranges: airy_ranges.clone(),
                    wkb_ranges: wkb_ranges.clone(),
                    scaling: s,
                };
                let factor = renormalize_factor(&unscaled, approx_inf);
                WaveFunction {
                    phase,
                    view,
                    parts,
                    airy_ranges,
                    wkb_ranges,
                    scaling: s * factor,
                }
            }
        }
    }

    pub fn calc_psi(&self, x: f64) -> Complex64 {
        for part in self.parts.as_slice() {
            if is_in_range(part.range(), x) {
                return part.eval(x);
            }
        }
        panic!(
            "[WkbWaveFunction::calc_psi] x out of range (x = {}, ranges: {:#?})",
            x,
            self.parts
                .iter()
                .map(|p| p.range())
                .collect::<Vec<(f64, f64)>>()
        );
    }

    pub fn get_airy_ranges(&self) -> &[(f64, f64)] {
        self.airy_ranges.as_slice()
    }

    pub fn get_wkb_ranges(&self) -> &[(f64, f64)] {
        self.wkb_ranges.as_slice()
    }

    pub fn get_wkb_ranges_in_view(&self) -> Vec<(f64, f64)> {
        self.wkb_ranges
            .iter()
            .map(|range| {
                (
                    f64::max(self.get_view().0, range.0),
                    f64::min(self.get_view().1, range.1),
                )
            })
            .collect::<Vec<(f64, f64)>>()
    }

    pub fn is_wkb(&self, x: f64) -> bool {
        self.wkb_ranges
            .iter()
            .map(|r| is_in_range(*r, x))
            .collect::<Vec<bool>>()
            .contains(&true)
    }

    pub fn is_airy(&self, x: f64) -> bool {
        self.airy_ranges
            .iter()
            .map(|r| is_in_range(*r, x))
            .collect::<Vec<bool>>()
            .contains(&true)
    }

    pub fn get_view(&self) -> (f64, f64) {
        self.view
    }

    pub fn set_view(&mut self, view: (f64, f64)) {
        self.view = view
    }
}

impl Func<f64, Complex64> for WaveFunction {
    fn eval(&self, x: f64) -> Complex64 {
        self.scaling * self.calc_psi(x)
    }
}

pub struct SuperPosition {
    wave_funcs: Vec<WaveFunction>,
    scaling: Complex64,
}

impl SuperPosition {
    pub fn new<F: Fn(f64) -> f64 + Send + Sync>(
        potential: &'static F,
        mass: f64,
        n_energies_scaling: &[(usize, Complex64)],
        approx_inf: (f64, f64),
        view_factor: f64,
        scaling: ScalingType,
    ) -> SuperPosition {
        let wave_funcs = n_energies_scaling
            .iter()
            .map(|(e, scale)| {
                let wave = WaveFunction::new(
                    potential,
                    mass,
                    *e,
                    approx_inf,
                    view_factor,
                    ScalingType::Mul(*scale),
                );
                println!("Calculated Energy {}\n", *e);
                return wave;
            })
            .collect();

        match scaling {
            ScalingType::Mul(s) => SuperPosition {
                wave_funcs,
                scaling: s,
            },
            ScalingType::None => SuperPosition {
                wave_funcs,
                scaling: 1.0.into(),
            },
            ScalingType::Renormalize(s) => {
                let unscaled = SuperPosition {
                    wave_funcs: wave_funcs.clone(),
                    scaling: s,
                };
                let factor = renormalize_factor(&unscaled, approx_inf);
                SuperPosition {
                    wave_funcs,
                    scaling: s * factor,
                }
            }
        }
    }

    pub fn get_view(&self) -> (f64, f64) {
        let view_a = self
            .wave_funcs
            .iter()
            .map(|w| w.get_view().0)
            .min_by(cmp_f64)
            .unwrap();
        let view_b = self
            .wave_funcs
            .iter()
            .map(|w| w.get_view().1)
            .max_by(cmp_f64)
            .unwrap();
        (view_a, view_b)
    }
}

impl Func<f64, Complex64> for SuperPosition {
    fn eval(&self, x: f64) -> Complex64 {
        self.scaling * self.wave_funcs.iter().map(|w| w.eval(x)).sum::<Complex64>()
    }
}

struct Scaled<A, R>
where
    R: std::ops::Mul<R, Output = R> + Sync + Send + Clone,
{
    scale: R,
    func: Box<dyn Func<A, R>>,
}

impl<A, R> Func<A, R> for Scaled<A, R>
where
    R: std::ops::Mul<R, Output = R> + Sync + Send + Clone,
{
    fn eval(&self, x: A) -> R {
        self.func.eval(x) * self.scale.clone()
    }
}

fn renormalize_factor(wave_func: &dyn Func<f64, Complex64>, approx_inf: (f64, f64)) -> f64 {
    integrate(
        evaluate_function_between(
            wave_func,
            approx_inf.0 * (1.0 - f64::EPSILON),
            approx_inf.1 * (1.0 - f64::EPSILON),
            INTEG_STEPS,
        )
        .par_iter()
        .map(|p| Point {
            x: p.x,
            y: p.y.norm_sqr(),
        })
        .collect(),
        TRAPEZE_PER_THREAD,
    )
}

pub fn renormalize(
    wave_func: Box<dyn Func<f64, Complex64>>,
    approx_inf: (f64, f64),
) -> Box<dyn Func<f64, Complex64>> {
    let area = renormalize_factor(wave_func.as_ref(), approx_inf);
    return Box::new(Scaled::<f64, Complex64> {
        scale: area.into(),
        func: wave_func,
    });
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sign_check_complex_test() {
        let range = (-50.0, 50.0);
        let n = 100000;
        for ri1 in 0..n {
            for ii1 in 0..n {
                for ri2 in 0..n {
                    for ii2 in 0..n {
                        let re1 = index_to_range(ri1 as f64, 0.0, n as f64, range.0, range.1);
                        let im1 = index_to_range(ii1 as f64, 0.0, n as f64, range.0, range.1);
                        let re2 = index_to_range(ri2 as f64, 0.0, n as f64, range.0, range.1);
                        let im2 = index_to_range(ii2 as f64, 0.0, n as f64, range.0, range.1);

                        assert_eq!(
                            sign_match_complex(complex(re1, im1), complex(re2, im2)),
                            sign_match_complex(complex(re2, im2), complex(re1, im1))
                        );
                    }
                }
            }
        }
    }
}
