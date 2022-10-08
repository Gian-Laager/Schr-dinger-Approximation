use crate::*;
use std::mem::*;
use std::sync::*;

const AIRY_JOINT_WIDTH_PERCENT: f64 = 0.2;

trait WaveFunctionPart: Func<f64, Complex64> + Sync + Send {
    fn next(&self) -> Option<Arc<dyn WaveFunctionPart>>;
    fn previous(&self) -> Option<Arc<dyn WaveFunctionPart>>;
    fn range(&self) -> (f64, f64);
}

pub fn is_in_range(range: (f64, f64), x: f64) -> bool {
    return range.0 <= x && range.1 > x;
}

struct WaveAssemblyStart {
    next: Option<Arc<dyn WaveFunctionPart>>,
}

impl WaveFunctionPart for WaveAssemblyStart {
    fn next(&self) -> Option<Arc<dyn WaveFunctionPart>> {
        return self.next.clone();
    }
    fn range(&self) -> (f64, f64) {
        (
            APPROX_INF.0,
            self.next
                .as_ref()
                .map(|p| p.range().0)
                .unwrap_or(APPROX_INF.1),
        )
    }

    fn previous(&self) -> Option<Arc<dyn WaveFunctionPart>> {
        None
    }
}
impl Func<f64, Complex64> for WaveAssemblyStart {
    fn eval(&self, _x: f64) -> Complex64 {
        complex(0.0, 0.0)
    }
}

struct WaveAssemblyEnd {
    previous: Option<Arc<dyn WaveFunctionPart>>,
}

impl WaveFunctionPart for WaveAssemblyEnd {
    fn next(&self) -> Option<Arc<dyn WaveFunctionPart>> {
        None
    }
    fn range(&self) -> (f64, f64) {
        (
            self.previous
                .as_ref()
                .map(|p| p.range().1)
                .unwrap_or(APPROX_INF.0),
            APPROX_INF.1,
        )
    }

    fn previous(&self) -> Option<Arc<dyn WaveFunctionPart>> {
        self.previous.clone()
    }
}

impl Func<f64, Complex64> for WaveAssemblyEnd {
    fn eval(&self, _x: f64) -> Complex64 {
        complex(0.0, 0.0)
    }
}

pub struct Joint {
    pub left: Arc<dyn Func<f64, Complex64>>,
    pub right: Arc<dyn Func<f64, Complex64>>,
    pub cut: f64,
    pub delta: f64,
}

impl Joint {
    pub fn range(&self) -> (f64, f64) {
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
            + right.eval(x) * f64::cos((x - self.cut) / (-self.delta) * f64::consts::PI / 2.0).powi(2)
    }
}

struct JointPart {
    left: Arc<dyn WaveFunctionPart>,
    right: Arc<dyn WaveFunctionPart>,
    cut: f64,
    delta: f64,
}

impl Func<f64, Complex64> for JointPart {
    fn eval(&self, x: f64) -> Complex64 {
        self.left.eval(x)
            + (self.right.eval(x) - self.left.eval(x))
                * f64::tanh((x - self.cut) / self.delta * 2.0 * f64::consts::PI - f64::consts::PI)
    }
}

impl WaveFunctionPart for JointPart {
    fn next(&self) -> Option<Arc<dyn WaveFunctionPart>> {
        Some(self.right.clone())
    }
    fn range(&self) -> (f64, f64) {
        (self.cut, self.cut + self.delta)
    }
    fn previous(&self) -> Option<Arc<dyn WaveFunctionPart>> {
        Some(self.left.clone())
    }
}

struct ApproxPart {
    airy: Arc<AiryWaveFunction>,
    wkb: Arc<WkbWaveFunction>,
    airy_join_l: Joint,
    airy_join_r: Joint,
    range: (f64, f64),
    parity_re: bool,
    parity_im: bool,
    next: Option<Arc<dyn WaveFunctionPart>>,
    previous: Option<Arc<dyn WaveFunctionPart>>,
}

impl ApproxPart {
    fn new(
        airy: AiryWaveFunction,
        wkb: WkbWaveFunction,
        range: (f64, f64),
        parity_re: bool,
        parity_im: bool,
    ) -> ApproxPart {
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
            parity_re,
            parity_im,
            next: None,
            previous: None,
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
        } else if x < self.range.0 {
            return self.previous.as_ref().unwrap().eval(x);
        } else if x >= self.range.1 {
            return self.next.as_ref().unwrap().eval(x);
        }
        panic!("invalid value for x: {}", x);
    }
}

struct WaveFunction {
    phase: Arc<Phase>,
    view: Option<(f64, f64)>,
    start_part: WaveAssemblyStart,
    end_part: WaveAssemblyEnd,
    approx_first: Option<Arc<dyn WaveFunctionPart>>,
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
            approx_inf.clone()
        };

        let phase = Arc::new(Phase::new(energy, MASS, potential, f64::consts::PI / 4.0));

        let (airy_wave_funcs, boundaries) = AiryWaveFunction::new(phase.clone(), (view.0, view.1));
        if boundaries.ts.len() == 0 {
            panic!("No turning points found!");
        }

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

        let mut parity = (false, false);
        let approx_parts: Vec<ApproxPart> = wkb_airy_pair
            .iter()
            .map(|((wkb, range), airy)| -> ApproxPart {
                return ApproxPart::new(airy.clone(), wkb.clone(), *range, parity.0, parity.1);
            })
            .collect();

        todo!();
    }
}
