use crate::cmp_f64;
use crate::newtons_method::*;
use crate::wkb_wave_func::*;
use crate::*;
use num::signum;

const MAX_TURNING_POINTS: usize = 256;
const ACCURACY: f64 = 1e-9;

pub struct TGroup {
    pub ts: Vec<((f64, f64), f64)>,
    // pub tn: Option<f64>,
}

impl TGroup {
    pub fn new() -> TGroup {
        TGroup { ts: vec![] }
    }

    pub fn add_ts(&mut self, new_t: ((f64, f64), f64)) {
        self.ts.push(new_t);
    }
}

fn validity_func(phase: Phase) -> Arc<dyn Fn(f64) -> f64> {
    Arc::new(move |x: f64| {
        1.0 / (2.0 * phase.mass).sqrt()
            * derivative(&|t| (phase.potential)(t), x).abs()
            * VALIDITY_LL_FACTOR
            - ((phase.potential)(x) - phase.energy).pow(2)
    })
}

fn group_ts(zeros: &Vec<f64>, phase: &Phase) -> TGroup {
    let mut zeros = zeros.clone();
    let valid = validity_func(phase.clone());

    zeros.sort_by(cmp_f64);
    let mut derivatives = zeros
        .iter()
        .map(|x| derivative(valid.as_ref(), *x))
        .map(signum)
        .zip(zeros.clone())
        .collect::<Vec<(f64, f64)>>();

    let mut groups = TGroup { ts: vec![] };

    if let Some((deriv, z)) = derivatives.first() {
        if *deriv < 0.0 {
            let mut guess = z - ACCURACY.sqrt();
            let mut new_deriv = *deriv;
            let mut missing_t = *z;

            while new_deriv < 0.0 {
                missing_t =
                    regula_falsi_bisection(valid.as_ref(), guess, -ACCURACY.sqrt(), ACCURACY);
                new_deriv = signum(derivative(valid.as_ref(), missing_t));
                guess -= ACCURACY.sqrt();
            }

            derivatives.insert(
                0,
                (signum(derivative(valid.as_ref(), missing_t)), missing_t),
            );
        }
    }

    if let Some((deriv, z)) = derivatives.last() {
        if *deriv > 0.0 {
            let mut guess = z + ACCURACY.sqrt();
            let mut new_deriv = *deriv;
            let mut missing_t = *z;

            while new_deriv > 0.0 {
                missing_t =
                    regula_falsi_bisection(valid.as_ref(), guess, ACCURACY.sqrt(), ACCURACY);
                new_deriv = signum(derivative(valid.as_ref(), missing_t));
                guess += ACCURACY.sqrt();
            }

            derivatives.push((signum(derivative(valid.as_ref(), missing_t)), missing_t));
        }
    }

    assert_eq!(derivatives.len() % 2, 0);

    for i in (0..derivatives.len()).step_by(2) {
        let (t1_deriv, t1) = derivatives[i];
        let (t2_deriv, t2) = derivatives[i + 1];
        assert!(t1_deriv > 0.0);
        assert!(t2_deriv < 0.0);

        let turning_point = newtons_method(
            &|x| phase.energy - (phase.potential)(x),
            (t1 + t2) / 2.0,
            1e-7,
        );
        groups.add_ts(((t1, t2), turning_point));
    }

    return groups;
}

pub fn calc_ts(phase: &Phase, view: (f64, f64)) -> TGroup {
    let zeros = find_zeros(phase, view);
    let groups = group_ts(&zeros, phase);
    println!(
        "Turning Points: {:.7?}",
        groups.ts.iter().map(|(_, t)| *t).collect::<Vec<f64>>()
    );
    return groups;
}

fn find_zeros(phase: &Phase, view: (f64, f64)) -> Vec<f64> {
    let phase_clone = phase.clone();
    let validity_func = Arc::new(move |x: f64| {
        1.0 / (2.0 * phase_clone.mass).sqrt()
            * derivative(&|t| (phase_clone.potential)(t), x).abs()
            * VALIDITY_LL_FACTOR
            - ((phase_clone.potential)(x) - phase_clone.energy).pow(2)
    });
    let mut zeros = NewtonsMethodFindNewZero::new(validity_func, ACCURACY, 1e4 as usize);

    (0..MAX_TURNING_POINTS).into_iter().for_each(|_| {
        let modified_func = |x| zeros.modified_func(x);

        let guess = make_guess(&modified_func, view, 1000);
        guess.map(|g| zeros.next_zero(g));
    });

    let view = if view.0 < view.1 {
        view
    } else {
        (view.1, view.0)
    };
    let unique_zeros = zeros
        .get_previous_zeros()
        .iter()
        .filter(|x| **x > view.0 && **x < view.1)
        .map(|x| *x)
        .collect::<Vec<f64>>();
    return unique_zeros;
}
