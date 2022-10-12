use crate::*;

#[allow(unused)]
pub fn smooth_step(x: f64) -> f64 {
    const TRANSITION: f64 = 0.5;
    let step = Arc::new(Function::new(|x: f64| -> Complex64 {
        if x.abs() < 2.0 {
            complex(10.0, 0.0)
        } else {
            complex(0.0, 0.0)
        }
    }));
    let zero = Arc::new(Function::new(|_: f64| -> Complex64 { complex(0.0, 0.0) }));
    let inf = Arc::new(Function::new(|x: f64| -> Complex64 {
        if x.abs() > 5.0 {
            complex(ENERGY_INF, 0.0)
        } else {
            complex(0.0, 0.0)
        }
    }));

    let joint_inf_zero_l = wave_function_builder::Joint {
        left: inf.clone(),
        right: zero.clone(),
        cut: -5.0 + TRANSITION / 2.0,
        delta: TRANSITION,
    };

    let joint_zero_step_l = wave_function_builder::Joint {
        left: zero.clone(),
        right: step.clone(),
        cut: -2.0 + TRANSITION / 2.0,
        delta: TRANSITION,
    };

    let joint_zero_inf_r = wave_function_builder::Joint {
        left: zero.clone(),
        right: inf.clone(),
        cut: 5.0 - TRANSITION / 2.0,
        delta: TRANSITION,
    };

    let joint_step_zero_r = wave_function_builder::Joint {
        left: step.clone(),
        right: zero.clone(),
        cut: 2.0 - TRANSITION / 2.0,
        delta: TRANSITION,
    };

    if wave_function_builder::is_in_range(joint_zero_inf_r.range(), x) {
        return joint_zero_inf_r.eval(x).re;
    }

    if wave_function_builder::is_in_range(joint_inf_zero_l.range(), x) {
        return joint_inf_zero_l.eval(x).re;
    }

    if wave_function_builder::is_in_range(joint_step_zero_r.range(), x) {
        return joint_step_zero_r.eval(x).re;
    }

    if wave_function_builder::is_in_range(joint_zero_step_l.range(), x) {
        return joint_zero_step_l.eval(x).re;
    }

    return zero.eval(x).re.max(inf.eval(x).re.max(step.eval(x).re));
}

#[allow(unused)]
pub fn mexican_hat(x: f64) -> f64 {
    (x - 4.0).powi(2) * (x + 4.0).powi(2)
}

#[allow(unused)]
pub fn double_mexican_hat(x: f64) -> f64 {
    (x - 4.0).powi(2) * x.powi(2) * (x + 4.0).powi(2)
}

#[allow(unused)]
pub fn triple_mexican_hat(x: f64) -> f64 {
    (x - 6.0).powi(2) * (x - 3.0).powi(2) * (x + 3.0).powi(2) * (x + 6.0).powi(2)
}

pub fn square(x: f64) -> f64 {
    x * x
}
