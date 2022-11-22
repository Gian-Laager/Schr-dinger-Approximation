
#[cfg(test)]
mod test {
    extern crate test;
    use test::Bencher;

    use crate::*;
    use std::f64;

    use duplicate::duplicate_item;
    use paste::paste;

    const INTEG_STEPS: usize = 64000;
    const TRAPEZE_PER_THREAD: usize = 1000;
    const NUMBER_OF_POINTS: usize = 100000;

    const AIRY_TRANSITION_FRACTION: f64 = 0.5;
    const ENABLE_AIRY_JOINTS: bool = true;

    const VALIDITY_LL_FACTOR: f64 = 3.5;

    const APPROX_INF: (f64, f64) = (-200.0, 200.0);
    const VIEW_FACTOR: f64 = 0.5;

    #[duplicate_item(
              num; 
              [1];
              [2];
              [3];
              [4];
              [5];
              [6];
              [7];
              [8];
              [9];
            )]
    paste! {
        #[bench]
        fn [< evaluate_bench_nenergy_ num >](b: &mut Bencher){
            let wave_function = wave_function_builder::WaveFunction::new(
                &potentials::square,
                1.0, // mass
                num,  // nth energy
                APPROX_INF,
                VIEW_FACTOR,
                ScalingType::Renormalize(complex(0.0, f64::consts::PI / 4.0).exp()),
            );

            b.iter(|| {
                let end = test::black_box(10.0);
                evaluate_function_between(&wave_function, -10.0, end, 100);
            })
        }
    }

    #[duplicate_item(
              num; 
              [1];
              [2];
              [3];
              [4];
              [5];
              [6];
              [7];
              [8];
              [9];
            )]
    paste! {
    #[bench]
    #[ignore]
    fn [< energy_bench_nenergy_ num >](b: &mut Bencher) {
              b.iter(|| {
                  let n = test::black_box(num);
                  let wave_function = wave_function_builder::WaveFunction::new(
                      &potentials::square,
                      1.0, // mass
                      n,   // nth energy
                      APPROX_INF,
                      VIEW_FACTOR,
                      ScalingType::None,
                  );
                  let _ = test::black_box(&wave_function);
              })
          }
    }
}

