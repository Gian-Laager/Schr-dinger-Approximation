use crate::{AiryWaveFunction, Phase, ReToC};

type WaveFunction = Vec<Box<dyn ReToC>>;

struct WaveFunctionBuilder {
    phase: Phase,
    view: (f64, f64)
}

impl WaveFunctionBuilder {
    pub fn new() -> WaveFunctionBuilder {
         WaveFunctionBuilder {
            phase: Phase::default(),
            view: (0.0,0.0)
        }
    }

    pub fn phase(mut self, phase: Phase) -> WaveFunctionBuilder {
        self.phase = phase;
        self
    }

    pub fn view(mut self, view: (f64, f64)) -> WaveFunctionBuilder {
        self.view = view;
        self
    }
    //
    // pub fn build(self) -> WaveFunction {
    //     let turning_point_boundaries = AiryWaveFunction::calc_ts(&self.phase, self.view);
    //
    // }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn calc_integration_points() {
        // let parts = WaveFunctionBuilder::new().;
    }
}