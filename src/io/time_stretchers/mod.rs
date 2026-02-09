use super::dsps::Dsp;

pub trait TimeStretcher: Send + Dsp {
    fn set_params(&mut self, speed: f32, pitch_semitones: f32);

    fn speed(&self) -> f32;
    fn pitch_semitones(&self) -> f32;
}

pub struct NoopTimeStretcher;

impl Dsp for NoopTimeStretcher {
    fn process(&mut self, _buffer: &mut [f32], _sample_rate: u32, _channels: usize) {}

    fn reset(&mut self) {}

    fn latency_frames(&self) -> usize { 0 }
}

impl TimeStretcher for NoopTimeStretcher {
    fn set_params(&mut self, _speed: f32, _pitch_semitones: f32) {}
    fn speed(&self) -> f32 { 1.0 }
    fn pitch_semitones(&self) -> f32 { 0.0 }
}