pub mod pitch;

pub trait Analyzer: Send {
    fn analyze(&mut self, input: &[f32], sample_rate: u32, channels: usize);
}
