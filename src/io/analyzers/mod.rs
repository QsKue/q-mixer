use crate::mixer::Channel;

pub trait Analyzer: Send {
    type Output: Send + 'static;

    fn analyze(&mut self, input: &[f32], sample_rate: u32, channels: usize) -> Option<Self::Output>;

    fn on_result(channel: &mut Channel, result: Option<Self::Output>);
    fn on_result_mut_buffer(buffer: &mut [f32], sample_rate: u32, channels: usize);
}