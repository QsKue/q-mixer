// TODO: refactor everything, try everything, and if something kicks you... KICK IT BACK
mod rubato;

pub use rubato::RubatoResampler;

pub struct ResamplerResult {
    pub out_frames: usize,
    pub src_frames_used: usize,
}

pub enum ResamplerStatus {
    Progress { result: ResamplerResult },
    NeedMoreInput,
    Flushed,
}

pub trait Resampler: Send { 

    fn produce_into(
        &mut self,
        decode_cache: &mut std::collections::VecDeque<f32>,
        out_sample_rate: u32,
        min_out_frames: usize,
        buffer: Option<&mut [f32]>,
        out_channels: usize,
        eof: bool,
    ) -> ResamplerStatus;

    fn drain_out_with_conv(&mut self, dst: &mut [f32], out_channels: usize) -> usize;

    fn reset(&mut self);
}