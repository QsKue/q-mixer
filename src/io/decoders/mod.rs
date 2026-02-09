mod symphonia;

pub use symphonia::SymphoniaDecoder;

// TODO: better error handling / refactor AI code
pub trait Decoder: Send {

    fn sample_rate(&self) -> u32;
    fn channels(&self) -> usize;
    fn total_samples(&self) -> Option<u64>;

    fn decode(&mut self, buffer: &mut [f32]) -> Result<usize, String>;

    fn position_samples(&self) -> u64;
    
    fn seekable(&self) -> bool { false }

    fn seek(&mut self, sample: u64) -> Result<u64, String> {
        Err("seek not supported".into())
    }
    
    fn is_eof(&self) -> bool;
    
    fn reset(&mut self) -> Result<(), String>;
}