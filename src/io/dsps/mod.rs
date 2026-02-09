
pub trait Dsp: Send {
    fn process(&mut self, buffer: &mut [f32], sample_rate: u32, channels: usize);

    fn reset(&mut self);

    fn latency_frames(&self) -> usize;
}

pub(crate) struct DspChain {
    processors: Vec<Box<dyn Dsp>>,
}

impl DspChain {

    pub fn new() -> Self {
        Self { processors: Vec::new() }
    }

    pub fn push(&mut self, processor: Box<dyn Dsp>) {
        self.processors.push(processor);
    }

    pub fn process(&mut self, buffer: &mut [f32], sample_rate: u32, channels: usize) {
        for processor in &mut self.processors {
            processor.process(buffer, sample_rate, channels);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }
}