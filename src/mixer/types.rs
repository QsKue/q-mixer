use crate::io::{decoders::Decoder, resamplers::Resampler, time_stretchers::TimeStretcher};

pub enum ChannelSource {
    File { path: String }
}

pub(crate) enum MixerTask {
    Initialize,
    RefillBuffer,
    ChannelLoad { index: usize, decoder: Box<dyn Decoder>, resampler: Box<dyn Resampler>, time_stretcher: Box<dyn TimeStretcher> },
    ChannelPlay { index: usize },
}

pub enum MixerEvent {
    
}
