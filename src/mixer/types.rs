use crate::io::{
    decoders::{Decoder, GeneratedWaveformPattern},
    resamplers::Resampler,
    time_stretchers::TimeStretcher,
};

pub enum ChannelSource {
    File {
        path: String,
    },
    GeneratedAudio {
        sample_rate: u32,
        channels: usize,
        pattern: GeneratedWaveformPattern,
    },
}

pub(crate) enum MixerTask {
    Initialize,
    RefillBuffer,
    ChannelLoad {
        index: usize,
        decoder: Box<dyn Decoder>,
        resampler: Box<dyn Resampler>,
        time_stretcher: Box<dyn TimeStretcher>,
    },
    ChannelPlay {
        index: usize,
    },
}

pub enum MixerEvent {}
