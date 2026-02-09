use crate::io::{
    decoders::Decoder,
    resamplers::Resampler,
    time_stretchers::TimeStretcher,
};

pub enum ChannelSource {
    File {
        path: String,
    },
    #[cfg(debug_assertions)]
    GeneratedAudio {
        sample_rate: u32,
        channels: usize,
        samples: Vec<f32>,
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
