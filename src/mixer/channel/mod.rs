use crate::io::{decoders::Decoder, resamplers::Resampler, stream::Stream, time_stretchers::TimeStretcher};

pub struct Channel {
    is_failed: bool,
    is_playing: bool,

    stream: Option<Stream>,
}

impl Channel {

    pub fn new(
        decoder: Box<dyn Decoder>,
        resampler: Box<dyn Resampler>,
        time_stretcher: Box<dyn TimeStretcher>,
    ) -> Self {

        // load in a separate thread
        Self {
            is_failed: false,
            is_playing: false,

            stream: Some(Stream::new(decoder, resampler, time_stretcher, None, None, None)),
        }
    }

    pub fn is_ready(&self) -> bool {
        self.stream.is_some()
    }

    pub fn play(&mut self) {
        self.is_playing = true;
    }

    pub fn get_data(&mut self, buffer: &mut [f32], out_sample_rate: u32, out_channels: usize) -> usize {

        if self.is_failed {
            return 0;
        }

        if !self.is_playing {
            return 0;
        }

        let Some(stream) = self.stream.as_mut() else {
            return 0;
        };

        stream.get_data(buffer, out_sample_rate, out_channels)
    }
}