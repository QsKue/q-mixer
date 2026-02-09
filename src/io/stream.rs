use std::collections::VecDeque;

use crate::io::{
    analyzers::Analyzer, decoders::Decoder, dsps::DspChain, resamplers::{Resampler, ResamplerStatus}, time_stretchers::TimeStretcher, types::StreamTime
};

// TODO: variable ?
const STREAM_CACHE_DURATION_SECONDS: f32 = 0.5;

pub(crate) struct Stream {
    decoder: Box<dyn Decoder>,
    decoder_cache: VecDeque<f32>,

    analyzers: Vec<Box<dyn Analyzer>>,
    pre_fx: DspChain,
    time_stretcher: Box<dyn TimeStretcher>,
    resampler: Box<dyn Resampler>,
    post_fx: DspChain,

    gain: Option<f32>,
    volume: f32,
}

impl Stream {

    pub fn new(
        mut decoder: Box<dyn Decoder>,
        resampler: Box<dyn Resampler>,
        time_stretcher: Box<dyn TimeStretcher>,
        pos: Option<StreamTime>,
        gain: Option<f32>,
        volume: Option<f32>,
    ) -> Self {

        let sample_rate = decoder.sample_rate();
        let channels = decoder.channels();

        let pos = pos.unwrap_or(StreamTime::Beat(0.0));
        decoder.seek(pos.sample(decoder.sample_rate()));
        
        Self {
            decoder,
            decoder_cache: VecDeque::with_capacity(((sample_rate * channels as u32) as f32 * STREAM_CACHE_DURATION_SECONDS) as usize),

            analyzers: Vec::new(),
            resampler: resampler,
            pre_fx: DspChain::new(),
            time_stretcher: time_stretcher,
            post_fx: DspChain::new(),

            gain: gain, // TODO: compute zero gain or have the deck do it
            volume: volume.unwrap_or(1f32),
        }
    }

    pub fn position_get(&self) -> StreamTime {
        StreamTime::from_sample(self.decoder.position_samples(), self.decoder.sample_rate())
    }

    pub fn position_set(&mut self, duration: StreamTime) {
        self.decoder.seek(duration.sample(self.decoder.sample_rate()));
    }

    pub fn gain_set(&mut self, gain: f32) {
        self.gain = Some(gain);
    }

    pub fn gain_get(&self) -> f32 {
        self.gain.unwrap_or(1.0)
    }

    pub fn volume_set(&mut self, volume: f32) {
        self.volume = volume;
    }

    pub fn volume_get(&self) -> f32 {
        self.volume
    }

    pub fn analyzers(&mut self) -> &mut Vec<Box<dyn Analyzer>> {
        &mut self.analyzers
    }

    pub fn pre_fx_mut(&mut self) -> &mut DspChain {
        &mut self.pre_fx
    }

    pub fn post_fx_mut(&mut self) -> &mut DspChain {
        &mut self.post_fx
    }

    pub fn get_data(&mut self, buffer: &mut [f32], out_sample_rate: u32, out_channels: usize) -> usize {

        buffer.fill(0.0);
        
        let samples = self.get_stream_data(buffer, out_sample_rate, out_channels);
        if samples == 0 { return 0; }

        if !self.post_fx.is_empty() {
            self.post_fx.process(&mut buffer[..samples], out_sample_rate, out_channels);
        }

        // Volume / Gain
        for s in buffer[..samples].iter_mut() {
            *s *= self.volume * self.gain.unwrap_or(1.0);
        }

        samples
    }

    fn get_stream_data(&mut self, buffer: &mut [f32], target_sample_rate: u32, target_channels: usize) -> usize {
        let mut written = 0;

        // AI code, refactor to make simpler
        written += self.resampler.drain_out_with_conv(
            &mut buffer[written..],
            target_channels,
        );

        while written < buffer.len() {
            let need_frames = (buffer.len() - written) / target_channels;
            if need_frames == 0 {
                break;
            }

            match self.resampler.produce_into(
                &mut self.decoder_cache,
                target_sample_rate,
                need_frames,
                Some(&mut buffer[written..]),
                target_channels,
                self.decoder.is_eof(),
            ) {
                ResamplerStatus::Progress { result } => {
                    if result.out_frames == 0 && result.src_frames_used == 0 {
                        if !self.decode_more_input() {
                            break;
                        }
                    }
                    written += result.out_frames * target_channels;
                }
                ResamplerStatus::NeedMoreInput => {
                    if !self.decode_more_input() {
                        break;
                    }
                }
                ResamplerStatus::Flushed => break,
            }
        }

        written
    }

    fn decode_more_input(&mut self) -> bool {
        let mut temp = vec![0.0; 4096];

        // this as well, also we need to account for length processed
        match self.decoder.decode(&mut temp) {
            Ok(frames) if frames > 0 => {
                let samples = frames * self.decoder.channels();
                
                if !self.analyzers.is_empty() {
                    for analyzer in &mut self.analyzers {
                        let processed = analyzer.analyze(&temp[..samples], self.decoder.sample_rate(), self.decoder.channels());
                    }
                }

                if !self.pre_fx.is_empty() {
                    let processed = self.pre_fx.process(&mut temp[..samples], self.decoder.sample_rate(), self.decoder.channels());
                }

                let stretched = self.time_stretcher.process(&mut temp[..samples], self.decoder.sample_rate(), self.decoder.channels());

                self.decoder_cache.extend(temp[..samples].to_vec());

                true
            }
            _ => false,
        }
    }
}
