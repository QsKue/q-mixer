use std::collections::VecDeque;

use super::dsps::Dsp;
use tdpsola::{AlternatingHann, Speed, TdpsolaAnalysis, TdpsolaSynthesis};

pub trait TimeStretcher: Send + Dsp {
    fn set_params(&mut self, speed: f32, pitch_semitones: f32);

    fn speed(&self) -> f32;
    fn pitch_semitones(&self) -> f32;

    fn can_change_speed(&self) -> bool;
    fn can_change_pitch(&self) -> bool;
}

pub struct NoopTimeStretcher;

impl Dsp for NoopTimeStretcher {
    fn process(&mut self, _buffer: &mut [f32], _sample_rate: u32, _channels: usize) {}

    fn reset(&mut self) {}

    fn latency_frames(&self) -> usize {
        0
    }
}

impl TimeStretcher for NoopTimeStretcher {
    fn set_params(&mut self, _speed: f32, _pitch_semitones: f32) {}
    fn speed(&self) -> f32 {
        1.0
    }
    fn pitch_semitones(&self) -> f32 {
        0.0
    }
    fn can_change_speed(&self) -> bool {
        false
    }
    fn can_change_pitch(&self) -> bool {
        false
    }
}

pub struct TdPsolaTimeStretcher {
    pitch_semitones: f32,
    sample_rate: u32,
    channels: usize,
    states: Vec<TdPsolaChannelState>,
}

impl TdPsolaTimeStretcher {
    pub fn new() -> Self {
        Self {
            pitch_semitones: 0.0,
            sample_rate: 0,
            channels: 0,
            states: Vec::new(),
        }
    }

    fn ensure_config(&mut self, sample_rate: u32, channels: usize) {
        if self.sample_rate == sample_rate && self.channels == channels && !self.states.is_empty() {
            return;
        }

        self.sample_rate = sample_rate;
        self.channels = channels;
        self.states = (0..channels)
            .map(|_| TdPsolaChannelState::new(sample_rate, self.pitch_semitones))
            .collect();
    }
}

impl Default for TdPsolaTimeStretcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Dsp for TdPsolaTimeStretcher {
    fn process(&mut self, buffer: &mut [f32], sample_rate: u32, channels: usize) {
        if channels == 0 {
            return;
        }

        self.ensure_config(sample_rate, channels);

        for frame in buffer.chunks_exact_mut(channels) {
            for (ch, sample) in frame.iter_mut().enumerate() {
                *sample = self.states[ch].process_sample(*sample, self.pitch_semitones);
            }
        }
    }

    fn reset(&mut self) {
        self.states.clear();
        self.sample_rate = 0;
        self.channels = 0;
    }

    fn latency_frames(&self) -> usize {
        0
    }
}

impl TimeStretcher for TdPsolaTimeStretcher {
    fn set_params(&mut self, _speed: f32, pitch_semitones: f32) {
        self.pitch_semitones = pitch_semitones;
    }

    fn speed(&self) -> f32 {
        1.0
    }

    fn pitch_semitones(&self) -> f32 {
        self.pitch_semitones
    }

    fn can_change_speed(&self) -> bool {
        false
    }

    fn can_change_pitch(&self) -> bool {
        true
    }
}

struct TdPsolaChannelState {
    window: AlternatingHann,
    analysis: TdpsolaAnalysis,
    synthesis: TdpsolaSynthesis,
    source_wavelength: f32,
    estimator_buffer: VecDeque<f32>,
    samples_since_estimate: usize,
    sample_rate: u32,
}

impl TdPsolaChannelState {
    const MIN_F0_HZ: f32 = 70.0;
    const MAX_F0_HZ: f32 = 700.0;
    const ESTIMATE_PERIOD_SAMPLES: usize = 128;
    const ESTIMATE_WINDOW_SAMPLES: usize = 1024;

    fn new(sample_rate: u32, pitch_semitones: f32) -> Self {
        let sample_rate_f = sample_rate.max(1) as f32;
        let source_wavelength = (sample_rate_f / 220.0).clamp(8.0, 1024.0);
        let window = AlternatingHann::new(source_wavelength);
        let analysis = TdpsolaAnalysis::new(&window);
        let target_wavelength = Self::target_wavelength(source_wavelength, pitch_semitones);
        let synthesis = TdpsolaSynthesis::new(Speed::from_f32(1.0), target_wavelength);

        Self {
            window,
            analysis,
            synthesis,
            source_wavelength,
            estimator_buffer: VecDeque::with_capacity(Self::ESTIMATE_WINDOW_SAMPLES),
            samples_since_estimate: 0,
            sample_rate,
        }
    }

    fn process_sample(&mut self, input: f32, pitch_semitones: f32) -> f32 {
        self.push_estimator_sample(input);
        self.samples_since_estimate += 1;

        if self.samples_since_estimate >= Self::ESTIMATE_PERIOD_SAMPLES {
            self.samples_since_estimate = 0;
            if let Some(wavelength) = self.estimate_wavelength() {
                self.source_wavelength = wavelength;
                self.window = AlternatingHann::new(self.source_wavelength);
            }
        }

        self.analysis.push_sample(input, &mut self.window);

        let target_wavelength = Self::target_wavelength(self.source_wavelength, pitch_semitones);
        self.synthesis.set_wavelength(target_wavelength);

        let output = self
            .synthesis
            .try_get_sample(&self.analysis)
            .unwrap_or(input);
        self.synthesis.step(&self.analysis);
        output
    }

    fn target_wavelength(source_wavelength: f32, pitch_semitones: f32) -> f32 {
        let ratio = 2.0_f32.powf(pitch_semitones / 12.0);
        (source_wavelength / ratio).clamp(2.0, 4096.0)
    }

    fn push_estimator_sample(&mut self, sample: f32) {
        if self.estimator_buffer.len() == Self::ESTIMATE_WINDOW_SAMPLES {
            self.estimator_buffer.pop_front();
        }
        self.estimator_buffer.push_back(sample);
    }

    fn estimate_wavelength(&self) -> Option<f32> {
        if self.estimator_buffer.len() < Self::ESTIMATE_WINDOW_SAMPLES / 2 {
            return None;
        }

        let samples: Vec<f32> = self.estimator_buffer.iter().copied().collect();
        let sr = self.sample_rate.max(1) as f32;
        let min_lag = (sr / Self::MAX_F0_HZ).max(2.0) as usize;
        let max_lag = (sr / Self::MIN_F0_HZ).max((min_lag + 1) as f32) as usize;

        let mut best_lag = 0usize;
        let mut best_score = 0.0_f32;

        for lag in min_lag..max_lag {
            if lag >= samples.len() {
                break;
            }

            let mut corr = 0.0_f32;
            let mut energy_a = 0.0_f32;
            let mut energy_b = 0.0_f32;
            for i in lag..samples.len() {
                let a = samples[i];
                let b = samples[i - lag];
                corr += a * b;
                energy_a += a * a;
                energy_b += b * b;
            }

            let norm = (energy_a * energy_b).sqrt();
            if norm > 1e-9 {
                let score = corr / norm;
                if score > best_score {
                    best_score = score;
                    best_lag = lag;
                }
            }
        }

        if best_lag > 0 && best_score > 0.2 {
            Some(best_lag as f32)
        } else {
            None
        }
    }
}
