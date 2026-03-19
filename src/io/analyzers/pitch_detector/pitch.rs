use std::time::{Duration, Instant};

use crate::io::analyzers::{AnalysisEvent, Analyzer};

// Uses Bitstream Autocorrelation Function
pub struct BCFPitchDetector {
    ring: Vec<f64>,
    ring_pos: usize,
    filled: usize,

    sample_rate: u32,
    window: usize,
    hop: usize,
    samples_since_update: usize,
    min_interval: Duration,
    last_update: Instant,

    min_freq: f64,
    max_freq: f64,

    window_buf: Vec<f64>,

    last_note: Option<i32>,
}

impl BCFPitchDetector {
    pub fn new() -> Self {
        Self {
            ring: Vec::new(),
            ring_pos: 0,
            filled: 0,
            sample_rate: 0,
            window: 0,
            hop: 0,
            samples_since_update: 0,
            min_interval: Duration::from_millis(10),
            last_update: Instant::now(),
            min_freq: 80.0,
            max_freq: 1_000.0,
            window_buf: Vec::new(),
            last_note: None,
        }
    }

    pub fn set_min_interval(&mut self, interval: Duration) {
        self.min_interval = interval;
    }

    pub fn detected_note(&self) -> Option<i32> {
        self.last_note
    }

    pub fn detected_key(&self) -> Option<String> {
        self.last_note.map(|note| {
            let name = Self::midi_to_note_name(note);
            let octave = (note / 12) - 1;
            format!("{}{}", name, octave)
        })
    }

    fn configure_if_needed(&mut self, sample_rate: u32) {
        if self.sample_rate == sample_rate && self.window != 0 {
            return;
        }

        self.sample_rate = sample_rate;
        self.window = if sample_rate >= 44_100 { 2_048 } else { 1_024 };
        self.hop = (sample_rate as usize / 200).max(64);

        let ring_len = (self.window * 2).max(self.window + 64);
        self.ring.resize(ring_len, 0.0);
        self.ring_pos = 0;
        self.filled = 0;

        self.window_buf.resize(self.window, 0.0);

        self.samples_since_update = 0;
        self.last_update = Instant::now();
        self.last_note = None;
    }

    fn push_mono_sample(&mut self, sample: f64) {
        if self.ring.is_empty() {
            return;
        }
        self.ring[self.ring_pos] = sample;
        self.ring_pos = (self.ring_pos + 1) % self.ring.len();
        self.filled = self.filled.saturating_add(1).min(self.ring.len());
    }

    fn read_last_into_from_ring(ring: &[f64], ring_pos: usize, dst: &mut [f64]) {
        let n = dst.len();
        let len = ring.len();
        if n == 0 || len == 0 || n > len {
            return;
        }

        let mut idx = if ring_pos >= n {
            ring_pos - n
        } else {
            len + ring_pos - n
        };

        for d in dst.iter_mut() {
            *d = ring[idx];
            idx += 1;
            if idx == len {
                idx = 0;
            }
        }
    }

    fn f0_to_midi(f0: f64) -> Option<f64> {
        if !(f0.is_finite() && f0 > 0.0) {
            return None;
        }
        let midi = 69.0 + 12.0 * (f0 / 440.0).log2();
        midi.is_finite().then_some(midi)
    }

    fn midi_to_note_name(midi_rounded: i32) -> &'static str {
        const N: [&str; 12] = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let idx = ((midi_rounded % 12) + 12) % 12;
        N[idx as usize]
    }
}

impl Default for BCFPitchDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer for BCFPitchDetector {
    fn analyze(
        &mut self,
        input: &[f32],
        sample_rate: u32,
        channels: usize,
        out_events: &mut Vec<AnalysisEvent>,
    ) {
        if channels == 0 || input.is_empty() || sample_rate == 0 {
            return;
        }
        self.configure_if_needed(sample_rate);

        let frames = input.len() / channels;
        if frames == 0 {
            return;
        }

        for frame in 0..frames {
            let base = frame * channels;
            let mut sum = 0.0f64;
            for ch in 0..channels {
                sum += input[base + ch] as f64;
            }
            self.push_mono_sample(sum / channels as f64);
            self.samples_since_update += 1;
        }

        if self.filled < self.window || self.samples_since_update < self.hop {
            return;
        }
        if self.last_update.elapsed() < self.min_interval {
            return;
        }
        self.samples_since_update = 0;
        self.last_update = Instant::now();

        Self::read_last_into_from_ring(&self.ring, self.ring_pos, &mut self.window_buf);

        let (detected_hz, confidence) = pitch::detect(&self.window_buf);
        if !(detected_hz.is_finite() && detected_hz > 0.0) {
            return;
        }

        let f0 = detected_hz * sample_rate as f64 / 48_000.0;
        if !(f0 >= self.min_freq && f0 <= self.max_freq) {
            return;
        }

        let Some(midi) = Self::f0_to_midi(f0) else {
            return;
        };

        let rounded_midi = midi.round() as i32;
        self.last_note = Some(rounded_midi);
        out_events.push(AnalysisEvent::Pitch {
            midi: rounded_midi,
            hz: f0 as f32,
            confidence: confidence as f32,
        });
    }
}
