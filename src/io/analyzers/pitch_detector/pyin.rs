use std::ops::Range;
use std::time::{Duration, Instant};

use crate::io::analyzers::Analyzer;
use pyin_rs::{PitchDetector, Pyin};

pub struct PyinPitchDetectorAnalyzer {
    ring: Vec<f32>,
    ring_pos: usize,
    filled: usize,

    sample_rate: u32,
    window: usize,
    hop: usize,
    samples_since_update: usize,
    min_interval: Duration,
    last_update: Instant,

    min_freq: f32,
    max_freq: f32,

    power_threshold: f32,

    window_buf: Vec<f64>,
    detector: Option<Pyin>,

    last_note: Option<i32>,
}

impl PyinPitchDetectorAnalyzer {
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
            power_threshold: 5e-4,
            window_buf: Vec::new(),
            detector: None,
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
        self.hop = (self.window / 2).max(sample_rate as usize / 50);

        let ring_len = (self.window * 2).max(self.window + 64);
        self.ring.resize(ring_len, 0.0);
        self.ring_pos = 0;
        self.filled = 0;

        self.window_buf.resize(self.window, 0.0);
        self.detector = Some(Pyin::new(self.window, sample_rate as usize));

        self.samples_since_update = 0;
        self.last_update = Instant::now();
        self.last_note = None;
    }

    fn push_mono_sample(&mut self, sample: f32) {
        if self.ring.is_empty() {
            return;
        }
        self.ring[self.ring_pos] = sample;
        self.ring_pos = (self.ring_pos + 1) % self.ring.len();
        self.filled = self.filled.saturating_add(1).min(self.ring.len());
    }

    fn read_last_into_from_ring(ring: &[f32], ring_pos: usize, dst: &mut [f64]) {
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
            *d = ring[idx] as f64;
            idx += 1;
            if idx == len {
                idx = 0;
            }
        }
    }

    fn f0_to_midi(f0: f32) -> Option<f32> {
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

impl Analyzer for PyinPitchDetectorAnalyzer {
    fn analyze(&mut self, input: &[f32], sample_rate: u32, channels: usize) {
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
            let mut sum = 0.0f32;
            for ch in 0..channels {
                sum += input[base + ch];
            }
            self.push_mono_sample(sum / channels as f32);
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

        let Some(detector) = self.detector.as_mut() else {
            return;
        };

        let f0 = detector.pitch(
            &self.window_buf,
            Some(Range {
                start: self.min_freq as f64,
                end: self.max_freq as f64,
            }),
        ) as f32;

        if !(f0.is_finite() && f0 >= self.min_freq && f0 <= self.max_freq) {
            return;
        }

        let power = self
            .window_buf
            .iter()
            .map(|sample| sample * sample)
            .sum::<f64>()
            / self.window_buf.len() as f64;
        if power < self.power_threshold as f64 {
            return;
        }

        let Some(midi) = Self::f0_to_midi(f0) else {
            return;
        };

        self.last_note = Some(midi.round() as i32);
    }
}
