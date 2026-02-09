// "DSP + heuristics" realtime pitch/key detector (no ML).
// - Uses McLeod Pitch Method (MPM) via the `pitch-detection` crate.
// - Heuristics: confidence gating, octave correction, smoothing, hysteresis.
// - No per-callback heap allocation (ring buffer + window buffer preallocated/reused).
//
// Usage:
//   pub mod realtime_pitch_analyzer;
//   stream.analyzer_mut().push(Box::new(RealtimePitchAnalyzer::new()));
//
// Your Analyzer trait:
//   pub trait Analyzer: Send {
//       fn analyze(&mut self, input: &[f32], sample_rate: u32, channels: usize);
//   }

use std::time::{Duration, Instant};

use crate::io::analyzers::Analyzer;

use pitch_detection::detector::mcleod::McLeodDetector;
use pitch_detection::detector::PitchDetector;

pub struct RealtimePitchAnalyzer {
    // Ring buffer holding mono samples
    ring: Vec<f32>,
    ring_pos: usize,
    filled: usize,

    // Cached SR + derived sizes
    sample_rate: u32,
    window: usize, // analysis window size (samples)
    hop: usize,    // how often to update (samples)

    // Range constraints (still useful for sanity checks / tuning)
    min_freq: f32,
    max_freq: f32,

    // Preallocated analysis buffer
    window_buf: Vec<f32>,

    // MPM detector (pitch-detection crate)
    detector: Option<McLeodDetector<f32>>,

    // Control-rate scheduling
    samples_since_update: usize,
    last_update: Instant,
    min_interval: Duration,

    // State for heuristics
    last_f0: Option<f32>,
    smooth_log2: Option<f32>, // smoothing in log2 domain
    last_note: Option<i32>,
    candidate_note: Option<i32>,
    candidate_count: u32,

    // pitch-detection parameters
    power_threshold: f32,
    clarity_threshold: f32,

    // Smoothing & hysteresis knobs
    smoothing_alpha: f32,  // 0..1 (higher = more smoothing)
    note_hold_updates: u32, // require N consecutive updates before declaring a new note
}

impl RealtimePitchAnalyzer {
    pub fn new() -> Self {
        Self {
            ring: Vec::new(),
            ring_pos: 0,
            filled: 0,

            sample_rate: 0,
            window: 0,
            hop: 0,

            // Defaults suitable for vocal-ish pitch correction.
            // Adjust min_freq if you need bass notes.
            min_freq: 80.0,
            max_freq: 1000.0,

            window_buf: Vec::new(),
            detector: None,

            samples_since_update: 0,
            last_update: Instant::now(),
            // update at most 100 Hz (10ms). You can lower this to reduce CPU.
            min_interval: Duration::from_millis(10),

            last_f0: None,
            smooth_log2: None,
            last_note: None,
            candidate_note: None,
            candidate_count: 0,

            // Tune these for your input level.
            // If detection never triggers, lower power_threshold.
            // If it triggers on noise, raise it.
            power_threshold: 5e-4,
            clarity_threshold: 0.7,

            // Smoothing in log2 domain; ~0.85 is fairly smooth but responsive.
            smoothing_alpha: 0.85,
            note_hold_updates: 2,
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

        // Window: tradeoff between latency and stability.
        // 1024 @ 48k = 21.3ms; 1024 @ 44.1k = 23.2ms
        // For lower min_freq, increase this (e.g. 2048) at the cost of control latency.
        self.window = if sample_rate >= 44_100 { 1024 } else { 512 };

        // Hop: control update rate. 1–5ms feels "snappy"; 10ms is often enough.
        // Here we tie it to SR to be ~5ms.
        self.hop = ((sample_rate as usize) / 200).max(64);

        // Ring buffer: keep at least one window + some slack
        let ring_len = (self.window * 2).max(self.window + 64);
        self.ring.resize(ring_len, 0.0);
        self.ring_pos = 0;
        self.filled = 0;

        // Preallocate analysis buffer
        self.window_buf.resize(self.window, 0.0);

        // McLeod detector (docs often use padding = size/2)
        let padding = self.window / 2;
        self.detector = Some(McLeodDetector::new(self.window, padding));

        // Reset heuristics when SR changes
        self.last_f0 = None;
        self.smooth_log2 = None;
        self.last_note = None;
        self.candidate_note = None;
        self.candidate_count = 0;
        self.samples_since_update = 0;
    }

    #[inline]
    fn push_mono_sample(&mut self, s: f32) {
        if self.ring.is_empty() {
            return;
        }
        self.ring[self.ring_pos] = s;
        self.ring_pos = (self.ring_pos + 1) % self.ring.len();
        self.filled = self.filled.saturating_add(1).min(self.ring.len());
    }

    // Copy last `n` samples (ending at ring_pos) into `dst` (length n).
    // IMPORTANT: This is a free function-style helper to avoid borrowing `&self`
    // while also mutably borrowing `self.window_buf` (Rust borrow checker issue).
    #[inline]
    fn read_last_into_from_ring(ring: &[f32], ring_pos: usize, dst: &mut [f32]) {
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

    fn octave_correct(&self, f0: f32, clarity: f32) -> f32 {
        let Some(prev) = self.last_f0 else { return f0 };

        // If we're very confident, trust the new estimate.
        if clarity >= 0.85 {
            return f0;
        }

        // Prefer continuity: if f0 is close to 2x or 0.5x prev, pull it toward prev.
        let two = f0 * 2.0;
        let half = f0 * 0.5;

        let d = |a: f32, b: f32| (a - b).abs();

        let mut best = f0;
        let mut best_dist = d(f0, prev);

        if d(two, prev) < best_dist {
            best = two;
            best_dist = d(two, prev);
        }
        if d(half, prev) < best_dist {
            best = half;
        }
        best
    }

    fn smooth_f0(&mut self, f0: f32) -> f32 {
        // Smooth in log2 domain (musically linear)
        let log2 = f0.log2();
        let out_log2 = if let Some(prev) = self.smooth_log2 {
            self.smoothing_alpha * prev + (1.0 - self.smoothing_alpha) * log2
        } else {
            log2
        };
        self.smooth_log2 = Some(out_log2);
        2.0f32.powf(out_log2)
    }

    fn f0_to_midi(f0: f32) -> Option<f32> {
        if !(f0.is_finite() && f0 > 0.0) {
            return None;
        }
        let midi = 69.0 + 12.0 * (f0 / 440.0).log2();
        midi.is_finite().then_some(midi)
    }

    fn midi_to_note_name(midi_rounded: i32) -> &'static str {
        const N: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        let idx = ((midi_rounded % 12) + 12) % 12;
        N[idx as usize]
    }
}

impl Analyzer for RealtimePitchAnalyzer {
    fn analyze(&mut self, input: &[f32], sample_rate: u32, channels: usize) {
        if channels == 0 || input.is_empty() || sample_rate == 0 {
            return;
        }
        self.configure_if_needed(sample_rate);

        let frames = input.len() / channels;
        if frames == 0 {
            return;
        }

        // Push mono samples into ring buffer (no allocation)
        for f in 0..frames {
            let mut sum = 0.0f32;
            let base = f * channels;
            for ch in 0..channels {
                sum += input[base + ch];
            }
            self.push_mono_sample(sum / channels as f32);
            self.samples_since_update += 1;
        }

        // Don’t update too often (CPU protection); also require enough data.
        if self.filled < self.window || self.samples_since_update < self.hop {
            return;
        }
        if self.last_update.elapsed() < self.min_interval {
            return;
        }
        self.last_update = Instant::now();
        self.samples_since_update = 0;

        // Reuse window buffer (no per-call alloc)
        Self::read_last_into_from_ring(&self.ring, self.ring_pos, &mut self.window_buf);

        let Some(det) = self.detector.as_mut() else { return; };

        // McLeod pitch (MPM) from pitch-detection crate.
        // get_pitch(signal, sample_rate, power_threshold, clarity_threshold)
        let pitch = det.get_pitch(
            &self.window_buf,
            sample_rate as usize,
            self.power_threshold,
            self.clarity_threshold,
        );
        let pitch = match pitch {
            Some(p) => p,
            None => return,
        };

        let raw_f0 = pitch.frequency as f32;
        let clarity = pitch.clarity as f32;

        // Keep your frequency sanity checks
        if !(raw_f0.is_finite() && raw_f0 >= self.min_freq * 0.8 && raw_f0 <= self.max_freq * 1.2) {
            return;
        }

        // Heuristics pipeline
        let f0 = self.octave_correct(raw_f0, clarity);
        let f0 = self.smooth_f0(f0);
        self.last_f0 = Some(f0);

        let midi = match Self::f0_to_midi(f0) {
            Some(m) => m,
            None => return,
        };

        let note = midi.round() as i32;

        // Hysteresis: require the same candidate note for a few updates
        match self.candidate_note {
            Some(c) if c == note => {
                self.candidate_count = self.candidate_count.saturating_add(1);
            }
            _ => {
                self.candidate_note = Some(note);
                self.candidate_count = 1;
            }
        }

        if self.candidate_count >= self.note_hold_updates && self.last_note != Some(note) {
            self.last_note = Some(note);
        }
    }
}
