// "DSP + heuristics" realtime pitch/key detector (no ML).
// - Time-domain pitch detection via normalized autocorrelation (MPM-ish).
// - Heuristics: confidence gating, octave correction, smoothing, hysteresis.
// - No per-callback heap allocation (ring buffer preallocated/reused).
//
// Usage:
//   pub mod fcpe_pitch_analyzer;
//   stream.analyzer_mut().push(Box::new(FcpePitchAnalyzer::new()));
//
// Your Analyzer trait:
//   pub trait Analyzer: Send {
//       fn analyze(&mut self, input: &[f32], sample_rate: u32, channels: usize);
//   }

use std::io::{self, Write};
use std::time::{Duration, Instant};

use crate::io::analyzers::Analyzer;

pub struct RealtimePitchAnalyzer {
    // Ring buffer holding mono samples
    ring: Vec<f32>,
    ring_pos: usize,
    filled: usize,

    // Cached SR + derived sizes
    sample_rate: u32,
    window: usize,       // analysis window size (samples)
    hop: usize,          // how often to update (samples)
    min_tau: usize,      // max_freq -> min lag
    max_tau: usize,      // min_freq -> max lag

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

    // Tuning knobs
    min_freq: f32,
    max_freq: f32,
    clarity_threshold: f32,
    smoothing_alpha: f32, // 0..1 (higher = more smoothing)
    note_hold_updates: u32,
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
            min_tau: 0,
            max_tau: 0,

            samples_since_update: 0,
            last_update: Instant::now(),
            // update at most 100 Hz (10ms). You can lower this to reduce CPU.
            min_interval: Duration::from_millis(10),

            last_f0: None,
            smooth_log2: None,
            last_note: None,
            candidate_note: None,
            candidate_count: 0,

            // Defaults suitable for vocal-ish pitch correction.
            // Adjust min_freq if you need bass notes.
            min_freq: 80.0,
            max_freq: 1000.0,
            clarity_threshold: 0.35,
            // Smoothing in log2 domain; ~0.85 is fairly smooth but responsive.
            smoothing_alpha: 0.85,
            // Require N consecutive updates before declaring a new note.
            note_hold_updates: 2,
        }
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

        // Lag range from frequency range:
        // f0 = sr / tau  => tau = sr / f0
        self.min_tau = ((sample_rate as f32) / self.max_freq).floor().max(2.0) as usize;
        self.max_tau = ((sample_rate as f32) / self.min_freq).ceil() as usize;

        // Ensure tau range fits window
        // We need at least window - tau >= 1
        self.max_tau = self.max_tau.min(self.window.saturating_sub(2)).max(self.min_tau + 1);

        // Ring buffer: keep at least one window + some slack
        let ring_len = (self.window * 2).max(self.window + self.max_tau + 4);
        self.ring.resize(ring_len, 0.0);
        self.ring_pos = 0;
        self.filled = 0;

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

    // Copy last `n` samples (ending at ring_pos) into `dst` (length n)
    fn read_last_into(&self, dst: &mut [f32]) {
        let n = dst.len();
        let len = self.ring.len();
        let mut idx = if self.ring_pos >= n { self.ring_pos - n } else { len + self.ring_pos - n };

        for d in dst.iter_mut() {
            *d = self.ring[idx];
            idx += 1;
            if idx == len {
                idx = 0;
            }
        }
    }

    // Normalized autocorrelation pitch estimate (MPM-ish):
    // - Compute r[tau] for tau in [min_tau, max_tau]
    // - Choose tau with max r[tau]
    // - Compute clarity = r_max / r0
    fn estimate_f0(&self, window_buf: &[f32]) -> Option<(f32, f32)> {
        let w = window_buf;
        if w.len() < self.window || self.min_tau >= self.max_tau {
            return None;
        }

        // Remove DC (simple mean) for better correlation
        let mut mean = 0.0f32;
        for &v in w.iter() {
            mean += v;
        }
        mean /= w.len() as f32;

        // r0 = energy
        let mut r0 = 0.0f32;
        for &v in w.iter() {
            let x = v - mean;
            r0 += x * x;
        }
        if !(r0 > 1e-8) {
            return None;
        }

        let mut best_tau = 0usize;
        let mut best_r = -1.0f32;

        // Compute correlation for each tau
        // r[tau] = sum (x[i] * x[i+tau])
        for tau in self.min_tau..=self.max_tau {
            let mut r = 0.0f32;
            let limit = w.len().saturating_sub(tau);
            if limit < 4 {
                break;
            }

            let mut i = 0usize;
            while i < limit {
                let a = w[i] - mean;
                let b = w[i + tau] - mean;
                r += a * b;
                i += 1;
            }

            if r > best_r {
                best_r = r;
                best_tau = tau;
            }
        }

        if best_tau == 0 {
            return None;
        }

        let clarity = (best_r / r0).max(0.0);
        let f0 = (self.sample_rate as f32) / (best_tau as f32);

        // sanity
        if !(f0.is_finite() && f0 >= self.min_freq * 0.8 && f0 <= self.max_freq * 1.2) {
            return None;
        }

        Some((f0, clarity))
    }

    fn octave_correct(&self, f0: f32, clarity: f32) -> f32 {
        let Some(prev) = self.last_f0 else { return f0 };
        // If we're very confident, trust the new estimate.
        if clarity >= 0.65 {
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

    fn print_key(&self, note: i32, f0: f32) {
        let name = Self::midi_to_note_name(note);
        let octave = (note / 12) - 1;
        println!("\rDetected: {}{} ({:.2} Hz)   ", name, octave, f0);
        // let _ = io::stdout().flush();
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

        // Pull latest window into a stack buffer (small) or reused Vec (no per-call alloc)
        // Here we reuse a local Vec but avoid realloc by keeping capacity.
        // If you want ZERO alloc ever, store this Vec in self.
        let mut wbuf = vec![0.0f32; self.window];
        self.read_last_into(&mut wbuf);

        let (raw_f0, clarity) = match self.estimate_f0(&wbuf) {
            Some(v) => v,
            None => return,
        };

        // Confidence gating (voicing)
        if clarity < self.clarity_threshold {
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

        if self.candidate_count >= self.note_hold_updates {
            if self.last_note != Some(note) {
                self.last_note = Some(note);
                self.print_key(note, f0);
            } else {
                // Still print (optional) to keep UI alive. Comment out if you prefer silence.
                self.print_key(note, f0);
            }
        }
    }
}
