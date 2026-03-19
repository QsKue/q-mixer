use std::collections::VecDeque;

use super::TimeStretcher;
use crate::io::dsps::Dsp;

pub struct WsolaTimeStretcher {
    speed: f32,
    pitch_semitones: f32,
    frame_size: usize,
    overlap: usize,
    search: usize,
    analysis_hop: usize,
    synth_hop: usize,
    channels: usize,
    states: Vec<ChannelState>,
}

struct ChannelState {
    input: Vec<f32>,
    output: VecDeque<f32>,
    nominal_pos: usize,
    tail: Vec<f32>,
    initialized: bool,
}

impl ChannelState {
    fn new() -> Self {
        Self {
            input: Vec::new(),
            output: VecDeque::new(),
            nominal_pos: 0,
            tail: Vec::new(),
            initialized: false,
        }
    }

    fn reset(&mut self) {
        self.input.clear();
        self.output.clear();
        self.nominal_pos = 0;
        self.tail.clear();
        self.initialized = false;
    }
}

impl WsolaTimeStretcher {
    pub fn new() -> Self {
        Self {
            speed: 1.0,
            pitch_semitones: 0.0,
            frame_size: 0,
            overlap: 0,
            search: 0,
            analysis_hop: 0,
            synth_hop: 0,
            channels: 0,
            states: Vec::new(),
        }
    }

    fn effective_speed(&self) -> f32 {
        let pitch_factor = 2f32.powf(self.pitch_semitones / 12.0);
        (self.speed * pitch_factor).clamp(0.05, 4.0)
    }

    fn ensure_layout(&mut self, sample_rate: u32, channels: usize) {
        let desired_frame_size = ((sample_rate as f32) * 0.02).round() as usize;
        let frame_size = desired_frame_size.max(128);
        let overlap = ((frame_size as f32) * 0.5).round() as usize;
        let overlap = overlap.clamp(32, frame_size.saturating_sub(1));
        let synth_hop = frame_size - overlap;
        let analysis_hop = ((synth_hop as f32) * self.effective_speed()).round() as usize;
        let analysis_hop = analysis_hop.max(1);
        let search = ((sample_rate as f32) * 0.01).round() as usize;
        let search = search.max(8);

        let changed = self.frame_size != frame_size
            || self.overlap != overlap
            || self.analysis_hop != analysis_hop
            || self.synth_hop != synth_hop
            || self.search != search
            || self.channels != channels;

        if changed {
            self.frame_size = frame_size;
            self.overlap = overlap;
            self.analysis_hop = analysis_hop;
            self.synth_hop = synth_hop;
            self.search = search;
            self.channels = channels;
            self.states = (0..channels).map(|_| ChannelState::new()).collect();
        }
    }

    fn process_channel(&mut self, channel: usize, input: &[f32], output_frames: usize) -> Vec<f32> {
        let state = &mut self.states[channel];
        state.input.extend_from_slice(input);

        let target = output_frames + self.frame_size;
        while state.output.len() < target {
            if !state.initialized {
                if state.input.len() < self.frame_size {
                    break;
                }
                for &v in &state.input[..self.frame_size] {
                    state.output.push_back(v);
                }
                state.tail = state.input[self.frame_size - self.overlap..self.frame_size].to_vec();
                state.nominal_pos = self.analysis_hop;
                state.initialized = true;
                continue;
            }

            if state.nominal_pos + self.frame_size > state.input.len() {
                break;
            }

            let search_start = state.nominal_pos.saturating_sub(self.search);
            let search_end = (state.nominal_pos + self.search)
                .min(state.input.len().saturating_sub(self.frame_size));

            let mut best_pos = state.nominal_pos;
            let mut best_score = f64::NEG_INFINITY;

            for pos in search_start..=search_end {
                let score = normalized_cross_correlation(
                    &state.tail,
                    &state.input[pos..pos + self.overlap],
                );
                if score > best_score {
                    best_score = score;
                    best_pos = pos;
                }
            }

            let segment = &state.input[best_pos..best_pos + self.frame_size];
            let mut mixed_overlap = vec![0.0f32; self.overlap];
            for i in 0..self.overlap {
                let t = i as f32 / self.overlap as f32;
                let fade_in = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
                let fade_out = 1.0 - fade_in;
                mixed_overlap[i] = state.tail[i] * fade_out + segment[i] * fade_in;
            }

            for &v in &mixed_overlap {
                state.output.push_back(v);
            }
            for &v in &segment[self.overlap..] {
                state.output.push_back(v);
            }

            let start_tail = self.frame_size - self.overlap;
            state.tail.clear();
            state
                .tail
                .extend_from_slice(&segment[start_tail..self.frame_size]);
            state.nominal_pos = best_pos + self.analysis_hop;

            let drop_before = state
                .nominal_pos
                .saturating_sub(self.frame_size + self.search);
            if drop_before > 0 {
                state.input.drain(..drop_before);
                state.nominal_pos = state.nominal_pos.saturating_sub(drop_before);
            }
        }

        let mut out = vec![0.0f32; output_frames];
        for sample in &mut out {
            if let Some(v) = state.output.pop_front() {
                *sample = v;
            }
        }
        out
    }
}

impl Dsp for WsolaTimeStretcher {
    fn process(&mut self, buffer: &mut [f32], sample_rate: u32, channels: usize) {
        if channels == 0 || buffer.is_empty() {
            return;
        }

        self.ensure_layout(sample_rate, channels);

        let frames = buffer.len() / channels;
        if frames == 0 {
            return;
        }

        let mut inputs = vec![vec![0.0f32; frames]; channels];
        for frame in 0..frames {
            for ch in 0..channels {
                inputs[ch][frame] = buffer[frame * channels + ch];
            }
        }

        let mut outputs = vec![vec![0.0f32; frames]; channels];
        for ch in 0..channels {
            outputs[ch] = self.process_channel(ch, &inputs[ch], frames);
        }

        for frame in 0..frames {
            for ch in 0..channels {
                buffer[frame * channels + ch] = outputs[ch][frame];
            }
        }
    }

    fn reset(&mut self) {
        for state in &mut self.states {
            state.reset();
        }
    }

    fn latency_frames(&self) -> usize {
        self.frame_size
    }
}

impl TimeStretcher for WsolaTimeStretcher {
    fn set_params(&mut self, speed: f32, pitch_semitones: f32) {
        self.speed = speed;
        self.pitch_semitones = pitch_semitones;
        if self.synth_hop > 0 {
            self.analysis_hop = ((self.synth_hop as f32) * self.effective_speed()).round() as usize;
            self.analysis_hop = self.analysis_hop.max(1);
        }
    }

    fn speed(&self) -> f32 {
        self.speed
    }

    fn pitch_semitones(&self) -> f32 {
        self.pitch_semitones
    }
}

fn normalized_cross_correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    for i in 0..n {
        sum_a += a[i] as f64;
        sum_b += b[i] as f64;
    }
    let mean_a = sum_a / n as f64;
    let mean_b = sum_b / n as f64;

    let mut ab = 0.0f64;
    let mut a2 = 0.0f64;
    let mut b2 = 0.0f64;
    for i in 0..n {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        ab += da * db;
        a2 += da * da;
        b2 += db * db;
    }

    let denom = (a2 * b2).sqrt();
    if denom <= 1e-12 { 0.0 } else { ab / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wsola_produces_non_silent_output() {
        let mut s = WsolaTimeStretcher::new();
        s.set_params(1.2, 0.0);
        let sr = 48_000;
        let channels = 1;
        let mut buf = vec![0.0f32; 4096];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = (2.0 * std::f32::consts::PI * 220.0 * (i as f32) / (sr as f32)).sin();
        }
        s.process(&mut buf, sr, channels);
        let energy: f32 = buf.iter().map(|v| v * v).sum();
        assert!(energy > 1.0);
    }

    #[test]
    fn ncc_identity_is_one() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let c = normalized_cross_correlation(&x, &x);
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pitch_parameter_changes_processing_rate() {
        let mut normal = WsolaTimeStretcher::new();
        normal.set_params(1.0, 0.0);
        let mut pitched = WsolaTimeStretcher::new();
        pitched.set_params(1.0, 12.0);

        let sr = 48_000;
        let channels = 1;
        let mut probe = vec![0.0f32; 4096];
        normal.process(&mut probe, sr, channels);
        pitched.process(&mut probe, sr, channels);

        assert!(pitched.analysis_hop > normal.analysis_hop);
    }
}
