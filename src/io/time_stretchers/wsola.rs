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
    synth: Vec<f32>,
    synth_read_pos: usize,
    nominal_pos: usize,
    initialized: bool,
}

impl ChannelState {
    fn new() -> Self {
        Self {
            input: Vec::new(),
            synth: Vec::new(),
            synth_read_pos: 0,
            nominal_pos: 0,
            initialized: false,
        }
    }

    fn reset(&mut self) {
        self.input.clear();
        self.synth.clear();
        self.synth_read_pos = 0;
        self.nominal_pos = 0;
        self.initialized = false;
    }

    fn available(&self) -> usize {
        self.synth.len().saturating_sub(self.synth_read_pos)
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

    fn pitch_factor(&self) -> f32 {
        2f32.powf(self.pitch_semitones / 12.0)
    }

    fn time_scale(&self) -> f32 {
        let factor = self.pitch_factor();
        (self.speed / factor).clamp(0.05, 4.0)
    }

    fn ensure_layout(&mut self, sample_rate: u32, channels: usize) {
        let desired_frame_size = ((sample_rate as f32) * 0.02).round() as usize;
        let frame_size = desired_frame_size.max(128);
        let overlap = ((frame_size as f32) * 0.5).round() as usize;
        let overlap = overlap.clamp(32, frame_size.saturating_sub(1));
        let synth_hop = frame_size - overlap;
        let analysis_hop = ((synth_hop as f32) * self.time_scale()).round() as usize;
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

    fn apply_pitch_shift_with_factor(samples: &mut [f32], factor: f32) {
        if (factor - 1.0).abs() < 1e-4 || samples.len() < 2 {
            return;
        }
        let src = samples.to_vec();
        let len = src.len();
        for (n, out) in samples.iter_mut().enumerate() {
            let pos = (n as f32 * factor).min((len - 1) as f32);
            let i0 = pos.floor() as usize;
            let i1 = (i0 + 1).min(len - 1);
            let frac = pos - i0 as f32;
            *out = src[i0] * (1.0 - frac) + src[i1] * frac;
        }
    }

    fn process_channel(&mut self, channel: usize, input: &[f32], output_frames: usize) -> Vec<f32> {
        let frame_size = self.frame_size;
        let overlap = self.overlap;
        let search = self.search;
        let analysis_hop = self.analysis_hop;
        let pitch_factor = self.pitch_factor();

        let state = &mut self.states[channel];
        state.input.extend_from_slice(input);

        while state.available() < output_frames {
            if !state.initialized {
                if state.input.len() < frame_size {
                    break;
                }
                let segment = state.input[0..frame_size].to_vec();
                append_segment(state, &segment, overlap);
                state.nominal_pos = analysis_hop;
                continue;
            }

            if state.nominal_pos + frame_size > state.input.len() {
                break;
            }

            let tail_start = state.synth.len().saturating_sub(overlap);
            let ref_overlap = &state.synth[tail_start..tail_start + overlap];

            let search_start = state.nominal_pos.saturating_sub(search);
            let search_end = (state.nominal_pos + search).min(state.input.len() - frame_size);

            let mut best_pos = state.nominal_pos;
            let mut best_score = f64::NEG_INFINITY;
            for pos in search_start..=search_end {
                let cand = &state.input[pos..pos + overlap];
                let score = normalized_cross_correlation(ref_overlap, cand);
                if score > best_score {
                    best_score = score;
                    best_pos = pos;
                }
            }

            let segment = state.input[best_pos..best_pos + frame_size].to_vec();
            append_segment(state, &segment, overlap);
            state.nominal_pos = best_pos + analysis_hop;

            let drop_before = state.nominal_pos.saturating_sub(frame_size + search);
            if drop_before > 0 {
                state.input.drain(..drop_before);
                state.nominal_pos -= drop_before;
            }

            if state.synth_read_pos > frame_size * 8 {
                state.synth.drain(..state.synth_read_pos);
                state.synth_read_pos = 0;
            }
        }

        let mut out = vec![0.0f32; output_frames];
        let available = state.available().min(output_frames);
        if available > 0 {
            let start = state.synth_read_pos;
            let end = start + available;
            out[..available].copy_from_slice(&state.synth[start..end]);
            state.synth_read_pos += available;
        }

        if available < output_frames {
            let rem = output_frames - available;
            let tail_start = input.len().saturating_sub(rem);
            out[available..].copy_from_slice(&input[tail_start..tail_start + rem]);
        }

        Self::apply_pitch_shift_with_factor(&mut out, pitch_factor);
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

        for ch in 0..channels {
            let out = self.process_channel(ch, &inputs[ch], frames);
            for (frame, sample) in out.iter().enumerate() {
                buffer[frame * channels + ch] = *sample;
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
            self.analysis_hop = ((self.synth_hop as f32) * self.time_scale()).round() as usize;
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

fn append_segment(state: &mut ChannelState, segment: &[f32], overlap: usize) {
    if !state.initialized {
        state.synth.extend_from_slice(segment);
        state.initialized = true;
        return;
    }

    if state.synth.len() < overlap {
        state.synth.extend_from_slice(segment);
        return;
    }

    let tail_start = state.synth.len() - overlap;
    for i in 0..overlap {
        let t = i as f32 / overlap as f32;
        let fade_in = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
        let fade_out = 1.0 - fade_in;
        state.synth[tail_start + i] = state.synth[tail_start + i] * fade_out + segment[i] * fade_in;
    }

    state.synth.extend_from_slice(&segment[overlap..]);
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

    fn estimate_freq(signal: &[f32], sample_rate: f32) -> f32 {
        let mut zero_cross = 0usize;
        for i in 1..signal.len() {
            let a = signal[i - 1];
            let b = signal[i];
            if (a <= 0.0 && b > 0.0) || (a >= 0.0 && b < 0.0) {
                zero_cross += 1;
            }
        }
        (zero_cross as f32 * sample_rate) / (2.0 * signal.len() as f32)
    }

    #[test]
    fn identity_keeps_frequency_close() {
        let mut s = WsolaTimeStretcher::new();
        s.set_params(1.0, 0.0);
        let sr = 48_000;
        let channels = 1;
        let total = 48_000;
        let mut out = Vec::new();

        let mut input = vec![0.0f32; total];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (2.0 * std::f32::consts::PI * 220.0 * (i as f32) / (sr as f32)).sin();
        }

        for chunk in input.chunks(1024) {
            let mut buf = chunk.to_vec();
            s.process(&mut buf, sr, channels);
            out.extend_from_slice(&buf);
        }

        let in_f = estimate_freq(&input[sr as usize / 4..sr as usize], sr as f32);
        let out_f = estimate_freq(&out[sr as usize / 4..sr as usize], sr as f32);
        assert!((out_f - in_f).abs() < 8.0);
    }

    #[test]
    fn pitch_shift_changes_frequency() {
        let sr = 48_000;
        let channels = 1;
        let total = 48_000;

        let mut input = vec![0.0f32; total];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (2.0 * std::f32::consts::PI * 220.0 * (i as f32) / (sr as f32)).sin();
        }

        let mut baseline = WsolaTimeStretcher::new();
        baseline.set_params(1.0, 0.0);
        let mut shifted = WsolaTimeStretcher::new();
        shifted.set_params(1.0, 12.0);

        let mut out_base = Vec::new();
        let mut out_shift = Vec::new();
        for chunk in input.chunks(1024) {
            let mut a = chunk.to_vec();
            let mut b = chunk.to_vec();
            baseline.process(&mut a, sr, channels);
            shifted.process(&mut b, sr, channels);
            out_base.extend_from_slice(&a);
            out_shift.extend_from_slice(&b);
        }

        let mut diff = 0.0f32;
        for (a, b) in out_base.iter().zip(out_shift.iter()) {
            diff += (a - b).abs();
        }
        diff /= out_base.len() as f32;
        assert!(diff > 0.01);
    }

    #[test]
    fn no_large_zero_gaps() {
        let mut s = WsolaTimeStretcher::new();
        s.set_params(1.0, 0.0);
        let sr = 48_000;
        let channels = 1;
        let total = 48_000;
        let mut zeroish = 0usize;
        let mut count = 0usize;

        let mut input = vec![0.0f32; total];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (2.0 * std::f32::consts::PI * 220.0 * (i as f32) / (sr as f32)).sin();
        }

        for chunk in input.chunks(1024) {
            let mut buf = chunk.to_vec();
            s.process(&mut buf, sr, channels);
            for x in &buf {
                if x.abs() < 1e-5 {
                    zeroish += 1;
                }
                count += 1;
            }
        }

        assert!((zeroish as f32) / (count as f32) < 0.2);
    }
}
