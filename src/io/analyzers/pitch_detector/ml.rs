use std::time::{Duration, Instant};

use crate::io::analyzers::{AnalysisEvent, Analyzer};
use pitch_detection::detector::PitchDetector;
use pitch_detection::detector::mcleod::McLeodDetector;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MlPitchModel {
    SwiftF0,
    Rmvpe,
    Crepe,
}

#[derive(Debug, Clone, Copy)]
pub struct MlPitchEstimate {
    pub hz: f32,
    pub confidence: f32,
}

pub trait MlPitchEngine {
    fn prepare(&mut self, _model: MlPitchModel, _sample_rate: u32, _window: usize) {}

    fn infer(
        &mut self,
        model: MlPitchModel,
        mono_window: &[f32],
        sample_rate: u32,
        min_freq: f32,
        max_freq: f32,
    ) -> Option<MlPitchEstimate>;
}

#[derive(Debug, Clone)]
pub struct OnnxModelParams {
    pub model_path: String,
    pub input_tensor_name: String,
    pub f0_output_tensor_name: String,
    pub confidence_output_tensor_name: Option<String>,
    pub expected_sample_rate: u32,
    pub expected_window_size: usize,
    pub hop_size: usize,
    pub model_min_freq: f32,
    pub model_max_freq: f32,
    pub output_kind: OnnxOutputKind,
}

#[derive(Debug, Clone)]
pub enum OnnxOutputKind {
    DirectHz,
    CentsBinLogits { min_hz: f32, bins_per_octave: f32 },
}

#[derive(Debug, Clone)]
pub struct OnnxRawOutput {
    pub pitch: Vec<f32>,
    pub confidence: Option<Vec<f32>>,
}

impl OnnxModelParams {
    pub fn for_swiftf0(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            input_tensor_name: "input_audio".to_string(),
            f0_output_tensor_name: "pitch_hz".to_string(),
            confidence_output_tensor_name: Some("confidence".to_string()),
            expected_sample_rate: 16_000,
            expected_window_size: 1024,
            hop_size: 160,
            model_min_freq: 50.0,
            model_max_freq: 1_100.0,
            output_kind: OnnxOutputKind::DirectHz,
        }
    }

    pub fn for_rmvpe(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            input_tensor_name: "waveform".to_string(),
            f0_output_tensor_name: "f0".to_string(),
            confidence_output_tensor_name: Some("periodicity".to_string()),
            expected_sample_rate: 16_000,
            expected_window_size: 1024,
            hop_size: 160,
            model_min_freq: 40.0,
            model_max_freq: 1_100.0,
            output_kind: OnnxOutputKind::DirectHz,
        }
    }

    pub fn for_crepe(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            input_tensor_name: "input".to_string(),
            f0_output_tensor_name: "pitch".to_string(),
            confidence_output_tensor_name: Some("confidence".to_string()),
            expected_sample_rate: 16_000,
            expected_window_size: 1024,
            hop_size: 160,
            model_min_freq: 50.0,
            model_max_freq: 2_000.0,
            output_kind: OnnxOutputKind::DirectHz,
        }
    }

    pub fn with_output_kind(mut self, output_kind: OnnxOutputKind) -> Self {
        self.output_kind = output_kind;
        self
    }
}

#[derive(Debug, Clone)]
pub struct OnnxPitchConfig {
    pub swiftf0: Option<OnnxModelParams>,
    pub rmvpe: Option<OnnxModelParams>,
    pub crepe: Option<OnnxModelParams>,
}

impl OnnxPitchConfig {
    pub fn empty() -> Self {
        Self {
            swiftf0: None,
            rmvpe: None,
            crepe: None,
        }
    }

    fn get(&self, model: MlPitchModel) -> Option<&OnnxModelParams> {
        match model {
            MlPitchModel::SwiftF0 => self.swiftf0.as_ref(),
            MlPitchModel::Rmvpe => self.rmvpe.as_ref(),
            MlPitchModel::Crepe => self.crepe.as_ref(),
        }
    }
}

pub trait OnnxRuntime {
    fn infer(&mut self, _params: &OnnxModelParams, _mono_window: &[f32]) -> Option<OnnxRawOutput>;
}

#[derive(Default)]
pub struct StubOnnxRuntime;

impl OnnxRuntime for StubOnnxRuntime {
    fn infer(&mut self, _params: &OnnxModelParams, _mono_window: &[f32]) -> Option<OnnxRawOutput> {
        None
    }
}

pub struct OnnxMlPitchEngine {
    config: OnnxPitchConfig,
    runtime: Box<dyn OnnxRuntime>,
    fallback: McLeodMlFallbackEngine,
}

impl OnnxMlPitchEngine {
    pub fn new(config: OnnxPitchConfig) -> Self {
        Self::with_runtime(config, Box::new(StubOnnxRuntime))
    }

    pub fn with_runtime(config: OnnxPitchConfig, runtime: Box<dyn OnnxRuntime>) -> Self {
        Self {
            config,
            runtime,
            fallback: McLeodMlFallbackEngine::new(),
        }
    }

    fn validate_input(params: &OnnxModelParams, mono_window: &[f32], sample_rate: u32) -> bool {
        params.expected_sample_rate > 0
            && params.expected_window_size > 0
            && !mono_window.is_empty()
            && sample_rate > 0
    }

    fn resample_to_model_window(
        mono_window: &[f32],
        source_sample_rate: u32,
        params: &OnnxModelParams,
    ) -> Option<Vec<f32>> {
        if source_sample_rate == 0 || params.expected_sample_rate == 0 || mono_window.is_empty() {
            return None;
        }

        let needed_source_window = ((params.expected_window_size as f64 * source_sample_rate as f64
            / params.expected_sample_rate as f64)
            .round() as usize)
            .max(2);

        let mut source = vec![0.0f32; needed_source_window];
        if mono_window.len() >= needed_source_window {
            let start = mono_window.len() - needed_source_window;
            source.copy_from_slice(&mono_window[start..]);
        } else {
            let pad = needed_source_window - mono_window.len();
            source[pad..].copy_from_slice(mono_window);
        }

        let mut out = vec![0.0f32; params.expected_window_size];
        let src_last = source.len() - 1;
        let out_last = out.len().saturating_sub(1).max(1);

        for (i, value) in out.iter_mut().enumerate() {
            let pos = i as f64 * src_last as f64 / out_last as f64;
            let idx = pos.floor() as usize;
            let frac = (pos - idx as f64) as f32;
            let right = (idx + 1).min(src_last);
            *value = source[idx] * (1.0 - frac) + source[right] * frac;
        }

        Some(out)
    }

    fn decode_output(params: &OnnxModelParams, raw: OnnxRawOutput) -> Option<MlPitchEstimate> {
        let confidence = raw
            .confidence
            .as_ref()
            .and_then(|values| values.first().copied());

        match params.output_kind {
            OnnxOutputKind::DirectHz => {
                let hz = *raw.pitch.first()?;
                let conf = confidence.unwrap_or(1.0).clamp(0.0, 1.0);
                Some(MlPitchEstimate {
                    hz,
                    confidence: conf,
                })
            }
            OnnxOutputKind::CentsBinLogits {
                min_hz,
                bins_per_octave,
            } => {
                if raw.pitch.is_empty() || min_hz <= 0.0 || bins_per_octave <= 0.0 {
                    return None;
                }

                let (best_idx, best_logit) = raw
                    .pitch
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))?;

                let hz = min_hz * 2.0f32.powf(best_idx as f32 / bins_per_octave);

                let logit_sum: f32 = raw.pitch.iter().map(|x| x.exp()).sum();
                let prob = if logit_sum > 0.0 {
                    best_logit.exp() / logit_sum
                } else {
                    0.0
                };

                Some(MlPitchEstimate {
                    hz,
                    confidence: confidence.unwrap_or(prob).clamp(0.0, 1.0),
                })
            }
        }
    }
}

impl MlPitchEngine for OnnxMlPitchEngine {
    fn prepare(&mut self, model: MlPitchModel, sample_rate: u32, window: usize) {
        self.fallback.prepare(model, sample_rate, window);
    }

    fn infer(
        &mut self,
        model: MlPitchModel,
        mono_window: &[f32],
        sample_rate: u32,
        min_freq: f32,
        max_freq: f32,
    ) -> Option<MlPitchEstimate> {
        let clamped = min_freq.max(1.0);
        let params = self.config.get(model);

        let onnx_estimate = params.and_then(|params| {
            if !Self::validate_input(params, mono_window, sample_rate) {
                return None;
            }

            let model_input = Self::resample_to_model_window(mono_window, sample_rate, params)?;
            let raw = self.runtime.infer(params, &model_input)?;
            let estimate = Self::decode_output(params, raw)?;
            (estimate.hz >= params.model_min_freq && estimate.hz <= params.model_max_freq)
                .then_some(estimate)
        });

        let estimate = onnx_estimate.or_else(|| {
            self.fallback
                .infer(model, mono_window, sample_rate, clamped, max_freq)
        })?;

        (estimate.hz >= min_freq && estimate.hz <= max_freq).then_some(estimate)
    }
}

/// Drop-in engine that mimics an ML backend contract while relying on McLeod
/// internally. This keeps the API stable while real SwiftF0/RMVPE/CREPE
/// inference engines are wired in later.
pub struct McLeodMlFallbackEngine {
    detector: Option<McLeodDetector<f32>>,
    window: usize,
    power_threshold: f32,
    clarity_threshold: f32,
}

impl McLeodMlFallbackEngine {
    pub fn new() -> Self {
        Self {
            detector: None,
            window: 0,
            power_threshold: 5e-4,
            clarity_threshold: 0.7,
        }
    }
}

impl Default for McLeodMlFallbackEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MlPitchEngine for McLeodMlFallbackEngine {
    fn prepare(&mut self, _model: MlPitchModel, _sample_rate: u32, window: usize) {
        if self.window == window && self.detector.is_some() {
            return;
        }
        self.window = window;
        let padding = window / 2;
        self.detector = Some(McLeodDetector::new(window, padding));
    }

    fn infer(
        &mut self,
        model: MlPitchModel,
        mono_window: &[f32],
        sample_rate: u32,
        min_freq: f32,
        max_freq: f32,
    ) -> Option<MlPitchEstimate> {
        let detector = self.detector.as_mut()?;
        let pitch = detector.get_pitch(
            mono_window,
            sample_rate as usize,
            self.power_threshold,
            self.clarity_threshold,
        )?;

        let hz = pitch.frequency;
        if !(hz.is_finite() && hz >= min_freq && hz <= max_freq) {
            return None;
        }

        let model_conf_scale = match model {
            MlPitchModel::SwiftF0 => 1.0,
            MlPitchModel::Rmvpe => 0.98,
            MlPitchModel::Crepe => 0.95,
        };

        Some(MlPitchEstimate {
            hz,
            confidence: (pitch.clarity * model_conf_scale).clamp(0.0, 1.0),
        })
    }
}

pub struct MlPitchDetector {
    model: MlPitchModel,
    engine: Box<dyn MlPitchEngine>,

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

    window_buf: Vec<f32>,

    last_note: Option<i32>,
}

impl MlPitchDetector {
    pub fn new(model: MlPitchModel) -> Self {
        Self::with_engine(model, Box::new(McLeodMlFallbackEngine::new()))
    }

    pub fn with_engine(model: MlPitchModel, engine: Box<dyn MlPitchEngine>) -> Self {
        Self {
            model,
            engine,
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

    pub fn with_onnx(model: MlPitchModel, onnx: OnnxPitchConfig) -> Self {
        Self::with_engine(model, Box::new(OnnxMlPitchEngine::new(onnx)))
    }

    pub fn set_frequency_range(&mut self, min_freq: f32, max_freq: f32) {
        self.min_freq = min_freq.max(1.0);
        self.max_freq = max_freq.max(self.min_freq);
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
        self.window = if sample_rate >= 44_100 { 1024 } else { 512 };
        self.hop = (sample_rate as usize / 200).max(64);

        let ring_len = (self.window * 2).max(self.window + 64);
        self.ring.resize(ring_len, 0.0);
        self.ring_pos = 0;
        self.filled = 0;

        self.window_buf.resize(self.window, 0.0);
        self.engine.prepare(self.model, sample_rate, self.window);

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

impl Analyzer for MlPitchDetector {
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

        let Some(estimate) = self.engine.infer(
            self.model,
            &self.window_buf,
            sample_rate,
            self.min_freq,
            self.max_freq,
        ) else {
            return;
        };

        let Some(midi) = Self::f0_to_midi(estimate.hz) else {
            return;
        };

        let rounded_midi = midi.round() as i32;
        self.last_note = Some(rounded_midi);
        out_events.push(AnalysisEvent::Pitch {
            midi: rounded_midi,
            hz: estimate.hz,
            confidence: estimate.confidence,
        });
    }
}

pub struct SwiftF0PitchDetector {
    inner: MlPitchDetector,
}

impl SwiftF0PitchDetector {
    pub fn new() -> Self {
        Self {
            inner: MlPitchDetector::new(MlPitchModel::SwiftF0),
        }
    }

    pub fn with_onnx(params: OnnxModelParams) -> Self {
        let mut config = OnnxPitchConfig::empty();
        config.swiftf0 = Some(params);
        Self {
            inner: MlPitchDetector::with_onnx(MlPitchModel::SwiftF0, config),
        }
    }
}

impl Default for SwiftF0PitchDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer for SwiftF0PitchDetector {
    fn analyze(
        &mut self,
        input: &[f32],
        sample_rate: u32,
        channels: usize,
        out_events: &mut Vec<AnalysisEvent>,
    ) {
        self.inner.analyze(input, sample_rate, channels, out_events);
    }
}

pub struct RmvpePitchDetector {
    inner: MlPitchDetector,
}

impl RmvpePitchDetector {
    pub fn new() -> Self {
        Self {
            inner: MlPitchDetector::new(MlPitchModel::Rmvpe),
        }
    }

    pub fn with_onnx(params: OnnxModelParams) -> Self {
        let mut config = OnnxPitchConfig::empty();
        config.rmvpe = Some(params);
        Self {
            inner: MlPitchDetector::with_onnx(MlPitchModel::Rmvpe, config),
        }
    }
}

impl Default for RmvpePitchDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer for RmvpePitchDetector {
    fn analyze(
        &mut self,
        input: &[f32],
        sample_rate: u32,
        channels: usize,
        out_events: &mut Vec<AnalysisEvent>,
    ) {
        self.inner.analyze(input, sample_rate, channels, out_events);
    }
}

pub struct CrepePitchDetector {
    inner: MlPitchDetector,
}

impl CrepePitchDetector {
    pub fn new() -> Self {
        Self {
            inner: MlPitchDetector::new(MlPitchModel::Crepe),
        }
    }

    pub fn with_onnx(params: OnnxModelParams) -> Self {
        let mut config = OnnxPitchConfig::empty();
        config.crepe = Some(params);
        Self {
            inner: MlPitchDetector::with_onnx(MlPitchModel::Crepe, config),
        }
    }
}

impl Default for CrepePitchDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer for CrepePitchDetector {
    fn analyze(
        &mut self,
        input: &[f32],
        sample_rate: u32,
        channels: usize,
        out_events: &mut Vec<AnalysisEvent>,
    ) {
        self.inner.analyze(input, sample_rate, channels, out_events);
    }
}
