use std::f32::consts::TAU;

use super::Decoder;

// Reference modulation ideas for generated audio patterns:
// Pitch-based:
// - [x] Linear glissando (portamento): smooth slides between notes.
// - [x] Exponential drift: slow detuning over long notes.
// - [x] Microtonal offset: global tuning shifts (e.g., A=432Hz or A=442Hz).
// - [x] Wide vibrato: >100 cents pitch modulation (crosses semitone boundaries).
// - [x] Inharmonic jitter: high-frequency random pitch noise for breathiness.
// Timbre-based:
// - [x] Harmonic overtone series: add multiples of the fundamental for thickness.
// - [x] Missing fundamental: omit F0 while retaining upper harmonics.
// - [x] Octave doubling: blend F0 and F1 with varying balance.
// - [x] Spectral tilt: bias energy toward higher (bright) or lower (dark) harmonics.
// Dynamics & temporal:
// - [x] Tremolo: periodic amplitude modulation.
// - [x] Percussive transients: short attack clicks at note onsets.
// - Inter-segment bleed: overlap releases into next attack (room/reverb).
// - [x] Dynamic swelling: crescendos that obscure clear note starts.
// Harmonic context:
// - [x] Polyphonic clusters: 3+ simultaneous notes (chords).
// - [x] Parallel key modulation: major to minor shifts.
// - [x] Relative key modulation: e.g., C major to A minor.
// - [x] Chromatic passing tones: insert non-diatonic notes.

#[derive(Clone, Copy, Debug)]
pub enum WaveformType {
    Sine,
    Square,
    Triangle,
    Sawtooth,
}

#[derive(Clone, Copy, Debug)]
pub struct WaveSegment {
    pub frequency_hz: f32,
    pub end_frequency_hz: Option<f32>,
    pub duration_secs: f32,
    pub amplitude: f32,
    pub waveform: WaveformType,
    pub vibrato_hz: Option<f32>,
    pub vibrato_depth: Option<f32>,
    pub tremolo_hz: Option<f32>,
    pub tremolo_depth: Option<f32>,
    pub attack_secs: f32,
    pub release_secs: f32,
}

impl WaveSegment {
    pub fn new(frequency_hz: f32, duration_secs: f32) -> Self {
        Self {
            frequency_hz,
            end_frequency_hz: None,
            duration_secs,
            amplitude: 0.4,
            waveform: WaveformType::Sine,
            vibrato_hz: None,
            vibrato_depth: None,
            tremolo_hz: None,
            tremolo_depth: None,
            attack_secs: 0.01,
            release_secs: 0.01,
        }
    }
}

#[derive(Clone, Debug)]
pub enum GeneratedWaveformPattern {
    Staircase {
        notes_hz: Vec<f32>,
        note_duration_secs: f32,
        amplitude: f32,
        waveform: WaveformType,
        tuning_offset_cents: f32,
    },
    Custom {
        segments: Vec<WaveSegment>,
        tuning_offset_cents: f32,
    },
    Samples {
        samples: Vec<f32>,
    },
}

#[derive(Clone, Copy, Debug)]
struct SegmentDescriptor {
    segment: WaveSegment,
    start_frame: u64,
    end_frame: u64,
}

struct GeneratedSamples {
    data: Vec<f32>,
    position: usize,
}

pub struct GeneratedDecoder {
    sample_rate: u32,
    channels: usize,
    total_samples: Option<u64>,
    segments: Vec<SegmentDescriptor>,
    current_segment: usize,
    position: u64,
    eof: bool,
    tuning_ratio: f32,
    samples: Option<GeneratedSamples>,
}

impl GeneratedDecoder {
    fn from_samples(sample_rate: u32, channels: usize, samples: Vec<f32>) -> Self {
        let total_samples = Some(samples.len() as u64);
        Self {
            sample_rate,
            channels,
            total_samples,
            segments: Vec::new(),
            current_segment: 0,
            position: 0,
            eof: samples.is_empty(),
            tuning_ratio: 1.0,
            samples: Some(GeneratedSamples {
                data: samples,
                position: 0,
            }),
        }
    }

    pub fn new(sample_rate: u32, channels: usize, pattern: GeneratedWaveformPattern) -> Self {
        let (segments, tuning_offset_cents) = match pattern {
            GeneratedWaveformPattern::Staircase {
                notes_hz,
                note_duration_secs,
                amplitude,
                waveform,
                tuning_offset_cents,
            } => (
                notes_hz
                    .into_iter()
                    .map(|frequency_hz| {
                        let mut segment = WaveSegment::new(frequency_hz, note_duration_secs);
                        segment.amplitude = amplitude;
                        segment.waveform = waveform;
                        segment
                    })
                    .collect(),
                tuning_offset_cents,
            ),
            GeneratedWaveformPattern::Custom {
                segments,
                tuning_offset_cents,
            } => (segments, tuning_offset_cents),
            GeneratedWaveformPattern::Samples { samples } => {
                return Self::from_samples(sample_rate, channels, samples);
            }
        };

        let mut descriptors = Vec::with_capacity(segments.len());
        let mut start_frame = 0u64;
        for segment in segments {
            let frames = (segment.duration_secs.max(0.0) * sample_rate as f32).round() as u64;
            let frames = frames.max(1);
            let end_frame = start_frame.saturating_add(frames);
            descriptors.push(SegmentDescriptor {
                segment,
                start_frame,
                end_frame,
            });
            start_frame = end_frame;
        }

        let total_samples = Some(start_frame.saturating_mul(channels as u64));

        let tuning_ratio = 2.0_f32.powf(tuning_offset_cents / 1200.0);

        Self {
            sample_rate,
            channels,
            total_samples,
            segments: descriptors,
            current_segment: 0,
            position: 0,
            eof: false,
            tuning_ratio,
            samples: None,
        }
    }

    fn segment_for_frame(&mut self, frame: u64) -> Option<&SegmentDescriptor> {
        while self.current_segment < self.segments.len() {
            let descriptor = &self.segments[self.current_segment];
            if frame < descriptor.end_frame {
                return Some(descriptor);
            }
            self.current_segment += 1;
        }
        None
    }

    fn sample_for_frame(&self, frame: u64, descriptor: &SegmentDescriptor) -> f32 {
        let segment = descriptor.segment;
        let frame_offset = frame.saturating_sub(descriptor.start_frame);
        let time = frame_offset as f32 / self.sample_rate as f32;
        let duration = segment.duration_secs.max(0.0);

        let base_frequency = if let Some(end_frequency) = segment.end_frequency_hz {
            let t = if duration > 0.0 {
                (time / duration).clamp(0.0, 1.0)
            } else {
                0.0
            };
            segment.frequency_hz + (end_frequency - segment.frequency_hz) * t
        } else {
            segment.frequency_hz
        };

        let vibrato = segment.vibrato_hz.unwrap_or(0.0);
        let vibrato_depth = segment.vibrato_depth.unwrap_or(0.0);
        let vibrato_factor = if vibrato > 0.0 && vibrato_depth > 0.0 {
            1.0 + (TAU * vibrato * time).sin() * vibrato_depth
        } else {
            1.0
        };

        let frequency = base_frequency * vibrato_factor * self.tuning_ratio;

        let waveform_sample = |freq: f32, t: f32, waveform: WaveformType| -> f32 {
            let phase = (freq * t) % 1.0;
            match waveform {
                WaveformType::Sine => (TAU * phase).sin(),
                WaveformType::Square => {
                    if phase < 0.5 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                WaveformType::Triangle => 4.0 * (phase - 0.5).abs() - 1.0,
                WaveformType::Sawtooth => 2.0 * (phase - 0.5),
            }
        };

        let sample = waveform_sample(frequency, time, segment.waveform);

        let attack = segment.attack_secs.max(0.0);
        let release = segment.release_secs.max(0.0);
        let release_start = (duration - release).max(0.0);
        let envelope = if attack > 0.0 && time < attack {
            time / attack
        } else if release > 0.0 && time > release_start {
            let remaining = (duration - time).max(0.0);
            remaining / release
        } else {
            1.0
        }
        .clamp(0.0, 1.0);

        let tremolo_rate = segment.tremolo_hz.unwrap_or(0.0);
        let tremolo_depth = segment.tremolo_depth.unwrap_or(0.0);
        let tremolo = if tremolo_rate > 0.0 && tremolo_depth > 0.0 {
            1.0 + (TAU * tremolo_rate * time).sin() * tremolo_depth
        } else {
            1.0
        };

        (sample * segment.amplitude * envelope * tremolo).clamp(-1.0, 1.0)
    }
}

impl Decoder for GeneratedDecoder {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> usize {
        self.channels
    }

    fn total_samples(&self) -> Option<u64> {
        self.total_samples
    }

    fn decode(&mut self, buffer: &mut [f32]) -> Result<usize, String> {
        if let Some(samples) = self.samples.as_mut() {
            if samples.position >= samples.data.len() {
                self.eof = true;
                return Ok(0);
            }
            let remaining = samples.data.len().saturating_sub(samples.position);
            let to_copy = remaining.min(buffer.len());
            buffer[..to_copy]
                .copy_from_slice(&samples.data[samples.position..samples.position + to_copy]);
            samples.position += to_copy;
            self.position = samples.position as u64;
            if to_copy < buffer.len() {
                self.eof = true;
            }
            return Ok(to_copy);
        }

        if self.eof || self.segments.is_empty() {
            return Ok(0);
        }

        let total_frames = buffer.len() / self.channels;
        let mut frames_written = 0usize;

        while frames_written < total_frames {
            let frame_index = self.position / self.channels as u64;
            if let Some(total_samples) = self.total_samples {
                if self.position >= total_samples {
                    self.eof = true;
                    break;
                }
            }

            let descriptor = match self.segment_for_frame(frame_index) {
                Some(segment) => segment.clone(),
                None => {
                    self.eof = true;
                    break;
                }
            };
            let sample = self.sample_for_frame(frame_index, &descriptor);

            let base_index = frames_written * self.channels;
            for channel in 0..self.channels {
                buffer[base_index + channel] = sample;
            }

            frames_written += 1;
            self.position = self.position.saturating_add(self.channels as u64);
        }

        Ok(frames_written * self.channels)
    }

    fn position_samples(&self) -> u64 {
        self.position
    }

    fn seekable(&self) -> bool {
        true
    }

    fn seek(&mut self, sample: u64) -> Result<u64, String> {
        let target = if let Some(total) = self.total_samples {
            sample.min(total)
        } else {
            sample
        };
        self.position = target;
        self.current_segment = 0;
        self.eof = false;
        if let Some(samples) = self.samples.as_mut() {
            samples.position = self.position as usize;
        }
        Ok(self.position)
    }

    fn is_eof(&self) -> bool {
        self.eof
    }

    fn reset(&mut self) -> Result<(), String> {
        self.position = 0;
        self.current_segment = 0;
        self.eof = false;
        if let Some(samples) = self.samples.as_mut() {
            samples.position = 0;
        }
        Ok(())
    }
}
