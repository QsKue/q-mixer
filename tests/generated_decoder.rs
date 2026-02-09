use audio::io::analyzers::Analyzer;
use dasp_signal::{self as signal, Signal};
use std::f64::consts::TAU;
use std::time::Duration;

use audio::io::analyzers::pitch::RealtimePitchAnalyzer;

const SAMPLE_RATE: u32 = 44_100;

fn render_signal<S>(mut signal: S, seconds: f32) -> Vec<f32>
where
    S: Signal<Frame = f64>,
{
    let samples = (SAMPLE_RATE as f32 * seconds) as usize;
    let mut data = Vec::with_capacity(samples);
    for _ in 0..samples {
        data.push(signal.next() as f32);
    }
    data
}

fn triangle_hz(hz: f64) -> impl Signal<Frame = f64> {
    let phase = signal::phase(signal::rate(SAMPLE_RATE as f64).const_hz(hz));
    phase.map(|p| 1.0 - 4.0 * (p - 0.5).abs())
}

fn render_note_with_vibrato(
    base_hz: f64,
    note_secs: f32,
    vibrato_hz: f64,
    depth: f64,
) -> Vec<f32> {
    let samples = (SAMPLE_RATE as f32 * note_secs) as usize;
    let mut data = Vec::with_capacity(samples);
    let mut lfo = signal::rate(SAMPLE_RATE as f64).const_hz(vibrato_hz).sine();
    let mut phase = 0.0;
    for _ in 0..samples {
        let hz = base_hz * (1.0 + lfo.next() * depth);
        phase = (phase + TAU * hz / SAMPLE_RATE as f64) % TAU;
        data.push(phase.sin() as f32);
    }
    data
}

fn render_notes_with_vibrato(
    notes: &[f64],
    note_secs: f32,
    vibrato_hz: f64,
    depth: f64,
) -> Vec<f32> {
    let mut samples = Vec::new();
    for &note in notes {
        samples.extend(render_note_with_vibrato(note, note_secs, vibrato_hz, depth));
    }
    samples
}

fn analyze_samples(samples: &[f32]) -> Option<String> {
    let mut analyzer = RealtimePitchAnalyzer::new();
    analyzer.set_min_interval(Duration::from_millis(0));
    for chunk in samples.chunks(1024) {
        analyzer.analyze(chunk, SAMPLE_RATE, 1);
    }
    analyzer.detected_key()
}

fn expected_key_from_hz(freq_hz: f64) -> String {
    let midi = 69.0 + 12.0 * (freq_hz / 440.0).log2();
    let note = midi.round() as i32;
    let name = match ((note % 12) + 12) % 12 {
        0 => "C",
        1 => "C#",
        2 => "D",
        3 => "D#",
        4 => "E",
        5 => "F",
        6 => "F#",
        7 => "G",
        8 => "G#",
        9 => "A",
        10 => "A#",
        _ => "B",
    };
    let octave = (note / 12) - 1;
    format!("{}{}", name, octave)
}

fn assert_detected_key(samples: &[f32], expected: &str) {
    let detected = analyze_samples(samples);
    assert_eq!(detected.as_deref(), Some(expected));
}

fn assert_detected_key_in(samples: &[f32], expected: &[&str]) {
    let detected = analyze_samples(samples);
    let detected = detected.as_deref().unwrap_or("<none>");
    assert!(
        expected.contains(&detected),
        "Expected one of {:?}, got {}",
        expected,
        detected
    );
}

#[test]
fn test_play_vocal_melisma() {
    let notes = [
        220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 392.0, 329.63,
    ];
    let samples = render_notes_with_vibrato(&notes, 1.0, 5.5, 0.03);
    let expected = expected_key_from_hz(*notes.last().unwrap());
    assert_detected_key(&samples, &expected);
}

#[test]
fn test_play_instrument_arpeggio() {
    let notes = [
        130.81, 164.81, 196.0, 261.63, 329.63, 392.0, 523.25, 392.0, 329.63, 261.63,
    ];
    let mut data = Vec::new();
    for &note in notes.iter() {
        let carrier = triangle_hz(note);
        let segment = render_signal(carrier, 1.0);
        data.extend(segment);
    }
    let expected = expected_key_from_hz(*notes.last().unwrap());
    assert_detected_key(&data, &expected);
}

#[test]
fn test_play_sustained_vibrato_line() {
    let notes = [
        440.0, 392.0, 349.23, 392.0, 440.0, 493.88, 523.25, 493.88, 440.0, 392.0,
    ];
    let samples = render_notes_with_vibrato(&notes, 1.0, 6.2, 0.09);
    let expected = expected_key_from_hz(*notes.last().unwrap());
    assert_detected_key(&samples, &expected);
}

#[test]
fn test_play_wide_glissando_sweep() {
    let mut data = Vec::new();
    let steps = (SAMPLE_RATE * 10) as usize;
    let mut phase = 0.0;
    for i in 0..steps {
        let t = i as f64 / steps as f64;
        let freq = 110.0 + (880.0 - 110.0) * t;
        let phase_delta = TAU * freq / SAMPLE_RATE as f64;
        let sample = (phase / TAU) * 2.0 - 1.0;
        phase = (phase + phase_delta) % TAU;
        data.push(sample as f32);
    }
    let expected = expected_key_from_hz(880.0);
    assert_detected_key(&data, &expected);
}

#[test]
fn test_play_drifted_swell() {
    let mut data = Vec::new();
    let steps = (SAMPLE_RATE * 10) as usize;
    let mut phase = 0.0;
    for i in 0..steps {
        let t = i as f64 / steps as f64;
        let drift = 2.0_f64.powf((6.0 * t * 10.0) / 1200.0);
        let freq = 220.0 * drift;
        let phase_delta = TAU * freq / SAMPLE_RATE as f64;
        let env = if t < 0.4 {
            t / 0.4
        } else if t > 0.8 {
            (1.0 - t) / 0.2
        } else {
            1.0
        };
        phase = (phase + phase_delta) % TAU;
        data.push((phase.sin() * env) as f32);
    }
    let expected = expected_key_from_hz(220.0 * 2.0_f64.powf((6.0 * 10.0) / 1200.0));
    assert_detected_key(&data, &expected);
}

#[test]
fn test_play_jitter_breathy() {
    let mut data = Vec::new();
    let mut noise = signal::noise(0);
    let steps = (SAMPLE_RATE * 10) as usize;
    let mut phase = 0.0;
    for _ in 0..steps {
        let jitter = 2.0_f64.powf((12.0 * noise.next()) / 1200.0);
        let freq = 196.0 * jitter;
        phase = (phase + TAU * freq / SAMPLE_RATE as f64) % TAU;
        data.push((phase.sin() * 0.8) as f32);
    }
    assert_detected_key_in(&data, &["G3", "A3", "F#3"]);
}

#[test]
fn test_play_harmonic_series_tilt() {
    let mut data = Vec::new();
    let steps = (SAMPLE_RATE * 10) as usize;
    let base_freq = 110.0;
    let mut harmonics: Vec<signal::Sine<signal::ConstHz>> = (1..=8)
        .map(|h| {
            signal::rate(SAMPLE_RATE as f64)
                .const_hz(base_freq * h as f64)
                .sine()
        })
        .collect();
    for _ in 0..steps {
        let mut sample = 0.0;
        for (index, osc) in harmonics.iter_mut().enumerate() {
            let harmonic = index as f64 + 1.0;
            let tilt = 10.0_f64.powf((-3.0 / 20.0) * harmonic.log2());
            sample += osc.next() * (1.0 / harmonic) * tilt;
        }
        data.push(sample as f32);
    }
    let expected = expected_key_from_hz(base_freq);
    assert_detected_key(&data, &expected);
}

#[test]
fn test_play_missing_fundamental() {
    let mut data = Vec::new();
    let steps = (SAMPLE_RATE * 10) as usize;
    let base_freq = 110.0;
    let mut harmonics: Vec<signal::Sine<signal::ConstHz>> = (2..=8)
        .map(|h| {
            signal::rate(SAMPLE_RATE as f64)
                .const_hz(base_freq * h as f64)
                .sine()
        })
        .collect();
    for _ in 0..steps {
        let mut sample = 0.0;
        for (index, osc) in harmonics.iter_mut().enumerate() {
            let harmonic = index as f64 + 2.0;
            let tilt = 10.0_f64.powf((-1.5 / 20.0) * harmonic.log2());
            sample += osc.next() * (1.0 / harmonic) * tilt;
        }
        data.push(sample as f32);
    }
    let expected = expected_key_from_hz(base_freq);
    assert_detected_key(&data, &expected);
}

#[test]
fn test_play_octave_doubling() {
    let mut data = Vec::new();
    let steps = (SAMPLE_RATE * 10) as usize;
    let base_freq = 220.0;
    let octave_freq = 440.0;
    let mut base = signal::rate(SAMPLE_RATE as f64)
        .const_hz(base_freq)
        .sine();
    let mut octave = signal::rate(SAMPLE_RATE as f64)
        .const_hz(octave_freq)
        .sine();
    for _ in 0..steps {
        data.push((base.next() + octave.next() * 0.6) as f32);
    }
    let expected = expected_key_from_hz(base_freq);
    assert_detected_key(&data, &expected);
}

#[test]
fn test_play_polyphonic_cluster() {
    let mut data = Vec::new();
    let steps = (SAMPLE_RATE * 10) as usize;
    let freqs = [261.63, 277.18, 293.66, 311.13];
    let mut tones = vec![
        signal::rate(SAMPLE_RATE as f64).const_hz(freqs[0]).sine(),
        signal::rate(SAMPLE_RATE as f64).const_hz(freqs[1]).sine(),
        signal::rate(SAMPLE_RATE as f64).const_hz(freqs[2]).sine(),
        signal::rate(SAMPLE_RATE as f64).const_hz(freqs[3]).sine(),
    ];
    for _ in 0..steps {
        let mut sample = 0.0;
        for osc in tones.iter_mut() {
            sample += osc.next() * 0.25;
        }
        data.push(sample as f32);
    }
    assert_detected_key_in(&data, &["C4", "C#4", "D4", "D#4"]);
}

#[test]
fn test_play_chromatic_passing_tones() {
    let mut data = Vec::new();
    let notes = [
        261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.0, 415.3, 440.0,
    ];
    for &note in notes.iter() {
        let carrier = triangle_hz(note);
        let segment = render_signal(carrier, 1.0);
        data.extend(segment);
    }
    let expected = expected_key_from_hz(*notes.last().unwrap());
    assert_detected_key(&data, &expected);
}

#[test]
fn test_play_parallel_key_modulation() {
    let mut data = Vec::new();
    for &note in [261.63, 329.63, 392.0, 523.25, 392.0].iter() {
        let segment = render_signal(
            signal::rate(SAMPLE_RATE as f64).const_hz(note).sine(),
            1.0,
        );
        data.extend(segment);
    }
    for &note in [261.63, 311.13, 392.0, 493.88, 392.0].iter() {
        let segment = render_signal(
            signal::rate(SAMPLE_RATE as f64).const_hz(note).sine(),
            1.0,
        );
        data.extend(segment);
    }
    let expected = expected_key_from_hz(392.0);
    assert_detected_key(&data, &expected);
}

#[test]
fn test_play_relative_key_modulation() {
    let mut data = Vec::new();
    for &note in [261.63, 329.63, 392.0, 523.25, 392.0].iter() {
        let segment = render_signal(
            signal::rate(SAMPLE_RATE as f64).const_hz(note).sine(),
            1.0,
        );
        data.extend(segment);
    }
    for &note in [220.0, 261.63, 329.63, 440.0, 329.63].iter() {
        let segment = render_signal(
            signal::rate(SAMPLE_RATE as f64).const_hz(note).sine(),
            1.0,
        );
        data.extend(segment);
    }
    let expected = expected_key_from_hz(329.63);
    assert_detected_key(&data, &expected);
}

#[test]
fn test_play_percussive_sequence() {
    let mut data = Vec::new();
    for &note in [
        196.0, 220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 392.0,
    ]
    .iter()
    {
        let mut carrier = signal::rate(SAMPLE_RATE as f64).const_hz(note).square();
        let samples = (SAMPLE_RATE as f32) as usize;
        for i in 0..samples {
            let transient = if i < (SAMPLE_RATE as f32 * 0.03) as usize {
                0.6 * (1.0 - i as f32 / (SAMPLE_RATE as f32 * 0.03))
            } else {
                0.0
            };
            data.push((carrier.next() as f32 + transient).clamp(-1.0, 1.0));
        }
    }
    let expected = expected_key_from_hz(392.0);
    assert_detected_key(&data, &expected);
}
