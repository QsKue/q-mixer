mod pitch_detection_fixtures;

use audio::io::analyzers::Analyzer;
use audio::io::analyzers::pitch_detector::McleodPitchDetector;
use pitch_detection_fixtures::{SAMPLE_RATE, render_linear_glissando, render_note_with_vibrato};
use plotters::prelude::*;
use std::f64::consts::TAU;
use std::fs;
use std::path::Path;
use std::time::Duration;

#[derive(Clone, Copy, Debug)]
struct FramePitch {
    time_s: f32,
    midi_note: Option<i32>,
}

#[derive(Debug)]
struct RegressionMetrics {
    cent_errors: Vec<f64>,
    voiced_unvoiced_f1: f64,
    note_onset_abs_error_s: f64,
    octave_error_rate: f64,
}

fn midi_to_hz(midi: i32) -> f64 {
    440.0 * 2.0_f64.powf((midi as f64 - 69.0) / 12.0)
}

fn hz_to_midi_float(hz: f64) -> f64 {
    69.0 + 12.0 * (hz / 440.0).log2()
}

fn cents_between_hz(reference_hz: f64, estimate_hz: f64) -> f64 {
    1200.0 * (estimate_hz / reference_hz).log2()
}

fn collect_frame_pitches(samples: &[f32], chunk_size: usize) -> Vec<FramePitch> {
    let mut analyzer = McleodPitchDetector::new();
    analyzer.set_min_interval(Duration::from_millis(0));

    let mut frames = Vec::new();
    let mut processed = 0usize;
    let mut analysis_events = Vec::new();
    for chunk in samples.chunks(chunk_size) {
        analyzer.analyze(chunk, SAMPLE_RATE, 1, &mut analysis_events);
        analysis_events.clear();
        processed += chunk.len();
        frames.push(FramePitch {
            time_s: processed as f32 / SAMPLE_RATE as f32,
            midi_note: analyzer.detected_note(),
        });
    }
    frames
}

fn first_voiced_time(frames: &[FramePitch]) -> Option<f64> {
    frames
        .iter()
        .find(|f| f.midi_note.is_some())
        .map(|f| f.time_s as f64)
}

fn evaluate_against_ground_truth<F>(
    frames: &[FramePitch],
    expected_hz_at: F,
    expected_onset_s: f64,
) -> RegressionMetrics
where
    F: Fn(f64) -> Option<f64>,
{
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut cent_errors = Vec::new();
    let mut octave_errors = 0usize;
    let mut voiced_pairs = 0usize;

    for frame in frames {
        let t = frame.time_s as f64;
        let expected_hz = expected_hz_at(t);
        let expected_voiced = expected_hz.is_some();
        let detected_voiced = frame.midi_note.is_some();

        match (expected_voiced, detected_voiced) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
            (false, false) => {}
        }

        if let (Some(reference_hz), Some(note)) = (expected_hz, frame.midi_note) {
            voiced_pairs += 1;
            let estimated_hz = midi_to_hz(note);
            cent_errors.push(cents_between_hz(reference_hz, estimated_hz).abs());

            let expected_midi = hz_to_midi_float(reference_hz).round() as i32;
            let diff = (note - expected_midi).abs();
            if (diff - 12).abs() <= 1 || (diff - 24).abs() <= 1 {
                octave_errors += 1;
            }
        }
    }

    cent_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let precision = if tp + fp == 0 {
        1.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let recall = if tp + fn_ == 0 {
        1.0
    } else {
        tp as f64 / (tp + fn_) as f64
    };
    let f1 = if (precision + recall) == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    let detected_onset_s = first_voiced_time(frames).unwrap_or(f64::INFINITY);
    let note_onset_abs_error_s = (detected_onset_s - expected_onset_s).abs();

    let octave_error_rate = if voiced_pairs == 0 {
        0.0
    } else {
        octave_errors as f64 / voiced_pairs as f64
    };

    RegressionMetrics {
        cent_errors,
        voiced_unvoiced_f1: f1,
        note_onset_abs_error_s,
        octave_error_rate,
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn maybe_plot_frames<F>(path: &str, title: &str, frames: &[FramePitch], expected_hz_at: F)
where
    F: Fn(f64) -> Option<f64>,
{
    if std::env::var_os("Q_MIXER_PITCH_PLOTS").is_none() {
        return;
    }

    let out_path = Path::new("target").join("plots").join(path);
    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let x_end = frames.last().map(|f| f.time_s).unwrap_or(1.0).max(1.0);
    let root = BitMapBackend::new(&out_path, (1280, 720)).into_drawing_area();
    let _ = root.fill(&WHITE);
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f32..x_end, 40f32..85f32)
        .expect("plot chart");

    let _ = chart
        .configure_mesh()
        .x_desc("Time (s)")
        .y_desc("MIDI note")
        .draw();

    let expected_series = frames.iter().filter_map(|f| {
        expected_hz_at(f.time_s as f64)
            .map(hz_to_midi_float)
            .map(|m| (f.time_s, m as f32))
    });
    let _ = chart.draw_series(LineSeries::new(expected_series, &BLUE));

    let detected_series = frames
        .iter()
        .filter_map(|f| f.midi_note.map(|n| (f.time_s, n as f32)));
    let _ = chart.draw_series(LineSeries::new(detected_series, &RED));
    let _ = root.present();
}

#[test]
fn pitch_regression_short_sine_onset_and_voicing() {
    let onset_s = 0.08;
    let sustain_s = 0.55;
    let tail_s = 0.08;
    let mut samples = vec![0.0; (onset_s * SAMPLE_RATE as f64) as usize];
    samples.extend(render_note_with_vibrato(220.0, sustain_s as f32, 0.0, 0.0));
    samples.extend(vec![0.0; (tail_s * SAMPLE_RATE as f64) as usize]);

    let frames = collect_frame_pitches(&samples, 512);
    let expected = |t: f64| {
        if (onset_s..(onset_s + sustain_s)).contains(&t) {
            Some(220.0)
        } else {
            None
        }
    };
    let metrics = evaluate_against_ground_truth(&frames, expected, onset_s);
    maybe_plot_frames(
        "regression_short_sine.png",
        "Regression short sine",
        &frames,
        expected,
    );

    let p95 = percentile(&metrics.cent_errors, 0.95);
    assert!(
        p95 <= 55.0,
        "p95 cent error too high: {p95:.2} cents; metrics={metrics:?}"
    );
    assert!(
        metrics.voiced_unvoiced_f1 >= 0.88,
        "voiced/unvoiced F1 too low: {:.3}; metrics={metrics:?}",
        metrics.voiced_unvoiced_f1
    );
    assert!(
        metrics.note_onset_abs_error_s <= 0.08,
        "onset timing error too high: {:.4}s; metrics={metrics:?}",
        metrics.note_onset_abs_error_s
    );
    assert!(
        metrics.octave_error_rate <= 0.02,
        "octave error rate too high: {:.3}; metrics={metrics:?}",
        metrics.octave_error_rate
    );
}

#[test]
fn pitch_regression_vibrato_and_glissando_frame_accuracy() {
    let vibrato_secs = 0.9;
    let mut samples = render_note_with_vibrato(246.94, vibrato_secs as f32, 5.0, 0.02);
    let gliss_secs = 0.9;
    samples.extend(render_linear_glissando(196.0, 392.0, gliss_secs as f32));

    let frames = collect_frame_pitches(&samples, 512);
    let expected = |t: f64| {
        if t < vibrato_secs {
            Some(246.94 * (1.0 + (TAU * 5.0 * t).sin() * 0.02))
        } else if t < vibrato_secs + gliss_secs {
            let u = (t - vibrato_secs) / gliss_secs;
            Some(196.0 + (392.0 - 196.0) * u)
        } else {
            None
        }
    };
    let metrics = evaluate_against_ground_truth(&frames, expected, 0.0);
    maybe_plot_frames(
        "regression_vibrato_glissando.png",
        "Regression vibrato + glissando",
        &frames,
        expected,
    );

    let p90 = percentile(&metrics.cent_errors, 0.90);
    assert!(
        p90 <= 95.0,
        "p90 cent error too high: {p90:.2} cents; metrics={metrics:?}"
    );
    assert!(
        metrics.voiced_unvoiced_f1 >= 0.90,
        "voiced/unvoiced F1 too low: {:.3}; metrics={metrics:?}",
        metrics.voiced_unvoiced_f1
    );
    assert!(
        metrics.octave_error_rate <= 0.08,
        "octave error rate too high: {:.3}; metrics={metrics:?}",
        metrics.octave_error_rate
    );
}
