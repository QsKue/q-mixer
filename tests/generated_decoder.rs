use dasp_signal::{self as signal, Signal};
use plotters::prelude::*;
use std::f64::consts::TAU;

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

struct NoteSeries {
    label: String,
    color: RGBColor,
    freqs: Vec<f64>,
}

struct PatternGraph {
    samples: Vec<f32>,
    series: Vec<NoteSeries>,
}

fn triangle_hz(hz: f64) -> impl Signal<Frame = f64> {
    let phase = signal::phase(signal::rate(SAMPLE_RATE as f64).const_hz(hz));
    phase.map(|p| 1.0 - 4.0 * (p - 0.5).abs())
}

fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

fn save_note_graph_png(label: &str, series: &[NoteSeries], samples_len: usize) {
    if series.is_empty() || samples_len == 0 {
        return;
    }
    let width = 900usize;
    let height = 360usize;
    let filename = format!("target/note_graphs/{}.png", sanitize_label(label));
    let _ = std::fs::create_dir_all("target/note_graphs");
    let root = BitMapBackend::new(&filename, (width as u32, height as u32))
        .into_drawing_area();
    let _ = root.fill(&WHITE);

    let mut min_freq = f64::MAX;
    let mut max_freq = f64::MIN;
    for line in series {
        for &freq in &line.freqs {
            if freq > 0.0 {
                min_freq = min_freq.min(freq);
                max_freq = max_freq.max(freq);
            }
        }
    }
    if !min_freq.is_finite() || !max_freq.is_finite() || min_freq == max_freq {
        min_freq = 0.0;
        max_freq = 1.0;
    }
    let padding = (max_freq - min_freq) * 0.1;
    let min_freq = (min_freq - padding).max(0.0);
    let max_freq = max_freq + padding;
    let range = (max_freq - min_freq).max(1.0);

    let x_max = samples_len.saturating_sub(1).max(1) as i32;
    let mut chart = ChartBuilder::on(&root)
        .margin(12)
        .x_label_area_size(32)
        .y_label_area_size(48)
        .build_cartesian_2d(0..x_max, min_freq..(min_freq + range))
        .unwrap();
    let _ = chart
        .configure_mesh()
        .x_desc("Sample")
        .y_desc("Frequency (Hz)")
        .draw();

    let stride = (samples_len / width).max(1);
    for line in series {
        let points = line
            .freqs
            .iter()
            .enumerate()
            .step_by(stride)
            .filter(|(_, freq)| freq.is_finite() && **freq > 0.0)
            .map(|(idx, freq)| (idx as i32, *freq));
        let _ = chart.draw_series(LineSeries::new(points, &line.color));
    }

    let _ = root.present();
}

fn save_waveform_png(label: &str, samples: &[f32]) {
    if samples.is_empty() {
        return;
    }
    let width = 900usize;
    let height = 360usize;
    let filename = format!("target/waveform_graphs/{}.png", sanitize_label(label));
    let _ = std::fs::create_dir_all("target/waveform_graphs");
    let root = BitMapBackend::new(&filename, (width as u32, height as u32))
        .into_drawing_area();
    let _ = root.fill(&WHITE);

    let x_max = samples.len().saturating_sub(1).max(1) as i32;
    let mut chart = ChartBuilder::on(&root)
        .margin(12)
        .x_label_area_size(32)
        .y_label_area_size(48)
        .build_cartesian_2d(0..x_max, -1.0f32..1.0f32)
        .unwrap();
    let _ = chart
        .configure_mesh()
        .x_desc("Sample")
        .y_desc("Amplitude")
        .draw();

    let stride = (samples.len() / width).max(1);
    let points = samples
        .iter()
        .enumerate()
        .step_by(stride)
        .map(|(idx, sample)| (idx as i32, sample.clamp(-1.0, 1.0)));
    let _ = chart.draw_series(LineSeries::new(points, &RGBColor(30, 90, 200)));

    let _ = root.present();
}

fn render_note_with_vibrato(
    base_hz: f64,
    note_secs: f32,
    vibrato_hz: f64,
    depth: f64,
) -> (Vec<f32>, Vec<f64>) {
    let samples = (SAMPLE_RATE as f32 * note_secs) as usize;
    let mut data = Vec::with_capacity(samples);
    let mut freqs = Vec::with_capacity(samples);
    let mut lfo = signal::rate(SAMPLE_RATE as f64).const_hz(vibrato_hz).sine();
    let mut phase = 0.0;
    for _ in 0..samples {
        let hz = base_hz * (1.0 + lfo.next() * depth);
        phase = (phase + TAU * hz / SAMPLE_RATE as f64) % TAU;
        data.push(phase.sin() as f32);
        freqs.push(hz);
    }
    (data, freqs)
}

fn render_notes_with_vibrato(
    notes: &[f64],
    note_secs: f32,
    vibrato_hz: f64,
    depth: f64,
) -> PatternGraph {
    let mut samples = Vec::new();
    let mut freqs = Vec::new();
    for &note in notes {
        let (segment_samples, segment_freqs) =
            render_note_with_vibrato(note, note_secs, vibrato_hz, depth);
        samples.extend(segment_samples);
        freqs.extend(segment_freqs);
    }
    PatternGraph {
        samples,
        series: vec![NoteSeries {
            label: "main".to_string(),
            color: RGBColor(30, 90, 200),
            freqs,
        }],
    }
}

fn push_constant_series(freqs: &mut Vec<f64>, freq: f64, samples: usize) {
    freqs.extend(std::iter::repeat(freq).take(samples));
}

fn run_test(label: &str, graph: PatternGraph) {
    save_note_graph_png(label, &graph.series, graph.samples.len());
    save_waveform_png(label, &graph.samples);
    // play(graph.samples)    
}

fn play(samples: Vec<f32>) {
    use std::sync::mpsc;
    use audio::mixer::{ChannelSource, Mixer};

    let (tx_event, _rx_event) = mpsc::channel();
    let mixer = Mixer::new(None, tx_event);

    mixer.setup();
    mixer
        .load_channel(
            0,
            ChannelSource::GeneratedAudio {
                sample_rate: SAMPLE_RATE,
                channels: 1,
                samples,
            },
        )
        .unwrap();
    mixer.play_channel(0);

    println!("Running test, press enter to exit");
    let stdin = std::io::stdin();
    for _line in std::io::BufRead::lines(stdin.lock()) {
        break;
    }
}

#[test]
#[ignore]
fn test_play_vocal_melisma() {
    let graph = render_notes_with_vibrato(
        &[
            220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 392.0, 329.63,
        ],
        1.0,
        5.5,
        0.03,
    );
    let label = "vocal melisma (gliss + vibrato)";
    run_test(label, graph);
}

#[test]
#[ignore]
fn test_play_instrument_arpeggio() {
    let notes = [
        130.81, 164.81, 196.0, 261.63, 329.63, 392.0, 523.25, 392.0, 329.63, 261.63,
    ];
    let mut data = Vec::new();
    let mut freqs = Vec::new();
    for &note in notes.iter() {
        let carrier = triangle_hz(note);
        let segment = render_signal(carrier, 1.0);
        push_constant_series(&mut freqs, note, segment.len());
        data.extend(segment);
    }
    let graph = PatternGraph {
        samples: data,
        series: vec![NoteSeries {
            label: "main".to_string(),
            color: RGBColor(30, 90, 200),
            freqs,
        }],
    };
    let label = "instrument arpeggio (tremolo)";
    run_test(label, graph);
}

#[test]
#[ignore]
fn test_play_sustained_vibrato_line() {
    let graph = render_notes_with_vibrato(
        &[
            440.0, 392.0, 349.23, 392.0, 440.0, 493.88, 523.25, 493.88, 440.0, 392.0,
        ],
        1.0,
        6.2,
        0.09,
    );
    let label = "sustained wide vibrato line";
    run_test(label, graph);
}

#[test]
#[ignore]
fn test_play_wide_glissando_sweep() {
    let mut data = Vec::new();
    let mut freqs = Vec::new();
    let steps = (SAMPLE_RATE * 10) as usize;
    let mut phase = 0.0;
    for i in 0..steps {
        let t = i as f64 / steps as f64;
        let freq = 110.0 + (880.0 - 110.0) * t;
        let phase_delta = TAU * freq / SAMPLE_RATE as f64;
        let sample = (phase / TAU) * 2.0 - 1.0;
        phase = (phase + phase_delta) % TAU;
        data.push(sample as f32);
        freqs.push(freq);
    }
    let graph = PatternGraph {
        samples: data,
        series: vec![NoteSeries {
            label: "main".to_string(),
            color: RGBColor(30, 90, 200),
            freqs,
        }],
    };
    let label = "wide glissando sweep";
    run_test(label, graph);
}

#[test]
#[ignore]
fn test_play_drifted_swell() {
    let mut data = Vec::new();
    let mut freqs = Vec::new();
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
        freqs.push(freq);
    }
    let graph = PatternGraph {
        samples: data,
        series: vec![NoteSeries {
            label: "main".to_string(),
            color: RGBColor(30, 90, 200),
            freqs,
        }],
    };
    let label = "drifted swell (exp drift + long attack)";
    run_test(label, graph);
}

#[test]
#[ignore]
fn test_play_jitter_breathy() {
    let mut data = Vec::new();
    let mut freqs = Vec::new();
    let mut noise = signal::noise(0);
    let steps = (SAMPLE_RATE * 10) as usize;
    let mut phase = 0.0;
    for _ in 0..steps {
        let jitter = 2.0_f64.powf((12.0 * noise.next()) / 1200.0);
        let freq = 196.0 * jitter;
        phase = (phase + TAU * freq / SAMPLE_RATE as f64) % TAU;
        data.push((phase.sin() * 0.8) as f32);
        freqs.push(freq);
    }
    let graph = PatternGraph {
        samples: data,
        series: vec![NoteSeries {
            label: "main".to_string(),
            color: RGBColor(30, 90, 200),
            freqs,
        }],
    };
    let label = "breathy jitter (inharmonic)";
    run_test(label, graph);
}

#[test]
#[ignore]
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
    let mut series = Vec::new();
    for (index, color) in [
        RGBColor(30, 90, 200),
        RGBColor(50, 110, 220),
        RGBColor(70, 130, 235),
        RGBColor(90, 150, 245),
        RGBColor(110, 170, 255),
    ]
    .iter()
    .enumerate()
    {
        let harmonic = (index + 1) as f64;
        let mut freqs = Vec::with_capacity(steps);
        push_constant_series(&mut freqs, base_freq * harmonic, steps);
        series.push(NoteSeries {
            label: format!("H{}", index + 1),
            color: *color,
            freqs,
        });
    }
    let graph = PatternGraph { samples: data, series };
    let label = "harmonic series (spectral tilt)";
    run_test(label, graph);
}

#[test]
#[ignore]
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
    let mut series = Vec::new();
    for (index, color) in [
        RGBColor(40, 100, 210),
        RGBColor(60, 120, 225),
        RGBColor(80, 140, 240),
        RGBColor(100, 160, 255),
    ]
    .iter()
    .enumerate()
    {
        let harmonic = (index + 2) as f64;
        let mut freqs = Vec::with_capacity(steps);
        push_constant_series(&mut freqs, base_freq * harmonic, steps);
        series.push(NoteSeries {
            label: format!("H{}", index + 2),
            color: *color,
            freqs,
        });
    }
    let graph = PatternGraph { samples: data, series };
    let label = "missing fundamental";
    run_test(label, graph);
}

#[test]
#[ignore]
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
    let mut base_freqs = Vec::with_capacity(steps);
    let mut octave_freqs = Vec::with_capacity(steps);
    push_constant_series(&mut base_freqs, base_freq, steps);
    push_constant_series(&mut octave_freqs, octave_freq, steps);
    let graph = PatternGraph {
        samples: data,
        series: vec![
            NoteSeries {
                label: "F0".to_string(),
                color: RGBColor(30, 90, 200),
                freqs: base_freqs,
            },
            NoteSeries {
                label: "F1".to_string(),
                color: RGBColor(80, 140, 240),
                freqs: octave_freqs,
            },
        ],
    };
    let label = "octave doubling";
    run_test(label, graph);
}

#[test]
#[ignore]
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
    let mut series = Vec::new();
    for (index, freq) in freqs.iter().enumerate() {
        let mut line = Vec::with_capacity(steps);
        push_constant_series(&mut line, *freq, steps);
        let color_shift = (index as u8) * 35;
        series.push(NoteSeries {
            label: format!("N{}", index + 1),
            color: RGBColor(30, 90 + color_shift, 200),
            freqs: line,
        });
    }
    let graph = PatternGraph { samples: data, series };
    let label = "polyphonic cluster";
    run_test(label, graph);
}

#[test]
#[ignore]
fn test_play_chromatic_passing_tones() {
    let mut data = Vec::new();
    let mut freqs = Vec::new();
    for &note in [
        261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.0, 415.3, 440.0,
    ]
    .iter()
    {
        let carrier = triangle_hz(note);
        let segment = render_signal(carrier, 1.0);
        push_constant_series(&mut freqs, note, segment.len());
        data.extend(segment);
    }
    let graph = PatternGraph {
        samples: data,
        series: vec![NoteSeries {
            label: "main".to_string(),
            color: RGBColor(30, 90, 200),
            freqs,
        }],
    };
    let label = "chromatic passing tones";
    run_test(label, graph);
}

#[test]
#[ignore]
fn test_play_parallel_key_modulation() {
    let mut data = Vec::new();
    let mut freqs = Vec::new();
    for &note in [261.63, 329.63, 392.0, 523.25, 392.0].iter() {
        let segment = render_signal(
            signal::rate(SAMPLE_RATE as f64).const_hz(note).sine(),
            1.0,
        );
        push_constant_series(&mut freqs, note, segment.len());
        data.extend(segment);
    }
    for &note in [261.63, 311.13, 392.0, 493.88, 392.0].iter() {
        let segment = render_signal(
            signal::rate(SAMPLE_RATE as f64).const_hz(note).sine(),
            1.0,
        );
        push_constant_series(&mut freqs, note, segment.len());
        data.extend(segment);
    }
    let graph = PatternGraph {
        samples: data,
        series: vec![NoteSeries {
            label: "main".to_string(),
            color: RGBColor(30, 90, 200),
            freqs,
        }],
    };
    let label = "parallel key modulation (C major to C minor)";
    run_test(label, graph);
}

#[test]
#[ignore]
fn test_play_relative_key_modulation() {
    let mut data = Vec::new();
    let mut freqs = Vec::new();
    for &note in [261.63, 329.63, 392.0, 523.25, 392.0].iter() {
        let segment = render_signal(
            signal::rate(SAMPLE_RATE as f64).const_hz(note).sine(),
            1.0,
        );
        push_constant_series(&mut freqs, note, segment.len());
        data.extend(segment);
    }
    for &note in [220.0, 261.63, 329.63, 440.0, 329.63].iter() {
        let segment = render_signal(
            signal::rate(SAMPLE_RATE as f64).const_hz(note).sine(),
            1.0,
        );
        push_constant_series(&mut freqs, note, segment.len());
        data.extend(segment);
    }
    let graph = PatternGraph {
        samples: data,
        series: vec![NoteSeries {
            label: "main".to_string(),
            color: RGBColor(30, 90, 200),
            freqs,
        }],
    };
    let label = "relative key modulation (C major to A minor)";
    run_test(label, graph);
}

#[test]
#[ignore]
fn test_play_percussive_sequence() {
    let mut data = Vec::new();
    let mut freqs = Vec::new();
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
        push_constant_series(&mut freqs, note, samples);
    }
    let graph = PatternGraph {
        samples: data,
        series: vec![NoteSeries {
            label: "main".to_string(),
            color: RGBColor(30, 90, 200),
            freqs,
        }],
    };
    let label = "percussive transients";
    run_test(label, graph);
}
