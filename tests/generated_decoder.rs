use std::{sync::mpsc, thread, time::Duration};

use audio::io::decoders_gen::GeneratedWaveformPattern;
use audio::mixer::{ChannelSource, Mixer};
use dasp_signal::{self as signal, Signal};
use image::{Rgb, RgbImage};
use std::f64::consts::TAU;

const SAMPLE_RATE: u32 = 44_100;
const CHANNELS: usize = 1;

fn play_pattern(label: &str, pattern: GeneratedWaveformPattern) {
    let (tx_event, _rx_event) = mpsc::channel();
    let mixer = Mixer::new(None, tx_event);

    mixer.setup();
    mixer
        .load_channel(
            0,
            ChannelSource::GeneratedAudio {
                sample_rate: SAMPLE_RATE,
                channels: CHANNELS,
                pattern,
            },
        )
        .unwrap();
    mixer.play_channel(0);

    println!("Playing {label} for 10 seconds...");
    thread::sleep(Duration::from_secs(10));
}

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
    color: Rgb<u8>,
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
    let mut img = RgbImage::from_pixel(width as u32, height as u32, Rgb([255, 255, 255]));
    let total = samples_len.max(1);

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

    for line in series {
        let mut prev = None;
        for x in 0..width {
            let idx = x * total / width;
            let freq = line.freqs.get(idx).copied().unwrap_or(0.0);
            let y = if freq <= 0.0 {
                height as i32 - 1
            } else {
                let norm = ((freq - min_freq) / range).clamp(0.0, 1.0);
                ((height as f64 - 1.0) * (1.0 - norm)).round() as i32
            };
            if let Some((px, py)) = prev {
                draw_line_colored(&mut img, px, py, x as i32, y, line.color);
            }
            prev = Some((x as i32, y));
        }
    }

    let filename = format!("target/note_graphs/{}.png", sanitize_label(label));
    let _ = std::fs::create_dir_all("target/note_graphs");
    let _ = img.save(&filename);
}

fn draw_line_colored(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgb<u8>) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        if x >= 0 && y >= 0 && x < img.width() as i32 && y < img.height() as i32 {
            img.put_pixel(x as u32, y as u32, color);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
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
            color: Rgb([30, 30, 30]),
            freqs,
        }],
    }
}

fn push_constant_series(freqs: &mut Vec<f64>, freq: f64, samples: usize) {
    freqs.extend(std::iter::repeat(freq).take(samples));
}

fn play_pattern_with_graph(label: &str, graph: PatternGraph) {
    save_note_graph_png(label, &graph.series, graph.samples.len());
    let pattern = GeneratedWaveformPattern::Samples {
        samples: graph.samples,
    };
    play_pattern(label, pattern);
}

#[test]
#[ignore = "manual listening test"]
fn test_play_vocal_melisma() {
    let graph = render_notes_with_vibrato(
        &[
            220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 392.0, 329.63,
        ],
        1.0,
        5.5,
        0.03,
    );
    play_pattern_with_graph("vocal melisma (gliss + vibrato)", graph);
}

#[test]
#[ignore = "manual listening test"]
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
            color: Rgb([30, 30, 30]),
            freqs,
        }],
    };
    play_pattern_with_graph("instrument arpeggio (tremolo)", graph);
}

#[test]
#[ignore = "manual listening test"]
fn test_play_sustained_vibrato_line() {
    let graph = render_notes_with_vibrato(
        &[
            440.0, 392.0, 349.23, 392.0, 440.0, 493.88, 523.25, 493.88, 440.0, 392.0,
        ],
        1.0,
        6.2,
        0.09,
    );
    play_pattern_with_graph("sustained wide vibrato line", graph);
}

#[test]
#[ignore = "manual listening test"]
fn test_play_wide_glissando_sweep() {
    let mut data = Vec::new();
    let mut freqs = Vec::new();
    let steps = (SAMPLE_RATE * 10) as usize;
    for i in 0..steps {
        let t = i as f64 / steps as f64;
        let freq = 110.0 + (880.0 - 110.0) * t;
        let mut osc = signal::rate(SAMPLE_RATE as f64).const_hz(freq).saw();
        data.push(osc.next() as f32);
        freqs.push(freq);
    }
    let graph = PatternGraph {
        samples: data,
        series: vec![NoteSeries {
            label: "main".to_string(),
            color: Rgb([30, 30, 30]),
            freqs,
        }],
    };
    play_pattern_with_graph("wide glissando sweep", graph);
}

#[test]
#[ignore = "manual listening test"]
fn test_play_drifted_swell() {
    let mut data = Vec::new();
    let mut freqs = Vec::new();
    let steps = (SAMPLE_RATE * 10) as usize;
    for i in 0..steps {
        let t = i as f64 / steps as f64;
        let drift = 2.0_f64.powf((6.0 * t * 10.0) / 1200.0);
        let freq = 220.0 * drift;
        let mut osc = signal::rate(SAMPLE_RATE as f64)
            .const_hz(freq)
            .sine();
        let env = if t < 0.4 {
            t / 0.4
        } else if t > 0.8 {
            (1.0 - t) / 0.2
        } else {
            1.0
        };
        data.push((osc.next() * env) as f32);
        freqs.push(freq);
    }
    let graph = PatternGraph {
        samples: data,
        series: vec![NoteSeries {
            label: "main".to_string(),
            color: Rgb([30, 30, 30]),
            freqs,
        }],
    };
    play_pattern_with_graph("drifted swell (exp drift + long attack)", graph);
}

#[test]
#[ignore = "manual listening test"]
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
            color: Rgb([30, 30, 30]),
            freqs,
        }],
    };
    play_pattern_with_graph("breathy jitter (inharmonic)", graph);
}

#[test]
#[ignore = "manual listening test"]
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
        Rgb([30, 30, 30]),
        Rgb([90, 90, 90]),
        Rgb([130, 130, 130]),
        Rgb([170, 170, 170]),
        Rgb([200, 200, 200]),
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
    play_pattern_with_graph("harmonic series (spectral tilt)", graph);
}

#[test]
#[ignore = "manual listening test"]
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
        Rgb([90, 90, 90]),
        Rgb([130, 130, 130]),
        Rgb([170, 170, 170]),
        Rgb([200, 200, 200]),
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
    play_pattern_with_graph("missing fundamental", graph);
}

#[test]
#[ignore = "manual listening test"]
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
                color: Rgb([30, 30, 30]),
                freqs: base_freqs,
            },
            NoteSeries {
                label: "F1".to_string(),
                color: Rgb([140, 140, 140]),
                freqs: octave_freqs,
            },
        ],
    };
    play_pattern_with_graph("octave doubling", graph);
}

#[test]
#[ignore = "manual listening test"]
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
        series.push(NoteSeries {
            label: format!("N{}", index + 1),
            color: Rgb([30 + (index as u8 * 40), 30, 30]),
            freqs: line,
        });
    }
    let graph = PatternGraph { samples: data, series };
    play_pattern_with_graph("polyphonic cluster", graph);
}

#[test]
#[ignore = "manual listening test"]
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
            color: Rgb([30, 30, 30]),
            freqs,
        }],
    };
    play_pattern_with_graph("chromatic passing tones", graph);
}

#[test]
#[ignore = "manual listening test"]
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
            color: Rgb([30, 30, 30]),
            freqs,
        }],
    };
    play_pattern_with_graph("parallel key modulation (C major to C minor)", graph);
}

#[test]
#[ignore = "manual listening test"]
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
            color: Rgb([30, 30, 30]),
            freqs,
        }],
    };
    play_pattern_with_graph("relative key modulation (C major to A minor)", graph);
}

#[test]
#[ignore = "manual listening test"]
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
            color: Rgb([30, 30, 30]),
            freqs,
        }],
    };
    play_pattern_with_graph("percussive transients", graph);
}
