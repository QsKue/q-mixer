use dasp_signal::{self as signal, Signal};
use std::f64::consts::TAU;

pub const SAMPLE_RATE: u32 = 44_100;

pub fn render_signal<S>(mut signal: S, seconds: f32) -> Vec<f32>
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

pub fn triangle_hz(hz: f64) -> impl Signal<Frame = f64> {
    let phase = signal::phase(signal::rate(SAMPLE_RATE as f64).const_hz(hz));
    phase.map(|p| 1.0 - 4.0 * (p - 0.5).abs())
}

pub fn render_note_with_vibrato(
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

pub fn render_notes_with_vibrato(
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

pub fn render_linear_glissando(start_hz: f64, end_hz: f64, seconds: f32) -> Vec<f32> {
    let mut data = Vec::new();
    let steps = (SAMPLE_RATE as f32 * seconds) as usize;
    let mut phase = 0.0;
    for i in 0..steps {
        let t = i as f64 / steps as f64;
        let freq = start_hz + (end_hz - start_hz) * t;
        let phase_delta = TAU * freq / SAMPLE_RATE as f64;
        phase = (phase + phase_delta) % TAU;
        data.push(phase.sin() as f32);
    }
    data
}
