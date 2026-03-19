pub mod pitch_detector;

/// Control-plane information emitted by analyzers while audio chunks are processed.
///
/// Analyzers observe the stream and produce events; they do **not** mutate the
/// audio buffer directly.
#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisEvent {
    Pitch {
        midi: i32,
        hz: f32,
        confidence: f32,
    },
    Onset {
        strength: f32,
    },
    Beat {
        bpm: f32,
        phase: f32,
    },
}

pub trait Analyzer {
    /// Analyze a chunk of audio and append any produced control events to `out_events`.
    ///
    /// Callers are expected to reuse the same output buffer across calls to avoid
    /// unnecessary allocations.
    fn analyze(
        &mut self,
        input: &[f32],
        sample_rate: u32,
        channels: usize,
        out_events: &mut Vec<AnalysisEvent>,
    );
}
