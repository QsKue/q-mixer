use crate::io::analyzers::AnalysisEvent;

/// Commands emitted by control policy and applied to stream processors.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlCommand {
    SetTimeStretchPitchSemitones(f32),
    SetSpeed(f32),
    SetDspParam {
        target: DspTarget,
        id: u32,
        value: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DspTarget {
    PreFx,
    PostFx,
}

/// Deterministic event router for chunk-scoped control decisions.
#[derive(Default)]
pub struct Controller;

impl Controller {
    pub fn new() -> Self {
        Self
    }

    /// Convert analyzer events from a single decoded chunk into control commands.
    pub fn route(&mut self, events: &[AnalysisEvent], out_commands: &mut Vec<ControlCommand>) {
        out_commands.clear();

        for event in events {
            match event {
                AnalysisEvent::Pitch {
                    midi, confidence, ..
                } if *confidence >= 0.80 => {
                    // Keep pitch near A4 (69) in semitone-domain to avoid runaway shifts.
                    let semitones = (*midi as f32 - 69.0).clamp(-12.0, 12.0);
                    out_commands.push(ControlCommand::SetTimeStretchPitchSemitones(semitones));
                }
                AnalysisEvent::Beat { bpm, .. } if *bpm > 0.0 => {
                    // Map [60..180] BPM into speed [0.75..1.25].
                    let normalized = ((*bpm - 60.0) / 120.0).clamp(0.0, 1.0);
                    let speed = 0.75 + normalized * 0.5;
                    out_commands.push(ControlCommand::SetSpeed(speed));
                }
                AnalysisEvent::Onset { strength } => {
                    let wet = strength.clamp(0.0, 1.0);
                    out_commands.push(ControlCommand::SetDspParam {
                        target: DspTarget::PreFx,
                        id: 0,
                        value: wet,
                    });
                }
                _ => {}
            }
        }
    }
}
