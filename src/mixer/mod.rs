mod channel;
mod runner;
mod types;

pub(crate) use channel::Channel;
pub(crate) use types::MixerTask;
pub use types::{ChannelSource, MixerEvent};

use std::sync::{Arc, RwLock, mpsc};

use crate::io::{
    decoders::Decoder, decoders::GeneratedDecoder, decoders::SymphoniaDecoder,
    resamplers::RubatoResampler, sources::FileSource, time_stretchers::NoopTimeStretcher,
};

pub struct MixerSettings {
    buffer_size_ms: u64,
    buffer_refill_interval_ms: u64,
}

impl Default for MixerSettings {
    fn default() -> Self {
        Self {
            buffer_size_ms: 500,
            buffer_refill_interval_ms: 10,
        }
    }
}

pub struct Mixer {
    tx_task: mpsc::Sender<MixerTask>,
    tx_event: mpsc::Sender<MixerEvent>,

    settings: Arc<RwLock<MixerSettings>>,
}

impl Mixer {
    pub fn new(settings: Option<MixerSettings>, tx_event: mpsc::Sender<MixerEvent>) -> Self {
        let settings = Arc::new(RwLock::new(settings.unwrap_or_default()));
        let tx_task = runner::Runner::new(settings.clone());
        Self {
            tx_task,
            tx_event,
            settings,
        }
    }

    pub fn setup(&self) {
        // TODO: add output source
        let _ = self.tx_task.send(MixerTask::Initialize);
    }

    // TODO: better error handling
    pub fn load_channel(&self, index: usize, source: ChannelSource) -> Result<(), String> {
        // TODO: add callback maybe (or only if function may fail)

        let decoder: Box<dyn Decoder> = match source {
            ChannelSource::File { path } => {
                let source =
                    FileSource::new(path).map_err(|err| "Error creating source".to_string())?;
                let decoder = SymphoniaDecoder::new(Box::new(source))
                    .map_err(|err| "Error creating decoder".to_string())?;
                Box::new(decoder)
            }
            ChannelSource::GeneratedAudio {
                sample_rate,
                channels,
                pattern,
            } => Box::new(GeneratedDecoder::new(sample_rate, channels, pattern)),
        };

        let resampler = Box::new(RubatoResampler::new());
        let time_stretcher = Box::new(NoopTimeStretcher {});

        let _ = self.tx_task.send(MixerTask::ChannelLoad {
            index,
            decoder,
            resampler,
            time_stretcher,
        });
        Ok(())
    }

    pub fn play_channel(&self, index: usize) {
        let _ = self.tx_task.send(MixerTask::ChannelPlay { index });
    }
}
