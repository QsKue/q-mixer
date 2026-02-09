mod output;

use std::{collections::HashMap, sync::{Arc, RwLock, mpsc}, time::Duration};

use super::{Channel, MixerTask, MixerSettings};

pub(super) struct Runner {
    output: output::Output,
    channels: HashMap<usize, Channel>,
}

impl Runner {

    pub(super) fn new(settings: Arc<RwLock<MixerSettings>>) -> mpsc::Sender<MixerTask> {
        let (tx_task, rx_task) = mpsc::channel::<MixerTask>();

        // ring buffer interval tick
        let settings_thread = settings.clone();
        let tx_task_thread = tx_task.clone();
        std::thread::spawn(move || {
            let period = Duration::from_millis(settings_thread.read().unwrap().buffer_refill_interval_ms);
            loop {
                std::thread::sleep(period);
                if tx_task_thread.send(MixerTask::RefillBuffer).is_err() { break; }
            }
        });

        std::thread::spawn(move || {
            Self::run(settings, rx_task);
        });

        tx_task
    }

    fn run(settings: Arc<RwLock<MixerSettings>>, rx_task: mpsc::Receiver<MixerTask>) {
        let mut runner: Option<Runner> = None;

        while let Ok(task) = rx_task.recv() {
            match task {
                MixerTask::Initialize => {
                    
                    if let Some(runner) = runner.as_mut() {
                        runner.output.set_paused(true);
                    }

                    // TODO: fix unwraps
                    runner.take();
                    runner = Some(
                        Runner {
                            output: output::Output::new(settings.read().unwrap().buffer_size_ms).unwrap(),
                            channels: HashMap::new(),
                        }
                    )
                }
                MixerTask::RefillBuffer => {
                    if let Some(runner) = runner.as_mut() {
                        runner.output.refill_buffer(|buffer, buffer_len, sample_rate, channels| {
                            let mut len = 0;

                            for (index, channel) in runner.channels.iter_mut() {

                                let mut channel_buffer = vec![0.0f32; buffer_len];
                                let samples = channel.get_data(&mut channel_buffer, sample_rate, channels);
                                
                                for i in 0..samples  {
                                    buffer[i] += channel_buffer[i];
                                }

                                len = len.max(samples);
                            }

                            len
                        });
                    }
                }
                MixerTask::ChannelLoad { index: channel, decoder, resampler, time_stretcher } => {
                    if let Some(runner) = runner.as_mut() {
                        if let Some(old) = runner.channels.insert(channel, Channel::new(decoder, resampler, time_stretcher)) {
                            // TODO: Drop stream
                        }
                    }
                }
                MixerTask::ChannelPlay { index } => {
                    if let Some(runner) = runner.as_mut() {
                        if let Some(channel) = runner.channels.get_mut(&index) {
                            channel.play();
                        }
                    }
                }
            }
        }

        if let Some(runner) = runner.as_mut() {
            runner.output.set_paused(true);
        }
    }
}
