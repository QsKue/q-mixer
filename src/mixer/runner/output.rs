use std::sync::Arc;

// TODO: stop using anyhow
use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{HeapRb, SharedRb, storage::Heap, traits::{Consumer, Observer, Producer, Split}, wrap::caching::Caching};

pub(super) struct Output {
    stream: cpal::Stream,
    config: cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    
    buffer: Caching<Arc<SharedRb<Heap<f32>>>, true, false>,
    state: Arc<OutputState>,
}

struct OutputState {
    paused: std::sync::atomic::AtomicBool,
    volume: std::sync::atomic::AtomicU32,
    needs_flush: std::sync::atomic::AtomicBool,
}

impl Output {

    pub fn new(buffer_size_ms: u64) -> Result<Self> {
        let host = cpal::default_host();

        let device = host
            .default_output_device()
            .context("No output device found")?;
        
        let supported_config = device
            .default_output_config()
            .context("No default output config")?;
        
        Self::build_with_device(device, supported_config, buffer_size_ms)
    }

    fn build_with_device(
        device: cpal::Device,
        supported_config: cpal::SupportedStreamConfig,
        buffer_size_ms: u64,
    ) -> Result<Self> {
        let config = supported_config.config();
        let sample_format = supported_config.sample_format();
        
        let sample_rate = config.sample_rate;
        let channels = config.channels as usize;
        
        let capacity_frames = (sample_rate as u64 * buffer_size_ms / 1000) as usize;
        let capacity_samples = capacity_frames * channels;
        
        let ring_buffer = HeapRb::<f32>::new(capacity_samples);
        let (buffer, consumer) = ring_buffer.split();
        
        let state = Arc::new(OutputState {
            paused: std::sync::atomic::AtomicBool::new(false),
            volume: std::sync::atomic::AtomicU32::new(1.0f32.to_bits()),
            needs_flush: std::sync::atomic::AtomicBool::new(false),
        });
        
        let stream = Self::build_stream_dynamic(
            &device,
            &config,
            sample_format,
            consumer,
            state.clone()
        )?;
        
        stream.play()?;
        
        Ok(Self {
            stream,
            config,
            sample_format,
            buffer,
            state,
        })
    }

    fn build_stream_dynamic(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        sample_format: cpal::SampleFormat,
        consumer: Caching<Arc<SharedRb<Heap<f32>>>, false, true>,
        state: Arc<OutputState>,
    ) -> Result<cpal::Stream> {
        use cpal::SampleFormat;
        
        match sample_format {
            SampleFormat::F32 => Self::build_stream::<f32>(device, config, consumer, state),
            SampleFormat::I16 => Self::build_stream::<i16>(device, config, consumer, state),
            SampleFormat::U16 => Self::build_stream::<u16>(device, config, consumer, state),
            _ => bail!("Unsupported sample format: {:?}", sample_format),
        }
    }

    fn build_stream<T>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        mut consumer: Caching<Arc<SharedRb<Heap<f32>>>, false, true>,
        state: Arc<OutputState>,
    ) -> Result<cpal::Stream>
    where
        T: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
    {
        let channels = config.channels as usize;
        
        let stream = device.build_output_stream(
            config,
            move |output: &mut [T], _: &cpal::OutputCallbackInfo| {

                if state.needs_flush.load(std::sync::atomic::Ordering::Relaxed) {
                    while consumer.try_pop().is_some() {}
                    state.needs_flush.store(false, std::sync::atomic::Ordering::Relaxed);
                }

                if state.paused.load(std::sync::atomic::Ordering::Relaxed) {

                    for sample in output.iter_mut() {
                        *sample = T::from_sample(0.0);
                    }
                    
                    return;
                }
                
                let volume = f32::from_bits(
                    state.volume.load(std::sync::atomic::Ordering::Relaxed)
                );
                
                for frame in output.chunks_mut(channels) {
                    for sample in frame.iter_mut() {
                        let value = consumer.try_pop().unwrap_or(0.0) * volume;
                        *sample = T::from_sample(value);
                    }
                }
            },
            |err| eprintln!("Output stream error: {}", err),
            None,
        )?;
        
        Ok(stream)
    }

    pub fn set_paused(&self, paused: bool) {
        self.state.paused.store(paused, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn refill_buffer<F>(&mut self, mut get_data: F) where F: FnMut(&mut [f32], usize, u32, usize) -> usize {

        let sr_out = self.config.sample_rate;
        let ch_out = self.config.channels as usize;

        let mut writable = self.buffer.vacant_len();
        writable -= writable % ch_out;
        if writable == 0 { return; }

        let mut mix_buf = vec![0.0f32; writable];
        let len = get_data(&mut mix_buf, writable, sr_out, ch_out);

        for sample in &mix_buf[..len] {
            if self.buffer.try_push(*sample).is_err() {
                break;
            }
        }
    }
}
