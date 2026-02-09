use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom};

// TODO stop using anyhow
use anyhow::{anyhow, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{Decoder as SymDecoder, DecoderOptions};
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia::core::io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::units::TimeBase;
use symphonia::default::get_probe;

use crate::io::sources::AudioSource;

pub struct SymphoniaDecoder {
    sample_rate: u32,
    channels: usize,
    total_samples: Option<u64>,
    time_base: TimeBase,

    format: Box<dyn FormatReader>,
    decoder: Box<dyn SymDecoder>,
    track_id: u32,

    priming_frames: u64,
    tail_padding_frames: u64,

    sample_buffer: Option<SampleBuffer<f32>>,
    sample_spec: Option<symphonia::core::audio::SignalSpec>,
    sample_capacity: u64,

    decode_cache: VecDeque<f32>,
    pos_frames: u64,
    
    eof_reached: bool,
}

impl SymphoniaDecoder {
    
    pub fn new(source: Box<dyn AudioSource>) -> Result<Self> {
        
        let adapter = SourceAdapter(source);
        let media_source = MediaSourceStream::new(
            Box::new(adapter),
            MediaSourceStreamOptions::default()
        );
        
        let probed = get_probe().format(
            &Default::default(),
            media_source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )?;
        
        let mut format = probed.format;
        let track = format
            .default_track()
            .ok_or_else(|| anyhow!("No default track"))?
            .clone();

        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())?;

        let sample_rate = track
            .codec_params
            .sample_rate
            .or(decoder.codec_params().sample_rate)
            .unwrap_or(44_100);

        let channels = track
            .codec_params
            .channels
            .or(decoder.codec_params().channels)
            .unwrap_or_default()
            .count();

        let time_base = track
            .codec_params
            .time_base
            .or(decoder.codec_params().time_base)
            .unwrap_or_else(|| TimeBase::new(1, sample_rate));

        let mut priming_frames = 0u64;
        let mut tail_padding_frames = 0u64;

        if let Some(meta) = format.metadata().current() {
            for tag in meta.tags() {
                let key = tag.key.to_ascii_lowercase();
                let val = tag.value.to_string();
                if key.contains("itunsmpb") {
                    if let Some((d, p)) = Self::parse_itunsmpb_hex_pair(&val) {
                        priming_frames = d as u64;
                        tail_padding_frames = p as u64;
                    }
                }
            }
        }

        let total_samples = track.codec_params.n_frames.map(|n| {
            n.saturating_sub(priming_frames).saturating_sub(tail_padding_frames)
        });

        let cache_seconds = 0.5;
        let cache_capacity = (sample_rate as f32 * cache_seconds) as usize * channels;

        Ok(Self {
            sample_rate,
            channels,
            total_samples,
            time_base,
            format,
            decoder,
            track_id: track.id,
            priming_frames,
            tail_padding_frames,
            sample_buffer: None,
            sample_spec: None,
            sample_capacity: 0,
            decode_cache: VecDeque::with_capacity(cache_capacity),
            pos_frames: 0,
            eof_reached: false,
        })
    }

    fn parse_itunsmpb_hex_pair(s: &str) -> Option<(u32, u32)> {
        let mut hexes = vec![];
        for tok in s.split_whitespace() {
            if let Ok(v) = u32::from_str_radix(tok.trim_start_matches("0x"), 16) {
                hexes.push(v);
            }
        }
        if hexes.len() >= 3 {
            Some((hexes[1], hexes[2]))
        } else {
            None
        }
    }

    fn decode_next_packet(&mut self) -> Result<bool, String> {
        use symphonia::core::errors::Error;

        loop {
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(Error::IoError(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Ok(false);
                }
                Err(_) => return Ok(false),
            };

            if packet.track_id() != self.track_id {
                continue;
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(buffer) => buffer,
                Err(_) => continue,
            };

            let spec = *decoded.spec();
            let capacity = decoded.capacity() as u64;

            let need_new = self.sample_buffer.is_none() 
                || self.sample_capacity < capacity 
                || self.sample_spec != Some(spec);

            if need_new {
                self.sample_buffer = Some(SampleBuffer::<f32>::new(capacity, spec));
                self.sample_capacity = capacity;
                self.sample_spec = Some(spec);
            }

            let sample_buffer = self.sample_buffer.as_mut().unwrap();
            sample_buffer.copy_interleaved_ref(decoded);
            
            let samples = sample_buffer.samples();
            self.decode_cache.extend(samples);
            
            return Ok(true);
        }
    }
}

impl super::Decoder for SymphoniaDecoder {

    #[inline]
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline]
    fn channels(&self) -> usize {
        self.channels
    }

    #[inline]
    fn total_samples(&self) -> Option<u64> {
        self.total_samples
    }

    fn decode(&mut self, buffer: &mut [f32]) -> Result<usize, String> {
        let frames_requested = buffer.len() / self.channels;
        let mut frames_written = 0;

        while frames_written < frames_requested && !self.eof_reached {
            // Drain from cache first
            let frames_available = self.decode_cache.len() / self.channels;
            if frames_available > 0 {
                let frames_to_copy = (frames_requested - frames_written).min(frames_available);
                let samples_to_copy = frames_to_copy * self.channels;
                
                for i in 0..samples_to_copy {
                    buffer[frames_written * self.channels + i] = 
                        self.decode_cache.pop_front().unwrap();
                }
                
                frames_written += frames_to_copy;
                self.pos_frames += frames_to_copy as u64;
            }

            // Need more data?
            if frames_written < frames_requested {
                match self.decode_next_packet() {
                    Ok(true) => continue,
                    Ok(false) => {
                        self.eof_reached = true;
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        Ok(frames_written)
    }

    fn position_samples(&self) -> u64 {
        self.pos_frames
    }

    fn seek(&mut self, frame: u64) -> Result<u64, String> {
        let target_frame = frame + self.priming_frames;
        
        let secs = target_frame as f64 / self.sample_rate as f64;
        let time = symphonia::core::units::Time {
            seconds: secs.trunc() as u64,
            frac: secs.fract(),
        };

        self.format.seek(
            SeekMode::Coarse,
            SeekTo::Time { time, track_id: Some(self.track_id) },
        ).map_err(|e| format!("Seek failed: {:?}", e))?;

        self.decoder.reset();
        self.decode_cache.clear();
        self.pos_frames = target_frame;
        self.eof_reached = false;

        Ok(frame)
    }

    fn is_eof(&self) -> bool {
        self.eof_reached && self.decode_cache.is_empty()
    }

    fn reset(&mut self) -> Result<(), String> {
        self.seek(0)?;
        Ok(())
    }
}

struct SourceAdapter(Box<dyn AudioSource>);
        
impl MediaSource for SourceAdapter {
    fn is_seekable(&self) -> bool {
        self.0.is_seekable()
    }
    
    fn byte_len(&self) -> Option<u64> {
        self.0.size()
    }
}

impl Read for SourceAdapter {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

impl Seek for SourceAdapter {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match pos {
            SeekFrom::Start(p) => self.0.seek(p),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "only SeekFrom::Start supported"
            ))
        }
    }
}