// ShitCode, AI Generated, refactor as soon as possible

use std::collections::VecDeque;

use anyhow::Context as _;
use rubato::{
    audioadapter::{Adapter, AdapterMut},
    Async, FixedAsync, Indexing,
    Resampler as RubatoResamplerTrait,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use super::{Resampler, ResamplerResult, ResamplerStatus};

pub struct RubatoSettings {
    pub sinc_len: usize,
    pub f_cutoff: f32,
    pub oversampling_factor: usize,
    pub interpolation: SincInterpolationType,
    pub window: WindowFunction,
    pub chunk_in_frames: usize,
    pub max_resample_ratio_relative: f64,
}

impl Default for RubatoSettings {
    fn default() -> Self {
        Self {
            sinc_len: 128,
            f_cutoff: 0.95,
            oversampling_factor: 80,
            interpolation: SincInterpolationType::Linear,
            window: WindowFunction::BlackmanHarris2,
            chunk_in_frames: 1024,
            max_resample_ratio_relative: 8.0,
        }
    }
}

pub struct RubatoResampler {
    sample_rate: u32,
    channels: usize,
    settings: RubatoSettings,

    rub: Option<Async<f32>>,
    out_sample_rate: u32,

    in_buf: Vec<f32>,
    out_buf: Vec<f32>,

    tmp_frame: Vec<f32>,
    out_cache: VecDeque<f32>,
}

impl RubatoResampler {
    pub fn new() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            settings: RubatoSettings::default(),

            rub: None,
            out_sample_rate: 0,

            in_buf: Vec::new(),
            out_buf: Vec::new(),
            tmp_frame: Vec::new(),

            out_cache: VecDeque::with_capacity(44_100 * 2),
        }
    }

    fn ensure_rubato(&mut self, out_sample_rate: u32) -> anyhow::Result<()> {
        if self.rub.is_some() && self.out_sample_rate == out_sample_rate {
            return Ok(());
        }

        if out_sample_rate == self.sample_rate {
            self.rub = None;
            self.out_sample_rate = out_sample_rate;
            return Ok(());
        }

        let chunk = self.settings.chunk_in_frames.max(16);
        let ch = self.channels;

        let params = SincInterpolationParameters {
            sinc_len: self.settings.sinc_len,
            f_cutoff: self.settings.f_cutoff,
            oversampling_factor: self.settings.oversampling_factor,
            interpolation: self.settings.interpolation,
            window: self.settings.window,
        };

        let ratio = out_sample_rate as f64 / self.sample_rate as f64;

        let r = Async::<f32>::new_sinc(
            ratio,
            self.settings.max_resample_ratio_relative,
            &params,
            chunk,
            ch,
            FixedAsync::Input,
        )
        .context("failed to create rubato::Async sinc resampler")?;

        let in_max = r.input_frames_max();
        self.in_buf.resize(in_max * ch, 0.0);

        let out_max = r.output_frames_max();
        self.out_buf.resize(out_max * ch, 0.0);

        self.tmp_frame.resize(ch, 0.0);

        let approx_max_expand = ((out_sample_rate as f64 / self.sample_rate as f64) * chunk as f64).ceil() as usize;
        self.out_cache = VecDeque::with_capacity(approx_max_expand.saturating_mul(ch).saturating_mul(2));

        self.rub = Some(r);
        self.out_sample_rate = out_sample_rate;
        Ok(())
    }

    #[inline]
    fn map_channels_per_frame(input: &[f32], in_channels: usize, output: &mut [f32], out_channels: usize) {
        match (in_channels, out_channels) {
            (i, o) if i == o => {
                output[..i].copy_from_slice(&input[..i]);
            }
            (i, 1) => {
                let sum: f32 = input[..i].iter().copied().sum();
                output[0] = sum / (i as f32);
            }
            (1, o) => {
                for c in 0..o {
                    output[c] = input[0];
                }
            }
            (_, 2) => {
                let l = input[0];
                let r = if in_channels > 1 { input[1] } else { l };
                output[0] = l;
                output[1] = r;
            }
            (i, o) if o > i => {
                output[..i].copy_from_slice(&input[..i]);
                for c in i..o {
                    output[c] = 0.0;
                }
            }
            (_i, o) => {
                output[..o].copy_from_slice(&input[..o]);
            }
        }
    }

    #[inline]
    fn convert_from_deque(
        src: &mut VecDeque<f32>,
        buffer: &mut [f32],
        in_channels: usize,
        out_channels: usize,
        tmp_frame: &mut [f32],
    ) -> usize {
        if in_channels == 0 || out_channels == 0 {
            return 0;
        }

        let frames_avail = src.len() / in_channels;
        let frames_writable = buffer.len() / out_channels;
        let frames = frames_avail.min(frames_writable);
        if frames == 0 {
            return 0;
        }

        if in_channels == out_channels {
            let need = frames * in_channels;
            let (a, b) = src.as_slices();
            let a_take = a.len().min(need);

            buffer[..a_take].copy_from_slice(&a[..a_take]);

            let rem = need - a_take;
            if rem > 0 {
                buffer[a_take..a_take + rem].copy_from_slice(&b[..rem]);
            }

            src.drain(..need);
            return frames;
        }

        for f in 0..frames {
            for c in 0..in_channels {
                tmp_frame[c] = src.pop_front().unwrap();
            }
            let base = f * out_channels;
            Self::map_channels_per_frame(&tmp_frame[..in_channels], in_channels, &mut buffer[base..base + out_channels], out_channels);
        }

        frames
    }

    #[inline]
    fn stage_or_write_out(
        out_cache: &mut VecDeque<f32>,
        tmp_frame: &mut Vec<f32>,
        out_interleaved: &[f32],
        out_frames: usize,
        out_channels_in_buf: usize,
        buffer: Option<&mut [f32]>,
        out_channels: usize,
    ) -> usize {
        match buffer {
            None => {
                out_cache.extend(
                    out_interleaved[..out_frames * out_channels_in_buf]
                        .iter()
                        .copied(),
                );
                out_frames
            }
            Some(buf) => {
                let frames_writable = buf.len() / out_channels;
                let n = out_frames.min(frames_writable);

                if out_channels_in_buf == out_channels {
                    let samples = n * out_channels;
                    buf[..samples].copy_from_slice(&out_interleaved[..samples]);
                } else {
                    tmp_frame.resize(out_channels_in_buf, 0.0);
                    for i in 0..n {
                        let in_base = i * out_channels_in_buf;
                        let out_base = i * out_channels;

                        tmp_frame[..out_channels_in_buf].copy_from_slice(
                            &out_interleaved[in_base..in_base + out_channels_in_buf],
                        );

                        RubatoResampler::map_channels_per_frame(
                            &tmp_frame[..out_channels_in_buf],
                            out_channels_in_buf,
                            &mut buf[out_base..out_base + out_channels],
                            out_channels,
                        );
                    }
                }

                // Stage remainder if produced more than fits
                if out_frames > n {
                    let start = n * out_channels_in_buf;
                    let end = out_frames * out_channels_in_buf;
                    out_cache.extend(out_interleaved[start..end].iter().copied());
                }

                n
            }
        }
    }
}

impl Resampler for RubatoResampler {
    fn produce_into(
        &mut self,
        decode_cache: &mut VecDeque<f32>,
        out_sample_rate: u32,
        min_out_frames: usize,
        mut buffer: Option<&mut [f32]>,
        out_channels: usize,
        eof: bool,
    ) -> ResamplerStatus {
        if self.channels == 0 || out_channels == 0 {
            return ResamplerStatus::NeedMoreInput;
        }

        if self.ensure_rubato(out_sample_rate).is_err() {
            return ResamplerStatus::NeedMoreInput;
        }

        if self.rub.is_none() {
            let frames_available = decode_cache.len() / self.channels;
            if frames_available == 0 {
                return if eof { ResamplerStatus::Flushed } else { ResamplerStatus::NeedMoreInput };
            }

            let want = frames_available.min(min_out_frames.max(1));
            match buffer.as_deref_mut() {
                Some(buf) => {
                    let max_samples = want.saturating_mul(out_channels).min(buf.len());
                    let (head, _) = buf.split_at_mut(max_samples);

                    let mut tmp = vec![0.0f32; self.channels];
                    let frames_done = Self::convert_from_deque(
                        decode_cache,
                        head,
                        self.channels,
                        out_channels,
                        &mut tmp,
                    );

                    return ResamplerStatus::Progress {
                        result: ResamplerResult { out_frames: frames_done, src_frames_used: frames_done }
                    };
                }
                None => {
                    let samples = want * self.channels;
                    let (a, b) = decode_cache.as_slices();
                    let a_take = a.len().min(samples);
                    self.out_cache.extend(a[..a_take].iter().copied());

                    let rem = samples - a_take;
                    if rem > 0 {
                        self.out_cache.extend(b[..rem].iter().copied());
                    }

                    decode_cache.drain(..samples);

                    return ResamplerStatus::Progress {
                        result: ResamplerResult { out_frames: want, src_frames_used: want }
                    };
                }
            }
        }

        let r = self.rub.as_mut().unwrap();

        let mut out_frames_total = 0usize;
        let mut src_used_total = 0usize;

        for _ in 0..4 {
            if out_frames_total >= min_out_frames {
                break;
            }

            let need_in = r.input_frames_next();
            let have_frames = decode_cache.len() / self.channels;

            if have_frames == 0 && !eof {
                break;
            }

            if have_frames < need_in && !eof {
                break;
            }

            let feed_frames = have_frames.min(need_in);

            let need_samples = need_in * self.channels;
            if self.in_buf.len() < need_samples {
                self.in_buf.resize(need_samples, 0.0);
            }
            self.in_buf[..need_samples].fill(0.0);

            for i in 0..(feed_frames * self.channels) {
                if let Some(v) = decode_cache.pop_front() {
                    self.in_buf[i] = v;
                } else {
                    break;
                }
            }

            let out_cap_frames = r.output_frames_next().min(r.output_frames_max());
            let out_need_samples = out_cap_frames * self.channels;
            if self.out_buf.len() < out_need_samples {
                self.out_buf.resize(out_need_samples, 0.0);
            }

            let input = InterleavedSlice::new(&self.in_buf[..need_samples], self.channels, need_in).unwrap();

            let indexing = Indexing {
                input_offset: 0,
                output_offset: 0,
                partial_len: if feed_frames < need_in { Some(feed_frames) } else { None },
                active_channels_mask: None,
            };

            let (in_used, out_done) = {
                let mut output = InterleavedSliceMut::new(
                    &mut self.out_buf[..out_need_samples],
                    self.channels,
                    out_cap_frames,
                ).unwrap();

                r.process_into_buffer(&input, &mut output, Some(&indexing))
                    .unwrap_or((0, 0))
            };

            src_used_total += in_used;

            let wrote = Self::stage_or_write_out(
                &mut self.out_cache,
                &mut self.tmp_frame,
                &self.out_buf,
                out_done,
                self.channels,
                buffer.as_deref_mut(),
                out_channels,
            );

            out_frames_total += wrote;

            if let Some(buf) = buffer {
                let advance = (wrote * out_channels).min(buf.len());
                let (_, tail) = buf.split_at_mut(advance);
                buffer = Some(tail);
            }

            if out_done == 0 && in_used == 0 {
                break;
            }
        }

        if out_frames_total > 0 || src_used_total > 0 {
            return ResamplerStatus::Progress {
                result: ResamplerResult {
                    out_frames: out_frames_total,
                    src_frames_used: src_used_total,
                }
            };
        }

        if eof {
            ResamplerStatus::Flushed
        } else {
            ResamplerStatus::NeedMoreInput
        }
    }

    fn drain_out_with_conv(&mut self, buffer: &mut [f32], out_channels: usize) -> usize {
        self.tmp_frame.resize(self.channels.max(1), 0.0);
        let frames = Self::convert_from_deque(
            &mut self.out_cache,
            buffer,
            self.channels,
            out_channels,
            &mut self.tmp_frame,
        );
        frames * out_channels
    }

    fn reset(&mut self) {
        self.out_cache.clear();
        if let Some(r) = self.rub.as_mut() {
            r.reset();
        }
        self.rub = None;
        self.out_sample_rate = 0;
    }
}

pub struct InterleavedSlice<'a, T> {
    data: &'a [T],
    channels: usize,
    frames: usize,
}

impl<'a, T> InterleavedSlice<'a, T> {
    pub fn new(data: &'a [T], channels: usize, frames: usize) -> Option<Self> {
        if channels == 0 { return None; }
        if data.len() < channels * frames { return None; }
        Some(Self { data, channels, frames })
    }

    #[inline]
    fn idx(&self, ch: usize, fr: usize) -> usize {
        fr * self.channels + ch
    }
}

impl<'a, T: Clone + 'a> Adapter<'a, T> for InterleavedSlice<'a, T> {
    #[inline]
    unsafe fn read_sample_unchecked(&self, channel: usize, frame: usize) -> T {
        unsafe { self.data.get_unchecked(self.idx(channel, frame)).clone() }
    }
    #[inline]
    fn channels(&self) -> usize { self.channels }
    #[inline]
    fn frames(&self) -> usize { self.frames }
}

/// Mutable adapter over interleaved audio.
pub struct InterleavedSliceMut<'a, T> {
    data: &'a mut [T],
    channels: usize,
    frames: usize,
}

impl<'a, T> InterleavedSliceMut<'a, T> {
    pub fn new(data: &'a mut [T], channels: usize, frames: usize) -> Option<Self> {
        if channels == 0 { return None; }
        if data.len() < channels * frames { return None; }
        Some(Self { data, channels, frames })
    }

    #[inline]
    fn idx(&self, ch: usize, fr: usize) -> usize {
        fr * self.channels + ch
    }
}

impl<'a, T: Clone + 'a> Adapter<'a, T> for InterleavedSliceMut<'a, T> {
    #[inline]
    unsafe fn read_sample_unchecked(&self, channel: usize, frame: usize) -> T {
        unsafe { self.data.get_unchecked(self.idx(channel, frame)).clone() }
    }
    #[inline]
    fn channels(&self) -> usize { self.channels }
    #[inline]
    fn frames(&self) -> usize { self.frames }
}

impl<'a, T: Clone + 'a> AdapterMut<'a, T> for InterleavedSliceMut<'a, T> {
    #[inline]
    unsafe fn write_sample_unchecked(
        &mut self,
        channel: usize,
        frame: usize,
        value: &T,
    ) -> bool {
        let i = self.idx(channel, frame);
        *unsafe { self.data.get_unchecked_mut(i) } = value.clone();
        true
    }
}