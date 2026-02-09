
#[derive(Clone, Copy)]
pub enum StreamTime {
    Beat(f32),
    Second(f64),
}

impl StreamTime {

    pub(crate) fn from_sample(sample: u64, sample_rate: u32) -> Self {
        StreamTime::Second(sample as f64 / sample_rate as f64)
    }

    #[inline]
    pub fn seconds(&self) -> f64 {
        match self {
            StreamTime::Beat(beat) => 0.0, // TODO: BPM grid
            StreamTime::Second(seconds) => *seconds,
        }
    }

    #[inline]
    pub(crate) fn sample(&self, sample_rate: u32) -> u64 {
        (self.seconds() * sample_rate as f64).round() as u64
    }
}