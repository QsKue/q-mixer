use std::{sync::mpsc, thread, time::Duration};

use audio::mixer::{ChannelSource, Mixer};

#[test]
#[ignore = "used for local listening checks"]
fn test_play_generated_pitch_shift_local() {
    let (tx_event, _rx_event) = mpsc::channel();
    let mixer = Mixer::new(None, tx_event);

    let sample_rate = 44_100u32;
    let seconds = 6.0f32;
    let total = (sample_rate as f32 * seconds) as usize;
    let mut samples = vec![0.0f32; total];
    for (i, s) in samples.iter_mut().enumerate() {
        *s = (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin() * 0.2;
    }

    mixer.setup();
    mixer
        .load_channel(
            0,
            ChannelSource::GeneratedAudio {
                sample_rate,
                channels: 1,
                samples,
            },
        )
        .unwrap();

    mixer.play_channel(0);
    thread::sleep(Duration::from_millis(1200));
    mixer.set_channel_pitch_semitones(0, 12.0);
    thread::sleep(Duration::from_millis(1200));
    mixer.set_channel_pitch_semitones(0, -5.0);
    thread::sleep(Duration::from_millis(1200));
    mixer.set_channel_speed(0, 0.9);
    thread::sleep(Duration::from_millis(1200));
}
