use std::sync::mpsc;

use audio::mixer::{ChannelSource, Mixer};

#[test]
#[ignore = "used for local testing"]
fn test_play_file_local() {

    let current_path = std::env::current_dir().unwrap();
    println!("The current directory is: {}", current_path.display());
    
    let (tx_event, rx_event) = mpsc::channel();
    let mixer = Mixer::new(None, tx_event);

    let source = ChannelSource::File { path: "tests/Aylex - Sunny Day (freetouse.com).mp3".to_string() };

    mixer.setup();
    mixer.load_channel(0, source).unwrap();
    mixer.play_channel(0);

    println!("Running test, press enter to exit");
    
    let stdin = std::io::stdin();
    for line in std::io::BufRead::lines(stdin.lock()) {
        break;
    }
}
