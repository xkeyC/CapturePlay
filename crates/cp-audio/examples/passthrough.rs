//! Lists audio endpoints, matches the capture card's input to a video
//! device name and runs the passthrough for a few seconds, reporting the
//! ring fill and underruns.
//!
//! `cargo run -p cp-audio --example passthrough -- ["video device name"] [seconds]`

use std::sync::atomic::Ordering;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let video = args.first().map_or("Live Gamer Ultra 2.1", String::as_str);
    let secs: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);

    let inputs = cp_audio::endpoints::list(true)?;
    let outputs = cp_audio::endpoints::list(false)?;
    println!("inputs:");
    for e in &inputs {
        println!("  {}  [{}]", e.name, e.id);
    }
    println!("outputs:");
    for e in &outputs {
        println!("  {}  [{}]", e.name, e.id);
    }
    let input = cp_audio::match_capture_endpoint(video, &inputs)
        .ok_or_else(|| anyhow::anyhow!("no input matches {video:?}"))?;
    println!("matched {video:?} -> {}", input.name);

    let p = cp_audio::AudioPassthrough::start(cp_audio::AudioConfig {
        input_id: input.id.clone(),
        output_id: None,
        target_latency_ms: 30,
        av_offset_ms: 0,
        volume: 1.0,
    })?;
    for _ in 0..secs {
        std::thread::sleep(Duration::from_secs(1));
        let s = p.shared();
        println!(
            "rate {} ch {} fill {} ms underruns {}",
            s.sample_rate.load(Ordering::Relaxed),
            s.channels.load(Ordering::Relaxed),
            s.fill_ms.load(Ordering::Relaxed),
            s.underruns.load(Ordering::Relaxed)
        );
    }
    p.stop();
    Ok(())
}
