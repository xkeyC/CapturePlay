//! Audio endpoints and capture-card matching (platform independent).

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    pub id: String,
    pub name: String,
}

/// Normalized product name of a capture device, e.g.
/// `"Live Gamer Ultra 2.1-Video"` → `"live gamer ultra 2.1"`.
fn product_key(name: &str) -> String {
    let lower = name.to_lowercase();
    let trimmed = lower.trim_end_matches(|c: char| c == ')' || c.is_whitespace());
    let base = ["-video", " video", "-audio", " audio"]
        .iter()
        .fold(trimmed.to_string(), |acc, suffix| {
            acc.trim_end_matches(suffix).to_string()
        });
    base.trim().to_string()
}

/// Finds the capture endpoint belonging to a video capture device by name.
/// Never falls back to an unrelated input (that would play a microphone).
pub fn match_capture_endpoint<'a>(
    video_name: &str,
    endpoints: &'a [Endpoint],
) -> Option<&'a Endpoint> {
    let key = product_key(video_name);
    if key.len() < 3 {
        return None;
    }
    endpoints.iter().find(|e| {
        let n = e.name.to_lowercase();
        // Endpoint names look like "HDMI/Line In (Live Gamer Ultra 2.1-Audio)".
        let inner = n.rsplit_once('(').map_or(n.as_str(), |(_, r)| r);
        product_key(inner) == key || n.contains(&key)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(name: &str) -> Endpoint {
        Endpoint {
            id: name.into(),
            name: name.into(),
        }
    }

    #[test]
    fn matches_avermedia_and_elgato_naming() {
        let eps = [
            ep("麦克风 (Steam Streaming Microphone)"),
            ep("HDMI/Line In (Live Gamer Ultra 2.1-Audio)"),
            ep("Game Capture 4K X Audio (Elgato)"),
        ];
        assert_eq!(
            match_capture_endpoint("Live Gamer Ultra 2.1-Video", &eps).map(|e| e.name.as_str()),
            Some("HDMI/Line In (Live Gamer Ultra 2.1-Audio)")
        );
        assert_eq!(
            match_capture_endpoint("Game Capture 4K X", &eps).map(|e| e.name.as_str()),
            Some("Game Capture 4K X Audio (Elgato)")
        );
        assert_eq!(match_capture_endpoint("OBS Virtual Camera", &eps), None);
    }
}
