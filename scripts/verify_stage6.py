"""Stage 6 local integration checks using real saved models; no Gemini key needed."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage5_samples import validation_request
from src.chatbot import ask_copilot


def main():
    reqs = {t: validation_request(t) for t in ("iranian", "ibm")}
    for track, req in reqs.items():
        reply = ask_copilot("আমাকে ব্যাখ্যা করো", track=track, features=req["features"])
        assert reply.mode == "offline" and reply.track == track
        assert reply.result and reply.result["track"] == track
        assert reply.source_label in reply.message
        assert reply.result["disclaimer"] in reply.message
        no_key = req["features"].copy()
        no_key.pop(next(iter(no_key)))
        missing = ask_copilot("Explain", track=track, features=no_key, live=True)
        assert missing.mode == "needs_input" and missing.result is None
        bad = req["features"].copy()
        bad["Churn"] = 1
        rejected = ask_copilot("Explain", track=track, features=bad, live=True)
        assert rejected.mode == "invalid" and rejected.result is None
        offers = ask_copilot("Recommend a Teletalk package price", track=track, features=req["features"], live=True)
        assert offers.mode == "unavailable" and offers.result is None
        print(f"PASS: {track} | source label, real verified prediction, missing-field, extra-target, offer refusal")
    cross = ask_copilot("Explain", track="ibm", features=reqs["iranian"]["features"], live=True)
    assert cross.mode in {"invalid", "needs_input"} and cross.result is None
    no_track = ask_copilot("Explain", features=reqs["iranian"]["features"])
    assert no_track.mode == "needs_input" and no_track.result is None
    print("PASS: missing/ambiguous tracks and cross-track input rejected without LLM")
    print("SUCCESS: Stage 6 offline integration verified. Live Gemini API call NOT tested.")


if __name__ == "__main__":
    main()
