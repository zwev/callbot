"""Download call recordings from Twilio and render readable transcripts.

Reads calls/manifest.jsonl, fetches the MP3 for each call SID, and writes a
timestamped .txt beside the .json transcript the bot saved during the call.

Usage:
    python fetch_calls.py            # everything in the manifest
    python fetch_calls.py CAxxx      # one call
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

CALL_DIR = Path("calls")
MANIFEST = CALL_DIR / "manifest.jsonl"


def client() -> Client:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        sys.exit("Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env")
    return Client(sid, token)


def manifest_entries() -> list[dict]:
    if not MANIFEST.exists():
        sys.exit(f"No manifest at {MANIFEST} - place some calls first")
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]


def download_recording(twilio: Client, call_sid: str, scenario: str) -> Path | None:
    """Fetch the MP3 for one call. Returns the path, or None if not available."""
    recordings = twilio.recordings.list(call_sid=call_sid)
    if not recordings:
        print(f"  no recording yet for {call_sid}")
        return None

    rec = recordings[0]
    out = CALL_DIR / f"{scenario}-{call_sid[-6:]}.mp3"
    if out.exists():
        print(f"  already have {out.name}")
        return out

    # Twilio serves the media off api.twilio.com, not the REST resource URL.
    url = f"https://api.twilio.com{rec.uri.replace('.json', '.mp3')}"
    response = requests.get(
        url,
        auth=(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]),
        timeout=60,
    )
    response.raise_for_status()
    out.write_bytes(response.content)
    print(f"  saved {out.name} ({len(response.content) // 1024} KB)")
    return out


def render_transcript(call_sid: str, scenario: str) -> Path | None:
    """Turn the bot's JSON transcript into a readable, timestamped .txt."""
    src = CALL_DIR / f"{call_sid}.json"
    if not src.exists():
        print(f"  no transcript json for {call_sid}")
        return None

    data = json.loads(src.read_text(encoding="utf-8"))
    turns = data.get("turns", [])
    if not turns:
        print(f"  transcript for {call_sid} is empty")
        return None

    def parsed(ts: str | None) -> datetime | None:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None

    start = next((parsed(t.get("timestamp")) for t in turns if parsed(t.get("timestamp"))), None)

    lines = [
        f"Call:     {call_sid}",
        f"Scenario: {scenario}",
        "-" * 60,
        "",
    ]
    for turn in turns:
        when = parsed(turn.get("timestamp"))
        if when and start:
            offset = (when - start).total_seconds()
            stamp = f"[{int(offset // 60)}:{int(offset % 60):02d}]"
        else:
            stamp = "[--:--]"
        # Column-aligned
        lines.append(f"{stamp} {turn['role'].upper():<8} {turn['content']}")

    out = CALL_DIR / f"{scenario}-{call_sid[-6:]}.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out.name} ({len(turns)} turns)")
    return out


def main() -> None:
    twilio = client()

    if len(sys.argv) > 1:
        entries = [e for e in manifest_entries() if e["call_sid"] == sys.argv[1]]
        if not entries:
            sys.exit(f"{sys.argv[1]} not found in manifest")
    else:
        entries = manifest_entries()

    for entry in entries:
        call_sid = entry["call_sid"]
        scenario = entry.get("scenario", "unknown")
        print(f"{scenario} ({call_sid})")
        download_recording(twilio, call_sid, scenario)
        render_transcript(call_sid, scenario)


if __name__ == "__main__":
    main()