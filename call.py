"""Place test calls to the target voice agent from YAML scenario files.

Calls run one at a time: each is polled to completion before the next dials,
so the single-scenario hand-off in server_utils stays valid and the callee
never gets a second call while the first is still up.

Usage:
    python call.py --scenario refill-vague
    python call.py --all
    python call.py --scenario refill-vague --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

SCENARIO_DIR = Path("scenarios")
MANIFEST = Path("calls/manifest.jsonl")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")

# Twilio call states that mean "this call is over, one way or another".
TERMINAL_STATUSES = {"completed", "busy", "no-answer", "failed", "canceled"}

POLL_INTERVAL = 3       # seconds between status checks
MAX_CALL_SECONDS = 300  # give up waiting and force the hangup
SETTLE_SECONDS = 10     # breathing room after hangup before the next dial


def twilio_client() -> Client:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        sys.exit("Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env")
    return Client(sid, token)


def load_scenario(name: str) -> dict:
    """Read scenarios/{name}.yaml and return it as a dict."""
    path = SCENARIO_DIR / f"{name}.yaml"
    if not path.exists():
        sys.exit(f"No scenario file at {path}")

    scenario = yaml.safe_load(path.read_text())
    if not isinstance(scenario, dict):
        sys.exit(f"{path} did not parse as a mapping")

    # Filename is the source of truth for the name, so they can't drift.
    scenario["name"] = name
    return scenario


def all_scenario_names() -> list[str]:
    names = sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))
    if not names:
        sys.exit(f"No scenario files found in {SCENARIO_DIR}/")
    return names


def build_payload(scenario: dict) -> dict:
    to_number = os.getenv("TARGET_NUMBER")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    if not to_number or not from_number:
        sys.exit("Set TARGET_NUMBER and TWILIO_FROM_NUMBER in .env")

    return {
        "to_number": to_number,
        "from_number": from_number,
        "scenario": scenario,
    }


def record(entry: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def place_call(scenario: dict) -> str | None:
    """POST to the dialout endpoint. Returns the call SID, or None on failure."""
    payload = build_payload(scenario)

    try:
        response = requests.post(f"{SERVER_URL}/dialout", json=payload, timeout=30)
    except requests.RequestException as e:
        print(f"  request failed: {e}")
        return None

    if response.status_code == 422:
        print(f"  scenario rejected by server:\n    {response.json()['detail']}")
        return None

    if not response.ok:
        print(f"  server returned {response.status_code}: {response.text[:200]}")
        return None

    call_sid = response.json()["call_sid"]
    record(
        {
            "call_sid": call_sid,
            "scenario": scenario["name"],
            "to_number": payload["to_number"],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return call_sid


def wait_for_completion(client: Client, call_sid: str) -> str:
    """Poll Twilio until the call reaches a terminal state. Returns the status.

    Twilio is the authority here rather than our own server: a call that is
    never answered (busy, no-answer) never opens a WebSocket, so the bot side
    has no idea it happened.
    """
    deadline = time.monotonic() + MAX_CALL_SECONDS
    last_status = ""

    while time.monotonic() < deadline:
        status = client.calls(call_sid).fetch().status
        if status != last_status:
            print(f"  {status}")
            last_status = status
        if status in TERMINAL_STATUSES:
            return status
        time.sleep(POLL_INTERVAL)

    print(f"  still {last_status} after {MAX_CALL_SECONDS}s - forcing hangup")
    client.calls(call_sid).update(status="completed")
    return "timed-out"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", help="scenario name, without the .yaml")
    group.add_argument("--all", action="store_true", help="run every scenario")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the payload and the rendered system prompt, then exit",
    )
    args = parser.parse_args()

    names = all_scenario_names() if args.all else [args.scenario]

    if args.dry_run:
        from persona import build_system_prompt
        from server_utils import Scenario

        for name in names:
            scenario = load_scenario(name)
            print(json.dumps(build_payload(scenario), indent=2))
            print("\n--- system prompt ---")
            print(build_system_prompt(Scenario(**scenario)))
            print()
        return

    client = twilio_client()

    for i, name in enumerate(names):
        scenario = load_scenario(name)
        print(f"[{i + 1}/{len(names)}] {name}")

        call_sid = place_call(scenario)
        if not call_sid:
            continue

        print(f"  placed: {call_sid}")
        status = wait_for_completion(client, call_sid)
        print(f"  finished: {status}")

        if i < len(names) - 1:
            time.sleep(SETTLE_SECONDS)


if __name__ == "__main__":
    main()