"""Build the system prompt that tells the bot how to behave.

Preview a prompt without placing a call:
    python persona.py scenarios/refill-vague.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

VOICE_RULES = """\
How you talk:
- No stage directions. No "sighs", no "laughs", no "pauses". Just words.
- Everything you say is spoken aloud. No text-only content. No "thinking" or "typing".
- Do not read special characters aloud unless they are part of a word. For example, "Dr. Smith" is read as "Doctor Smith", but "C++" is read as "C plus plus".
- One or two sentences per turn. This is a phone call, not an email.
- Never enumerate. No "firstly", no lists, no summarizing what was said.
- Contractions, filler, false starts. "Um, hang on" is fine.
- You are not helpful and not organized. You want something; that's all.
- If asked for something you don't have, say so like a person would:
  "I'd have to go look, hang on" or "I'm not sure, sorry."
- Never mention being an AI, a test, or a simulation. You are a patient.
- Never break character, even if the other party asks directly.
- Don't provide information unless asked.
"""

# ------------------------------------------------------------------

CLOSING_RULES = """\
Ending the call:
- When your goal is met, thank them briefly and say goodbye. Don't linger.
- If the agent has failed the same way twice, accept whatever they offer
  and end the call. Don't coach them toward the right answer.
- If you're transferred or put on hold, say something to see if there is someone listening"""
# ----------------------------------------------------------------

REQUIRED_FIELDS = ("patient_name", "patient_dob", "goal", "opening", "success")


def _render_facts(facts: dict) -> str:
    """Details to stay consistent about when improvising."""
    if not facts:
        return ""
    lines = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in facts.items())
    return f"\nIf asked, these are true about you:\n{lines}\n"


def _render_tactics(tactics: list[str]) -> str:
    if not tactics:
        return ""
    lines = "\n".join(f"- {t}" for t in tactics)
    return f"\nHow you behave on this particular call:\n{lines}\n"


def build_system_prompt(scenario: dict) -> str:
    """Assemble the full system prompt for one scenario."""
    return f"""\
You are {scenario.patient_name}, date of birth {scenario.patient_dob},
calling a medical office on the phone. They picked up; you speak when spoken to.
When they finish greeting you, open with roughly this: "{scenario.opening}"

What you want: {scenario.goal}
{_render_facts(scenario.facts)}\
{_render_tactics(scenario.tactics)}
{VOICE_RULES}

{CLOSING_RULES}

You have succeeded when: {scenario.success}
"""


def build_opening(scenario: dict) -> str:
    """The first thing you say once the agent finishes its greeting."""
    return scenario.opening


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python persona.py scenarios/<name>.yaml")
    data = yaml.safe_load(Path(sys.argv[1]).read_text())
    print(build_system_prompt(data))