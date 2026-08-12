"""
Purely decorative opener phrases observed in the dataset. Frequency analysis
(src/_analyze2.py) showed each of these prefixed ~90 different core messages
across every category in roughly equal measure — i.e. they carry no category
signal and are noise, unlike content markers such as "Calendar update:" or
"Personal note:" which DO correlate with a category and are deliberately left
alone. Stripped only before task/event title extraction, so a clean action
phrase can be pulled out (e.g. from "For today: Complete the onboarding form
is due on 2026-09-10." -> "Complete the onboarding form is due on
2026-09-10.").
"""
import re

_NOISE_PREFIXES = [
    "For today:", "FYI:", "One more thing:", "Hi,", "Important:",
    "Just checking—", "Just checking-", "Please note:", "Quick update:",
    "Can you help?",
]


def strip_noise_prefixes(message: str) -> str:
    text = message.strip()
    changed = True
    while changed:
        changed = False
        for p in _NOISE_PREFIXES:
            if text.startswith(p):
                text = text[len(p):].strip()
                changed = True
    return text
