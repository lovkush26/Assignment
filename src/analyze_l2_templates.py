"""
Frequency analysis over the L2 message batch (l2_messages.csv + l2_demo_messages.csv),
same methodology as analyze_templates.py used for the original L1 dataset: strip the
known noise-prefix openers, normalize dates/times/numbers, and count the distinct
underlying sentence shapes. Used to design the L2 rule extensions in prefixes.py,
classify.py, extract.py, and sensitive.py so every template is covered by design,
not guessed.
"""
import re
from collections import Counter

import pandas as pd

PREFIXES = [
    "For today:", "FYI:", "One more thing:", "Hi,", "Important:",
    "Just checking—", "Just checking-", "Please note:", "Quick update:",
    "Can you help?", "If possible,",
    "Follow-up:", "Additional update:", "Update:",
]


def strip_prefixes(msg):
    changed = True
    while changed:
        changed = False
        m2 = msg.strip()
        for p in PREFIXES:
            if m2.startswith(p):
                m2 = m2[len(p):].strip()
                changed = True
        msg = m2
    return msg


def normalize(s):
    s = re.sub(r"\d{4}-\d{2}-\d{2}", "<DATE>", s)
    s = re.sub(r"\b\d{1,2}:\d{2}\b", "<TIME>", s)
    s = re.sub(r"[A-Za-z0-9_#\-]*\d[A-Za-z0-9_#\-]*", "<VAL>", s)
    return s


for path in ["data/l2_messages.csv", "data/l2_demo_messages.csv"]:
    df = pd.read_csv(path)
    df["core"] = df["message"].apply(strip_prefixes)
    df["template"] = df["core"].apply(normalize)
    counts = Counter(df["template"])
    print(f"=== {path}: {len(df)} messages, {len(counts)} unique templates ===")
    for tpl, c in counts.most_common(200):
        print(c, "|", tpl)
    print()
