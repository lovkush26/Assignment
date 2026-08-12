import pandas as pd
import re
from collections import Counter

df = pd.read_csv("data/messages.csv")

PREFIXES = [
    "For today:", "FYI:", "One more thing:", "Hi,", "Important:",
    "Just checking—", "Just checking-", "Please note:", "Quick update:",
    "Can you help?", "If possible,",
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

df['core'] = df['message'].apply(strip_prefixes)

# normalize dates/times/numbers to find template shape
def normalize(s):
    s = re.sub(r'\d{4}-\d{2}-\d{2}', '<DATE>', s)
    s = re.sub(r'\b\d{1,2}(:\d{2})?\s?(AM|PM|am|pm)?\b', '<NUM>', s)
    s = re.sub(r'\d+', '<NUM>', s)
    return s

df['template'] = df['core'].apply(normalize)
counts = Counter(df['template'])
print("unique templates:", len(counts))
for tpl, c in counts.most_common(120):
    print(c, "|", tpl)
