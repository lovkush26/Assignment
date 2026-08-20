"""
Runs every query in data/l2_demo_queries.csv against the assistant, after
processing L1 + L2 + the demo batch as three sequential batches (the demo
batch loaded last, as a "new unseen batch" per L2_Candidate_Dataset/README.txt).

For the video: run this once on screen to show all 8 mandatory query answers
at a glance, then go back into the Streamlit app to narrate 1-2 in depth.

Usage: python src/run_demo_queries.py
"""
import csv

import pandas as pd

from pipeline import process_full, DATA_CSV, L2_CSV, L2_DEMO_CSV, ROOT
from search import Assistant

df = pd.concat([pd.read_csv(DATA_CSV), pd.read_csv(L2_CSV), pd.read_csv(L2_DEMO_CSV)], ignore_index=True)
result = process_full(df)
assistant = Assistant(
    result["messages_df"], result["classifications"], result["items"], result["sensitive_records"],
    result["priority_records"], result["groups"], result["routing_records"],
)

with open(ROOT / "data" / "l2_demo_queries.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        ans = assistant.answer(row["query"])
        print(f"\n{row['query_id']}: {row['query']}")
        print(f"  Answer: {ans['answer']}")
        print(f"  Reason: {ans['reason']}")
