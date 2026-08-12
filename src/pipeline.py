"""
End-to-end pipeline: classification (Part 1), task/event extraction (Part 2)
and sensitive-info detection (Part 3) over a dataframe of messages.

process_dataframe() is the single shared entry point used by both the CLI
script (`python src/pipeline.py`, reads data/messages.csv, writes outputs/)
and the Streamlit app (app.py, processes an uploaded CSV in-memory) so both
surfaces run the exact same logic.

Runs entirely locally: pandas for tabular handling, the stdlib `re`/
`datetime` for all logic in classify.py / extract.py / sensitive.py. No
network calls, no external AI services — required by the assignment rules.
"""
import json
import csv
from pathlib import Path

import pandas as pd

from classify import classify_message
from extract import extract_item
from sensitive import detect_sensitive, mask_message

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "messages.csv"
OUT_DIR = ROOT / "outputs"


def process_dataframe(df: pd.DataFrame):
    """Run classification + extraction + sensitive-detection over df.

    df must have columns: message_id, timestamp, sender, message.
    Returns (classifications, items, sensitive_records) as lists of dicts.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)  # chronological order

    classifications = []
    items = []
    sensitive_records = []

    task_counter = 0
    event_counter = 0

    for _, row in df.iterrows():
        msg_id = row["message_id"]
        sender = row["sender"]
        message = row["message"]
        timestamp = row["timestamp"]

        clf = classify_message(sender, message)
        classifications.append({
            "message_id": msg_id,
            "category": clf.category,
            "confidence": round(clf.confidence, 2),
            "reason": clf.reason,
        })

        item = extract_item(clf.category, message, timestamp, msg_id)
        if item:
            if item.type == "task":
                task_counter += 1
                item_id = f"TASK_{task_counter:03d}"
            else:
                event_counter += 1
                item_id = f"EVENT_{event_counter:03d}"
            items.append({
                "item_id": item_id,
                "type": item.type,
                "title": item.title,
                "description": item.description,
                "deadline": item.deadline,
                "time": item.time,
                "person": item.person,
                "priority": item.priority,
                "source_message_id": item.source_message_id,
            })

        matches = detect_sensitive(message)
        if matches:
            masked = mask_message(message, matches)
            for m in matches:
                sensitive_records.append({
                    "message_id": msg_id,
                    "sensitivity_type": m.sensitivity_type,
                    "risk": m.risk,
                    "masked_text": masked,
                    "recommended_action": m.recommended_action,
                    "reason": m.reason,
                })

    return classifications, items, sensitive_records


def run_pipeline():
    """CLI entry point: read data/messages.csv, write outputs/*.json + *.csv."""
    df = pd.read_csv(DATA_CSV)
    classifications, items, sensitive_records = process_dataframe(df)

    OUT_DIR.mkdir(exist_ok=True)

    _write_json(OUT_DIR / "classifications.json", classifications)
    _write_json(OUT_DIR / "tasks_events.json", items)
    _write_json(OUT_DIR / "sensitive_findings.json", sensitive_records)

    _write_csv(OUT_DIR / "classifications.csv", classifications,
               ["message_id", "category", "confidence", "reason"])
    _write_csv(OUT_DIR / "tasks_events.csv", items,
               ["item_id", "type", "title", "description", "deadline", "time",
                "person", "priority", "source_message_id"])
    _write_csv(OUT_DIR / "sensitive_findings.csv", sensitive_records,
               ["message_id", "sensitivity_type", "risk", "masked_text",
                "recommended_action", "reason"])

    print(f"messages: {len(df)}")
    print(f"classifications: {len(classifications)}")
    tasks = sum(1 for i in items if i["type"] == "task")
    events = sum(1 for i in items if i["type"] == "event")
    print(f"extracted tasks/events: {len(items)}  (tasks={tasks}, events={events})")
    print(f"sensitive findings: {len(sensitive_records)}")

    return classifications, items, sensitive_records


def _write_json(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, records: list, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


if __name__ == "__main__":
    run_pipeline()
