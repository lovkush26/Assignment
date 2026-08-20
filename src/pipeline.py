"""
End-to-end pipeline.

L1 (unchanged): classification (Part 1), task/event extraction (Part 2), and
sensitive-info detection (Part 3) over a dataframe of messages —
`process_dataframe()`.

L2 additions: `process_full()` runs those three plus the new priority engine
(priority.py), related-message grouping (grouping.py), and privacy-aware
routing (routing.py) over a combined, chronologically-sorted L1+L2
dataframe. `run_pipeline()` is the CLI entry point: it reads
data/messages.csv (L1, 900) + data/l2_messages.csv (L2, 180) — and
optionally data/l2_demo_messages.csv (24) for the recorded demo — and writes
every output file used across the assignment.

Item IDs continue across L1 and L2 automatically: process_dataframe assigns
TASK_/EVENT_ ids by iterating the dataframe in timestamp order, and L1's
timestamps (2026-09) are all earlier than L2's (2026-09-26 onward), so
concatenating the two CSVs and sorting once is sufficient — no separate
counter-continuation bookkeeping needed.

Runs entirely locally: pandas/scikit-learn for tabular + TF-IDF handling,
the stdlib `re`/`datetime` for the rule logic in classify.py / extract.py /
sensitive.py / l2_patterns.py. No network calls, no external AI service.
"""
import json
import csv
from pathlib import Path

import pandas as pd

from classify import classify_message
from extract import extract_item
from sensitive import detect_sensitive, mask_message
from grouping import build_groups
from priority import compute_priorities
from routing import route_messages

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "messages.csv"
L2_CSV = ROOT / "data" / "l2_messages.csv"
L2_DEMO_CSV = ROOT / "data" / "l2_demo_messages.csv"
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


def process_full(df: pd.DataFrame) -> dict:
    """L2 orchestration: L1's three parts, plus priority/grouping/routing.
    Returns every registry the Streamlit app / search assistant needs."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    classifications, items, sensitive_records = process_dataframe(df)
    sensitive_ids = {s["message_id"] for s in sensitive_records}

    groups, unresolved_references, groups_by_id = build_groups(df, items, sensitive_ids)
    priority_records = compute_priorities(groups_by_id)
    routing_records = route_messages(df, sensitive_records)

    return {
        "messages_df": df,
        "classifications": classifications,
        "items": items,
        "sensitive_records": sensitive_records,
        "groups": groups,
        "unresolved_references": unresolved_references,
        "priority_records": priority_records,
        "routing_records": routing_records,
    }


def run_pipeline(include_demo: bool = False) -> dict:
    """CLI entry point: read data/messages.csv + data/l2_messages.csv (and
    optionally data/l2_demo_messages.csv), write every outputs/* file."""
    frames = [pd.read_csv(DATA_CSV), pd.read_csv(L2_CSV)]
    if include_demo:
        frames.append(pd.read_csv(L2_DEMO_CSV))
    df = pd.concat(frames, ignore_index=True)

    result = process_full(df)
    _write_all_outputs(result)
    _print_summary(df, result)
    return result


def _print_summary(df, result):
    items = result["items"]
    tasks = sum(1 for i in items if i["type"] == "task")
    events = sum(1 for i in items if i["type"] == "event")
    print(f"messages: {len(df)}")
    print(f"classifications: {len(result['classifications'])}")
    print(f"extracted tasks/events: {len(items)}  (tasks={tasks}, events={events})")
    print(f"sensitive findings: {len(result['sensitive_records'])}")
    print(f"related-message groups: {len(result['groups'])} "
          f"(multi-message: {sum(1 for g in result['groups'] if len(g['related_message_ids']) > 1)})")
    print(f"unresolved references (no confident group match): {len(result['unresolved_references'])}")
    print(f"priority decisions: {len(result['priority_records'])}")
    routes = result["routing_records"]
    print(f"privacy routing: local={sum(1 for r in routes if r['route']=='process_locally')} "
          f"confirm={sum(1 for r in routes if r['route']=='ask_for_confirmation')} "
          f"blocked={sum(1 for r in routes if r['route']=='blocked')}")


def _write_all_outputs(result: dict):
    OUT_DIR.mkdir(exist_ok=True)

    _write_json(OUT_DIR / "classifications.json", result["classifications"])
    _write_json(OUT_DIR / "tasks_events.json", result["items"])
    _write_json(OUT_DIR / "sensitive_findings.json", result["sensitive_records"])
    _write_json(OUT_DIR / "priority.json", result["priority_records"])
    _write_json(OUT_DIR / "related_message_groups.json", result["groups"])
    _write_json(OUT_DIR / "privacy_routing.json", result["routing_records"])
    _write_json(OUT_DIR / "unresolved_references.json", result["unresolved_references"])

    _write_csv(OUT_DIR / "classifications.csv", result["classifications"],
               ["message_id", "category", "confidence", "reason"])
    _write_csv(OUT_DIR / "tasks_events.csv", result["items"],
               ["item_id", "type", "title", "description", "deadline", "time",
                "person", "priority", "source_message_id"])
    _write_csv(OUT_DIR / "sensitive_findings.csv", result["sensitive_records"],
               ["message_id", "sensitivity_type", "risk", "masked_text",
                "recommended_action", "reason"])
    _write_csv(OUT_DIR / "priority.csv", result["priority_records"],
               ["message_id", "item_id", "group_id", "priority", "reason", "signals", "confidence"],
               list_fields=["signals"])
    _write_csv(OUT_DIR / "related_message_groups.csv", result["groups"],
               ["group_id", "title", "related_message_ids", "related_item_ids", "status",
                "latest_deadline", "summary", "confidence", "conflicting_deadlines"],
               list_fields=["related_message_ids", "related_item_ids"], dict_list_fields=["conflicting_deadlines"])
    _write_csv(OUT_DIR / "privacy_routing.csv", result["routing_records"],
               ["message_id", "route", "risk", "sensitivity_types", "recommended_action", "reason"],
               list_fields=["sensitivity_types"])


def _write_json(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, records: list, fieldnames: list, list_fields=None, dict_list_fields=None):
    list_fields = list_fields or []
    dict_list_fields = dict_list_fields or []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = dict(r)
            for lf in list_fields:
                row[lf] = "; ".join(str(v) for v in row.get(lf, []))
            for df_ in dict_list_fields:
                row[df_] = "; ".join(f'{d.get("date")}({d.get("message_id")})' for d in row.get(df_, []))
            writer.writerow(row)


if __name__ == "__main__":
    run_pipeline(include_demo=False)
