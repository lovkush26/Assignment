"""
Priority and Action Engine (L2 Part 1).

Assigns Critical/High/Medium/Low to every actionable message using a small
weighted-signal score, not a single keyword or a random choice — mirroring
the "stated, reproducible heuristic" philosophy L1 already used for its
day-gap task priority (see extract.py's _priority_from_dates / README).

One priority record is produced per (message, item) touch-point, using
grouping.py's per-message history snapshots — i.e. every time a message
creates or updates a tracked task/event, its priority is (re)computed from
the state as of that message's own timestamp. This is what lets priority
change over time: the same item can score "medium" when first requested and
later score "critical" once an urgent earlier-deadline message, a conflicting
deadline, or simple overdue time passing pushes its score up (see DQ01-style
"which task became critical" queries — the answer is: whichever item's
history shows a low/medium record followed by a critical one).

Score inputs (all literally present in the message/group state, nothing
inferred):
  - deadline proximity as of the message's own timestamp (overdue > due
    today > within 2 days > within a week > further out > unresolved)
  - explicit urgency language ("Treat this as urgent", "This is urgent")
  - a conflicting deadline having just been reported
  - status being "unclear" (ambiguous/hedged latest information)
  - whether the message still requires a response (vs. a pure status report
    like a completion/cancellation, which is always scored "low")
  - sender: a small bump for senders who assign work with organizational
    authority in this dataset (Project Lead, HR Team, Mentor)
  - the message overlapping a detected sensitive-information finding
  - message category: a scheduled event (meeting_or_event) close to its time
    gets a small bump over a task at the same deadline proximity — a missed
    meeting slot can't be caught up on later the way a late task can
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

PRIORITY_CRITICAL = "critical"
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"

_AUTHORITATIVE_SENDERS = {"Project Lead", "HR Team", "Mentor"}
_TERMINAL_STATUSES = {"completed", "cancelled"}


def _deadline_signal(deadline: Optional[str], as_of: datetime):
    """Returns (score, signal_name) for how close/overdue a concrete
    deadline is relative to the message's own timestamp. None/unresolved
    deadlines score low (can't judge urgency without a date — same rule L1
    used for vague dates in extract.py)."""
    if not deadline or deadline == "unresolved":
        return 1, "deadline_unresolved_or_unknown"
    try:
        deadline_date = datetime.strptime(deadline, "%Y-%m-%d").date()
    except ValueError:
        return 1, "deadline_unresolved_or_unknown"
    days = (deadline_date - as_of.date()).days
    if days < 0:
        return 4, "overdue"
    if days == 0:
        return 4, "deadline_today"
    if days <= 2:
        return 3, "deadline_within_2_days"
    if days <= 7:
        return 2, "deadline_within_a_week"
    return 1, "deadline_more_than_a_week_out"


def _score_snapshot(snap: dict) -> dict:
    as_of = datetime.strptime(snap["timestamp"], "%Y-%m-%d %H:%M:%S")

    if snap["status"] in _TERMINAL_STATUSES:
        return {
            "priority": PRIORITY_LOW,
            "reason": f'Item status is "{snap["status"]}" as of this message — no further action is needed.',
            "signals": [f'status_{snap["status"]}'],
            "confidence": 0.9,
        }

    score = 0
    signals = []

    d_score, d_signal = _deadline_signal(snap["deadline"], as_of)
    score += d_score
    signals.append(d_signal)

    if snap["urgent"]:
        score += 3
        signals.append("explicitly_marked_urgent")

    if snap["conflict_count"] > 0:
        score += 2
        signals.append("conflicting_deadline_reported")

    if snap["ambiguous"]:
        score += 1
        signals.append("status_unclear")

    if snap["response_required"]:
        score += 1
        signals.append("response_required")

    if snap["sender"] in _AUTHORITATIVE_SENDERS:
        score += 1
        signals.append(f'assigned_by_{snap["sender"].replace(" ", "_").lower()}')

    if snap["is_sensitive"]:
        score += 1
        signals.append("contains_sensitive_information")

    # Message category: a scheduled event (meeting_or_event) close to its time is
    # time-sensitive in a way a task deadline isn't — you can still finish a task late,
    # but a missed meeting slot is simply missed. Only applies once the event is
    # already close (today/overdue/within 2 days), not to every event outright.
    if snap["item_type"] == "event" and d_signal in ("deadline_today", "overdue", "deadline_within_2_days"):
        score += 1
        signals.append("time_sensitive_meeting_category")

    if score >= 7:
        priority = PRIORITY_CRITICAL
    elif score >= 5:
        priority = PRIORITY_HIGH
    elif score >= 3:
        priority = PRIORITY_MEDIUM
    else:
        priority = PRIORITY_LOW

    reason_bits = []
    if "overdue" in signals:
        reason_bits.append("the deadline has already passed")
    elif "deadline_today" in signals:
        reason_bits.append("the deadline is today")
    elif "deadline_within_2_days" in signals:
        reason_bits.append("the deadline is within 2 days")
    elif "deadline_within_a_week" in signals:
        reason_bits.append("the deadline is within a week")
    elif "deadline_more_than_a_week_out" in signals:
        reason_bits.append("the deadline is more than a week out")
    else:
        reason_bits.append("no concrete deadline is known")
    if "explicitly_marked_urgent" in signals:
        reason_bits.append("a message explicitly marked it urgent")
    if "conflicting_deadline_reported" in signals:
        reason_bits.append("a conflicting deadline was reported")
    if "status_unclear" in signals:
        reason_bits.append("the latest status is ambiguous")
    if snap["sender"] in _AUTHORITATIVE_SENDERS:
        reason_bits.append(f'it was assigned by {snap["sender"]}')
    if snap["is_sensitive"]:
        reason_bits.append("the message also contains sensitive information")
    if "time_sensitive_meeting_category" in signals:
        reason_bits.append("it is a scheduled meeting/event, not a task, so a missed slot can't be caught up on later")

    reason = "; ".join(reason_bits).capitalize() + "."

    confidence = 0.55 + 0.06 * len(signals)
    if snap["ambiguous"]:
        confidence -= 0.15
    if "deadline_unresolved_or_unknown" in signals:
        confidence -= 0.1
    confidence = round(max(0.4, min(confidence, 0.97)), 2)

    return {"priority": priority, "reason": reason, "signals": signals, "confidence": confidence}


def compute_priorities(groups_by_id: Dict[str, "grouping.Group"]) -> List[dict]:
    """One record per (message_id, item_id) touch-point across every group's
    history, in chronological arrival order within each group."""
    records = []
    for g in groups_by_id.values():
        for snap in g.history:
            scored = _score_snapshot(snap)
            records.append({
                "message_id": snap["message_id"],
                "item_id": snap["primary_item_id"],
                "group_id": snap["group_id"],
                "priority": scored["priority"],
                "reason": scored["reason"],
                "signals": scored["signals"],
                "confidence": scored["confidence"],
            })
    return records


def current_priorities(priority_records: List[dict]) -> Dict[str, dict]:
    """Reduce the full history down to each item's latest (current) priority
    record — used by the assistant for "what's pending right now"-style
    queries, keeping the full history available separately for "why"/"when
    did this change" questions."""
    latest: Dict[str, dict] = {}
    for rec in priority_records:
        latest[rec["item_id"]] = rec  # records arrive in chronological order per group
    return latest
