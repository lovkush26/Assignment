"""
Related-Message Grouping (L2 Part 2).

Links messages that refer to the same task/event/subject across time, using
two tiers of matching — literal first, meaning-based fallback second — so
every link stays explainable:

1. **Exact subject match** (high confidence): every extracted task/event
   (from extract.py, including the L2 new-task/new-event signals) seeds a
   group keyed by `textnorm.normalize_subject(title)`. Any later message
   whose L2 signal (l2_patterns.detect_l2_signal) names that same normalized
   subject — a follow-up, completion, cancellation, reschedule, deadline
   change, etc. — joins that group and updates its tracked status/deadline.
   Two items that literally share a subject (e.g. the dataset's "prepare the
   offline inference demo" is announced once as an event and once as a task)
   are merged into one group rather than kept as duplicates.

2. **TF-IDF fallback** (lower confidence, via retrieval.py): when a
   message's subject phrase doesn't exactly match any known title — e.g.
   "our earlier model-results review" vs. the task titled "Review the model
   results" — cosine similarity over title/description text finds the best
   match above a threshold. Below the threshold, the message is left
   unlinked rather than guessed into a group (the assignment requires "must
   not invent" — an unresolved reference is reported as such, not forced
   into a group).

Group status follows "latest message wins": each linked message's L2 signal
can move status forward (completed/cancelled/rescheduled) or back to
"unclear" (an ambiguous/hedged later message, or a newly-stated conflicting
deadline). This is what lets Part 1's priority engine react to "a later
message changes the deadline or status" without re-deriving it itself.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from l2_patterns import (
    detect_l2_signal,
    SIGNAL_STATUS_CHECK, SIGNAL_DUPLICATE_STATUS_CHECK, SIGNAL_COMPLETED,
    SIGNAL_CANCELLED, SIGNAL_EVENT_CANCELLED, SIGNAL_RESCHEDULED,
    SIGNAL_DEADLINE_URGENT_EARLIER, SIGNAL_DEADLINE_CONFLICT,
    SIGNAL_DEADLINE_EXTENDED, SIGNAL_AMBIGUOUS, SIGNAL_NEW_TASK, SIGNAL_NEW_EVENT,
)
from prefixes import strip_noise_prefixes
from retrieval import Corpus
from textnorm import normalize_subject

FUZZY_MIN_SCORE = 0.35

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_RESCHEDULED = "rescheduled"
STATUS_CANCELLED = "cancelled"
STATUS_UNCLEAR = "unclear"

_TERMINAL_STATUSES = {STATUS_COMPLETED, STATUS_CANCELLED}


@dataclass
class ConflictRecord:
    message_id: str
    date: str
    reason: str


@dataclass
class Group:
    group_id: str
    title: str
    item_type: str  # "task" | "event" | "mixed"
    related_message_ids: List[str] = field(default_factory=list)
    related_item_ids: List[str] = field(default_factory=list)
    status: str = STATUS_PENDING
    latest_deadline: Optional[str] = None
    latest_time: Optional[str] = None
    conflicts: List[ConflictRecord] = field(default_factory=list)
    link_confidences: List[float] = field(default_factory=list)
    ever_urgent: bool = False
    ever_ambiguous: bool = False
    last_signal_type: Optional[str] = None
    last_signal_message_id: Optional[str] = None
    history: List[dict] = field(default_factory=list)  # per-message snapshots, consumed by priority.py

    def to_record(self) -> dict:
        confidence = round(min(self.link_confidences), 2) if self.link_confidences else 0.5
        return {
            "group_id": self.group_id,
            "title": self.title,
            "related_message_ids": self.related_message_ids,
            "related_item_ids": self.related_item_ids,
            "status": self.status,
            "latest_deadline": self.latest_deadline,
            "summary": _summarize(self),
            "confidence": confidence,
            "conflicting_deadlines": [
                {"message_id": c.message_id, "date": c.date, "reason": c.reason} for c in self.conflicts
            ],
            "ever_urgent": self.ever_urgent,
            "ever_ambiguous": self.ever_ambiguous,
            "last_signal_type": self.last_signal_type,
            "last_signal_message_id": self.last_signal_message_id,
        }


def _summarize(g: Group) -> str:
    kind = {"task": "Task", "event": "Event", "mixed": "Item"}[g.item_type]
    n_msgs = len(g.related_message_ids)
    parts = [f'{kind} "{g.title}" was raised in {g.related_message_ids[0]}']
    followups = n_msgs - 1
    if followups > 0:
        parts.append(f"referenced again in {followups} later message(s)")
    if g.conflicts:
        dates = ", ".join(f"{c.date} ({c.message_id})" for c in g.conflicts)
        parts.append(f"conflicting deadline(s) were also stated: {dates}")
    if g.latest_deadline and g.latest_deadline != "unresolved":
        parts.append(f"latest known deadline is {g.latest_deadline}")
    summary = ", ".join(parts) + f". Current status: {g.status}."
    return summary[0].upper() + summary[1:]


# Signal types that still leave something outstanding for the recipient
# (used as the "response_required" priority signal). Pure status reports
# (completed/cancelled) need no further response.
_RESPONSE_REQUIRED_SIGNALS = {
    SIGNAL_STATUS_CHECK, SIGNAL_DUPLICATE_STATUS_CHECK, SIGNAL_NEW_TASK,
    SIGNAL_NEW_EVENT, SIGNAL_DEADLINE_URGENT_EARLIER, SIGNAL_DEADLINE_CONFLICT,
    SIGNAL_DEADLINE_EXTENDED, SIGNAL_RESCHEDULED,
}


def _snapshot(g: Group, msg_id: str, meta: dict, response_required: bool, sensitive_ids: set) -> None:
    g.history.append({
        "message_id": msg_id,
        "timestamp": meta["timestamp"],
        "sender": meta["sender"],
        "group_id": g.group_id,
        "primary_item_id": g.related_item_ids[0],
        "item_type": g.item_type,
        "status": g.status,
        "deadline": g.latest_deadline,
        "urgent": g.ever_urgent,
        "ambiguous": g.status == STATUS_UNCLEAR,
        "conflict_count": len(g.conflicts),
        "response_required": response_required,
        "is_sensitive": msg_id in sensitive_ids,
    })


def build_groups(messages_df, items: List[dict], sensitive_ids: Optional[set] = None):
    """messages_df: DataFrame sorted chronologically with message_id/timestamp/
    sender/message. items: the Part-2 task/event list from extract.py (as
    produced by pipeline.process_dataframe), already in creation order.
    sensitive_ids: set of message_ids that Part 3 flagged as sensitive
    (used only as a minor priority signal, see priority.py).

    Returns (group_records, unresolved_references, groups_by_id) — the last
    is the raw Group objects (with per-message .history) that priority.py
    needs; group_records is the public, JSON-serializable list.
    """
    sensitive_ids = sensitive_ids or set()
    messages_by_id = {
        row["message_id"]: {"timestamp": row["timestamp"], "sender": row["sender"]}
        for _, row in messages_df.iterrows()
    }

    items_by_id = {it["item_id"]: it for it in items}

    groups: Dict[str, Group] = {}
    subject_to_group: Dict[str, str] = {}
    item_to_group: Dict[str, str] = {}
    group_counter = 0

    def new_group_id():
        nonlocal group_counter
        group_counter += 1
        return f"GROUP_{group_counter:03d}"

    # Seed one group per item, merging items that share a normalized subject
    # (e.g. the same real-world thing announced once as an event, once as a
    # task) into a single group instead of duplicating.
    for it in items:
        key = normalize_subject(it["title"])
        if key in subject_to_group:
            gid = subject_to_group[key]
            g = groups[gid]
            g.related_item_ids.append(it["item_id"])
            g.related_message_ids.append(it["source_message_id"])
            g.link_confidences.append(1.0)
            if g.item_type != it["type"]:
                g.item_type = "mixed"
        else:
            gid = new_group_id()
            status = STATUS_UNCLEAR if it["deadline"] == "unresolved" else STATUS_PENDING
            g = Group(
                group_id=gid,
                title=it["title"],
                item_type=it["type"],
                related_message_ids=[it["source_message_id"]],
                related_item_ids=[it["item_id"]],
                status=status,
                latest_deadline=it["deadline"] if it["deadline"] != "unresolved" else None,
                latest_time=it["time"] if it["time"] != "unresolved" else None,
                link_confidences=[1.0],
            )
            groups[gid] = g
            subject_to_group[key] = gid
        item_to_group[it["item_id"]] = gid
        _snapshot(g, it["source_message_id"], messages_by_id[it["source_message_id"]],
                  response_required=True, sensitive_ids=sensitive_ids)

    origin_message_ids = {it["source_message_id"] for it in items}

    # Fuzzy fallback corpus: one "document" per group, built from its title
    # (the titles already read like short natural-language phrases).
    fuzzy_corpus = Corpus(ids=list(groups.keys()), texts=[g.title for g in groups.values()])

    unresolved_references = []  # messages whose L2 signal named a subject we
    # could not confidently link to anything — surfaced for the assistant's
    # "insufficient evidence" answers and the benchmark/limitations notes.

    for _, row in messages_df.iterrows():
        msg_id = row["message_id"]
        if msg_id in origin_message_ids:
            continue  # already seeded its group above

        core = strip_noise_prefixes(row["message"])
        l2 = detect_l2_signal(core)
        if l2 is None:
            continue
        if l2.signal_type in (SIGNAL_NEW_TASK, SIGNAL_NEW_EVENT):
            continue  # handled at seeding time

        gid = None
        confidence = l2.confidence
        if l2.subject_key:
            gid = subject_to_group.get(l2.subject_key)
            if gid is None:
                match = fuzzy_corpus.best_match(l2.subject_title or "", min_score=FUZZY_MIN_SCORE)
                if match:
                    gid, score = match
                    confidence = round(l2.confidence * score, 2)

        if gid is None:
            unresolved_references.append({
                "message_id": msg_id,
                "subject": l2.subject_title,
                "signal_type": l2.signal_type,
                "reason": "No task/event with a matching or similar-enough title was found; not linked to avoid "
                          "inventing a connection.",
            })
            continue

        g = groups[gid]
        g.related_message_ids.append(msg_id)
        g.link_confidences.append(confidence)
        _apply_signal(g, l2, msg_id)
        _snapshot(g, msg_id, messages_by_id[msg_id],
                  response_required=l2.signal_type in _RESPONSE_REQUIRED_SIGNALS and g.status not in _TERMINAL_STATUSES,
                  sensitive_ids=sensitive_ids)

    ordered_groups = sorted(groups.values(), key=lambda g: g.group_id)
    result = [g.to_record() for g in ordered_groups]
    return result, unresolved_references, groups


def _apply_signal(g: Group, l2, msg_id: str) -> None:
    st = l2.signal_type
    g.last_signal_type = st
    g.last_signal_message_id = msg_id
    if l2.urgent:
        g.ever_urgent = True
    if st == SIGNAL_AMBIGUOUS:
        g.ever_ambiguous = True
    if st in (SIGNAL_STATUS_CHECK, SIGNAL_DUPLICATE_STATUS_CHECK):
        # A status-check doesn't state a new outcome, but it does mean someone is
        # actively tracking/following up on this item — the first such check on an
        # otherwise-untouched item moves it from "pending" (nothing yet asked) to
        # "in_progress" (now being actively followed up on), matching the assignment's
        # documented status vocabulary. Anything already resolved/rescheduled/unclear
        # is left as-is — a status-check doesn't undo a real outcome.
        if g.status == STATUS_PENDING:
            g.status = STATUS_IN_PROGRESS
        return
    if st == SIGNAL_COMPLETED:
        g.status = STATUS_COMPLETED
    elif st in (SIGNAL_CANCELLED, SIGNAL_EVENT_CANCELLED):
        g.status = STATUS_CANCELLED
    elif st == SIGNAL_RESCHEDULED:
        if l2.date == "unresolved":
            g.status = STATUS_UNCLEAR
        elif g.status not in _TERMINAL_STATUSES:
            g.status = STATUS_RESCHEDULED
        if l2.date and l2.date != "unresolved":
            g.latest_deadline = l2.date
        if l2.time:
            g.latest_time = l2.time
    elif st == SIGNAL_DEADLINE_URGENT_EARLIER:
        if l2.date and l2.date != "unresolved" and l2.date != g.latest_deadline:
            if g.latest_deadline:
                g.conflicts.append(ConflictRecord(msg_id, l2.date, "Stated as a new, earlier, urgent deadline."))
            g.latest_deadline = l2.date
    elif st == SIGNAL_DEADLINE_CONFLICT:
        if l2.date and l2.date != g.latest_deadline:
            g.conflicts.append(ConflictRecord(
                msg_id, l2.date, 'Message explicitly notes "the earlier message listed another date".'))
            g.latest_deadline = l2.date
            if g.status not in _TERMINAL_STATUSES:
                g.status = STATUS_UNCLEAR
    elif st == SIGNAL_DEADLINE_EXTENDED:
        g.latest_deadline = l2.date
    elif st == SIGNAL_AMBIGUOUS:
        if g.status not in _TERMINAL_STATUSES:
            g.status = STATUS_UNCLEAR
