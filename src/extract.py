"""
Task and event extraction (Part 2).

Only messages already classified as action_required or meeting_or_event are
considered. Extraction uses regex templates tailored to the small set of
recurring sentence structures found in the dataset (verified by frequency
analysis over all 900 messages, see src/analyze_templates.py). Each template captures
exactly the fields it names; nothing is inferred beyond what is written in
the message text (per the assignment's "do not guess" rule).

When a date/time is present but only as a vague/relative phrase ("tomorrow",
"next week", "soon", "Friday afternoon") it is stored as the string
"unresolved" (not a guessed absolute date) together with the raw phrase kept
in the description, and priority defaults to "low" since urgency can't be
judged without a concrete date.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from prefixes import strip_noise_prefixes

KNOWN_NAMES = [
    "Meera", "Ishaan", "Kabir", "Aarav", "Ananya",
    "Neha", "Vikram", "Tara", "Rohan", "Maya",
]
NAME_RE = re.compile(r"\b(" + "|".join(KNOWN_NAMES) + r")\b")

DATE_RE = r"\d{4}-\d{2}-\d{2}"
TIME_RE = r"\d{1,2}:\d{2}"


@dataclass
class ExtractedItem:
    type: str  # "task" | "event"
    title: str
    description: str
    deadline: Optional[str]
    time: Optional[str]
    person: Optional[str]
    priority: str
    source_message_id: str


def _find_person(text: str) -> Optional[str]:
    m = NAME_RE.search(text)
    return m.group(1) if m else None


def _priority_from_dates(msg_date: datetime, deadline_str: Optional[str]) -> str:
    if not deadline_str or deadline_str == "unresolved":
        return "low"
    deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d")
    delta = (deadline_date.date() - msg_date.date()).days
    if delta <= 2:
        return "high"
    if delta <= 7:
        return "medium"
    return "low"


# --- Action Required (task) templates -------------------------------------

_TASK_PATTERNS = [
    re.compile(rf"Don't forget to (?P<action>.+?); deadline is (?P<date>{DATE_RE})\."),
    re.compile(rf"I need you to (?P<action>.+?) by (?P<date>{DATE_RE})\."),
    re.compile(rf"Can you (?P<action>.+?) before (?P<date>{DATE_RE})\??"),
    re.compile(rf"Please (?P<action>(?:submit|reply|confirm|complete)[^.]*?) by (?P<date>{DATE_RE})\."),
    re.compile(rf"(?P<action>.+?) is due on (?P<date>{DATE_RE})\."),
]

_TASK_VAGUE_PATTERNS = [
    (re.compile(r"Please call (?P<person>\w+) when you are free\."),
     lambda m: ("Call " + m.group("person"), None, None, m.group("person"))),
    (re.compile(r"Could you send it soon\?"),
     lambda m: ("Send requested item (unspecified)", "unresolved", None, None)),
    (re.compile(r"review the file before the meeting\."),
     lambda m: ("Review the file", "unresolved", None, None)),
    (re.compile(r"The report may be needed tomorrow\."),
     lambda m: ("Prepare the report", "unresolved", None, None)),
]


def _extract_task(message: str, msg_date: datetime, msg_id: str) -> Optional[ExtractedItem]:
    for pat in _TASK_PATTERNS:
        m = pat.search(message)
        if m:
            action = m.group("action").strip()
            date_str = m.group("date")
            title = action[0].upper() + action[1:]
            person = _find_person(action)
            return ExtractedItem(
                type="task",
                title=title,
                description=f"Action requested: {action}.",
                deadline=date_str,
                time=None,
                person=person,
                priority=_priority_from_dates(msg_date, date_str),
                source_message_id=msg_id,
            )
    for pat, builder in _TASK_VAGUE_PATTERNS:
        m = pat.search(message)
        if m:
            title, deadline, time_, person = builder(m)
            desc = f"Extracted from: \"{m.group(0)}\""
            if deadline == "unresolved":
                desc += " — deadline phrase is relative/vague and was not resolved to an exact date."
            return ExtractedItem(
                type="task",
                title=title,
                description=desc,
                deadline=deadline,
                time=time_,
                person=person,
                priority=_priority_from_dates(msg_date, deadline),
                source_message_id=msg_id,
            )
    return None


# --- Meeting or Event templates --------------------------------------------

_EVENT_PATTERNS = [
    # "Calendar update: <event>, <date> at <time>, <location>."
    re.compile(
        rf"Calendar update:\s*(?P<event>.+?),\s*(?P<date>{DATE_RE})\s+at\s+(?P<time>{TIME_RE}),\s*(?P<location>[^.]+)\."
    ),
    # "Reminder: <event> happens on <date> at <time> in <location>."
    re.compile(
        rf"Reminder:\s*(?P<event>.+?)\s+happens on\s+(?P<date>{DATE_RE})\s+at\s+(?P<time>{TIME_RE})\s+in\s+(?P<location>[^.]+)\."
    ),
    # "Please join the <event> on <date>, <time> at <location>."
    re.compile(
        rf"Please join the\s+(?P<event>.+?)\s+on\s+(?P<date>{DATE_RE}),\s*(?P<time>{TIME_RE})\s+at\s+(?P<location>[^.]+)\."
    ),
    # "The <event> is scheduled for <date> at <time> in <location>."
    re.compile(
        rf"The\s+(?P<event>.+?)\s+is scheduled for\s+(?P<date>{DATE_RE})\s+at\s+(?P<time>{TIME_RE})\s+in\s+(?P<location>[^.]+)\."
    ),
    # "Are you available for the <event> at <time> on <date>? Location: <location>."
    re.compile(
        rf"Are you available for the\s+(?P<event>.+?)\s+at\s+(?P<time>{TIME_RE})\s+on\s+(?P<date>{DATE_RE})\?\s*Location:\s*(?P<location>[^.]+)\."
    ),
]

_EVENT_VAGUE_PATTERNS = [
    (re.compile(r"Let us meet sometime next week\."),
     ("Team meeting (unscheduled)", "next week")),
    (re.compile(r"The review could be Friday afternoon\."),
     ("Review (tentative)", "Friday afternoon")),
]


def _extract_event(message: str, msg_date: datetime, msg_id: str) -> Optional[ExtractedItem]:
    for pat in _EVENT_PATTERNS:
        m = pat.search(message)
        if m:
            event = m.group("event").strip()
            date_str = m.group("date")
            time_str = m.group("time")
            location = m.group("location").strip()
            title = event[0].upper() + event[1:]
            person = _find_person(message)
            return ExtractedItem(
                type="event",
                title=title,
                description=f"Scheduled event: {event} at {location}.",
                deadline=date_str,
                time=time_str,
                person=person,
                priority=_priority_from_dates(msg_date, date_str),
                source_message_id=msg_id,
            )
    for pat, (title, raw_phrase) in _EVENT_VAGUE_PATTERNS:
        m = pat.search(message)
        if m:
            return ExtractedItem(
                type="event",
                title=title,
                description=f"Extracted from: \"{m.group(0)}\" — date/time phrase "
                f"(\"{raw_phrase}\") is relative/vague and was not resolved to an exact value.",
                deadline="unresolved",
                time="unresolved",
                person=None,
                priority="low",
                source_message_id=msg_id,
            )
    return None


def extract_item(category: str, message: str, timestamp: str, msg_id: str) -> Optional[ExtractedItem]:
    msg_date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    core = strip_noise_prefixes(message)
    if category == "action_required":
        return _extract_task(core, msg_date, msg_id)
    if category == "meeting_or_event":
        return _extract_event(core, msg_date, msg_id)
    return None
