"""
L2 message-template recognizer (extends the L1 rule-engine approach to the
new sentence structures introduced by the L2 dataset: follow-ups/status
checks, completions, cancellations, reschedules, deadline changes/conflicts,
ambiguous hedges, and new tasks/events).

Same design philosophy as L1's classify.py/extract.py: every pattern here was
derived from src/analyze_l2_templates.py's frequency dump of l2_messages.csv
and l2_demo_messages.csv (not guessed), each tied to a named regex group, and
nothing is inferred beyond what a template literally states. This module is
the single source of truth for "what kind of L2 event does this message
describe", consumed by classify.py (category), extract.py (new task/event
items), priority.py (Part 1 signals), and grouping.py (Part 2 status/deadline
updates).
"""
import re
from dataclasses import dataclass
from typing import Optional

from textnorm import normalize_subject, titlecase_first

DATE_RE = r"\d{4}-\d{2}-\d{2}"
TIME_RE = r"\d{1,2}:\d{2}"

SIGNAL_STATUS_CHECK = "status_check"
SIGNAL_DUPLICATE_STATUS_CHECK = "duplicate_status_check"
SIGNAL_COMPLETED = "completed"
SIGNAL_CANCELLED = "cancelled"
SIGNAL_RESCHEDULED = "rescheduled"
SIGNAL_EVENT_CANCELLED = "event_cancelled"
SIGNAL_DEADLINE_URGENT_EARLIER = "deadline_urgent_earlier"
SIGNAL_DEADLINE_CONFLICT = "deadline_conflict"
SIGNAL_DEADLINE_EXTENDED = "deadline_extended"
SIGNAL_AMBIGUOUS = "ambiguous"
SIGNAL_NEW_TASK = "new_task"
SIGNAL_NEW_EVENT = "new_event"


@dataclass
class L2Signal:
    signal_type: str
    subject_key: Optional[str]        # normalize_subject() key, or None
    subject_title: Optional[str]      # human-readable, title-cased
    date: Optional[str] = None        # "YYYY-MM-DD" or "unresolved" or None
    time: Optional[str] = None
    location: Optional[str] = None
    urgent: bool = False
    conflict: bool = False
    matched_text: str = ""
    reason: str = ""
    confidence: float = 0.9


def _subj(m, group="subj"):
    raw = m.group(group).strip()
    return normalize_subject(raw), titlecase_first(raw)


# Order matters: more specific / higher-confidence patterns first.
_STATUS_CHECK_PATTERNS = [
    r"Can you share an update on (?P<subj>.+?)\?",
    r"Following up on (?P<subj>.+?); is it in progress\?",
    r"Please confirm whether you started to (?P<subj>.+?)\.",
    r"Any progress on the item concerning (?P<subj>.+?)\?",
    r"Please check the latest status of (?P<subj>.+?)\.",
    r"The work we discussed about (?P<subj>.+?) still needs attention\.",
    r"Has the (?P<subj>.+?) item been handled yet\?",
    r"I am referring to our earlier request about (?P<subj>.+?)\.",
    r"Any update on (?P<subj>.+?)\?",
    r"Has the material for our earlier (?P<subj>.+?) been handled\?",
]

_DUPLICATE_STATUS_PATTERN = r"This is another status request about (?P<subj>.+?), not a new task\."

_COMPLETED_PATTERNS = [
    r"Confirmed: (?P<subj>.+?) has been completed\.",
    r"(?P<subj>.+?) has been completed successfully\.",
]

_CANCELLED_TASK_PATTERNS = [
    r"You can cancel (?P<subj>.+?); it is no longer required\.",
    r"Cancel (?P<subj>.+?); it is no longer needed\.",
]

_CANCELLED_EVENT_PATTERNS = [
    r"The (?P<subj>.+?) has been cancelled\.",
]

_RESCHEDULED_FULL_PATTERNS = [
    rf"The (?P<subj>.+?) has been moved to (?P<date>{DATE_RE}) at (?P<time>{TIME_RE})\. Please use the new schedule\.",
    rf"The (?P<subj>.+?) has moved to (?P<date>{DATE_RE}) at (?P<time>{TIME_RE})\.",
]
_RESCHEDULED_TIME_ONLY_PATTERN = (
    rf"The date for (?P<subj>.+?) stays the same, but the time is now (?P<time>{TIME_RE})\."
)
_RESCHEDULED_TENTATIVE_PATTERN = (
    r"We may move the (?P<subj>.+?); I will confirm the schedule later\."
)

_DEADLINE_URGENT_EARLIER_PATTERN = (
    rf"The deadline to (?P<subj>.+?) is now (?P<date>{DATE_RE}), earlier than previously planned\. "
    r"Treat this as urgent\."
)
_DEADLINE_URGENT_RELATIVE_PATTERN = (
    r"The deadline to (?P<subj>.+?) is now tomorrow at (?P<hour>\d{1,2}) AM\. This is urgent\."
)
_DEADLINE_CONFLICT_PATTERN = (
    rf"Please note that (?P<subj>.+?) is due on (?P<date>{DATE_RE}), "
    r"although the earlier message listed another date\."
)
_DEADLINE_CONFLICT_STATED_PATTERN = (
    rf"One message says \w+, but the latest instruction says (?P<subj>.+?) is due on (?P<date>{DATE_RE})\."
)
_DEADLINE_EXTENDED_PATTERN = (
    rf"The deadline for (?P<subj>.+?) has been extended to (?P<date>{DATE_RE})\."
)

_AMBIGUOUS_WITH_SUBJECT_PATTERN = r"(?P<subj>.+?) might already be finished, but I cannot confirm it\."
_AMBIGUOUS_NO_SUBJECT_PATTERNS = [
    r"The report might already be done, but I am not completely sure\.",
    r"We may move the meeting, I will confirm later\.",
    r"The deadline could be tomorrow or Monday; please wait for confirmation\.",
    r"Maya said someone probably handled the task\.",
    r"This may no longer be urgent\.",
    r"The deadline may be Monday, or it may be Wednesday\. Wait for the official update\.",
]

_NEW_TASK_PATTERN = rf"New task: (?P<subj>.+?) by (?P<date>{DATE_RE})\."
_NEW_EVENT_PATTERNS = [
    rf"A new (?P<subj>.+?) meeting is scheduled for (?P<date>{DATE_RE}) at (?P<time>{TIME_RE}) on (?P<location>.+?)\.",
    rf"A new (?P<subj>.+?) session is scheduled for (?P<date>{DATE_RE}) at (?P<time>{TIME_RE})\.",
]


def detect_l2_signal(core_text: str) -> Optional[L2Signal]:
    """core_text should already have noise prefixes stripped (see
    prefixes.strip_noise_prefixes, which now also strips the L2 wrapper
    labels 'Follow-up:', 'Additional update:', 'Update:')."""
    text = core_text.strip()

    m = re.fullmatch(_DUPLICATE_STATUS_PATTERN, text, re.IGNORECASE)
    if m:
        key, title = _subj(m)
        return L2Signal(SIGNAL_DUPLICATE_STATUS_CHECK, key, title, matched_text=m.group(0),
                         reason="Explicitly marked as a repeat status request about an existing item, not a new task.",
                         confidence=0.95)

    for pat in _COMPLETED_PATTERNS:
        m = re.fullmatch(pat, text, re.IGNORECASE)
        if m:
            key, title = _subj(m)
            return L2Signal(SIGNAL_COMPLETED, key, title, matched_text=m.group(0),
                             reason=f'"{title}" is reported completed.',
                             confidence=0.95)

    for pat in _CANCELLED_TASK_PATTERNS:
        m = re.fullmatch(pat, text, re.IGNORECASE)
        if m:
            key, title = _subj(m)
            return L2Signal(SIGNAL_CANCELLED, key, title, matched_text=m.group(0),
                             reason=f'"{title}" is reported cancelled/no longer required.',
                             confidence=0.95)

    for pat in _CANCELLED_EVENT_PATTERNS:
        m = re.fullmatch(pat, text, re.IGNORECASE)
        if m:
            key, title = _subj(m)
            return L2Signal(SIGNAL_EVENT_CANCELLED, key, title, matched_text=m.group(0),
                             reason=f'"{title}" is reported cancelled.',
                             confidence=0.95)

    for pat in _RESCHEDULED_FULL_PATTERNS:
        m = re.fullmatch(pat, text, re.IGNORECASE)
        if m:
            key, title = _subj(m)
            return L2Signal(SIGNAL_RESCHEDULED, key, title, date=m.group("date"), time=m.group("time"),
                             matched_text=m.group(0),
                             reason=f'Message states the event was moved to {m.group("date")} at {m.group("time")}.',
                             confidence=0.95)

    m = re.fullmatch(_RESCHEDULED_TIME_ONLY_PATTERN, text, re.IGNORECASE)
    if m:
        key, title = _subj(m)
        return L2Signal(SIGNAL_RESCHEDULED, key, title, date=None, time=m.group("time"),
                         matched_text=m.group(0),
                         reason=f'Message states the date is unchanged but the time moved to {m.group("time")}.',
                         confidence=0.9)

    m = re.fullmatch(_RESCHEDULED_TENTATIVE_PATTERN, text, re.IGNORECASE)
    if m:
        key, title = _subj(m)
        return L2Signal(SIGNAL_RESCHEDULED, key, title, date="unresolved", time=None,
                         matched_text=m.group(0),
                         reason="Message tentatively suggests moving the event but gives no confirmed new date/time.",
                         confidence=0.55)

    m = re.fullmatch(_DEADLINE_URGENT_EARLIER_PATTERN, text, re.IGNORECASE)
    if m:
        key, title = _subj(m)
        return L2Signal(SIGNAL_DEADLINE_URGENT_EARLIER, key, title, date=m.group("date"), urgent=True,
                         matched_text=m.group(0),
                         reason=f'Message moves the deadline earlier to {m.group("date")} and explicitly flags it as urgent.',
                         confidence=0.95)

    m = re.fullmatch(_DEADLINE_URGENT_RELATIVE_PATTERN, text, re.IGNORECASE)
    if m:
        key, title = _subj(m)
        return L2Signal(SIGNAL_DEADLINE_URGENT_EARLIER, key, title, date="unresolved", urgent=True,
                         matched_text=m.group(0),
                         reason='Message states the deadline is now "tomorrow" (relative, not resolved to an exact '
                                'date) and explicitly flags it as urgent.',
                         confidence=0.8)

    m = re.fullmatch(_DEADLINE_CONFLICT_PATTERN, text, re.IGNORECASE)
    if m:
        key, title = _subj(m)
        return L2Signal(SIGNAL_DEADLINE_CONFLICT, key, title, date=m.group("date"), conflict=True,
                         matched_text=m.group(0),
                         reason=f'Message states a due date of {m.group("date")} while explicitly noting "the '
                                'earlier message listed another date" — a conflicting deadline.',
                         confidence=0.9)

    m = re.fullmatch(_DEADLINE_CONFLICT_STATED_PATTERN, text, re.IGNORECASE)
    if m:
        key, title = _subj(m)
        return L2Signal(SIGNAL_DEADLINE_CONFLICT, key, title, date=m.group("date"), conflict=True,
                         matched_text=m.group(0),
                         reason=f'Message explicitly contrasts two stated deadlines and gives {m.group("date")} as '
                                'the latest instruction.',
                         confidence=0.9)

    m = re.fullmatch(_DEADLINE_EXTENDED_PATTERN, text, re.IGNORECASE)
    if m:
        key, title = _subj(m)
        return L2Signal(SIGNAL_DEADLINE_EXTENDED, key, title, date=m.group("date"),
                         matched_text=m.group(0),
                         reason=f'Message extends the deadline to {m.group("date")}.',
                         confidence=0.95)

    m = re.fullmatch(_NEW_TASK_PATTERN, text, re.IGNORECASE)
    if m:
        key, title = _subj(m)
        return L2Signal(SIGNAL_NEW_TASK, key, title, date=m.group("date"),
                         matched_text=m.group(0),
                         reason=f'New task explicitly requested with a deadline of {m.group("date")}.',
                         confidence=0.95)

    for pat in _NEW_EVENT_PATTERNS:
        m = re.fullmatch(pat, text, re.IGNORECASE)
        if m:
            key, title = _subj(m)
            loc = m.groupdict().get("location")
            return L2Signal(SIGNAL_NEW_EVENT, key, title, date=m.group("date"), time=m.group("time"),
                             location=loc.strip() if loc else None,
                             matched_text=m.group(0),
                             reason=f'New event explicitly scheduled for {m.group("date")} at {m.group("time")}.',
                             confidence=0.95)

    m = re.fullmatch(_AMBIGUOUS_WITH_SUBJECT_PATTERN, text, re.IGNORECASE)
    if m:
        key, title = _subj(m)
        return L2Signal(SIGNAL_AMBIGUOUS, key, title, matched_text=m.group(0),
                         reason="Hedged/uncertain language about this item's status (\"might already be finished, "
                                "but I cannot confirm it\") — status cannot be reliably determined.",
                         confidence=0.5)

    for pat in _AMBIGUOUS_NO_SUBJECT_PATTERNS:
        m = re.fullmatch(pat, text, re.IGNORECASE)
        if m:
            return L2Signal(SIGNAL_AMBIGUOUS, None, None, matched_text=m.group(0),
                             reason="Hedged/uncertain language with no concrete subject named — cannot be linked "
                                    "to a specific task/event without guessing.",
                             confidence=0.4)

    for pat in _STATUS_CHECK_PATTERNS:
        m = re.fullmatch(pat, text, re.IGNORECASE)
        if m:
            key, title = _subj(m)
            return L2Signal(SIGNAL_STATUS_CHECK, key, title, matched_text=m.group(0),
                             reason=f'Follow-up/status-check request about "{title}" — no new status stated by '
                                    'this message itself.',
                             confidence=0.85)

    return None
