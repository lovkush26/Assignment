"""
Message classification into six categories (Part 1).

Approach: ordered rule-based / keyword-regex matching, not a trained model.
The dataset ships with no labels, so a supervised classifier has nothing to
learn from; the messages themselves are also highly templated (a small set of
recurring sentence patterns wrapped in random "framing" phrases like
"For today:", "FYI:", "Quick update:" that carry no category signal). A
transparent rule engine is the more defensible choice here: every decision
traces back to a specific matched phrase, which satisfies the assignment's
requirement that every classification carry an explainable reason.

Rules are evaluated in priority order, most-specific first, and the first
category whose rules match wins:
  1. sensitive_information   (a concrete secret/PII value is present)
  2. promotional              (marketing language / promo sender)
  3. meeting_or_event         (a scheduled meeting/event with date+time)
  4. action_required          (an explicit task/reminder/deadline)
  5. personal_information     (a non-sensitive personal fact/preference)
  6. general_information      (fallback: plain FYI/status statement)

This ordering matters: e.g. "You can contact me on 98765 43210-86." mentions
a person but is caught by the sensitive phone-number rule before it could be
miscategorized as personal_information.
"""
import re
from dataclasses import dataclass

from sensitive import detect_sensitive
from prefixes import strip_noise_prefixes
from l2_patterns import (
    detect_l2_signal,
    SIGNAL_STATUS_CHECK, SIGNAL_DUPLICATE_STATUS_CHECK, SIGNAL_COMPLETED,
    SIGNAL_CANCELLED, SIGNAL_EVENT_CANCELLED, SIGNAL_RESCHEDULED,
    SIGNAL_DEADLINE_URGENT_EARLIER, SIGNAL_DEADLINE_CONFLICT,
    SIGNAL_DEADLINE_EXTENDED, SIGNAL_AMBIGUOUS, SIGNAL_NEW_TASK, SIGNAL_NEW_EVENT,
)

PROMO_KEYWORDS = re.compile(
    r"\buse code\b|\bdiscount\b|\bsale\b|\d+%\s*off|\bsave\s+\d+%|\bcashback\b|\bpremium plan\b|"
    r"\bsubscription\b|\breward points\b|\bfree delivery\b|\bcoupon\b|"
    r"\blimited-time offer\b|\bstudent plan\b|\bexclusive benefits\b",
    re.IGNORECASE,
)

# L2 signal type -> (category, confidence override or None to use the
# signal's own confidence). See l2_patterns.py for how each signal is
# detected; the category choice is documented in the L2 README section:
# status-check/new-task/deadline-change messages are still directives
# ("action_required"); reschedule/new-event messages are still about a
# scheduled event ("meeting_or_event"); pure status reports (completed/
# cancelled/ambiguous) carry no outstanding request ("general_information").
_L2_SIGNAL_CATEGORY = {
    SIGNAL_STATUS_CHECK: "action_required",
    SIGNAL_DUPLICATE_STATUS_CHECK: "action_required",
    SIGNAL_COMPLETED: "general_information",
    SIGNAL_CANCELLED: "general_information",
    SIGNAL_EVENT_CANCELLED: "meeting_or_event",
    SIGNAL_RESCHEDULED: "meeting_or_event",
    SIGNAL_DEADLINE_URGENT_EARLIER: "action_required",
    SIGNAL_DEADLINE_CONFLICT: "action_required",
    SIGNAL_DEADLINE_EXTENDED: "action_required",
    SIGNAL_AMBIGUOUS: "general_information",
    SIGNAL_NEW_TASK: "action_required",
    SIGNAL_NEW_EVENT: "meeting_or_event",
}

MEETING_KEYWORDS = re.compile(
    r"\bcalendar update\b|\bhappens on\b|\bscheduled for\b|\bplease join the\b|"
    r"\bavailable for the\b.*\bon\b|\border\b",
    re.IGNORECASE,
)
MEETING_TENTATIVE = re.compile(
    r"\blet us meet\b|\bmeet sometime\b|\breview could be\b", re.IGNORECASE
)

ACTION_KEYWORDS = re.compile(
    r"don't forget to|\bi need you to\b|\bplease (submit|reply|confirm|complete|"
    r"call|send|update|review|upload|email|back up|verify|share)\b|"
    r"\bcan you (review|update|finish|send)\b|\bis due on\b|\bdeadline is\b|"
    r"\bcould you send it\b|\breview the file before\b|\bmay be needed\b",
    re.IGNORECASE,
)

PERSONAL_KEYWORDS = re.compile(
    r"\bfor my profile\b|\bpersonal note\b|\bjust so you know\b|\bremember that\b|"
    r"\bemergency contact\b|\bfavourite\b|\bprefer\b|\bvegetarian\b|"
    r"\bt-shirt size\b|\bdark mode\b|\bstudy after dinner\b|\blive near\b|"
    r"\bmight prefer\b|\bi drink coffee\b",
    re.IGNORECASE,
)


@dataclass
class Classification:
    category: str
    confidence: float
    reason: str


def classify_message(sender: str, message: str) -> Classification:
    sensitive_matches = detect_sensitive(message)
    if sensitive_matches:
        types = ", ".join(sorted({m.sensitivity_type for m in sensitive_matches}))
        return Classification(
            category="sensitive_information",
            confidence=0.97,
            reason=f"Message contains a detected sensitive value ({types}).",
        )

    if sender == "Promotions" and PROMO_KEYWORDS.search(message):
        return Classification(
            category="promotional",
            confidence=0.96,
            reason="Sent by the Promotions sender and contains marketing/discount language.",
        )
    if PROMO_KEYWORDS.search(message):
        return Classification(
            category="promotional",
            confidence=0.88,
            reason="Contains marketing/discount language (e.g. a promo code or sale offer).",
        )

    l2_signal = detect_l2_signal(strip_noise_prefixes(message))
    if l2_signal:
        return Classification(
            category=_L2_SIGNAL_CATEGORY[l2_signal.signal_type],
            confidence=l2_signal.confidence,
            reason=l2_signal.reason,
        )

    if MEETING_KEYWORDS.search(message):
        return Classification(
            category="meeting_or_event",
            confidence=0.93,
            reason="Describes a scheduled meeting/event with a date and time.",
        )
    if MEETING_TENTATIVE.search(message):
        return Classification(
            category="meeting_or_event",
            confidence=0.6,
            reason="Refers to a possible meeting/get-together but uses tentative language "
            "and lacks a confirmed date, lowering confidence.",
        )

    if ACTION_KEYWORDS.search(message):
        confidence = 0.9 if re.search(r"\bdeadline is\b|\bis due on\b", message, re.I) else 0.85
        return Classification(
            category="action_required",
            confidence=confidence,
            reason="Contains an explicit request/instruction directed at the recipient.",
        )

    if PERSONAL_KEYWORDS.search(message):
        return Classification(
            category="personal_information",
            confidence=0.87,
            reason="Shares a personal fact or preference about the sender/user that is "
            "not a secret or identifying credential.",
        )

    return Classification(
        category="general_information",
        confidence=0.65,
        reason="Plain informational/status statement; no action, schedule, personal, "
        "promotional, or sensitive signal detected.",
    )
