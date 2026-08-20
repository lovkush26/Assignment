"""
Sensitive-information detection and masking (Part 3).

Every rule is a plain regex tied to one concrete sensitive value pattern found
in the dataset's templates. Detection is intentionally literal/explainable
rather than a black-box classifier: each rule names exactly which substring it
treats as sensitive, and only that substring is masked.

Recommended-action mapping (documented rationale, see README):
- Credentials / financial secrets / government-style IDs (password, OTP,
  access token, recovery code, card number, bank account number,
  identification number) -> "do_not_store": these must never be persisted.
- Private contact details (home address, phone number) and health
  information -> "ask_for_confirmation": personal but sometimes legitimately
  needed, so a human should confirm before it is stored or acted on.

Regardless of the per-message recommended_action, the pipeline as a whole
never sends raw message text to an external AI/service (a system-level rule,
not a per-message one).

**L2 extension:** the L2 dataset re-expresses each of these ten sensitive
types with new sentence templates (e.g. L1's "Use password X to sign in" vs
L2's "The temporary password is X." / "Use temporary password X for the
sample account.") and adds "Follow-up:"-prefixed duplicates whose values
carry a trailing "-B" marker. Each rule below now tries a short list of
patterns (most specific first) for the same type/risk/action, instead of one
regex — additive only, so every L1 match still fires identically (verified:
a full 900-message re-run still yields exactly 100 sensitive findings, see
src/pipeline.py output).
"""
import re
from dataclasses import dataclass
from typing import Optional

MASK = "******"


@dataclass
class SensitiveMatch:
    sensitivity_type: str
    risk: str
    recommended_action: str
    span: tuple  # (start, end) of the sensitive value in the original message
    reason: str


def _pat(p):
    return re.compile(p, re.IGNORECASE)


# Each rule: (sensitivity_type, risk, recommended_action, [compiled regex
# variants, most specific first, each with a named group 'value' spanning
# exactly the sensitive substring], reason). Only the first variant that
# matches contributes a record, so a message never gets double-counted for
# the same type even if two variants could technically both match.
_RULES = [
    (
        "password",
        "high",
        "do_not_store",
        [
            _pat(r"use password\s+(?P<value>.+?)\s+to sign in"),                 # L1
            _pat(r"use temporary password\s+(?P<value>\S+)\s+for"),              # L2 demo
            _pat(r"temporary password is\s+(?P<value>\S+?)\."),                   # L2
        ],
        "Message contains an explicit login password.",
    ),
    (
        "one_time_password",
        "high",
        "do_not_store",
        [
            _pat(r"your (?:new |fictional )?otp is\s+(?P<value>[\w\-]+)\."),      # L1 + L2 + L2 demo
        ],
        "Message contains a one-time password (OTP) used for authentication.",
    ),
    (
        "authentication_token",
        "high",
        "do_not_store",
        [
            _pat(r"account recovery code is\s+(?P<value>[A-Za-z0-9\-]+)\."),      # L1
            _pat(r"recovery code\s+(?P<value>[A-Za-z0-9\-]+)\."),                 # L2 ("Save recovery code X.")
            _pat(r"temporary access token is\s+(?P<value>\S+)\."),                # L1
            _pat(r"use access token\s+(?P<value>\S+)\s+for"),                     # L2
            _pat(r"integration token:\s*(?P<value>\S+?)\."),                       # L2 demo
        ],
        "Message contains an authentication/access token or account recovery code that can be "
        "used to bypass authentication.",
    ),
    (
        "bank_or_payment_details",
        "high",
        "do_not_store",
        [
            _pat(r"card number is\s+(?P<value>[\w\s\-]+?)\."),                    # L1 + L2 (broadened for -B suffix)
            _pat(r"bank account(?:\s+number)?(?:\s+is)?\s+(?P<value>[\w\-]+)\."), # L1 + L2
        ],
        "Message contains a payment card number or bank account number.",
    ),
    (
        "personal_identification_number",
        "high",
        "do_not_store",
        [
            _pat(r"(?:identification|id) number is\s+(?P<value>[A-Za-z0-9\-]+)\."),  # L1 + L2
        ],
        "Message contains a personal identification number.",
    ),
    (
        "private_contact_detail",
        "medium",
        "ask_for_confirmation",
        [
            _pat(r"contact me on\s+(?P<value>[\d\s\-]+)\."),                      # L1
            _pat(r"call me on\s+(?P<value>[\d\s\-]+?)\s+after\b"),                # L2 + L2 demo
        ],
        "Message contains a private phone number.",
    ),
    (
        "private_address",
        "medium",
        "ask_for_confirmation",
        [
            _pat(r"home address is\s+(?P<value>.+?)\."),                          # L1
            _pat(r"deliver.{0,40}?\bto\s+(?P<value>\d[^.]*)\."),                  # L2 + L2 demo
        ],
        "Message contains a private home/delivery address.",
    ),
    (
        "health_information",
        "medium",
        "ask_for_confirmation",
        [
            _pat(r"recent test result says\s+(?P<value>.+?)\."),                  # L1
            _pat(r"medical note mentions\s+(?:a\s+)?(?P<value>[^.]+)\."),         # L2 + L2 demo
        ],
        "Message contains a personal health/medical detail. Not explicitly listed in the "
        "assignment's sensitive-type examples, but treated as sensitive personal data "
        "(extended category, see README).",
    ),
]


def detect_sensitive(message: str) -> list:
    """Return all SensitiveMatch hits in a message (usually 0 or 1)."""
    matches = []
    for sensitivity_type, risk, action, patterns, reason in _RULES:
        for pattern in patterns:
            m = pattern.search(message)
            if m:
                matches.append(
                    SensitiveMatch(
                        sensitivity_type=sensitivity_type,
                        risk=risk,
                        recommended_action=action,
                        span=m.span("value"),
                        reason=reason,
                    )
                )
                break  # first matching variant wins; don't double-count this type
    return matches


def mask_message(message: str, matches: list) -> str:
    """Replace every detected sensitive value with a fixed-length mask."""
    if not matches:
        return message
    # Replace from the end so earlier spans stay valid.
    masked = message
    for m in sorted(matches, key=lambda x: x.span[0], reverse=True):
        start, end = m.span
        masked = masked[:start] + MASK + masked[end:]
    return masked
