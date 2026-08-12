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


# Each rule: (sensitivity_type, risk, recommended_action, compiled regex with
# a named group 'value' spanning exactly the sensitive substring, reason)
_RULES = [
    (
        "password",
        "high",
        "do_not_store",
        re.compile(r"Use password\s+(?P<value>.+?)\s+to sign in", re.IGNORECASE),
        "Message contains an explicit login password.",
    ),
    (
        "one_time_password",
        "high",
        "do_not_store",
        re.compile(r"Your OTP is\s+(?P<value>[\d\-]+)\.", re.IGNORECASE),
        "Message contains a one-time password (OTP) used for authentication.",
    ),
    (
        "authentication_token",
        "high",
        "do_not_store",
        re.compile(r"account recovery code is\s+(?P<value>[A-Z0-9\-]+)\.", re.IGNORECASE),
        "Message contains an account recovery code that can be used to bypass authentication.",
    ),
    (
        "authentication_token",
        "high",
        "do_not_store",
        re.compile(r"temporary access token is\s+(?P<value>\S+)\.", re.IGNORECASE),
        "Message contains an authentication/access token.",
    ),
    (
        "bank_or_payment_details",
        "high",
        "do_not_store",
        re.compile(r"card number is\s+(?P<value>[\d\s\-]+?)\.", re.IGNORECASE),
        "Message contains a payment card number.",
    ),
    (
        "bank_or_payment_details",
        "high",
        "do_not_store",
        re.compile(r"bank account number\s+(?P<value>[\d\-]+)\.", re.IGNORECASE),
        "Message contains a bank account number.",
    ),
    (
        "personal_identification_number",
        "high",
        "do_not_store",
        re.compile(r"identification number is\s+(?P<value>[A-Z0-9\-]+)\.", re.IGNORECASE),
        "Message contains a personal identification number.",
    ),
    (
        "private_contact_detail",
        "medium",
        "ask_for_confirmation",
        re.compile(r"contact me on\s+(?P<value>[\d\s\-]+)\.", re.IGNORECASE),
        "Message contains a private phone number.",
    ),
    (
        "private_address",
        "medium",
        "ask_for_confirmation",
        re.compile(r"home address is\s+(?P<value>.+?)\.", re.IGNORECASE),
        "Message contains a private home address.",
    ),
    (
        "health_information",
        "medium",
        "ask_for_confirmation",
        re.compile(r"recent test result says\s+(?P<value>.+?)\.", re.IGNORECASE),
        "Message contains a personal health/medical detail. Not explicitly listed in the "
        "assignment's sensitive-type examples, but treated as sensitive personal data "
        "(extended category, see README).",
    ),
]


def detect_sensitive(message: str) -> list:
    """Return all SensitiveMatch hits in a message (usually 0 or 1)."""
    matches = []
    for sensitivity_type, risk, action, pattern, reason in _RULES:
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
