"""
Privacy-aware routing.

Turns Part 3's per-value sensitive-information findings (sensitive.py) into
one routing decision per message — the three states the assignment's demo
explicitly asks to show:

  - process_locally   — no sensitive value detected; safe to run through the
                         pipeline (classification/extraction/etc.) with no
                         restriction.
  - ask_for_confirmation — a medium-risk value is present (private phone
                         number, home/delivery address, health detail): may
                         legitimately be needed, but a human must confirm
                         before it is stored or acted on further.
  - blocked           — a high-risk value is present (password, OTP, access
                         token, recovery code, card number, bank account,
                         ID number): the message is blocked from any further
                         (especially external) processing outright.

A message can trigger multiple sensitive.py findings; the route is driven by
the single highest-risk finding present (high > medium), so the decision is
never softer than the most sensitive value actually found. This is a
one-message decision derived entirely from already-computed Part 3 output —
no new detection logic, just a routing rule on top of it, so it can never
disagree with what Part 3 reports as masked/detected.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

ROUTE_LOCAL = "process_locally"
ROUTE_CONFIRM = "ask_for_confirmation"
ROUTE_BLOCKED = "blocked"

_RISK_RANK = {"high": 2, "medium": 1}


def route_messages(messages_df, sensitive_records: List[dict]) -> List[dict]:
    """Returns one routing record per message in messages_df (chronological),
    each with: message_id, route, risk, sensitivity_types, reason."""
    findings_by_msg: Dict[str, List[dict]] = {}
    for rec in sensitive_records:
        findings_by_msg.setdefault(rec["message_id"], []).append(rec)

    routes = []
    for _, row in messages_df.iterrows():
        msg_id = row["message_id"]
        findings = findings_by_msg.get(msg_id, [])
        if not findings:
            routes.append({
                "message_id": msg_id,
                "route": ROUTE_LOCAL,
                "risk": "none",
                "sensitivity_types": [],
                "recommended_action": "process_locally",
                "reason": "No sensitive information detected; safe to process locally with no restriction.",
            })
            continue

        top = max(findings, key=lambda f: _RISK_RANK.get(f["risk"], 0))
        types = sorted({f["sensitivity_type"] for f in findings})
        if top["risk"] == "high":
            routes.append({
                "message_id": msg_id,
                "route": ROUTE_BLOCKED,
                "risk": "high",
                "sensitivity_types": types,
                "recommended_action": "do_not_store",
                "reason": f'Contains high-risk sensitive information ({", ".join(types)}); '
                          "blocked from any further/external processing.",
            })
        else:
            routes.append({
                "message_id": msg_id,
                "route": ROUTE_CONFIRM,
                "risk": "medium",
                "sensitivity_types": types,
                "recommended_action": "ask_for_confirmation",
                "reason": f'Contains medium-risk personal information ({", ".join(types)}); '
                          "requires human confirmation before storage or further action.",
            })
    return routes
