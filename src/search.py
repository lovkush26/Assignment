"""
Semantic Search and Intelligent Assistant (L2 Part 3).

Answers natural-language questions using only the structured output already
produced by Parts 1-3 of the pipeline (classifications, tasks/events,
sensitive findings, priority records, related-message groups, privacy
routes) plus the original message text where the routing decision permits
it (never for a blocked/high-risk message — those are only ever shown
masked, same as Part 3's own masking rule).

Two layers, in order, same "literal first, meaning-based fallback second"
design as grouping.py:

1. **Intent rules** (`_INTENT_HANDLERS`): a small ordered set of regex/
   keyword matchers, each covering one of the assignment's example query
   shapes ("today", "critical or high", "rescheduled", "completed",
   "confirmation", "why was X critical", "deadlines changed", "conflicting
   messages", "blocked"). Each handler queries the structured registries
   directly and returns real IDs — nothing about the answer is generated
   freeform.
2. **TF-IDF fallback** (`retrieval.Corpus`, the same local, no-network
   vector-space technique grouping.py uses for its fuzzy subject match):
   for open-ended "show me / tell me about X" questions, or to resolve which
   group a query is asking about, cosine similarity over group titles/
   summaries and raw message text finds the best evidence. Below a minimum
   score, the assistant reports it has insufficient evidence rather than
   guessing — required by the assignment ("must not generate an unsupported
   answer").
"""
import re
from datetime import datetime
from typing import Dict, List, Optional

from retrieval import Corpus

MIN_EVIDENCE_SCORE = 0.2
# Stricter threshold for the assistant's own fuzzy subject resolution
# (deciding which group a free-text question is about). A single shared
# generic word (e.g. "form" in "compliance form" vs. "onboarding form") can
# clear a low bar without the messages being genuinely related — reporting
# insufficient evidence is safer than a confident-looking wrong answer here.
FALLBACK_MIN_SCORE = 0.4
_PRIORITY_TIER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_ID_TOKEN_RE = re.compile(r"\b([A-Z]+_\d+)\b")
_TERMINAL_STATUSES = {"completed", "cancelled"}


class Assistant:
    def __init__(self, messages_df, classifications, items, sensitive_records,
                 priority_records, groups, routing_records):
        self.messages_by_id = {row["message_id"]: row for _, row in messages_df.iterrows()}
        self.classifications_by_id = {c["message_id"]: c for c in classifications}
        self.items_by_id = {i["item_id"]: i for i in items}
        self.sensitive_by_msg: Dict[str, list] = {}
        for s in sensitive_records:
            self.sensitive_by_msg.setdefault(s["message_id"], []).append(s)
        self.routing_by_msg = {r["message_id"]: r for r in routing_records}
        self.groups = groups
        self.groups_by_id = {g["group_id"]: g for g in groups}
        self.group_by_message: Dict[str, dict] = {}
        self.group_by_item: Dict[str, dict] = {}
        for g in groups:
            for mid in g["related_message_ids"]:
                self.group_by_message[mid] = g
            for iid in g["related_item_ids"]:
                self.group_by_item[iid] = g

        self.priority_records = priority_records
        self.priority_history_by_item: Dict[str, list] = {}
        for rec in priority_records:
            self.priority_history_by_item.setdefault(rec["item_id"], []).append(rec)
        self.current_priority_by_item = {
            iid: hist[-1] for iid, hist in self.priority_history_by_item.items()
        }

        timestamps = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in messages_df["timestamp"]]
        self.now = max(timestamps) if timestamps else datetime.now()
        self.today = self.now.date().isoformat()

        self._group_corpus = Corpus(
            ids=[g["group_id"] for g in groups],
            texts=[f'{g["title"]} {g["summary"]}' for g in groups],
        )
        self._message_corpus = Corpus(
            ids=list(self.messages_by_id.keys()),
            texts=[self._display_text(mid) for mid in self.messages_by_id],
        )

    # ------------------------------------------------------------------ #
    # Privacy-safe text access: a blocked (high-risk) message's raw text
    # is never surfaced by the assistant, even internally for retrieval —
    # only its masked form, same guarantee Part 3 gives everywhere else.
    def _display_text(self, msg_id: str) -> str:
        row = self.messages_by_id[msg_id]
        findings = self.sensitive_by_msg.get(msg_id)
        if findings:
            return findings[0]["masked_text"]
        return row["message"]

    def _resolve_group_by_id_token(self, query: str) -> Optional[dict]:
        for token in _ID_TOKEN_RE.findall(query):
            if token in self.group_by_message:
                return self.group_by_message[token]
            if token in self.group_by_item:
                return self.group_by_item[token]
            if token in self.groups_by_id:
                return self.groups_by_id[token]
        return None

    def _fuzzy_group(self, phrase: str):
        match = self._group_corpus.best_match(phrase, min_score=FALLBACK_MIN_SCORE)
        if not match:
            return None, 0.0
        gid, score = match
        return self.groups_by_id[gid], score

    # ------------------------------------------------------------------ #
    def answer(self, query: str) -> dict:
        q = query.strip()
        ql = q.lower()
        for pattern, handler in _INTENT_HANDLERS:
            if re.search(pattern, ql):
                result = handler(self, q, ql)
                if result is not None:
                    result["query"] = q
                    return result
        return self._fallback_semantic_answer(q, ql)

    def _fallback_semantic_answer(self, q: str, ql: str) -> dict:
        group, score = self._fuzzy_group(q)
        if group:
            return {
                "query": q,
                "answer": group["summary"],
                "supporting_message_ids": group["related_message_ids"],
                "related_item_ids": group["related_item_ids"],
                "group_id": group["group_id"],
                "relevance_scores": [{"id": group["group_id"], "score": round(score, 3)}],
                "reason": "No specific query pattern matched; the best-matching related-message group by "
                          "TF-IDF similarity to the question was used as evidence.",
            }
        msg_matches = self._message_corpus.top_matches(q, k=5, min_score=MIN_EVIDENCE_SCORE)
        # A near-verbatim match (score ~1.0) means the dataset contains a message that IS this
        # question — e.g. someone asked it — not a message that answers it. That alone isn't
        # evidence of an answer, so it's excluded before deciding whether real evidence exists.
        corroborating = [(m, s) for m, s in msg_matches if s < 0.95]
        if corroborating:
            return {
                "query": q,
                "answer": "I found related messages but no confident group/task match - see the supporting "
                          "messages below.",
                "supporting_message_ids": [m for m, _ in corroborating],
                "related_item_ids": [],
                "relevance_scores": [{"id": m, "score": round(s, 3)} for m, s in corroborating],
                "reason": "Matched by TF-IDF term similarity to the question text.",
            }
        if msg_matches:
            return _no_evidence(
                q, f"The dataset contains a message asking this exact question ({msg_matches[0][0]}), but no "
                   "other message provides an answer to it.")
        return _no_evidence(q, "No message, task/event, or related-message group in the processed data is "
                                "similar enough to this question.")


def _no_evidence(q: str, reason: str) -> dict:
    return {
        "query": q,
        "answer": "I don't have sufficient evidence in the processed messages to answer this.",
        "supporting_message_ids": [],
        "related_item_ids": [],
        "relevance_scores": [],
        "reason": reason,
    }


# ---------------------------------------------------------------------- #
# Intent handlers. Each takes (assistant, original_query, lowercased_query)
# and returns a response dict, or None to fall through to the next rule.

def _handle_today(a: Assistant, q, ql):
    due_today = []
    for iid, rec in a.current_priority_by_item.items():
        item = a.items_by_id.get(iid)
        group = a.group_by_item.get(iid)
        if not item or group["status"] in _TERMINAL_STATUSES:
            continue
        if group.get("latest_deadline") == a.today:
            due_today.append((iid, group, rec))
    if not due_today:
        return _no_evidence(q, f"No open task/event currently has a latest known deadline of {a.today} "
                                f'(the most recent message timestamp in the processed data, used as "today").')
    lines = [f'{g["title"]} ({iid}, priority {rec["priority"]})' for iid, g, rec in due_today]
    return {
        "answer": f'{len(due_today)} item(s) due today ({a.today}): ' + "; ".join(lines) + ".",
        "supporting_message_ids": [g["related_message_ids"][-1] for _, g, _ in due_today],
        "related_item_ids": [iid for iid, _, _ in due_today],
        "relevance_scores": [{"id": iid, "score": 1.0} for iid, _, _ in due_today],
        "reason": f'"Today" is taken as {a.today}, the latest message timestamp in the processed dataset '
                  "(there is no real wall-clock date inside this fictional dataset's timeline).",
    }


def _handle_critical_high_pending(a: Assistant, q, ql):
    hits = []
    for iid, rec in a.current_priority_by_item.items():
        group = a.group_by_item.get(iid)
        if rec["priority"] in ("critical", "high") and group["status"] not in _TERMINAL_STATUSES:
            hits.append((iid, group, rec))
    if not hits:
        return _no_evidence(q, "No item currently has priority critical/high while still pending.")
    hits.sort(key=lambda x: (x[2]["priority"] != "critical", x[0]))
    lines = [f'{g["title"]} ({iid}, {rec["priority"]})' for iid, g, rec in hits]
    return {
        "answer": f"{len(hits)} pending critical/high-priority item(s): " + "; ".join(lines) + ".",
        "supporting_message_ids": [rec["message_id"] for _, _, rec in hits],
        "related_item_ids": [iid for iid, _, _ in hits],
        "relevance_scores": [{"id": iid, "score": {"critical": 1.0, "high": 0.85}[rec["priority"]]}
                              for iid, _, rec in hits],
        "reason": "Filtered current_priorities for priority in {critical, high} and group status not in "
                  "{completed, cancelled}.",
    }


def _handle_rescheduled(a: Assistant, q, ql):
    hits = [g for g in a.groups if g["status"] == "rescheduled"]
    if not hits:
        return _no_evidence(q, "No related-message group currently has status 'rescheduled'.")
    lines = [f'{g["title"]} -> {g["latest_deadline"]}' for g in hits]
    return {
        "answer": f"{len(hits)} rescheduled meeting/event group(s): " + "; ".join(lines) + ".",
        "supporting_message_ids": [g["related_message_ids"][-1] for g in hits],
        "related_item_ids": [iid for g in hits for iid in g["related_item_ids"]],
        "relevance_scores": [{"id": g["group_id"], "score": g["confidence"]} for g in hits],
        "reason": "Groups whose current status was updated to 'rescheduled' by a later 'moved to ...' message.",
    }


def _handle_completed_or_cancelled(a: Assistant, q, ql):
    wants_cancelled = "cancel" in ql
    wants_completed = "complet" in ql
    statuses = set()
    if wants_completed:
        statuses.add("completed")
    if wants_cancelled:
        statuses.add("cancelled")
    if not statuses:
        statuses = {"completed", "cancelled"}
    hits = [g for g in a.groups if g["status"] in statuses]
    if not hits:
        return _no_evidence(q, f"No related-message group currently has status in {sorted(statuses)}.")
    lines = [f'{g["title"]} ({g["status"]})' for g in hits]
    return {
        "answer": f"{len(hits)} item(s): " + "; ".join(lines) + ".",
        "supporting_message_ids": [g["related_message_ids"][-1] for g in hits],
        "related_item_ids": [iid for g in hits for iid in g["related_item_ids"]],
        "relevance_scores": [{"id": g["group_id"], "score": g["confidence"]} for g in hits],
        "reason": f"Groups whose current status is in {sorted(statuses)}.",
    }


def _handle_confirmation_required(a: Assistant, q, ql):
    hits = [r for r in a.routing_by_msg.values() if r["route"] == "ask_for_confirmation"]
    if not hits:
        return _no_evidence(q, "No message is currently routed as 'ask_for_confirmation'.")
    return {
        "answer": f"{len(hits)} message(s) require confirmation before processing: " +
                  ", ".join(r["message_id"] for r in hits) + ".",
        "supporting_message_ids": [r["message_id"] for r in hits],
        "related_item_ids": [],
        "relevance_scores": [{"id": r["message_id"], "score": 1.0} for r in hits],
        "reason": "Privacy routing marked these messages medium-risk (private phone/address/health detail), "
                  "which requires human confirmation before storage or further action.",
    }


def _handle_blocked(a: Assistant, q, ql):
    hits = [r for r in a.routing_by_msg.values() if r["route"] == "blocked"]
    if not hits:
        return _no_evidence(q, "No message is currently routed as 'blocked'.")
    return {
        "answer": f"{len(hits)} message(s) must be blocked from external processing: " +
                  ", ".join(r["message_id"] for r in hits) + ".",
        "supporting_message_ids": [r["message_id"] for r in hits],
        "related_item_ids": [],
        "relevance_scores": [{"id": r["message_id"], "score": 1.0} for r in hits],
        "reason": "Privacy routing marked these messages high-risk (password/OTP/token/card/bank/ID), which "
                  "blocks them from any further or external processing.",
    }


def _handle_deadlines_changed(a: Assistant, q, ql):
    hits = [g for g in a.groups if g["conflicting_deadlines"]]
    if not hits:
        return _no_evidence(q, "No group has more than one distinct stated deadline on record.")
    lines = [f'{g["title"]}: ' + ", ".join(c["date"] for c in g["conflicting_deadlines"]) +
             f' -> now {g["latest_deadline"]}' for g in hits]
    return {
        "answer": f"{len(hits)} item(s) with a changed deadline: " + "; ".join(lines) + ".",
        "supporting_message_ids": [c["message_id"] for g in hits for c in g["conflicting_deadlines"]],
        "related_item_ids": [iid for g in hits for iid in g["related_item_ids"]],
        "relevance_scores": [{"id": g["group_id"], "score": g["confidence"]} for g in hits],
        "reason": "Groups that logged one or more conflicting_deadlines entries (a later message stated a "
                  "different concrete date than previously known).",
    }


def _handle_conflicting_same_event(a: Assistant, q, ql):
    hits = [g for g in a.groups if g["conflicting_deadlines"] or g["status"] == "unclear"]
    if not hits:
        return _no_evidence(q, "No group shows conflicting or uncertain deadline information.")
    lines = [f'{g["title"]} ({g["status"]})' for g in hits]
    return {
        "answer": f"{len(hits)} item(s) with conflicting/uncertain deadline messages: " + "; ".join(lines) + ".",
        "supporting_message_ids": list({mid for g in hits for c in g["conflicting_deadlines"]
                                         for mid in [c["message_id"]]}) or
                                   [g["related_message_ids"][-1] for g in hits],
        "related_item_ids": [iid for g in hits for iid in g["related_item_ids"]],
        "relevance_scores": [{"id": g["group_id"], "score": g["confidence"]} for g in hits],
        "reason": "Groups with an explicit conflicting_deadlines record, or whose current status is 'unclear' "
                  "because of ambiguous/hedged later messages.",
    }


def _handle_became_critical(a: Assistant, q, ql):
    require_demo = "demo" in ql
    hits = []
    for iid, hist in a.priority_history_by_item.items():
        if not hist:
            continue
        first_priority = hist[0]["priority"]
        last = hist[-1]
        touched_by_demo = any(r["message_id"].startswith("DEMO_") for r in hist)
        if last["priority"] != "critical" or first_priority == "critical":
            continue  # only items that started out lower and are currently critical
        if require_demo and not touched_by_demo:
            continue
        first_critical = next(r for r in hist if r["priority"] == "critical")
        hits.append((iid, first_critical, last))
    if not hits:
        scope = " touched by the demo batch" if require_demo else ""
        return _no_evidence(q, f"No item{scope} shows a priority history that started below 'critical' and is "
                                "now 'critical'.")
    lines = [f'{a.group_by_item[iid]["title"]} ({iid}) - first reached critical at {fc["message_id"]}, '
             f'currently critical as of {last["message_id"]}' for iid, fc, last in hits]
    return {
        "answer": "; ".join(lines) + ".",
        "supporting_message_ids": [fc["message_id"] for _, fc, _ in hits] + [last["message_id"] for _, _, last in hits],
        "related_item_ids": [iid for iid, _, _ in hits],
        "relevance_scores": [{"id": last["message_id"], "score": last["confidence"]} for _, _, last in hits],
        "reason": "; ".join(f'{iid}: {last["reason"]} (signals: {", ".join(last["signals"])})'
                             for iid, _, last in hits),
    }


def _handle_latest_status(a: Assistant, q, ql):
    group = a._resolve_group_by_id_token(q)
    if group is None:
        m = re.search(r"(?:status of|about|regarding)\s+(?:the\s+|this\s+)?(.+?)(?:\?|$)", ql)
        phrase = m.group(1) if m else q
        group, score = a._fuzzy_group(phrase)
        if group is None:
            return _no_evidence(q, f'No task/event/group matching "{phrase.strip()}" was found.')
    return {
        "answer": f'Latest status of "{group["title"]}": {group["status"]}'
                  + (f' (latest deadline {group["latest_deadline"]})' if group["latest_deadline"] else "") + ".",
        "supporting_message_ids": group["related_message_ids"],
        "related_item_ids": group["related_item_ids"],
        "group_id": group["group_id"],
        "relevance_scores": [{"id": group["group_id"], "score": group["confidence"]}],
        "reason": group["summary"],
    }


def _handle_why_critical(a: Assistant, q, ql):
    group = a._resolve_group_by_id_token(q)
    if group is None:
        m = re.search(r"why (?:was|is)\s+(?:this\s+)?(.+?)\s+marked", ql)
        phrase = m.group(1) if m else q
        group, score = a._fuzzy_group(phrase)
    if group is None:
        return _no_evidence(q, "No matching task/event/group was found to explain a priority for.")
    iid = group["related_item_ids"][0]
    rec = a.current_priority_by_item.get(iid)
    if rec is None:
        return _no_evidence(q, f'No priority record exists for "{group["title"]}".')
    return {
        "answer": f'"{group["title"]}" is currently priority {rec["priority"]}: {rec["reason"]}',
        "supporting_message_ids": [rec["message_id"]],
        "related_item_ids": [iid],
        "group_id": group["group_id"],
        "relevance_scores": [{"id": rec["message_id"], "score": rec["confidence"]}],
        "reason": f'Signals used: {", ".join(rec["signals"])}.',
    }


def _handle_related_messages(a: Assistant, q, ql):
    m = re.search(r"related to\s+(?:the\s+)?(.+?)(?:\?|$)", ql)
    phrase = m.group(1) if m else q
    group = a._resolve_group_by_id_token(q)
    score = 1.0
    if group is None:
        group, score = a._fuzzy_group(phrase)
    if group is None:
        return _no_evidence(q, f'No task/event/group matching "{phrase.strip()}" was found.')
    return {
        "answer": group["summary"],
        "supporting_message_ids": group["related_message_ids"],
        "related_item_ids": group["related_item_ids"],
        "group_id": group["group_id"],
        "relevance_scores": [{"id": group["group_id"], "score": round(score, 3)}],
        "reason": f'All messages linked into {group["group_id"]} by exact or TF-IDF subject match.',
    }


_INTENT_HANDLERS = [
    (r"became critical|become critical|which .*(task|item|meeting).*critical", _handle_became_critical),
    (r"\btoday\b", _handle_today),
    (r"critical or high|high[- ]priority|high or critical", _handle_critical_high_pending),
    (r"reschedul", _handle_rescheduled),
    (r"complet|cancel", _handle_completed_or_cancelled),
    (r"blocked|must be blocked", _handle_blocked),
    (r"confirmation", _handle_confirmation_required),
    (r"why (was|is).*(critical|priority)", _handle_why_critical),
    (r"deadlines? (have )?changed|changed deadline", _handle_deadlines_changed),
    (r"conflicting|uncertain", _handle_conflicting_same_event),
    (r"latest status", _handle_latest_status),
    (r"related to|show (all )?messages", _handle_related_messages),
]
