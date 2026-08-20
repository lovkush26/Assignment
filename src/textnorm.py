"""
Shared text-normalization helpers used by extract.py, l2_patterns.py, and
grouping.py to turn an action/event phrase ("Review the privacy checklist.",
"review the privacy checklist", "the privacy checklist") into a single
comparable key, so the same real-world task/event can be recognized across
differently-worded messages without guessing — only literal shared wording
(after case/article/punctuation normalization) counts as a match here.
Fuzzy/meaning-based matching beyond this is handled separately in grouping.py
via TF-IDF similarity.
"""
import re

_LEADING_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_TRAILING_PUNCT = re.compile(r"[.?!,;:]+$")
_WS = re.compile(r"\s+")


def normalize_subject(text: str) -> str:
    """Canonical key for an action/event phrase: lowercase, strip leading
    articles (repeatedly — the dataset has a "the the X" double-article typo
    in a few templates), strip trailing punctuation, collapse whitespace."""
    if not text:
        return ""
    t = text.strip()
    t = _TRAILING_PUNCT.sub("", t)
    changed = True
    while changed:
        new_t = _LEADING_ARTICLES.sub("", t)
        changed = new_t != t
        t = new_t
    t = _WS.sub(" ", t)
    return t.strip().lower()


def titlecase_first(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]
