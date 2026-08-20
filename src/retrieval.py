"""
Shared local semantic retrieval used by grouping.py (fuzzy subject fallback,
when a message's normalized subject doesn't exactly match a known task/event
title) and search.py (Part 3's semantic search / assistant).

Uses scikit-learn's TF-IDF vector space model + cosine similarity — a
classic, explainable meaning-based retrieval technique (matches on word/
bigram overlap and importance, not just a single exact keyword), computed
entirely in-process over data already loaded in memory. No network call, no
trained/black-box model, no external AI service — consistent with the rest
of this pipeline's "runs entirely locally" rule (see README). It is a
deliberately lightweight, transparent stand-in for embedding-based semantic
search: every match can be explained by which shared terms drove the score.
"""
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Corpus:
    ids: List[str]
    texts: List[str]

    def __post_init__(self):
        self._vectorizer = None
        self._matrix = None
        if self.texts:
            try:
                self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
                self._matrix = self._vectorizer.fit_transform(self.texts)
            except ValueError:
                # Can happen on a degenerate all-stopword corpus; fall back to
                # no-vocabulary-filtering so retrieval still works.
                self._vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
                self._matrix = self._vectorizer.fit_transform(self.texts)

    def top_matches(self, query: str, k: int = 5, min_score: float = 0.0) -> List[Tuple[str, float]]:
        """Return up to k (id, cosine_similarity) pairs, highest first,
        restricted to score >= min_score. Empty corpus -> empty list."""
        if not self.texts or self._matrix is None or not query.strip():
            return []
        qvec = self._vectorizer.transform([query])
        sims = cosine_similarity(qvec, self._matrix)[0]
        ranked = sorted(zip(self.ids, sims), key=lambda pair: -pair[1])
        return [(i, float(s)) for i, s in ranked[:k] if s >= min_score]

    def best_match(self, query: str, min_score: float = 0.0):
        m = self.top_matches(query, k=1, min_score=min_score)
        return m[0] if m else None
