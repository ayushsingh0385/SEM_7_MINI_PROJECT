"""Stylometric features for AI-vs-human text.

TF-IDF captures *which words* appear; these features capture *how the writing
behaves*. The signals that separate the two classes most reliably:

  * burstiness - humans vary sentence length a lot, LLM prose is metronomic,
    so the spread of sentence lengths is a strong discriminator
  * vocabulary richness - type/token and hapax ratios
  * punctuation habits - humans scatter semicolons, dashes, parentheses and
    exclamation marks unevenly; model output is comma-heavy and regular
  * surface noise - stray double spaces, ALL-CAPS, digits and contractions
    show up far more in human writing

Every feature is scale-invariant
--------------------------------
None of these measure how *long* a document is, and that is deliberate.

An earlier version included raw ``n_chars`` / ``n_words`` / ``n_paragraphs``
counts. Because HC3's human answers are shorter than its ChatGPT answers
(median 725 vs 1025 characters), the classifier learned "short means human" and
then confidently mislabelled short AI-written passages as human. Document
length is a property of the corpus, not of the author, and it is trivial to
change, so it has no place in the feature set.

Three consequences of that rule:

  * counts are gone; only rates (per 1000 characters) and ratios remain
  * sentence-length spread is reported as a coefficient of variation
    (std / mean) as well as a raw std, so it does not drift with register
  * vocabulary richness is measured on a fixed ``VOCAB_WINDOW``-word window,
    because type/token ratio falls as a document grows and would otherwise
    smuggle length back in

Implemented with plain regex (no NLTK corpora to download), which keeps it fast
enough for 40k documents and removes a runtime dependency from the app.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_SENT_SPLIT = re.compile(r"[.!?]+(?:\s+|$)")
_WORD = re.compile(r"[A-Za-z']+")
_PARA_SPLIT = re.compile(r"\n\s*\n")

# Vocabulary richness is computed on this many words so that the measure does
# not depend on total document length.
#
# The window only does its job for documents that actually reach it, so it is
# set below the shortest document the data pipeline admits (MIN_CHARS = 350
# characters, roughly 60 words) rather than at a comfortable-looking 200. Most
# HC3 answers are 120-180 words; a 200-word window would bind on almost none of
# them and type/token ratio would quietly go back to tracking length, which is
# the exact shortcut this feature set exists to avoid. A 50-word estimate is
# noisier, but it is unbiased with respect to length, and that matters more.
VOCAB_WINDOW = 50

# A tiny inline stop list, so the transformer has no external data dependency.
_STOPWORDS = frozenset(
    """
a an the and or but if while of to in on at by for with about against between
into through during before after above below from up down out off over under
is are was were be been being am do does did doing have has had having
i you he she it we they me him her us them my your his its our their this that
these those as not no nor so than then there here when where why how all any
both each few more most other some such only own same too very can will just
""".split()
)

# Feature names, in the exact order _row() emits them.
FEATURE_NAMES: list[str] = [
    # sentence shape
    "avg_word_len", "avg_sent_len", "std_sent_len", "sent_len_cv", "max_sent_len",
    # vocabulary (measured on a fixed window)
    "type_token_ratio", "hapax_ratio", "repeat_bigram_ratio",
    "stopword_ratio", "long_word_ratio",
    # punctuation, per 1000 characters
    "comma_rate", "period_rate", "semicolon_rate", "colon_rate",
    "exclaim_rate", "question_rate", "quote_rate", "dash_rate",
    "paren_rate", "apostrophe_rate",
    # surface
    "uppercase_ratio", "digit_ratio", "whitespace_anomaly",
    # structure, as a rate rather than a count
    "paragraph_rate",
]
N_FEATURES = len(FEATURE_NAMES)

# Curly punctuation, kept as escapes so the file stays pure ASCII.
_LQUOTE, _RQUOTE = "“", "”"
_ENDASH, _EMDASH = "–", "—"
_RSQUO = "’"


def _row(text: str) -> list[float]:
    """Compute the scale-invariant feature vector for a single document."""
    text = text or ""
    n_chars = len(text)
    if n_chars == 0:
        return [0.0] * N_FEATURES

    words = _WORD.findall(text.lower())
    n_words = len(words)
    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    n_sents = len(sentences)

    # Rates are per-1000-characters so they do not scale with document length.
    per_k = 1000.0 / n_chars

    # ---- vocabulary, on a fixed window -----------------------------------
    window = words[:VOCAB_WINDOW]
    n_win = len(window)
    if n_win:
        counts: dict[str, int] = {}
        for w in window:
            counts[w] = counts.get(w, 0) + 1
        type_token = len(counts) / n_win
        hapax = sum(1 for c in counts.values() if c == 1) / n_win
        bigrams = list(zip(window, window[1:]))
        repeat_bi = 1.0 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    else:
        type_token = hapax = repeat_bi = 0.0

    # ---- word shape, over the whole document (already an average) --------
    if n_words:
        word_lens = [len(w) for w in words]
        avg_word_len = float(np.mean(word_lens))
        stop_ratio = sum(1 for w in words if w in _STOPWORDS) / n_words
        long_ratio = sum(1 for n in word_lens if n >= 7) / n_words
    else:
        avg_word_len = stop_ratio = long_ratio = 0.0

    # ---- sentence rhythm -------------------------------------------------
    if n_sents:
        sent_lens = [len(_WORD.findall(s)) for s in sentences]
        avg_sent = float(np.mean(sent_lens))
        std_sent = float(np.std(sent_lens))            # burstiness
        # Coefficient of variation: burstiness with the overall pace divided
        # out, so terse and expansive writers are compared on equal terms.
        cv = std_sent / avg_sent if avg_sent > 0 else 0.0
        max_sent = float(np.max(sent_lens))
    else:
        avg_sent = std_sent = cv = max_sent = 0.0

    # ---- structure, as a rate --------------------------------------------
    # Paragraph breaks per 1000 characters. Expressed as a rate on purpose:
    # "paragraphs per sentence" or "sentences per paragraph" both collapse to
    # a function of document length whenever a document has a single
    # paragraph, which is most of them, so they smuggle length back in.
    n_breaks = len(_PARA_SPLIT.findall(text))
    para_rate = n_breaks * per_k

    n_quotes = text.count('"') + text.count(_LQUOTE) + text.count(_RQUOTE)
    n_dashes = text.count("-") + text.count(_ENDASH) + text.count(_EMDASH)
    n_apos = text.count("'") + text.count(_RSQUO)

    return [
        avg_word_len, avg_sent, std_sent, cv, max_sent,
        type_token, hapax, repeat_bi,
        stop_ratio, long_ratio,
        text.count(",") * per_k,
        text.count(".") * per_k,
        text.count(";") * per_k,
        text.count(":") * per_k,
        text.count("!") * per_k,
        text.count("?") * per_k,
        n_quotes * per_k,
        n_dashes * per_k,
        (text.count("(") + text.count(")")) * per_k,
        n_apos * per_k,
        sum(1 for c in text if c.isupper()) / n_chars,
        sum(1 for c in text if c.isdigit()) / n_chars,
        text.count("  ") * per_k,
        para_rate,
    ]


class StylometricFeatures(BaseEstimator, TransformerMixin):
    """Turn an iterable of raw documents into a dense stylometric matrix.

    A stateless sklearn transformer, so it drops straight into a Pipeline or
    FeatureUnion and pickles cleanly with joblib.
    """

    def fit(self, X, y=None):  # noqa: N803  (sklearn API)
        return self

    def transform(self, X):  # noqa: N803
        return np.asarray([_row(str(doc)) for doc in X], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)


def describe(text: str) -> dict[str, float]:
    """Feature dict for one document - used by the UI to explain a verdict."""
    return {name: float(val) for name, val in zip(FEATURE_NAMES, _row(text))}


__all__ = [
    "StylometricFeatures", "FEATURE_NAMES", "N_FEATURES", "VOCAB_WINDOW",
    "describe",
]
