"""Independent, behaviour-level metrics on generated text.

These do NOT depend on the activation direction that builds the mask, so they
carry the real claim (unlike the ``RoleSensor`` diagnostic).
"""
from __future__ import annotations

import re
from math import sqrt

from data import PIRATE_MARKERS, META_MARKERS

_DROP_G = re.compile(r"\b\w+in'\b")   # takin', sailin'
_OF = re.compile(r"\bo'\b")           # o'
_YE = re.compile(r"\bye\b")           # ye as a standalone word
_ARR = re.compile(r"\barr+\b")        # arr, arrr, arrrr ... ("arr" is NOT in PIRATE_MARKERS)

# Pre-compiled WORD-BOUNDARY patterns for every marker. Using \b avoids substring
# false positives that an earlier .count() had ("aye" in "maybe", "arr" in "narrative").
_MARKER_RE = [re.compile(r"\b" + re.escape(m.strip()) + r"\b") for m in PIRATE_MARKERS]


def pirate_lexicon(text: str, per_100_words: bool = True) -> float:
    """Surface pirate-marker rate in ``text`` (independent of the mask subspace).

    Word-boundary matched (no substring false positives) and normalized per 100 words
    so that shorter masked/disclaimer outputs are not scored as "more suppressed" merely
    for being shorter. Set ``per_100_words=False`` for the raw count.
    """
    low = text.lower()
    n_words = max(len(re.findall(r"[a-z']+", low)), 1)
    count = sum(len(p.findall(low)) for p in _MARKER_RE)
    count += (len(_DROP_G.findall(low)) + len(_OF.findall(low))
              + len(_YE.findall(low)) + len(_ARR.findall(low)))
    return 100.0 * count / n_words if per_100_words else count


def is_meta(text: str) -> bool:
    """True if the model reverted to an 'I am an AI' disclaimer (expected cascade)."""
    low = text.lower()
    return any(m in low for m in META_MARKERS)


def lexical_diversity(text: str) -> float:
    """Type/token ratio (rises mechanically for shorter texts — interpret with care)."""
    words = re.findall(r"[a-z']+", text.lower())
    return len(set(words)) / max(len(words), 1)


def cohens_d(a, b) -> float:
    """Standardized mean difference between two 1-D arrays."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a.mean() - b.mean()) / sqrt((a.var() + b.var()) / 2 + 1e-9)


def clustered_paired_ci(x_a, x_b, n_prompts, n_seeds):
    """95% CI on a paired difference, with the PROMPT as the independent unit.

    Inputs are flat arrays of length ``n_prompts * n_seeds`` ordered prompt-major
    (all seeds of prompt 0, then prompt 1, ...). Averaging seeds within each prompt
    first gives ``n_prompts`` independent values — the correct unit for the CLT here.
    Treating all ``n_prompts*n_seeds`` samples as independent understates the CI
    (seeds of one prompt are correlated and share the same controller lambda).

    Returns ``(mean_diff, ci95_halfwidth, n_independent)``.
    """
    import numpy as np
    a = np.asarray(x_a, float).reshape(n_prompts, n_seeds).mean(1)
    b = np.asarray(x_b, float).reshape(n_prompts, n_seeds).mean(1)
    d = a - b
    se = d.std(ddof=1) / sqrt(n_prompts)
    return float(d.mean()), float(1.96 * se), n_prompts
