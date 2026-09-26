"""Offline token counter, fitted against Vertex `count_tokens` on real renderings.

Token accounting is the whole point of the schema-compression measurement, so an
estimator that is wrong in a format-dependent way would invent the result. This
one is a structural approximation of a byte-pair vocabulary rather than a
character-count heuristic: short alphabetic runs cost one token, longer ones pay
per extra chunk, and runs of punctuation merge the way `","` and `":{"` do in a
real vocabulary.

The six constants were fitted by grid search on 224 recorded renderings
(`tests/fixtures/token_calibration.json`, gemini-2.5-flash-lite, 2026-09-25) and
score 3.6% mean absolute error there, 4.0% on a held-out split. `atg calibrate`
recomputes both numbers offline from the fixture.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

_RUN = re.compile(r"[A-Za-z]+|\d+|\n+|[ \t]+|[^\sA-Za-z\d]+")

_FREE_CHARS = 9  # alphabetic run length that still costs a single token
_CHUNK = 6  # characters per additional token beyond that
_DIGITS_PER_TOKEN = 3
_PUNCT_FIRST = 1.0  # first character of a punctuation run
_PUNCT_EXTRA = 0.1  # each subsequent one, since runs merge
_NEWLINE = 1.5


@lru_cache(maxsize=16384)
def count_tokens(text: str) -> int:
    """Approximate model token count for a string."""
    total = 0.0
    for run in _RUN.findall(text):
        head = run[0]
        if head.isalpha():
            length = len(run)
            total += 1 if length <= _FREE_CHARS else 1 + math.ceil((length - _FREE_CHARS) / _CHUNK)
        elif head.isdigit():
            total += math.ceil(len(run) / _DIGITS_PER_TOKEN)
        elif head == "\n":
            total += _NEWLINE * len(run)
        elif head in " \t":
            continue
        else:
            total += _PUNCT_FIRST + _PUNCT_EXTRA * (len(run) - 1)
    return int(round(total))


def count_tokens_real(text: str, *, model: str | None = None) -> int:
    """True token count from Vertex. Needs credentials; used only for calibration."""
    from atg.gemini_adapter import DEFAULT_MODEL, build_client

    client = build_client()
    resp = client.models.count_tokens(model=model or DEFAULT_MODEL, contents=text)
    return int(resp.total_tokens)
