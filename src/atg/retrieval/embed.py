"""Deterministic local embeddings, plus an optional Vertex embedding backend.

The local embedder is a signed-hash random projection of a tf-idf space over word
unigrams and character 4-grams. It is not a learned model and does not know that
"reimburse" and "refund" are related unless the synonym table says so, but the
character n-grams give it real morphological tolerance that exact-match BM25 does
not have, which is the axis the comparison is actually about. Nothing here needs
network access, so the default measurement path runs offline.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from atg.catalog import Catalog
from atg.retrieval.text import char_ngrams, tokenize

DIM = 512
_NGRAM = 4
_NGRAM_WEIGHT = 0.45


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


def _hash_slot(feature: str) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    value = struct.unpack("<Q", digest)[0]
    return value % DIM, 1.0 if (value >> 63) & 1 else -1.0


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


@dataclass(slots=True)
class HashingEmbedder:
    """Signed-hash projection with corpus idf weighting."""

    idf: dict[str, float]
    name: str = "local-hash-512"

    def _features(self, text: str) -> dict[str, float]:
        feats: dict[str, float] = {}
        for term in tokenize(text):
            feats[f"w:{term}"] = feats.get(f"w:{term}", 0.0) + 1.0
        for gram in char_ngrams(text, _NGRAM):
            key = f"g:{gram}"
            feats[key] = feats.get(key, 0.0) + _NGRAM_WEIGHT
        return feats

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * DIM
            for feature, tf in self._features(text).items():
                slot, sign = _hash_slot(feature)
                weight = (1.0 + math.log(tf)) * self.idf.get(feature, self._default_idf)
                vec[slot] += sign * weight
            out.append(_normalize(vec))
        return out

    @property
    def _default_idf(self) -> float:
        return 1.0


def fit_hashing_embedder(catalog: Catalog) -> HashingEmbedder:
    """Compute idf over the catalog's retrieval text."""
    docs = [t.retrieval_text() for t in catalog.tools]
    df: dict[str, int] = {}
    for doc in docs:
        seen = set()
        for term in tokenize(doc):
            seen.add(f"w:{term}")
        for gram in char_ngrams(doc, _NGRAM):
            seen.add(f"g:{gram}")
        for feature in seen:
            df[feature] = df.get(feature, 0) + 1
    n = max(1, len(docs))
    idf = {feature: math.log(1.0 + n / count) for feature, count in df.items()}
    return HashingEmbedder(idf=idf)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class EmbeddingIndex:
    """Dense index. Brute force, because 1200 rows times 512 floats is nothing."""

    def __init__(self, catalog: Catalog, embedder: Embedder) -> None:
        self.catalog = catalog
        self.embedder = embedder
        self.names = [t.name for t in catalog.tools]
        self.vectors = embedder.encode([t.retrieval_text() for t in catalog.tools])

    def score(self, query: str) -> dict[str, float]:
        qv = self.embedder.encode([query])[0]
        return {name: cosine(qv, vec) for name, vec in zip(self.names, self.vectors)}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scored = self.score(query)
        return sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


@dataclass(slots=True)
class VertexEmbedder:
    """Optional hosted embeddings. Never required: tests never construct this."""

    model: str = "text-embedding-005"
    name: str = "vertex-text-embedding-005"
    _cache: dict[str, list[float]] | None = None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        from atg.gemini_adapter import build_client

        if self._cache is None:
            self._cache = {}
        client = build_client()
        missing = [t for t in texts if t not in self._cache]
        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            resp = client.models.embed_content(model=self.model, contents=list(batch))
            for text, emb in zip(batch, resp.embeddings):
                self._cache[text] = _normalize(list(emb.values))
        return [self._cache[t] for t in texts]


def cached_embedder(inner: Embedder, store: dict[str, list[float]]) -> Embedder:
    """Wrap an embedder in a dict cache so recorded vectors can replay offline."""

    class _Cached:
        name = inner.name

        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            missing = [t for t in texts if t not in store]
            if missing:
                for text, vec in zip(missing, inner.encode(missing)):
                    store[text] = vec
            return [store[t] for t in texts]

    return _Cached()


def replay_embedder(store: dict[str, list[float]], name: str) -> Embedder:
    """An embedder that can only answer from a recorded fixture."""

    class _Replay:
        def __init__(self) -> None:
            self.name = name

        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            out = []
            for text in texts:
                if text not in store:
                    raise KeyError(f"no recorded embedding for {text[:60]!r}")
                out.append(store[text])
            return out

    return _Replay()


MakeEmbedder = Callable[[Catalog], Embedder]
