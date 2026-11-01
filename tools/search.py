"""Text search tool.

Two backends:
  * paragraphs - BM25 over the 20 Wikipedia paragraphs that ship with each MuSiQue question
                 (the standard, reproducible MuSiQue "distractor" setting).
  * wikipedia  - live MediaWiki full-text search, returning the intro of the top pages.
"""
import re

import requests
from rank_bm25 import BM25Okapi

from cache import Cache
from config import HTTP_HEADERS, SEARCH_TOP_K

DESCRIPTION = (
    "search[query]: searches Wikipedia text and returns the most relevant paragraphs. "
    "Use short, specific queries (an entity name plus the relation you need), e.g. "
    "search[Steve Hillage spouse]."
)

_cache = Cache("wikipedia_search")
_TOKEN = re.compile(r"\w+")
_STOP = set(
    "a an the of in on at to for by with from and or is was are were be been who whom whose what "
    "which when where why how did does do that this as it its".split()
)


def _tok(text: str):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class ParagraphSearch:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
        docs = [_tok(p["title"] + " " + p["title"] + " " + p["paragraph_text"]) for p in paragraphs]
        self.bm25 = BM25Okapi(docs)

    def __call__(self, query: str, k: int = SEARCH_TOP_K) -> str:
        q = _tok(query)
        if not q:
            return "Empty query. Provide keywords to search for."
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return "\n\n".join(
            f"[{n + 1}] {self.paragraphs[i]['title']}: {self.paragraphs[i]['paragraph_text']}"
            for n, i in enumerate(ranked)
        )


_API = "https://en.wikipedia.org/w/api.php"


def _wiki_get(params):
    r = requests.get(_API, params={**params, "format": "json"}, headers=HTTP_HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def wikipedia_search(query: str, k: int = SEARCH_TOP_K) -> str:
    key = Cache.key(query, k)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    try:
        hits = _wiki_get({"action": "query", "list": "search", "srsearch": query, "srlimit": k})
        titles = [h["title"] for h in hits["query"]["search"]]
        if not titles:
            out = "No results. Try different keywords."
        else:
            pages = _wiki_get({
                "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                "titles": "|".join(titles), "redirects": 1,
            })["query"]["pages"]
            extracts = {p["title"]: p.get("extract", "") for p in pages.values()}
            out = "\n\n".join(
                f"[{n + 1}] {t}: {extracts.get(t, '')[:1200]}" for n, t in enumerate(titles)
            )
    except requests.RequestException as exc:
        return f"Search failed ({exc.__class__.__name__}). Try again or rephrase."
    _cache.set(key, out)
    return out


def make_search_tool(backend: str, question: dict):
    if backend == "paragraphs":
        return ParagraphSearch(question["paragraphs"])
    if backend == "wikipedia":
        return wikipedia_search
    raise ValueError(f"unknown search backend: {backend}")
