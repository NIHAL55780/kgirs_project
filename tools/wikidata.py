"""Wikidata knowledge-graph tool.

Input: "entity | property"  (property optional), entity may be a name or a QID like Q42.
  1. wbsearchentities resolves the name to candidate QIDs.
  2. A SPARQL query fetches every direct (truthy) statement of the chosen entity with labels.
  3. Statements are filtered to those whose property label matches the requested property.
"""
import re
import time

import requests

from cache import Cache
from config import HTTP_HEADERS

DESCRIPTION = (
    "wikidata[entity | property]: looks up structured facts in the Wikidata knowledge graph. "
    "Give an entity name (or a QID such as Q42) and optionally a property, e.g. "
    "wikidata[Steve Hillage | spouse], wikidata[Paris | country], wikidata[Q90 | population]. "
    "Without a property it lists the entity's facts. If the wrong entity is picked, retry with "
    "a more specific name or with the QID shown among the other matches."
)

_API = "https://www.wikidata.org/w/api.php"
_SPARQL = "https://query.wikidata.org/sparql"
_cache = Cache("wikidata")
_QID = re.compile(r"^Q\d+$", re.I)

_PROP_SYNONYMS = {
    "wife": "spouse", "husband": "spouse", "married": "spouse",
    "born": "date of birth", "birthday": "date of birth", "birthplace": "place of birth",
    "died": "date of death", "death": "date of death",
    "capital city": "capital", "located": "located in the administrative territorial entity",
    "located in": "located in the administrative territorial entity",
    "director": "director", "directed": "director",
    "singer": "performer", "artist": "performer", "band": "performer",
    "author": "author", "writer": "author", "wrote": "author",
    "founded": "inception", "founder": "founded by", "established": "inception",
    "owner": "owned by", "owns": "owned by", "parent company": "parent organization",
    "headquarters": "headquarters location", "hq": "headquarters location",
    "nationality": "country of citizenship", "citizenship": "country of citizenship",
    "father": "father", "mother": "mother", "child": "child", "children": "child",
    "label": "record label", "record company": "record label",
    "team": "member of sports team", "league": "league", "employer": "employer",
    "education": "educated at", "school": "educated at", "university": "educated at",
    "producer": "producer", "composer": "composer", "creator": "creator",
    "mouth": "mouth of the watercourse", "flows into": "mouth of the watercourse",
    "continent": "continent", "country": "country", "head of state": "head of state",
    "president": "head of state", "prime minister": "head of government", "mayor": "head of government",
    "leader": "head of government", "language": "official language",
}


def _get(url, params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Wikidata rate limited")


def _search_entities(name: str, limit: int = 5):
    key = Cache.key("search", name, limit)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    data = _get(_API, {
        "action": "wbsearchentities", "search": name, "language": "en",
        "uselang": "en", "type": "item", "limit": limit, "format": "json",
    })
    out = [
        {"id": e["id"], "label": e.get("label", ""), "description": e.get("description", "")}
        for e in data.get("search", [])
    ]
    _cache.set(key, out)
    return out


def _entity_label(qid: str):
    key = Cache.key("label", qid)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    data = _get(_API, {
        "action": "wbgetentities", "ids": qid, "props": "labels|descriptions",
        "languages": "en", "format": "json",
    })
    ent = data.get("entities", {}).get(qid, {})
    out = {
        "id": qid,
        "label": ent.get("labels", {}).get("en", {}).get("value", qid),
        "description": ent.get("descriptions", {}).get("en", {}).get("value", ""),
    }
    _cache.set(key, out)
    return out


def _facts(qid: str):
    key = Cache.key("facts", qid)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    query = f"""
    SELECT ?propLabel ?valueLabel WHERE {{
      wd:{qid} ?p ?value .
      ?prop wikibase:directClaim ?p .
      FILTER(!isLiteral(?value) || lang(?value) = "" || langMatches(lang(?value), "en"))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT 600"""
    data = _get(_SPARQL, {"query": query, "format": "json"})
    facts = {}
    for b in data["results"]["bindings"]:
        prop = b["propLabel"]["value"]
        val = b["valueLabel"]["value"]
        if re.match(r"^\d{4}-\d\d-\d\dT", val):
            val = val[:10].lstrip("+")
        if prop.startswith("P") and prop[1:].isdigit():
            continue
        if re.match(r"^(Q\d+|https?://)", val) or "ID" in prop.split() or prop.endswith(" ID"):
            continue
        facts.setdefault(prop, [])
        if val not in facts[prop] and len(facts[prop]) < 15:
            facts[prop].append(val)
    _cache.set(key, facts)
    return facts


def _match_props(facts: dict, wanted: str):
    w = wanted.lower().strip()
    targets = {w, _PROP_SYNONYMS.get(w, w)}
    exact = [p for p in facts if p.lower() in targets]
    if exact:
        return exact
    words = set(re.findall(r"\w+", " ".join(targets))) - {"of", "the", "in", "a"}
    scored = []
    for p in facts:
        pw = set(re.findall(r"\w+", p.lower()))
        overlap = len(words & pw)
        if overlap or any(t in p.lower() for t in targets):
            scored.append((overlap + (2 if any(t in p.lower() for t in targets) else 0), p))
    scored.sort(reverse=True)
    return [p for _, p in scored[:5]]


def wikidata(query: str) -> str:
    entity, _, prop = (s.strip() for s in query.partition("|"))
    if not entity:
        return "Provide an entity name, e.g. wikidata[Barack Obama | spouse]."
    try:
        if _QID.match(entity):
            chosen = _entity_label(entity.upper())
            others = []
        else:
            cands = _search_entities(entity)
            if not cands:
                return f"No Wikidata entity found for '{entity}'. Try a different or more complete name."
            chosen, others = cands[0], cands[1:]
        facts = _facts(chosen["id"])
    except Exception as exc:  # noqa: BLE001
        return f"Wikidata request failed ({exc.__class__.__name__}). Try again."

    header = f"Entity: {chosen['label']} ({chosen['id']}) - {chosen['description'] or 'no description'}"
    if others:
        header += "\nOther matches: " + "; ".join(
            f"{o['label']} ({o['id']}) - {o['description'] or 'no description'}" for o in others
        )
    if not facts:
        return header + "\nNo facts found for this entity."

    if prop:
        props = _match_props(facts, prop)
        if props:
            body = "\n".join(f"{p}: {', '.join(facts[p])}" for p in props)
            return f"{header}\nFacts matching '{prop}':\n{body}"
        available = ", ".join(sorted(facts)[:60])
        return f"{header}\nNo property matching '{prop}'. Available properties: {available}"

    body = "\n".join(f"{p}: {', '.join(v[:5])}" for p, v in list(facts.items())[:40])
    return f"{header}\nFacts:\n{body}"
