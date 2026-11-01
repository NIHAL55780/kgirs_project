from tools import calculator as _calc
from tools import search as _search
from tools import wikidata as _wd

TOOL_DESCRIPTIONS = {
    "search": _search.DESCRIPTION,
    "wikidata": _wd.DESCRIPTION,
    "calculator": _calc.DESCRIPTION,
}


def build_tools(names, question, search_backend):
    """Return {tool_name: callable(str) -> str} for one question."""
    registry = {}
    for name in names:
        if name == "search":
            registry[name] = _search.make_search_tool(search_backend, question)
        elif name == "wikidata":
            registry[name] = _wd.wikidata
        elif name == "calculator":
            registry[name] = _calc.calculator
        else:
            raise ValueError(f"unknown tool: {name}")
    return registry
