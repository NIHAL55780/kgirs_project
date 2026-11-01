"""Quick manual check that every tool works (no LLM needed)."""
import json
import sys

from config import SAMPLE_FILE
from tools import build_tools

sys.stdout.reconfigure(encoding="utf-8")

q = json.loads(open(SAMPLE_FILE, encoding="utf-8").readline())
tools = build_tools(["search", "wikidata", "calculator"], q, "paragraphs")
print("Question:", q["question"], "| gold:", q["answer"])
print("Decomposition:", [d["question"] for d in q["question_decomposition"]])

checks = [
    ("calculator", "2024 - 1867"),
    ("calculator", "__import__('os')"),
    ("search", q["question_decomposition"][0]["question"].replace(">>", "")),
    ("wikidata", "Steve Hillage | spouse"),
    ("wikidata", "Paris | country"),
    ("wikidata", "Christopher Nolan | born"),
]
for name, arg in checks:
    print(f"\n=== {name}[{arg}]")
    print(tools[name](arg)[:800])
