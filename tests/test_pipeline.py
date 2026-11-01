"""Offline end-to-end check of the agent plumbing with a scripted fake LLM (no API key needed).

    python -m tests.test_pipeline
"""
import json
import sys

from agent.llm import LLMResponse
from agent.react_agent import ReActAgent, RouterAgent, closed_book, parse_action
from config import SAMPLE_FILE
from evaluation.metrics import score


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, stop=None):
        return LLMResponse(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=5)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    q = json.loads(open(SAMPLE_FILE, encoding="utf-8").readline())

    assert parse_action("Thought: x\nAction: search[foo bar]") == ("search", "foo bar")
    assert parse_action("Action: calculator[round(max(1, 2))]") == ("calculator", "round(max(1, 2))")
    assert parse_action("Action: finish[Paris") == ("finish", "Paris")
    assert score("the Paris.", ["paris"]) == {"em": 1.0, "f1": 1.0}

    r = closed_book(q, ScriptedLLM(["Thinking...\nAnswer: 1235"]))
    assert r["prediction"] == "1235", r

    llm = ScriptedLLM([
        "Thought: find host\nAction: search[tournament host]",
        "Thought: check KG\nAction: wikidata[Burma | official name]",
        "Thought: math\nAction: calculator[2024 - 1235]",
        "Thought: done\nAction: finish[1235]",
    ])
    r = ReActAgent(llm, ["search", "wikidata", "calculator"], "paragraphs").run(q)
    assert r["prediction"] == "1235" and r["tool_calls"] == {"search": 1, "wikidata": 1, "calculator": 1}, r
    print("routed_react trajectory:")
    for s in r["trajectory"]:
        print(f"  {s['action']}[{s['input']}] -> {str(s.get('observation', ''))[:100]!r}")

    r = ReActAgent(ScriptedLLM(["Action: wikidata[x]", "Action: finish[a]"]), ["search"], "paragraphs").run(q)
    assert r["invalid_actions"] == 1 and r["tool_calls"] == {}, r

    llm = ScriptedLLM([
        "Next: who hosted?\nTool: search", "Thought: t\nAction: search[tournament host]",
        "Next: done\nTool: finish", "Thought: t\nAction: finish[1235]",
    ])
    r = RouterAgent(llm, "paragraphs").run(q)
    assert r["prediction"] == "1235" and r["routes"] == ["search", "finish"], r

    llm = ScriptedLLM([
        "Plan: none\nNext: who hosted?\nTool: search", "Thought: I know it\nAction: finish[wrong]",
        "Plan: done\nNext: none\nTool: finish", "Thought: t\nAction: finish[1235]",
    ])
    r = RouterAgent(llm, "paragraphs", version=2).run(q)
    assert r["prediction"] == "1235" and r["tool_calls"] == {"search": 1} and r["invalid_actions"] == 1, r

    print("All pipeline checks passed.")


if __name__ == "__main__":
    main()
