"""Agents for the experiment.

* closed_book       - no tools, single LLM call.
* ReActAgent        - Thought/Action/Observation loop over a given tool set. With one tool this is
                      the "single tool" condition; with all tools the LLM routes implicitly.
* RouterAgent       - explicit routing: a router LLM call picks the tool for the next sub-question,
                      then an executor LLM call (seeing only that tool) writes the tool input.
"""
import re
import time
from collections import Counter

import config
from agent import prompts
from agent.llm import LLM
from tools import build_tools

_ACTION = re.compile(r"Action\s*:\s*([A-Za-z_]+)\s*\[(.*?)(?:\]\s*$|\]\s*\n|$)", re.S | re.M)
_OBS_LIMIT = 1500


def parse_action(text: str):
    matches = _ACTION.findall(text)
    if not matches:
        return None, None
    tool, arg = matches[-1]
    return tool.strip().lower(), arg.strip().rstrip("]").strip()


def clean_answer(ans: str) -> str:
    ans = ans.strip().strip('"\'`*').strip()
    ans = re.sub(r"^(the answer is|answer:)\s*", "", ans, flags=re.I)
    return ans.rstrip(".").strip()


class Tracker:
    def __init__(self):
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tool_calls = Counter()
        self.invalid_actions = 0
        self.repeated_queries = 0
        self.seen_queries = set()
        self.trajectory = []
        self.start = time.time()

    def add_llm(self, resp):
        self.llm_calls += 1
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens

    def result(self, answer, finished, steps, **extra):
        return {
            "prediction": clean_answer(answer or ""),
            "finished": finished,
            "steps": steps,
            "llm_calls": self.llm_calls,
            "tool_calls": dict(self.tool_calls),
            "num_tool_calls": sum(self.tool_calls.values()),
            "invalid_actions": self.invalid_actions,
            "repeated_queries": self.repeated_queries,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_s": round(time.time() - self.start, 2),
            "trajectory": self.trajectory,
            **extra,
        }


def run_tool(tools, name, arg, tracker):
    tracker.tool_calls[name] += 1
    key = (name, " ".join(sorted(set(re.findall(r"\w+", arg.lower())))))
    if key in tracker.seen_queries:
        tracker.repeated_queries += 1
        return ("You already ran this query and the result has not changed. Do not repeat it: "
                "try a different query or tool, or finish with your best answer.")
    tracker.seen_queries.add(key)
    try:
        obs = tools[name](arg)
    except Exception as exc:  # noqa: BLE001
        obs = f"Tool error: {exc}"
    return obs[:_OBS_LIMIT]


def closed_book(question: dict, llm: LLM) -> dict:
    t = Tracker()
    messages = [
        {"role": "system", "content": prompts.CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": f"Question: {question['question']}"},
    ]
    resp = llm.chat(messages)
    t.add_llm(resp)
    t.trajectory.append({"llm": resp.text})
    m = re.findall(r"Answer\s*:\s*(.+)", resp.text)
    if not m:
        resp = llm.chat(messages + [
            {"role": "assistant", "content": resp.text},
            {"role": "user", "content": prompts.CLOSED_BOOK_FOLLOWUP},
        ])
        t.add_llm(resp)
        t.trajectory.append({"llm": resp.text})
        m = re.findall(r"Answer\s*:\s*(.+)", resp.text)
    answer = m[-1] if m else resp.text.strip().splitlines()[-1] if resp.text.strip() else ""
    return t.result(answer, finished=bool(m), steps=1)


class ReActAgent:
    def __init__(self, llm: LLM, tool_names, search_backend=None, max_steps=None):
        self.llm = llm
        self.tool_names = list(tool_names)
        self.search_backend = search_backend or config.SEARCH_BACKEND
        self.max_steps = max_steps or config.MAX_STEPS
        self.system = prompts.react_system(self.tool_names)

    def run(self, question: dict) -> dict:
        tools = build_tools(self.tool_names, question, self.search_backend)
        t = Tracker()
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": f"Question: {question['question']}"},
        ]
        for step in range(1, self.max_steps + 1):
            resp = self.llm.chat(messages, stop=["Observation:", "\nObservation"])
            t.add_llm(resp)
            text = resp.text.strip()
            tool, arg = parse_action(text)
            messages.append({"role": "assistant", "content": text})

            if tool == "finish":
                t.trajectory.append({"step": step, "llm": text, "action": "finish", "input": arg})
                return t.result(arg, finished=True, steps=step)

            if tool in tools:
                obs = run_tool(tools, tool, arg, t)
            else:
                t.invalid_actions += 1
                allowed = ", ".join(self.tool_names + ["finish"])
                obs = (f"Invalid action. Use exactly 'Action: <tool>[<input>]' with one of: {allowed}."
                       if tool is None else f"Tool '{tool}' is not available. Use one of: {allowed}.")
            t.trajectory.append({"step": step, "llm": text, "action": tool, "input": arg, "observation": obs})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        messages.append({"role": "user", "content":
                         "Step limit reached. Reply now with your best final answer as "
                         "'Action: finish[<answer>]'."})
        resp = self.llm.chat(messages)
        t.add_llm(resp)
        tool, arg = parse_action(resp.text)
        answer = arg if tool == "finish" else resp.text.strip().splitlines()[-1] if resp.text.strip() else ""
        t.trajectory.append({"step": "forced", "llm": resp.text, "action": "finish", "input": answer})
        return t.result(answer, finished=False, steps=self.max_steps)


class RouterAgent:
    TOOLS = ["search", "wikidata", "calculator"]

    def __init__(self, llm: LLM, search_backend=None, max_steps=None):
        self.llm = llm
        self.search_backend = search_backend or config.SEARCH_BACKEND
        self.max_steps = max_steps or config.MAX_STEPS

    @staticmethod
    def _render(question, history):
        lines = [f"Question: {question['question']}", "", "Evidence gathered so far:"]
        if not history:
            lines.append("(none yet)")
        for i, h in enumerate(history, 1):
            lines.append(f"{i}. Sub-question: {h['subq']}\n   {h['tool']}[{h['input']}]\n   Result: {h['observation']}")
        return "\n".join(lines)

    def _finish(self, question, history, t):
        resp = self.llm.chat([
            {"role": "system", "content": prompts.actor_system("finish")},
            {"role": "user", "content": self._render(question, history)},
        ])
        t.add_llm(resp)
        tool, arg = parse_action(resp.text)
        answer = arg if tool == "finish" else resp.text.strip().splitlines()[-1] if resp.text.strip() else ""
        return answer, resp.text

    def run(self, question: dict) -> dict:
        tools = build_tools(self.TOOLS, question, self.search_backend)
        t = Tracker()
        history = []
        routes = []
        for step in range(1, self.max_steps + 1):
            context = self._render(question, history)
            r = self.llm.chat([
                {"role": "system", "content": prompts.ROUTER_SYSTEM},
                {"role": "user", "content": context},
            ])
            t.add_llm(r)
            route = re.findall(r"Tool\s*:\s*\**\s*([A-Za-z_]+)", r.text)
            route = route[-1].lower() if route else "search"
            if route not in self.TOOLS + ["finish"]:
                t.invalid_actions += 1
                route = "search"
            subq = re.findall(r"Next\s*:\s*(.+)", r.text)
            subq = subq[-1].strip() if subq else question["question"]
            routes.append(route)

            if route == "finish":
                answer, text = self._finish(question, history, t)
                t.trajectory.append({"step": step, "router": r.text, "action": "finish", "input": answer, "llm": text})
                return t.result(answer, finished=True, steps=step, routes=routes)

            a = self.llm.chat([
                {"role": "system", "content": prompts.actor_system(route)},
                {"role": "user", "content": f"{context}\n\nNext sub-question: {subq}"},
            ], stop=["Observation:"])
            t.add_llm(a)
            tool, arg = parse_action(a.text)
            if tool == "finish":
                t.trajectory.append({"step": step, "router": r.text, "llm": a.text, "action": "finish", "input": arg})
                return t.result(arg, finished=True, steps=step, routes=routes)
            if tool != route:
                t.invalid_actions += 1
                tool, arg = route, arg or subq
            obs = run_tool(tools, tool, arg, t)
            history.append({"subq": subq, "tool": tool, "input": arg, "observation": obs[:800]})
            t.trajectory.append({"step": step, "router": r.text, "llm": a.text, "action": tool,
                                 "input": arg, "observation": obs})

        answer, text = self._finish(question, history, t)
        t.trajectory.append({"step": "forced", "llm": text, "action": "finish", "input": answer})
        return t.result(answer, finished=False, steps=self.max_steps, routes=routes)
