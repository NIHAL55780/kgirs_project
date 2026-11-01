"""Run every experimental condition over the 100-question MuSiQue sample.

Results go to results/<model>/<condition>.jsonl (one line per question). Runs are resumable:
questions already present in a results file are skipped.

Examples:
    python run_experiments.py                                  # all conditions
    python run_experiments.py --conditions search_only routed_react --limit 5
"""
import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import config
from agent.llm import LLM, DailyLimitReached, FatalLLMError
from agent.react_agent import ReActAgent, RouterAgent, closed_book
from evaluation.metrics import score

ALL_TOOLS = ["search", "wikidata", "calculator"]
CONDITIONS = {
    "closed_book": "No tools (LLM memory only) - baseline",
    "search_only": "Single tool: search",
    "wikidata_only": "Single tool: Wikidata",
    "routed_react": "All tools, the ReAct agent routes implicitly at every step",
    "routed_router": "All tools, an explicit router LLM picks the tool per sub-question",
    "calculator_only": "Single tool: calculator",
    "routed_router_v2": "All tools, explicit router v2 (only the router may finish, keeps a plan, full observations)",
}


def make_runner(condition, llm, search_backend):
    if condition == "closed_book":
        return lambda q: closed_book(q, llm)
    if condition.endswith("_only"):
        return ReActAgent(llm, [condition[: -len("_only")]], search_backend).run
    if condition == "routed_react":
        return ReActAgent(llm, ALL_TOOLS, search_backend).run
    if condition == "routed_router":
        return RouterAgent(llm, search_backend).run
    if condition == "routed_router_v2":
        return RouterAgent(llm, search_backend, version=2).run
    raise ValueError(condition)


def model_slug():
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{config.LLM_PROVIDER}_{config.LLM_MODEL}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel questions; keep low on free tiers (tokens-per-minute limits)")
    ap.add_argument("--search-backend", default=config.SEARCH_BACKEND, choices=["paragraphs", "wikipedia"])
    args = ap.parse_args()

    questions = [json.loads(l) for l in open(config.SAMPLE_FILE, encoding="utf-8")]
    if args.limit:
        questions = questions[: args.limit]

    out_dir = config.RESULTS_DIR / model_slug()
    if args.search_backend != "paragraphs":
        out_dir = out_dir.with_name(out_dir.name + f"__search-{args.search_backend}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model: {config.LLM_PROVIDER} / {config.LLM_MODEL} | search backend: {args.search_backend}")
    print(f"Writing to {out_dir}")

    try:
        llm = LLM()
    except FatalLLMError as exc:
        sys.exit(f"\nERROR: {exc}")
    stop = threading.Event()
    fatal = []
    for cond in args.conditions:
        path = out_dir / f"{cond}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")}
        todo = [q for q in questions if q["id"] not in done]
        print(f"\n== {cond}: {CONDITIONS[cond]} ({len(done)} done, {len(todo)} to go)")
        if not todo:
            continue

        runner = make_runner(cond, llm, args.search_backend)
        em_sum = n = failed = 0

        def work(q):
            if stop.is_set():
                return q, None, None
            try:
                res = runner(q)
            except DailyLimitReached as exc:
                stop.set()
                return q, None, f"daily limit: {exc}"
            except FatalLLMError as exc:
                fatal.append(str(exc))
                stop.set()
                return q, None, None
            except Exception as exc:  # noqa: BLE001
                return q, None, f"{exc.__class__.__name__}: {exc}"
            golds = [q["answer"]] + q.get("answer_aliases", [])
            return q, {
                "id": q["id"], "hops": int(q["id"][0]), "question": q["question"],
                "gold": q["answer"], "aliases": q.get("answer_aliases", []),
                "condition": cond, **score(res.get("prediction", ""), golds), **res,
            }, None

        with ThreadPoolExecutor(args.workers) as pool, open(path, "a", encoding="utf-8") as f:
            futures = [pool.submit(work, q) for q in todo]
            bar = tqdm(as_completed(futures), total=len(futures), desc=cond)
            for fut in bar:
                q, rec, err = fut.result()
                if rec is None:
                    if err:
                        failed += 1
                        tqdm.write(f"  not saved (will retry on next run) {q['id']}: {err[:200]}")
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                em_sum += rec["em"]
                n += 1
                bar.set_postfix(EM=f"{em_sum / n:.2f}", tokens=llm.session_tokens)

        if fatal:
            sys.exit(f"\nERROR: {fatal[0]}\nNothing was saved for the failed questions; fix this and re-run.")
        if stop.is_set():
            print(f"\nDaily quota reached after {llm.session_calls} LLM calls / {llm.session_tokens:,} tokens "
                  "this session. Progress is saved; run the same command again later to resume.")
            return
        if failed:
            print(f"  {failed} question(s) failed and will be retried when you re-run.")

    print(f"\nDone ({llm.session_calls} LLM calls / {llm.session_tokens:,} tokens this session). "
          "Run `python analyze.py` to build the comparison report.")


if __name__ == "__main__":
    main()
