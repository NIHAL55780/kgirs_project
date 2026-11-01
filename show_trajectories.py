"""Print agent trajectories for error analysis.

    python show_trajectories.py routed_react            # all questions of a condition
    python show_trajectories.py routed_router --wrong   # only questions with EM = 0
"""
import argparse
import json
import sys

import config


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("condition")
    ap.add_argument("--run", help="results sub-folder (defaults to the most recently modified)")
    ap.add_argument("--wrong", action="store_true", help="only show questions answered incorrectly")
    ap.add_argument("--chars", type=int, default=200, help="observation characters to show")
    args = ap.parse_args()

    runs = sorted((d for d in config.RESULTS_DIR.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    run_dir = config.RESULTS_DIR / args.run if args.run else runs[-1]
    for line in open(run_dir / f"{args.condition}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if args.wrong and r["em"] == 1:
            continue
        print("=" * 100)
        print(f"{r['id']}\nQ: {r['question']}")
        print(f"GOLD: {r['gold']} | PRED: {r['prediction']} | EM {r['em']:.0f} F1 {r['f1']:.2f} | "
              f"tools {r.get('tool_calls')} | tokens {r.get('prompt_tokens', 0) + r.get('completion_tokens', 0)}")
        for s in r.get("trajectory", []):
            if "router" in s:
                print(f"  router: {s['router'].strip()!r}")
            obs = str(s.get("observation", ""))[: args.chars].replace("\n", " | ")
            print(f"  [{s.get('step')}] {s.get('action')}[{s.get('input')}] -> {obs}")


if __name__ == "__main__":
    main()
