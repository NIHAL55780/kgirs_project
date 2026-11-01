"""Build the single-tool vs routed comparison: summary tables, per-hop breakdown, tool usage,
complementarity (oracle routing upper bound), charts, and a markdown report."""
import argparse
import json
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

ORDER = ["closed_book", "search_only", "wikidata_only", "calculator_only", "routed_react", "routed_router"]
SINGLE = ["search_only", "wikidata_only", "calculator_only"]
ROUTED = ["routed_react", "routed_router"]
TOOLS = ["search", "wikidata", "calculator"]


def load(run_dir):
    recs = []
    for cond in ORDER:
        p = run_dir / f"{cond}.jsonl"
        if p.exists():
            recs += [json.loads(l) for l in open(p, encoding="utf-8")]
    return pd.DataFrame(recs)


def pct(x):
    return round(100 * x, 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="results sub-folder (defaults to the most recently modified)")
    args = ap.parse_args()

    runs = sorted((d for d in config.RESULTS_DIR.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    if not runs:
        sys.exit("No results found. Run run_experiments.py first.")
    run_dir = config.RESULTS_DIR / args.run if args.run else runs[-1]
    df = load(run_dir)
    conds = [c for c in ORDER if c in set(df["condition"])]
    df["condition"] = pd.Categorical(df["condition"], conds, ordered=True)
    for t in TOOLS:
        df[f"calls_{t}"] = df["tool_calls"].apply(lambda d: (d or {}).get(t, 0) if isinstance(d, dict) else 0)
    df["tools_used"] = df["tool_calls"].apply(lambda d: sum(1 for v in (d or {}).values() if v) if isinstance(d, dict) else 0)

    g = df.groupby("condition", observed=True)
    summary = pd.DataFrame({
        "n": g.size(),
        "EM %": g["em"].mean().map(pct),
        "F1 %": g["f1"].mean().map(pct),
        "finished %": g["finished"].mean().map(pct),
        "avg steps": g["steps"].mean().round(2),
        "avg tool calls": g["num_tool_calls"].mean().round(2),
        "avg search": g["calls_search"].mean().round(2),
        "avg wikidata": g["calls_wikidata"].mean().round(2),
        "avg calculator": g["calls_calculator"].mean().round(2),
        "multi-tool Q %": g["tools_used"].apply(lambda s: pct((s > 1).mean())),
        "avg LLM calls": g["llm_calls"].mean().round(2),
        "avg tokens": (g["prompt_tokens"].mean() + g["completion_tokens"].mean()).round(0),
        "avg latency s": g["latency_s"].mean().round(1),
        "errors": g["error"].apply(lambda s: int(s.notna().sum())) if "error" in df else 0,
    })

    by_hop_em = df.pivot_table(index="condition", columns="hops", values="em", aggfunc="mean", observed=True).map(pct)
    by_hop_f1 = df.pivot_table(index="condition", columns="hops", values="f1", aggfunc="mean", observed=True).map(pct)
    by_hop_em.columns = [f"{c}-hop EM %" for c in by_hop_em.columns]
    by_hop_f1.columns = [f"{c}-hop F1 %" for c in by_hop_f1.columns]
    hop_table = pd.concat([by_hop_em, by_hop_f1], axis=1)

    # Complementarity: if a perfect router picked the best single tool per question.
    em = df.pivot_table(index="id", columns="condition", values="em", observed=True)
    comp_rows = []
    singles = [c for c in SINGLE if c in em]
    if singles:
        oracle = em[singles].max(axis=1)
        comp_rows.append(("Oracle best single tool per question (upper bound)", pct(oracle.mean())))
        for r in [c for c in ROUTED if c in em]:
            both = em[[r] + singles].dropna()
            best_single = both[singles].max(axis=1)
            comp_rows.append((f"{r}: correct where NO single tool was correct", int(((both[r] == 1) & (best_single == 0)).sum())))
            comp_rows.append((f"{r}: wrong where SOME single tool was correct", int(((both[r] == 0) & (best_single == 1)).sum())))
        if {"search_only", "wikidata_only"} <= set(em):
            comp_rows.append(("Only search_only correct (not wikidata_only)", int(((em.search_only == 1) & (em.wikidata_only == 0)).sum())))
            comp_rows.append(("Only wikidata_only correct (not search_only)", int(((em.search_only == 0) & (em.wikidata_only == 1)).sum())))
    comp = pd.DataFrame(comp_rows, columns=["metric", "value"])

    route_counts = Counter()
    if "routes" in df:
        route_counts = Counter(x for r in df[df.condition == "routed_router"]["routes"].dropna() for x in r)

    out = run_dir
    summary.to_csv(out / "summary.csv")
    hop_table.to_csv(out / "by_hops.csv")
    comp.to_csv(out / "complementarity.csv", index=False)

    # Charts
    fig, ax = plt.subplots(figsize=(9, 4.5))
    summary[["EM %", "F1 %"]].plot.bar(ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title(f"MuSiQue (n={int(summary['n'].max())}) - single tool vs routed\n{run_dir.name}")
    ax.set_ylabel("%")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "em_f1_by_condition.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    by_hop_f1.T.plot(ax=ax, marker="o")
    ax.set_xticks(range(len(by_hop_f1.columns)))
    ax.set_xticklabels([c.split()[0] for c in by_hop_f1.columns])
    ax.set_ylabel("F1 %")
    ax.set_title("F1 by number of hops")
    plt.tight_layout()
    plt.savefig(out / "f1_by_hops.png", dpi=150)
    plt.close()

    tool_cols = ["avg search", "avg wikidata", "avg calculator"]
    tool_df = summary.loc[[c for c in conds if c != "closed_book"], tool_cols]
    if len(tool_df):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        tool_df.plot.bar(stacked=True, ax=ax, color=["#55A868", "#8172B2", "#C44E52"])
        ax.set_ylabel("tool calls per question")
        ax.set_title("Tool usage per condition")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out / "tool_usage.png", dpi=150)
        plt.close()

    md = [
        f"# Results: {run_dir.name}",
        "",
        "## Overall",
        summary.to_markdown(),
        "",
        "## By number of hops",
        hop_table.to_markdown(),
        "",
        "## Complementarity of tools",
        comp.to_markdown(index=False) if len(comp) else "(needs single-tool runs)",
        "",
    ]
    if route_counts:
        md += ["## Explicit router decisions", pd.Series(route_counts).to_frame("count").to_markdown(), ""]
    md += ["![EM/F1](em_f1_by_condition.png)", "", "![F1 by hops](f1_by_hops.png)", "",
           "![Tool usage](tool_usage.png)", ""]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(summary.to_string())
    print()
    print(hop_table.to_string())
    print()
    print(comp.to_string(index=False))
    if route_counts:
        print("\nExplicit router decisions:", dict(route_counts))
    print(f"\nSaved report, CSVs and charts to {out}")


if __name__ == "__main__":
    main()
