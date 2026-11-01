"""Draw a reproducible, hop-stratified sample of 100 MuSiQue (answerable) dev questions."""
import argparse
import json
import random
from collections import defaultdict

from config import RAW_DEV_FILE, SAMPLE_FILE


def hop_count(qid: str) -> int:
    return int(qid[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(RAW_DEV_FILE, encoding="utf-8")]
    rows = [r for r in rows if r.get("answerable", True)]

    by_hop = defaultdict(list)
    for r in rows:
        by_hop[hop_count(r["id"])].append(r)

    rng = random.Random(args.seed)
    quotas = {h: round(args.n * len(v) / len(rows)) for h, v in by_hop.items()}
    quotas[2] += args.n - sum(quotas.values())

    sample = []
    for h in sorted(by_hop):
        sample.extend(rng.sample(by_hop[h], quotas[h]))
    rng.shuffle(sample)

    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} questions to {SAMPLE_FILE}")
    print("Hop distribution:", {f"{h}-hop": quotas[h] for h in sorted(quotas)})


if __name__ == "__main__":
    main()
