"""Score a retrieval run against gold labels.

PROVIDED - do not modify. Reviewers score every submission with this exact file.

Run file format (JSONL, one line per query):
    {"query_id": "TS-001", "ranked_passage_ids": ["KB-031", "KB-007", ...]}

Query file format (JSONL, one line per query):
    {"query_id": "TS-001", "query": "...", "passage_id": "KB-031"}
    An optional "group" field (for example "seen" / "unseen") adds a per-group breakdown.

Each query has exactly one relevant passage, so:
    Recall@k = 1 if the gold passage is in the top k, else 0
    MRR@10   = 1 / rank of the gold passage (0 if it is not in the top 10)
    nDCG@10  = 1 / log2(rank + 1)           (0 if it is not in the top 10)

Usage:
    python score.py --run runs/bm25_test_seen.jsonl --queries data/test_seen.jsonl
    python score.py --run runs/a.jsonl --queries data/test_seen.jsonl --json   # machine-readable
"""
import argparse
import json
import math
import sys
from collections import defaultdict

CUTOFF = 10
RECALL_KS = (1, 5, 10)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_one(ranked, gold):
    ranked = ranked[:CUTOFF]
    rank = ranked.index(gold) + 1 if gold in ranked else None
    out = {f"Recall@{k}": float(rank is not None and rank <= k) for k in RECALL_KS}
    out["MRR@10"] = 1.0 / rank if rank else 0.0
    out["nDCG@10"] = 1.0 / math.log2(rank + 1) if rank else 0.0
    return out


def mean(rows):
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run file (JSONL) produced by your search.py")
    ap.add_argument("--queries", required=True, help="query file with gold passage_id")
    ap.add_argument("--json", action="store_true", help="print metrics as JSON")
    args = ap.parse_args()

    queries = read_jsonl(args.queries)
    run = {r["query_id"]: r["ranked_passage_ids"] for r in read_jsonl(args.run)}

    missing = [q["query_id"] for q in queries if q["query_id"] not in run]
    if missing:
        print(f"WARNING: {len(missing)} queries missing from run (scored as 0), e.g. {missing[:3]}",
              file=sys.stderr)

    per_query, by_group = [], defaultdict(list)
    for q in queries:
        s = score_one(run.get(q["query_id"], []), q["passage_id"])
        per_query.append(s)
        if "group" in q:
            by_group[q["group"]].append(s)

    results = {"all": {"n": len(per_query), **mean(per_query)}}
    for g, rows in sorted(by_group.items()):
        results[g] = {"n": len(rows), **mean(rows)}

    if args.json:
        print(json.dumps(results, indent=2))
        return

    metrics = list(results["all"].keys())[1:]
    print(f"{'subset':<10}{'n':>5}" + "".join(f"{m:>11}" for m in metrics))
    for name, r in results.items():
        print(f"{name:<10}{r['n']:>5}" + "".join(f"{r[m]:>11.3f}" for m in metrics))


if __name__ == "__main__":
    main()
