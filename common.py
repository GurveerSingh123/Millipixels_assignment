# PROVIDED helpers: file I/O and a quick validation metric. Use them to save time, or write your own.

# from common import read_jsonl, write_run, mrr_at_10

# corpus  = read_jsonl("data/corpus.jsonl")
# queries = read_jsonl("data/test_seen.jsonl")
# ranked  = [...]                      # one list of passage_ids per query, best first
# write_run("runs/x.jsonl", queries, ranked)
# print(mrr_at_10(queries, ranked))    # same MRR@10 as score.py

import json
import os


def read_jsonl(path):
    """Read a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_run(path, queries, ranked_ids, k=10):
    """Write a run file for score.py. ranked_ids[i] is the ranked passage_id list for queries[i]."""
    assert len(queries) == len(ranked_ids), "need one ranking per query"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q, ids in zip(queries, ranked_ids):
            f.write(json.dumps({"query_id": q["query_id"], "ranked_passage_ids": list(ids)[:k]}) + "\n")


def mrr_at_10(queries, ranked_ids):
    """MRR@10 for queries that have a gold 'passage_id' (e.g. your validation split)."""
    total = 0.0
    for q, ids in zip(queries, ranked_ids):
        top = list(ids)[:10]
        if q["passage_id"] in top:
            total += 1.0 / (top.index(q["passage_id"]) + 1)
    return total / len(queries)
