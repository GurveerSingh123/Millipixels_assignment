"""PROVIDED: run this once after `pip install -r requirements.txt`.

It checks your Python and packages, downloads the base model (about 90 MB, first run only), runs a
2-step training job, and estimates how long a full fine-tuning run will take on YOUR machine.

    python check_setup.py
"""
import sys
import time

if sys.version_info < (3, 9):
    sys.exit(f"Python 3.9+ needed, you have {sys.version.split()[0]}")

try:
    import accelerate  # noqa: F401  (needed by the trainer)
    import datasets
    import rank_bm25  # noqa: F401
    import torch
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    import sentence_transformers
except ImportError as e:
    sys.exit(f"Missing package: {e}. Run: pip install -r requirements.txt")

from common import read_jsonl

print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | "
      f"sentence-transformers {sentence_transformers.__version__}")

corpus = read_jsonl("data/corpus.jsonl")
train = read_jsonl("data/train.jsonl")
print(f"Data OK: {len(corpus)} articles, {len(train)} training pairs")

t0 = time.time()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print(f"Model loaded in {time.time() - t0:.0f}s")

t0 = time.time()
model.encode([p["text"] for p in corpus], normalize_embeddings=True)
print(f"Encoded all {len(corpus)} articles in {time.time() - t0:.1f}s")

by_id = {p["passage_id"]: p["text"] for p in corpus}
pairs = train[:64]
ds = datasets.Dataset.from_dict({"anchor": [r["query"] for r in pairs],
                                 "positive": [by_id[r["passage_id"]] for r in pairs]})
args = SentenceTransformerTrainingArguments(output_dir="checkpoints", max_steps=2, per_device_train_batch_size=32,
                                            save_strategy="no", report_to="none", use_cpu=True,
                                            disable_tqdm=True)
t0 = time.time()
SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                           loss=losses.MultipleNegativesRankingLoss(model)).train()
per_step = (time.time() - t0) / 2
full = per_step * (len(train) / 32) * 4  # 288 pairs, batch 32, 4 epochs
print(f"Training step OK ({per_step:.1f}s/step). A 4-epoch run on all {len(train)} pairs ≈ {full / 60:.1f} min here.")
print("\nSetup OK - start Part 1.")
