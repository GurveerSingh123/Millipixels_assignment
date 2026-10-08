# halcyon-retrieval-finetune

Fine-tunes `sentence-transformers/all-MiniLM-L6-v2` to retrieve Halcyon KB articles for Hinglish/slang
field-support queries. The task brief is in [ASSIGNMENT.md](ASSIGNMENT.md); the results are in `REPORT.md`.

## Setup (Python 3.9+, CPU only)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python check_setup.py
```

## Reproduce every number

```bash
# 1. (optional) validation grid on an article-held-out split of train.jsonl
python train.py --validate

# 2. train the final model on all of train.jsonl -> models/halcyon-minilm
python train.py

# 3. rank both test sets with every system (writes runs/*.jsonl)
python search.py --model bm25 --queries data/test_seen.jsonl --out runs/bm25_test_seen.jsonl
python search.py --model bm25 --queries data/test_unseen.jsonl --out runs/bm25_test_unseen.jsonl
python search.py --model sentence-transformers/all-MiniLM-L6-v2 --queries data/test_seen.jsonl --out runs/minilm_test_seen.jsonl
python search.py --model sentence-transformers/all-MiniLM-L6-v2 --queries data/test_unseen.jsonl --out runs/minilm_test_unseen.jsonl
python search.py --model models/halcyon-minilm --queries data/test_seen.jsonl --out runs/ft_test_seen.jsonl
python search.py --model models/halcyon-minilm --queries data/test_unseen.jsonl --out runs/ft_test_unseen.jsonl

# 4. score a run (swap in any run file / query file pair)
python score.py --run runs/ft_test_seen.jsonl --queries data/test_seen.jsonl
python score.py --run runs/ft_test_unseen.jsonl --queries data/test_unseen.jsonl
```

Extra flag: `search.py --hybrid` fuses the dense model with BM25 via Reciprocal Rank Fusion (k=60).

## Layout

- `search.py`: BM25, dense and hybrid retrieval; writes top-10 run files
- `train.py`: seeded fine-tuning with MultipleNegativesRankingLoss and a no-duplicates batch sampler
- `common.py`: provided JSONL, run-file and MRR@10 helpers
- `score.py`: provided scorer (unchanged)
- `check_setup.py`: provided setup check
- `data/`: corpus, training pairs and the two test sets
- `runs/`: run files behind the reported numbers
