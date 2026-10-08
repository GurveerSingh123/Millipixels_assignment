# halcyon-retrieval-finetune

**Start with [ASSIGNMENT.md](ASSIGNMENT.md).**

Replace this file with your own README before you submit. It should cover:

1. Setup (Python version, `pip install -r requirements.txt`)
2. The exact commands that reproduce every number in `REPORT.md` (train → search → score)
3. Any extra flags or options you added
4. Project layout: one line per file


## Setup (Python 3.9+, cpu only)
bash
pip install -r requirements.txt
python check_setup.py

## Reproduce every number in REPORT.md
bash
# validation grid (article-held out split from train.jsonl)
python train.py --validate
#final model -> models/halcyon-minilm

python train.py

# scoring
python score.py --run runs/ft_test_seen.jsonl --queries data/$SET.jsonl

# layout
. search.py: BM25, dense and hybrid retreival; writes top-10 run files
. train.py: seeded fine tuning with MNRL + no duplicates
