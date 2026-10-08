# AI/ML Engineer Intern Assignment: Fine-tuning a Retriever for Field-Support Search

**Timebox: 2 hours in total, including setup** · **Hardware: any laptop, CPU only** · **Python 3.9+**

---

## 0. Time plan: read this first

The 2 hours start when you open this file. The plan below has a 10-minute buffer. If a part runs over,
move on: a finished Part 3 (report) is worth more than a perfect Part 2.

| Clock | Part | Minutes |
|---|---|---|
| 0:00 – 0:15 | **Setup:** start the install *first*, then read this brief while it downloads | 15 |
| 0:15 – 0:40 | **Part 1:** BM25 and off-the-shelf baselines | 25 |
| 0:40 – 1:15 | **Part 2:** fine-tune the embedding model | 35 |
| 1:15 – 1:40 | **Part 3:** error analysis and report | 25 |
| 1:40 – 1:50 | **Part 4:** README and submit | 10 |
| 1:50 – 2:00 | Buffer, or the optional bonus | 10 |

Training is fast (under a minute on a modern laptop, a few minutes on an older one), so time goes on
writing code and thinking, not on waiting. We provide the boring parts (data loading, run-file writing, the
scorer and a setup checker) so your time goes on the ML decisions.

---

## 1. The problem

**Halcyon Agrobotics** (a fictional company) sells agricultural spray drones (*FieldHawk F2/F3*), an
autonomous ground robot (*Tiller-X*) and a farm-management app (*AgroSense*). Its support team keeps
a knowledge base (KB) of 92 short technical articles.

The people searching that KB are farmers, FPO staff and field technicians. They don't type
*"Electronic speed controller fault on rotor 3"*. They type:

> `bhai ek pankha dheere ghoom raha baaki fast, bird ek side jhuk raha`

The KB articles are written in formal technical English. The queries are in Hinglish, slang and
symptom descriptions. Keyword search breaks on this mismatch, and so does an off-the-shelf English
embedding model. In a RAG system, if retrieval misses the right article, the LLM never sees it,
however good the LLM is.

**Your job:** fine-tune a small sentence-embedding model so that it finds the right KB article for these
queries, prove that it helps with honest evaluation, and explain where it still fails.

This is the retrieval half of a RAG system. You don't need an LLM for this assignment.

---

## 2. What you get

```
halcyon-retrieval-finetune/
├── ASSIGNMENT.md          ← this file
├── README.md              ← replace with your own README
├── REPORT_TEMPLATE.md     ← copy to REPORT.md and fill in
├── requirements.txt
├── check_setup.py         ← PROVIDED: verifies your setup in about 15 s
├── common.py              ← PROVIDED helpers: read_jsonl, write_run, mrr_at_10
├── score.py               ← PROVIDED scorer. Do not modify it.
└── data/
    ├── corpus.jsonl       92 KB articles
    ├── train.jsonl        288 (query → article) training pairs
    ├── test_seen.jsonl    72 new queries about articles that ALSO have training queries
    └── test_unseen.jsonl  40 queries about 20 articles that have NO training queries
```

**Formats** (one JSON object per line):

```json
// corpus.jsonl
{"passage_id": "KB-024", "title": "Cleaning a Clogged Spray Nozzle", "product": "FieldHawk F2/F3", "text": "A partially clogged nozzle produces..."}
// train.jsonl / test_seen.jsonl / test_unseen.jsonl
{"query_id": "TR-0034", "query": "bhai fawwara kitni der garam paani me bhigo ke rakhu, aur peeche wali jaali bhi nikalni hai kya", "passage_id": "KB-024"}
```

Each query has **exactly one** correct article (`passage_id`).

Why two test sets? `test_seen` checks whether the model handles *new phrasings* of problems it was
trained on. `test_unseen` checks whether it learned the *domain's language* well enough to handle articles
it never saw a training query for. In real life, new KB articles get added every week.

**We also hold a hidden test set** of 92 queries in the same style, both seen and unseen. We will run
your code on it.

---

## 3. Setup (0:00 – 0:15)

**Do step 1 straight away.** The install downloads about 200–300 MB (mostly PyTorch) and takes 2–8 minutes,
depending on your internet. Read the rest of this brief while it runs.

```bash
# 1. Start the install (macOS / Linux)
cd halcyon-retrieval-finetune
python3 -m venv .venv
source .venv/bin/activate
# Linux only, run this first, or PyTorch pulls ~2.5 GB of GPU libraries:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

#    Windows (PowerShell):
#    py -m venv .venv ; .venv\Scripts\Activate.ps1 ; pip install -r requirements.txt

# 2. When it finishes, check everything (downloads the ~90 MB base model on first run):
python check_setup.py
```

`check_setup.py` should end with `Setup OK`. It also prints how long a full training run will take **on
your machine** (we measured about 40 s on an Apple M-series laptop; expect 2–3 min on an older Intel/AMD laptop).

**Requirements:** 8 GB RAM, about 1.5 GB free disk, no GPU.

**Stuck on setup at 0:20?** Don't lose more time. Upload the folder to [Google Colab](https://colab.research.google.com)
(a free CPU runtime is enough). Run `!pip install -r requirements.txt` there, do the work, and say so in your report.

---

## 4. Tasks

### Part 1: Baselines (0:15 – 0:40)

Write `search.py`. It ranks the 92 articles for every query in a query file and writes the **top 10**
to a *run file* (use `write_run` from `common.py`):

```bash
python search.py --model bm25 --queries data/test_seen.jsonl --out runs/bm25_test_seen.jsonl
python search.py --model sentence-transformers/all-MiniLM-L6-v2 --queries data/test_seen.jsonl --out runs/minilm_test_seen.jsonl
python score.py  --run runs/minilm_test_seen.jsonl --queries data/test_seen.jsonl
```

Run-file format, one line per query:

```json
{"query_id": "TS-001", "ranked_passage_ids": ["KB-024", "KB-007", "...10 ids in total"]}
```

- **BM25** baseline, using `rank_bm25`. Pick a tokenisation and say what it is.
- **Dense** baseline: `all-MiniLM-L6-v2`, with no training. Use cosine similarity between query and article embeddings.
- Decide what text represents an article (title? body? both?) and stay consistent.
- Score both baselines on **both** test files. Commit: `git commit -m "Part 1: baselines"`.

### Part 2: Fine-tune the embedding model (0:40 – 1:15)

Write `train.py`. Run it with no arguments and it must **re-create your final model from scratch**, seeded,
saving it to `models/halcyon-minilm/`. Then `search.py --model models/halcyon-minilm` must work.

1. **Base model:** `sentence-transformers/all-MiniLM-L6-v2`. We fix the base model so results are comparable across candidates.
2. **Validation:** carve a validation set out of `train.jsonl` and use it, not the test files, to
   choose epochs, learning rate and so on. **How** you split matters, so justify your choice in the report.
   Keep the search small: 2–4 training runs is plenty.
3. **Loss:** `MultipleNegativesRankingLoss` (in-batch negatives) is the standard starting point. You may
   use something else if you can say why.
4. Train the final model, then score it on both test files. Commit.

### Part 3: Error analysis and report (1:15 – 1:40)

Copy `REPORT_TEMPLATE.md` to `REPORT.md` and fill it in: the results table, your key decisions, **at least 5
failure cases grouped into categories**, the seen-vs-unseen comparison, and 3–5 production bullets.
Bullet points are fine. We value clear reasoning over polished prose. Commit.

### Part 4: README and submit (1:40 – 1:50)

Replace `README.md` with yours (setup, the exact commands behind every number in your report), then submit
as described in [Section 8](#8-how-to-submit).

### Optional bonus: only if you have time left within the 2 hours

Try **one** improvement and measure it against your Part 2 model on validation **and** both test sets:
**(a)** hybrid BM25 + fine-tuned model (for example Reciprocal Rank Fusion), **(b)** hard negatives mined from BM25
or the base model (watch out for *false* negatives), **(c)** extra training queries for articles that have none,
made from **corpus text only**, or **(d)** a multilingual base model such as
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Keep the Part 2 model as your official submission.
A well-measured negative result counts as much as a positive one.

**Don't start the bonus at the cost of Part 3 or Part 4.** We'd rather have a complete, honest submission.

---

## 5. Rules

- **Never train on, or tune against, `test_seen.jsonl` / `test_unseen.jsonl`.** Use them to report
  results. The hidden test set exposes over-tuning, and a solid validation method earns more marks
  than a slightly higher test number.
- `python train.py` must finish in **≤ 10 minutes on CPU**.
- **Don't commit model weights.** `.gitignore` already excludes `models/` and `checkpoints/`. We will run
  `train.py` ourselves.
- **Don't modify `score.py`.**
- **Stop at 2:00.** Submit what you have, and list in the report anything unfinished and what you would do next.
- You may use documentation, Stack Overflow and AI coding assistants. Say which ones you used in the
  report. In the follow-up call you will be asked to explain any line of your code and any number in your
  report.

---

## 6. Implementation guide

This is a suggested route, not a requirement. The finished `search.py` + `train.py` together come to roughly
100–150 lines.

**Step 1: load the data** with `common.read_jsonl`, and build `{passage_id: article}`.

**Step 2: BM25.**

```python
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi([tokenize(article_text) for article_text in corpus_texts])
scores = bm25.get_scores(tokenize(query))           # one score per article
top10 = numpy.argsort(-scores)[:10]
```

**Step 3: dense retrieval.**

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_name_or_path, device="cpu")
P = model.encode(corpus_texts, normalize_embeddings=True)   # (92, 384)
Q = model.encode(query_texts,  normalize_embeddings=True)   # (n, 384)
S = Q @ P.T                                                  # cosine similarity, (n, 92)
```

**Step 4: fine-tuning** (sentence-transformers ≥ 3 API):

```python
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
train_ds = Dataset.from_dict({"anchor": train_queries, "positive": train_passage_texts})
loss = losses.MultipleNegativesRankingLoss(model)
args = SentenceTransformerTrainingArguments(
    output_dir="checkpoints", num_train_epochs=..., per_device_train_batch_size=...,
    learning_rate=..., warmup_ratio=0.1, seed=42,
    save_strategy="no", report_to="none", use_cpu=True,
)
SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss).train()
model.save("models/halcyon-minilm")
```

Before you train, think about these. They are the core of the assignment:

- `MultipleNegativesRankingLoss` treats every *other* positive in the batch as a negative. Each article
  here has **4** training queries. What happens when two of them land in the same batch? Look at the
  `batch_sampler` option in the training arguments.
- How does batch size change the number of negatives each query sees?
- With 288 pairs, how quickly does the model overfit? How would you notice?
- Your validation split: if you hold out *queries*, what does that measure? If you hold out *articles*, what does that measure?
  Which one matches `test_seen`, and which matches `test_unseen`?

**Step 5: validation.** Inside `train.py`, compute `common.mrr_at_10` on your validation split before and after
training. That tells you in seconds whether a change helped.

Useful docs: [sentence-transformers training overview](https://sbert.net/docs/sentence_transformer/training_overview.html) ·
[MultipleNegativesRankingLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) ·
[rank_bm25](https://github.com/dorianbrown/rank_bm25)

---

## 7. How we grade

| Area | Weight | What we look for |
|---|---|---|
| **Works and reproduces** | 25 | We run the commands in your README on a fresh clone. `train.py` and `search.py` work. Your report numbers reproduce (within small noise). |
| **Evaluation rigour** | 25 | Both baselines, a sensible validation split with no test leakage, seen and unseen reported separately, fixed seeds. |
| **Error analysis and insight** | 20 | Real failure categories with examples. Correct interpretation of the seen/unseen comparison. |
| **Hidden-set result** | 15 | Improvement of your fine-tuned model over the off-the-shelf MiniLM on our hidden queries. |
| **Code quality and README** | 15 | Clear, small, readable. A README that lets a stranger run everything in 3 commands. |
| *Bonus* | +10 | A well-measured optional improvement, or something we didn't think of. |

You can get full marks without the bonus.

---

## 8. How to submit

Your final repo should look roughly like this:

```
ASSIGNMENT.md   README.md (yours)   REPORT.md   requirements.txt   check_setup.py   common.py
train.py   search.py   score.py (unchanged)   data/   runs/*.jsonl
```

Your `README.md` must say: how to set up, the exact commands that reproduce every number in `REPORT.md`,
and any extra flags (for example `--hybrid`). **Commit the run files** in `runs/`. They are small, and they
show which outputs your numbers came from.

### Option A (preferred): public GitHub repository

Commit as you go, at least once per part. We read the commit history as a rough timeline, so please don't squash it.

```bash
cd halcyon-retrieval-finetune
git init
git add .
git commit -m "Part 1: BM25 and MiniLM baselines"
# ... keep committing after each part ...

# Create an EMPTY public repo on github.com (no README/licence), named halcyon-retrieval-finetune, then:
git remote add origin https://github.com/<your-username>/halcyon-retrieval-finetune.git
git branch -M main
git push -u origin main
```

If you have the GitHub CLI, `gh repo create halcyon-retrieval-finetune --public --source=. --push` does the last three steps.

**Before you send the link:** open it in a private/incognito browser window to confirm it is public. Check that
`models/` and `.venv/` are **not** in the repo (it should be under ~5 MB).

### Option B: share the source code directly

If you can't use GitHub, zip the folder **without** the virtualenv and model weights:

```bash
# macOS / Linux
zip -r halcyon-<yourname>.zip . -x ".venv/*" "models/*" "checkpoints/*" "__pycache__/*"
# Windows (PowerShell): copy the folder, delete .venv, models and checkpoints from the copy, then
Compress-Archive -Path .\halcyon-retrieval-finetune-copy\* -DestinationPath halcyon-<yourname>.zip
```

Send the zip by email, or upload it to Google Drive / OneDrive and share a link with "anyone with the link can view".
If you used git locally, include the `.git` folder so we can see your commits.

**Send the link or zip to:** `<reviewer email>` · **Deadline:** `<date and time>`

Good luck, and have fun with it. We care more about how you reason than about the final number.
