# Report: Halcyon Retrieval Fine-tuning

> Copy this file to `REPORT.md` and fill it in (budget: about 25 minutes). Bullet points and tables are fine,
> and preferred over paragraphs. Every number must be reproducible with the commands in your README.

**Name:**
**Time spent:** (be honest, e.g. "2h, setup took 12 min")
**Machine:** (e.g. MacBook Air M1, 8 GB, CPU only / Windows laptop i5, 16 GB / Google Colab)

## 1. Results

All numbers from `score.py`. `test_seen` = 72 queries, `test_unseen` = 40 queries.

| System | test_seen R@1 | test_seen MRR@10 | test_unseen R@1 | test_unseen MRR@10 |
|---|---|---|---|---|
| BM25 | | | | |
| all-MiniLM-L6-v2 (no fine-tuning) | | | | |
| Fine-tuned MiniLM | | | | |
| *(optional bonus)* | | | | |

Training time on your machine: ___ s · Validation MRR@10 before / after fine-tuning: ___ / ___

## 2. Key decisions (3–5 bullets)

- **Validation split:** how you split `train.jsonl`, and what that split does and doesn't measure.
- **Training setup:** loss, batch size, epochs, learning rate, and how you chose them.
- Anything else that mattered, such as article text representation or BM25 tokenisation.

## 3. Error analysis

At least 5 failed queries from your fine-tuned model (gold article not at rank 1):

| query_id | query | gold article (title) | ranked #1 instead | why it failed |
|---|---|---|---|---|

Then, in 2–3 bullets: the main failure categories, and what would fix each one.

## 4. Seen vs unseen (2–4 bullets)

Compare `test_seen` and `test_unseen`, before and after fine-tuning. Is the difference what you expected?
Is it meaningful with only 40–72 queries? What did the model actually learn: the domain's vocabulary,
or a lookup table of the training articles? What is your evidence?

## 5. If this went to production (3–5 bullets)

What would you change before real field technicians relied on this? For example: data, evaluation,
monitoring, new KB articles, latency.

## 6. Unfinished work and AI tools

- Anything you didn't finish, and what you would do next.
- Which AI tools you used, if any, and for what. You will be asked to explain any line of your code.
