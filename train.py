import argparse
import random
import time
import numpy as np
import torch

from datasets import Dataset
from sentence_transformers import (SentenceTransformer,SentenceTransformerTrainer,SentenceTransformerTrainingArguments,losses)
from sentence_transformers.training_args import BatchSamplers

from common import mrr_at_10, read_jsonl
from search import article_text,rank

BASE="sentence-transformers/all-MiniLM-L6-v2"
OUT="models/halcyon-minilm"
SEED=42
FINAL=dict(epochs=4,batch_size=32,lr=2e-5)
GRID=[dict(epochs=2,batch_size=32,lr=2e-5),dict(epochs=4,batch_size=32,lr=2e-5),dict(epochs=8,batch_size=32,lr=2e-5)]
N_VAL_ARTICLES=14

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def split_by_article(train,n_val,seed):
    pids=sorted({r["passage_id"] for r in train})
    val_pids=set(random.Random(seed).sample(pids,n_val))
    return ([r for r in train if r["passage_id"] not in val_pids], [r for r in train if r["passage_id"]in val_pids])

def evaluate(model,corpus,queries):
    return mrr_at_10(queries,rank(corpus,[q["query"] for q in queries], None, model=model))

def train_model(pairs, corpus,epochs,batch_size,lr):
    set_seed(SEED)
    by_id={p["passage_id"]:article_text(p) for p in corpus}
    model=SentenceTransformer(BASE,device='cpu')
    ds = Dataset.from_dict({"anchor": [r["query"] for r in pairs], "positive": [by_id[r["passage_id"]] for r in pairs]})
    args = SentenceTransformerTrainingArguments(
        output_dir="checkpoints", num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, warmup_ratio=0.1, seed=SEED,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no", report_to="none", use_cpu=True,
    )
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=losses.MultipleNegativesRankingLoss(model)).train()

    return model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--validate",action='store_true',help="run the validation grid instead of final fit")
    args=ap.parse_args()
    corpus=read_jsonl("data/corpus.jsonl")
    train=read_jsonl("data/train.jsonl")
    tr,val=split_by_article(train,N_VAL_ARTICLES,SEED)

    if args.validate:
        base=SentenceTransformer(BASE,device='cpu')
        print(f"split:{len(tr)}train/{len(val)} val queries ({N_VAL_ARTICLES} held-out articles)")
        print(f"Base model val MRR@10={evaluate(base,corpus,val):.3f}")

        for cfg in GRID:
            t0=time.time()
            model=train_model(tr,corpus,**cfg)
            print(f"{cfg} val MRR@10 = {evaluate(model,corpus,val):.3f} | " f"train MRR@10 = {evaluate(model,corpus,tr):.3f} ({time.time()-t0:.0f}s)")
        return
    t0=time.time()
    model=train_model(train,corpus,**FINAL)
    print(f"trained on all {len(train)} pairs with {FINAL} in {time.time()-t0:.0f}s")
    model.save(OUT)
    print("Saved to",OUT)

if __name__=="__main__":
    main()
