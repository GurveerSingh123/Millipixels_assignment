import argparse
import re 
import numpy as np
from rank_bm25 import BM25Okapi

from common import read_jsonl, write_run


TOKEN_RE=re.compile(r"[a-z0-9]+")

def article_text(p):
    return f"{p['title']}.{p['product']}.{p['text']}"

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

def bm25_scores(corpus,query_texts):
    bm25=BM25Okapi([tokenize(article_text(p)) for p in corpus])
    return np.stack([bm25.get_scores(tokenize(q)) for q in query_texts])

def dense_scores(corpus,query_texts,model_name,model=None):
    if model is None:
        from sentence_transformers import SentenceTransformer
    P=model.encode([article_text(p) for p in corpus],normalize_embeddings=True)
    Q=model.encode(query_texts, normalize_embeddings=True)
    return Q @ P.T

def rrf(*score_matrices,k=60):
    fused=np.zeros_like(score_matrices[0],dtype=float)
    for S in score_matrices:
        ranks=np.argsort(np.argsort(-S,axis=1),axis=1)+1
        fused +=1.0/(k+ranks)
    return fused

def rank(corpus, query_texts,model_name,hybrid=False,model=None):
    if model_name=="bm25":
        S=bm25_scores(corpus,query_texts)
    else:
        S=dense_scores(corpus,query_texts,model_name,model)
        if hybrid:
            S=rrf(S,bm25_scores(corpus,query_texts))
    ids=[p["passage_id"] for p in corpus]
    return [[ids[j] for j in np.argsort(-row,kind="stable")[:10]]for row in S]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True,help="'bm25' or a sentence-transformers name/path" )
    ap.add_argument("--queries",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--corpus",default="data/corpus.jsonl")
    ap.add_argument("--hybrid",action="store_true",help="Fuse the dense model with  bm25 via rrf")
    args=ap.parse_args()

    corpus=read_jsonl(args.corpus)
    queries=read_jsonl(args.queries)
    ranked=rank(corpus,[q["query"]for q in queries],args.models,args.hybrid)
    write_run(args.out,queries,ranked)
    print(f"wrote {len(queries)} rankings to {args.out}")

if __name__=="__main__":
    main()
