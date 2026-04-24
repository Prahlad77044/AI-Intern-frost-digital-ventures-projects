import json
import math
from search import search  # your BM25 search function

QRELS = "qrels.json"

def mrr_at_k(results, relevant_docs):
    for rank, r in enumerate(results, 1):
        if r["doc_id"] in relevant_docs:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(results, relevant_docs):
    dcg = 0.0
    for rank, r in enumerate(results, 1):
        if r["doc_id"] in relevant_docs:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_dcg = sum([1.0 / math.log2(i + 2) for i in range(len(relevant_docs))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def top5_hit(results, relevant_docs):
    """Return 1 if any relevant doc is in top 5, else 0"""
    return int(any(r["doc_id"] in relevant_docs for r in results))

def evaluate(k=5):
    with open(QRELS, "r") as f:
        queries = json.load(f)["queries"]

    mrrs, ndcgs, hits = [], [], []

    for q in queries:
        results = search(q["query"], k)
        mrrs.append(mrr_at_k(results, q["relevant"]))
        ndcgs.append(ndcg_at_k(results, q["relevant"]))
        hits.append(top5_hit(results, q["relevant"]))

    print("MRR@5:", sum(mrrs) / len(mrrs))
    print("nDCG@5:", sum(ndcgs) / len(ndcgs))
    print("Top-5 Hit Rate:", sum(hits), "/", len(hits),
          f"({sum(hits)/len(hits)*100:.2f}%)")

if __name__ == "__main__":
    evaluate()
