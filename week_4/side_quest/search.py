import pickle
from rank_bm25 import BM25Okapi

INDEX_PATH = "index/bm25_index.pkl"

# Load BM25 index and corpus
def load_index():
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["corpus"]

bm25, corpus = load_index()

# Search function
def search(query, k=5):
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:k]
    
    # Return as list of dicts
    results = [{"doc_id": doc_id, "score": score} for doc_id, score in ranked]
    return results
