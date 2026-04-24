import json
import pickle
import os
from rank_bm25 import BM25Okapi

DATA_PATH = "data/corpus.json"
INDEX_FOLDER = "index"
INDEX_FILE = os.path.join(INDEX_FOLDER, "bm25_index.pkl")

def build_index():
    # Create index folder if not exists
    if not os.path.exists(INDEX_FOLDER):
        os.makedirs(INDEX_FOLDER)

    # Load corpus
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Tokenize documents (split by whitespace)
    tokenized_corpus = [doc["text"].lower().split() for doc in corpus]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Save BM25 index and corpus together
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus}, f)

    print(f"Index built and saved to {INDEX_FILE}")
    print(f"Total documents indexed: {len(corpus)}")

if __name__ == "__main__":
    build_index()
