import streamlit as st
import pickle
from rank_bm25 import BM25Okapi

INDEX_FILE = "index/bm25_index.pkl"

@st.cache_resource
def load_index():
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["corpus"]

# Load index and corpus
bm25, corpus = load_index()

# Streamlit UI
st.title("🔎 Mini BM25 Search Engine")

query = st.text_input("Enter a search query:")
k = st.slider("Top K results", 1, 10, 5)

if st.button("Search") and query.strip():
    # Tokenize query
    tokenized_query = query.lower().split()

    # Get scores and rank
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    # Display results
    for doc_id, score in ranked:
        item = corpus[doc_id]
        st.subheader(f"{item['title']} (doc_id={doc_id})")
        st.caption(f"Score: {score:.4f}")
        st.write(item["text"])
