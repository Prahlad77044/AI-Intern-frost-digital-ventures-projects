# Mini BM25 Search Engine

This project implements a simple measurable IR system using BM25.

## Features
- Full BM25 indexing + retrieval
- CLI search interface
- Streamlit UI
- qrels and evaluation
- Metrics: MRR@5, nDCG@5

## How to run

### Build index
python build_index.py

### Search (CLI)
python search.py "cnn for images"

### Evaluate ranking quality
python eval.py

### Streamlit App
streamlit run streamlit_app.py
