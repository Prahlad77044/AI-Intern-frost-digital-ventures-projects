# Week 5: Graph, RAG, and System Design

## Overview
This week focused on building a Knowledge Graph (KG) from the Constitution of Nepal (2015), exploring graph-based insights, and applying Graph Machine Learning (Graph ML) techniques. The project also included designing an application for querying the KG and optimizing ingestion pipelines.

## Project Structure

### Extraction
- **extract_text.py**: Extracts text from the Constitution PDF while preserving layout. Skips unnecessary pages and removes page numbers.
- **parse_pdf.py**: Parses the extracted text into a structured JSON format, chunking at the Part, Article, and Clause levels. It also identifies relationships, institutions, and tags for each article.

### App
- **api.py**: Implements a FastAPI server for classifying constitutional articles into categories (e.g., Rights, Governance, Judiciary) using a pre-trained GCN model.
- **app.py**: A Streamlit-based UI for exploring the Knowledge Graph. Users can search articles by keywords or numbers, view related articles, and explore institutions and tags.

### Notebooks
- **week_5.ipynb**: Covers data ingestion, graph construction, and Graph ML tasks. Key steps include:
  - Fetching nodes and edges from Neo4j.
  - Building a heterogeneous graph with embeddings and labels.
  - Splitting data into train/val/test sets.
  - Training GCN and GraphSAGE models for node classification.
  - Evaluating models using accuracy, F1-score, and confusion matrices.

### Constitution Data
- **constitution_kg.json**: The structured Knowledge Graph in JSON format.
- **constitution.txt**: Raw text extracted from the Constitution PDF.

### Saved Models
- **gcn_article_classifier.pt**: Pre-trained GCN model for article classification.
- **sage_article_classifier.pt**: Pre-trained GraphSAGE model for article classification.
- **metadata.json**: Metadata for category mappings.

## 5-Day Plan Execution

### Day 1: Schema & Ingestion
- Designed the KG schema with nodes (e.g., Part, Article, Clause) and edges (e.g., HAS_PART, RELATES_TO).
- Built ingestion scripts to parse the Constitution into a structured JSON format.
- Optional Neo4j load and basic graph visualizations.

### Day 2: Exploration & Queries
- Computed graph metrics like degrees, centralities, and components.
- Wrote Cypher queries for 2-hop relationships and cross-references.
- Identified most-referenced articles and key institutions.

### Day 3: Graph ML Task
- Chose node classification to categorize articles into Rights, Governance, Judiciary, etc.
- Created train/val/test splits respecting legal/temporal structure.

### Day 4: Modeling
- Trained GCN and GraphSAGE models using PyTorch Geometric.
- Compared AUC, F1-score, and inference costs.

### Day 5: Use-case Write-up
- Designed an application for querying the KG and retrieving references.
- Explored productization opportunities in legal search and civic tech.

## Side Quest: Optimization
- Proposed incremental ingestion pipelines to handle new articles/clauses without recreating the entire KG.
- Suggested using change detection and delta updates for efficient ingestion.

## Deliverables
- **KG Build Scripts**: `extract_text.py`, `parse_pdf.py`.
- **Query Notebook**: `week_5.ipynb`.
- **Graph ML Notebook**: Training and evaluation of GCN/GraphSAGE models.
- **Rebuild Instructions**: Provided in this README.

## Future Work
- Enhance robustness of the KG to handle amendments dynamically.
- Integrate the KG with LLMs for retrieval-augmented generation.
- Optimize graph queries for large-scale datasets.

## How to Run
1. **Setup**: Install dependencies from `requirements.txt`.
2. **Extraction**: Run `extract_text.py` and `parse_pdf.py` to build the KG.
3. **App**: Start the FastAPI server (`api.py`) and Streamlit app (`app.py`).
4. **Notebook**: Execute `week_5.ipynb` for graph exploration and ML tasks.

## Acknowledgments
- Neo4j for graph database support.
- PyTorch Geometric for Graph ML models.
- Streamlit and FastAPI for application development.