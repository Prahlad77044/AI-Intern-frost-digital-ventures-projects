# Week 4: NLP and LLMs

## Overview
This week focuses on Natural Language Processing (NLP) and the use of Large Language Models (LLMs) for sentiment analysis. The project includes classic NLP baselines, prompt engineering, and lightweight adaptation techniques. Additionally, a side quest explores the implementation of a BM25 search engine.

## Project Structure

### Notebooks

#### 1. Classic NLP Baselines
- **Vectorization**: Utilized `TfidfVectorizer` to transform text data into numerical features.
- **Models**: Trained Logistic Regression and Linear SVM models.
- **Evaluation**: Assessed models using metrics like F1-score, confusion matrix, and classification reports.
- **Error Analysis**: Created error buckets to analyze model predictions.
- **Model Saving**: Saved trained models and vectorizers for future use.

#### 2. Prompt Engineering
- **Objective**: Experimented with 0-shot, 1-shot, and 3-shot prompts to evaluate LLM responses.
- **Templates**: Designed templates for sentiment classification of movie reviews.
- **Evaluation**: Compared accuracy, F1-score, precision, and recall across different prompting styles.

#### 3. Lightweight Adaptation
- **Data Preparation**: Preprocessed text data and converted it into Hugging Face datasets.
- **Tokenization**: Implemented various truncation strategies (head, tail, head-tail).
- **Model Loading**: Loaded pre-trained models like DistilBERT and applied quantization for efficiency.
- **Fine-Tuning**: Fine-tuned models using LoRA (Low-Rank Adaptation) with custom training arguments.
- **Evaluation**: Conducted pre- and post-training evaluations, including metrics comparison and confusion matrix visualization.
- **Robustness Testing**: Tested the model on negations, sarcasm, and out-of-domain inputs.
- **Calibration**: Evaluated model calibration using reliability diagrams and Expected Calibration Error (ECE).

### Side Quest: BM25 Search Engine
- **Implementation**: Developed a BM25-based search engine for information retrieval.
- **Features**: Supports ranking documents based on query relevance.
- **Evaluation**: Tested the search engine on a sample dataset.

## Key Results
- **Classic NLP Models**: Achieved competitive performance with Logistic Regression and SVM.
- **Prompt Engineering**: 3-shot prompting yielded the best results in terms of accuracy and F1-score.
- **Fine-Tuned Models**: Significant improvement in metrics after fine-tuning with LoRA.
- **Robustness**: Identified challenges with negations and sarcasm, highlighting areas for improvement.

## Future Work
- Enhance robustness to linguistic phenomena like sarcasm and negations.
- Explore domain adaptation techniques for better generalization.
- Integrate the BM25 search engine with LLMs for hybrid retrieval-augmented generation.

## How to Run
1. **Setup**: Install dependencies from `requirements.txt`.
2. **Notebooks**: Execute the notebooks in the `notebooks/` directory for step-by-step workflows.
3. **BM25 Search Engine**: Run the side quest script to test the search engine.

## Acknowledgments
- Hugging Face Transformers for pre-trained models.
- Scikit-learn for classic NLP pipelines.
- Rank-BM25 for the search engine implementation.
