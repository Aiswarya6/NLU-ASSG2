# Natural Language Understanding Assignment

## Overview
This repository contains the implementation and analysis for two core Natural Language Processing tasks:
1. [cite_start]Learning word embeddings from a custom corpus[cite: 377].
2. [cite_start]Character-level name generation using recurrent neural networks[cite: 589].

For a detailed analysis, evaluation metrics, and qualitative findings, please refer to the accompanying report.

## Problem 1: Word Embeddings
[cite_start]This section focuses on training Word2Vec models from scratch using textual data collected from IIT Jodhpur sources[cite: 377]. 
* [cite_start]**Models Implemented:** Continuous Bag of Words (CBOW) and Skip-gram with Negative Sampling[cite: 379].
* [cite_start]**Implementations:** Custom PyTorch architectures defining embedding layers, hidden layers, and forward passes[cite: 380].
* [cite_start]**Evaluations:** Cosine similarity for nearest neighbors, vector arithmetic for analogies, and PCA for 2D visualization[cite: 548, 554, 570].

## Problem 2: Character-Level Name Generation
[cite_start]This section explores sequence modeling to generate Indian names based on a dataset created using Large Language Models[cite: 588, 598].
* [cite_start]**Models Implemented:** Vanilla Recurrent Neural Network (RNN), Bidirectional Long Short-Term Memory (BLSTM), and an RNN with a basic Attention mechanism[cite: 593, 594, 595].
* [cite_start]**Evaluations:** Quantitative metrics (novelty and diversity) and qualitative analysis (realism and failure modes)[cite: 596]. 

## Prerequisites
Ensure you have Python 3.x installed. The following libraries are required:
* `torch` (PyTorch)
* `nltk`
* `matplotlib`
* `wordcloud`
* `scikit-learn` (for PCA)

Install the dependencies using pip:
```bash
pip install torch nltk matplotlib wordcloud scikit-learn
```

## How to Run

### Task 1: Word Embeddings
1. Ensure the scraped dataset files are placed in the `raw_data` directory.
2. Run the preprocessing and model training script:
   ```bash
   python word_embeddings.py
   ```
   *(Note: Replace `word_embeddings.py` with the exact filename of your script or run the respective Jupyter Notebook cells).*
3. The script will automatically clean the corpus, train the CBOW and Skip-gram models, and output the loss per epoch.
4. Visualizations and nearest neighbor computations will be printed to the console or displayed in standard plot windows.

### Task 2: Character-Level Name Generation
1. [cite_start]Ensure the `Training Names.txt` dataset file is located in the root directory[cite: 598].
2. Run the name generation script:
   ```bash
   python name_generation.py
   ```
   *(Note: Replace `name_generation.py` with the exact filename of your script or run the respective Jupyter Notebook cells).*
3. The script will train the Vanilla RNN, BLSTM, and Attention models sequentially.
4. [cite_start]Upon completion, the script will output 20 generated sample names for each architecture[cite: 673, 674, 701, 702].

## File Structure
* `raw_data/` - Directory containing the raw IIT Jodhpur text files.
* [cite_start]`Training Names.txt` - Dataset containing Indian names for the RNN generation task[cite: 598].
* `cleaned_corpus.txt` - The processed and tokenized text used for embedding training.
* `report.pdf` - Detailed findings, architecture explanations, and model analysis.
