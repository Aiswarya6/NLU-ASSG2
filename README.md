# NLU-ASSG2

This repository contains the implementations, experiments, and analyses for Natural Language Understanding Assignment 2. The project is divided into two main problems:
1. Learning custom Word2Vec embeddings from a domain-specific corpus.
2. Character-level name generation using Recurrent Neural Networks (RNNs).

---

# Problem 1: Learning Word Embeddings from IIT Jodhpur Data

## Project Overview
The objective of this task was to train custom Word2Vec architectures on a preprocessed text corpus scraped from the official IIT Jodhpur website. Two distinct architectures were implemented from scratch using PyTorch: Continuous Bag of Words (CBOW) and Skip-gram with Negative Sampling. 

## Model Architectures & Experiments
Custom PyTorch implementations were used to define the embedding layers, hidden layers, and forward passes. 

Experiments were conducted to observe the effects of varying hyperparameters on the training loss:
- **Embedding Dimension**: Increasing the dimension from 50 to 100 did not significantly improve performance, likely due to the relatively small dataset size.
- **Context Window**: A window size of 2 resulted in slightly lower loss compared to a window size of 1, indicating that a larger window captures richer contextual information. However, expanding the window too broadly introduced semantic noise.
- **Negative Samples**: Increasing the number of negative samples provided stronger contrastive signals, improving the model's ability to differentiate between true contextual neighbors and random noise.

## Results & Semantic Analysis

### 1. Nearest Neighbors (Cosine Similarity)
The Skip-gram model successfully captured domain-specific academic contexts. The top nearest neighbors for selected terms were:
- research: small (0.43), art (0.40), social (0.40), vacant (0.37), lab (0.37)
- student: drc (0.45), admission (0.42), carrying (0.40), should (0.39), qualifier (0.38)
- phd: wild (0.39), minors (0.39), programmes (0.38), dues (0.38), comprehensive (0.37)

### 2. Analogy Experiments
Using vector arithmetic, semantic relationships were evaluated:
- UG : BTech :: PG : regularexternalparttime (0.39)
- BTech : Four :: MTech : deadline (0.39)

The overall cosine similarity scores remained relatively low (~0.37-0.45), and analogy outputs occasionally yielded frequent but loosely related institutional words rather than perfect logical matches. This reflects the limitations of training a Word2Vec model on a small, domain-specific corpus.

### 3. PCA Visualization
A 2D Principal Component Analysis (PCA) projection was utilized to visualize the clustering behavior of the learned embeddings.

![PCA Visualization](image_a7c69c.png)

---

# Problem 2: Character-Level Name Generation using RNNs

## Project Overview
The objective of this assignment was to design, implement, and compare different recurrent neural architectures for character-level name generation. 

The architectures evaluated include:
- Vanilla Recurrent Neural Network (RNN)
- Bidirectional Long Short-Term Memory (BLSTM)
- RNN with a basic Attention mechanism

## Dataset
A dataset of Indian names was generated using Large Language Models (LLMs).
- Total unique names: 895
- Vocabulary size: 27

Preprocessing steps included converting all names to lowercase, adding special start (^) and end ($) tokens, and including a space character to model full names (first + last).

## Model Evaluations

### 1. Vanilla RNN
- **Architecture**: Embedding Layer -> RNN Layer -> Fully Connected Layer.
- **Quantitative Results**: Achieved a high novelty rate of 0.74 and a diversity rate of 0.97. 
- **Qualitative Results**: Successfully learned the first-name + last-name pattern, producing realistic and structured names while maintaining phonetic consistency. 

### 2. Bidirectional LSTM (BLSTM)
- **Architecture**: Embedding Layer -> BLSTM -> Dropout -> Fully Connected Layer.
- **Results**: Achieved an extremely low training loss (dropping from 414 to 0.87). However, it failed completely during text generation, producing incoherent sequences with broken word boundaries. 
- **Key Insight**: The model failed due to its non-causal nature; it leveraged future context during training, leading to a mismatch at inference time and poor generalization. 

### 3. RNN with Attention
- **Architecture**: Introduced an Attention Layer to compute weighted importance over hidden states.
- **Results**: Exhibited unstable generation, mode collapse, and highly repetitive sequences. 
- **Key Insight**: While attention mechanisms are powerful, they require careful tuning and sufficient data to be effective in sequence generation tasks.

## Conclusion
The Vanilla RNN performed the best overall for character-level name generation. It achieved the best balance between stability, diversity, and realism. The experiments demonstrated that low training loss does not guarantee good generative quality, and that proper causal modeling is critical for
