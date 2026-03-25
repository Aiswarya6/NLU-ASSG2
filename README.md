Natural Language Understanding Assignment 2
Overview
This repository contains the implementation and analysis for two core Natural Language Processing tasks. For detailed methodologies, hyperparameter tuning, and qualitative evaluations, please refer to the included report.pdf.

Problem 1: Word Embeddings
Training custom Word2Vec models (Continuous Bag of Words and Skip-gram with Negative Sampling) from scratch using textual data collected from IIT Jodhpur sources.

Problem 2: Character-Level Name Generation
Designing and comparing recurrent neural architectures (Vanilla RNN, Bidirectional LSTM, and Attention-based RNN) to generate sequence-based names.

Repository Structure
prob1.ipynb: Jupyter Notebook containing the data preprocessing, model training, and semantic analysis for the Word2Vec architectures.

prob2.ipynb: Jupyter Notebook containing the architecture definitions, training loops, and generation scripts for the RNN models.

cleaned_corpus.txt: The preprocessed and tokenized text corpus used to train the embeddings in Problem 1.

report.pdf: The comprehensive assignment report detailing the findings, evaluation metrics, and comparative analysis of all models.

Prerequisites
To run the notebooks, ensure you have Python 3.x installed along with the following libraries:

torch

nltk

matplotlib

wordcloud

scikit-learn

You can install the required packages using:

Bash
pip install torch nltk matplotlib wordcloud scikit-learn
How to Run
Problem 1: Word Embeddings
Launch Jupyter Notebook or upload prob1.ipynb to Google Colab.

Ensure cleaned_corpus.txt is located in the same working directory as the notebook.

Execute the notebook cells sequentially. The code will load the corpus, train the CBOW and Skip-gram models, and output the epoch losses, cosine similarities, and PCA visualizations.

Problem 2: Character-Level Name Generation
Launch Jupyter Notebook or upload prob2.ipynb to Google Colab.

Ensure your required training dataset (Training Names.txt) is located in the same working directory.

Execute the notebook cells sequentially. The script will train the Vanilla RNN, BLSTM, and Attention models, ultimately printing the generated sample names for evaluation.
