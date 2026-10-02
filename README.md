# Comparative Analysis of Deep Learning Models for Text Sentiment Classification

ICT-4442 Deep Learning mini project, School of Computer Engineering, MIT Manipal.

This repository contains six executed Jupyter notebooks for a controlled comparison
of a classical baseline and three neural sentiment-classification models.

## Notebook workflow

| # | Notebook | Model / purpose | Family | Owner |
|---|---|---|---|---|
| 01 | `01_data_preprocessing.ipynb` | Download, clean, split, build vocabulary and GloVe matrix | Shared pipeline | All |
| 02 | `02_tfidf_logreg.ipynb` | TF-IDF + Logistic Regression | Classical baseline | Sonaksh Jain |
| 03 | `03_text_cnn.ipynb` | Text CNN with 300-dimensional GloVe embeddings | Convolutional | Rishit Kumar |
| 04 | `04_bilstm.ipynb` | BiLSTM with 300-dimensional GloVe embeddings | Recurrent | Akash Shaw |
| 05 | `05_distilbert.ipynb` | DistilBERT fine-tuning | Transformer | Anvith Choudhary Jasti |
| 06 | `06_comparison_and_error_analysis.ipynb` | Comparison, significance tests, ablations and error analysis | Cross-model | All |

The notebooks include their outputs so that the reported results can be inspected
without rerunning the full experiments.

## Experimental protocol

- **Dataset:** IMDb Large Movie Review Dataset v1.0, with 25,000 training and
  25,000 test reviews.
- **Split:** 2,500 stratified training reviews (seed 42) are held out for
  validation, leaving 22,500 training, 2,500 validation and 25,000 test reviews.
- **Cleaning:** HTML unescaping, accent folding, tag and URL removal, lowercasing,
  while retaining apostrophes and selected punctuation.
- **Tokenisation:** a shared regex tokenizer for non-Transformer models; DistilBERT
  uses its native WordPiece tokenizer.
- **Input budget:** 256 tokens for sequence models, with truncation selected on the
  validation set.
- **Selection:** hyperparameters, ablations and early stopping are selected using
  validation F1 only.
- **Metrics:** accuracy, precision, recall, F1, macro-F1, ROC-AUC, confusion
  matrices, training time and inference time. Neural models report mean ± standard
  deviation over seeds 42, 43 and 44.

## Running the notebooks

Create an environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open the notebooks and run them in order from `01` through `06`. Notebook 01
creates the shared processed data and vocabulary artifacts consumed by the model
notebooks. The dataset and GloVe files are downloaded automatically when needed.

For GPU-enabled PyTorch installations, follow the installation command appropriate
for the CUDA version available on the machine before installing the remaining
requirements.

## Repository layout

```text
01_data_preprocessing.ipynb
02_tfidf_logreg.ipynb
03_text_cnn.ipynb
04_bilstm.ipynb
05_distilbert.ipynb
06_comparison_and_error_analysis.ipynb
README.md
```
