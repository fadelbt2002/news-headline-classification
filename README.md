# news-headline-classification# Beyond the Words: News Headline Classification

A machine learning project that classifies news headlines by source (NBC vs Fox News) using advanced NLP models including LSTM, BERT, and GPT-2.

## 📊 Project Overview

This project investigates how different text preprocessing techniques and model architectures affect the classification of news headlines between Fox News and NBC News. Our research focuses on three key hypotheses:

1. **Data Cleaning Hypothesis**: Standard text preprocessing may hurt performance by removing contextual cues
2. **Model Architecture Hypothesis**: Transformer models (BERT/GPT-2) will outperform LSTM models
3. **Feature Enhancement Hypothesis**: POS tags will improve LSTM performance but not transformer performance

## 🎯 Key Findings

- **Best Model**: BiLSTM with POS tags achieved **95% accuracy** and **0.95 F1 score**
- **Surprising Result**: LSTM outperformed both BERT and GPT-2 (which plateaued around 84-85% F1)
- **Preprocessing Impact**: Aggressive text normalization degrades performance across all models
- **Feature Engineering**: POS tags significantly help LSTM but don't improve transformers

## 📈 Model Performance

| Model | Accuracy | F1 Score | NBC F1 | Fox F1 |
|-------|----------|----------|---------|---------|
| **BiLSTM + POS** | **95%** | **0.95** | **0.95** | **0.95** |
| BERT | 84% | 0.84 | 0.84 | 0.85 |
| GPT-2 | 83% | 0.85 | 0.85 | 0.86 |
| TF-IDF + LR | 68% | 0.67 | 0.63 | 0.71 |

## 🏗️ Project Structure

```
news-headline-classification/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── raw/                    # Original scraped data
│   ├── processed/              # Cleaned and preprocessed data
│   └── urls.csv               # Source URLs for scraping
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   │   ├── __init__.py
│   │   └── scraper.py         # Beautiful Soup web scraper
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py    # Text preprocessing utilities
│   │   └── feature_extractor.py # POS/NER feature extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py        # TF-IDF + Logistic Regression
│   │   ├── lstm_model.py      # BiLSTM with POS embeddings
│   │   ├── bert_model.py      # Fine-tuned BERT classifier
│   │   └── gpt2_model.py      # GPT-2 with POS features
│   ├── evaluation/
│   │   ├── _
