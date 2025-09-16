# Beyond the Words: News Headline Classification

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
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   │   ├── __init__.py
│   │   └── scraper.py         # Beautiful Soup web scraper
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py    # Text preprocessing utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py        # TF-IDF + Logistic Regression
│   │   ├── lstm_model.py      # BiLSTM with POS embeddings
│   │   ├── bert_model.py      # Fine-tuned BERT classifier
│   │   └── gpt2_model.py      # GPT-2 with POS features
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # Evaluation utilities
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Configuration settings
├── scripts/
│   ├── train_models.py        # Training pipeline
│   └── evaluate_models.py     # Evaluation pipeline
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/news-headline-classification.git
cd news-headline-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Collection

```bash
# Scrape headlines from provided URLs
python scripts/collect_data.py --urls data/urls.csv --output data/raw/
```

### Training Models

```bash
# Train all models
python scripts/train_models.py

# Train specific model
python scripts/train_models.py --model lstm --use-pos-tags
```

### Evaluation

```bash
# Evaluate all models
python scripts/evaluate_models.py

```

## 🔬 Methodology

### Data Collection
- Web scraping using Beautiful Soup
- Headlines collected from provided URL lists
- 80/20 train/validation split with stratification

### Preprocessing Variants
1. **Raw**: Minimal preprocessing (best for transformers)
2. **Normalized**: Lowercasing, punctuation removal, lemmatization
3. **Enhanced**: Raw text + POS tags (best for LSTM)

### Model Architectures

**LSTM**: BiLSTM with GloVe embeddings and POS tag embeddings
**BERT**: Fine-tuned BERT-base-uncased with custom classification head
**GPT-2**: GPT-2 with custom POS integration layer
**Baseline**: TF-IDF features with Logistic Regression

## 📚 References
[1] HaCohen-Kerner, Y., Miller, J., & Yigal, G. (2020). The influence of preprocessing on text classification
using a bag-of-words representation. PLOS ONE, 15(5), e0232525. https://journals.plos.org/
plosone/article?id=10.1371/journal.pone.02325258

[2] Aggarwal, A., Agrahari, A., & Janghel, R. R. (2020). Classification of fake news by fine-tuning deep
bidirectional transformer-based language model. EAI Endorsed Transactions on Scalable Information
Systems, 7(25). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7519200

[3] Sabiri, B., Elhassouny, A., & Maalmi, K. (2023). Analyzing BERT’s performance compared to tradi-
tional text classification models. In Proceedings of the 25th International Conference on Enterprise In-
formation Systems (ICEIS), Vol. 2, pp. 572–582. https://www.scitepress.org/Papers/2023/118605

[4] Chotirat, S., & Meesad, P. (2021). Part-of-speech tagging enhancement for question classification
with deep learning. Heliyon, 7(10), e08216. https://www.ncbi.nlm.nih.gov/pmc/articles/
PMC8540490

[5] Tenney, I., Das, D., & Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. In Proceedings
of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 4593–4601.
https://arxiv.org/abs/1905.05950


## 📧 Contact

For questions about this project, please open an issue.

---
