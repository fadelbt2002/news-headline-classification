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
│   │   ├── __init__.py
│   │   └── metrics.py         # Evaluation utilities
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Configuration settings
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── scripts/
│   ├── train_models.py        # Training pipeline
│   └── evaluate_models.py     # Evaluation pipeline
├── results/
│   ├── model_performance.json
│   ├── confusion_matrices/
│   └── visualizations/
├── docs/
│   └── final_report.pdf       # Detailed research report
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_scraper.py
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

# Generate visualizations
python scripts/generate_plots.py
```

## 🛠️ Usage Examples

### Basic Classification

```python
from src.models.lstm_model import LSTMClassifier
from src.preprocessing.text_cleaner import TextPreprocessor

# Initialize model and preprocessor
model = LSTMClassifier.load('results/best_lstm_model.pkl')
preprocessor = TextPreprocessor(use_pos_tags=True)

# Classify a headline
headline = "Biden announces new climate initiative"
processed = preprocessor.transform([headline])
prediction = model.predict(processed)
print(f"Predicted source: {'NBC' if prediction[0] == 0 else 'Fox News'}")
```

### Model Training

```python
from src.models.bert_model import BERTClassifier
from src.utils.config import Config

# Initialize and train BERT model
config = Config()
model = BERTClassifier(config)
model.train(train_data, val_data, epochs=10)
model.save('results/bert_model.pkl')
```

## 📊 Data Analysis

Our exploratory data analysis revealed key differences between Fox News and NBC headlines:

- **Length**: Fox News headlines average 12.3 words vs NBC's 11.8 words
- **Vocabulary**: Different emphasis on political figures and institutions
- **POS Patterns**: NBC uses more proper nouns and determiners; Fox uses more adjectives
- **Stylistic Cues**: Punctuation and capitalization patterns differ significantly

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

## 📝 Key Insights

1. **Preprocessing Paradox**: Common NLP preprocessing techniques hurt performance
2. **Context Limitations**: Short headlines limit the advantage of transformer models
3. **Syntactic Signals**: Explicit POS features help simpler models compete with transformers
4. **Style Over Content**: Classification relies more on stylistic differences than topic

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📚 References

1. HaCohen-Kerner, Y., Miller, J., & Yigal, G. (2020). The influence of preprocessing on text classification using a bag-of-words representation. PLOS ONE, 15(5), e0232525.


## 📧 Contact

For questions about this project, please open an issue or contact the team members.

---
