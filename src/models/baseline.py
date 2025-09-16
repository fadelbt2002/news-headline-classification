"""
Baseline TF-IDF + Logistic Regression model for news headline classification.
This serves as the baseline model that achieved 68% accuracy in our experiments.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import pickle
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

logger = logging.getLogger(__name__)


class BaselineClassifier:
    """TF-IDF + Logistic Regression baseline classifier."""
    
    def __init__(self, 
                 max_features: int = 100,
                 remove_stopwords: bool = True,
                 max_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize the baseline classifier.
        
        Args:
            max_features: Maximum number of TF-IDF features
            remove_stopwords: Whether to remove stopwords
            max_iter: Maximum iterations for LogisticRegression
            random_state: Random state for reproducibility
        """
        self.max_features = max_features
        self.remove_stopwords = remove_stopwords
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Initialize pipeline
        self.pipeline = None
        self._create_pipeline()
    
    def _create_pipeline(self) -> None:
        """Create the TF-IDF + Logistic Regression pipeline."""
        # TF-IDF Vectorizer
        tfidf_params = {
            'max_features': self.max_features,
            'lowercase': True,
            'analyzer': 'word',
            'token_pattern': r'\b\w+\b',
            'ngram_range': (1, 1)
        }
        
        if self.remove_stopwords:
            tfidf_params['stop_words'] = 'english'
        
        vectorizer = TfidfVectorizer(**tfidf_params)
        
        # Logistic Regression
        classifier = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', classifier)
        ])
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        """
        Train the baseline model.
        
        Args:
            texts: Training texts
            labels: Training labels (0 for NBC, 1 for Fox)
        """
        logger.info("Training baseline TF-IDF + Logistic Regression model")
        
        # Convert to numpy arrays
        X = np.array(texts)
        y = np.array(labels)
        
        # Train pipeline
        self.pipeline.fit(X, y)
        
        logger.info("Baseline model training completed")
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            Array of predictions (0 for NBC, 1 for Fox)
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before making predictions")
        
        X = np.array(texts)
        return self.pipeline.predict(X)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            Array of prediction probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before making predictions")
        
        X = np.array(texts)
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            texts: Test texts
            labels: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None)
        macro_f1 = f1_score(labels, predictions, average='macro')
        
        results = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'nbc_precision': precision[0],
            'nbc_recall': recall[0],
            'nbc_f1': f1[0],
            'fox_precision': precision[1],
            'fox_recall': recall[1],
            'fox_f1': f1[1],
            'classification_report': classification_report(labels, predictions, 
                                                         target_names=['NBC', 'Fox News']),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List]:
        """
        Get most important features for each class.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top features for each class
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before extracting features")
        
        # Get feature names and coefficients
        vectorizer = self.pipeline.named_steps['tfidf']
        classifier = self.pipeline.named_steps['classifier']
        
        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]
        
        # Get top features for each class
        # Positive coefficients indicate Fox News (class 1)
        # Negative coefficients indicate NBC (class 0)
        
        # Sort by coefficient value
        feature_coef_pairs = list(zip(feature_names, coefficients))
        feature_coef_pairs.sort(key=lambda x: x[1])
        
        # Top NBC features (most negative coefficients)
        nbc_features = feature_coef_pairs[:top_n]
        
        # Top Fox features (most positive coefficients)
        fox_features = feature_coef_pairs[-top_n:]
        fox_features.reverse()  # Most positive first
        
        return {
            'nbc_features': [(feat, coef) for feat, coef in nbc_features],
            'fox_features': [(feat, coef) for feat, coef in fox_features]
        }
    
    def cross_validate(self, texts: List[str], labels: List[int], cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            texts: All texts
            labels: All labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        X = np.array(texts)
        y = np.array(labels)
        
        # Stratified cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Calculate scores
        accuracy_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='f1_macro')
        
        results = {
            'accuracy_mean': accuracy_scores.mean(),
            'accuracy_std': accuracy_scores.std(),
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'accuracy_scores': accuracy_scores.tolist(),
            'f1_scores': f1_scores.tolist()
        }
        
        logger.info(f"CV Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
        logger.info(f"CV F1: {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.pipeline is None:
            raise ValueError("No model to save")
        
        model_data = {
            'pipeline': self.pipeline,
            'max_features': self.max_features,
            'remove_stopwords': self.remove_stopwords,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Baseline model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaselineClassifier':
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded BaselineClassifier instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create classifier instance
        classifier = cls(
            max_features=model_data['max_features'],
            remove_stopwords=model_data['remove_stopwords'],
            max_iter=model_data['max_iter'],
            random_state=model_data['random_state']
        )
        
        # Restore trained pipeline
        classifier.pipeline = model_data['pipeline']
        
        logger.info(f"Baseline model loaded from {filepath}")
        return classifier


def analyze_baseline_features(texts: List[str], 
                            labels: List[int], 
                            max_features: int = 100) -> Dict:
    """
    Analyze which features are most important for baseline classification.
    
    Args:
        texts: All texts
        labels: All labels
        max_features: Maximum TF-IDF features
        
    Returns:
        Feature analysis results
    """
    # Train baseline model
    classifier = BaselineClassifier(max_features=max_features)
    classifier.train(texts, labels)
    
    # Get feature importance
    feature_importance = classifier.get_feature_importance(top_n=20)
    
    # Get TF-IDF vocabulary statistics
    vectorizer = classifier.pipeline.named_steps['tfidf']
    vocab = vectorizer.vocabulary_
    
    # Separate texts by class
    nbc_texts = [texts[i] for i, label in enumerate(labels) if label == 0]
    fox_texts = [texts[i] for i, label in enumerate(labels) if label == 1]
    
    analysis = {
        'total_features': len(vocab),
        'nbc_samples': len(nbc_texts),
        'fox_samples': len(fox_texts),
        'feature_importance': feature_importance,
        'vocabulary_size': len(vocab)
    }
    
    return analysis


# Example usage
if __name__ == "__main__":
    # Example data
    sample_texts = [
        "Biden announces new climate initiative for renewable energy",
        "Trump criticizes latest Democratic proposal in fiery speech",
        "Stock market reaches new highs amid economic recovery",
        "Local weather updates show severe storms approaching",
        "Breaking news from Washington on infrastructure bill",
        "Fox News investigation reveals new evidence",
        "NBC reports on latest polling data from swing states",
        "Analysis of current political climate and voter sentiment"
    ]
    sample_labels = [0, 1, 0, 1, 0, 1, 0, 1]  # 0 = NBC, 1 = Fox
    
    # Train baseline model
    baseline = BaselineClassifier(max_features=50)
    baseline.train(sample_texts, sample_labels)
    
    # Make predictions
    test_texts = [
        "New policy announcement from White House",
        "Conservative analysis of recent legislation"
    ]
    
    predictions = baseline.predict(test_texts)
    probabilities = baseline.predict_proba(test_texts)
    
    print("Baseline Model Results:")
    print("=" * 30)
    
    for i, (text, pred, prob) in enumerate(zip(test_texts, predictions, probabilities)):
        source = "Fox News" if pred == 1 else "NBC"
        confidence = prob[pred]
        print(f"\nText {i+1}: {text}")
        print(f"Prediction: {source} (confidence: {confidence:.3f})")
    
    # Evaluate on training data (just for demonstration)
    results = baseline.evaluate(sample_texts, sample_labels)
    print(f"\nTraining Accuracy: {results['accuracy']:.3f}")
    print(f"Macro F1: {results['macro_f1']:.3f}")
    
    # Show most important features
    feature_importance = baseline.get_feature_importance(top_n=5)
    print(f"\nTop NBC features:")
    for feat, coef in feature_importance['nbc_features']:
        print(f"  {feat}: {coef:.3f}")
    
    print(f"\nTop Fox features:")
    for feat, coef in feature_importance['fox_features']:
        print(f"  {feat}: {coef:.3f}")
