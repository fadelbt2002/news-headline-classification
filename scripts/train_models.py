#!/usr/bin/env python3
"""
Training script for news headline classification models.
Supports LSTM, BERT, and GPT-2 models with various preprocessing options.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.lstm_model import LSTMClassifier, cross_validate_lstm
from models.baseline import BaselineClassifier
from preprocessing.text_cleaner import TextPreprocessor
from utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> Tuple[List[str], List[int]]:
    """
    Load headlines data from CSV.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Convert source to binary labels
    label_map = {'nbc': 0, 'fox': 1}
    texts = df['headline'].tolist()
    labels = [label_map[source.lower()] for source in df['source']]
    
    logger.info(f"Loaded {len(texts)} headlines")
    logger.info(f"NBC headlines: {labels.count(0)}")
    logger.info(f"Fox headlines: {labels.count(1)}")
    
    return texts, labels


def train_baseline_model(texts: List[str], 
                        labels: List[int], 
                        config: Config) -> Dict:
    """
    Train baseline TF-IDF + Logistic Regression model.
    
    Args:
        texts: Training texts
        labels: Training labels
        config: Configuration object
        
    Returns:
        Training results
    """
    logger.info("Training baseline model (TF-IDF + Logistic Regression)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train model
    baseline = BaselineClassifier(
        max_features=config.baseline.max_features,
        remove_stopwords=config.baseline.remove_stopwords
    )
    
    baseline.train(X_train, y_train)
    
    # Evaluate
    results = baseline.evaluate(X_test, y_test)
    
    # Save model
    os.makedirs(config.model_save_dir, exist_ok=True)
    baseline.save(os.path.join(config.model_save_dir, 'baseline_model.pkl'))
    
    logger.info(f"Baseline model - Accuracy: {results['accuracy']:.3f}, F1: {results['macro_f1']:.3f}")
    
    return {
        'model_type': 'baseline',
        'results': results,
        'config': config.baseline.__dict__
    }


def train_lstm_model(texts: List[str], 
                    labels: List[int], 
                    config: Config,
                    use_pos_tags: bool = True,
                    cross_validate: bool = False) -> Dict:
    """
    Train LSTM model.
    
    Args:
        texts: Training texts
        labels: Training labels
        config: Configuration object
        use_pos_tags: Whether to use POS tag features
        cross_validate: Whether to perform cross-validation
        
    Returns:
        Training results
    """
    logger.info(f"Training LSTM model (POS tags: {use_pos_tags})")
    
    if cross_validate:
        logger.info("Performing cross-validation")
        results = cross_validate_lstm(
            texts, labels, 
            use_pos_tags=use_pos_tags,
            n_folds=5,
            **config.lstm.__dict__
        )
        
        logger.info(f"CV Results - Accuracy: {results['accuracy_mean']:.3f}±{results['accuracy_std']:.3f}")
        logger.info(f"CV Results - F1: {results['macro_f1_mean']:.3f}±{results['macro_f1_std']:.3f}")
        
        return {
            'model_type': 'lstm',
            'use_pos_tags': use_pos_tags,
            'cross_validation': True,
            'results': results,
            'config': config.lstm.__dict__
        }
    
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Train model
        classifier = LSTMClassifier(**config.lstm.__dict__)
        history = classifier.train(
            X_train, y_train,
            validation_data=(X_val, y_val),
            use_pos_tags=use_pos_tags,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            verbose=1
        )
        
        # Evaluate
        results = classifier.evaluate(X_test, y_test, use_pos_tags)
        
        # Save model
        os.makedirs(config.model_save_dir, exist_ok=True)
        model_name = f"lstm_model{'_pos' if use_pos_tags else ''}.pkl"
        classifier.save(os.path.join(config.model_save_dir, model_name))
        
        logger.info(f"LSTM model - Accuracy: {results['accuracy']:.3f}, F1: {results['macro_f1']:.3f}")
        
        return {
            'model_type': 'lstm',
            'use_pos_tags': use_pos_tags,
            'cross_validation': False,
            'results': results,
            'training_history': history,
            'config': config.lstm.__dict__
        }


def train_all_models(data_path: str, 
                    output_dir: str,
                    config_path: str = None,
                    models: List[str] = None,
                    cross_validate: bool = False) -> Dict:
    """
    Train all specified models.
    
    Args:
        data_path: Path to training data
        output_dir: Directory to save results
        config_path: Path to configuration file
        models: List of models to train
        cross_validate: Whether to perform cross-validation
        
    Returns:
        All training results
    """
    # Load configuration
    config = Config(config_path) if config_path else Config()
    config.model_save_dir = output_dir
    
    # Load data
    texts, labels = load_data(data_path)
    
    # Default models if not specified
    if models is None:
        models = ['baseline', 'lstm', 'lstm_pos']
    
    all_results = {}
    
    # Train models
    if 'baseline' in models:
        try:
            results = train_baseline_model(texts, labels, config)
            all_results['baseline'] = results
        except Exception as e:
            logger.error(f"Error training baseline model: {e}")
    
    if 'lstm' in models:
        try:
            results = train_lstm_model(texts, labels, config, use_pos_tags=False, cross_validate=cross_validate)
            all_results['lstm'] = results
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
    
    if 'lstm_pos' in models:
        try:
            results = train_lstm_model(texts, labels, config, use_pos_tags=True, cross_validate=cross_validate)
            all_results['lstm_pos'] = results
        except Exception as e:
            logger.error(f"Error training LSTM+POS model: {e}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'training_results.json')
    
    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = make_serializable(all_results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return all_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Train news headline classification models')
    
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--output', required=True, help='Output directory for models and results')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--models', nargs='+', 
                       choices=['baseline', 'lstm', 'lstm_pos', 'bert', 'gpt2'],
                       default=['baseline', 'lstm', 'lstm_pos'],
                       help='Models to train')
    parser.add_argument('--cross-validate', action='store_true',
                       help='Perform cross-validation instead of single train/test split')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Train models
    results = train_all_models(
        data_path=args.data,
        output_dir=args.output,
        config_path=args.config,
        models=args.models,
        cross_validate=args.cross_validate
    )
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        
        if model_results.get('cross_validation'):
            cv_results = model_results['results']
            print(f"  Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
            print(f"  Macro F1: {cv_results['macro_f1_mean']:.3f} ± {cv_results['macro_f1_std']:.3f}")
        else:
            eval_results = model_results['results']
            print(f"  Accuracy: {eval_results['accuracy']:.3f}")
            print(f"  Macro F1: {eval_results['macro_f1']:.3f}")
            print(f"  NBC F1: {eval_results['nbc_f1']:.3f}")
            print(f"  Fox F1: {eval_results['fox_f1']:.3f}")


if __name__ == "__main__":
    main()
