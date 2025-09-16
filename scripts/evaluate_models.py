#!/usr/bin/env python3
"""
Evaluation script for news headline classification models.
Loads trained models and evaluates them on test data.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.baseline import BaselineClassifier
from models.lstm_model import LSTMClassifier
from preprocessing.text_cleaner import TextPreprocessor
from evaluation.metrics import ModelEvaluator
from utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str, test_size: float = 0.2) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load and split data for evaluation.
    
    Args:
        data_path: Path to data CSV
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels)
    """
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Convert source to binary labels
    label_map = {'nbc': 0, 'fox': 1}
    texts = df['headline'].tolist()
    labels = [label_map[source.lower()] for source in df['source']]
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    logger.info(f"Test set: {len(test_texts)} headlines")
    logger.info(f"NBC test headlines: {test_labels.count(0)}")
    logger.info(f"Fox test headlines: {test_labels.count(1)}")
    
    return train_texts, train_labels, test_texts, test_labels


def load_model(model_path: str, model_type: str):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to saved model
        model_type: Type of model ('baseline', 'lstm', 'bert', 'gpt2')
        
    Returns:
        Loaded model instance
    """
    logger.info(f"Loading {model_type} model from {model_path}")
    
    if model_type == 'baseline':
        return BaselineClassifier.load(model_path)
    elif model_type in ['lstm', 'lstm_pos']:
        return LSTMClassifier.load(model_path)
    elif model_type == 'bert':
        # Import here to avoid dependency issues if not using BERT
        from models.bert_model import BERTClassifier
        return BERTClassifier.load(model_path)
    elif model_type == 'gpt2':
        # Import here to avoid dependency issues if not using GPT-2
        from models.gpt2_model import GPT2Classifier
        return GPT2Classifier.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_single_model(model, 
                         model_name: str,
                         test_texts: List[str], 
                         test_labels: List[int],
                         output_dir: str) -> Dict:
    """
    Evaluate a single model and save results.
    
    Args:
        model: Trained model instance
        model_name: Name of the model
        test_texts: Test texts
        test_labels: Test labels
        output_dir: Directory to save results
        
    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Evaluating {model_name} model...")
    
    # Make predictions
    if hasattr(model, 'predict'):
        if model_name in ['lstm', 'lstm_pos']:
            predictions = model.predict(test_texts, use_pos_tags='pos' in model_name)
            probabilities = model.predict_proba(test_texts, use_pos_tags='pos' in model_name)
        else:
            predictions = model.predict(test_texts)
            probabilities = model.predict_proba(test_texts)
    else:
        raise ValueError(f"Model {model_name} does not have predict method")
    
    # Evaluate using model's evaluate method
    if hasattr(model, 'evaluate'):
        if model_name in ['lstm', 'lstm_pos']:
            results = model.evaluate(test_texts, test_labels, use_pos_tags='pos' in model_name)
        else:
            results = model.evaluate(test_texts, test_labels)
    else:
        # Manual evaluation
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
        
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average=None)
        macro_f1 = f1_score(test_labels, predictions, average='macro')
        
        results = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'nbc_precision': precision[0],
            'nbc_recall': recall[0],
            'nbc_f1': f1[0],
            'fox_precision': precision[1],
            'fox_recall': recall[1],
            'fox_f1': f1[1],
            'classification_report': classification_report(test_labels, predictions, 
                                                         target_names=['NBC', 'Fox News']),
            'confusion_matrix': confusion_matrix(test_labels, predictions).tolist()
        }
    
    # Add additional analysis
    results['model_name'] = model_name
    results['test_size'] = len(test_texts)
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'text': test_texts,
        'true_label': test_labels,
        'predicted_label': predictions,
        'nbc_probability': probabilities[:, 0] if len(probabilities.shape) > 1 else 1 - probabilities,
        'fox_probability': probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities
    })
    predictions_df.to_csv(os.path.join(output_dir, f'{model_name}_predictions.csv'), index=False)
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NBC', 'Fox News'], yticklabels=['NBC', 'Fox News'])
    plt.title(f'{model_name.upper()} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"{model_name} - Accuracy: {results['accuracy']:.3f}, F1: {results['macro_f1']:.3f}")
    
    return results


def compare_models(all_results: Dict[str, Dict], output_dir: str) -> None:
    """
    Create comparison visualizations and analysis.
    
    Args:
        all_results: Dictionary of model results
        output_dir: Directory to save comparison plots
    """
    logger.info("Creating model comparison analysis...")
    
    # Extract metrics for comparison
    models = list(all_results.keys())
    metrics = ['accuracy', 'macro_f1', 'nbc_f1', 'fox_f1']
    
    comparison_data = {metric: [] for metric in metrics}
    comparison_data['model'] = []
    
    for model_name in models:
        comparison_data['model'].append(model_name)
        for metric in metrics:
            comparison_data[metric].append(all_results[model_name][metric])
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        bars = ax.bar(models, comparison_data[metric], 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
        
        # Add value labels on bars
        for bar, value in zip(bars, comparison_data[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed performance table
    detailed_metrics = ['accuracy', 'macro_f1', 'nbc_precision', 'nbc_recall', 'nbc_f1', 
                       'fox_precision', 'fox_recall', 'fox_f1']
    
    detailed_comparison = pd.DataFrame()
    for model_name in models:
        model_metrics = {}
        for metric in detailed_metrics:
            model_metrics[metric] = all_results[model_name][metric]
        detailed_comparison[model_name] = model_metrics
    
    # Save detailed comparison
    detailed_comparison.to_csv(os.path.join(output_dir, 'detailed_model_comparison.csv'))
    
    # Print summary to console
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    # Sort models by macro F1 score
    sorted_models = sorted(models, key=lambda x: all_results[x]['macro_f1'], reverse=True)
    
    for i, model_name in enumerate(sorted_models):
        result = all_results[model_name]
        print(f"\n{i+1}. {model_name.upper()}:")
        print(f"   Accuracy: {result['accuracy']:.3f}")
        print(f"   Macro F1: {result['macro_f1']:.3f}")
        print(f"   NBC F1:   {result['nbc_f1']:.3f}")
        print(f"   Fox F1:   {result['fox_f1']:.3f}")
    
    # Best model analysis
    best_model = sorted_models[0]
    best_result = all_results[best_model]
    
    print(f"\n🏆 BEST MODEL: {best_model.upper()}")
    print(f"   Overall Performance: {best_result['macro_f1']:.3f} F1 Score")
    
    # Performance insights
    print(f"\n📊 KEY INSIGHTS:")
    
    # Check if LSTM with POS performed best
    if 'lstm_pos' in all_results and best_model == 'lstm_pos':
        print("   ✅ LSTM + POS tags achieved the best performance")
        print("   ✅ Feature engineering outperformed complex transformers")
    
    # Compare transformer performance
    transformer_models = [m for m in models if m in ['bert', 'gpt2']]
    if transformer_models:
        transformer_avg = np.mean([all_results[m]['macro_f1'] for m in transformer_models])
        lstm_score = all_results.get('lstm_pos', {}).get('macro_f1', 0)
        if lstm_score > transformer_avg:
            print(f"   📈 LSTM+POS ({lstm_score:.3f}) outperformed transformers (avg: {transformer_avg:.3f})")
    
    # Preprocessing impact
    if 'lstm' in all_results and 'lstm_pos' in all_results:
        lstm_basic = all_results['lstm']['macro_f1']
        lstm_pos = all_results['lstm_pos']['macro_f1']
        improvement = lstm_pos - lstm_basic
        print(f"   🔧 POS tags improved LSTM by {improvement:.3f} F1 points")


def analyze_errors(all_results: Dict[str, Dict], test_texts: List[str], test_labels: List[int], output_dir: str) -> None:
    """
    Analyze prediction errors across models.
    
    Args:
        all_results: Dictionary of model results
        test_texts: Test texts
        test_labels: Test labels
        output_dir: Directory to save analysis
    """
    logger.info("Analyzing prediction errors...")
    
    # Load predictions from all models
    error_analysis = pd.DataFrame({
        'text': test_texts,
        'true_label': test_labels,
        'true_source': ['NBC' if label == 0 else 'Fox News' for label in test_labels]
    })
    
    # Add predictions from each model
    for model_name in all_results.keys():
        predictions_file = os.path.join(output_dir, f'{model_name}_predictions.csv')
        if os.path.exists(predictions_file):
            model_preds = pd.read_csv(predictions_file)
            error_analysis[f'{model_name}_pred'] = model_preds['predicted_label']
            error_analysis[f'{model_name}_correct'] = (model_preds['predicted_label'] == model_preds['true_label'])
    
    # Find commonly misclassified examples
    prediction_cols = [col for col in error_analysis.columns if col.endswith('_correct')]
    error_analysis['total_correct'] = error_analysis[prediction_cols].sum(axis=1)
    error_analysis['all_wrong'] = error_analysis['total_correct'] == 0
    error_analysis['all_correct'] = error_analysis['total_correct'] == len(prediction_cols)
    
    # Save error analysis
    error_analysis.to_csv(os.path.join(output_dir, 'error_analysis.csv'), index=False)
    
    # Examples that all models got wrong
    all_wrong = error_analysis[error_analysis['all_wrong']]
    if not all_wrong.empty:
        print(f"\n❌ EXAMPLES ALL MODELS GOT WRONG ({len(all_wrong)} total):")
        for _, row in all_wrong.head(5).iterrows():
            print(f"   True: {row['true_source']} | Text: {row['text'][:80]}...")
    
    # Examples that all models got right
    all_correct = error_analysis[error_analysis['all_correct']]
    if not all_correct.empty:
        print(f"\n✅ EXAMPLES ALL MODELS GOT RIGHT ({len(all_correct)} total):")
        for _, row in all_correct.head(3).iterrows():
            print(f"   True: {row['true_source']} | Text: {row['text'][:80]}...")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate news headline classification models')
    
    parser.add_argument('--models-dir', required=True, help='Directory containing trained models')
    parser.add_argument('--data', required=True, help='Path to data CSV file')
    parser.add_argument('--output', required=True, help='Output directory for evaluation results')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--models', nargs='+', 
                       choices=['baseline', 'lstm', 'lstm_pos', 'bert', 'gpt2'],
                       help='Specific models to evaluate (default: all available)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    train_texts, train_labels, test_texts, test_labels = load_data(args.data, args.test_size)
    
    # Find available models
    model_files = {
        'baseline': os.path.join(args.models_dir, 'baseline_model.pkl'),
        'lstm': os.path.join(args.models_dir, 'lstm_model.pkl'),
        'lstm_pos': os.path.join(args.models_dir, 'lstm_model_pos.pkl'),
        'bert': os.path.join(args.models_dir, 'bert_model.pkl'),
        'gpt2': os.path.join(args.models_dir, 'gpt2_model.pkl')
    }
    
    # Filter to requested models or available models
    if args.models:
        models_to_evaluate = args.models
    else:
        models_to_evaluate = [name for name, path in model_files.items() if os.path.exists(path)]
    
    if not models_to_evaluate:
        logger.error("No models found to evaluate!")
        return
    
    logger.info(f"Evaluating models: {models_to_evaluate}")
    
    # Evaluate each model
    all_results = {}
    
    for model_name in models_to_evaluate:
        model_path = model_files[model_name]
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            continue
        
        try:
            # Load and evaluate model
            model = load_model(model_path, model_name)
            results = evaluate_single_model(
                model, model_name, test_texts, test_labels, args.output
            )
            all_results[model_name] = results
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue
    
    if not all_results:
        logger.error("No models were successfully evaluated!")
        return
    
    # Save overall results
    with open(os.path.join(args.output, 'evaluation_results.json'), 'w') as f:
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
        json.dump(serializable_results, f, indent=2)
    
    # Create comparison analysis
    compare_models(all_results, args.output)
    
    # Error analysis
    analyze_errors(all_results, test_texts, test_labels, args.output)
    
    logger.info(f"Evaluation complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
