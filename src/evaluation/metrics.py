"""
Evaluation metrics and utilities for news headline classification.
Provides comprehensive evaluation tools for model comparison.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: Names of classes (default: ['NBC', 'Fox News'])
        """
        self.class_names = class_names or ['NBC', 'Fox News']
    
    def evaluate_predictions(self, 
                           y_true: List[int], 
                           y_pred: List[int], 
                           y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        for i, class_name in enumerate(self.class_names):
            results[f'{class_name.lower()}_precision'] = precision[i]
            results[f'{class_name.lower()}_recall'] = recall[i]
            results[f'{class_name.lower()}_f1'] = f1[i]
            results[f'{class_name.lower()}_support'] = support[i]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # Classification report
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names
        )
        
        # Probability-based metrics (if available)
        if y_proba is not None:
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                # Multi-class probabilities
                positive_proba = y_proba[:, 1]  # Probability of positive class (Fox News)
            else:
                # Single probability score
                positive_proba = y_proba.flatten()
            
            try:
                results['roc_auc'] = roc_auc_score(y_true, positive_proba)
                results['average_precision'] = average_precision_score(y_true, positive_proba)
            except ValueError as e:
                logger.warning(f"Could not compute ROC AUC or Average Precision: {e}")
        
        return results
    
    def plot_confusion_matrix(self, 
                            y_true: List[int], 
                            y_pred: List[int],
                            title: str = "Confusion Matrix",
                            figsize: Tuple[int, int] = (8, 6),
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot (optional)
            
        Returns:
            matplotlib Figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add accuracy to title
        accuracy = accuracy_score(y_true, y_pred)
        plt.suptitle(f'{title}\nAccuracy: {accuracy:.3f}', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curve(self, 
                      y_true: List[int], 
                      y_proba: np.ndarray,
                      title: str = "ROC Curve",
                      figsize: Tuple[int, int] = (8, 6),
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot (optional)
            
        Returns:
            matplotlib Figure
        """
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            positive_proba = y_proba[:, 1]
        else:
            positive_proba = y_proba.flatten()
        
        fpr, tpr, _ = roc_curve(y_true, positive_proba)
        auc = roc_auc_score(y_true, positive_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, 
                                   y_true: List[int], 
                                   y_proba: np.ndarray,
                                   title: str = "Precision-Recall Curve",
                                   figsize: Tuple[int, int] = (8, 6),
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot (optional)
            
        Returns:
            matplotlib Figure
        """
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            positive_proba = y_proba[:, 1]
        else:
            positive_proba = y_proba.flatten()
        
        precision, recall, _ = precision_recall_curve(y_true, positive_proba)
        avg_precision = average_precision_score(y_true, positive_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, linewidth=2, 
                label=f'PR Curve (AP = {avg_precision:.3f})')
        
        # Baseline (random classifier)
        baseline = sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                   label=f'Random Classifier (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_calibration_curve(self, 
                             y_true: List[int], 
                             y_proba: np.ndarray,
                             n_bins: int = 10,
                             title: str = "Calibration Curve",
                             figsize: Tuple[int, int] = (8, 6),
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curve to assess probability calibration.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            n_bins: Number of bins for calibration
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot (optional)
            
        Returns:
            matplotlib Figure
        """
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            positive_proba = y_proba[:, 1]
        else:
            positive_proba = y_proba.flatten()
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, positive_proba, n_bins=n_bins
        )
        
        plt.figure(figsize=figsize)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                linewidth=2, label="Model")
        plt.
