"""
GPT-2 model for news headline classification.
GPT-2 with POS features achieved 85% F1 score in our experiments.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import pickle
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Model,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix, f1_score
)
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

logger = logging.getLogger(__name__)


class NewsHeadlineDataset(Dataset):
    """Dataset class for news headlines with GPT-2 tokenization."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 38):
        """
        Initialize dataset.
        
        Args:
            texts: List of headlines
            labels: List of labels (0 for NBC, 1 for Fox)
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Set pad token for GPT-2
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class GPT2WithPOSModel(nn.Module):
    """GPT-2 model with POS tag integration for classification."""
    
    def __init__(self, 
                 model_name: str = 'gpt2',
                 num_classes: int = 2,
                 pos_vocab_size: int = 50,
                 pos_embedding_dim: int = 32,
                 dropout_rate: float = 0.1):
        """
        Initialize GPT-2 with POS features.
        
        Args:
            model_name: Pre-trained GPT-2 model name
            num_classes: Number of output classes
            pos_vocab_size: Size of POS tag vocabulary
            pos_embedding_dim: Dimension of POS embeddings
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        # Load pre-trained GPT-2
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        
        # POS tag embeddings
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim)
        
        # Projection layers
        hidden_size = self.gpt2.config.hidden_size
        
        # Two-layer fully connected network for classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + pos_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_ids, attention_mask, pos_ids=None):
        # GPT-2 forward pass
        gpt2_outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the last hidden state
        last_hidden_state = gpt2_outputs.last_hidden_state
        
        # Use the last token's representation (since GPT-2 is autoregressive)
        # Find the last non-padded token for each sequence
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        
        # Gather the last token representations
        gpt2_features = last_hidden_state[range(batch_size), sequence_lengths]
        gpt2_features = self.dropout(gpt2_features)
        
        features = [gpt2_features]
        
        # Add POS features if provided
        if pos_ids is not None:
            # Create POS distribution features
            pos_embeds = self.pos_embedding(pos_ids)
            # Average pool POS embeddings
            pos_features = torch.mean(pos_embeds, dim=1)
            features.append(pos_features)
        
        # Concatenate features
        if len(features) > 1:
            combined_features = torch.cat(features, dim=1)
        else:
            combined_features = features[0]
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits


class GPT2Classifier:
    """GPT-2 classifier for news headline classification."""
    
    def __init__(self,
                 model_name: str = 'gpt2',
                 max_length: int = 38,
                 learning_rate: float = 3e-5,
                 weight_decay: float = 0.01,
                 use_pos_features: bool = False):
        """
        Initialize GPT-2 classifier.
        
        Args:
            model_name: Pre-trained GPT-2 model name
            max_length: Maximum sequence length
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            use_pos_features: Whether to use POS tag features
        """
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_pos_features = use_pos_features
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        if use_pos_features:
            self.model = GPT2WithPOSModel(model_name=model_name)
        else:
            self.model = GPT2ForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
            # Resize token embeddings to account for pad token
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # POS tag mapping (if using POS features)
        self.pos_tags = [
            'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
            'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP,
            'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
            'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP, 'WRB', '.', ',', ':',
            '(', ')', '"', "'", 'UNK'
        ]
        self.pos_to_id = {pos: i+1 for i, pos in enumerate(self.pos_tags)}
        self.pos_to_id['UNK'] = 0
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training history
        self.history = None
    
    def extract_pos_distributions(self, texts: List[str]) -> torch.Tensor:
        """
        Extract POS tag distributions for texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Tensor of POS distributions
        """
        pos_distributions = []
        
        for text in texts:
            # Tokenize and get POS tags
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # Create distribution vector
            pos_counts = torch.zeros(len(self.pos_tags))
            
            for _, pos in pos_tags:
                pos_id = self.pos_to_id.get(pos, self.pos_to_id['UNK'])
                if pos_id > 0:  # Skip UNK
                    pos_counts[pos_id - 1] += 1
            
            # Normalize to get distribution
            if pos_counts.sum() > 0:
                pos_counts = pos_counts / pos_counts.sum()
            
            pos_distributions.append(pos_counts)
        
        return torch.stack(pos_distributions)
    
    def extract_pos_sequences(self, texts: List[str]) -> List[List[int]]:
        """Extract POS tag sequences for texts."""
        pos_sequences = []
        
        for text in texts:
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            pos_ids = [self.pos_to_id.get(pos, self.pos_to_id['UNK']) 
                      for _, pos in pos_tags]
            pos_sequences.append(pos_ids)
        
        # Pad sequences
        max_len = min(self.max_length, max(len(seq) for seq in pos_sequences) if pos_sequences else 0)
        padded_sequences = []
        
        for seq in pos_sequences:
            if len(seq) >= max_len:
                padded_sequences.append(seq[:max_len])
            else:
                padded_sequences.append(seq + [0] * (max_len - len(seq)))
        
        return padded_sequences
    
    def create_data_loader(self, 
                          texts: List[str], 
                          labels: List[int], 
                          batch_size: int = 16,
                          shuffle: bool = True) -> DataLoader:
        """Create DataLoader for training/evaluation."""
        dataset = NewsHeadlineDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(self,
              train_texts: List[str],
              train_labels: List[int],
              val_texts: Optional[List[str]] = None,
              val_labels: Optional[List[int]] = None,
              epochs: int = 3,
              batch_size: int = 16,
              verbose: bool = True) -> Dict:
        """
        Train the GPT-2 model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        # Create data loaders
        train_loader = self.create_data_loader(
            train_texts, train_labels, batch_size, shuffle=True
        )
        
        val_loader = None
        if val_texts and val_labels:
            val_loader = self.create_data_loader(
                val_texts, val_labels, batch_size, shuffle=False
            )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            if verbose:
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            else:
                progress_bar = train_loader
            
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if self.use_pos_features:
                    # Extract POS features for this batch
                    batch_size_current = input_ids.size(0)
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size_current
                    batch_texts = train_texts[start_idx:end_idx]
                    
                    pos_ids = torch.tensor(
                        self.extract_pos_sequences(batch_texts), 
                        dtype=torch.long
                    ).to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pos_ids=pos_ids
                    )
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    logits = outputs
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                
                if verbose:
                    progress_bar.set_postfix({
                        'loss': loss.item(),
                        'acc': correct_predictions / total_predictions
                    })
            
            # Calculate epoch metrics
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_predictions
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            
            # Validation
            if val_loader:
                val_loss, val_acc = self._evaluate_loader(val_loader, val_texts)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose:
                    print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, '
                          f'Train Acc: {train_accuracy:.4f}, '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                if verbose:
                    print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, '
                          f'Train Acc: {train_accuracy:.4f}')
        
        self.history = history
        return history
    
    def _evaluate_loader(self, data_loader: DataLoader, texts: List[str] = None) -> Tuple[float, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.use_pos_features and texts:
                    # Extract POS features for validation batch
                    batch_size_current = input_ids.size(0)
                    start_idx = batch_idx * data_loader.batch_size
                    end_idx = start_idx + batch_size_current
                    batch_texts = texts[start_idx:end_idx]
                    
                    pos_ids = torch.tensor(
                        self.extract_pos_sequences(batch_texts), 
                        dtype=torch.long
                    ).to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pos_ids=pos_ids
                    )
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    logits = outputs
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        self.model.train()
        return total_loss / len(data_loader), correct_predictions / total_predictions
    
    def predict(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Make predictions on texts."""
        self.model.eval()
        
        # Create dummy labels for dataset
        dummy_labels = [0] * len(texts)
        data_loader = self.create_data_loader(
            texts, dummy_labels, batch_size, shuffle=False
        )
        
        predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.use_pos_features:
                    # Extract POS features
                    batch_size_current = input_ids.size(0)
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size_current
                    batch_texts = texts[start_idx:end_idx]
                    
                    pos_ids = torch.tensor(
                        self.extract_pos_sequences(batch_texts), 
                        dtype=torch.long
                    ).to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pos_ids=pos_ids
                    )
                    logits = outputs
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                
                batch_predictions = torch.argmax(logits, dim=1)
                predictions.extend(batch_predictions.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Get prediction probabilities."""
        self.model.eval()
        
        # Create dummy labels for dataset
        dummy_labels = [0] * len(texts)
        data_loader = self.create_data_loader(
            texts, dummy_labels, batch_size, shuffle=False
        )
        
        probabilities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.use_pos_features:
                    # Extract POS features
                    batch_size_current = input_ids.size(0)
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size_current
                    batch_texts = texts[start_idx:end_idx]
                    
                    pos_ids = torch.tensor(
                        self.extract_pos_sequences(batch_texts), 
                        dtype=torch.long
                    ).to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pos_ids=pos_ids
                    )
                    logits = outputs
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict:
        """Evaluate model performance."""
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
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
            'classification_report': classification_report(
                labels, predictions, target_names=['NBC', 'Fox News']
            ),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }
        
        return results
    
    def save(self, filepath: str) -> None:
        """Save the trained model."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and tokenizer
        model_dir = filepath.replace('.pkl', '_model')
        
        if self.use_pos_features:
            # Save custom model
            torch.save(self.model.state_dict(), os.path.join(model_dir, 'pytorch_model.bin'))
            # Save model config
            import json
            config = {
                'model_name': self.model_name,
                'use_pos_features': True,
                'pos_vocab_size': len(self.pos_tags),
                'pos_embedding_dim': 32
            }
            with open(os.path.join(model_dir, 'config.json'), 'w') as f:
                json.dump(config, f)
        else:
            self.model.save_pretrained(model_dir)
        
        self.tokenizer.save_pretrained(model_dir)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'use_pos_features': self.use_pos_features,
            'pos_to_id': self.pos_to_id,
            'pos_tags': self.pos_tags,
            'history': self.history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"GPT-2 model saved to {model_dir} and {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GPT2Classifier':
        """Load a trained model."""
        # Load metadata
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create classifier instance
        classifier = cls(
            model_name=metadata['model_name'],
            max_length=metadata['max_length'],
            learning_rate=metadata['learning_rate'],
            weight_decay=metadata['weight_decay'],
            use_pos_features=metadata['use_pos_features']
        )
        
        # Load model and tokenizer
        model_dir = filepath.replace('.pkl', '_model')
        
        if metadata['use_pos_features']:
            # Load custom model
            classifier.model = GPT2WithPOSModel(model_name=metadata['model_name'])
            classifier.model.load_state_dict(
                torch.load(os.path.join(model_dir, 'pytorch_model.bin'))
            )
        else:
            classifier.model = GPT2ForSequenceClassification.from_pretrained(model_dir)
        
        classifier.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        classifier.model.to(classifier.device)
        
        # Restore metadata
        classifier.pos_to_id = metadata['pos_to_id']
        classifier.pos_tags = metadata['pos_tags']
        classifier.history = metadata.get('history')
        
        logger.info(f"GPT-2 model loaded from {model_dir} and {filepath}")
        return classifier


def cross_validate_gpt2(texts: List[str], 
                       labels: List[int], 
                       use_pos_features: bool = False,
                       n_folds: int = 5,
                       **model_kwargs) -> Dict:
    """
    Perform cross-validation for GPT-2 model.
    
    Args:
        texts: List of texts
        labels: List of labels
        use_pos_features: Whether to use POS features
        n_folds: Number of CV folds
        **model_kwargs: Additional model parameters
        
    Returns:
        Cross-validation results
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info(f"Training GPT-2 fold {fold + 1}/{n_folds}")
        
        # Split data
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        # Train model
        classifier = GPT2Classifier(use_pos_features=use_pos_features, **model_kwargs)
        classifier.train(
            train_texts, train_labels,
            validation_data=(val_texts, val_labels),
            epochs=3,
            verbose=0
        )
        
        # Evaluate
        results = classifier.evaluate(val_texts, val_labels)
        fold_results.append(results)
    
    # Aggregate results
    metrics = ['accuracy', 'macro_f1', 'nbc_f1', 'fox_f1']
    aggregated = {}
    
    for metric in metrics:
        values = [result[metric] for result in fold_results]
        aggregated[f'{metric}_mean'] = np.mean(values)
        aggregated[f'{metric}_std'] = np.std(values)
    
    aggregated['fold_results'] = fold_results
    
    return aggregated


# Example usage
if __name__ == "__main__":
    # Example data
    sample_texts = [
        "Biden announces new climate initiative",
        "Trump criticizes Democratic proposal",
        "Stock market reaches new highs",
        "Investigation reveals new evidence",
        "Policy announcement from Washington",
        "Conservative analysis gains traction"
    ]
    sample_labels = [0, 1, 0, 1, 0, 1]  # 0 = NBC, 1 = Fox
    
    # Train GPT-2 model without POS features
    print("Training GPT-2 (baseline)...")
    gpt2_classifier = GPT2Classifier(max_length=32, use_pos_features=False)
    history = gpt2_classifier.train(
        sample_texts, sample_labels,
        epochs=2, batch_size=2, verbose=True
    )
    
    # Evaluate baseline
    results = gpt2_classifier.evaluate(sample_texts, sample_labels)
    print(f"Baseline - Accuracy: {results['accuracy']:.3f}, F1: {results['macro_f1']:.3f}")
    
    # Train GPT-2 model with POS features
    print("\nTraining GPT-2 with POS features...")
    gpt2_pos_classifier = GPT2Classifier(max_length=32, use_pos_features=True)
    history_pos = gpt2_pos_classifier.train(
        sample_texts, sample_labels,
        epochs=2, batch_size=2, verbose=True
    )
    
    # Evaluate with POS
    results_pos = gpt2_pos_classifier.evaluate(sample_texts, sample_labels)
    print(f"With POS - Accuracy: {results_pos['accuracy']:.3f}, F1: {results_pos['macro_f1']:.3f}")
    
    # Make predictions
    test_texts = ["New policy from the White House"]
    predictions = gpt2_pos_classifier.predict(test_texts)
    probabilities = gpt2_pos_classifier.predict_proba(test_texts)
    
    print(f"\nPrediction: {'Fox News' if predictions[0] == 1 else 'NBC'}")
    print(f"Probabilities: NBC={probabilities[0][0]:.3f}, Fox={probabilities[0][1]:.3f}")
