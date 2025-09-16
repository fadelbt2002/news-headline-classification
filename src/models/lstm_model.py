"""
BiLSTM model with POS tag embeddings for news headline classification.
This was the best-performing model in our experiments (95% accuracy).
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import pickle
import logging

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Embedding, 
    Input, Concatenate, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

logger = logging.getLogger(__name__)


class LSTMClassifier:
    """BiLSTM classifier with POS tag embeddings for news headline classification."""
    
    def __init__(self, 
                 max_features: int = 10000,
                 max_length: int = 50,
                 embedding_dim: int = 100,
                 lstm_units: int = 128,
                 pos_embedding_dim: int = 20,
                 dropout_rate: float = 0.3):
        """
        Initialize the LSTM classifier.
        
        Args:
            max_features: Maximum number of words in vocabulary
            max_length: Maximum sequence length
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
            pos_embedding_dim: Dimension of POS tag embeddings
            dropout_rate: Dropout rate for regularization
        """
        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.pos_embedding_dim = pos_embedding_dim
        self.dropout_rate = dropout_rate
        
        self.tokenizer = None
        self.pos_tokenizer = None
        self.model = None
        self.history = None
        
        # POS tag mapping
        self.pos_tags = [
            'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 
            'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 
            'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 
            'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.', ',', ':', 
            '(', ')', '"', "'", 'UNK'
        ]
        self.pos_to_id = {pos: i+1 for i, pos in enumerate(self.pos_tags)}
        self.pos_to_id['UNK'] = 0
    
    def extract_pos_tags(self, texts: List[str]) -> List[List[str]]:
        """
        Extract POS tags from texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of POS tag sequences
        """
        pos_sequences = []
        
        for text in texts:
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            pos_sequence = [pos for _, pos in pos_tags]
            pos_sequences.append(pos_sequence)
        
        return pos_sequences
    
    def encode_pos_sequences(self, pos_sequences: List[List[str]]) -> np.ndarray:
        """
        Encode POS tag sequences to numerical arrays.
        
        Args:
            pos_sequences: List of POS tag sequences
            
        Returns:
            Encoded POS sequences
        """
        encoded_sequences = []
        
        for sequence in pos_sequences:
            encoded = [self.pos_to_id.get(pos, self.pos_to_id['UNK']) for pos in sequence]
            encoded_sequences.append(encoded)
        
        return pad_sequences(encoded_sequences, maxlen=self.max_length, padding='post')
    
    def prepare_data(self, texts: List[str], use_pos_tags: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare text data for training/prediction.
        
        Args:
            texts: List of text strings
            use_pos_tags: Whether to include POS tag features
            
        Returns:
            Tuple of (text_sequences, pos_sequences)
        """
        # Tokenize text
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
        
        text_sequences = self.tokenizer.texts_to_sequences(texts)
        text_sequences = pad_sequences(text_sequences, maxlen=self.max_length, padding='post')
        
        pos_sequences = None
        if use_pos_tags:
            pos_tags = self.extract_pos_tags(texts)
            pos_sequences = self.encode_pos_sequences(pos_tags)
        
        return text_sequences, pos_sequences
    
    def build_model(self, use_pos_tags: bool = True) -> Model:
        """
        Build the BiLSTM model architecture.
        
        Args:
            use_pos_tags: Whether to include POS tag embeddings
            
        Returns:
            Compiled Keras model
        """
        # Text input branch
        text_input = Input(shape=(self.max_length,), name='text_input')
        text_embedding = Embedding(
            input_dim=self.max_features,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            mask_zero=True,
            name='text_embedding'
        )(text_input)
        
        text_lstm = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate),
            name='text_bilstm'
        )(text_embedding)
        text_pooled = GlobalMaxPooling1D(name='text_pooling')(text_lstm)
        
        inputs = [text_input]
        features = [text_pooled]
        
        # POS tag input branch (if enabled)
        if use_pos_tags:
            pos_input = Input(shape=(self.max_length,), name='pos_input')
            pos_embedding = Embedding(
                input_dim=len(self.pos_tags) + 1,
                output_dim=self.pos_embedding_dim,
                input_length=self.max_length,
                mask_zero=True,
                name='pos_embedding'
            )(pos_input)
            
            pos_lstm = Bidirectional(
                LSTM(32, return_sequences=True, dropout=self.dropout_rate),
                name='pos_bilstm'
            )(pos_embedding)
            pos_pooled = GlobalMaxPooling1D(name='pos_pooling')(pos_lstm)
            
            inputs.append(pos_input)
            features.append(pos_pooled)
        
        # Combine features
        if len(features) > 1:
            combined = Concatenate(name='feature_concat')(features)
        else:
            combined = features[0]
        
        # Classification head
        dense1 = Dense(64, activation='relu', name='dense1')(combined)
        dropout1 = Dropout(self.dropout_rate, name='dropout1')(dense1)
        dense2 = Dense(32, activation='relu', name='dense2')(dropout1)
        dropout2 = Dropout(self.dropout_rate, name='dropout2')(dense2)
        
        output = Dense(1, activation='sigmoid', name='output')(dropout2)
        
        model = Model(inputs=inputs, outputs=output, name='BiLSTM_Classifier')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=5e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, 
              texts: List[str], 
              labels: List[int],
              validation_data: Optional[Tuple[List[str], List[int]]] = None,
              use_pos_tags: bool = True,
              epochs: int = 20,
              batch_size: int = 16,
              verbose: int = 1) -> Dict:
        """
        Train the model.
        
        Args:
            texts: Training texts
            labels: Training labels (0 for NBC, 1 for Fox)
            validation_data: Optional validation data
            use_pos_tags: Whether to use POS tag features
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Prepare training data
        X_text, X_pos = self.prepare_data(texts, use_pos_tags)
        y = np.array(labels)
        
        # Prepare model inputs
        if use_pos_tags:
            X_train = [X_text, X_pos]
        else:
            X_train = X_text
        
        # Prepare validation data
        X_val = None
        y_val = None
        if validation_data:
            val_texts, val_labels = validation_data
            X_val_text, X_val_pos = self.prepare_data(val_texts, use_pos_tags)
            
            if use_pos_tags:
                X_val = [X_val_text, X_val_pos]
            else:
                X_val = X_val_text
            y_val = np.array(val_labels)
        
        # Build model
        self.model = self.build_model(use_pos_tags)
        
        if verbose:
            self.model.summary()
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=verbose
            )
        ]
        
        # Train model
        validation_data_keras = (X_val, y_val) if validation_data else None
        
        self.history = self.model.fit(
            X_train, y,
            validation_data=validation_data_keras,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history.history
    
    def predict(self, texts: List[str], use_pos_tags: bool = True) -> np.ndarray:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to classify
            use_pos_tags: Whether to use POS tag features
            
        Returns:
            Array of predictions (0 for NBC, 1 for Fox)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_text, X_pos = self.prepare_data(texts, use_pos_tags)
        
        if use_pos_tags:
            X = [X_text, X_pos]
        else:
            X = X_text
        
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, texts: List[str], use_pos_tags: bool = True) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: List of texts to classify
            use_pos_tags: Whether to use POS tag features
            
        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_text, X_pos = self.prepare_data(texts, use_pos_tags)
        
        if use_pos_tags:
            X = [X_text, X_pos]
        else:
            X = X_text
        
        return self.model.predict(X).flatten()
    
    def evaluate(self, texts: List[str], labels: List[int], use_pos_tags: bool = True) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            texts: Test texts
            labels: True labels
            use_pos_tags: Whether to use POS tag features
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(texts, use_pos_tags)
        probabilities = self.predict_proba(texts, use_pos_tags)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
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
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model and tokenizers.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model weights
        model_path = filepath.replace('.pkl', '_model.h5')
        self.model.save(model_path)
        
        # Save tokenizers and metadata
        metadata = {
            'tokenizer': self.tokenizer,
            'max_features': self.max_features,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'pos_embedding_dim': self.pos_embedding_dim,
            'dropout_rate': self.dropout_rate,
            'pos_to_id': self.pos_to_id,
            'pos_tags': self.pos_tags
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {model_path} and {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMClassifier':
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded LSTMClassifier instance
        """
        # Load metadata
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create classifier instance
        classifier = cls(
            max_features=metadata['max_features'],
            max_length=metadata['max_length'],
            embedding_dim=metadata['embedding_dim'],
            lstm_units=metadata['lstm_units'],
            pos_embedding_dim=metadata['pos_embedding_dim'],
            dropout_rate=metadata['dropout_rate']
        )
        
        # Restore tokenizer and metadata
        classifier.tokenizer = metadata['tokenizer']
        classifier.pos_to_id = metadata['pos_to_id']
        classifier.pos_tags = metadata['pos_tags']
        
        # Load model weights
        model_path = filepath.replace('.pkl', '_model.h5')
        classifier.model = tf.keras.models.load_model(model_path)
        
        logger.info(f"Model loaded from {model_path} and {filepath}")
        return classifier


def cross_validate_lstm(texts: List[str], 
                       labels: List[int], 
                       use_pos_tags: bool = True,
                       n_folds: int = 5,
                       **model_kwargs) -> Dict:
    """
    Perform cross-validation for LSTM model.
    
    Args:
        texts: List of texts
        labels: List of labels
        use_pos_tags: Whether to use POS tag features
        n_folds: Number of CV folds
        **model_kwargs: Additional model parameters
        
    Returns:
        Cross-validation results
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info(f"Training fold {fold + 1}/{n_folds}")
        
        # Split data
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        # Train model
        classifier = LSTMClassifier(**model_kwargs)
        classifier.train(
            train_texts, train_labels,
            validation_data=(val_texts, val_labels),
            use_pos_tags=use_pos_tags,
            verbose=0
        )
        
        # Evaluate
        results = classifier.evaluate(val_texts, val_labels, use_pos_tags)
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
        "Trump criticizes latest Democratic proposal",
        "Stock market reaches new highs",
        "Local weather updates for the weekend"
    ]
    sample_labels = [0, 1, 0, 1]  # 0 = NBC, 1 = Fox
    
    # Train model
    classifier = LSTMClassifier()
    classifier.train(sample_texts, sample_labels, use_pos_tags=True, epochs=5, verbose=1)
    
    # Make predictions
    test_texts = ["New policy announcement from Washington"]
    predictions = classifier.predict(test_texts)
    probabilities = classifier.predict_proba(test_texts)
    
    print(f"Prediction: {'Fox News' if predictions[0] == 1 else 'NBC'}")
    print(f"Confidence: {probabilities[0]:.3f}")
    
    # Evaluate
    results = classifier.evaluate(sample_texts, sample_labels)
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Macro F1: {results['macro_f1']:.3f}")
