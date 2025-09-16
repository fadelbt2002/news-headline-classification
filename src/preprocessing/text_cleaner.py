"""
Text preprocessing utilities for news headline classification.
Includes various cleaning and feature extraction methods.
"""

import re
import string
from typing import List, Dict, Tuple, Optional
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

import spacy
from spacy import displacy

# Download required NLTK data
nltk_downloads = [
    'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 
    'maxent_ne_chunker', 'words'
]

for download in nltk_downloads:
    try:
        nltk.data.find(f'tokenizers/{download}')
    except LookupError:
        try:
            nltk.data.find(f'corpora/{download}')
        except LookupError:
            try:
                nltk.data.find(f'taggers/{download}')
            except LookupError:
                try:
                    nltk.data.find(f'chunkers/{download}')
                except LookupError:
                    nltk.download(download)

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing and feature extraction for news headlines."""
    
    def __init__(self, 
                 lowercase: bool = False,
                 remove_punctuation: bool = False,
                 remove_stopwords: bool = False,
                 lemmatize: bool = False,
                 use_pos_tags: bool = True,
                 use_ner_tags: bool = False):
        """
        Initialize text preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            use_pos_tags: Whether to extract POS tags
            use_ner_tags: Whether to extract NER tags
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.use_pos_tags = use_pos_tags
        self.use_ner_tags = use_ner_tags
        
        # Initialize tools
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
        # Try to load spaCy model for NER
        self.nlp = None
        if use_ner_tags:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. NER features disabled.")
                self.use_ner_tags = False
    
    def clean_text(self, text: str) -> str:
        """
        Apply basic text cleaning.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Tokenize text and apply filtering.
        
        Args:
            text: Input text
            
        Returns:
            List of filtered tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def extract_pos_tags(self, text: str) -> List[str]:
        """
        Extract POS tags from text.
        
        Args:
            text: Input text
            
        Returns:
            List of POS tags
        """
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        return [pos for _, pos in pos_tags]
    
    def extract_ner_tags(self, text: str) -> List[str]:
        """
        Extract Named Entity Recognition tags using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of NER tags
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        ner_tags = []
        
        # Get entity labels for each token
        for token in doc:
            if token.ent_iob_ != 'O':
                ner_tags.append(f"{token.ent_iob_}-{token.ent_type_}")
            else:
                ner_tags.append('O')
        
        return ner_tags
    
    def extract_ner_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities as structured data.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def get_stylistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract stylistic features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of stylistic features
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'length': 0, 'word_count': 0, 'avg_word_length': 0,
                'punctuation_ratio': 0, 'uppercase_ratio': 0, 'digit_ratio': 0
            }
        
        # Basic counts
        length = len(text)
        words = text.split()
        word_count = len(words)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        # Character type ratios
        punctuation_count = sum(1 for char in text if char in string.punctuation)
        uppercase_count = sum(1 for char in text if char.isupper())
        digit_count = sum(1 for char in text if char.isdigit())
        
        punctuation_ratio = punctuation_count / length if length > 0 else 0
        uppercase_ratio = uppercase_count / length if length > 0 else 0
        digit_ratio = digit_count / length if length > 0 else 0
        
        return {
            'length': length,
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'punctuation_ratio': punctuation_ratio,
            'uppercase_ratio': uppercase_ratio,
            'digit_ratio': digit_ratio
        }
    
    def process_single_text(self, text: str) -> Dict[str, any]:
        """
        Process a single text and extract all features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with processed text and features
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and filter
        tokens = self.tokenize_and_filter(cleaned_text)
        processed_text = ' '.join(tokens)
        
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'processed_text': processed_text,
            'tokens': tokens
        }
        
        # Add POS tags
        if self.use_pos_tags:
            pos_tags = self.extract_pos_tags(text)  # Use original text for POS
            result['pos_tags'] = pos_tags
        
        # Add NER tags
        if self.use_ner_tags:
            ner_tags = self.extract_ner_tags(text)
            ner_entities = self.extract_ner_entities(text)
            result['ner_tags'] = ner_tags
            result['ner_entities'] = ner_entities
        
        # Add stylistic features
        stylistic_features = self.get_stylistic_features(text)
        result['stylistic_features'] = stylistic_features
        
        return result
    
    def process_texts(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Process multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of processed text dictionaries
        """
        return [self.process_single_text(text) for text in texts]
    
    def transform(self, texts: List[str]) -> List[str]:
        """
        Transform texts to processed format (for compatibility with sklearn).
        
        Args:
            texts: List of input texts
            
        Returns:
            List of processed texts
        """
        processed = self.process_texts(texts)
        return [item['processed_text'] for item in processed]
    
    def get_pos_sequences(self, texts: List[str]) -> List[List[str]]:
        """
        Get POS tag sequences for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of POS tag sequences
        """
        return [self.extract_pos_tags(text) for text in texts]
    
    def get_ner_sequences(self, texts: List[str]) -> List[List[str]]:
        """
        Get NER tag sequences for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of NER tag sequences
        """
        return [self.extract_ner_tags(text) for text in texts]
    
    def analyze_preprocessing_impact(self, texts: List[str]) -> Dict[str, any]:
        """
        Analyze the impact of different preprocessing steps.
        
        Args:
            texts: List of input texts
            
        Returns:
            Analysis of preprocessing impact
        """
        # Original texts
        original_stats = self._calculate_text_stats(texts)
        
        # Apply different preprocessing steps
        preprocessors = {
            'lowercase': TextPreprocessor(lowercase=True),
            'remove_punctuation': TextPreprocessor(remove_punctuation=True),
            'remove_stopwords': TextPreprocessor(remove_stopwords=True),
            'lemmatize': TextPreprocessor(lemmatize=True),
            'all_cleaning': TextPreprocessor(
                lowercase=True, remove_punctuation=True, 
                remove_stopwords=True, lemmatize=True
            )
        }
        
        analysis = {'original': original_stats}
        
        for name, preprocessor in preprocessors.items():
            processed_texts = preprocessor.transform(texts)
            stats = self._calculate_text_stats(processed_texts)
            analysis[name] = stats
        
        return analysis
    
    def _calculate_text_stats(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate basic statistics for a list of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary with text statistics
        """
        if not texts:
            return {}
        
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        return {
            'avg_length': sum(lengths) / len(lengths),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'total_texts': len(texts),
            'total_chars': sum(lengths),
            'total_words': sum(word_counts)
        }


def compare_preprocessing_strategies(texts: List[str], labels: List[int]) -> Dict:
    """
    Compare different preprocessing strategies on the given dataset.
    
    Args:
        texts: List of texts
        labels: List of labels
        
    Returns:
        Comparison results
    """
    strategies = {
        'raw': TextPreprocessor(),
        'minimal': TextPreprocessor(lowercase=True),
        'standard': TextPreprocessor(
            lowercase=True, remove_punctuation=True, remove_stopwords=True
        ),
        'aggressive': TextPreprocessor(
            lowercase=True, remove_punctuation=True, 
            remove_stopwords=True, lemmatize=True
        )
    }
    
    results = {}
    
    for name, preprocessor in strategies.items():
        logger.info(f"Analyzing preprocessing strategy: {name}")
        
        # Process texts
        processed_texts = preprocessor.transform(texts)
        
        # Calculate statistics
        stats = preprocessor._calculate_text_stats(processed_texts)
        
        # Separate by class
        nbc_texts = [processed_texts[i] for i, label in enumerate(labels) if label == 0]
        fox_texts = [processed_texts[i] for i, label in enumerate(labels) if label == 1]
        
        nbc_stats = preprocessor._calculate_text_stats(nbc_texts)
        fox_stats = preprocessor._calculate_text_stats(fox_texts)
        
        results[name] = {
            'overall': stats,
            'nbc': nbc_stats,
            'fox': fox_stats,
            'sample_texts': processed_texts[:3]  # Show first 3 examples
        }
    
    return results


# Example usage
if __name__ == "__main__":
    # Example texts
    sample_texts = [
        "Biden Announces New Climate Initiative for Renewable Energy!",
        "Trump Criticizes Latest Democratic Proposal in Fiery Speech",
        "Stock Market Reaches Record Highs Amid Economic Recovery",
        "Breaking: Investigation Reveals New Evidence in Political Scandal"
    ]
    
    # Test different preprocessing strategies
    print("Text Preprocessing Examples")
    print("=" * 50)
    
    strategies = {
        'Raw': TextPreprocessor(),
        'Lowercase': TextPreprocessor(lowercase=True),
        'No Punctuation': TextPreprocessor(remove_punctuation=True),
        'No Stopwords': TextPreprocessor(remove_stopwords=True),
        'Full Cleaning': TextPreprocessor(
            lowercase=True, remove_punctuation=True, 
            remove_stopwords=True, lemmatize=True
        )
    }
    
    for strategy_name, preprocessor in strategies.items():
        print(f"\n{strategy_name}:")
        processed = preprocessor.transform(sample_texts)
        for i, (orig, proc) in enumerate(zip(sample_texts, processed)):
            print(f"  {i+1}. Original: {orig}")
            print(f"     Processed: {proc}")
            break  # Just show first example
    
    # Feature extraction example
    print(f"\n\nFeature Extraction Example")
    print("=" * 50)
    
    feature_extractor = TextPreprocessor(use_pos_tags=True, use_ner_tags=True)
    result = feature_extractor.process_single_text(sample_texts[0])
    
    print(f"Text: {result['original_text']}")
    print(f"Tokens: {result['tokens']}")
    print(f"POS Tags: {result['pos_tags']}")
    if result.get('ner_entities'):
        print(f"Named Entities: {result['ner_entities']}")
    print(f"Stylistic Features: {result['stylistic_features']}")
