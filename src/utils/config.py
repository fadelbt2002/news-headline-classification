"""
Configuration management for the news headline classification project.
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class BaselineConfig:
    """Configuration for baseline TF-IDF + Logistic Regression model."""
    max_features: int = 100
    remove_stopwords: bool = True
    max_iter: int = 1000
    random_state: int = 42


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    max_features: int = 10000
    max_length: int = 50
    embedding_dim: int = 100
    lstm_units: int = 128
    pos_embedding_dim: int = 20
    dropout_rate: float = 0.3


@dataclass
class BERTConfig:
    """Configuration for BERT model."""
    model_name: str = 'bert-base-uncased'
    max_length: int = 128
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01


@dataclass
class GPT2Config:
    """Configuration for GPT-2 model."""
    model_name: str = 'gpt2'
    max_length: int = 38
    learning_rate: float = 3e-5
    weight_decay: float = 0.01


@dataclass
class TrainingConfig:
    """General training configuration."""
    epochs: int = 20
    batch_size: int = 16
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    random_state: int = 42
    use_gpu: bool = True


@dataclass
class PreprocessingConfig:
    """Text preprocessing configuration."""
    lowercase: bool = False
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    lemmatize: bool = False
    use_pos_tags: bool = True
    use_ner_tags: bool = False


@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str = 'data/processed/train.csv'
    test_path: str = 'data/processed/test.csv'
    urls_path: str = 'data/urls.csv'
    raw_data_path: str = 'data/raw/'
    processed_data_path: str = 'data/processed/'


class Config:
    """Main configuration class that holds all sub-configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Initialize with defaults
        self.baseline = BaselineConfig()
        self.lstm = LSTMConfig()
        self.bert = BERTConfig()
        self.gpt2 = GPT2Config()
        self.training = TrainingConfig()
        self.preprocessing = PreprocessingConfig()
        self.data = DataConfig()
        
        # Model saving configuration
        self.model_save_dir = 'results/models'
        self.results_dir = 'results'
        self.plots_dir = 'results/plots'
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations
        if 'baseline' in config_dict:
            self._update_config(self.baseline, config_dict['baseline'])
        
        if 'lstm' in config_dict:
            self._update_config(self.lstm, config_dict['lstm'])
        
        if 'bert' in config_dict:
            self._update_config(self.bert, config_dict['bert'])
        
        if 'gpt2' in config_dict:
            self._update_config(self.gpt2, config_dict['gpt2'])
        
        if 'training' in config_dict:
            self._update_config(self.training, config_dict['training'])
        
        if 'preprocessing' in config_dict:
            self._update_config(self.preprocessing, config_dict['preprocessing'])
        
        if 'data' in config_dict:
            self._update_config(self.data, config_dict['data'])
        
        # Update other configurations
        if 'model_save_dir' in config_dict:
            self.model_save_dir = config_dict['model_save_dir']
        
        if 'results_dir' in config_dict:
            self.results_dir = config_dict['results_dir']
        
        if 'plots_dir' in config_dict:
            self.plots_dir = config_dict['plots_dir']
    
    def _update_config(self, config_obj: Any, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration object with values from dictionary.
        
        Args:
            config_obj: Configuration object to update
            config_dict: Dictionary with new values
        """
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_dict = {
            'baseline': self.baseline.__dict__,
            'lstm': self.lstm.__dict__,
            'bert': self.bert.__dict__,
            'gpt2': self.gpt2.__dict__,
            'training': self.training.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'data': self.data.__dict__,
            'model_save_dir': self.model_save_dir,
            'results_dir': self.results_dir,
            'plots_dir': self.plots_dir
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_model_config(self, model_name: str) -> Any:
        """
        Get configuration for specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration object
        """
        model_configs = {
            'baseline': self.baseline,
            'lstm': self.lstm,
            'bert': self.bert,
            'gpt2': self.gpt2
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")
        
        return model_configs[model_name]
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.model_save_dir,
            self.results_dir,
            self.plots_dir,
            self.data.raw_data_path,
            self.data.processed_data_path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        config_str = "Configuration Summary:\n"
        config_str += "=" * 50 + "\n"
        
        config_str += f"\nBaseline: {self.baseline}\n"
        config_str += f"LSTM: {self.lstm}\n"
        config_str += f"BERT: {self.bert}\n"
        config_str += f"GPT-2: {self.gpt2}\n"
        config_str += f"Training: {self.training}\n"
        config_str += f"Preprocessing: {self.preprocessing}\n"
        config_str += f"Data: {self.data}\n"
        
        config_str += f"\nDirectories:\n"
        config_str += f"  Models: {self.model_save_dir}\n"
        config_str += f"  Results: {self.results_dir}\n"
        config_str += f"  Plots: {self.plots_dir}\n"
        
        return config_str


# Default configuration as YAML string
DEFAULT_CONFIG_YAML = """
baseline:
  max_features: 100
  remove_stopwords: true
  max_iter: 1000
  random_state: 42

lstm:
  max_features: 10000
  max_length: 50
  embedding_dim: 100
  lstm_units: 128
  pos_embedding_dim: 20
  dropout_rate: 0.3

bert:
  model_name: 'bert-base-uncased'
  max_length: 128
  learning_rate: 2e-5
  warmup_steps: 500
  weight_decay: 0.01

gpt2:
  model_name: 'gpt2'
  max_length: 38
  learning_rate: 3e-5
  weight_decay: 0.01

training:
  epochs: 20
  batch_size: 16
  validation_split: 0.2
  early_stopping_patience: 5
  random_state: 42
  use_gpu: true

preprocessing:
  lowercase: false
  remove_punctuation: false
  remove_stopwords: false
  lemmatize: false
  use_pos_tags: true
  use_ner_tags: false

data:
  train_path: 'data/processed/train.csv'
  test_path: 'data/processed/test.csv'
  urls_path: 'data/urls.csv'
  raw_data_path: 'data/raw/'
  processed_data_path: 'data/processed/'

model_save_dir: 'results/models'
results_dir: 'results'
plots_dir: 'results/plots'
"""


def create_default_config(config_path: str) -> None:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path to save the configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(DEFAULT_CONFIG_YAML)


if __name__ == "__main__":
    # Example usage
    config = Config()
    print(config)
    
    # Save default configuration
    create_default_config('config/default_config.yaml')
    print("Default configuration saved to config/default_config.yaml")
