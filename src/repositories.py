"""
Repository Pattern for data access.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class TransactionRepository(ABC):
    """Abstract repository for transaction data access."""

    @abstractmethod
    def load_dataset(self) -> pd.DataFrame:
        """Load the transaction dataset."""
        pass

    @abstractmethod
    def save_dataset(self, df: pd.DataFrame) -> None:
        """Save the transaction dataset."""
        pass


class CSVTransactionRepository(TransactionRepository):
    """CSV-based implementation of transaction repository."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_dataset(self) -> pd.DataFrame:
        """Load the transaction dataset from CSV."""
        return pd.read_csv(self.dataset_path)

    def save_dataset(self, df: pd.DataFrame) -> None:
        """Save the transaction dataset to CSV."""
        df.to_csv(self.dataset_path, index=False)


class ModelRepository(ABC):
    """Abstract repository for model persistence."""

    @abstractmethod
    def load_model(self):
        """Load the trained model."""
        pass

    @abstractmethod
    def save_model(self, model) -> None:
        """Save the trained model."""
        pass

    @abstractmethod
    def load_feature_names(self) -> list:
        """Load feature names."""
        pass

    @abstractmethod
    def save_feature_names(self, feature_names: list) -> None:
        """Save feature names."""
        pass


class JoblibModelRepository(ModelRepository):
    """Joblib-based implementation of model repository."""

    def __init__(self, model_path: str, feature_names_path: str):
        self.model_path = model_path
        self.feature_names_path = feature_names_path

    def load_model(self):
        """Load the trained model from disk."""
        import joblib

        return joblib.load(self.model_path)

    def save_model(self, model) -> None:
        """Save the trained model to disk."""
        import joblib

        Path(self.model_path).parent.mkdir(exist_ok=True)
        joblib.dump(model, self.model_path)

    def load_feature_names(self) -> list:
        """Load feature names from JSON."""
        if Path(self.feature_names_path).exists():
            with open(self.feature_names_path, "r") as f:
                return json.load(f)
        return []

    def save_feature_names(self, feature_names: list) -> None:
        """Save feature names to JSON."""
        Path(self.feature_names_path).parent.mkdir(exist_ok=True)
        with open(self.feature_names_path, "w") as f:
            json.dump(feature_names, f)
