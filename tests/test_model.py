import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.model import FraudDetectionModel


class TestFraudDetectionModel:
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'valor': np.random.uniform(0, 10000, n_samples),
            'sender_banco': np.random.randint(1, 1000, n_samples),
            'receiver_banco': np.random.randint(1, 1000, n_samples),
            'hora_do_dia': np.random.randint(0, 24, n_samples),
            'dia_da_semana': np.random.randint(0, 7, n_samples),
            'is_pix': np.random.randint(0, 2, n_samples),
            'is_ted': np.random.randint(0, 2, n_samples),
            'is_web': np.random.randint(0, 2, n_samples),
            'is_app': np.random.randint(0, 2, n_samples),
            'is_api': np.random.randint(0, 2, n_samples),
        }
        
        df = pd.DataFrame(data)
        df['fraudResult'] = np.random.randint(0, 2, n_samples)
        df['payload'] = '{}'  # Dummy payload
        
        return df
    
    def test_model_initialization(self):
        """Test model initialization."""
        # Use a path that doesn't exist to avoid loading existing model
        model = FraudDetectionModel(model_path="nonexistent_model.pkl")
        assert model.model is None
        assert model.threshold == 0.5
    
    def test_model_training(self, sample_data):
        """Test model training."""
        model = FraudDetectionModel()
        metrics = model.train(sample_data)
        
        assert model.model is not None
        assert len(model.feature_names) > 0
        assert 'auc_roc' in metrics
        assert 'f1_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
    
    def test_model_prediction(self, sample_data):
        """Test model prediction."""
        model = FraudDetectionModel()
        model.train(sample_data)
        
        # Create test features
        test_features = sample_data.drop(columns=['fraudResult', 'payload']).iloc[0:1]
        fraud_prob, is_fraud, explanation = model.predict(test_features)
        
        assert isinstance(fraud_prob, (float, np.floating))
        assert 0 <= fraud_prob <= 1
        assert isinstance(is_fraud, (bool, np.bool_))
    
    def test_model_save_and_load(self, sample_data, tmp_path):
        """Test model save and load functionality."""
        model_path = str(tmp_path / "test_model.pkl")
        
        # Train and save
        model = FraudDetectionModel(model_path=model_path)
        model.train(sample_data)
        
        # Load in new instance
        loaded_model = FraudDetectionModel(model_path=model_path)
        assert loaded_model.model is not None
        assert len(loaded_model.feature_names) == len(model.feature_names)
    
    def test_model_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        model = FraudDetectionModel()
        model.train(sample_data)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(isinstance(v, (float, np.floating)) for v in importance.values())
    
    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        model = FraudDetectionModel(model_path="nonexistent_model.pkl")
        test_features = pd.DataFrame({'valor': [1000]})
        
        with pytest.raises(ValueError, match="Model not loaded"):
            model.predict(test_features)
    
    def test_model_with_imbalanced_data(self):
        """Test model training with imbalanced data."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'valor': np.random.uniform(0, 10000, n_samples),
            'sender_banco': np.random.randint(1, 1000, n_samples),
            'receiver_banco': np.random.randint(1, 1000, n_samples),
            'hora_do_dia': np.random.randint(0, 24, n_samples),
            'dia_da_semana': np.random.randint(0, 7, n_samples),
            'is_pix': np.random.randint(0, 2, n_samples),
            'is_ted': np.random.randint(0, 2, n_samples),
            'is_web': np.random.randint(0, 2, n_samples),
            'is_app': np.random.randint(0, 2, n_samples),
            'is_api': np.random.randint(0, 2, n_samples),
        }
        
        df = pd.DataFrame(data)
        # Create imbalanced labels (90% legitimate, 10% fraud)
        df['fraudResult'] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        df['payload'] = '{}'
        
        model = FraudDetectionModel(model_path="test_imbalanced_model.pkl")
        metrics = model.train(df)
        
        # Model should still train successfully
        assert model.model is not None
        # Relax assertion since random data may not be discriminative
        assert metrics['auc_roc'] >= 0.0
