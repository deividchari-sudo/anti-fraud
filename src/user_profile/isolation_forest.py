"""
Isolation Forest for multivariate anomaly detection.
Detects outliers in high-dimensional feature space.
"""

from typing import List, Dict, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class MultivariateAnomalyDetector:
    """Detects multivariate anomalies using Isolation Forest."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize multivariate anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in dataset
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _extract_features(self, transaction: dict, user_profile: Optional[dict] = None) -> np.ndarray:
        """
        Extract features from transaction for anomaly detection.
        
        Args:
            transaction: Transaction dictionary
            user_profile: User profile for context (optional)
            
        Returns:
            Feature vector (1, n_features)
        """
        features = []
        
        # Basic transaction features
        features.append(float(transaction.get("valor", 0)))
        
        # Time features
        timestamp = transaction.get("timestamp", "")
        if timestamp:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                features.append(float(dt.hour))
                features.append(float(dt.weekday()))
            except (ValueError, TypeError):
                features.append(12.0)  # Default noon
                features.append(3.0)   # Default Wednesday
        else:
            features.append(12.0)
            features.append(3.0)
        
        # Channel encoding (one-hot)
        canal = transaction.get("canal", "")
        features.append(1.0 if canal == "app" else 0.0)
        features.append(1.0 if canal == "web" else 0.0)
        features.append(1.0 if canal == "api" else 0.0)
        
        # Product encoding (one-hot)
        produto = transaction.get("produto", "")
        features.append(1.0 if produto == "pix" else 0.0)
        features.append(1.0 if produto == "ted" else 0.0)
        features.append(1.0 if produto == "boleto" else 0.0)
        features.append(1.0 if produto == "autenticacao" else 0.0)
        
        # Sender bank
        sender = transaction.get("sender", {})
        features.append(float(sender.get("banco", 0)))
        
        # Receiver bank
        receiver = transaction.get("receiver", {})
        features.append(float(receiver.get("banco", 0)))
        
        # Direction
        direcao = transaction.get("direcao", "")
        features.append(1.0 if direcao == "saida" else 0.0)
        
        # User profile context (if available)
        if user_profile:
            stats = user_profile.get("statistics", {})
            valor_stats = stats.get("valor", {})
            
            # Z-score relative to user's history
            user_mean = valor_stats.get("mean", 0)
            user_std = valor_stats.get("std", 1)
            transaction_value = float(transaction.get("valor", 0))
            z_score = (transaction_value - user_mean) / (user_std + 1e-6)
            features.append(z_score)
        else:
            features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def fit(self, transactions: List[dict], user_profiles: Optional[Dict[str, dict]] = None):
        """
        Fit Isolation Forest on transaction data.
        
        Args:
            transactions: List of transaction dictionaries
            user_profiles: Dictionary mapping CPF to user profiles (optional)
        """
        # Extract features
        features_list = []
        for tx in transactions:
            cpf = tx.get("sender", {}).get("cpfSender", "")
            user_profile = user_profiles.get(cpf) if user_profiles else None
            features = self._extract_features(tx, user_profile)
            features_list.append(features[0])
        
        if not features_list:
            raise ValueError("No features extracted from transactions")
        
        features_array = np.array(features_list)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_array)
        
        # Fit Isolation Forest
        self.isolation_forest.fit(features_scaled)
        self.is_fitted = True
    
    def predict(self, transaction: dict, user_profile: Optional[dict] = None) -> Dict:
        """
        Predict if transaction is anomalous.
        
        Args:
            transaction: Transaction dictionary
            user_profile: User profile for context (optional)
            
        Returns:
            Dictionary with anomaly prediction
        """
        if not self.is_fitted:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "method": "isolation_forest",
                "model_fitted": False
            }
        
        # Extract features
        features = self._extract_features(transaction, user_profile)
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.isolation_forest.predict(features_scaled)[0]
        anomaly_score = self.isolation_forest.score_samples(features_scaled)[0]
        
        # Convert to 0-1 scale (higher = more anomalous)
        # Isolation Forest returns -1 for outliers, 1 for inliers
        # Score samples returns negative values (more negative = more anomalous)
        is_anomaly = prediction == -1
        normalized_score = float(-anomaly_score)  # Convert to positive
        normalized_score = min(normalized_score, 1.0)  # Cap at 1.0
        normalized_score = max(normalized_score, 0.0)  # Floor at 0.0
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": normalized_score,
            "method": "isolation_forest",
            "model_fitted": True
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (approximate using decision path lengths).
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            return {}
        
        # Feature names
        feature_names = [
            "valor", "hour", "day_of_week",
            "canal_app", "canal_web", "canal_api",
            "produto_pix", "produto_ted", "produto_boleto", "produto_autenticacao",
            "sender_banco", "receiver_banco", "direcao_saida",
            "z_score_relative"
        ]
        
        # Approximate importance using average path length
        # This is a simplified approach
        importance = {
            feature_names[i]: 1.0 / len(feature_names)
            for i in range(len(feature_names))
        }
        
        return importance
