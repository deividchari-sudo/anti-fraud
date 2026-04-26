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
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42, auto_contamination: bool = False):
        """
        Initialize multivariate anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in dataset (ignored if auto_contamination=True)
            random_state: Random seed for reproducibility
            auto_contamination: If True, automatically find optimal contamination
        """
        self.contamination = contamination
        self.random_state = random_state
        self.auto_contamination = auto_contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.optimal_contamination = None
    
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
        
        # Find optimal contamination if auto_contamination is enabled
        if self.auto_contamination:
            self.optimal_contamination = self._find_optimal_contamination(features_scaled)
            # Reinitialize with optimal contamination
            self.contamination = self.optimal_contamination
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100
            )
        
        # Fit Isolation Forest
        self.isolation_forest.fit(features_scaled)
        self.is_fitted = True
    
    def _find_optimal_contamination(self, features: np.ndarray, n_trials: int = 10) -> float:
        """
        Find optimal contamination using score distribution.
        
        Args:
            features: Scaled feature matrix
            n_trials: Number of contamination values to try
            
        Returns:
            Optimal contamination value
        """
        # Train initial model to get scores
        temp_model = IsolationForest(
            contamination=0.1,
            random_state=self.random_state,
            n_estimators=50  # Faster for exploration
        )
        temp_model.fit(features)
        scores = temp_model.score_samples(features)
        
        # Use score distribution to estimate contamination
        # Points with scores below certain threshold are outliers
        # Try different thresholds and find one that gives reasonable number of outliers
        contamination_values = np.linspace(0.01, 0.3, n_trials)
        best_contamination = 0.1
        
        for cont in contamination_values:
            model = IsolationForest(
                contamination=cont,
                random_state=self.random_state,
                n_estimators=50
            )
            model.fit(features)
            predictions = model.predict(features)
            outlier_count = np.sum(predictions == -1)
            outlier_ratio = outlier_count / len(features)
            
            # Prefer contamination that gives 5-15% outliers
            if 0.05 <= outlier_ratio <= 0.15:
                best_contamination = cont
                break
        
        return float(best_contamination)
    
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
        
        result = {
            "is_anomaly": is_anomaly,
            "anomaly_score": normalized_score,
            "method": "isolation_forest",
            "model_fitted": True,
            "contamination": self.contamination
        }
        
        if self.auto_contamination and self.optimal_contamination is not None:
            result["optimal_contamination"] = self.optimal_contamination
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance using permutation importance.
        
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
        
        # Use permutation importance approximation
        # This is a simplified version that uses decision path depth
        # More accurate would require actual permutation importance
        try:
            # Get feature importances from the underlying trees
            importances = []
            for estimator in self.isolation_forest.estimators_:
                # Each estimator is a decision tree
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            
            if importances:
                # Average importance across all trees
                avg_importance = np.mean(importances, axis=0)
                
                # Normalize to sum to 1
                total = np.sum(avg_importance)
                if total > 0:
                    avg_importance = avg_importance / total
                
                importance = {
                    feature_names[i]: float(avg_importance[i])
                    for i in range(len(feature_names))
                }
            else:
                # Fallback to uniform distribution
                importance = {
                    feature_names[i]: 1.0 / len(feature_names)
                    for i in range(len(feature_names))
                }
        except Exception:
            # Fallback to uniform distribution
            importance = {
                feature_names[i]: 1.0 / len(feature_names)
                for i in range(len(feature_names))
            }
        
        return importance


class EnsembleIsolationForest:
    """Ensemble of Isolation Forest models for improved anomaly detection."""
    
    def __init__(self, n_estimators: int = 5, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize ensemble Isolation Forest.
        
        Args:
            n_estimators: Number of Isolation Forest models in ensemble
            contamination: Expected proportion of outliers
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.models = []
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize multiple models with different random states
        for i in range(n_estimators):
            model = IsolationForest(
                contamination=contamination,
                random_state=random_state + i,
                n_estimators=100
            )
            self.models.append(model)
    
    def _extract_features(self, transaction: dict, user_profile: Optional[dict] = None) -> np.ndarray:
        """Extract features from transaction (same as MultivariateAnomalyDetector)."""
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
                features.append(12.0)
                features.append(3.0)
        else:
            features.append(12.0)
            features.append(3.0)
        
        # Channel encoding
        canal = transaction.get("canal", "")
        features.append(1.0 if canal == "app" else 0.0)
        features.append(1.0 if canal == "web" else 0.0)
        features.append(1.0 if canal == "api" else 0.0)
        
        # Product encoding
        produto = transaction.get("produto", "")
        features.append(1.0 if produto == "pix" else 0.0)
        features.append(1.0 if produto == "ted" else 0.0)
        features.append(1.0 if produto == "boleto" else 0.0)
        features.append(1.0 if produto == "autenticacao" else 0.0)
        
        # Sender and receiver banks
        sender = transaction.get("sender", {})
        features.append(float(sender.get("banco", 0)))
        
        receiver = transaction.get("receiver", {})
        features.append(float(receiver.get("banco", 0)))
        
        # Direction
        direcao = transaction.get("direcao", "")
        features.append(1.0 if direcao == "saida" else 0.0)
        
        # User profile context
        if user_profile:
            stats = user_profile.get("statistics", {})
            valor_stats = stats.get("valor", {})
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
        Fit ensemble on transaction data.
        
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
        
        # Fit all models
        for model in self.models:
            model.fit(features_scaled)
        
        self.is_fitted = True
    
    def predict(self, transaction: dict, user_profile: Optional[dict] = None) -> Dict:
        """
        Predict if transaction is anomalous using ensemble voting.
        
        Args:
            transaction: Transaction dictionary
            user_profile: User profile for context (optional)
            
        Returns:
            Dictionary with ensemble prediction
        """
        if not self.is_fitted:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "method": "ensemble_isolation_forest",
                "model_fitted": False
            }
        
        # Extract features
        features = self._extract_features(transaction, user_profile)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from all models
        predictions = []
        scores = []
        
        for model in self.models:
            pred = model.predict(features_scaled)[0]
            score = model.score_samples(features_scaled)[0]
            predictions.append(pred)
            scores.append(score)
        
        # Voting: majority vote for anomaly detection
        anomaly_votes = sum(1 for p in predictions if p == -1)
        is_anomaly = anomaly_votes > (self.n_estimators / 2)
        
        # Average score
        avg_score = np.mean(scores)
        normalized_score = float(-avg_score)
        normalized_score = min(normalized_score, 1.0)
        normalized_score = max(normalized_score, 0.0)
        
        # Vote distribution
        vote_distribution = {
            "anomaly_votes": anomaly_votes,
            "normal_votes": self.n_estimators - anomaly_votes,
            "total_votes": self.n_estimators
        }
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": normalized_score,
            "method": "ensemble_isolation_forest",
            "model_fitted": True,
            "vote_distribution": vote_distribution,
            "contamination": self.contamination
        }
