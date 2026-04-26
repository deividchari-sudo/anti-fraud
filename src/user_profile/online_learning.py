"""
Online learning for adaptive anomaly detection.
Implements incremental learning with forgetting factor for concept drift.
"""

from typing import List, Dict, Optional
import numpy as np
from collections import deque


class OnlineAnomalyThreshold:
    """Adaptive threshold that learns from feedback."""
    
    def __init__(self, initial_threshold: float = 0.5, 
                 learning_rate: float = 0.1,
                 window_size: int = 100):
        """
        Initialize online anomaly threshold.
        
        Args:
            initial_threshold: Initial threshold value (0.0 to 1.0)
            learning_rate: Rate of adaptation (0.0 to 1.0)
            window_size: Number of recent samples to consider
        """
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.recent_scores = deque(maxlen=window_size)
        self.recent_labels = deque(maxlen=window_size)
    
    def update(self, anomaly_score: float, is_true_anomaly: bool):
        """
        Update threshold based on feedback.
        
        Args:
            anomaly_score: Predicted anomaly score
            is_true_anomaly: True if this was actually an anomaly
        """
        self.recent_scores.append(anomaly_score)
        self.recent_labels.append(is_true_anomaly)
        
        # Calculate optimal threshold based on recent data
        if len(self.recent_scores) >= 10:
            # Find threshold that maximizes accuracy on recent data
            best_threshold = self._find_optimal_threshold()
            
            # Adapt towards optimal threshold
            self.threshold = self.threshold + self.learning_rate * (best_threshold - self.threshold)
            self.threshold = max(0.0, min(1.0, self.threshold))
    
    def _find_optimal_threshold(self) -> float:
        """Find optimal threshold based on recent data."""
        scores = np.array(list(self.recent_scores))
        labels = np.array(list(self.recent_labels))
        
        best_threshold = self.threshold
        best_accuracy = 0.0
        
        # Try different thresholds
        for threshold in np.linspace(0.0, 1.0, 20):
            predictions = scores >= threshold
            accuracy = np.mean(predictions == labels)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold
    
    def predict(self, anomaly_score: float) -> bool:
        """
        Predict if score is anomalous based on current threshold.
        
        Args:
            anomaly_score: Anomaly score (0.0 to 1.0)
            
        Returns:
            True if anomalous
        """
        return anomaly_score >= self.threshold


class ConceptDriftDetector:
    """Detects concept drift in user behavior patterns."""
    
    def __init__(self, window_size: int = 50, drift_threshold: float = 0.3):
        """
        Initialize concept drift detector.
        
        Args:
            window_size: Size of sliding window for comparison
            drift_threshold: Threshold for detecting drift
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
    
    def add_sample(self, feature_value: float):
        """
        Add a sample for drift detection.
        
        Args:
            feature_value: Feature value to monitor
        """
        self.current_window.append(feature_value)
    
    def detect_drift(self) -> bool:
        """
        Detect if concept drift has occurred.
        
        Returns:
            True if drift detected
        """
        if len(self.current_window) < self.window_size:
            return False
        
        # Calculate statistics
        ref_mean = np.mean(self.reference_window) if self.reference_window else 0
        ref_std = np.std(self.reference_window) if self.reference_window else 1
        
        curr_mean = np.mean(self.current_window)
        curr_std = np.std(self.current_window)
        
        # Calculate drift score
        mean_diff = abs(curr_mean - ref_mean) / (ref_std + 1e-6)
        std_diff = abs(curr_std - ref_std) / (ref_std + 1e-6)
        
        drift_score = (mean_diff + std_diff) / 2
        
        # Update reference if no drift
        if drift_score < self.drift_threshold:
            self.reference_window.extend(self.current_window)
            self.current_window = deque(maxlen=self.window_size)
        
        return drift_score >= self.drift_threshold
    
    def reset(self):
        """Reset the detector."""
        self.reference_window = deque(maxlen=self.window_size)
        self.current_window = deque(maxlen=self.window_size)


class AdaptiveUserProfile:
    """Adaptive user profile with online learning."""
    
    def __init__(self, cpf: str, forgetting_factor: float = 0.95):
        """
        Initialize adaptive user profile.
        
        Args:
            cpf: User CPF
            forgetting_factor: Weight for old data (0.0 to 1.0)
        """
        self.cpf = cpf
        self.forgetting_factor = forgetting_factor
        
        # Exponential moving averages
        self.ema_value = 0.0
        self.ema_hour = 12.0
        self.ema_std = 0.0
        
        # Counts for exponential moving average
        self.count = 0
        
        # Drift detector
        self.drift_detector = ConceptDriftDetector()
        
        # Adaptive threshold
        self.adaptive_threshold = OnlineAnomalyThreshold()
    
    def update(self, transaction: dict, anomaly_score: float, is_true_anomaly: Optional[bool] = None):
        """
        Update profile with new transaction.
        
        Args:
            transaction: Transaction dictionary
            anomaly_score: Anomaly score for this transaction
            is_true_anomaly: True if this was actually an anomaly (feedback)
        """
        # Extract features
        value = float(transaction.get("valor", 0))
        
        # Parse timestamp
        timestamp = transaction.get("timestamp", "")
        if timestamp:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                hour = float(dt.hour)
            except (ValueError, TypeError):
                hour = 12.0
        else:
            hour = 12.0
        
        # Update exponential moving averages
        self.count += 1
        alpha = 1.0 / (self.count + 1)  # Adaptive learning rate
        
        self.ema_value = self.forgetting_factor * self.ema_value + (1 - self.forgetting_factor) * value
        self.ema_hour = self.forgetting_factor * self.ema_hour + (1 - self.forgetting_factor) * hour
        
        # Update running std
        delta = value - self.ema_value
        self.ema_std = self.forgetting_factor * self.ema_std + (1 - self.forgetting_factor) * abs(delta)
        
        # Add sample to drift detector
        self.drift_detector.add_sample(value)
        
        # Update adaptive threshold if feedback provided
        if is_true_anomaly is not None:
            self.adaptive_threshold.update(anomaly_score, is_true_anomaly)
    
    def detect_anomaly(self, transaction: dict) -> Dict:
        """
        Detect if transaction is anomalous based on adaptive profile.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Dictionary with anomaly detection result
        """
        # Extract features
        value = float(transaction.get("valor", 0))
        
        # Calculate z-score based on EMA
        z_score = (value - self.ema_value) / (self.ema_std + 1e-6)
        
        # Check for concept drift
        has_drift = self.drift_detector.detect_drift()
        
        # If drift detected, reset profile
        if has_drift:
            self.drift_detector.reset()
            self.ema_value = value
            self.ema_std = 0.0
        
        # Anomaly score based on z-score
        anomaly_score = min(abs(z_score) / 3.0, 1.0)  # Cap at 1.0
        
        # Apply adaptive threshold
        is_anomaly = self.adaptive_threshold.predict(anomaly_score)
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "z_score": z_score,
            "ema_mean": self.ema_value,
            "ema_std": self.ema_std,
            "has_drift": has_drift,
            "adaptive_threshold": self.adaptive_threshold.threshold,
            "method": "adaptive_online_learning"
        }
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            "cpf": self.cpf,
            "count": self.count,
            "ema_value": self.ema_value,
            "ema_hour": self.ema_hour,
            "ema_std": self.ema_std,
            "adaptive_threshold": self.adaptive_threshold.threshold
        }
