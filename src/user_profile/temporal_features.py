"""
Temporal feature engineering for behavioral profiling.
Implements sliding window features and trend analysis.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
from collections import deque


class TemporalFeatureExtractor:
    """Extracts temporal features from transaction history."""
    
    def __init__(self, max_window_days: int = 90):
        """
        Initialize temporal feature extractor.
        
        Args:
            max_window_days: Maximum window size in days for feature calculation
        """
        self.max_window_days = max_window_days
    
    def extract_sliding_window_features(self, transactions: List[dict]) -> Dict:
        """
        Extract features from sliding windows (7 days, 30 days, 90 days).
        
        Args:
            transactions: List of transaction dictionaries with timestamp
            
        Returns:
            Dictionary with sliding window features
        """
        if not transactions:
            return self._empty_sliding_features()
        
        # Parse timestamps and sort
        parsed_transactions = []
        for t in transactions:
            try:
                timestamp = t.get("timestamp", "")
                if timestamp:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    parsed_transactions.append({
                        "datetime": dt,
                        "valor": float(t.get("valor", 0))
                    })
            except (ValueError, TypeError):
                continue
        
        if not parsed_transactions:
            return self._empty_sliding_features()
        
        parsed_transactions.sort(key=lambda x: x["datetime"])
        
        # Get latest transaction time as reference
        latest_time = parsed_transactions[-1]["datetime"]
        
        # Extract features for different windows
        features = {}
        for window_days in [7, 30, 90]:
            window_start = latest_time - timedelta(days=window_days)
            window_txns = [t for t in parsed_transactions if t["datetime"] >= window_start]
            
            if window_txns:
                values = [t["valor"] for t in window_txns if t["valor"] > 0]
                
                features[f"window_{window_days}_days"] = {
                    "transaction_count": len(window_txns),
                    "total_amount": sum(values) if values else 0,
                    "avg_amount": np.mean(values) if values else 0,
                    "max_amount": max(values) if values else 0,
                    "min_amount": min(values) if values else 0,
                    "std_amount": np.std(values) if len(values) > 1 else 0
                }
            else:
                features[f"window_{window_days}_days"] = {
                    "transaction_count": 0,
                    "total_amount": 0,
                    "avg_amount": 0,
                    "max_amount": 0,
                    "min_amount": 0,
                    "std_amount": 0
                }
        
        return features
    
    def extract_trend_features(self, transactions: List[dict]) -> Dict:
        """
        Extract trend features (increasing/decreasing spending patterns).
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dictionary with trend features
        """
        if not transactions:
            return self._empty_trend_features()
        
        # Parse and sort transactions
        parsed_transactions = []
        for t in transactions:
            try:
                timestamp = t.get("timestamp", "")
                if timestamp:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    parsed_transactions.append({
                        "datetime": dt,
                        "valor": float(t.get("valor", 0))
                    })
            except (ValueError, TypeError):
                continue
        
        if len(parsed_transactions) < 2:
            return self._empty_trend_features()
        
        parsed_transactions.sort(key=lambda x: x["datetime"])
        
        # Calculate trend over last 10 transactions
        recent_txns = parsed_transactions[-10:]
        values = [t["valor"] for t in recent_txns if t["valor"] > 0]
        
        if len(values) < 2:
            return self._empty_trend_features()
        
        # Linear trend (slope)
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend direction
        if slope > 10:
            trend = "increasing"
        elif slope < -10:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Compare recent vs older transactions
        recent_avg = np.mean(values[-5:]) if len(values) >= 5 else np.mean(values)
        older_avg = np.mean(values[:-5]) if len(values) >= 5 else recent_avg
        
        change_percent = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        
        return {
            "trend_direction": trend,
            "trend_slope": float(slope),
            "recent_avg_amount": float(recent_avg),
            "older_avg_amount": float(older_avg),
            "change_percent": float(change_percent)
        }
    
    def extract_seasonal_features(self, transactions: List[dict]) -> Dict:
        """
        Extract seasonal features (day of week, hour of day patterns).
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dictionary with seasonal features
        """
        if not transactions:
            return self._empty_seasonal_features()
        
        # Parse transactions
        days_of_week = []
        hours_of_day = []
        values = []
        
        for t in transactions:
            try:
                timestamp = t.get("timestamp", "")
                if timestamp:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    days_of_week.append(dt.weekday())  # 0 = Monday, 6 = Sunday
                    hours_of_day.append(dt.hour)
                    values.append(float(t.get("valor", 0)))
            except (ValueError, TypeError):
                continue
        
        if not days_of_week:
            return self._empty_seasonal_features()
        
        # Day of week distribution
        day_counts = np.bincount(days_of_week, minlength=7)
        day_distribution = [float(c / len(days_of_week)) for c in day_counts]
        
        # Hour of day distribution
        hour_counts = np.bincount(hours_of_day, minlength=24)
        hour_distribution = [float(c / len(hours_of_day)) for c in hour_counts]
        
        # Most active day and hour
        most_active_day = int(np.argmax(day_counts))
        most_active_hour = int(np.argmax(hour_counts))
        
        return {
            "day_of_week_distribution": day_distribution,
            "hour_of_day_distribution": hour_distribution,
            "most_active_day": most_active_day,
            "most_active_hour": most_active_hour,
            "avg_value_per_day": float(np.mean(values)) if values else 0
        }
    
    def extract_all_temporal_features(self, transactions: List[dict]) -> Dict:
        """
        Extract all temporal features.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dictionary with all temporal features
        """
        return {
            "sliding_windows": self.extract_sliding_window_features(transactions),
            "trends": self.extract_trend_features(transactions),
            "seasonal": self.extract_seasonal_features(transactions)
        }
    
    def _empty_sliding_features(self) -> Dict:
        """Return empty sliding window features."""
        return {
            f"window_{days}_days": {
                "transaction_count": 0,
                "total_amount": 0,
                "avg_amount": 0,
                "max_amount": 0,
                "min_amount": 0,
                "std_amount": 0
            } for days in [7, 30, 90]
        }
    
    def _empty_trend_features(self) -> Dict:
        """Return empty trend features."""
        return {
            "trend_direction": "stable",
            "trend_slope": 0.0,
            "recent_avg_amount": 0.0,
            "older_avg_amount": 0.0,
            "change_percent": 0.0
        }
    
    def _empty_seasonal_features(self) -> Dict:
        """Return empty seasonal features."""
        return {
            "day_of_week_distribution": [0.0] * 7,
            "hour_of_day_distribution": [0.0] * 24,
            "most_active_day": 0,
            "most_active_hour": 0,
            "avg_value_per_day": 0.0
        }
