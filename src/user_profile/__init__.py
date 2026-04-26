"""
User Profile Module for Behavioral Profiling.
"""

from .models import (
    UserProfile,
    AnomalyResult,
    AnomalyDetail,
    BehavioralAnalysis,
    AnomalyFeedback,
    AnomalyType,
    AnomalySeverity,
    Statistics,
    ValueStatistics,
    HourStatistics,
    FrequencyStatistics,
    Destinations,
    Canais,
    Produtos
)
from .service import UserProfileService
from .anomaly_detector import AnomalyDetector
from ..repositories import UserProfileRepository
from .clustering import UserClusterer
from .isolation_forest import MultivariateAnomalyDetector, EnsembleIsolationForest
from .online_learning import AdaptiveUserProfile
from .graph_features import GraphFeatureExtractor

__all__ = [
    "UserProfile",
    "AnomalyResult",
    "AnomalyDetail",
    "BehavioralAnalysis",
    "AnomalyFeedback",
    "AnomalyType",
    "AnomalySeverity",
    "Statistics",
    "ValueStatistics",
    "HourStatistics",
    "FrequencyStatistics",
    "Destinations",
    "Canais",
    "Produtos",
    "UserProfileService",
    "UserProfileRepository",
    "AnomalyDetector",
    "UserClusterer",
    "MultivariateAnomalyDetector",
    "EnsembleIsolationForest",
    "AdaptiveUserProfile",
    "GraphFeatureExtractor"
]
