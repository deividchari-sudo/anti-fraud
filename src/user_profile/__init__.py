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
    "AnomalyDetector"
]
