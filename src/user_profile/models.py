"""
Pydantic models for User Profile and Behavioral Analysis.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class AnomalySeverity(str, Enum):
    """Severity level of anomaly."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnomalyType(str, Enum):
    """Type of anomaly detected."""
    VALOR_ANOMALY = "valor_anomaly"
    HORARIO_ANOMALY = "horario_anomaly"
    DESTINO_ANOMALY = "destino_anomaly"
    CANAL_ANOMALY = "canal_anomaly"
    FREQUENCIA_ANOMALY = "frequencia_anomaly"


class ValueStatistics(BaseModel):
    """Statistical measures for transaction values."""
    mean: float
    std: float
    median: float
    p25: float
    p75: float
    p95: float
    min: float
    max: float


class HourStatistics(BaseModel):
    """Statistical measures for transaction hours."""
    mean: float
    std: float
    median: float
    p25: float
    p75: float


class FrequencyStatistics(BaseModel):
    """Statistical measures for transaction frequency."""
    transactions_per_day_mean: float
    transactions_per_day_std: float
    days_active: int


class DestinationInfo(BaseModel):
    """Information about a destination (CPF or bank)."""
    count: int
    last_seen: Optional[datetime] = None


class Destinations(BaseModel):
    """Destination statistics."""
    common_cpfs: Dict[str, DestinationInfo] = Field(default_factory=dict)
    common_bancos: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class CanalInfo(BaseModel):
    """Information about channel usage."""
    count: int
    percentage: float


class Canais(BaseModel):
    """Channel usage statistics."""
    app: CanalInfo
    web: CanalInfo
    api: CanalInfo


class ProdutoInfo(BaseModel):
    """Information about product usage."""
    count: int
    percentage: float


class Produtos(BaseModel):
    """Product usage statistics."""
    pix: ProdutoInfo
    ted: ProdutoInfo
    boleto: ProdutoInfo
    autenticacao: Optional[ProdutoInfo] = None


class Statistics(BaseModel):
    """Overall statistics for user profile."""
    valor: ValueStatistics
    hora: HourStatistics
    frequencia: FrequencyStatistics


class UserProfile(BaseModel):
    """User behavioral profile."""
    cpf: str
    created_at: datetime
    last_updated: datetime
    transaction_count: int
    is_cold_start: bool = False
    statistics: Statistics
    destinations: Destinations = Field(default_factory=Destinations)
    canais: Canais
    produtos: Produtos


class AnomalyDetail(BaseModel):
    """Detail of a specific anomaly detected."""
    type: AnomalyType
    severity: AnomalySeverity
    description: str
    expected_range: Optional[str] = None
    actual_value: Optional[Any] = None
    z_score: Optional[float] = None


class AnomalyResult(BaseModel):
    """Result of anomaly detection."""
    is_anomaly: bool
    anomaly_score: float  # 0.0 to 1.0
    anomalies: List[AnomalyDetail] = Field(default_factory=list)


class BehavioralAnalysis(BaseModel):
    """Behavioral analysis result for API response."""
    has_profile: bool
    transaction_count: int
    is_cold_start: bool
    is_anomaly: bool
    anomaly_score: float
    anomalies: List[AnomalyDetail] = Field(default_factory=list)


class AnomalyFeedback(BaseModel):
    """Feedback from analyst about anomaly."""
    transaction_id: str
    cpf: str
    timestamp: datetime
    is_true_anomaly: bool
    analyst_id: str
    notes: Optional[str] = None
    anomaly_type: Optional[AnomalyType] = None
