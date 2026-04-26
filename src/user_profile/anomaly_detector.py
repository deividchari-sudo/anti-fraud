"""
Anomaly Detection Algorithm for Behavioral Profiling.
Uses z-score and percentile-based methods to detect anomalies.
"""

from datetime import datetime
from typing import Optional
from .models import (
    UserProfile,
    AnomalyResult,
    AnomalyDetail,
    AnomalyType,
    AnomalySeverity
)


class AnomalyDetector:
    """Detects anomalies in transactions based on user behavioral profile."""
    
    # Thresholds for anomaly detection
    Z_SCORE_THRESHOLD_HIGH = 3.0
    Z_SCORE_THRESHOLD_MEDIUM = 2.0
    Z_SCORE_THRESHOLD_CRITICAL = 5.0
    
    HOUR_DEVIATION_THRESHOLD = 2.0  # hours
    FREQUENCY_MULTIPLIER_THRESHOLD = 3.0  # times normal frequency
    
    def __init__(self, use_global_profile_fallback: bool = True):
        """
        Initialize anomaly detector.
        
        Args:
            use_global_profile_fallback: If True, use global profile for cold start
        """
        self.use_global_profile_fallback = use_global_profile_fallback
    
    def detect(self, transaction: dict, profile: UserProfile) -> AnomalyResult:
        """
        Detect anomalies in a transaction based on user profile.
        
        Args:
            transaction: Transaction dictionary
            profile: User behavioral profile
            
        Returns:
            AnomalyResult with detected anomalies
        """
        anomalies = []
        
        # Skip detection if cold start and no fallback
        if profile.is_cold_start and not self.use_global_profile_fallback:
            return AnomalyResult(is_anomaly=False, anomaly_score=0.0, anomalies=[])
        
        # 1. Detect value anomaly
        valor_anomaly = self._detect_valor_anomaly(transaction, profile)
        if valor_anomaly:
            anomalies.append(valor_anomaly)
        
        # 2. Detect hour anomaly
        horario_anomaly = self._detect_horario_anomaly(transaction, profile)
        if horario_anomaly:
            anomalies.append(horario_anomaly)
        
        # 3. Detect destination anomaly
        destino_anomaly = self._detect_destino_anomaly(transaction, profile)
        if destino_anomaly:
            anomalies.append(destino_anomaly)
        
        # 4. Detect channel anomaly
        canal_anomaly = self._detect_canal_anomaly(transaction, profile)
        if canal_anomaly:
            anomalies.append(canal_anomaly)
        
        # Calculate anomaly score (0.0 to 1.0)
        anomaly_score = self._calculate_anomaly_score(anomalies)
        
        return AnomalyResult(
            is_anomaly=len(anomalies) > 0,
            anomaly_score=anomaly_score,
            anomalies=anomalies
        )
    
    def _detect_valor_anomaly(self, transaction: dict, profile: UserProfile) -> Optional[AnomalyDetail]:
        """Detect value anomaly using z-score."""
        try:
            valor = float(transaction.get("valor", 0))
            if valor == 0:
                return None
            
            stats = profile.statistics.valor
            z_score = abs((valor - stats.mean) / stats.std) if stats.std > 0 else 0
            
            if z_score >= self.Z_SCORE_THRESHOLD_HIGH:
                severity = (
                    AnomalySeverity.HIGH if z_score >= self.Z_SCORE_THRESHOLD_CRITICAL
                    else AnomalySeverity.MEDIUM
                )
                
                return AnomalyDetail(
                    type=AnomalyType.VALOR_ANOMALY,
                    severity=severity,
                    description=f"Transaction value {valor:.2f} is {z_score:.2f} standard deviations from mean",
                    expected_range=f"{stats.p25:.2f} - {stats.p95:.2f}",
                    actual_value=valor,
                    z_score=z_score
                )
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        
        return None
    
    def _detect_horario_anomaly(self, transaction: dict, profile: UserProfile) -> Optional[AnomalyDetail]:
        """Detect hour anomaly."""
        try:
            timestamp = transaction.get("timestamp", "")
            if not timestamp:
                return None
            
            # Parse timestamp and extract hour
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            hora = dt.hour
            
            stats = profile.statistics.hora
            hour_diff = abs(hora - stats.mean)
            
            # Check if transaction is outside habitual hour range
            if hour_diff > self.HOUR_DEVIATION_THRESHOLD:
                return AnomalyDetail(
                    type=AnomalyType.HORARIO_ANOMALY,
                    severity=AnomalySeverity.MEDIUM,
                    description=f"Transaction at {hora}:00 is outside habitual hour range",
                    expected_range=f"{stats.p25:.0f}h - {stats.p75:.0f}h",
                    actual_value=f"{hora}:00"
                )
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _detect_destino_anomaly(self, transaction: dict, profile: UserProfile) -> Optional[AnomalyDetail]:
        """Detect destination anomaly (new CPF or bank)."""
        try:
            receiver = transaction.get("receiver", {})
            receiver_cpf = receiver.get("cpfReceiver")
            receiver_banco = receiver.get("banco")
            
            anomalies = []
            
            # Check if receiver CPF is common
            if receiver_cpf and profile.destinations.common_cpfs:
                if receiver_cpf not in profile.destinations.common_cpfs:
                    anomalies.append("new CPF")
            
            # Check if receiver bank is common
            if receiver_banco and profile.destinations.common_bancos:
                banco_str = str(receiver_banco)
                if banco_str not in profile.destinations.common_bancos:
                    anomalies.append("new bank")
            
            if anomalies:
                return AnomalyDetail(
                    type=AnomalyType.DESTINO_ANOMALY,
                    severity=AnomalySeverity.MEDIUM,
                    description=f"Transaction to {', '.join(anomalies)}",
                    expected_range="common destinations",
                    actual_value=f"CPF: {receiver_cpf}, Banco: {receiver_banco}"
                )
        except (ValueError, TypeError, AttributeError):
            pass
        
        return None
    
    def _detect_canal_anomaly(self, transaction: dict, profile: UserProfile) -> Optional[AnomalyDetail]:
        """Detect channel anomaly."""
        try:
            canal = transaction.get("canal", "").lower()
            if not canal:
                return None
            
            canais = profile.canais
            canal_usage = getattr(canais, canal, None)
            
            if canal_usage is None:
                return AnomalyDetail(
                    type=AnomalyType.CANAL_ANOMALY,
                    severity=AnomalySeverity.LOW,
                    description=f"Transaction using uncommon channel: {canal}",
                    expected_range="common channels",
                    actual_value=canal
                )
            
            # Check if channel usage is very low (< 5%)
            if canal_usage.percentage < 0.05:
                return AnomalyDetail(
                    type=AnomalyType.CANAL_ANOMALY,
                    severity=AnomalySeverity.LOW,
                    description=f"Transaction using rarely used channel: {canal} ({canal_usage.percentage*100:.1f}%)",
                    expected_range="common channels (>5%)",
                    actual_value=f"{canal} ({canal_usage.percentage*100:.1f}%)"
                )
        except (ValueError, TypeError, AttributeError):
            pass
        
        return None
    
    def _calculate_anomaly_score(self, anomalies: list) -> float:
        """
        Calculate overall anomaly score based on detected anomalies.
        
        Args:
            anomalies: List of AnomalyDetail
            
        Returns:
            Anomaly score between 0.0 and 1.0
        """
        if not anomalies:
            return 0.0
        
        # Base score from number of anomalies
        base_score = min(len(anomalies) * 0.3, 0.6)
        
        # Add severity weights
        severity_weights = {
            AnomalySeverity.LOW: 0.1,
            AnomalySeverity.MEDIUM: 0.2,
            AnomalySeverity.HIGH: 0.3
        }
        
        severity_bonus = sum(
            severity_weights.get(anomaly.severity, 0.1)
            for anomaly in anomalies
        )
        
        total_score = min(base_score + severity_bonus, 1.0)
        
        return round(total_score, 2)
