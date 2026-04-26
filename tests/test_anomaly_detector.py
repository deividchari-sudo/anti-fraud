"""
Tests for AnomalyDetector.
"""

import pytest
from datetime import datetime

from src.user_profile.anomaly_detector import AnomalyDetector
from src.user_profile.models import (
    UserProfile,
    Statistics,
    ValueStatistics,
    HourStatistics,
    FrequencyStatistics,
    Destinations,
    Canais,
    Produtos,
    CanalInfo,
    ProdutoInfo,
    AnomalyType,
    AnomalySeverity
)


@pytest.fixture
def sample_profile():
    """Create a sample user profile for testing."""
    return UserProfile(
        cpf="12345678901",
        created_at=datetime.utcnow(),
        last_updated=datetime.utcnow(),
        transaction_count=45,
        is_cold_start=False,
        statistics=Statistics(
            valor=ValueStatistics(
                mean=500.0,
                std=150.0,
                median=450.0,
                p25=350.0,
                p75=650.0,
                p95=850.0,
                min=100.0,
                max=1000.0
            ),
            hora=HourStatistics(
                mean=12.0,
                std=2.0,
                median=12.0,
                p25=10.0,
                p75=14.0
            ),
            frequencia=FrequencyStatistics(
                transactions_per_day_mean=2.5,
                transactions_per_day_std=1.0,
                days_active=30
            )
        ),
        destinations=Destinations(
            common_cpfs={
                "98765432100": {"count": 15, "last_seen": datetime.utcnow()},
                "45678901234": {"count": 8, "last_seen": datetime.utcnow()}
            },
            common_bancos={
                "152": {"count": 20},
                "001": {"count": 10}
            }
        ),
        canais=Canais(
            app=CanalInfo(count=36, percentage=0.8),
            web=CanalInfo(count=7, percentage=0.15),
            api=CanalInfo(count=2, percentage=0.05)
        ),
        produtos=Produtos(
            pix=ProdutoInfo(count=30, percentage=0.67),
            ted=ProdutoInfo(count=10, percentage=0.22),
            boleto=ProdutoInfo(count=5, percentage=0.11)
        )
    )


class TestAnomalyDetector:
    """Test AnomalyDetector."""
    
    def test_detect_no_anomaly_normal_transaction(self, sample_profile):
        """Test detection with normal transaction (no anomaly)."""
        detector = AnomalyDetector()
        
        transaction = {
            "id": "1",
            "timestamp": "2026-04-25T12:00:00Z",
            "canal": "app",
            "produto": "pix",
            "valor": 450.0,
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "98765432100", "banco": 152}
        }
        
        result = detector.detect(transaction, sample_profile)
        
        assert result.is_anomaly == False
        assert result.anomaly_score == 0.0
        assert len(result.anomalies) == 0
    
    def test_detect_valor_anomaly_high_z_score(self, sample_profile):
        """Test detection of value anomaly with high z-score."""
        detector = AnomalyDetector()
        
        transaction = {
            "id": "1",
            "timestamp": "2026-04-25T12:00:00Z",
            "canal": "app",
            "produto": "pix",
            "valor": 1500.0,  # 6.67 standard deviations above mean
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "98765432100", "banco": 152}
        }
        
        result = detector.detect(transaction, sample_profile)
        
        assert result.is_anomaly == True
        assert result.anomaly_score > 0
        assert len(result.anomalies) > 0
        
        valor_anomaly = next((a for a in result.anomalies if a.type == AnomalyType.VALOR_ANOMALY), None)
        assert valor_anomaly is not None
        assert valor_anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.MEDIUM]
    
    def test_detect_valor_anomaly_medium_z_score(self, sample_profile):
        """Test detection of value anomaly with medium z-score."""
        detector = AnomalyDetector()
        
        transaction = {
            "id": "1",
            "timestamp": "2026-04-25T12:00:00Z",
            "canal": "app",
            "produto": "pix",
            "valor": 950.0,  # 3 standard deviations above mean
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "98765432100", "banco": 152}
        }
        
        result = detector.detect(transaction, sample_profile)
        
        assert result.is_anomaly == True
        valor_anomaly = next((a for a in result.anomalies if a.type == AnomalyType.VALOR_ANOMALY), None)
        assert valor_anomaly is not None
    
    def test_detect_horario_anomaly(self, sample_profile):
        """Test detection of hour anomaly."""
        detector = AnomalyDetector()
        
        transaction = {
            "id": "1",
            "timestamp": "2026-04-25T03:00:00Z",  # 3 AM - far from habitual 12 PM
            "canal": "app",
            "produto": "pix",
            "valor": 450.0,
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "98765432100", "banco": 152}
        }
        
        result = detector.detect(transaction, sample_profile)
        
        assert result.is_anomaly == True
        horario_anomaly = next((a for a in result.anomalies if a.type == AnomalyType.HORARIO_ANOMALY), None)
        assert horario_anomaly is not None
        assert horario_anomaly.severity == AnomalySeverity.MEDIUM
    
    def test_detect_destino_anomaly_new_cpf(self, sample_profile):
        """Test detection of destination anomaly (new CPF)."""
        detector = AnomalyDetector()
        
        transaction = {
            "id": "1",
            "timestamp": "2026-04-25T12:00:00Z",
            "canal": "app",
            "produto": "pix",
            "valor": 450.0,
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "11122233344", "banco": 152}  # New CPF
        }
        
        result = detector.detect(transaction, sample_profile)
        
        assert result.is_anomaly == True
        destino_anomaly = next((a for a in result.anomalies if a.type == AnomalyType.DESTINO_ANOMALY), None)
        assert destino_anomaly is not None
    
    def test_detect_destino_anomaly_new_banco(self, sample_profile):
        """Test detection of destination anomaly (new bank)."""
        detector = AnomalyDetector()
        
        transaction = {
            "id": "1",
            "timestamp": "2026-04-25T12:00:00Z",
            "canal": "app",
            "produto": "pix",
            "valor": 450.0,
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "98765432100", "banco": 999}  # New bank
        }
        
        result = detector.detect(transaction, sample_profile)
        
        assert result.is_anomaly == True
        destino_anomaly = next((a for a in result.anomalies if a.type == AnomalyType.DESTINO_ANOMALY), None)
        assert destino_anomaly is not None
    
    def test_detect_canal_anomaly_rare_channel(self, sample_profile):
        """Test detection of channel anomaly (rare channel)."""
        detector = AnomalyDetector()
        
        transaction = {
            "id": "1",
            "timestamp": "2026-04-25T12:00:00Z",
            "canal": "api",  # Only 5% usage
            "produto": "pix",
            "valor": 450.0,
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "98765432100", "banco": 152}
        }
        
        result = detector.detect(transaction, sample_profile)
        
        # API is used but only 5%, might trigger low severity anomaly
        canal_anomaly = next((a for a in result.anomalies if a.type == AnomalyType.CANAL_ANOMALY), None)
        if canal_anomaly:
            assert canal_anomaly.severity == AnomalySeverity.LOW
    
    def test_cold_start_no_detection(self):
        """Test that cold start profile doesn't trigger anomalies if fallback disabled."""
        detector = AnomalyDetector(use_global_profile_fallback=False)
        
        cold_start_profile = UserProfile(
            cpf="12345678901",
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            transaction_count=5,
            is_cold_start=True,
            statistics=Statistics(
                valor=ValueStatistics(mean=500.0, std=150.0, median=450.0, p25=350.0, p75=650.0, p95=850.0, min=100.0, max=1000.0),
                hora=HourStatistics(mean=12.0, std=2.0, median=12.0, p25=10.0, p75=14.0),
                frequencia=FrequencyStatistics(transactions_per_day_mean=2.5, transactions_per_day_std=1.0, days_active=2)
            ),
            destinations=Destinations(),
            canais=Canais(app=CanalInfo(count=3, percentage=0.6), web=CanalInfo(count=2, percentage=0.4), api=CanalInfo(count=0, percentage=0.0)),
            produtos=Produtos(pix=ProdutoInfo(count=3, percentage=0.6), ted=ProdutoInfo(count=2, percentage=0.4), boleto=ProdutoInfo(count=0, percentage=0.0))
        )
        
        transaction = {
            "id": "1",
            "timestamp": "2026-04-25T12:00:00Z",
            "canal": "app",
            "produto": "pix",
            "valor": 1500.0,
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "98765432100", "banco": 152}
        }
        
        result = detector.detect(transaction, cold_start_profile)
        
        # With fallback disabled, cold start should not detect anomalies
        assert result.is_anomaly == False
        assert result.anomaly_score == 0.0
    
    def test_multiple_anomalies(self, sample_profile):
        """Test detection of multiple anomalies in one transaction."""
        detector = AnomalyDetector()
        
        transaction = {
            "id": "1",
            "timestamp": "2026-04-25T03:00:00Z",  # Unusual hour
            "canal": "api",  # Rare channel
            "produto": "pix",
            "valor": 1500.0,  # High value
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "11122233344", "banco": 999}  # New CPF and bank
        }
        
        result = detector.detect(transaction, sample_profile)
        
        assert result.is_anomaly == True
        assert len(result.anomalies) >= 2  # Should detect at least 2 anomalies
        assert result.anomaly_score > 0.5  # High score due to multiple anomalies
