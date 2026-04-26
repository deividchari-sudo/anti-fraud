"""Tests for Sprint 3 models: AutoEncoder + SimpleGraphSAGE."""

import numpy as np
import pandas as pd
import pytest

from src.autoencoder_anomaly import AutoEncoderAnomalyDetector
from src.simple_graph_sage import SimpleGraphSAGE


class TestAutoEncoderAnomalyDetector:
    @pytest.fixture
    def legitimate_data(self):
        np.random.seed(42)
        n = 200
        return pd.DataFrame(
            {
                "valor": np.random.uniform(50, 1000, n),
                "hora": np.random.randint(8, 22, n),
                "dia_semana": np.random.randint(0, 5, n),
                "canal_app": np.random.randint(0, 2, n),
                "valor_log": np.log1p(np.random.uniform(50, 1000, n)),
            }
        )

    def test_initialization(self):
        ae = AutoEncoderAnomalyDetector()
        assert ae.is_fitted is False
        assert ae.threshold == 0.0

    def test_fit_returns_metrics(self, legitimate_data):
        ae = AutoEncoderAnomalyDetector(max_iter=20)
        metrics = ae.fit(legitimate_data)
        assert ae.is_fitted is True
        assert "anomaly_threshold" in metrics
        assert metrics["n_train"] == len(legitimate_data)
        assert metrics["mean_error"] >= 0

    def test_predict_returns_score_in_unit_interval(self, legitimate_data):
        ae = AutoEncoderAnomalyDetector(max_iter=20)
        ae.fit(legitimate_data)
        result = ae.predict(legitimate_data.head(1))
        assert 0.0 <= result["anomaly_score"] <= 1.0
        assert isinstance(result["is_zero_day_anomaly"], bool)
        assert result["model_fitted"] is True

    def test_predict_unfitted_returns_safe_default(self, legitimate_data):
        ae = AutoEncoderAnomalyDetector()
        result = ae.predict(legitimate_data.head(1))
        assert result["model_fitted"] is False
        assert result["anomaly_score"] == 0.0

    def test_anomalous_input_has_higher_score(self, legitimate_data):
        ae = AutoEncoderAnomalyDetector(max_iter=20, anomaly_percentile=95.0)
        ae.fit(legitimate_data)

        # Out-of-distribution sample: very high amount
        anomalous = pd.DataFrame(
            {
                "valor": [100_000.0],
                "hora": [3],
                "dia_semana": [6],
                "canal_app": [1],
                "valor_log": [np.log1p(100_000)],
            }
        )
        normal = legitimate_data.head(1)

        anomalous_score = ae.predict(anomalous)["anomaly_score"]
        normal_score = ae.predict(normal)["anomaly_score"]
        assert anomalous_score >= normal_score

    def test_save_and_load_roundtrip(self, legitimate_data, tmp_path):
        path = str(tmp_path / "ae.pkl")
        ae = AutoEncoderAnomalyDetector(max_iter=20, model_path=path)
        ae.fit(legitimate_data)
        ae.save()

        ae2 = AutoEncoderAnomalyDetector(model_path=path)
        ae2.load()
        assert ae2.is_fitted
        assert ae2.threshold == ae.threshold

    def test_fit_raises_for_too_few_samples(self):
        ae = AutoEncoderAnomalyDetector()
        df = pd.DataFrame({"valor": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 10"):
            ae.fit(df)


class TestSimpleGraphSAGE:
    @pytest.fixture
    def transactions(self):
        return [
            {"sender": {"cpfSender": "A"}, "receiver": {"cpfReceiver": "B"}, "valor": 100},
            {"sender": {"cpfSender": "A"}, "receiver": {"cpfReceiver": "C"}, "valor": 200},
            {"sender": {"cpfSender": "B"}, "receiver": {"cpfReceiver": "D"}, "valor": 50},
            {"sender": {"cpfSender": "C"}, "receiver": {"cpfReceiver": "D"}, "valor": 150},
            {"sender": {"cpfSender": "D"}, "receiver": {"cpfReceiver": "E"}, "valor": 75},
        ]

    def test_initialization_validates_args(self):
        with pytest.raises(ValueError, match="num_hops"):
            SimpleGraphSAGE(num_hops=3)
        with pytest.raises(ValueError, match="embedding_dim"):
            SimpleGraphSAGE(embedding_dim=2)

    def test_build_graph_populates_nodes(self, transactions):
        gsage = SimpleGraphSAGE()
        gsage.build_graph(transactions)
        assert gsage.nodes == {"A", "B", "C", "D", "E"}

    def test_get_embedding_correct_size(self, transactions):
        gsage = SimpleGraphSAGE(num_hops=2, embedding_dim=8)
        gsage.build_graph(transactions)
        emb = gsage.get_embedding("A")
        # 2-hop: own + hop1 + hop2 = 8 * 3
        assert emb.shape == (24,)

    def test_get_embedding_one_hop(self, transactions):
        gsage = SimpleGraphSAGE(num_hops=1, embedding_dim=8)
        gsage.build_graph(transactions)
        emb = gsage.get_embedding("A")
        assert emb.shape == (16,)

    def test_get_embedding_for_unknown_node_returns_zeros(self, transactions):
        gsage = SimpleGraphSAGE()
        gsage.build_graph(transactions)
        emb = gsage.get_embedding("UNKNOWN")
        assert np.allclose(emb, 0.0)

    def test_extract_features_dict_returns_named_keys(self, transactions):
        gsage = SimpleGraphSAGE()
        gsage.build_graph(transactions)
        feats = gsage.extract_features_dict("A", prefix="gsage")
        assert all(k.startswith("gsage_") for k in feats)
        assert len(feats) == gsage.output_dim

    def test_extract_features_dict_for_none_cpf(self):
        gsage = SimpleGraphSAGE()
        feats = gsage.extract_features_dict(None)
        assert all(v == 0.0 for v in feats.values())

    def test_embeddings_differ_between_nodes(self, transactions):
        gsage = SimpleGraphSAGE()
        gsage.build_graph(transactions)
        emb_a = gsage.get_embedding("A")
        emb_e = gsage.get_embedding("E")
        # A is sender (high out-degree), E is leaf - should differ
        assert not np.allclose(emb_a, emb_e)
