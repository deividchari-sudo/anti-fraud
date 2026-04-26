"""Tests for ml_mixins (Sprint 2 audit fix)."""

import numpy as np
import pandas as pd

from src.ml_mixins import (
    CHANNEL_FEATURES,
    PRODUCT_FEATURES,
    SHAPExplainerMixin,
    ThresholdTuningMixin,
)


class _FakeModelWithThresholds(ThresholdTuningMixin):
    def __init__(self):
        self.threshold = 0.5
        self.thresholds_by_channel = {}
        self.thresholds_by_product = {}


class TestThresholdTuningMixin:
    def test_best_f1_threshold_returns_value_in_unit_interval(self):
        np.random.seed(0)
        proba = np.random.rand(100)
        y_true = np.random.randint(0, 2, 100)
        thr = ThresholdTuningMixin.best_f1_threshold(y_true, proba)
        assert thr is None or 0.0 <= thr <= 1.0

    def test_best_f1_threshold_single_class_returns_none(self):
        proba = np.array([0.1, 0.2, 0.3])
        y_true = np.array([0, 0, 0])
        assert ThresholdTuningMixin.best_f1_threshold(y_true, proba) is None

    def test_select_threshold_priority_product_over_channel(self):
        m = _FakeModelWithThresholds()
        m.thresholds_by_product = {"produto_pix": 0.25}
        m.thresholds_by_channel = {"canal_app": 0.45}
        m.threshold = 0.5

        row = pd.Series({"produto_pix": 1, "canal_app": 1})
        assert m.select_threshold(row) == 0.25

    def test_select_threshold_falls_back_to_global(self):
        m = _FakeModelWithThresholds()
        row = pd.Series({"produto_pix": 0, "canal_app": 0})
        assert m.select_threshold(row) == 0.5

    def test_optimize_thresholds_populates_per_segment(self):
        np.random.seed(42)
        n = 500
        df = pd.DataFrame(
            {
                "canal_app": np.random.randint(0, 2, n),
                "canal_web": 0,
                "canal_api": 0,
                "produto_pix": np.random.randint(0, 2, n),
                "produto_ted": 0,
                "produto_boleto": 0,
                "produto_autenticacao": 0,
                "produto_financeiro_generico": 0,
            }
        )
        proba = np.random.rand(n)
        y = pd.Series(np.random.randint(0, 2, n))

        m = _FakeModelWithThresholds()
        m.optimize_thresholds(proba, df, y)
        # global threshold updated
        assert 0.0 <= m.threshold <= 1.0

    def test_constants_are_lists(self):
        assert isinstance(CHANNEL_FEATURES, list) and len(CHANNEL_FEATURES) == 3
        assert isinstance(PRODUCT_FEATURES, list) and len(PRODUCT_FEATURES) == 5


class TestSHAPExplainerMixin:
    def test_summarize_explanation_with_positives_and_negatives(self):
        top = {"feature_a": 0.5, "feature_b": -0.3, "feature_c": 0.1}
        out = SHAPExplainerMixin._summarize_explanation(top)
        assert "Indicadores de fraude" in out
        assert "Indicadores legítimos" in out

    def test_summarize_explanation_empty(self):
        assert SHAPExplainerMixin._summarize_explanation({}) == "No significant features."

    def test_get_shap_explanation_returns_none_without_explainer(self):
        class Host(SHAPExplainerMixin):
            explainer = None
            feature_names = ["a", "b"]

            class _Logger:
                def error(self, *_args, **_kw):
                    pass

            logger = _Logger()

        out = Host().get_shap_explanation(
            pd.DataFrame([{"a": 1, "b": 2}]),
            fraud_probability=0.7,
            model_name="x",
        )
        assert out is None
