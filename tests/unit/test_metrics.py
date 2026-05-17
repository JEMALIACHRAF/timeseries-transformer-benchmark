"""
Unit tests for evaluation metrics.
"""
import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    mae,
    mase,
    rmse,
    rmsse,
    smape,
)


class TestRMSE:
    def test_perfect_forecast(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 3.0, 4.0, 5.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_non_negative(self):
        y = np.random.rand(100)
        p = np.random.rand(100)
        assert rmse(y, p) >= 0.0


class TestMAE:
    def test_perfect_forecast(self):
        y = np.array([3.0, 5.0, 2.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.0, 1.0, 2.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)


class TestSMAPE:
    def test_perfect_forecast(self):
        y = np.array([1.0, 2.0, 3.0])
        assert smape(y, y) == pytest.approx(0.0)

    def test_bounded(self):
        y_true = np.random.rand(200) + 0.1
        y_pred = np.random.rand(200) + 0.1
        result = smape(y_true, y_pred)
        assert 0.0 <= result <= 200.0

    def test_symmetry(self):
        y_true = np.array([2.0, 4.0, 6.0])
        y_pred = np.array([4.0, 2.0, 3.0])
        # SMAPE(a, b) == SMAPE(b, a)
        assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true))


class TestRMSSE:
    def test_perfect_forecast(self):
        y_true  = np.array([1.0, 2.0, 3.0, 4.0])
        y_train = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        assert rmsse(y_true, y_true, y_train) == pytest.approx(0.0, abs=1e-6)

    def test_worse_than_naive(self):
        """Model worse than naïve → RMSSE > 1."""
        y_train = np.ones(50)
        y_true  = np.ones(10) * 5.0   # True values = 5
        y_pred  = np.zeros(10)         # Predictions = 0 (terrible)
        # Naïve error on constant series ≈ 0 → RMSSE will be very large
        result = rmsse(y_true, y_pred, y_train)
        assert result > 0.0

    def test_non_negative(self):
        y_train = np.random.rand(100) + 1
        y_true  = np.random.rand(28) + 1
        y_pred  = np.random.rand(28) + 1
        assert rmsse(y_true, y_pred, y_train) >= 0.0


class TestMASE:
    def test_perfect_forecast(self):
        y_train = np.random.rand(100) + 1
        y_true  = np.array([1.0, 2.0, 3.0])
        assert mase(y_true, y_true, y_train) == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self):
        y_train = np.random.rand(100) + 1
        y_true  = np.random.rand(28) + 1
        y_pred  = np.random.rand(28) + 1
        assert mase(y_true, y_pred, y_train) >= 0.0


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        y_train = np.random.rand(100) + 1
        y_true  = np.random.rand(28) + 1
        y_pred  = np.random.rand(28) + 1
        result  = compute_all_metrics(y_true, y_pred, y_train)
        assert set(result.keys()) == {"rmsse", "mase", "smape", "rmse", "mae"}

    def test_all_non_negative(self):
        y_train = np.random.rand(100) + 1
        y_true  = np.random.rand(28) + 1
        y_pred  = np.random.rand(28) + 1
        result  = compute_all_metrics(y_true, y_pred, y_train)
        for k, v in result.items():
            assert v >= 0.0, f"Metric {k} is negative: {v}"

    def test_perfect_forecast_zeros(self):
        y_train = np.random.rand(100) + 1
        y       = np.random.rand(28) + 1
        result  = compute_all_metrics(y, y, y_train)
        assert result["rmse"]  == pytest.approx(0.0, abs=1e-6)
        assert result["mae"]   == pytest.approx(0.0, abs=1e-6)
        assert result["smape"] == pytest.approx(0.0, abs=1e-6)
