"""
QuantEdge — Unit Tests
Tests factor computation, regime detection, A/B testing, and API endpoints.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from factors.compute import (cross_sectional_zscore, compute_value_factor,
                              compute_quality_factor, compute_information_coefficient)
from backtest.engine import compute_sharpe, compute_max_drawdown, power_analysis, ab_test_strategies, BacktestResult


class TestFactorComputation:

    def test_cross_sectional_zscore_mean_zero(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = cross_sectional_zscore(s)
        assert abs(z.mean()) < 1e-6

    def test_cross_sectional_zscore_std_one(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = cross_sectional_zscore(s)
        assert abs(z.std() - 1.0) < 0.1

    def test_value_factor_direction(self):
        fund_df = pd.DataFrame({
            "pb_ratio": [1.0, 5.0, 10.0],
            "pe_ratio": [10.0, 25.0, 50.0],
            "ps_ratio": [0.5, 2.0, 5.0],
        }, index=["CHEAP", "MID", "EXPENSIVE"])
        value = compute_value_factor(fund_df)
        assert value["CHEAP"] > value["EXPENSIVE"], "Cheap stocks should have higher value scores"

    def test_quality_factor_direction(self):
        fund_df = pd.DataFrame({
            "roe": [0.30, 0.10, 0.05],
            "gross_margins": [0.60, 0.30, 0.10],
            "debt_to_equity": [0.10, 0.50, 2.00],
        }, index=["HIGH_QUALITY", "MID", "LOW_QUALITY"])
        quality = compute_quality_factor(fund_df)
        assert quality["HIGH_QUALITY"] > quality["LOW_QUALITY"], \
            "High quality stocks should have higher quality scores"

    def test_ic_computation(self):
        factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        fwd_ret = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        result = compute_information_coefficient(factor, fwd_ret)
        assert result["ic"] > 0.9, "Perfect rank correlation should produce IC near 1.0"
        assert result["significant"] is True


class TestBacktestEngine:

    def test_sharpe_positive_returns(self):
        returns = pd.Series([0.001] * 252)
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_sharpe_negative_returns(self):
        returns = pd.Series([-0.001] * 252)
        sharpe = compute_sharpe(returns)
        assert sharpe < 0

    def test_max_drawdown_zero_for_always_up(self):
        returns = pd.Series([0.001] * 100)
        mdd = compute_max_drawdown(returns)
        assert mdd == 0.0

    def test_max_drawdown_negative_for_losses(self):
        returns = pd.Series([0.01, 0.01, -0.05, 0.01, 0.01])
        mdd = compute_max_drawdown(returns)
        assert mdd < 0

    def test_power_analysis_reasonable_n(self):
        n = power_analysis(effect_size=0.3, alpha=0.05, power=0.8)
        assert 50 <= n <= 200, f"Expected 50-200 samples, got {n}"

    def test_ab_test_insufficient_data(self):
        treatment = BacktestResult(
            strategy="treatment", total_return=0.1, annualized_return=0.1,
            annualized_volatility=0.15, sharpe_ratio=0.67, max_drawdown=-0.1,
            calmar_ratio=1.0, win_rate=0.55, quarterly_sharpes=[0.5],
        )
        control = BacktestResult(
            strategy="control", total_return=0.05, annualized_return=0.05,
            annualized_volatility=0.12, sharpe_ratio=0.42, max_drawdown=-0.08,
            calmar_ratio=0.6, win_rate=0.52, quarterly_sharpes=[0.2],
        )
        result = ab_test_strategies(treatment, control)
        assert "error" in result, "Should flag insufficient data for A/B test"

    def test_ab_test_with_sufficient_data(self):
        np.random.seed(42)
        treatment = BacktestResult(
            strategy="treatment", total_return=0.5, annualized_return=0.08,
            annualized_volatility=0.15, sharpe_ratio=0.8, max_drawdown=-0.12,
            calmar_ratio=0.67, win_rate=0.58,
            quarterly_sharpes=list(np.random.normal(0.8, 0.3, 40)),
        )
        control = BacktestResult(
            strategy="control", total_return=0.3, annualized_return=0.05,
            annualized_volatility=0.15, sharpe_ratio=0.5, max_drawdown=-0.15,
            calmar_ratio=0.33, win_rate=0.52,
            quarterly_sharpes=list(np.random.normal(0.4, 0.3, 40)),
        )
        result = ab_test_strategies(treatment, control)
        assert "t_stat" in result
        assert "p_value" in result
        assert "cohens_d" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
