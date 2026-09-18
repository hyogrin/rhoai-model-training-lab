"""Tests for evaluation: pass^k, failures_in_denominator, CI, retention_delta."""

from __future__ import annotations

import math

import pytest

from rhoai_model_training_lab.evaluation import (
    ConfidenceCalculator,
    TauEpisodeRunner,
)
from rhoai_model_training_lab.schemas.evaluation import (
    RetentionResult,
    TaskResult,
)


class TestPassK:
    """Test pass^k computation (NOT pass@k)."""

    def test_all_succeed(self):
        """All tasks succeed on all trials → pass^k = 1.0."""
        task_trials = {
            "task-1": [True, True, True],
            "task-2": [True, True, True],
        }
        assert TauEpisodeRunner.compute_pass_k(task_trials, k=3) == 1.0

    def test_all_fail(self):
        """All tasks fail at least one trial → pass^k = 0.0."""
        task_trials = {
            "task-1": [False, True, True],
            "task-2": [True, False, True],
        }
        assert TauEpisodeRunner.compute_pass_k(task_trials, k=3) == 0.0

    def test_mixed_results(self):
        """One task always succeeds, one fails → pass^k = 0.5."""
        task_trials = {
            "task-1": [True, True, True],
            "task-2": [True, True, False],
        }
        result = TauEpisodeRunner.compute_pass_k(task_trials, k=3)
        assert result == pytest.approx(0.5)

    def test_single_trial(self):
        """With k=1, pass^k reduces to task success rate."""
        task_trials = {
            "task-1": [True],
            "task-2": [False],
            "task-3": [True],
        }
        result = TauEpisodeRunner.compute_pass_k(task_trials, k=1)
        assert result == pytest.approx(2.0 / 3.0)

    def test_fewer_trials_than_k(self):
        """Tasks with fewer trials than k score 0."""
        task_trials = {
            "task-1": [True, True],
            "task-2": [True],
        }
        result = TauEpisodeRunner.compute_pass_k(task_trials, k=3)
        assert result == 0.0

    def test_empty_task_trials(self):
        result = TauEpisodeRunner.compute_pass_k({}, k=3)
        assert result == 0.0

    def test_pass_k_not_pass_at_k(self):
        """pass^k requires ALL k trials to succeed (unlike pass@k best-of-k)."""
        task_trials = {
            "task-1": [True, False, True],
        }
        pass_k = TauEpisodeRunner.compute_pass_k(task_trials, k=3)
        assert pass_k == 0.0, "pass^k=0 even though 2/3 trials succeeded (not best-of-k)"


class TestFailuresInDenominator:
    """Test that timeouts and failures are always counted in the denominator."""

    def test_timeout_counted_as_failure(self):
        """Timed-out episodes do NOT get excluded from the denominator."""
        task_trials = {
            "task-1": [True, True, True],
            "task-2": [False, False, False],
        }
        result = TauEpisodeRunner.compute_pass_k(task_trials, k=3)
        assert result == pytest.approx(0.5)

    def test_failure_in_pass_k(self):
        """A single failure in any trial makes that task score 0."""
        task_trials = {
            "task-1": [True, True, True],
            "task-2": [True, True, False],
            "task-3": [True, True, True],
        }
        result = TauEpisodeRunner.compute_pass_k(task_trials, k=3)
        assert result == pytest.approx(2.0 / 3.0)

    def test_task_results_success_rate_includes_failures(self):
        """Aggregate success rate always divides by total, including errors."""
        results = [
            TaskResult(task_id="t1", trial=0, success=True),
            TaskResult(task_id="t2", trial=0, success=False, error="timeout", error_type="timeout"),
            TaskResult(task_id="t3", trial=0, success=False, error="api error", error_type="api_error"),
        ]
        total = len(results)
        succeeded = sum(1 for r in results if r.success)
        rate = succeeded / total
        assert rate == pytest.approx(1.0 / 3.0)


class TestConfidenceInterval:
    """Test confidence_interval calculation."""

    @pytest.fixture
    def calculator(self):
        return ConfidenceCalculator(n_bootstrap=1000, alpha=0.05, seed=42)

    def test_perfect_results(self, calculator):
        results = [
            TaskResult(task_id=f"t{i}", trial=0, success=True) for i in range(20)
        ]
        pe, ci_lo, ci_hi = calculator.compute_ci(results)
        assert pe == 1.0
        assert ci_lo == 1.0
        assert ci_hi == 1.0

    def test_zero_results(self, calculator):
        results = [
            TaskResult(task_id=f"t{i}", trial=0, success=False) for i in range(20)
        ]
        pe, ci_lo, ci_hi = calculator.compute_ci(results)
        assert pe == 0.0
        assert ci_lo == 0.0
        assert ci_hi == 0.0

    def test_mixed_results_ci_bounds(self, calculator):
        results = []
        for i in range(50):
            results.append(TaskResult(
                task_id=f"t{i}",
                trial=0,
                success=(i % 2 == 0),
            ))
        pe, ci_lo, ci_hi = calculator.compute_ci(results)

        assert 0.0 <= ci_lo <= pe <= ci_hi <= 1.0
        assert pe == pytest.approx(0.5, abs=0.05)
        assert ci_lo < ci_hi

    def test_empty_results(self, calculator):
        pe, ci_lo, ci_hi = calculator.compute_ci([])
        assert pe == 0.0
        assert ci_lo == 0.0
        assert ci_hi == 0.0

    def test_ci_deterministic_with_seed(self):
        results = [
            TaskResult(task_id=f"t{i}", trial=0, success=(i % 3 != 0))
            for i in range(30)
        ]
        calc1 = ConfidenceCalculator(n_bootstrap=500, seed=123)
        calc2 = ConfidenceCalculator(n_bootstrap=500, seed=123)
        r1 = calc1.compute_ci(results)
        r2 = calc2.compute_ci(results)
        assert r1 == r2

    def test_multiple_trials_per_task(self, calculator):
        results = []
        for task_id in ["t1", "t2", "t3", "t4", "t5"]:
            for trial in range(3):
                results.append(TaskResult(
                    task_id=task_id,
                    trial=trial,
                    success=(task_id in ["t1", "t2", "t3"]),
                ))
        pe, ci_lo, ci_hi = calculator.compute_ci(results)
        assert pe == pytest.approx(3.0 / 5.0)


class TestRetentionDeltaPP:
    """Test retention_delta_pp computation."""

    def test_positive_delta(self):
        r = RetentionResult(
            benchmark="arc_challenge",
            variant="lora",
            accuracy=0.75,
            base_accuracy=0.70,
            retention_delta_pp=100.0 * (0.75 - 0.70),
        )
        assert r.retention_delta_pp == pytest.approx(5.0)

    def test_negative_delta(self):
        r = RetentionResult(
            benchmark="arc_challenge",
            variant="osft",
            accuracy=0.65,
            base_accuracy=0.70,
            retention_delta_pp=100.0 * (0.65 - 0.70),
        )
        assert r.retention_delta_pp == pytest.approx(-5.0)

    def test_zero_delta(self):
        r = RetentionResult(
            benchmark="arc_challenge",
            variant="lora",
            accuracy=0.70,
            base_accuracy=0.70,
            retention_delta_pp=100.0 * (0.70 - 0.70),
        )
        assert r.retention_delta_pp == pytest.approx(0.0)

    def test_formula_correctness(self):
        """retention_delta_pp = 100 * (adapted - base)."""
        adapted = 0.82
        base = 0.78
        delta = 100.0 * (adapted - base)
        r = RetentionResult(
            benchmark="arc_challenge",
            variant="lora",
            accuracy=adapted,
            base_accuracy=base,
            retention_delta_pp=delta,
        )
        assert r.retention_delta_pp == pytest.approx(4.0)
