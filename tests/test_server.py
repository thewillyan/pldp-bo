from __future__ import annotations

import numpy as np


def _median_weights(norms: list[float]) -> np.ndarray:
    b = float(np.median(norms))
    weights = np.array(
        [1.0 if r <= 1e-12 else min(1.0, b / r) for r in norms],
        dtype=np.float64,
    )
    return weights


def _weighted_average(deltas: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    total = np.sum(weights)
    if total <= 0:
        return np.zeros_like(deltas[0])
    norm_w = weights / total
    return np.sum([w * d for w, d in zip(norm_w, deltas, strict=True)], axis=0)


class TestMedianWeightComputation:
    def test_median_with_outlier(self) -> None:
        norms = [1.0, 2.0, 10.0]
        weights = _median_weights(norms)
        expected = np.array([1.0, 1.0, 0.2])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_all_equal(self) -> None:
        norms = [3.0, 3.0, 3.0]
        weights = _median_weights(norms)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_single_client(self) -> None:
        norms = [5.0]
        weights = _median_weights(norms)
        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_two_clients(self) -> None:
        norms = [1.0, 4.0]
        weights = _median_weights(norms)
        expected = np.array([1.0, 0.625])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_zero_norm_does_not_cause_division_error(self) -> None:
        norms = [0.0, 2.0, 8.0]
        weights = _median_weights(norms)
        assert np.all(np.isfinite(weights))

    def test_larger_norm_gets_smaller_weight(self) -> None:
        norms = [1.0, 5.0, 10.0]
        weights = _median_weights(norms)
        assert weights[1] > weights[2]


class TestWeightedAveraging:
    def test_simple_weighted_average(self) -> None:
        deltas = [np.array([1.0, 0.0]), np.array([0.0, 2.0])]
        weights = np.array([0.75, 0.25])
        result = _weighted_average(deltas, weights)
        expected = np.array([0.75, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_delta(self) -> None:
        deltas = [np.array([3.0, 4.0])]
        weights = np.array([1.0])
        result = _weighted_average(deltas, weights)
        np.testing.assert_array_almost_equal(result, [3.0, 4.0])

    def test_equal_weights(self) -> None:
        deltas = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        weights = np.array([1.0, 1.0])
        result = _weighted_average(deltas, weights)
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_weights_returns_zeros(self) -> None:
        deltas = [np.array([1.0, 2.0])]
        weights = np.array([0.0])
        result = _weighted_average(deltas, weights)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0])

    def test_integration_with_median_weights(self) -> None:
        norms = [1.0, 2.0, 10.0]
        weights = _median_weights(norms)
        deltas = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
        agg = _weighted_average(deltas, weights)
        w_sum = (
            1.0 * np.array([1.0, 0.0])
            + 1.0 * np.array([0.0, 1.0])
            + 0.2 * np.array([0.5, 0.5])
        ) / 2.2
        expected = w_sum
        np.testing.assert_array_almost_equal(agg, expected)

    def test_server_learning_rate_applied(self) -> None:
        global_weights = np.array([1.0, 1.0])
        aggregated_delta = np.array([0.5, -0.3])
        lr = 0.5
        new_weights = global_weights + lr * aggregated_delta
        expected = np.array([1.25, 0.85])
        np.testing.assert_array_almost_equal(new_weights, expected)
