import numpy as np

from anygoal.evaluation.counterfactual import (
    counterfactual_sensitivity,
    counterfactual_sensitivity_uniformity,
    evaluate_counterfactual_metrics,
    factual_output_variation,
    relative_action_sensitivity,
)
from anygoal.evaluation.counterfactual_runner import (
    _insert_peg_after_grip_mask,
)


def test_counterfactual_sensitivity_averages_actions():
    factual = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    counterfactual = np.asarray(
        [[[3.0, 4.0], [0.0, 0.0]], [[4.0, 5.0], [1.0, 2.0]]]
    )
    np.testing.assert_allclose(counterfactual_sensitivity(factual, counterfactual), [2.5, 3.0])


def test_ras_uses_exact_pairwise_median_and_is_scale_invariant():
    factual = np.asarray([[0.0], [2.0], [5.0]])
    counterfactual = factual[:, None] + np.asarray([[[1.0]], [[2.0]], [[4.0]]])
    result = evaluate_counterfactual_metrics(
        factual,
        counterfactual,
        {"axis": np.asarray([0.0, 1.0, 2.0])},
        {"axis": np.asarray([0.0, 1.0, 2.0])},
    )
    assert factual_output_variation(factual) == 3.0
    scaled = evaluate_counterfactual_metrics(
        factual * 7,
        counterfactual * 7,
        {"axis": np.asarray([0.0, 1.0, 2.0])},
        {"axis": np.asarray([0.0, 1.0, 2.0])},
    )
    assert np.isclose(result["ras"], scaled["ras"])


def test_csu_equal_bin_weighting_and_empty_bin_omission():
    sensitivity = np.asarray([1.0, 1.0, 3.0, 3.0])
    bias = np.asarray([0.0, 0.1, 2.0, 2.1])
    score, medians, populated = counterfactual_sensitivity_uniformity(
        sensitivity, bias, np.asarray([0.0, 1.0, 2.0, 3.0])
    )
    np.testing.assert_array_equal(populated, [0, 2])
    np.testing.assert_allclose(medians, [1.0, 3.0])
    assert np.isclose(score, 2.0 / 3.0)


def test_zero_sensitivity_metrics_are_well_defined():
    ras, relative = relative_action_sensitivity(np.zeros(3), 0.0)
    csu, _, _ = counterfactual_sensitivity_uniformity(
        np.zeros(3), np.arange(3), np.asarray([0.0, 1.0, 2.0])
    )
    assert ras == 0.0
    np.testing.assert_array_equal(relative, np.zeros(3))
    # This is 0.5 (not 1.0) under the report's literal epsilon-stabilized
    # formula: every zero bin contributes (0 / epsilon - 1)^2.
    assert csu == 0.5


def test_insert_peg_after_grip_requires_ten_close_synchronized_frames():
    observation = np.zeros((14, 44), dtype=np.float32)
    hand = np.arange(14, dtype=np.float32)[:, None] * np.asarray([[0.002, 0.001]])
    observation[:, 29:31] = hand
    observation[:, 33:35] = hand + np.asarray([0.06, 0.0])
    mask = _insert_peg_after_grip_mask(observation)
    np.testing.assert_array_equal(mask[:9], False)
    np.testing.assert_array_equal(mask[9:], True)

    observation[11, 33] += 0.02
    mask = _insert_peg_after_grip_mask(observation)
    assert not mask[11]
    assert not mask[12]
