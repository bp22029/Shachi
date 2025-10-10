import numpy as np
from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class StereotypingMetric(RatioScaleMetric):
    """
    A metric that measures the presence and strength of Stereotyping based on a set of test results.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results, k=np.array([1]))
