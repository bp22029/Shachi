from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class ReactanceMetric(RatioScaleMetric):
    """
    A metric that measures the presence and strength of Reactance based on a set of test results.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)
