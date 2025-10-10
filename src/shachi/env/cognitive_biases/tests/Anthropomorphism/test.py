from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class AnthropomorphismMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Anthropomorphism in a model.

    Metric:
    𝔅(â₁, â₂) = k ⋅ (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]
    where:
    â₂, â₁ are the chosen answers for the treatment and control versions, respectively.
    k is the parameter that reflects the order of reasons in the test case (k = 1 if the anthropomorphic reason is presented first, k = -1 otherwise).

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)
        self.k = 1
