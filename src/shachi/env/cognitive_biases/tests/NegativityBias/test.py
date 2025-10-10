from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class NegativityBiasMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Negativity bias in a model.

    Metric:
    𝔅(â₁, â₂) = (â₂ - â₁) / max(â₁, â₂) ∈ [-1, 1]

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): A list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)
        # Reflect the treatment options w.r.t. the central option
        self.flip_treatment = True
