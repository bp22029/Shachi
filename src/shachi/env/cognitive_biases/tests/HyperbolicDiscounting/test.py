import numpy as np
from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class HyperbolicDiscountingMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Hyperbolic Discounting in a model.

    Metric:
    𝔅(â₁, â₂) = k ⋅ (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]
    where:
    â₂, â₁ are the chosen answers for the treatment and control versions, respectively.
    k is the parameter that reflects the order of schemes in the test case (k = -1 if an immediate reward is presented first, k = 1 otherwise).

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)
        # set the coefficient in the metric: it depends on the 'index' custom value that we sampled
        # (and reflects which scheme is presented first, i.e., which scheme is A)
        self.k = [
            [
                insertion.text
                for insertion in control.TEMPLATE.get_insertions()
                if insertion.pattern == "control_scheme"
            ]
            for ((control, treatment), _) in self.test_results
        ]
        self.k = np.array([[-1] if "immediately" in k[0] else [1] for k in self.k])
