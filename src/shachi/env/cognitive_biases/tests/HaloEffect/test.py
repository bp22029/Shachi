import numpy as np
from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class HaloEffectMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Halo effect in a model.

    Metric:
    𝔅(â₁, â₂) = k ⋅ (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]
    where:
    â₂, â₁ are the chosen answers for the treatment and control versions, respectively.
    k is the parameter that reflects the type of halo (k = 1 for a positive one, k = -1 otherwise).

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)
        self.k = [
            [
                insertion.text
                for insertion in treatment.TEMPLATE.get_insertions()
                if insertion.pattern == "perception"
            ]
            for ((control, treatment), _) in test_results
        ]
        self.k = np.array([[1] if k == ["positively"] else [-1] for k in self.k])
