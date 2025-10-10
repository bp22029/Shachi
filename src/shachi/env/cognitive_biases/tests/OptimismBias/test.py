import numpy as np
from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class OptimismBiasMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the optimism bias in a model.

    Individual metric:
    𝔅(â₁, â₂) = k ⋅ (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    k is the kind of event (-1: positive or 1: negative).

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)
        # set the coefficient in the metric: it depends on the 'index' custom value that we sampled
        # (and reflects which event kind is used in the test case)
        self.k = [
            [
                insertion.text
                for insertion in treatment.TEMPLATE.get_insertions()
                if insertion.pattern == "event_kind"
            ]
            for ((control, treatment), _) in self.test_results
        ]
        self.k = np.array([[-1] if "positive" in k[0] else [1] for k in self.k])
