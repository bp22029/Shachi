import numpy as np
from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class BandwagonEffectMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Bandwagon Effect in a model.

    Metric:
    𝔅(â₁, â₂) = k ⋅ (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]
    where:
    â₂, â₁ are the chosen answers for the treatment and control versions, respectively.
    k is the parameter that reflects the majority opinion in the test case (k = -1 if it is A, k = 1 otherwise).

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)
        # set the coefficient in the metric: it depends on the 'index' custom value that we sampled
        # (and reflects which opinion is presented as the majority one)
        self.k = [
            [
                insertion.text
                for insertion in control.TEMPLATE.get_insertions()
                if insertion.pattern == "majority_opinion"
            ]
            for ((control, treatment), _) in self.test_results
        ]
        self.k = np.array([[-1] if "A" in k[0] else [1] for k in self.k])
        # we flip the treatment answers
        self.flip_treatment = True
