import numpy as np
from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class PlanningFallacyMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Planning fallacy in a model.

    Metric:
    𝔅(â₁, â₂) = (â₁ + x₁ - â₂) / max(â₁ + x₁, â₂) ∈ [-1, 1]

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    x₁ is the parameter that corresponds to the rational estimation update.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): A list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)
        # extract the estimation updates' values and set them as the parameter x_1.
        self.x_1 = [
            [
                insertion.text
                for insertion in treatment.TEMPLATE.get_insertions()
                if insertion.pattern == "estimation_update"
            ]
            for ((control, treatment), _) in test_results
        ]
        self.x_1 = np.array([[int(x[0]) // 10] for x in self.x_1])
        # account for the sign of the parameter x_1 in the metric
        self.x_1 = -self.x_1
        # to make the estimator unbiased, we set the parameter x_2 to -𝔼[x_1] = -3
        self.x_2 = -3
        self.k = 1
