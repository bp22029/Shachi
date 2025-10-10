import numpy as np
from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class HindsightBiasMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Hindsight bias in a model.

    Metric:
    𝔅 = (‖ â₁ − a' ‖₁ − ‖ â₂ − a' ‖₁) / max[‖ â₁ − a' ‖₁, ‖ â₂ − a' ‖₁] ∈ [-1, 1];

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a' is the option closest to the ground truth percentage (sampled using custom values);

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)
        # extract the options closest to the ground truth values and set them as parameters x_1 and x_2.
        self.x_1 = [
            [
                insertion.text
                for insertion in treatment.TEMPLATE.get_insertions()
                if insertion.pattern == "percentage"
            ]
            for ((control, treatment), _) in test_results
        ]
        self.x_1 = np.array([[round(int(x[0]) / 10) + 5] for x in self.x_1])
        self.x_2 = self.x_1
        self.k = 1
