import numpy as np
from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class AnchoringMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the anchoring in a model.

    Metric:
    𝔅 = (‖ â₁ − a' ‖₁ − ‖ â₂ − a' ‖₁) / max[‖ â₁ − a' ‖₁, ‖ â₂ − a' ‖₁] ∈ [-1, 1];

    where:
    â₁, â₂ are the chosen answers for the control and treatment versions, respectively;
    a' is the answer option closest to the anchor value;
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)
        # set the coefficient in the metric
        self.k = 1
        # set the anchor values as the parameters x_1 and x_2 in the metric
        self.x_1 = [
            [
                insertion.text
                for insertion in treatment.TEMPLATE.get_insertions()
                if insertion.pattern == "anchor"
            ]
            for ((control, treatment), _) in self.test_results
        ]
        self.x_1 = np.array([[round(int(anchor[0]) / 10)] for anchor in self.x_1])
        self.x_2 = self.x_1
