import ast

import numpy as np
from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class SocialDesirabilityBiasMetric(RatioScaleMetric):
    """
    A metric that measures the presence and strength of Social Desirability Bias based on a set of test results.

    Attributes:
        test_results (list[tuple[TestCase, DecisionResult]]): The list of test results to be used for the metric calculation.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)

        # Extract from the remarks whether the statements used for the tests where socially desirable or not
        desirable = [
            ast.literal_eval(treatment.REMARKS)["desirable"]
            for ((control, treatment), _) in self.test_results
        ]

        self.k = np.array([[1] if d else [-1] for d in desirable])
