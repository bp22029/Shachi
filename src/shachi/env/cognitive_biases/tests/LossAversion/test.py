import numpy as np
from core.base import RatioScaleMetric
from core.testing import DecisionResult, TestCase


class LossAversionMetric(RatioScaleMetric):
    """
    A class that describes the quantitative evaluation of the Loss aversion bias in a model.

    Individual metric:
    𝔅(â₁, â₂) = k ⋅ (â₁ - â₂) / max(â₁, â₂) ∈ [-1, 1]

    Batch metric:
    𝔅 = (∑ wᵢ𝔅ᵢ) / (∑ wᵢ) ∈ [-1, 1]

    where:
    â₂ is the chosen answer for the i-th test;
    â₁ is the fixed central (neutral) option of the scale;
    k is the parameter that reflects the order of choices in the test case (k = 1 if the guaranteed choice is presented first, k = -1 otherwise).
    wᵢ is the loss aversion hyperparameter in the i-th test (test_weights). Set as 1.
    """

    def __init__(self, test_results: list[tuple[(TestCase, TestCase), DecisionResult]]):
        super().__init__(test_results)
        # set the coefficient in the metric: it depends on the 'index' custom value that we sampled
        # (and reflects which scheme is presented first, i.e., which scheme is A)
        self.k = [
            [
                insertion.text
                for insertion in treatment.TEMPLATE.get_insertions()
                if insertion.pattern == "treatment_choice"
            ]
            for ((control, treatment), _) in self.test_results
        ]
        self.k = np.array([[-1] if "guarantees" in k[0] else [1] for k in self.k])
        # we also need to flip treatment options
        self.flip_treatment = True
        # extract lambda parameters from the test cases and set them as the test_weights in the metric
        # lambda_amounts = np.array([
        #     [
        #         float(insertion.text)
        #         for insertion in test_case.TREATMENT.get_insertions()
        #         if insertion.pattern == "lambda_amount"
        #     ]
        #     for (test_case, _) in self.test_results
        # ])
        # base_amounts = np.array([
        #     [
        #         float(insertion.text)
        #         for insertion in test_case.TREATMENT.get_insertions()
        #         if insertion.pattern == "base_amount"
        #     ]
        #     for (test_case, _) in self.test_results
        # ])
        # self.test_weights = lambda_amounts / base_amounts
