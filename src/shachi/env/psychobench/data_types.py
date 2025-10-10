import random
from statistics import mean, stdev
from typing import Dict, List, Literal, Optional, Tuple, TypeAlias, get_args

import pydantic
from pydantic import BaseModel

from .observation import PsychoBenchQuestionMessage

PsychoBenchTestName = Literal[
    "BFI",
    "DTDD",
    "EPQ-R",
    "ECR-R",
    "CABIN",
    "GSE",
    "LMS",
    "BSRI",
    "ICB",
    "LOT-R",
    "Empathy",
    "EIS",
    "WLEIS",
    "16P",
]

PSYCHOBENCH_TEST_NAMES: Tuple[PsychoBenchTestName, ...] = get_args(PsychoBenchTestName)

MIN_MAX_SCORES: Dict[PsychoBenchTestName, Tuple[int, int]] = {
    "BFI": (1, 5),
    "DTDD": (1, 9),
    "EPQ-R": (0, 1),
    "ECR-R": (1, 7),
    "CABIN": (1, 5),
    "GSE": (1, 4),
    "LMS": (1, 5),
    "BSRI": (1, 7),
    "ICB": (1, 6),
    "LOT-R": (0, 4),
    "Empathy": (1, 7),
    "EIS": (1, 5),
    "WLEIS": (1, 7),
    "16P": (1, 7),
}
Result: TypeAlias = Tuple[float, float, int]


class PsychoBenchTestCat(BaseModel):
    cat_name: str
    cat_questions: List[int]

    # Result from crowd
    crowd: Optional[List[Dict[str, str | float | int]]] = None


class TestQuestions(BaseModel):
    questions: Dict[str, str]
    all_tests: Dict[int, List[Tuple[int, str]]] = pydantic.Field(default_factory=dict)
    shuffle: bool = True

    def get_test(
        self, test_number: int, agent_id: int, rng: random.Random
    ) -> List[PsychoBenchQuestionMessage]:
        """
        Generate and get test for a given test_number.
        Different test_number uses differnt orderings of questions.
        """
        if test_number not in self.all_tests:
            question_ids = list(self.questions.keys())
            if self.shuffle:
                rng.shuffle(question_ids)

            questions = []
            for q_id, original_q_id in enumerate(question_ids, 1):
                question = PsychoBenchQuestionMessage(
                    time=0,
                    src_agent_id=None,
                    dst_agent_id=agent_id,
                    question_key=q_id,
                    question_text=self.questions[str(original_q_id)],
                )
                question._original_question_key = original_q_id
                questions.append(question)
            self.all_tests[test_number] = questions

        return self.all_tests[test_number]


class CatResult(BaseModel):
    cat_name: str
    score_mean: float
    score_std: float
    num_test: int


class PsychoBenchTestType(BaseModel):
    # Test type, e.g. Empathy, BFI...
    name: PsychoBenchTestName

    # Each question is indexed by some string number, e.g. "1", "2", ...
    questions: Dict[str, str]

    # We use this scale to reverse the score, i.e. scale - score is the reversed score.
    # For example, for the test with scores 1~5, scale is 6.
    scale: int

    # How to calculate the overall score for each category
    compute_mode: Literal["AVG", "SUM"]

    # Question indices where score should be reversed (i.e. 1 is higher if reversed)
    reverse: List[int]

    categories: List[PsychoBenchTestCat]

    # System prompt
    inner_setting: str

    # User prompt for task instruction
    prompt: str

    def generate_test_questions(self, shuffle: bool = True) -> TestQuestions:
        return TestQuestions(questions=self.questions, shuffle=shuffle)

    def get_cat_questions(self, cat_name: str) -> List[int]:
        for cat in self.categories:
            if cat_name == cat.cat_name:
                return cat.cat_questions
        raise KeyError(f"Invalid cat_name {cat_name}")

    def is_reversed(self, idx: int) -> bool:
        return idx in self.reverse

    def compute_statistics(self, data_list: List[Dict[int, int]]) -> list[CatResult]:
        results = []

        for cat in self.categories:
            scores_list = []

            for data in data_list:
                scores = []
                for key in data:
                    if int(key) in cat.cat_questions:
                        score = int(data[key])
                        if score == 0:
                            score = 0
                        elif self.is_reversed(int(key)):
                            score = self.scale - score
                        scores.append(score)

                if self.compute_mode == "SUM":
                    scores_list.append(sum(scores))
                else:
                    scores_list.append(mean(scores))

            results.append(
                CatResult(
                    cat_name=cat.cat_name,
                    score_mean=mean(scores_list),
                    score_std=stdev(scores_list) if len(scores_list) > 1 else float('nan'),
                    num_test=len(scores_list),
                )
            )

        return results
