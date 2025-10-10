from collections import defaultdict
from typing import Dict, List, Literal

from pydantic import BaseModel

from shachi.env.psychobench.analysis import ResultFor16P, analysis_personality
from shachi.env.psychobench.data_types import (
    CatResult,
    PsychoBenchTestName,
    PsychoBenchTestType,
)


class ResultForAQuestonnaire(BaseModel):
    questionnaire_name: PsychoBenchTestName
    compute_mode: Literal["SUM", "AVG"]

    results: List[CatResult]


class PsychoBenchResult(BaseModel):
    questionnaire: PsychoBenchTestType
    questions_to_answers: Dict[int, int]


class AggregatedPsychoBenchResult(BaseModel):
    results: List[ResultForAQuestonnaire | ResultFor16P]

    @classmethod
    def from_test_results(
        cls, results: List[PsychoBenchResult]
    ) -> "AggregatedPsychoBenchResult":
        q_name_to_results: dict[PsychoBenchTestName, list[PsychoBenchResult]] = (
            defaultdict(list)
        )
        for result in results:
            q_name_to_results[result.questionnaire.name].append(result)

        aggregated_result: List[ResultForAQuestonnaire | ResultFor16P] = []
        for results_for_q in q_name_to_results.values():
            questionnaire = results_for_q[0].questionnaire
            questions_to_answer_list = [
                result.questions_to_answers for result in results_for_q
            ]
            if questionnaire.name != "16P":
                cat_results = questionnaire.compute_statistics(questions_to_answer_list)
                aggregated_result.append(
                    ResultForAQuestonnaire(
                        questionnaire_name=questionnaire.name,
                        compute_mode=questionnaire.compute_mode,
                        results=cat_results,
                    )
                )
            else:
                aggregated_result.append(analysis_personality(questions_to_answer_list))

        return AggregatedPsychoBenchResult(results=aggregated_result)
