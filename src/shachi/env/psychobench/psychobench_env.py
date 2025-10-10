import json
import random
from collections.abc import AsyncIterator, Sequence
from logging import getLogger
from pathlib import Path
from typing import Any

import pydantic
import requests  # type: ignore[import-untyped]

from shachi.base import Environment, Observation, Task

from .data_types import MIN_MAX_SCORES, PsychoBenchTestName, PsychoBenchTestType
from .observation import (
    PsychoBenchIntroMessage,
    PsychoBenchObservation,
    PsychoBenchQuestionMessage,
    QuestionnaireAnswers,
)
from .test_results import AggregatedPsychoBenchResult, PsychoBenchResult

logger = getLogger(__name__)


def retrieve_questionnaire_json() -> list[Any]:
    questionnaire_json_url = "https://raw.githubusercontent.com/CUHK-ARISE/PsychoBench/main/questionnaires.json"

    questionnaire_json_path = Path(__file__).parent / "questionnaires.json"
    if not questionnaire_json_path.exists():
        response = requests.get(questionnaire_json_url)
        response.raise_for_status()

        with open(questionnaire_json_path, "wb") as f:
            f.write(response.content)

    q = json.loads(questionnaire_json_path.read_text())
    if not isinstance(q, list):
        raise RuntimeError(f"Internal Error: loaded questionnaire should be of type list, while we got {type(q)}; {q}")
    return q


def get_questionnaire(questionnaire_name: str) -> PsychoBenchTestType:
    try:
        data = retrieve_questionnaire_json()
    except FileNotFoundError:
        raise FileNotFoundError("The 'questionnaires.json' file does not exist.")

    questionnaire = None
    for item in data:
        if item["name"] == questionnaire_name:
            questionnaire = item

    if questionnaire is None:
        raise ValueError("Questionnaire not found.")

    return PsychoBenchTestType.model_validate(questionnaire)


class PsychoBenchEnv(Environment):
    def __init__(
        self,
        questionnaire: PsychoBenchTestName,
        max_questions_per_step: int | None = None,
        shuffle_questions: bool = True,
        question_shuffle_seed: int | None = None,
        max_parse_retries: int = 3,
    ):
        if question_shuffle_seed is not None:
            self.rng = random.Random(question_shuffle_seed)
        else:
            self.rng = random.Random()

        self.max_questions_per_step = max_questions_per_step
        self.shuffle_questions = shuffle_questions

        self.max_parse_retries = max_parse_retries
        self.current_parse_retry = 0

        self.questionnaire = get_questionnaire(questionnaire)
        self.test_questions_generator = self.questionnaire.generate_test_questions(shuffle=self.shuffle_questions)

        self._is_done: bool = False
        self.current_step_index = 0
        self.current_test_number = 0
        self.current_prepared_obs: list[PsychoBenchObservation] = []
        self.questions_to_answers: dict[int, int] = {}
        self.current_qkey_map: dict[int, int] = {}

    def prepare_obs(self, agent_id: int = 0, test_number: int = 0) -> list[PsychoBenchObservation]:
        """
        Prepares the observation batches.
        Observation is possibly split to multiple batches if max_questions_per_step is set.
        """

        questionnaire = self.questionnaire
        min_score, max_score = MIN_MAX_SCORES[self.questionnaire.name]

        intro_message = PsychoBenchIntroMessage(
            time=0,
            src_agent_id=None,
            dst_agent_id=agent_id,
            questionnaire_text=questionnaire.prompt,
            min_score=min_score,
            max_score=max_score,
        )
        all_questions_for_test = self.test_questions_generator.get_test(
            test_number=test_number, agent_id=0, rng=self.rng
        )

        if self.max_questions_per_step is not None:
            separated_question_messages = [
                all_questions_for_test[i : i + self.max_questions_per_step]
                for i in range(0, len(all_questions_for_test), self.max_questions_per_step)
            ]

            obs = []
            for msgs in separated_question_messages:
                obs.append(
                    PsychoBenchObservation(
                        agent_id=agent_id,
                        messages=[intro_message] + msgs,
                    )
                )
        else:
            obs = [PsychoBenchObservation(agent_id=agent_id, messages=[intro_message] + all_questions_for_test)]

        return obs

    def num_agents(self) -> int:
        """PsychoBench is a single-player env."""
        return 1

    def get_default_agent_configs(self) -> list[dict] | None:
        """
        Config for an agent. It is a signle agent env, so we return only a single config.
        """
        configs = [{"system_prompt": self.questionnaire.inner_setting}]
        return configs

    def done(self) -> bool:
        """Returns True if all tests have been completed."""
        return self._is_done

    def _get_current_observation(self, agent_id: int = 0) -> dict[int, Observation] | None:
        if self.current_step_index >= len(self.current_prepared_obs):
            return None

        current_batch_obs = self.current_prepared_obs[self.current_step_index]
        self.current_step_index += 1

        self.current_qkey_map = {}
        for msg in current_batch_obs.messages[1:]:
            if not isinstance(msg, PsychoBenchQuestionMessage):
                raise RuntimeError(
                    f"Internal Error: messages in obs except for the first one "
                    f"should be PsychoBenchQuestionMessage, while we got type {type(msg)}; {msg}"
                )
            self.current_qkey_map[int(msg.question_key)] = int(msg._original_question_key)

        return {agent_id: current_batch_obs}

    async def reset(self, agent_id: int = 0) -> dict[int, Observation]:
        """
        Resets the environment for a new run of all specified tests.
        Returns the first batch of questions.
        """
        logger.info("-- Resetting PsychoBench Environment --")
        self._is_done = False
        self.current_test_number = self.current_test_number + 1
        self.questions_to_answers = dict()
        self.current_step_index = 0

        self.current_prepared_obs = self.prepare_obs(agent_id=agent_id, test_number=self.current_test_number)

        obs = self._get_current_observation(agent_id)

        if obs is None:
            raise RuntimeError("Internal Error: obs is None at the first step, something went wrong!")

        logger.info(f"-- Questionnaire '{self.questionnaire.name}', Step {self.current_step_index + 1} --")
        return obs

    async def step(
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
        agent_id: int = 0,
    ) -> dict[int, Observation]:
        """
        Processes the agent's answers and returns the next batch of questions or terminates.
        """
        if self._is_done:
            raise RuntimeError("Please call reset since the run is already done.")

        response = responses[agent_id]

        assert isinstance(response, QuestionnaireAnswers), (
            f"Invalid response type {type(response)}, response: {response}"
        )
        logger.info(f"Received {len(response.answers)} answers for questionnaire '{self.questionnaire.name}'.")
        for answer in response.answers:
            displayed_key = int(answer.question_key)
            answer_value = int(answer.answer)

            if displayed_key in self.current_qkey_map:
                original_key = self.current_qkey_map[displayed_key]
                min_score, max_score = MIN_MAX_SCORES[self.questionnaire.name]
                if not (min_score <= answer_value <= max_score):
                    logger.warning(
                        f"Answer {answer_value} for Q{displayed_key}(orig:{original_key}) in "
                        f"{self.questionnaire.name} is out of range [{min_score}-{max_score}]. Retrying..."
                    )
                    self.current_parse_retry += 1
                    if self.current_parse_retry > self.max_parse_retries:
                        raise RuntimeError(
                            f"Max parse retry attempts {self.max_parse_retries} attempt exceeded, exiting..."
                        )
                    return {agent_id: self.current_prepared_obs[self.current_step_index - 1]}
                self.questions_to_answers[original_key] = answer_value
            else:
                logger.warning(
                    f"Received answer for unexpected displayed key {displayed_key} for questionnaire "
                    f"'{self.questionnaire.name}'. It might be from a previous step or invalid. Retrying..."
                )
                self.current_parse_retry += 1
                if self.current_parse_retry > self.max_parse_retries:
                    raise RuntimeError(
                        f"Max parse retry attempts {self.max_parse_retries} attempt exceeded, exiting..."
                    )
                return {agent_id: self.current_prepared_obs[self.current_step_index - 1]}

        next_obs = self._get_current_observation(agent_id)

        if next_obs is None:
            logger.info(
                f"-- Finished Questionnaire '{self.questionnaire.name}' for Test {self.current_test_number + 1} --"
            )
            self._is_done = True
            next_obs = dict()
        else:
            logger.info(
                f"-- Continuing Test, Questionnaire '{self.questionnaire.name}', Step {self.current_step_index + 1} --"
            )
        return next_obs

    def get_result(self) -> PsychoBenchResult:
        if not self.done():
            raise RuntimeError("get_result should be called after the env is done.")

        return PsychoBenchResult(
            questionnaire=self.questionnaire.model_dump(),  # type: ignore[arg-type] # dict arg is automatically parsed by pydantic
            questions_to_answers=self.questions_to_answers,
        )


class PsychoBenchTask(Task):
    def __init__(
        self,
        questionnaire: list[PsychoBenchTestName] | PsychoBenchTestName,
        num_tests: int = 100,
        max_questions_per_step: int | None = None,
        shuffle_questions: bool = True,
        max_parse_retries: int = 3,
    ):
        if isinstance(questionnaire, str):
            self.questionnaires = [questionnaire]
        else:
            self.questionnaires = questionnaire

        self.num_tests = num_tests
        self.max_questions_per_step = max_questions_per_step
        self.shuffle_questions = shuffle_questions
        self.max_parse_retries = max_parse_retries

    async def iterate_environments(
        self,
    ) -> AsyncIterator[Environment[PsychoBenchResult]]:
        seed = 0
        for questionnaire in self.questionnaires:
            for i in range(self.num_tests):
                seed += 1
                yield PsychoBenchEnv(
                    questionnaire=questionnaire,
                    max_questions_per_step=self.max_questions_per_step,
                    shuffle_questions=self.shuffle_questions,
                    question_shuffle_seed=seed,
                )

    def aggregate_results(self, results: Sequence[PsychoBenchResult]) -> AggregatedPsychoBenchResult:
        aggregated_result = AggregatedPsychoBenchResult.from_test_results(list(results))
        logger.info(f"Aggregated Result:\n{aggregated_result.model_dump_json(indent=4)}")
        return aggregated_result
