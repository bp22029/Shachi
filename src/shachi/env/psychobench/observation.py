from typing import Union

import pydantic

from shachi.base import Message, Observation


class PsychoBenchIntroMessage(Message):
    questionnaire_text: str = pydantic.Field(
        ..., description="The main instructions for the questionnaire."
    )
    min_score: int = pydantic.Field(..., description="Minimum score for the answer scale.")
    max_score: int = pydantic.Field(..., description="Maximum score for the answer scale.")


class PsychoBenchQuestionMessage(Message):
    """
    Message containing either the introductory prompt or a specific questionnaire question.
    Multiple question messages will be sent in a single observation.
    """

    question_key: int = pydantic.Field(
        ..., description="The key (integer) of the question in the questionnaire."
    )
    question_text: str = pydantic.Field(
        ..., description="The text of the specific questionnaire question."
    )

    _original_question_key: int = pydantic.PrivateAttr()

    def __str__(self) -> str:
        return f"{self.question_key}. {self.question_text}"


class QuestionnaireAnswer(pydantic.BaseModel):
    """Expected response format from the agent, containing all answers."""

    question_key: int = pydantic.Field(
        ..., description="The key (integer) of the question in the questionnaire."
    )
    answer: int = pydantic.Field(
        description="An answer score for a question with the question_key."
    )


class QuestionnaireAnswers(pydantic.BaseModel):
    """Expected response format from the agent, containing all answers."""

    answers: list[QuestionnaireAnswer] = pydantic.Field(
        ..., description="The answers to the questions."
    )


class PsychoBenchObservation(
    Observation[Union[PsychoBenchIntroMessage, PsychoBenchQuestionMessage]]
):
    """
    Observation for the PsychoBench environment.
    In the standard flow, reset() returns an observation with all questions,
    and step() processes the AllAnswers response.
    """

    response_type: type[pydantic.BaseModel] = QuestionnaireAnswers

    def format_as_prompt_text(self) -> str:
        """Formats the observation containing all questions as a single text prompt."""
        if not self.messages:
            raise RuntimeError("No messages included.")

        intro_message = self.messages[0]
        if not isinstance(intro_message, PsychoBenchIntroMessage):
            raise RuntimeError(
                f"Internal Error: The first message should be PsychoBenchIntroMessage, while self.messages is {self.messages}"
            )

        full_prompt = intro_message.questionnaire_text

        for msg in self.messages[1:]:
            if not isinstance(msg, PsychoBenchQuestionMessage):
                raise RuntimeError(
                    f"Internal Error: The second or later messages should be PsychoBenchQuestionMessage, while self.messages is {self.messages}"
                )
            full_prompt += f"\n{str(msg)}"

        return full_prompt
