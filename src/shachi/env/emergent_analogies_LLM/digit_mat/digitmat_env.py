import builtins
import os
from collections.abc import AsyncIterator, Sequence
from datetime import datetime

import numpy as np
import pandas as pd
import pydantic
from pydantic import Field

from shachi import Environment, Message, Observation, Task


class DigitMatResponse(pydantic.BaseModel):
    pred_list: list[int] = Field(description="The new row of digits to be added to the matrix.")


class DigitMatMessage(Message):
    prompt: str = pydantic.Field(description="The prompt text that includes the digit matrix question.")


class DigitMatObservation(Observation[DigitMatMessage]):
    def format_as_prompt_text(self) -> str:
        prompt = "[1] [1] [1]\n[2] [2] [2]\n[3] [3] [3]\n\n"
        prompt += self.messages[0].prompt
        return prompt


class DigitMatDecisionResult(pydantic.BaseModel):
    index: int = Field(description="The index of the problem.")
    pred_list: list[int] = Field(description="The predicted answer to the digit matrix question.")


class AggregatedDigitMatResult(pydantic.BaseModel):
    data: pd.DataFrame = Field(description="The data of the aggregated results.")

    model_config = {"arbitrary_types_allowed": True}


class DigitMatEnv(Environment):
    def __init__(
        self,
        index: int,
        prob: dict,
        num_agents: int,
        max_trial_steps: int = 1,
    ):
        self._num_agents = num_agents
        self.index = index
        self.prob = prob
        self.max_trial_steps = max_trial_steps
        self.complete = False
        self.current_step = 0
        self.pred_list: list[int]

    def num_agents(self) -> int:
        return self._num_agents

    def get_default_agent_configs(self) -> list[dict] | None:
        return None

    def done(self) -> bool:
        return self.current_step >= self.max_trial_steps or self.complete

    async def _get_observation(self) -> dict[int, Observation]:
        if self.done():
            return {}
        prompt = ""
        for r in range(3):
            for c in range(3):
                prompt += "["
                if not (r == 2 and c == 2):
                    for i in range(len(self.prob[r][c])):
                        if self.prob[r][c][i] == -1:
                            prompt += " "
                        else:
                            prompt += str(self.prob[r][c][i])
                        if i < len(self.prob[r][c]) - 1:
                            prompt += " "
                    prompt += "]"
                    if c < 2:
                        prompt += " "
                    else:
                        prompt += "\n"
        observations: dict[int, Observation] = {}
        messages: list[DigitMatMessage] = []
        messages.append(
            DigitMatMessage(
                time=self.current_step,
                src_agent_id=None,
                dst_agent_id=0,
                prompt=prompt,
            )
        )
        observations[0] = DigitMatObservation(
            agent_id=0,
            messages=messages,
            response_type=DigitMatResponse,
        )
        return observations

    async def reset(self) -> dict[int, Observation]:
        self.current_step = 0
        return await self._get_observation()

    async def step(
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
    ) -> dict[int, Observation]:
        action = responses[0]
        if isinstance(action, DigitMatResponse):
            self.pred_list = action.pred_list
            self.complete = True
        self.current_step += 1
        return await self._get_observation()

    def get_result(self) -> DigitMatDecisionResult:
        return DigitMatDecisionResult(
            index=self.index,
            pred_list=self.pred_list,
        )


class DigitMatTask(Task):
    def __init__(
        self,
        data_path: str = "all_problems_1thru5.npz",
        target_prob_types: list[str] | None = None,
        each_prob_num: int = 20,
        max_trial_steps: int = 1,
    ):
        self._num_agents = 1
        self.max_trial_steps = max_trial_steps
        all_prob = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path), allow_pickle=True)
        all_prob_types = builtins.list(all_prob["all_problems"].item().keys())

        self.task_list = []
        for prob_type in all_prob_types:
            if target_prob_types is not None and prob_type not in target_prob_types:
                continue
            prob_type_N_prob = all_prob["all_problems"].item()[prob_type]["prob"].shape[0]
            if each_prob_num > prob_type_N_prob:
                sampled_indices = np.arange(prob_type_N_prob)
            else:
                sampled_indices = np.random.choice(prob_type_N_prob, each_prob_num, replace=False)
            for prob_ind in sampled_indices:
                prob_info = all_prob["all_problems"].item()[prob_type]

                prob = prob_info["prob"][prob_ind]
                answer_choices = prob_info["answer_choices"][prob_ind]
                correct_ind = prob_info["correct_ind"][prob_ind]
                correct_answer = answer_choices[correct_ind]

                self.task_list.append(
                    {
                        "prob_type": prob_type,
                        "prob_ind": prob_ind,
                        "prob": prob,
                        "answer_choices": answer_choices,
                        "correct_ind": correct_ind,
                        "correct_answer": correct_answer,
                    }
                )

    async def iterate_environments(
        self,
    ) -> AsyncIterator[Environment[DigitMatDecisionResult]]:
        for i, task_info in enumerate(self.task_list):
            print(f"Creating environment {i + 1}/{len(self.task_list)}")
            yield DigitMatEnv(
                index=i,
                prob=task_info["prob"],
                num_agents=self._num_agents,
                max_trial_steps=self.max_trial_steps,
            )

    def aggregate_results(self, results: Sequence[DigitMatDecisionResult]) -> AggregatedDigitMatResult:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        results_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", timestamp)
        os.makedirs(results_directory, exist_ok=True)

        result_list = []
        for result in results:
            task_info = self.task_list[result.index]
            prob_type = task_info["prob_type"]
            prob_ind = task_info["prob_ind"]
            correct_answer = task_info["correct_answer"]
            correct_pred = False
            pred_list = result.pred_list
            if self.compare_arrays(pred_list, correct_answer):
                correct_pred = True
            result_list.append(
                {
                    "prob_type": prob_type,
                    "prob_ind": prob_ind,
                    "prediction": pred_list,
                    "correct_answer": correct_answer,
                    "correct": correct_pred,
                }
            )

        df = pd.DataFrame(result_list)
        df.to_csv(os.path.join(results_directory, "results.csv"), index=False)
        acc_by_type = df.groupby("prob_type")["correct"].mean()
        acc_by_type.to_csv(os.path.join(results_directory, "acc_by_type.csv"))

        df_two = df[df["prob_type"].str.contains("two", case=False)]
        df_three = df[df["prob_type"].str.contains("three", case=False)]
        df_four = df[df["prob_type"].str.contains("four", case=False)]
        df_five = df[df["prob_type"].str.contains("five", case=False)]
        df_else = df[~df["prob_type"].str.contains("two|three|four|five", case=False)]

        acc_two = df_two["correct"].mean() if not df_two.empty else np.nan
        acc_three = df_three["correct"].mean() if not df_three.empty else np.nan
        acc_four = df_four["correct"].mean() if not df_four.empty else np.nan
        acc_five = df_five["correct"].mean() if not df_five.empty else np.nan
        acc_else = df_else["correct"].mean() if not df_else.empty else np.nan

        sub_cat_acc_dict = {
            "one": float(acc_else),
            "two": float(acc_two),
            "three": float(acc_three),
            "four": float(acc_four),
            "five": float(acc_five),
        }
        sub_cat_acc_df = pd.DataFrame.from_dict(sub_cat_acc_dict, orient="index", columns=["accuracy"])
        sub_cat_acc_df.to_csv(os.path.join(results_directory, "sub_cat_acc.csv"))

        return AggregatedDigitMatResult(
            data=df,
        )

    def compare_arrays(self, pred_list: list[int], correct_answer: np.ndarray) -> bool:
        pred_array = np.array(pred_list)
        correct_answer = np.array(correct_answer)
        if pred_array.shape != correct_answer.shape:
            return False
        return bool(np.all(pred_array == correct_answer))
