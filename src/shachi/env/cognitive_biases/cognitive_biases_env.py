import datetime
import importlib
import logging
import os
import sys
from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
from typing import cast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import pandas as pd
import pydantic
from pydantic import Field

from shachi import Environment, Message, Observation, Task
from shachi.env.cognitive_biases.analysis import load_decision_data, plot_bias_heatmap
from shachi.env.cognitive_biases.core.base import RatioScaleMetric
from shachi.env.cognitive_biases.core.testing import DecisionResult, Template, TestCase


class CognitiveBiasResponse(pydantic.BaseModel):
    option: int = Field(
        description="The number of the option chosen by the model. This is the final decision made by the model."
    )


class CognitiveBiasMessage(Message):
    prompt: str = pydantic.Field(
        description="The prompt to be sent to the model. It should contain the scenario and the questions."
    )


class CognitiveBiasObservation(Observation[CognitiveBiasMessage]):
    def format_as_prompt_text(self) -> str:
        parts = [
            "You will be given a decision-making task with multiple answer options.\n\n",
            self.messages[0].prompt,
            "\n\nAt the end of your answer, identify the selected option.\n",
            "From a text with multiple entries like 'Option N: details', "
            "extract only one 'Option N' without additional details.\n",
        ]
        return "".join(parts)


class CognitiveBiasDecisionResult(pydantic.BaseModel):
    id: int = Field(description="The id of the test case.")
    bias: str = Field(description="The name of the bias.")
    condition: str = Field(description="The condition of the test case.")
    answer: int | None = Field(description="The answer given by the model.")
    option_texts: list[str] = Field(description="The texts of the options given to the model.")
    option_order: list[int] = Field(description="The order of the options given to the model.")


class AggregatedCognitiveBiasResult(pydantic.BaseModel):
    data: pd.DataFrame = Field(description="The data of the aggregated results.")

    model_config = {"arbitrary_types_allowed": True}


class CognitiveBiasEnv(Environment):
    def __init__(
        self,
        test_case: TestCase,
        num_agents: int = 1,
        max_trial_steps: int = 1,
        randomly_flip_options: bool = True,
        shuffle_answer_options: bool = False,
    ):
        self.test_case = test_case
        self._num_agents = num_agents
        self.max_trial_steps = max_trial_steps
        self.randomly_flip_options = randomly_flip_options
        self.shuffle_answer_options = shuffle_answer_options
        self.complete = False
        self.current_step = 0
        self.answer: int | None = None
        self.option_texts: list[str] = []
        self.option_order: list[int] = []

    def num_agents(self) -> int:
        return self._num_agents

    def get_default_agent_configs(self) -> list[dict] | None:
        return None

    def done(self) -> bool:
        return self.current_step >= self.max_trial_steps or self.complete

    async def _get_observation(self) -> dict[int, Observation]:
        if self.done():
            return {}
        prompt = self.test_case.TEMPLATE.format(
            randomly_flip_options=self.randomly_flip_options,
            shuffle_options=self.shuffle_answer_options,
            seed=self.test_case.SEED,
        )
        self.option_texts, self.option_order = self.test_case.TEMPLATE.get_options(
            randomly_flip_options=self.randomly_flip_options,
            shuffle_options=self.shuffle_answer_options,
            seed=self.test_case.SEED,
        )

        observations: dict[int, Observation] = {}
        messages: list[CognitiveBiasMessage] = []
        messages.append(
            CognitiveBiasMessage(
                time=self.current_step,
                src_agent_id=None,
                dst_agent_id=0,
                prompt=prompt,
            )
        )
        observations[0] = CognitiveBiasObservation(
            agent_id=0,
            messages=messages,
            response_type=CognitiveBiasResponse,
        )

        return observations

    async def reset(self) -> dict[int, Observation]:
        self.current_step = 0
        self.answer, self.option_texts, self.option_order = None, [], []
        return await self._get_observation()

    async def step(
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
    ) -> dict[int, Observation]:
        action = responses[0]
        if isinstance(action, CognitiveBiasResponse):
            self.answer = action.option
            self.complete = True
        self.current_step += 1
        return await self._get_observation()

    def get_result(self) -> CognitiveBiasDecisionResult:
        return CognitiveBiasDecisionResult(
            id=int(self.test_case.ID),
            bias=str(self.test_case.BIAS),
            condition=str(self.test_case.CONDITION),
            answer=self.answer,
            option_texts=self.option_texts,
            option_order=self.option_order,
        )


class CognitiveBiasTask(Task):
    def __init__(
        self,
        dataset_file_path: str,
        decision_results_dir: str,
        max_trial_steps: int = 1,
        randomly_flip_options: bool = True,
        shuffle_answer_options: bool = False,
        target_biases: list[str] | None = None,
    ) -> None:
        self._num_agents = 1
        self.max_trial_steps = max_trial_steps
        self.randomly_flip_options = randomly_flip_options
        self.shuffle_answer_options = shuffle_answer_options
        self.dataset_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_file_path)
        self.decision_results_dir = decision_results_dir
        self.dataset = pd.read_csv(self.dataset_file_path)
        self.dataset["seed"] = range(len(self.dataset))
        self.test_cases = []

        biases = [
            "".join(" " + char if char.isupper() else char for char in bias).strip().title().replace(" ", "")
            for bias in self.dataset["bias"].unique()
        ]
        for bias in biases:
            if target_biases is not None and bias not in target_biases:
                continue
            for _, row in self.dataset[
                self.dataset["bias"].str.strip().str.title().str.replace(" ", "") == bias
            ].iterrows():
                self.test_cases.append(
                    (
                        TestCase(
                            id=row["id"],
                            bias=bias,
                            condition="control",
                            template=Template(row["raw_control"]),
                            generator=row["generator"],
                            temperature=row["temperature"],
                            seed=row["seed"],
                            scenario=row["scenario"],
                            variant=row["variant"],
                            remarks=row["remarks"],
                        )
                        if row["raw_control"]
                        else None,
                        TestCase(
                            id=row["id"],
                            bias=bias,
                            condition="treatment",
                            template=Template(row["raw_treatment"]),
                            generator=row["generator"],
                            temperature=row["temperature"],
                            seed=row["seed"],
                            scenario=row["scenario"],
                            variant=row["variant"],
                            remarks=row["remarks"],
                        )
                        if row["raw_treatment"]
                        else None,
                    )
                )

    async def iterate_environments(
        self,
    ) -> AsyncIterator[Environment[CognitiveBiasDecisionResult]]:
        for i, test_case in enumerate(self.test_cases):
            logging.info(f"Creating environment {i * 2 + 1}/{len(self.test_cases) * 2}")
            for condition in test_case:
                if condition is not None:
                    yield CognitiveBiasEnv(
                        condition,
                        self._num_agents,
                        self.max_trial_steps,
                        self.randomly_flip_options,
                        self.shuffle_answer_options,
                    )

    def aggregate_results(self, results: Sequence[CognitiveBiasDecisionResult]) -> AggregatedCognitiveBiasResult:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.results_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.decision_results_dir, timestamp
        )
        os.makedirs(self.results_directory, exist_ok=True)

        grouped: defaultdict[int, dict[str, CognitiveBiasDecisionResult]] = defaultdict(dict)
        for r in results:
            grouped[r.id][r.condition] = r

        self.decision_results = []
        for id_, condition_map in grouped.items():
            control_res = condition_map.get("control")
            treatment_res = condition_map.get("treatment")
            try:
                decision_result = DecisionResult(
                    id=id_,
                    bias=control_res.bias if control_res else (treatment_res.bias if treatment_res else ""),
                    model="model_name",
                    control_answer="",
                    control_extraction="",
                    control_decision=(int(control_res.answer) if control_res and control_res.answer is not None else 0),
                    control_options=(
                        list(control_res.option_texts) if control_res and control_res.option_texts is not None else []
                    ),
                    control_option_order=(
                        list(control_res.option_order) if control_res and control_res.option_order is not None else []
                    ),
                    treatment_answer="",
                    treatment_extraction="",
                    treatment_decision=(
                        int(treatment_res.answer) if treatment_res and treatment_res.answer is not None else 0
                    ),
                    treatment_options=(
                        list(treatment_res.option_texts)
                        if treatment_res and treatment_res.option_texts is not None
                        else []
                    ),
                    treatment_option_order=(
                        list(treatment_res.option_order)
                        if treatment_res and treatment_res.option_order is not None
                        else []
                    ),
                    status="OK",
                )
            except Exception as e:
                decision_result = DecisionResult(
                    id=id_,
                    bias=control_res.bias if control_res else (treatment_res.bias if treatment_res else ""),
                    model="model_name",
                    control_answer="",
                    control_extraction="",
                    control_decision=0,
                    control_options=[],
                    control_option_order=[],
                    treatment_answer="",
                    treatment_extraction="",
                    treatment_decision=NotImplemented,
                    treatment_options=[],
                    treatment_option_order=[],
                    status="ERROR",
                    error_message=str(e),
                )
            self.decision_results.append(decision_result)
        self.summarise_results()
        df_decisions = load_decision_data(self.results_directory)
        df_decisions.to_csv(os.path.join(self.results_directory, "decisions.csv"), index=False)

        plot_bias_heatmap(df_decisions, legend=True, figsize=(11.0, 10.0), save_plot_dir=self.results_directory)
        return AggregatedCognitiveBiasResult(
            data=df_decisions,
        )

    def summarise_results(self) -> None:
        results_data = []
        for i, decision_result in enumerate(self.decision_results):
            row = {
                "model": decision_result.MODEL,
                "id": decision_result.ID,
                "bias": decision_result.BIAS,
                "control_decision": decision_result.CONTROL_DECISION,
                "control_options": decision_result.CONTROL_OPTIONS,
                "control_option_order": decision_result.CONTROL_OPTION_SHUFFLING,
                "treatment_decision": decision_result.TREATMENT_DECISION,
                "treatment_options": decision_result.TREATMENT_OPTIONS,
                "treatment_option_order": decision_result.TREATMENT_OPTION_SHUFFLING,
                "status": decision_result.STATUS,
                "error_message": decision_result.ERROR_MESSAGE,
            }
            results_data.append(row)
        decision_df = pd.DataFrame(results_data)

        failed_idx = [i for i, dr in enumerate(self.decision_results) if dr.STATUS == "ERROR"]

        for bias_name in decision_df["bias"].unique():
            idx_for_bias_ok = decision_df.index[(decision_df["bias"] == bias_name) & (decision_df["status"] == "OK")]

            if len(idx_for_bias_ok) == 0:
                continue

            test_cases_for_bias = [self.test_cases[i] for i in idx_for_bias_ok if i not in failed_idx]
            decision_results_for_bias = [self.decision_results[i] for i in idx_for_bias_ok if i not in failed_idx]

            metric = self.get_metric(bias_name)(test_results=list(zip(test_cases_for_bias, decision_results_for_bias)))
            individual_scores = metric.compute()

            if len(individual_scores) == len(idx_for_bias_ok):
                decision_df.loc[idx_for_bias_ok, "individual_score"] = individual_scores
            if len(metric.test_weights) == len(idx_for_bias_ok):
                decision_df.loc[idx_for_bias_ok, "weight"] = metric.test_weights

        file_name = os.path.join(
            self.results_directory,
            f"batch_{os.getpid()}_decided_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        )
        decision_df.to_csv(file_name, index=False)

        merge_datasets = self.merge_datasets(self.results_directory, add_id=False)
        merge_datasets.to_csv(os.path.join(self.results_directory, "merged_decisions.csv"), index=False)

    def get_metric(self, bias: str) -> type[RatioScaleMetric]:
        try:
            module_path = f"shachi.env.cognitive_biases.tests.{bias}.test"
            module = importlib.import_module(module_path)
            class_name = f"{bias}Metric"
            MetricClass = cast(type[RatioScaleMetric], getattr(module, class_name))
            return MetricClass
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f"Could not find the metric for bias '{bias}': {e}")

    def merge_datasets(self, datasets_directory: str, add_id: bool = True) -> pd.DataFrame:
        # Load and concatenate all CSV files from the datasets directory
        dataframes = []
        for csv_file in os.listdir(datasets_directory):
            if csv_file.endswith(".csv"):
                csv_path = os.path.join(datasets_directory, csv_file)
                df = pd.read_csv(csv_path)
                dataframes.append(df)

        # Concatenate all the loaded DataFrames into a single DataFrame
        merged_df = pd.concat(dataframes, ignore_index=True)

        # Add a new index column named 'id', if requested
        if add_id:
            merged_df.reset_index(drop=True, inplace=True)
            merged_df.index.name = "id"
            merged_df.reset_index(inplace=True)

        return merged_df
