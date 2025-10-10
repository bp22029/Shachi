import json
import os
import random
import shutil
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from statistics import mean, stdev
from typing import Any, Literal

import pandas as pd
import pydantic
import scipy.stats as stats
from omegaconf import DictConfig
from pydantic import BaseModel, Field, field_validator

from shachi import Environment, Message, Observation, Task

Score = Literal[1, 2, 3, 4, 5]


class PANASItem(BaseModel):
    name: str = Field(..., description="The name of the item.")
    score: Score = Field(..., description="The score of the item, between 1 and 5.")

    @field_validator("score", mode="before")
    def validate_score(cls, v: Any) -> int:
        if isinstance(v, str) and v.isdigit():
            v = int(v)
        if v < 1:
            return 1
        if v > 5:
            return 5
        return int(v)


class PANASResponse(pydantic.BaseModel):
    item1: PANASItem = Field(..., description="The first item of the PANAS questionnaire.")
    item2: PANASItem = Field(..., description="The second item of the PANAS questionnaire.")
    item3: PANASItem = Field(..., description="The third item of the PANAS questionnaire.")
    item4: PANASItem = Field(..., description="The fourth item of the PANAS questionnaire.")
    item5: PANASItem = Field(..., description="The fifth item of the PANAS questionnaire.")
    item6: PANASItem = Field(..., description="The sixth item of the PANAS questionnaire.")
    item7: PANASItem = Field(..., description="The seventh item of the PANAS questionnaire.")
    item8: PANASItem = Field(..., description="The eighth item of the PANAS questionnaire.")
    item9: PANASItem = Field(..., description="The ninth item of the PANAS questionnaire.")
    item10: PANASItem = Field(..., description="The tenth item of the PANAS questionnaire.")
    item11: PANASItem = Field(..., description="The eleventh item of the PANAS questionnaire.")
    item12: PANASItem = Field(..., description="The twelfth item of the PANAS questionnaire.")
    item13: PANASItem = Field(..., description="The thirteenth item of the PANAS questionnaire.")
    item14: PANASItem = Field(..., description="The fourteenth item of the PANAS questionnaire.")
    item15: PANASItem = Field(..., description="The fifteenth item of the PANAS questionnaire.")
    item16: PANASItem = Field(..., description="The sixteenth item of the PANAS questionnaire.")
    item17: PANASItem = Field(..., description="The seventeenth item of the PANAS questionnaire.")
    item18: PANASItem = Field(..., description="The eighteenth item of the PANAS questionnaire.")
    item19: PANASItem = Field(..., description="The nineteenth item of the PANAS questionnaire.")
    item20: PANASItem = Field(..., description="The twentieth item of the PANAS questionnaire.")


class EmotionOMessage(Message):
    prompt: str = pydantic.Field(
        description="The prompt to be sent to the model. It should contain the scenario and the questions."
    )


class EmotionObservation(Observation[EmotionOMessage]):
    def format_as_prompt_text(self) -> str:
        return self.messages[0].prompt + "\nMake sure to provide a rating for all 20 emotions."


class EmotionDecisionResult(pydantic.BaseModel):
    key: str = Field(description="The key of the decision result.")
    scenario: str = Field(description="The scenario of the decision result.")
    decisions: list[int] = Field(description="The decisions made by the agent, represented as a list of integers.")


class AggregatedEmotionResult(pydantic.BaseModel):
    data: pd.DataFrame = Field(description="The data of the aggregated results.")

    model_config = {"arbitrary_types_allowed": True}


class EmotionBenchEnv(Environment):
    def __init__(self, key: str, scenario: str, prompt: str, num_agents: int, max_trial_steps: int):
        self._num_agents = num_agents
        self.max_trial_steps = max_trial_steps
        self.key = key
        self.scenario = scenario
        self.prompt = prompt
        self.complete = False
        self.current_step = 0
        self.decisions: list[int] = []

    def num_agents(self) -> int:
        return self._num_agents

    def get_default_agent_configs(self) -> list[dict] | None:
        return None

    def done(self) -> bool:
        return self.current_step >= self.max_trial_steps or self.complete

    async def _get_observations(self) -> dict[int, Observation]:
        if self.done():
            return {}
        observations: dict[int, Observation] = {}
        messages: list[EmotionOMessage] = []
        messages.append(
            EmotionOMessage(
                time=self.current_step,
                src_agent_id=None,
                dst_agent_id=0,
                prompt=self.prompt,
            )
        )
        observations[0] = EmotionObservation(
            agent_id=0,
            messages=messages,
            response_type=PANASResponse,
        )
        return observations

    async def reset(self) -> dict[int, Observation]:
        self.current_step = 0
        return await self._get_observations()

    async def step(
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
    ) -> dict[int, Observation]:
        action = responses[0]
        if isinstance(action, PANASResponse):
            self.decisions = [getattr(action, f"item{i}").score for i in range(1, 21)]
            self.complete = True
        self.current_step += 1
        return await self._get_observations()

    def get_result(self) -> EmotionDecisionResult:
        return EmotionDecisionResult(
            key=self.key,
            scenario=self.scenario,
            decisions=self.decisions,
        )


class EmotionBenchTask(Task):
    def __init__(
        self,
        args: DictConfig,
        max_trial_steps: int = 1,
    ):
        self._num_agents = 1
        random.seed(args.seed)
        self.args = args
        self.max_trial_steps = max_trial_steps
        self.questionnaire = self.get_questionnaire(args.questionnaire)
        self.questionnaire_name = self.questionnaire["name"]
        self.result_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.result_dir, exist_ok=True)
        self.args.scenarios_file = f"{self.result_dir}/{self.questionnaire_name}-situations.csv"
        self.args.testing_file = f"{self.result_dir}/{self.questionnaire_name}-testing_base.csv"
        if os.path.exists(self.args.scenarios_file):
            os.remove(self.args.scenarios_file)
        if os.path.exists(self.args.testing_file):
            os.remove(self.args.testing_file)
        self.generate_testfile(self.questionnaire, self.args)
        testing_df = pd.read_csv(self.args.testing_file)
        headers = testing_df.columns.tolist()
        self.questions_list = {
            f"order-{header[-1]}": "\n".join(testing_df[header].astype(str))
            for header in headers
            if header.startswith("question")
        }
        self.testing_list = [
            {"key": header, "scenario": testing_df[header].iloc[0]}
            for header in headers
            if not header.startswith(("question", "order"))
        ]

    async def iterate_environments(
        self,
    ) -> AsyncIterator[Environment[EmotionDecisionResult]]:
        for i, test_case in enumerate(self.testing_list):
            print(f"Creating environment {i + 1}/{len(self.testing_list)}")
            order_key = test_case["key"].split("_")[-1]
            prompt = self.questions_list[order_key].replace("SCENARIO", test_case["scenario"])
            yield EmotionBenchEnv(
                test_case["key"],
                test_case["scenario"],
                prompt,
                self._num_agents,
                self.max_trial_steps,
            )

    def aggregate_results(self, results: Sequence[EmotionDecisionResult]) -> AggregatedEmotionResult:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        results_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.result_dir, timestamp)
        os.makedirs(results_directory, exist_ok=True)
        output_df_path = os.path.join(results_directory, f"{self.questionnaire_name}-testing.csv")
        shutil.copy(self.args.testing_file, output_df_path)

        output_df = pd.read_csv(output_df_path)
        for result in results:
            output_df[result.key] = [result.scenario] + result.decisions
        output_df.to_csv(output_df_path, index=False)

        self.analysis_results(self.questionnaire, self.args, output_df_path, results_directory)

        return AggregatedEmotionResult(
            data=output_df,
        )

    def get_scenarios(self) -> dict[str, Any]:
        try:
            with open(os.path.join(os.path.dirname(__file__), "situations.json")) as dataset:
                scenarios: dict[str, Any] = json.load(dataset)
        except FileNotFoundError:
            raise FileNotFoundError("'situations.json' file not exist.")

        return scenarios

    def generate_scenarios(self, scenario_file: str, select_time: int, target_emotions: list[str]) -> pd.DataFrame:
        scenarios = self.get_scenarios()
        headers_list = []
        scenarios_list = []

        for emotion in scenarios["emotions"]:
            emotion_name = emotion["name"]

            if ("ALL" not in target_emotions) and (emotion_name not in target_emotions):
                continue

            for index, factor in enumerate(emotion["factors"]):
                headers_list.append(f"{emotion_name}-{index}")
                if select_time > len(factor["scenarios"]):
                    selected_scenarios = factor["scenarios"]
                else:
                    selected_scenarios = random.sample(factor["scenarios"], select_time)
                scenarios_list.append([factor["name"]] + selected_scenarios)

        df_list = [pd.DataFrame(data, columns=[header]) for header, data in zip(headers_list, scenarios_list)]
        output_df = pd.concat(df_list, axis=1)
        output_df.to_csv(scenario_file, index=False)

        return output_df

    def generate_testfile(self, questionnaire: dict, args: DictConfig) -> None:
        scenarios_csv = args.scenarios_file
        output_csv = args.testing_file

        test_times = args.test_count
        default_shuffle_times = args.default_shuffle_count
        emotion_shuffle_times = args.emotion_shuffle_count
        shuffle_times = max(default_shuffle_times, emotion_shuffle_times)

        # Extract the scenarios file
        emotions_selection = args.emotion
        select_times = args.select_count
        if emotions_selection != "Customize":
            scenarios_df = self.generate_scenarios(scenarios_csv, select_times, emotions_selection.split(","))
        else:
            try:
                scenarios_df = pd.read_csv(scenarios_csv)
            except FileNotFoundError:
                raise FileNotFoundError("'situations.csv' file not exist.")

        output_df = pd.DataFrame()
        questions_list = questionnaire["questions"]
        question_num = len(list(questions_list.keys()))
        headers_list = []
        data_list = []

        for shuffle_count in range(shuffle_times + 1):
            question_indices = list(questions_list.keys())

            # Shuffle the question indices
            if shuffle_count != 0:
                random.shuffle(question_indices)
            questions = [f"{index}. {questions_list[question]}" for index, question in enumerate(question_indices, 1)]

            # Append questions column
            headers_list.append(f"question-{shuffle_count}")
            data_list.append([f"Prompt: {questionnaire['prompt']}"] + questions)

            # Append order column
            headers_list.append(f"order-{shuffle_count}")
            data_list.append([questionnaire["prompt"]] + question_indices)

        output_df.to_csv(output_csv, index=False)

        # For each shuffled order,
        for shuffle_count in range(default_shuffle_times + 1):
            for test_count in range(test_times):
                headers_list.append(f"General_test-{test_count}_order-{shuffle_count}")
                data_list.append([" "] + [""] * question_num)

        # For each scenario, create a new column in output_df for each question order
        headers = scenarios_df.columns.tolist()
        for factor in headers:
            for s_index, scenario in enumerate(scenarios_df[factor].iloc[1:].dropna().astype(str)):
                for shuffle_count in range(emotion_shuffle_times + 1):
                    for test_count in range(test_times):
                        # Append scenario column
                        headers_list.append(f"{factor}_scenario-{s_index}_test-{test_count}_order-{shuffle_count}")
                        data_list.append(
                            [f'Imagine you are the protagonist in the scenario: "{scenario}"'] + [""] * question_num
                        )

        # Create a DataFrame from the data_list and new_header_list, and save it to a CSV file
        data_dict = {header: data for header, data in zip(headers_list, data_list)}
        output_df = pd.DataFrame(data_dict)
        output_df.to_csv(output_csv, index=False)

    def get_questionnaire(self, questionnaire_name: str) -> dict[str, Any]:
        try:
            with open(os.path.join(os.path.dirname(__file__), "questionnaires.json")) as dataset:
                data = json.load(dataset)
        except FileNotFoundError:
            raise FileNotFoundError("'questionnaires.json' file not exist.")

        # Matching by questionnaire_name in dataset
        questionnaire: dict[str, Any] | None = None
        for item in data:
            if item["name"] == questionnaire_name:
                questionnaire = item

        if questionnaire is None:
            raise ValueError("Questionnaire not found.")

        return questionnaire

    def convert_data(self, questionnaire: dict[str, Any], scenarios_csv: str, testing_csv: str) -> dict[str, Any]:
        try:
            # Read scenarios_csv to extract all headers
            scenarios_df = pd.read_csv(scenarios_csv)
            scenarios_headers = scenarios_df.columns.tolist()

            # Extract unique emotions from the column headers
            emotions_list = []
            for h in scenarios_headers:
                if h.split("_")[0].split("-")[0] not in emotions_list:
                    emotions_list.append(h.split("_")[0].split("-")[0])

            # Create a dictionary of factors with the headers as keys and names as values (first row of the DataFrame)
            factors_list = {h: scenarios_df[h].iloc[0] for h in scenarios_headers}

        except Exception as e:
            raise ValueError(f"{scenarios_csv} has problems: {str(e)}")

        try:
            # Read testing_csv file to extract all headers
            testing_df = pd.read_csv(testing_csv)
            testing_headers = testing_df.columns.tolist()

            # Extract all questions orders
            orders_list = {
                h: testing_df[h].iloc[1:].astype(int).tolist() for h in testing_headers if h.startswith("order")
            }

        except Exception as e:
            raise ValueError(f"{testing_csv} has problems: {str(e)}")

        # Store the tested data into a dictionary with proper mapping
        tested_data = {}
        for h in testing_headers:
            if not h.startswith(("question", "order")):
                try:
                    tested_data[h] = {
                        # Map the data to its corresponding order
                        i: questionnaire["scale"] - int(val) if i in questionnaire["reverse"] else int(val)
                        for i, val in zip(orders_list[h.split("_")[-1]], testing_df[h].iloc[1:].tolist())
                    }
                except ValueError:
                    raise ValueError(
                        f'Error in {testing_csv}: Some cells in column "{h}" cannot be converted to integers.'
                    )

        # Create a dictionary to store the mapped data according to the corresponding emotion and factor
        organized_data: dict[str, Any] = {}
        organized_data = {
            emotion: {
                factor: {"factor_name": factors_list[factor], "data": []}
                for factor in factors_list
                if factor.startswith(emotion)
            }
            for emotion in emotions_list
        }
        organized_data["General"] = []

        # Organize the data into the mapped_data dictionary based on emotions and factors
        for key in tested_data:
            emotion = key.split("_")[0].split("-")[0]
            if emotion == "General":
                # If the emotion is 'General', add the data directly to the 'General' category
                organized_data["General"].append(tested_data[key])
            else:
                # Add other data to the corresponding emotion and factor category based on their keys
                factor = key.split("_")[0]
                organized_data[emotion][factor]["data"].append(tested_data[key])

        return organized_data

    def compute_statistics(self, questionnaire: dict, data_list: list) -> tuple[list, list]:
        cat_list = []
        results = []

        for cat in questionnaire["categories"]:
            scores_list = []

            for data in data_list:
                scores = []
                for key in data:
                    if key in cat["cat_questions"]:
                        scores.append(data[key])

                # Getting the computation mode (SUM or AVG)
                if questionnaire["compute_mode"] == "SUM":
                    scores_list.append(sum(scores))
                elif questionnaire["compute_mode"] == "SUM*2":
                    scores_list.append(sum(scores) * 2)
                else:
                    scores_list.append(mean(scores))

            results.append([mean(scores_list), stdev(scores_list), len(scores_list)])
            cat_list.append(cat["cat_name"])

        return results, cat_list  # ([mean, std, size], cat_list)

    def hypothesis_testing(
        self,
        result1: list[list[float]],
        result2: list[list[float]],
        cat_list: list[str],
        significance_level: float,
        title: str,
    ) -> tuple[str, str]:
        output_list = f"| {title} |"
        output_text = f"### {title}\n"

        for i, cat_name in enumerate(cat_list):
            output_text += f"\n##### {cat_name}"

            # Extract the mean, std and size for both data sets
            mean1, std1, n1 = result1[i]
            mean2, std2, n2 = result2[i]
            # Add an epsilon to prevent the zero standard deviarion
            epsilon = 1e-8
            std1 += epsilon
            std2 += epsilon

            output_text += "\n- **Statistic**:\n"
            output_text += f"Corresponding Factor:\tmean1 = {mean1:.1f},\tstd1 = {std1:.1f},\tn1 = {n1}\n"
            output_text += f"Default:\tmean2 = {mean2:.1f},\tstd2 = {std2:.1f},\tn2 = {n2}\n"

            # Perform F-test
            output_text += "\n- **F-Test:**\n\n"

            if std1 > std2:
                f_value = std1**2 / std2**2
                df1, df2 = n1 - 1, n2 - 1
            else:
                f_value = std2**2 / std1**2
                df1, df2 = n2 - 1, n1 - 1

            p_value = (1 - stats.f.cdf(f_value, df1, df2)) * 2
            equal_var = True if p_value > significance_level else False

            output_text += f"\tf-value = {f_value:.4f}\t($df_1$ = {df1}, $df_2$ = {df2})\n\n"
            output_text += f"\tp-value = {p_value:.4f}\t(two-tailed test)\n\n"
            output_text += "\tNull hypothesis $H_0$ ($s_1^2$ = $s_2^2$): "

            if p_value > significance_level:
                output_text += (
                    f"\tSince p-value ({p_value:.4f}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n"
                )
                output_text += (
                    "\t**Conclusion ($s_1^2$ = $s_2^2$):** The variance of LLM's average "
                    "responses in this factor is statistically equal to the variance of general.\n\n"
                )
            else:
                output_text += f"\tSince p-value ({p_value:.4f}) < α ({significance_level}), $H_0$ is rejected.\n\n"
                output_text += (
                    "\t**Conclusion ($s_1^2$ ≠ $s_2^2$):** The variance of LLM's average "
                    "responses in this factor is statistically unequal to the variance of general.\n\n"
                )

            # Performing T-test
            output_text += (
                "- **Two Sample T-Test (Equal Variance):**\n\n"
                if equal_var
                else "- **Two Sample T-test (Welch's T-Test):**\n\n"
            )

            df = (
                n1 + n2 - 2
                if equal_var
                else ((std1**2 / n1 + std2**2 / n2) ** 2)
                / ((std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1))
            )
            t_value, p_value = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=equal_var)

            output_text += f"\tt-value = {t_value:.4f}\t($df$ = {df:.1f})\n\n"
            output_text += f"\tp-value = {p_value:.4f}\t(two-tailed test)\n\n"

            output_text += "\tNull hypothesis $H_0$ ($µ_1$ = $µ_2$): "
            if p_value > significance_level:
                output_text += (
                    f"\tSince p-value ({p_value:.4f}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n"
                )
                output_text += (
                    "\t**Conclusion ($µ_1$ = $µ_2$):** The average of LLM's responses in this "
                    "factor is assumed to be equal to the average of general.\n\n"
                )

                output_list += f""" $-$ ({"+" if (mean1 - mean2) > 0 else ""}{(mean1 - mean2):.1f}) |"""

            else:
                output_text += f"Since p-value ({p_value:.4f}) < α ({significance_level}), $H_0$ is rejected.\n\n"
                if t_value > 0:
                    output_text += "\tAlternative hypothesis $H_1$ ($µ_1$ > $µ_2$): "
                    output_text += (
                        f"\tSince p-value ({(1 - p_value / 2):.1f}) > α ({significance_level}), "
                        f"$H_1$ cannot be rejected.\n\n"
                    )
                    output_text += (
                        "\t**Conclusion ($µ_1$ > $µ_2$):** The average of LLM's responses in "
                        "this factor is assumed to be larger than the average of general.\n\n"
                    )
                    output_list += f" $\\uparrow$ (+{(mean1 - mean2):.1f}) |"

                else:
                    output_text += "\tAlternative hypothesis $H_1$ ($µ_1$ < $µ_2$): "
                    output_text += (
                        f"\tSince p-value ({(1 - p_value / 2):.1f}) > α ({significance_level}), "
                        f"$H_1$ cannot be rejected.\n\n"
                    )
                    output_text += (
                        "\t**Conclusion ($µ_1$ < $µ_2$):** The average of LLM's responses in "
                        "this factor is assumed to be smaller than the average of general emotion.\n\n"
                    )
                    output_list += f" $\\downarrow$ ({(mean1 - mean2):.1f}) |"

        output_list += f" {result1[0][2]} |\n"
        return (output_text, output_list)

    def analysis_results(
        self, questionnaire: dict[str, Any], args: DictConfig, output_df_path: str, results_directory: str
    ) -> None:
        scenarios_csv = args.scenarios_file
        testing_csv = output_df_path
        significance_level = args.significance_level

        overall_list = "# PANAS Results Analysis\n"
        markdown_output = ""  # overall markdown output text
        overall_data = []

        data = self.convert_data(questionnaire, scenarios_csv, testing_csv)

        general_results, cat_list = self.compute_statistics(questionnaire, data["General"])

        overall_list += "| Emotions | " + " | ".join(cat_list) + " | N |\n"
        overall_list += "| :---: |" + " | ".join([":---:" for i in cat_list]) + " | :---: |\n"
        overall_list += (
            "| Default |"
            + " | ".join([f"{r[0]:.1f} $\\pm$ {r[1]:.1f}" for r in general_results])
            + f" | {general_results[0][2]} |\n"
        )

        # Analyze the results for each emotion
        for emotion in data:
            if emotion == "General":
                continue

            emotion_output = ""  # markdown output text for the current emotion

            emotion_list = f"## {emotion}\n"
            emotion_list += "| Factors | " + " | ".join(cat_list) + " | N |\n"
            emotion_list += "| :---: |" + " | ".join([":---:" for i in cat_list]) + " | :---: |\n"
            emotion_list += (
                "| Default |"
                + " | ".join([f"{r[0]:.1f} $\\pm$ {r[1]:.1f}" for r in general_results])
                + f" | {general_results[0][2]} |\n"
            )

            emotion_data = []  # the data that belongs to the current emotion

            # Analyze the results for each factor
            for factor in data[emotion]:
                emotion_data += data[emotion][factor]["data"]
                overall_data += data[emotion][factor]["data"]
                results, _ = self.compute_statistics(questionnaire, data[emotion][factor]["data"])
                text_msg, list_msg = self.hypothesis_testing(
                    results,
                    general_results,
                    cat_list,
                    significance_level,
                    data[emotion][factor]["factor_name"],
                )
                emotion_output += text_msg
                emotion_list += list_msg

            results, _ = self.compute_statistics(questionnaire, emotion_data)
            text_msg, list_msg = self.hypothesis_testing(
                results, general_results, cat_list, significance_level, "Overall"
            )
            emotion_output += text_msg
            emotion_list += list_msg
            overall_list += list_msg.replace("Overall", emotion)
            markdown_output += emotion_list + "\n" + emotion_output

        markdown_output += "## Emotions Overall\n"
        results, _ = self.compute_statistics(questionnaire, overall_data)
        text_msg, list_msg = self.hypothesis_testing(results, general_results, cat_list, significance_level, "Overall")
        markdown_output += text_msg
        overall_list += list_msg

        with open(f"{results_directory}/emotion.md", "w", encoding="utf-8") as f:
            f.write(overall_list + "\n\n" + markdown_output)
