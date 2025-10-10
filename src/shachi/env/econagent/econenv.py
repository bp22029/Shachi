# mypy: no-warn-unused-ignores
from __future__ import annotations

import json
import logging
import os
import pickle as pkl
import re
import sys
from collections import deque
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from time import time
from typing import Annotated

import numpy as np
import pydantic
import yaml
from dateutil.relativedelta import relativedelta
from pydantic import Field

import shachi

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

DEP_MODULE_DIR = os.path.join(SCRIPT_DIR, "dep")
if DEP_MODULE_DIR not in sys.path:
    sys.path.append(DEP_MODULE_DIR)

# ruff: noqa: E402
from ai_economist import foundation  #  type: ignore

RECOURSE_DIR = os.path.join(SCRIPT_DIR, "resource")  # Not an ideal solution.

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103]) * 100 / 12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]


def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r"\s+", " ", document).strip()
    return cleaned


def format_numbers(numbers: list[int | float]) -> str:
    return "[" + ", ".join(f"{num:.2f}" for num in numbers) + "]"


def format_percentages(numbers: list[int | float]) -> str:
    return "[" + ", ".join(f"{num:.2%}" for num in numbers) + "]"


world_start_time = datetime.strptime("2001.01", "%Y.%m")


class EconAgentState(pydantic.BaseModel):
    curr_rates: list[float]
    current_time: str
    skill: float
    wealth: float
    consumption: float
    interest_rate: float
    price: float
    tax_paid: float
    lump_sum: float
    max_l: float
    last_income: float
    expected_skill: float
    name: str
    age: int
    city: str
    job: str
    offer: str
    labor_market_increased_for_skill: bool
    no_consumption_due_to_shortage_of_goods: bool
    tax_model_is_us_federal_single_filer_2018_scaled: bool
    price_is_in_begining: bool
    price_has_increased: bool
    extra_instruction: str

    def make_prompt(self) -> str:
        problem_prompt = f"""
            You're {self.name}, a {self.age}-year-old individual living in {self.city}. 
            As with all Americans, a portion of your monthly income is taxed by the 
            federal government. This taxation system is tiered, income is taxed cumulatively 
            within defined brackets, combined with a redistributive policy: after collection, 
            the government evenly redistributes the tax revenue back to all citizens, 
            irrespective of their earnings.
            Now it's {self.current_time}.
        """
        if self.job == "Unemployment":
            job_prompt = f"""
                In the previous month, you became unemployed and had no income. 
                Now, you are invited to work as a(an) {self.offer} with monthly salary 
                of ${self.skill * self.max_l:.2f}.
            """
        else:
            if self.labor_market_increased_for_skill:
                job_prompt = f"""
                    In the previous month, you worked as a(an) {self.job}. 
                    If you continue working this month, your expected income will 
                    be ${self.skill * self.max_l:.2f}, which is increased compared 
                    to the last month due to the inflation of labor market.
                """
            else:
                job_prompt = f"""
                    In the previous month, you worked as a(an) {self.job}. 
                    If you continue working this month, your expected income will 
                    be ${self.skill * self.max_l:.2f}, which is decreased compared 
                    to the last month due to the deflation of labor market.
                """
        if self.no_consumption_due_to_shortage_of_goods:
            consumption_prompt = """
                        Besides, you had no consumption due to shortage of goods.
                    """
        else:
            consumption_prompt = f"""
                        Besides, your consumption was ${self.consumption:.2f}.
                    """
        if self.tax_model_is_us_federal_single_filer_2018_scaled:
            tax_prompt = f"""
                Your tax deduction amounted to ${self.tax_paid:.2f}. 
                However, as part of the government's redistribution program, 
                you received a credit of ${self.lump_sum:.2f}.
                In this month, the government sets the brackets: {format_numbers(brackets)} 
                and their corresponding rates: {format_numbers(self.curr_rates)}. 
                Income earned within each bracket is taxed only at that bracket's rate.
            """
        else:
            tax_prompt = f"""
                Your tax deduction amounted to ${self.tax_paid:.2f}. 
                However, as part of the government's redistribution program, 
                you received a credit of ${self.lump_sum:.2f}.
                In this month, according to the optimal taxation theory, Saez Tax, 
                the brackets are not changed: {format_numbers(brackets)} but the 
                government has updated corresponding rates: {format_percentages(self.curr_rates)}. 
                Income earned within each bracket is taxed only at that bracket's rate.
            """
        if self.price_is_in_begining:
            price_prompt = f"""
                Meanwhile, in the consumption market, 
                the average price of essential goods is now at ${self.price:.2f}.
            """
        else:
            if self.price_has_increased:
                price_prompt = f"""
                    Meanwhile, inflation has led to a price increase in the consumption market, 
                    with the average price of essential goods now at ${self.price:.2f}.
                """
            else:
                price_prompt = f"""
                    Meanwhile, deflation has led to a price decrease in the consumption market, 
                    with the average price of essential goods now at ${self.price:.2f}.
                """
        job_prompt = prettify_document(job_prompt)
        obs_prompt = f"""
            {problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt}
            Your current savings account balance is ${self.wealth:.2f}. 
            Interest rates, as set by your bank, stand at {self.interest_rate * 100:.2f}%. 
            With all these factors in play, and considering aspects like your living costs, 
            any future aspirations, and the broader economic trends, how is your willingness 
            to work this month? Furthermore, how would you plan your expenditures on essential 
            goods, keeping in mind good price?
        """

        if self.extra_instruction:
            obs_prompt = (
                obs_prompt
                + f"""
            {self.extra_instruction}
        """
            )
        obs_prompt = (
            obs_prompt
            + """
        Please share your decision with two values: 
        'work' (a value between 0 and 1 with intervals of 0.02, indicating the willingness or 
        propensity to work) and 'consumption' (a value between 0 and 1 with intervals of 0.02, 
        indicating the proportion of all your savings and income you intend to spend on essential goods).
        also, please a string value 'rationale' explaining your rationale (reason) of making such decision.
        """
        )
        obs_prompt = prettify_document(obs_prompt)

        return obs_prompt


class EconAgentRecordedResponse(pydantic.BaseModel):
    str_response: str


class EconAgentMessage(shachi.Message):
    states: list[EconAgentState | EconAgentResponse] | None = pydantic.Field(
        default=None,
        description="A list of states for agents to interpret",
    )


class EconAgentObervation(shachi.Observation):
    def format_as_prompt_text(self) -> str:
        payload = self.format_as_prompt_payload()
        return "\n".join([_["content"] for _ in payload])

    def format_as_prompt_payload(self) -> list[dict]:
        # results_from_dialogs = [dialog for message in self.messages for dialog in message.dialogs]

        results_from_states = []
        for message in self.messages:
            for state in message.states:
                if isinstance(state, EconAgentState):
                    result = {"type": "text", "role": "user", "text": f"{state.make_prompt()}"}
                elif isinstance(state, EconAgentResponse):
                    result = {"type": "text", "role": "assistant", "text": f"{state}"}
                else:
                    assert False
                results_from_states.append(result)

        return results_from_states


# WorkingRatio = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
# ConsumptionRatio = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]

# OpenAI seems to be unhappy with constraints here. So instead of use Field(strict=True, ge=0.0, le=1.0)
# We need to deal with it in our code.

WorkingRatio = Annotated[float, Field(strict=True)]
ConsumptionRatio = Annotated[float, Field(strict=True)]


class EconAgentResponse(pydantic.BaseModel):
    work: WorkingRatio = pydantic.Field(description="The ratio of working. This value should be in range of [0.,1.].")
    consumption: ConsumptionRatio = pydantic.Field(
        description="The ratio of consumption. This value should be in range of [0.,1.]."
    )
    rationale: str = pydantic.Field(description="The rationale behind the decision.")


class EconAgentResult(pydantic.BaseModel):
    one_saved_log: dict


class AggregatedEconAgentResult(pydantic.BaseModel):
    all_saved_logs: list[dict]


class EconAgentEnvironment(shachi.Environment):
    def __init__(
        self,
        model: str,
        num_agents: int,
        episode_length: int,
        dialog_len: int = 3,
        max_price_inflation: float = 0.1,
        max_wage_inflation: float = 0.05,
        save_suffix: str = "",
        save_path: str = "./output/econagent",
        extra_instruction_key: str = "0000",
    ):
        config_filepath = os.path.join(RECOURSE_DIR, "config.yaml")
        with open(config_filepath) as f:
            run_configuration = yaml.safe_load(f)
        env_config = run_configuration.get("env")

        env_config["n_agents"] = num_agents
        env_config["episode_length"] = episode_length

        env_config["flatten_masks"] = False
        env_config["flatten_observations"] = False
        env_config["components"][0]["SimpleLabor"]["scale_obs"] = False
        env_config["components"][1]["PeriodicBracketTax"]["scale_obs"] = False
        env_config["components"][3]["SimpleSaving"]["scale_obs"] = False
        env_config["components"][2]["SimpleConsumption"]["max_price_inflation"] = max_price_inflation
        env_config["components"][2]["SimpleConsumption"]["max_wage_inflation"] = max_wage_inflation
        env_config["resource_dir"] = RECOURSE_DIR

        gpt_error = 0
        state_queue: list[deque] = [deque(maxlen=dialog_len) for _ in range(env_config["n_agents"])]
        dialog_queue: list[deque] = [deque(maxlen=dialog_len) for _ in range(env_config["n_agents"])]
        policy_model = "gpt"
        model_str = str(model).replace("/", "-")  # model (name) may contains '/'
        policy_model_save = (
            f"{policy_model}-{model_str}-{dialog_len}-noperception-reflection-1-"
            f"{num_agents}agents-{episode_length}months{save_suffix}"
        )

        self.gpt_error = gpt_error
        self.state_queue = state_queue
        self.dialog_queue = dialog_queue
        self.env_config = env_config
        self.save_path = save_path
        self.policy_model_save = policy_model_save
        self.extra_instruction_key = extra_instruction_key

    def _reset(self) -> None:
        extra_instruction_key = self.extra_instruction_key
        save_path = self.save_path
        policy_model_save = self.policy_model_save
        env_config = self.env_config

        t = time()
        env = foundation.make_env_instance(**env_config)
        obs = env.reset()
        actions: dict = {}
        epi = 0  # current epi

        def load_extra_instruction() -> str:
            with open(os.path.join(RECOURSE_DIR, "data/extra_instructions.json")) as file:
                extra_instructions = json.load(file)
                result: str = extra_instructions.get(extra_instruction_key, "")
                return result

        extra_instruction = load_extra_instruction()

        logger.info(f"Saving location: {save_path}/data/{policy_model_save}")
        if not os.path.exists(f"{save_path}/data/{policy_model_save}"):
            os.makedirs(f"{save_path}/data/{policy_model_save}")
        if not os.path.exists(f"{save_path}/figs/{policy_model_save}"):
            os.makedirs(f"{save_path}/figs/{policy_model_save}")

        self.t = t
        self.env = env
        self.obs = obs
        self.actions = actions
        self.epi = epi
        self.extra_instruction = extra_instruction

    async def reset(self) -> dict[int, EconAgentObervation]:  # type: ignore
        # like https://gymnasium.farama.org/api/env/#gymnasium.Env.reset mentions,
        # reset() should (in the typical use case) be called with a seed after initialization and then never again.
        # So while we support reset for multiple times semantically, it is more than required.

        self._reset()
        return self._get_observations()

    async def step(  # type: ignore
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
    ) -> dict[int, EconAgentObervation]:
        self._step_tick(responses)
        return self._get_observations()

    def num_agents(self) -> int:
        result: int = self.env_config["n_agents"]
        return result

    def done(self) -> bool:
        result: bool = self.epi >= self.env.episode_length
        return result

    def _get_observations(self) -> dict[int, EconAgentObervation]:  # {agent_id -> observation}
        # get shared vars
        state_queue = self.state_queue
        dialog_queue = self.dialog_queue
        env = self.env
        obs = self.obs
        save_path = self.save_path
        policy_model_save = self.policy_model_save
        extra_instruction = self.extra_instruction

        gpt_path = f"{save_path}/data/{policy_model_save}/dialogs"

        # real work

        if not os.path.exists(gpt_path):
            os.makedirs(gpt_path)

        curr_rates = obs["p"]["PeriodicBracketTax-curr_rates"]
        current_time = (world_start_time + relativedelta(months=env.world.timestep)).strftime("%Y.%m")
        for idx in range(env.num_agents):
            this_agent = env.get_agent(str(idx))
            actions = env.dense_log["actions"]
            states = env.dense_log["states"]

            skill = this_agent.state["skill"]
            wealth = this_agent.inventory["Coin"]
            consumption = this_agent.consumption["Coin"]
            interest_rate = env.world.interest_rate[-1]
            price = env.world.price[-1]
            tax_paid = obs["p"][f"p{idx}"]["PeriodicBracketTax-tax_paid"]
            lump_sum = obs["p"][f"p{idx}"]["PeriodicBracketTax-lump_sum"]
            max_l = env._components_dict["SimpleLabor"].num_labor_hours
            last_income = this_agent.income["Coin"]
            expected_skill = this_agent.state["expected skill"]
            name = this_agent.endogenous["name"]
            age = this_agent.endogenous["age"]
            city = this_agent.endogenous["city"]
            job = this_agent.endogenous["job"]
            offer = this_agent.endogenous["offer"]
            labor_market_increased_for_skill = (job != "Unemployment") and (skill >= states[-1][str(idx)]["skill"])
            no_consumption_due_to_shortage_of_goods = (
                (consumption <= 0) and (len(actions) > 0) and (actions[-1].get("SimpleConsumption", 0) > 0)
            )
            tax_model_is_us_federal_single_filer_2018_scaled = (
                env._components_dict["PeriodicBracketTax"].tax_model == "us-federal-single-filer-2018-scaled"
            )
            price_is_in_begining = env.world.timestep == 0
            price_has_increased = not price_is_in_begining and (price >= env.world.price[-2])

            state = EconAgentState(
                curr_rates=curr_rates.tolist(),
                current_time=current_time,
                skill=skill,
                wealth=wealth,
                consumption=consumption,
                interest_rate=interest_rate,
                price=price,
                tax_paid=tax_paid,
                lump_sum=lump_sum,
                max_l=max_l,
                last_income=last_income,
                expected_skill=expected_skill,
                name=name,
                age=age,
                city=city,
                job=job,
                offer=offer,
                labor_market_increased_for_skill=labor_market_increased_for_skill,
                no_consumption_due_to_shortage_of_goods=no_consumption_due_to_shortage_of_goods,
                tax_model_is_us_federal_single_filer_2018_scaled=tax_model_is_us_federal_single_filer_2018_scaled,
                price_is_in_begining=price_is_in_begining,
                price_has_increased=price_has_increased,
                extra_instruction=extra_instruction,
            )
            obs_prompt = state.make_prompt()

            state_queue[idx].append(state)
            dialog_queue[idx].append({"role": "user", "content": obs_prompt})
        del idx  # type: ignore  # False negative

        states = [list(states) for states in state_queue]

        return {
            agent_id: EconAgentObervation(
                agent_id=agent_id,
                messages=[
                    EconAgentMessage(
                        time=env.world.timestep,
                        src_agent_id=None,
                        dst_agent_id=agent_id,
                        states=states[agent_id],
                    )
                ],
                reward=None,
                response_type=EconAgentResponse,
            )
            for agent_id in range(env.num_agents)
        }

    @staticmethod
    def _extract_typed_response(
        response: str | pydantic.BaseModel | None,
    ) -> tuple[EconAgentResponse, bool]:
        fallback_typed_response = EconAgentResponse(
            work=1.0,
            consumption=0.5,
            rationale="",
        )

        def type_response_is_in_valid_range(response: EconAgentResponse) -> bool:
            return (0.0 <= response.work <= 1.0) and (0.0 <= response.consumption <= 1.0)

        if response is None:
            return fallback_typed_response, False

        # EconAgentResponse case
        if isinstance(response, pydantic.BaseModel):
            if not isinstance(response, EconAgentResponse):
                raise ValueError(f"Invalid response type: expected EconAgentResponse, got {type(response)}")
            if type_response_is_in_valid_range(response):
                return response, True
            else:
                return fallback_typed_response, False

        # JSON str response case
        try:
            response = EconAgentResponse.model_validate_json(response)
            if type_response_is_in_valid_range(response):
                return response, True
            else:
                return fallback_typed_response, False
        except pydantic.ValidationError:
            pass

        # Text response case
        response = str(response)
        possible_floats_in_str = re.findall(r"[-+]?(?:\d*\.*\d+)", response)
        work = 1.0
        consumption = 0.5
        success = False
        try:
            # may be out of index when there is no enough floats in response.
            work = float(possible_floats_in_str[0])
            consumption = float(possible_floats_in_str[1])
            success = True
        except OSError:
            pass
        rationale = response

        if not success:
            return fallback_typed_response, False

        response = EconAgentResponse(work=work, consumption=consumption, rationale=rationale)
        if type_response_is_in_valid_range(response):
            return response, True
        else:
            return fallback_typed_response, False

    def _step_tick(self, responses: dict[int, str | pydantic.BaseModel | None]) -> None:
        # get shared vars
        gpt_error = self.gpt_error
        state_queue = self.state_queue
        dialog_queue = self.dialog_queue
        t = self.t
        env = self.env
        obs = self.obs
        save_path = self.save_path
        policy_model_save = self.policy_model_save
        epi = self.epi

        gpt_path = f"{save_path}/data/{policy_model_save}/dialogs"

        # real work
        actions = {}
        for idx in range(env.num_agents):
            response = responses[idx]
            typed_response, success = self._extract_typed_response(response=response)
            gpt_error += int(not success)
            extracted_actions = [typed_response.work, typed_response.consumption]
            extracted_actions[0] = int(np.random.uniform() <= extracted_actions[0])
            extracted_actions[1] /= 0.02
            actions[str(idx)] = extracted_actions
            state_queue[idx].append(typed_response)
            dialog_queue[idx].append({"role": "assistant", "content": f"{typed_response}"})
        actions["p"] = [0]
        for idx, agent_dialog in enumerate(dialog_queue):
            with open(f"""{gpt_path}/{env.get_agent(str(idx)).endogenous["name"]}""", "a") as f:
                for dialog in list(agent_dialog)[-2:]:
                    f.write(f""">>>>>>>>>{dialog["role"]}: {dialog["content"]}\n""")

        obs, _reward, _done, _info = env.step(actions)
        if (epi + 1) % 1 == 0:
            logger.info(f"step {epi + 1} done, cost {time() - t:.1f}s")
            logger.info(f"#errors: {gpt_error}")
            t = time()
        if (epi + 1) % 6 == 0 or epi + 1 == env.episode_length:
            with open(f"{save_path}/data/{policy_model_save}/actions_{epi + 1}.pkl", "wb") as f:
                pkl.dump(actions, f)
            with open(f"{save_path}/data/{policy_model_save}/obs_{epi + 1}.pkl", "wb") as f:
                pkl.dump(obs, f)
            with open(f"{save_path}/data/{policy_model_save}/env_{epi + 1}.pkl", "wb") as f:
                pkl.dump(env, f)
            with open(f"{save_path}/data/{policy_model_save}/dialog_{epi + 1}.pkl", "wb") as f:
                pkl.dump(dialog_queue, f)
            with open(f"{save_path}/data/{policy_model_save}/dense_log_{epi + 1}.pkl", "wb") as f:
                pkl.dump(env.dense_log, f)

        epi += 1

        if epi >= env.episode_length:
            with open(f"{save_path}/data/{policy_model_save}/dense_log.pkl", "wb") as f:
                pkl.dump(env.dense_log, f)

            logger.info(f"#gpt errors: {gpt_error}")

        # write back
        self.epi = epi
        self.gpt_error = gpt_error
        self.obs = obs

    def read_log(self) -> dict:
        env = self.env
        save_path = self.save_path
        policy_model_save = self.policy_model_save

        episode_length = env.episode_length
        log_save_dir = f"{save_path}/data/{policy_model_save}/"

        def load_entity(fmt_str: str) -> dict:
            result = dict()
            for episode in reversed(range(1, episode_length + 1)):
                pkl_filepath = fmt_str.format_map({"log_save_dir": log_save_dir, "episode": episode})
                try:
                    _part = pkl.load(open(pkl_filepath, "rb"))
                    result[episode] = _part
                except Exception:
                    pass
            return result

        # Here, only uncomment entities that is needed.
        # This saves time.
        entities = [
            "actions",
            "obs",
            "env",
            "dialog",
            "dialog4ref",
            "dense_log",
        ]

        return {entity: load_entity(fmt_str="{log_save_dir}/" + entity + "_{episode}.pkl") for entity in entities}

    def get_result(self) -> EconAgentResult:
        return EconAgentResult(one_saved_log=self.read_log())


class EconAgentTask(shachi.Task):
    def __init__(
        self,
        num_parallel: int,
        model: str,
        num_agents: int,
        episode_length: int,
        dialog_len: int = 3,
        max_price_inflation: float = 0.1,
        max_wage_inflation: float = 0.05,
        save_suffix: str = "",
        save_path: str = "./output/econagent",
        extra_instruction_key: str = "0000",
    ):
        self.num_parallel = num_parallel
        self.model = model
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.dialog_len = dialog_len
        self.max_price_inflation = max_price_inflation
        self.max_wage_inflation = max_wage_inflation
        self.save_suffix = save_suffix
        self.save_path = save_path
        self.extra_instruction_key = extra_instruction_key

    async def iterate_environments(self) -> AsyncIterator[shachi.Environment[EconAgentResult]]:
        for i_env in range(self.num_parallel):
            instance_save_suffix = self.save_suffix if i_env == 0 else self.save_suffix + f"_run{i_env + 1}"
            yield EconAgentEnvironment(
                model=self.model,
                num_agents=self.num_agents,
                episode_length=self.episode_length,
                dialog_len=self.dialog_len,
                max_price_inflation=self.max_price_inflation,
                max_wage_inflation=self.max_wage_inflation,
                save_suffix=instance_save_suffix,
                save_path=self.save_path,
                extra_instruction_key=self.extra_instruction_key,
            )

    def aggregate_results(self, results: Sequence[EconAgentResult]) -> AggregatedEconAgentResult:
        return AggregatedEconAgentResult(all_saved_logs=[result.one_saved_log for result in results])
