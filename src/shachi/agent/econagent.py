# mypy: no-warn-unused-ignores
import copy
import json
import logging
import os
from typing import Any

import litellm
import numpy as np
import pydantic

import shachi
from shachi import BaseMemory

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class HistoryMemory(BaseMemory):
    def __init__(self, history_length: int = 100, load_path: str | None = None, save_path: str | None = None):
        self.history_length = history_length
        self.load_path = load_path
        self.save_path = save_path

        self.memory: list[dict[str, str]] = []

    def add_record(self, messages: list[dict[str, str]]) -> None:
        self.memory.extend(messages)

    def retrieve_raw(self) -> list[dict[str, str]]:
        return copy.deepcopy(self.memory)

    def retrieve(self, query: str | None = None) -> str:
        messages = self.memory[-self.history_length :]
        return "\n".join([f"{message['role']}: {message['content']}" for message in messages])

    def clear(self) -> None:
        self.memory = []

    def load_memory(self, agent_id: int | None = None) -> None:
        if self.load_path is not None:
            load_filepath = os.path.join(self.load_path, f"agent_id-{agent_id}.json")
            if os.path.exists(load_filepath):
                logger.info(f"loading memory from {load_filepath}")
                with open(load_filepath) as fin:
                    self.memory = json.load(fin)
            else:
                logger.warning(f"{load_filepath=} does not exist, skip loading memory.")
        else:
            logger.warning(f"{self.load_path=} , skip loading memory.")

    def save_memory(self, agent_id: int | None = None) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            save_filepath = os.path.join(self.save_path, f"agent_id-{agent_id}.json")
            logger.info(f"{save_filepath} , saving memory.")
            with open(save_filepath, "w") as fout:
                json.dump(self.memory, fout)
        else:
            logger.warning(f"{self.save_path=} , skip saving memory.")


def get_messages_from_observation(observation: shachi.Observation) -> list[dict]:
    prompt_payload = observation.format_as_prompt_payload()
    messages = [
        {
            "role": payload_entry.get("role", "user"),  # "role" is default to "user" unless specified in payload entry
            "content": payload_entry["text"],
        }
        for payload_entry in prompt_payload  # Handles possible multiple entries.
    ]
    return messages


gpt_default_model = "gpt-4o-mini"


class EconAgentAgent_using_FunctionCalling(shachi.Agent):
    def __init__(
        self,
        model: str = gpt_default_model,
        temperature: float = 0.0,
        memory_save_path: str | None = None,
        *args: list,
        **kwargs: dict,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.temperature = temperature
        self.memory = HistoryMemory(save_path=memory_save_path)

    async def step(self, observation: shachi.Observation) -> str | pydantic.BaseModel | None:
        _logger_header = f"EconAgentAgent (obj hash={hash(self)}):"
        logger.debug(f"{_logger_header} {observation=}")

        messages = get_messages_from_observation(observation=observation)
        logger.debug(f"{_logger_header} {messages=}")

        if observation.response_type is None:
            raise ValueError(f"{observation.response_type=} must not be None.")

        response_type = observation.response_type
        json_schema = response_type.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_type.__name__,
                    "description": f"Generate a response of type {response_type.__name__}",
                    "parameters": json_schema,
                },
            }
        ]
        logger.debug(f"{_logger_header} {tools=}")

        completion = await litellm.acompletion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": response_type.__name__},
            },  # type: ignore
        )
        logger.debug(f"{_logger_header} {completion=}")

        tool_call = completion.choices[0].message.tool_calls[0]  # type: ignore
        logger.debug(f"{_logger_header} {tool_call=}")

        try:
            response = response_type.model_validate_json(tool_call.function.arguments)
        except Exception:
            response = None
        logger.debug(f"{_logger_header} {response=}")

        try:
            str_response = response.model_dump_json()  # type: ignore
        except Exception:
            str_response = ""
        self.memory.add_record(messages + [{"role": "assistant", "content": str_response}])
        try:
            agent_id = observation.agent_id
        except Exception:
            agent_id = None
        self.memory.save_memory(agent_id=agent_id)

        return response


class EconAgentAgent_using_StructuredOutput(shachi.Agent):
    def __init__(
        self,
        model: str = gpt_default_model,
        temperature: float = 0.0,
        memory_save_path: str | None = None,
        *args: list,
        **kwargs: dict,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.temperature = temperature
        self.memory = HistoryMemory(save_path=memory_save_path)

    async def step(self, observation: shachi.Observation) -> str | pydantic.BaseModel | None:
        _logger_header = f"EconAgentAgent (obj hash={hash(self)}):"
        logger.debug(f"{_logger_header} {observation=}")

        messages = get_messages_from_observation(observation=observation)
        logger.debug(f"{_logger_header} {messages=}")

        if observation.response_type is None:
            raise ValueError(f"{observation.response_type=} must not be None.")

        completion = await litellm.acompletion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            response_format=observation.response_type,
        )
        logger.debug(f"{_logger_header} {completion=}")

        json_response = completion.choices[0].message.content  # type: ignore
        try:
            response = observation.response_type.model_validate_json(json_response)  # type: ignore
        except Exception:
            response = None

        logger.debug(f"{_logger_header} {response=}")

        try:
            str_response = response.model_dump_json()  # type: ignore
        except Exception:
            str_response = ""
        self.memory.add_record(messages + [{"role": "assistant", "content": str_response}])
        try:
            agent_id = observation.agent_id
        except Exception:
            agent_id = None
        self.memory.save_memory(agent_id=agent_id)

        return response


def _make_kwargs_for_create_agent(model: str | None, temperature: float | None, memory_save_path: str | None) -> dict:
    kwargs: dict[str, Any] = dict()
    if model is not None:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature
    if memory_save_path is not None:
        kwargs["memory_save_path"] = memory_save_path
    return kwargs


def create_agents_functioncalling(
    num_agents: int,
    model: str = gpt_default_model,
    temperature: float | None = None,
    memory_save_path: str | None = None,
    *args: list,
    **kwargs: dict,
) -> list[shachi.Agent]:
    call_kwargs = _make_kwargs_for_create_agent(model=model, temperature=temperature, memory_save_path=memory_save_path)
    return [EconAgentAgent_using_FunctionCalling(**call_kwargs) for i in range(num_agents)]


def create_agents_structuredoutput(
    num_agents: int,
    model: str = gpt_default_model,
    temperature: float | None = None,
    memory_save_path: str | None = None,
    *args: list,
    **kwargs: dict,
) -> list[shachi.Agent]:
    call_kwargs = _make_kwargs_for_create_agent(model=model, temperature=temperature, memory_save_path=memory_save_path)
    return [EconAgentAgent_using_StructuredOutput(**call_kwargs) for i in range(num_agents)]


class EconAgentAgentStaticBaseline(shachi.Agent):
    def __init__(
        self,
        method: str,
        beta: float = 0.1,
        gamma: float = 0.1,
        h: float = 1.0,
        seed: int = 42,
        *args: list,
        **kwargs: dict,
    ):
        super().__init__(*args, **kwargs)

        self.method = method
        self.beta = beta
        self.gamma = gamma
        self.h = h

        self.seed = seed

        self.consumption_fun_idx: int | None = None
        self.work_fun_idx: int | None = None

        self.random_rng: np.random.Generator | None = None

    async def step(self, observation: shachi.Observation) -> str | pydantic.BaseModel | None:
        def find_last_message_of_state() -> Any:
            for message in reversed(observation.messages):
                for state in reversed(message.states):
                    type_name = type(state).__name__  # We don't have type variable here. Instead we judge by type name.
                    if type_name == "EconAgentState":
                        return state
            return None

        state = find_last_message_of_state()
        assert state is not None

        consumption_funs = {
            "len": [self.consumption_len],
            "cats": [self.consumption_cats],
            "complex": [self.consumption_len, self.consumption_cats],
        }[self.method.lower()]
        work_funs = [self.work_income_wealth]

        price = state.price
        wealth = state.wealth
        max_l = state.max_l
        skill = state.skill
        max_income = skill * max_l
        last_income = state.last_income
        expected_income = max_l * state.expected_skill
        interest_rate = state.interest_rate

        if self.random_rng is None:
            # ensure different agents have different seeds.
            self.random_rng = np.random.default_rng(
                seed=observation.agent_id * 997 + self.seed
            )  # 997 is a prime number.
        if self.consumption_fun_idx is None:
            self.consumption_fun_idx = self.random_rng.choice(range(len(consumption_funs)))
        assert self.consumption_fun_idx is not None
        if self.work_fun_idx is None:
            self.work_fun_idx = self.random_rng.choice(range(len(work_funs)))
        assert self.work_fun_idx is not None
        work_fun = work_funs[self.work_fun_idx]
        w = work_fun(price, wealth, max_income, last_income, expected_income, interest_rate)
        curr_income = w * max_income
        consumption_fun = consumption_funs[self.consumption_fun_idx]
        c = consumption_fun(price, wealth, curr_income, last_income, interest_rate)

        return observation.response_type(
            work=w, consumption=c, rationale=f"{self.work_fun_idx=} {self.consumption_fun_idx=} "
        )  # type: ignore

    def consumption_len(
        self,
        price: float,
        wealth: float,
        curr_income: float,
        last_income: float,
        interest_rate: float,
    ) -> float:
        beta = self.beta
        c: float = (price / (1e-8 + wealth + curr_income)) ** beta
        c = min(max(c // 0.02, 0), 50)
        c = c * 1.0 / 50  # Project to [0., 1.]
        return c

    def consumption_cats(
        self,
        price: float,
        wealth: float,
        curr_income: float,
        last_income: float,
        interest_rate: float,
    ) -> float:
        h = self.h
        h1 = h / (1 + interest_rate)
        g = curr_income / (last_income + 1e-8) - 1
        d = wealth / (last_income + 1e-8) - h1
        c = 1 + (d - h1 * g) / (1 + g + 1e-8)
        c = min(max(c * curr_income / (wealth + curr_income + 1e-8) // 0.02, 0), 50)
        c = c * 1.0 / 50  # Project to [0., 1.]
        return c

    def work_income_wealth(
        self,
        price: float,
        wealth: float,
        curr_income: float,
        last_income: float,
        expected_income: float,
        interest_rate: float,
    ) -> float:
        gamma = self.gamma
        assert 0.0 <= gamma <= 1.0
        w = int(np.random.uniform() < (curr_income / (wealth * (1 + interest_rate) + 1e-8)) ** gamma)
        # w should be in [0., 1.] as long as 0. <= gamma <= 1.0
        return w


def create_agents_static_baseline(
    num_agents: int,
    method: str = "complex",
    *args: list,
    **kwargs: dict,
) -> list[shachi.Agent]:
    return [EconAgentAgentStaticBaseline(method=method) for i in range(num_agents)]
