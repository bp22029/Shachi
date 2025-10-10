import importlib
import json
import logging
import random
from typing import Any

import litellm
import pydantic
from litellm import ModelResponse  # type: ignore[attr-defined]

from shachi import Agent, BaseMemory, Observation
from shachi.env.psychobench.observation import (
    PsychoBenchIntroMessage,
    PsychoBenchObservation,
    PsychoBenchQuestionMessage,
    QuestionnaireAnswer,
    QuestionnaireAnswers,
)

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def safe_cost_computation(completion: ModelResponse) -> float:
    try:
        return litellm.completion_cost(completion)  # type: ignore[attr-defined]
    except Exception:
        logger.warning("Cost calculation failed so the cost is set to be 0, be careful of API cost!")
        return 0.0


def request_debug(messages: list[dict], model_name: str) -> None:
    debug_msgs = []

    debug_msgs.append(f"{model_name} \nREQUEST START" + ("-" * 200))
    msgs = []
    for message in messages:
        msgs.append(f"{message['role']}:\n {message['content']}")
    debug_msgs.append("\n\n".join(msgs))

    debug_msgs.append(f"{model_name} \nREQUEST END" + ("-" * 200))
    logger.debug("\n".join(debug_msgs))


def response_debug(json_response_str: str, parse_json: bool = True) -> None:
    msgs: list[str] = []
    msgs.append("RESPONSE:\n" + "\n".join(msgs))
    if parse_json:
        try:
            for key, val in json.loads(json_response_str).items():
                msgs.append(f"{key}: {val}")
        except Exception:
            pass
    else:
        msgs.append(json_response_str)

    logger.debug("\n".join(msgs))


class PsychoBenchAgent_using_StructuredOutput(Agent):
    """
    An agent for the PschoBench environment that uses LiteLLM's
    structured output capabilities (response_format) to generate
    Pydantic model responses based on the observation type.
    """

    def __init__(
        self,
        memory: BaseMemory,
        id: int = 0,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.2,
        parser_model: str | None = None,
        model_api_base: str | None = None,
    ):
        self.model = model
        self.model_api_base = model_api_base

        if parser_model is not None:
            self.parser_model = parser_model
        else:
            self.parser_model = self.model

        self.id = id
        self.temperature = temperature
        self.memory = memory
        self.system_prompt = None

        self.total_api_cost = 0.0

    def update_config(self, config: dict) -> None:
        """Updates agent configuration, e.g., the system prompt."""
        # Allow overriding system prompt from config
        self.system_prompt = config.get("system_prompt", self.system_prompt)
        self.model = config.get("model", self.model)
        self.temperature = config.get("temperature", self.temperature)
        logging.info(
            "Agent config updated. "
            f"System Prompt: '{(self.system_prompt or 'None')[:100]}...', "
            f"Model: {self.model}, Temp: {self.temperature}"
        )

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        """Processes an observation and returns a structured response or None."""
        for message in observation.messages:
            assert message.dst_agent_id == self.id

        prompt_text = observation.format_as_prompt_payload()[0]["text"]

        response_type = observation.response_type
        assert response_type is not None

        system_message = self.system_prompt

        plan_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_text},
        ]

        logging.info(f"Agent calling LLM. Expecting response type: {response_type.__name__}")

        request_debug(plan_messages, self.model)
        plan_completion = await litellm.acompletion(
            messages=plan_messages,
            model=self.model,
            temperature=self.temperature,
            api_base=self.model_api_base,
        )

        response_str = plan_completion.choices[0].message.content
        response_debug(response_str, parse_json=False)

        cost = safe_cost_computation(plan_completion)

        action_messages = [
            {
                "role": "system",
                "content": "You will receive an agent plan. Your task is to parse that plan exactly as it is.",
            },
            {
                "role": "user",
                "content": (
                    f"User prompt: {prompt_text}\n\n"
                    f"Agent answer: {plan_completion.choices[0].message.content}. \n\n"
                    "Parse this answer."
                ),
            },
        ]
        request_debug(action_messages, self.model)
        action_completion = await litellm.acompletion(
            messages=action_messages,
            model=self.parser_model,
            temperature=self.temperature,
            response_format=response_type,
        )
        response_debug(response_str)
        cost = safe_cost_computation(action_completion)
        self.total_api_cost += cost
        logging.info(f"API cost: ${cost}, Total API cost: ${self.total_api_cost}")

        return response_type.model_validate_json(action_completion.choices[0].message.content)


class PsychoBenchAgent_using_FunctionCalling(Agent):
    """
    An agent for the PsychoBench environment that uses LiteLLM's
    Function Calling to generate Pydantic model responses based on the observation type.
    """

    def __init__(
        self,
        memory: BaseMemory,
        id: int = 0,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.2,
        parser_model: str | None = None,
        model_api_base: str | None = None,
    ):
        self.model = model
        self.model_api_base = model_api_base

        if parser_model is not None:
            self.parser_model = parser_model
        else:
            self.parser_model = self.model

        self.id = id
        self.temperature = temperature
        self.memory = memory
        self.system_prompt = None

        self.total_api_cost = 0.0

    def update_config(self, config: dict) -> None:
        """Updates agent configuration, e.g., the system prompt."""
        # Allow overriding system prompt from config
        self.system_prompt = config.get("system_prompt", self.system_prompt)
        self.model = config.get("model", self.model)
        self.temperature = config.get("temperature", self.temperature)
        logging.info(
            "Agent config updated. "
            f"System Prompt: '{(self.system_prompt or 'None')[:100]}...', "
            f"Model: {self.model}, Temp: {self.temperature}"
        )

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        """Processes an observation and returns a structured response or None."""
        for message in observation.messages:
            assert message.dst_agent_id == self.id

        # --- Prepare Inputs ---
        prompt_text = observation.format_as_prompt_payload()[0]["text"]

        response_type = observation.response_type
        assert response_type is not None

        # --- Construct Messages for LLM ---
        system_message = self.system_prompt

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_text},
        ]

        # --- Call LLM to Generate The Answer ---
        logging.info("Agent calling LLM.")

        # Use response_format for structured output
        request_debug(messages, self.model)
        plan_completion = await litellm.acompletion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            api_base=self.model_api_base,
        )
        response_debug(plan_completion.choices[0].message.content, parse_json=False)

        cost = safe_cost_computation(plan_completion)

        # --- Parse Response with Function Calling---
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_type.__name__,
                    "description": f"Generate a response of type {response_type.__name__}",
                    "parameters": response_type.model_json_schema(),
                },
            }
        ]

        action_messages = [
            {
                "role": "system",
                "content": "You will receive an agent plan. Your task is to parse that plan exactly as it is.",
            },
            {
                "role": "user",
                "content": (
                    f"User prompt: {prompt_text}\n\n"
                    f"Agent answer: {plan_completion.choices[0].message.content}. \n\n"
                    "Parse this answer."
                ),
            },
        ]
        action_completion = await litellm.acompletion(
            messages=action_messages,
            model=self.parser_model,
            temperature=self.temperature,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": response_type.__name__},
            },
        )
        tool_call = action_completion.choices[0].message.tool_calls[0]

        cost += safe_cost_computation(action_completion)
        self.total_api_cost += cost

        logging.info(f"API cost: ${cost}, Total API cost: ${self.total_api_cost}")

        return response_type.model_validate_json(tool_call.function.arguments)


class PsychoBenchRandomAgent(Agent):
    def __init__(self, **kwargs: Any):
        pass

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        assert isinstance(observation, PsychoBenchObservation), f"{type(observation)}"
        answers = []

        messages = observation.messages
        intro_msg, q_msgs = messages[0], messages[1:]
        if not isinstance(intro_msg, PsychoBenchIntroMessage):
            raise RuntimeError(
                "Internal Error: The first message of observation should be PsychoBenchIntroMessage, "
                f"but got type {type(intro_msg)}; {intro_msg}"
            )

        min_score, max_score = intro_msg.min_score, intro_msg.max_score

        for question in q_msgs:
            assert isinstance(question, PsychoBenchQuestionMessage)
            question_key = question.question_key
            ans = random.randint(min_score, max_score)
            answers.append(QuestionnaireAnswer(question_key=question_key, answer=ans))

        return QuestionnaireAnswers(answers=answers)


def create_agents_random(num_agents: int, **kwargs: Any) -> list[PsychoBenchRandomAgent]:
    """Creates agents, each with its own ObjectMemory instance."""
    agents: list[PsychoBenchRandomAgent] = []
    for id in range(num_agents):
        agents.append(PsychoBenchRandomAgent(id=id))
    return agents


def create_agents_structuredoutput(
    num_agents: int,
    model: str,
    model_api_base: str | None,
    parser_model: str | None,
    temperature: float,
    memory_cls_path: str,
    memory_cls_kwargs: dict,
) -> list[PsychoBenchAgent_using_StructuredOutput]:
    """Creates agents, each with its own ObjectMemory instance."""
    agents = []
    for id in range(num_agents):
        module_path, class_name = memory_cls_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        MemoryClass = getattr(module, class_name)
        agent_memory = MemoryClass(**memory_cls_kwargs)
        agents.append(
            PsychoBenchAgent_using_StructuredOutput(
                id=id,
                model=model,
                temperature=temperature,
                memory=agent_memory,
                parser_model=parser_model,
                model_api_base=model_api_base,
            )
        )
    return agents


def create_agents_functioncalling(
    num_agents: int,
    model: str,
    model_api_base: str | None,
    parser_model: str | None,
    temperature: float,
    memory_cls_path: str,
    memory_cls_kwargs: dict,
) -> list[PsychoBenchAgent_using_FunctionCalling]:
    """Creates agents, each with its own ObjectMemory instance."""
    agents = []
    for id in range(num_agents):
        # Create a new memory instance for each agent
        module_path, class_name = memory_cls_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        MemoryClass = getattr(module, class_name)
        agent_memory = MemoryClass(**memory_cls_kwargs)

        agents.append(
            PsychoBenchAgent_using_FunctionCalling(
                id=id,
                model=model,
                temperature=temperature,
                memory=agent_memory,
                parser_model=parser_model,
                model_api_base=model_api_base,
            )
        )
    return agents
