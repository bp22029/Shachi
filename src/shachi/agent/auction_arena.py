import importlib
import json
import logging
import random
from typing import Any

import litellm
import pydantic
from camel.memories import ChatHistoryMemory, MemoryRecord, ScoreBasedContextCreator
from camel.messages import BaseMessage
from camel.types import ModelType, OpenAIBackendRole
from camel.utils import OpenAITokenCounter
from litellm import ModelResponse  # type: ignore[attr-defined]

from shachi import Agent, BaseMemory, Observation
from shachi.env.auction_arena.observation import BiddingObservation, BidResponse

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


class EmptyMemory(BaseMemory):
    def __init__(self, **kwargs: Any):
        pass

    def add_record(self, messages: list[dict[str, str]]) -> None:
        pass

    def retrieve(self, query: str | None = None) -> str:
        return "No Memory"

    def clear(self) -> None:
        pass


class CamelMemory(BaseMemory):
    def __init__(self, window_size: int = 20, token_limit: int = 100_000) -> None:
        context_creator = ScoreBasedContextCreator(
            OpenAITokenCounter(ModelType.GPT_3_5_TURBO),
            token_limit,
        )
        self.memory = ChatHistoryMemory(context_creator, window_size=window_size)

    def add_record(self, messages: list[dict[str, str]]) -> None:
        for message in messages:
            role_name = message["role"]
            content = message["content"]
            if role_name == "user":
                msg = BaseMessage.make_user_message(
                    role_name=role_name,
                    content=content,
                )
                self.memory.write_record(
                    MemoryRecord(
                        message=msg,
                        role_at_backend=OpenAIBackendRole.USER,
                    )
                )
            elif role_name == "assistant":
                msg = BaseMessage.make_assistant_message(
                    role_name=role_name,
                    content=content,
                )
                self.memory.write_record(
                    MemoryRecord(
                        message=msg,
                        role_at_backend=OpenAIBackendRole.ASSISTANT,
                    )
                )
            else:  # skip system message
                continue

    def retrieve(self, query: str | None = None) -> str:
        messages = self.memory.get_context()
        if messages:
            return "\n".join([message["content"] for message in messages[0]])
        return ""

    def clear(self) -> None:
        self.memory.clear()


class AuctionArenaAgent_using_StructuredOutput(Agent):
    """
    An agent for the Auction Arena environment that uses LiteLLM's
    structured output capabilities (response_format) to generate
    Pydantic model responses based on the observation type.
    """

    def __init__(
        self,
        memory: BaseMemory,
        id: int = 0,
        model: str = "openai/gpt-4o-mini",  # Default model
        parser_model: str | None = None,
        temperature: float = 0.2,  # Slightly higher default temp
    ):
        self.model = model
        if parser_model is not None:
            self.parser_model = parser_model
        else:
            self.parser_model = self.model

        self.id = id
        self.temperature = temperature
        self.memory = memory
        self.system_prompt: str | None = None

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

        memory_context = self.memory.retrieve(query="")
        plan_messages = []
        if self.system_prompt is not None:
            if memory_context.strip() != "":
                system_message = f"{self.system_prompt}\n\nMemory from previous conversations:\n\n{memory_context}"
            else:
                system_message = self.system_prompt
            plan_messages.append({"role": "system", "content": system_message})

        plan_messages.append({"role": "user", "content": prompt_text})

        logging.info(f"Agent calling LLM. Expecting response type: {response_type.__name__}")

        request_debug(plan_messages, self.model)
        plan_completion = await litellm.acompletion(
            messages=plan_messages,
            model=self.model,
            temperature=self.temperature,
        )

        response_str = plan_completion.choices[0].message.content
        response_debug(response_str, parse_json=False)

        self.memory.add_record(plan_messages + [{"role": "assistant", "content": response_str}])
        cost = safe_cost_computation(plan_completion)

        action_messages = [
            {
                "role": "system",
                "content": "You will receive an agent plan. Your task is to parse that plan exactly as it is.",
            },
            {
                "role": "user",
                "content": f"Agent answer: {plan_completion.choices[0].message.content}. Parse this plan.",
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


class AuctionArenaAgent_using_FunctionCalling(Agent):
    """
    An agent for the Auction Arena environment that uses LiteLLM's
    Function Calling to generate Pydantic model responses based on the observation type.
    """

    def __init__(
        self,
        memory: BaseMemory,
        id: int = 0,
        model: str = "openai/gpt-4o-mini",
        parser_model: str | None = None,
        temperature: float = 0.2,
    ):
        self.model = model
        if parser_model is not None:
            self.parser_model = parser_model
        else:
            self.parser_model = self.model

        self.id = id
        self.temperature = temperature
        self.memory = memory
        self.system_prompt: str | None = None

        self.total_api_cost = 0.0

    def update_config(self, config: dict) -> None:
        """Updates agent configuration, e.g., the system prompt."""
        # Allow overriding system prompt from config
        self.system_prompt = config.get("system_prompt", self.system_prompt)
        self.model = config.get("model", self.model)
        self.temperature = config.get("temperature", self.temperature)
        logging.info(
            "Agent config updated. "
            f"System Prompt: '{self.system_prompt[:100] if self.system_prompt is not None else 'None'}...', "
            f"Model: {self.model}, "
            f"Temp: {self.temperature}"
        )

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        """Processes an observation and returns a structured response or None."""
        for message in observation.messages:
            assert message.dst_agent_id == self.id

        prompt_text = observation.format_as_prompt_payload()[0]["text"]

        response_type = observation.response_type
        assert response_type is not None

        memory_context = self.memory.retrieve(query="")
        messages = []
        if self.system_prompt is not None:
            if memory_context.strip() != "":
                system_message = f"{self.system_prompt}\n\nMemory from previous conversations:\n\n{memory_context}"
            else:
                system_message = self.system_prompt
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt_text})

        logging.info("Agent calling LLM.")

        request_debug(messages, self.model)
        plan_completion = await litellm.acompletion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
        )
        response_debug(plan_completion.choices[0].message.content, parse_json=False)

        self.memory.add_record(
            messages + [{"role": "assistant", "content": plan_completion.choices[0].message.content}]
        )
        cost = safe_cost_computation(plan_completion)

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
                "content": f"Agent answer: {plan_completion.choices[0].message.content}. Parse this plan.",
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


class AuctionArenaRandomAgent(Agent):
    def __init__(self, **kwargs: Any):
        pass

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        if isinstance(observation, BiddingObservation):
            return BidResponse(bid_amount=random.randint(0, 10000))
        else:
            return None


def create_agents_structuredoutput(num_agents: int, agents_config: list[dict]) -> list[Agent]:
    assert len(agents_config) == num_agents, (
        f"The number of agents_config ({len(agents_config)}) should match num_agents {num_agents}"
    )

    agents: list[Agent] = []
    for id, agent_cfg in enumerate(agents_config):
        if agent_cfg.get("is_random"):
            agents.append(AuctionArenaRandomAgent(id=id))
            continue

        model = agent_cfg["model"]
        parser_model = agent_cfg["parser_model"]
        temperature = agent_cfg["temperature"]
        memory_cls_name = agent_cfg["memory_cls_name"]
        memory_cls_kwargs = agent_cfg["memory_cls_kwargs"]

        module_path, class_name = memory_cls_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        MemoryClass = getattr(module, class_name)
        agent_memory = MemoryClass(**memory_cls_kwargs)

        agents.append(
            AuctionArenaAgent_using_StructuredOutput(
                id=id,
                model=model,
                parser_model=parser_model,
                temperature=temperature,
                memory=agent_memory,
            )
        )
        logger.info(f"Created agent {id} with model: {model}, parser: {parser_model}, temp: {temperature}")
    return agents


def create_agents_functioncalling(num_agents: int, agents_config: list[dict]) -> list[Agent]:
    assert len(agents_config) == num_agents, (
        f"The number of agents_config ({len(agents_config)}) should match num_agents {num_agents}"
    )
    agents: list[Agent] = []
    for id, agent_cfg in enumerate(agents_config):
        if agent_cfg.get("is_random"):
            agents.append(AuctionArenaRandomAgent(id=id))
            continue

        model = agent_cfg["model"]
        parser_model = agent_cfg["parser_model"]
        temperature = agent_cfg["temperature"]
        memory_cls_name = agent_cfg["memory_cls_name"]
        memory_cls_kwargs = agent_cfg["memory_cls_kwargs"]

        module_path, class_name = memory_cls_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        MemoryClass = getattr(module, class_name)
        agent_memory = MemoryClass(**memory_cls_kwargs)

        agents.append(
            AuctionArenaAgent_using_FunctionCalling(
                id=id,
                model=model,
                parser_model=parser_model,
                temperature=temperature,
                memory=agent_memory,
            )
        )
        logger.info(
            f"Created agent {id} (Function Calling) with model: {model}, parser: {parser_model}, temp: {temperature}"
        )
    return agents
