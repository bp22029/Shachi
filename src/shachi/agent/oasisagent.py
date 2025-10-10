import importlib
import logging

import litellm
import pydantic
from camel.memories import ChatHistoryMemory, MemoryRecord, ScoreBasedContextCreator
from camel.messages import BaseMessage
from camel.types import ModelType, OpenAIBackendRole
from camel.utils import OpenAITokenCounter

from shachi import Agent, BaseMemory, Observation

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_messages_from_observation(observation: Observation) -> list[dict]:
    prompt_payload = observation.format_as_prompt_payload()
    messages = [
        {
            "role": payload_entry.get("role", "user"),  # "role" is default to "user" unless specified in payload entry
            "content": payload_entry["text"],
        }
        for payload_entry in prompt_payload  # Handles possible multiple entries.
    ]
    return messages


class CamelMemory(BaseMemory):
    def __init__(self, window_size: int = 5):
        context_creator = ScoreBasedContextCreator(
            OpenAITokenCounter(ModelType.GPT_3_5_TURBO),
            4096,
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
            formatted = [f"{msg['role']}: {msg['content']}" for msg in messages[0]]
            return "\n".join(formatted)
        return ""

    def clear(self) -> None:
        self.memory.clear()


class SNSAgent_using_FunctionCalling(Agent):
    r"""Social Agent."""

    def __init__(
        self,
        agent_id: int,
        memory: BaseMemory,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.5,
    ):
        self.agent_id = agent_id
        self.model = model
        self.memory = memory
        self.temperature = temperature
        self.system_prompt = None

    def update_config(self, config: dict) -> None:
        self.system_prompt = config.get("system_prompt")

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        memory = self.memory.retrieve()

        messages = get_messages_from_observation(observation=observation)

        response_type = observation.response_type
        assert response_type is not None

        plan_completion = await litellm.acompletion(
            messages=[{"role": "system", "content": f"{self.system_prompt}\n Previous memory: {memory}"}] + messages,
            model=self.model,
            temperature=self.temperature,
        )
        messages.append({"role": "assistant", "content": plan_completion.choices[0].message.content})
        self.memory.add_record(messages)

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

        action_completion = await litellm.acompletion(
            messages=[
                {
                    "role": "system",
                    "content": "You will receive an agent plan. Your task is to parse that plan exactly as it is.",
                },
                {
                    "role": "user",
                    "content": f"Agent plan: {plan_completion.choices[0].message.content}. Parse this plan.",
                },
            ],
            model=self.model,
            temperature=self.temperature,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": response_type.__name__},
            },
        )
        tool_call = action_completion.choices[0].message.tool_calls[0]
        return response_type.model_validate_json(tool_call.function.arguments)


class SNSAgent_using_StructuredOutput(Agent):
    r"""Social Agent."""

    def __init__(
        self,
        agent_id: int,
        memory: BaseMemory,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.5,
    ):
        self.agent_id = agent_id
        self.model = model
        self.memory = memory
        self.temperature = temperature
        self.system_prompt = None

    def update_config(self, config: dict) -> None:
        self.system_prompt = config.get("system_prompt")

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        memory = self.memory.retrieve()
        messages = get_messages_from_observation(observation=observation)

        response_type = observation.response_type
        assert response_type is not None

        plan_completion = await litellm.acompletion(
            messages=[{"role": "system", "content": f"{self.system_prompt}\n Previous memory: {memory}"}] + messages,
            model=self.model,
            temperature=self.temperature,
        )

        messages.append({"role": "assistant", "content": plan_completion.choices[0].message.content})
        self.memory.add_record(messages)

        action_completion = await litellm.acompletion(
            messages=[
                {
                    "role": "system",
                    "content": "You will receive an agent plan. Your task is to parse that plan exactly as it is.",
                },
                {
                    "role": "user",
                    "content": f"Agent plan: {plan_completion.choices[0].message.content}. Parse this plan.",
                },
            ],
            model=self.model,
            temperature=self.temperature,
            response_format=response_type,
        )
        json_response = action_completion.choices[0].message.content
        return response_type.model_validate_json(json_response)


def create_agents_functioncalling(
    num_agents: int,
    model: str,
    temperature: float,
    memory_cls_path: str,
    memory_cls_kwargs: dict,
) -> list[SNSAgent_using_FunctionCalling]:
    module_path, class_name = memory_cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    MemoryClass = getattr(module, class_name)

    agents = []
    for i in range(num_agents):
        memory = MemoryClass(**memory_cls_kwargs)
        agent = SNSAgent_using_FunctionCalling(agent_id=i, model=model, temperature=temperature, memory=memory)
        agents.append(agent)
    return agents


def create_agents_structuredoutput(
    num_agents: int,
    model: str,
    temperature: float,
    memory_cls_path: str,
    memory_cls_kwargs: dict,
) -> list[SNSAgent_using_StructuredOutput]:
    module_path, class_name = memory_cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    MemoryClass = getattr(module, class_name)

    agents = []
    for i in range(num_agents):
        memory = MemoryClass(**memory_cls_kwargs)
        agent = SNSAgent_using_StructuredOutput(agent_id=i, model=model, temperature=temperature, memory=memory)
        agents.append(agent)
    return agents
