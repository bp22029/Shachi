import importlib
import logging

import litellm
import pydantic

from shachi import Agent, BaseMemory, Observation

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


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


class HistoryMemory(BaseMemory):
    def __init__(self, history_length: int = 5):
        self.history_length = history_length
        self.memory: list[dict[str, str]] = []

    def add_record(self, messages: list[dict[str, str]]) -> None:
        self.memory.extend(messages)

    def retrieve(self, query: str | None = None) -> str:
        messages = self.memory[-self.history_length :]
        return "\n".join([f"{message['role']}: {message['content']}" for message in messages])

    def clear(self) -> None:
        self.memory = []


class StockAgent_using_FunctionCalling(Agent):
    def __init__(self, memory: BaseMemory, model: str = "openai/gpt-4o-mini", temperature: float = 0):
        self.model = model
        self.temperature = temperature
        self.memory = memory
        self.system_prompt = None

    def update_config(self, config: dict) -> None:
        self.system_prompt = config.get("system_prompt")

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        memory = self.memory.retrieve()

        messages = get_messages_from_observation(observation=observation)

        response_type = observation.response_type
        assert response_type is not None

        if observation.tools:
            available_tools = observation.tools
            tools_for_llm = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_type.model_json_schema(),
                    },
                }
                for tool in available_tools
            ]

            completion = await litellm.acompletion(
                messages=messages,
                model=self.model,
                tools=tools_for_llm,
                tool_choice="auto",
            )
            assistant_message = completion.choices[0].message

            # If no tool calls, exit
            if not hasattr(assistant_message, "tool_calls") or not assistant_message.tool_calls:
                return assistant_message.content if assistant_message.content is not None else ""

            # Process tool calls
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments

                # Find corresponding tool
                matching_tool = None
                for tool in available_tools:
                    if tool.name == function_name:
                        matching_tool = tool
                        break

                if matching_tool:
                    # Execute tool
                    try:
                        parameters = (
                            matching_tool.parameters_type.model_validate_json(function_args)
                            if isinstance(function_args, str)
                            else matching_tool.parameters_type.model_validate(function_args)
                        )
                        tool_response = matching_tool.fun(parameters)
                        response_text = tool_response.format_as_prompt_text()

                        messages = [
                            {
                                "role": "assistant",
                                "content": response_text,
                            }
                        ] + messages

                        logger.info(f"Tool call: {function_name}")
                        logger.info(f"Arguments: {function_args}")
                        logger.info(f"Response: {response_text}")
                    except Exception as e:
                        logger.error(f"Error occurred during tool execution: {e}")

        plan_completion = await litellm.acompletion(
            messages=[
                {
                    "role": "system",
                    "content": f"Previous memory: {memory}. Your feature: {self.system_prompt}\n ",
                }
            ]
            + messages,
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


class StockAgent_using_StructuredOutput(Agent):
    def __init__(self, memory: BaseMemory, model: str = "openai/gpt-4o-mini", temperature: float = 0):
        self.model = model
        self.temperature = temperature
        self.memory = memory
        self.system_prompt = None

    def update_config(self, config: dict) -> None:
        self.system_prompt = config.get("system_prompt")

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        memory = self.memory.retrieve()

        messages = get_messages_from_observation(observation=observation)

        response_type = observation.response_type
        assert response_type is not None

        if observation.tools:
            available_tools = observation.tools
            tools_for_llm = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_type.model_json_schema(),
                    },
                }
                for tool in available_tools
            ]

            completion = await litellm.acompletion(
                messages=messages,
                model=self.model,
                tools=tools_for_llm,
                tool_choice="auto",
            )
            assistant_message = completion.choices[0].message

            # If no tool calls, exit
            if not hasattr(assistant_message, "tool_calls") or not assistant_message.tool_calls:
                return assistant_message.content if assistant_message.content is not None else ""

            # Process tool calls
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments

                # Find corresponding tool
                matching_tool = None
                for tool in available_tools:
                    if tool.name == function_name:
                        matching_tool = tool
                        break

                if matching_tool:
                    # Execute tool
                    try:
                        parameters = (
                            matching_tool.parameters_type.model_validate_json(function_args)
                            if isinstance(function_args, str)
                            else matching_tool.parameters_type.model_validate(function_args)
                        )
                        tool_response = matching_tool.fun(parameters)
                        response_text = tool_response.format_as_prompt_text()

                        messages = [
                            {
                                "role": "assistant",
                                "content": response_text,
                            }
                        ] + messages

                        logger.info(f"Tool call: {function_name}")
                        logger.info(f"Arguments: {function_args}")
                        logger.info(f"Response: {response_text}")
                    except Exception as e:
                        logger.error(f"Error occurred during tool execution: {e}")

        plan_completion = await litellm.acompletion(
            messages=[
                {
                    "role": "system",
                    "content": f"Previous memory: {memory}. Your feature: {self.system_prompt}\n ",
                }
            ]
            + messages,
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


class NoToolNoMemoryNoConfigStockAgent_using_FunctionCalling(Agent):
    def __init__(self, model: str = "openai/gpt-4o-mini", temperature: float = 0):
        self.model = model
        self.temperature = temperature

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        messages = get_messages_from_observation(observation=observation)

        response_type = observation.response_type
        assert response_type is not None

        plan_completion = await litellm.acompletion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
        )

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


def create_agents_functioncalling(
    num_agents: int,
    model: str,
    temperature: float,
    memory_cls_path: str,
    memory_cls_kwargs: dict,
) -> list[StockAgent_using_FunctionCalling]:
    module_path, class_name = memory_cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    MemoryClass = getattr(module, class_name)

    agents = []
    for _ in range(num_agents):
        memory = MemoryClass(**memory_cls_kwargs)
        agent = StockAgent_using_FunctionCalling(model=model, temperature=temperature, memory=memory)
        agents.append(agent)
    return agents


def create_agents_structuredoutput(
    num_agents: int,
    model: str,
    temperature: float,
    memory_cls_path: str,
    memory_cls_kwargs: dict,
) -> list[StockAgent_using_StructuredOutput]:
    module_path, class_name = memory_cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    MemoryClass = getattr(module, class_name)

    agents = []
    for _ in range(num_agents):
        memory = MemoryClass(**memory_cls_kwargs)
        agent = StockAgent_using_StructuredOutput(model=model, temperature=temperature, memory=memory)
        agents.append(agent)
    return agents


def create_agents_functioncalling_without_component(
    num_agents: int,
    model: str,
    temperature: float,
    memory_cls_path: str,
    memory_cls_kwargs: dict,
) -> list[NoToolNoMemoryNoConfigStockAgent_using_FunctionCalling]:
    agents = []
    for _ in range(num_agents):
        agent = NoToolNoMemoryNoConfigStockAgent_using_FunctionCalling(model=model, temperature=temperature)
        agents.append(agent)
    return agents
