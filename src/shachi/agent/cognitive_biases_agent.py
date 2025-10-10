import json
import logging
import random
import re

import litellm
import pydantic

from shachi import Agent, Observation

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


class CognitiveBiasAgent_using_FunctionCalling(Agent):
    def __init__(
        self,
        model: str = "openai/gpt-3.5-turbo",
        temperature: float = 0,
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = None

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
                    "role": "user",
                    "content": (
                        f"You answer is following.\n\n"
                        f"Answer: {plan_completion.choices[0].message.content}\n"
                        "From the answer provided above, identify the selected actions.\n"
                    ),
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


class CognitiveBiasAgent_using_StructuredOutput(Agent):
    def __init__(
        self,
        model: str = "openai/gpt-3.5-turbo",
        temperature: float = 0,
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = None

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        messages = get_messages_from_observation(observation=observation)

        response_type = observation.response_type
        assert response_type is not None

        plan_completion = await litellm.acompletion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
        )

        action_completion = await litellm.acompletion(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"You answer is following.\n\n"
                        f"Answer: {plan_completion.choices[0].message.content}\n"
                        "From the answer provided above, identify the selected actions.\n"
                    ),
                },
            ],
            model=self.model,
            temperature=self.temperature,
            response_format=response_type,
        )
        json_response = action_completion.choices[0].message.content
        return response_type.model_validate_json(json_response)


class RandomAgent(Agent):
    def __init__(self) -> None:
        pass

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        prompt = observation.format_as_prompt_payload()[0]["text"]
        nums = re.findall(r"Option\s+(\d+):", prompt)
        if not nums:
            raise ValueError("No options found in prompt")
        N = max(map(int, nums))
        sample = {"option": random.randint(0, N)}
        json_str = json.dumps(sample)

        response_type = observation.response_type
        assert response_type is not None
        return response_type.model_validate_json(json_str)


def create_agents_functioncalling(
    num_agents: int, model: str, temperature: float
) -> list[CognitiveBiasAgent_using_FunctionCalling]:
    return [CognitiveBiasAgent_using_FunctionCalling(model=model, temperature=temperature) for _ in range(num_agents)]


def create_agents_structuredoutput(
    num_agents: int, model: str, temperature: float
) -> list[CognitiveBiasAgent_using_StructuredOutput]:
    return [CognitiveBiasAgent_using_StructuredOutput(model=model, temperature=temperature) for _ in range(num_agents)]


def create_agents_random(num_agents: int) -> list[RandomAgent]:
    return [RandomAgent() for i in range(num_agents)]
