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


class DigitMatAgent_using_FunctionCalling(Agent):
    def __init__(
        self,
        model: str = "openai/gpt-3.5-turbo",
        temperature: float = 0,
        max_tokens: int = 1024,
        stop: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = None
        self.max_tokens = max_tokens
        self.stop = stop

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        messages = get_messages_from_observation(observation=observation)

        response_type = observation.response_type
        assert response_type is not None

        plan_completion = await litellm.acompletion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
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
                        "From the answer provided above, extract your final answer\n"
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


class DigitMatAgent_using_StructuredOutput(Agent):
    def __init__(
        self,
        model: str = "openai/gpt-3.5-turbo",
        temperature: float = 0,
        max_tokens: int = 1024,
        stop: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = None
        self.max_tokens = max_tokens
        self.stop = stop

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        messages = get_messages_from_observation(observation=observation)

        response_type = observation.response_type
        assert response_type is not None

        plan_completion = await litellm.acompletion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )

        action_completion = await litellm.acompletion(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"You answer is following.\n\n"
                        f"Answer: {plan_completion.choices[0].message.content}\n"
                        "From the answer provided above, extract your final answer\n"
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
        blocks = re.findall(r"\[([^\]]*)\]", prompt)
        last_block = blocks[-1].strip()
        items = [x for x in re.split(r"[,\s]+", last_block) if x]
        length = len(items)
        sample = {"pred_list": random.sample(range(10), length)}
        json_str = json.dumps(sample)

        response_type = observation.response_type
        assert response_type is not None
        return response_type.model_validate_json(json_str)


def create_agents_functioncalling(
    num_agents: int, model: str, temperature: float, max_tokens: int, stop: str | None = None
) -> list[DigitMatAgent_using_FunctionCalling]:
    return [
        DigitMatAgent_using_FunctionCalling(model=model, temperature=temperature, max_tokens=max_tokens, stop=stop)
        for _ in range(num_agents)
    ]


def create_agents_structuredoutput(
    num_agents: int, model: str, temperature: float, max_tokens: int, stop: str | None = None
) -> list[DigitMatAgent_using_StructuredOutput]:
    return [
        DigitMatAgent_using_StructuredOutput(model=model, temperature=temperature, max_tokens=max_tokens, stop=stop)
        for _ in range(num_agents)
    ]


def create_agents_random(num_agents: int) -> list[RandomAgent]:
    return [RandomAgent() for i in range(num_agents)]
