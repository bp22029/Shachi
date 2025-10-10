import re
from typing import Any, Literal, TypeVar

import litellm
import pydantic

import shachi

PROMPT_MODE = Literal["single_turn", "multi_turn"]

PARSING_MODE = Literal[
    "none",
    "structured_output",
    "function_calling",
    "two_steps_structured_output",
]
MAX_RETRIES = 10

TResponseType = TypeVar("TResponseType", bound=pydantic.BaseModel)


async def call_llm(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    parsing_mode: PARSING_MODE,
    parsing_model: str | None = "gpt-4.1-mini-2025-04-14",
    response_type: type[TResponseType] | None = None,
) -> str | TResponseType:
    if parsing_mode == "none" or response_type is None:
        completion = await litellm.acompletion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_retries=MAX_RETRIES,
        )
        response_text: str = completion.choices[0].message.content
        return response_text

    elif parsing_mode == "structured_output":
        assert response_type is not None, "response_type is required for structured_output"
        completion = await litellm.acompletion(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format=response_type,
            max_retries=MAX_RETRIES,
        )
        response_text = completion.choices[0].message.content
        response_obj = response_type.model_validate_json(response_text)
        return response_obj

    elif parsing_mode == "function_calling":
        assert response_type is not None, "response_type is required for function_calling"
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
        completion = await litellm.acompletion(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": response_type.__name__},
            },
            max_retries=MAX_RETRIES,
        )
        tool_call = completion.choices[0].message.tool_calls[0]
        tool_args = tool_call.function.arguments

        # The following error occurs when the LLM generates invalid JSON.
        # This is a workaround to remove control characters from the JSON string.
        #   Invalid JSON: control character (\u0000-\u001F) found while parsing a string at line 4 column 0
        # 　[type=json_invalid, input_value='{\n"action_type": "speak...isrupt the order?\n"\n}', input_type=str]
        tool_args = re.sub(r"[\x00-\x1F\x7F]", "", tool_args)

        response_obj = response_type.model_validate_json(tool_args)
        return response_obj

    elif parsing_mode == "two_steps_structured_output":
        if parsing_model is None:
            raise ValueError("parsing_model must be provided for two_steps_structured_output")
        assert response_type is not None, "response_type is required for two_steps_structured_output"

        # First step: generate in a plain text
        completion1 = await litellm.acompletion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_retries=MAX_RETRIES,
        )
        response_text_1 = completion1.choices[0].message.content

        # Second step: parse the plain text into a structured output
        completion2 = await litellm.acompletion(
            messages=[
                {
                    "role": "user",
                    "content": f"""
Based on the text provided below, output JSON. If the input is plain text, 
extract the necessary information while preserving the original wording as much as possible. 
If the input is JSON, output it unchanged, except fix any formatting errors you find.
```
{response_text_1}
```

The JSON should follow the schema below:
```
{response_type.model_json_schema()}
```
""".strip(),
                },
            ],
            model=parsing_model,
            temperature=temperature,
            response_format=response_type,
            max_retries=MAX_RETRIES,
        )
        response_text = completion2.choices[0].message.content
        response_obj = response_type.model_validate_json(response_text)
        return response_obj
    else:
        raise ValueError(f"Unknown parsing mode: {parsing_mode}")


# SotopiaAgentMT is an agent for reproducing Sotopia experiments using multi-turn LLM calls.
# The big difference from the original Sotopia is that this agent manages its own chat history.
# In the original Sotopia, all agents' chat history is concatenated into a single prompt.
class SotopiaAgentMT(shachi.Agent):
    # `gpt-4-0613` and `gpt-3.5-turbo-16k-0613` were used in the original Sotopia paper.
    # Temperature is set to 1.0 in the original Sotopia paper.
    def __init__(
        self,
        model: str,
        parsing_mode: PARSING_MODE,
        temperature: float,
        drop_memory: bool,
    ) -> None:
        assert drop_memory is False, "drop_memory is not supported in SotopiaAgentMT"

        self.model = model
        self.parsing_mode = parsing_mode
        self.temperature = temperature
        self.chat_history: list[dict[Any, Any]] = []

    async def step(self, observation: shachi.Observation) -> str | pydantic.BaseModel | None:
        self.chat_history.append({"role": "user", "content": observation.format_as_prompt_payload()})

        import rich

        rich.print(self.chat_history)
        response = await call_llm(
            messages=self.chat_history,
            model=self.model,
            temperature=self.temperature,
            parsing_mode=self.parsing_mode,
            response_type=observation.response_type,
        )

        if isinstance(response, str):
            self.chat_history.append({"role": "assistant", "content": response})
        elif isinstance(response, pydantic.BaseModel):
            self.chat_history.append({"role": "assistant", "content": response.json()})
        return response


# SotopiaAgentSP is an agent for reproducing Sotopia experiments using single-turn LLM calls.
# This agent will work as an exact reproduction of the Sotopia paper.
class SotopiaAgentST(shachi.Agent):
    # `gpt-4-0613` and `gpt-3.5-turbo-16k-0613` were used in the original Sotopia paper.
    # Temperature is set to 1.0 in the original Sotopia paper.
    def __init__(
        self,
        model: str,
        parsing_mode: PARSING_MODE,
        temperature: float,
        drop_memory: bool,
    ):
        self.model = model
        self.parsing_mode = parsing_mode
        self.temperature = temperature
        self.first_message: str | None = None
        self.drop_memory = drop_memory
        self.history = ""

    async def step(self, observation: shachi.Observation) -> str | pydantic.BaseModel | None:
        last_turn_prompt = observation.format_as_prompt_text()
        if self.first_message is None:
            self.first_message = last_turn_prompt
        else:
            if self.drop_memory:
                # Keep only the last turn in the prompt and drop the rest.
                self.history = last_turn_prompt
            else:
                self.history += "\n" + last_turn_prompt

        total_prompt = self.first_message
        if ".\nYou are at Turn #" in total_prompt:
            prv_turns = [int(n) for n in re.findall(r"Turn #(\d+):", total_prompt + self.history)]
            crr_turn = max(prv_turns) + 1 if prv_turns else 0

            total_prompt = re.sub(
                r".\nYou are at Turn #\d+\.\s*",
                self.history + f".\nYou are at Turn #{crr_turn}.\n",
                total_prompt,
            )
        else:
            total_prompt += "\n" + self.history

        import rich

        rich.print(rich.panel.Panel("Prompt"))
        rich.print(total_prompt)
        response = await call_llm(
            messages=[
                {"role": "user", "content": total_prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            parsing_mode=self.parsing_mode,
            response_type=observation.response_type,
        )

        return response


def create_agents(
    num_agents: int,
    model: str = "openai/gpt-4o",
    prompt_mode: PROMPT_MODE = "single_turn",
    parsing_mode: PARSING_MODE = "two_steps_structured_output",
    temperature: float = 1.0,
    drop_memory: bool = False,
) -> dict[int, shachi.Agent]:
    if prompt_mode == "single_turn":
        return {
            agent_id: SotopiaAgentST(
                model=model,
                parsing_mode=parsing_mode,
                temperature=temperature,
                drop_memory=drop_memory,
            )
            for agent_id in range(num_agents)
        }
    elif prompt_mode == "multi_turn":
        return {
            agent_id: SotopiaAgentMT(
                model=model,
                parsing_mode=parsing_mode,
                temperature=temperature,
                drop_memory=drop_memory,
            )
            for agent_id in range(num_agents)
        }
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
