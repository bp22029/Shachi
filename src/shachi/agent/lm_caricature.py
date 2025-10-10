import copy
import json
import logging
import os

import litellm

from shachi import Agent, BaseMemory, Observation

logger = logging.getLogger(__name__)

gpt_default_model = "openai/gpt-4o"


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


class LMCaricatureAgent(Agent):
    def __init__(
        self,
        model: str = gpt_default_model,
        temperature: float | None = 0.0,
        max_tokens: int = 256,
        context_memory_load_path: str | None = None,
        *args: list,
        **kwargs: dict,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if context_memory_load_path:
            self.use_context_memory = True
            self.context_memory = HistoryMemory(load_path=context_memory_load_path)
            logger.info(f"{self.use_context_memory=} {context_memory_load_path=}")
        else:
            self.use_context_memory = False

    async def step(self, observation: Observation) -> str | None:
        _logger_header = f"EconAgentAgent (obj hash={hash(self)}):"
        logger.debug(f"{_logger_header} {observation=}")

        messages = get_messages_from_observation(observation=observation)
        logger.debug(f"{_logger_header} {messages=}")

        if self.use_context_memory:
            try:
                agent_id = observation.agent_id
            except Exception:
                agent_id = None

            self.context_memory.load_memory(agent_id=agent_id)
            retrieved_messages = self.context_memory.retrieve_raw()
            logger.info(
                f"{_logger_header} load context memory, assuming {agent_id=}. "
                f"Totally {len(str(retrieved_messages))} chars of memory loaded."
            )
            messages = retrieved_messages + messages

        completion = await litellm.acompletion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            response_format=observation.response_type,
            max_tokens=self.max_tokens,
        )
        logger.debug(f"{_logger_header} {completion=}")

        response: str = completion.choices[0].message.content
        logger.debug(f"{_logger_header} {response=}")
        return response


def create_agents(
    num_agents: int,
    model: str = gpt_default_model,
    temperature: float | None = None,
    max_tokens: int = 256,
    context_memory_load_path: str | None = None,
) -> list[LMCaricatureAgent]:
    return [
        LMCaricatureAgent(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_memory_load_path=context_memory_load_path,
        )
        for i in range(num_agents)
    ]
