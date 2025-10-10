import os
import sys
from collections.abc import AsyncIterator, Sequence

import pydantic
from pydantic import BaseModel

from shachi import Environment, Message, Observation, Task

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(SCRIPT_DIR)

DEP_DIR = os.path.join(SCRIPT_DIR, "dep")
TOPICS_DIR = os.path.join(DEP_DIR, "topics")

available_scenarios = ["onlineforum", "pew", "twitter"]
scenario_to_topics_file = {
    scenario: os.path.join(TOPICS_DIR, f"{scenario}topics.txt") for scenario in available_scenarios
}


class Trait(BaseModel):
    persona: str
    persona_repr: str | None = (
        None  # Part of the key to identify traits. Default to persona if persona_repr is not specified.
    )
    persona_type: str | None = None  # If specified, identifies the type of persona
    topic: str
    scenario: str


def load_traits(scenario: str, num_per_gen: int = 10) -> list[Trait]:
    assert scenario in available_scenarios, f"{scenario=} must be one of {available_scenarios}"
    topics_file = scenario_to_topics_file[scenario]
    if scenario == "onlineforum":
        with open(topics_file) as f:
            topic_list = [line.rstrip("\n") for line in f]
        gender_list = ["woman", "man", "person", "nonbinary"]
        age_list = ["20-year-old", "80-year-old", "40-year-old"]
        ideo_list = ["conservative", "liberal", "moderate"]
        race_list = ["white", "Black", "Asian", "Hispanic", "Middle-Eastern"]
        chars = [age_list, ideo_list, race_list, gender_list]

        traits = []
        for topic in topic_list:
            for char_list in chars:
                for char in char_list:
                    persona = char
                    if "man" not in persona and "person" not in persona:
                        persona += " person"
                    for _repeat in range(num_per_gen):
                        trait = Trait(persona=persona, topic=topic, scenario=scenario)
                        traits.append(trait)
        return traits
    elif scenario == "pew":
        with open(topics_file) as f:
            topic_list = [line.rstrip("\n") for line in f]
        topic_list.append("comment")
        persona_dict = {
            "age": ["20-year-old", "person", "80-year-old", "40-year-old"],
            "ideo": ["conservative", "moderate", "liberal"],
            "race": ["white", "Black", "Asian", "Middle-Eastern", "Hispanic"],
            "gender": ["woman", "man", "nonbinary"],
        }
        traits = []
        for topic in topic_list:
            for char_type, char_list in persona_dict.items():
                for char in char_list:
                    persona_type = char_type
                    persona = char

                    persona_repr = persona
                    if "man" not in persona and "person" not in persona:
                        persona_repr = " person"

                    for _repeat in range(num_per_gen):
                        trait = Trait(
                            persona=persona,
                            persona_repr=persona_repr,
                            persona_type=persona_type,
                            topic=topic,
                            scenario=scenario,
                        )
                        traits.append(trait)
        return traits
    elif scenario == "twitter":
        with open(topics_file) as f:
            topic_list = [line.rstrip("\n") for line in f]
            topic_list.append("comment")
            traits = []
            for topic in topic_list:
                for char in ["Republican user", "Democrat user", "user"]:
                    persona = char
                    for _repeat in range(num_per_gen):
                        trait = Trait(persona=persona, topic=topic, scenario=scenario)
                        traits.append(trait)
        return traits
    else:
        raise ValueError(f"{scenario=} must be one of {available_scenarios}")


def trait_to_prompt(trait: Trait, use_format_instruction: bool = True) -> str:
    format_instruction = (
        "Please complete the following comment, using your best guess without asking for more "
        "context, and starting the completed content right after the prompt without extra "
        "description. Put all your responses in a single line without new lines."
    )

    if trait.scenario == "onlineforum":
        persona = trait.persona
        topic = trait.topic
        prompt = ""
        if use_format_instruction:
            prompt += format_instruction
        if topic == "comment":
            if persona[0] == "A":
                prompt += f"An {persona} posted the following comment to an online forum:"
            else:
                prompt += f"A {persona} posted the following comment to an online forum:"
        else:
            if persona[0] == "A":
                prompt += f"An {persona} posted the following comment on {topic} to an online forum:"
            else:
                prompt += f"A {persona} posted the following comment on {topic} to an online forum:"
        return prompt
    elif trait.scenario == "pew":
        persona = trait.persona
        persona_type = trait.persona_type
        topic = trait.topic

        question = topic

        prompt = ""
        if use_format_instruction:
            prompt += format_instruction

        prompt = (
            "Below you will be asked to provide a short description of your identity and then "
            "answer some questions.\nDescription: "
        )
        if "ideo" == persona_type:
            prompt += f"In politics today, I would describe my political views as {persona}. "
        if "race" == persona_type:
            prompt += f"I am {persona}. "
        if "age" == persona_type:
            prompt += f"I am {persona}. "
        if "gender" == persona_type:
            prompt += f"I identify as a {persona}."
        prompt += f"\nQuestion: {question}"
        prompt += "\nAnswer:"
        return prompt
    elif trait.scenario == "twitter":
        persona = trait.persona
        topic = trait.topic

        prompt = ""
        if use_format_instruction:
            prompt += format_instruction

        if topic == "comment":
            prompt += f"A {persona} posted on Twitter:"
        else:
            if "people" in topic or (topic.islower() and topic[-1] == "s"):
                prompt += f"A {persona} posted on Twitter: {topic} are"
            else:
                prompt += f"A {persona} posted on Twitter: {topic} is"
        return prompt
    else:
        raise ValueError(f"{trait.scenario=} must be one of {available_scenarios}")


class LMCaricatureMessage(Message):
    trait: Trait


class LMCaricatureObservation(Observation):
    messages: list[LMCaricatureMessage]
    # response_type = None by default.

    def format_as_prompt_text(self) -> str:
        assert len(self.messages) == 1
        trait = self.messages[0].trait
        return trait_to_prompt(trait)

    def format_as_prompt_payload(self) -> list[dict]:
        return super().format_as_prompt_payload()


class LMCaricatureRecord(BaseModel):
    scenario: str
    persona: str
    topic: str
    response: str


class LMCaricatureResult(BaseModel):
    records: list[LMCaricatureRecord]


class AggregatedLMCaricatureResult(BaseModel):
    records: list[LMCaricatureRecord]


class LMCaricatureEnv(Environment):
    def __init__(
        self,
        model: str,
        scenario: str,
        num_agents: int,
        num_per_gen: int,
        save_prefix: str,
        save_suffix: str = "",
    ):
        """
        We already know how many questions (which is `len(self.traits)`).
        We will ask agents for ceil( len(self.traits) / num_agents ) rounds for these questions and record them.
        Fake questions would be used to fill the batch so each round all agents get an action.
        Note that we assume agents don't use the memory as each questiosn are individual.
        """

        assert scenario in available_scenarios

        self.model = model
        self.scenario = scenario
        self._num_agents = num_agents
        self.num_per_gen = num_per_gen
        self.save_prefix = save_prefix
        self.save_suffix = save_suffix

    def num_agents(self) -> int:
        return self._num_agents

    def get_default_agent_configs(self) -> list[dict] | None:
        return None

    async def reset(self) -> dict[int, Observation]:
        self._setup()
        return self._get_observations()

    async def step(
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
    ) -> dict[int, Observation]:
        self._tick(responses)
        return self._get_observations()

    def done(self) -> bool:
        return self.i_round >= self.n_rounds

    def _setup(self) -> None:
        self.traits = load_traits(scenario=self.scenario, num_per_gen=self.num_per_gen)
        num_agents = self.num_agents()
        self.n_rounds = (len(self.traits) + num_agents - 1) / num_agents  # ceiling division
        self.i_round = 0
        self.records: list[LMCaricatureRecord | None] = [None for _ in range(len(self.traits))]

        self.save_dir = os.path.join(
            self.save_prefix,
            (
                f"scenario-{self.scenario}-model-{self.model}-num-per-gen-{self.num_per_gen}".replace("/", "-")
                + self.save_suffix  # allow '/' in save_suffix to enable sub dirs, which are useful for multiple runs.
            ),
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.record_file_path = os.path.join(self.save_dir, "records.tsv")
        with open(self.record_file_path, "w") as fout:  # clear the content and write the header
            fout.write("scenario\tpersona\ttopic\tresponse\n")

    def _tick(self, responses: dict[int, str | pydantic.BaseModel | None]) -> None:
        num_agents = self.num_agents()
        for index_agent in range(0, num_agents):
            response = str(responses[index_agent]).replace("\n", " ").replace("\r", " ")
            # ^ we turn response into type `str`, and remove any possible newlines.
            index_trait = self.current_round_index_mapping[index_agent]
            trait = self.current_round_observations[index_agent].messages[0].trait
            if index_trait is not None:
                record = LMCaricatureRecord(
                    scenario=trait.scenario,
                    persona=trait.persona,
                    topic=trait.topic,
                    response=response,
                )
                self.records[index_trait] = record
                with open(self.record_file_path, "a") as fout:  # open the file and append.
                    fout.write(f"{record.scenario}\t{record.persona}\t{record.topic}\t{record.response}\n")

        self.i_round = self.i_round + 1

    def _get_observations(self) -> dict[int, Observation]:
        num_agents = self.num_agents()
        offset = self.i_round * num_agents
        traits = self.traits

        self.current_round_index_mapping = {
            # for current round, the mapping from index_of_agent -> index_of_trait. None if it's overflow.
            index_agent: (offset + index_agent if offset + index_agent < len(traits) else None)
            for index_agent in range(0, num_agents)
        }

        self.current_round_observations = {
            index_agent: LMCaricatureObservation(
                agent_id=index_agent,
                messages=[
                    LMCaricatureMessage(
                        time=self.i_round,
                        src_agent_id=None,
                        dst_agent_id=index_agent,
                        trait=traits[(offset + index_agent) % len(traits)],
                    )
                ],
                response_type=None,  # the agent is expected to return a string
            )
            for index_agent in range(0, num_agents)
        }

        observations = self.current_round_observations
        return observations  # type: ignore[return-value]

    def get_result(self) -> LMCaricatureResult:
        return LMCaricatureResult(records=self.records)  # type: ignore[arg-type]  # for the same reason above.


class LMCaricatureTask(Task):
    def __init__(
        self,
        model: str,
        scenario: str,
        num_agents: int,
        num_per_gen: int,
        save_prefix: str,
        num_parallel: int,
    ):
        self.model = model
        self.scenario = scenario
        self.num_agents = num_agents
        self.num_per_gen = num_per_gen
        self.save_prefix = save_prefix
        self.num_parallel = num_parallel

    async def iterate_environments(self) -> AsyncIterator[Environment[LMCaricatureResult]]:
        for i_env in range(self.num_parallel):
            instance_save_suffix = f"/run{i_env + 1}"
            yield LMCaricatureEnv(
                model=self.model,
                scenario=self.scenario,
                num_agents=self.num_agents,
                num_per_gen=self.num_per_gen,
                save_prefix=self.save_prefix,
                save_suffix=instance_save_suffix,
            )

    def aggregate_results(self, results: Sequence[LMCaricatureResult]) -> AggregatedLMCaricatureResult:
        return AggregatedLMCaricatureResult(records=[record for result in results for record in result.records])
