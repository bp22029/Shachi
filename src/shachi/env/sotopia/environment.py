from collections.abc import AsyncIterator, Sequence
from typing import Literal, cast

import pydantic
import sotopia.messages
import sotopia.samplers
from sotopia.agents import Agents, BaseAgent
from sotopia.envs import ParallelSotopiaEnv
from sotopia.generation_utils.output_parsers import PydanticOutputParser
from sotopia.messages import ActionType, AgentAction

from shachi import base

from . import _generate, _server
from ._server import SotopiaState


class SotopiaMessage(base.Message):
    content: str = pydantic.Field(...)
    original: sotopia.messages.Message = pydantic.Field(...)


class SotopiaObservation(base.Observation[base.Message]):
    # contains the full history (not only the latest turn message)
    full_inbox: list[sotopia.messages.Observation] = pydantic.Field()
    turn_number: int = pydantic.Field()
    action_types: list[ActionType] = pydantic.Field()
    agent: str = pydantic.Field()
    goal: str = pydantic.Field()
    script_like: bool = pydantic.Field(default=False)

    def format_as_prompt_text(self) -> str:
        if self.turn_number <= 1:
            return cast(
                str,
                _generate._agenerate_action__pre(
                    model_name="",
                    history="\n".join(f"{y.to_natural_language()}" for y in self.full_inbox),
                    turn_number=self.turn_number,
                    action_types=self.action_types,
                    agent=self.agent,
                    goal=self.goal,
                    script_like=self.script_like,
                ),
            )
        else:
            last_turn_messages = [
                y for y in self.full_inbox if y.turn_number > 0 and y.turn_number >= self.turn_number - 1
            ]
            return "\n".join(f"{y.to_natural_language()}" for y in last_turn_messages)


def _messages_from_state(
    state: SotopiaState, env: ParallelSotopiaEnv, agents: Agents, script_like: bool
) -> dict[int, base.Observation]:
    shachi_observation: dict[int, base.Observation] = {}

    for agent_id, agent_name in enumerate(env.agents):
        agent = agents[agent_name]
        obs: sotopia.messages.Observation = state.environment_messages[agent_name]

        # `LLMAgent.aact`
        agent.recv_message("Environment", obs)
        assert agent._goal is not None, "TODO(shachi): support goal generation (see LLMAgent.aact)"

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            # Later, `AgentAction(action_type="none", argument="")` will be generated
            pass
        else:
            full_inbox = [cast(sotopia.messages.Observation, y) for x, y in agent.inbox]

            messages = [
                cast(
                    base.Message,
                    SotopiaMessage(
                        time=message.turn_number,
                        content=message.to_natural_language(),
                        original=message,
                        src_agent_id=(message.turn_number + 1) % 2,
                        dst_agent_id=(message.turn_number) % 2,
                    ),
                )
                for message in full_inbox
                if message.turn_number > 0 and message.turn_number >= obs.turn_number - 1
            ]

            shachi_observation[agent_id] = SotopiaObservation(
                # Fields of Observation
                agent_id=agent_id,
                messages=messages,
                reward=None,
                response_type=AgentAction,
                # Fields of SotopiaObservation
                full_inbox=[cast(sotopia.messages.Observation, y) for x, y in agent.inbox],
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=agent.agent_name,
                goal=agent.goal,
                script_like=script_like,
            )

    return shachi_observation


class SotopiaReward(pydantic.BaseModel):
    believability: float = pydantic.Field(...)
    relationship: float = pydantic.Field(...)
    knowledge: float = pydantic.Field(...)
    secret: float = pydantic.Field(...)
    social_rules: float = pydantic.Field(...)
    financial_and_material_benefits: float = pydantic.Field(...)
    goal: float = pydantic.Field(...)
    overall_score: float = pydantic.Field(...)


class SotopiaResult(pydantic.BaseModel):
    rewards: list[SotopiaReward] = pydantic.Field(description="The rewards for each agent")


class SotopiaEnvironment(base.Environment[SotopiaResult]):
    def __init__(
        self,
        env: ParallelSotopiaEnv,
        agent_list: list[BaseAgent[sotopia.messages.Observation, AgentAction]],
        omniscient: bool = False,
        script_like: bool = False,
        json_in_script: bool = False,
        tag: str | None = None,
        push_to_db: bool = False,
    ):
        self.agent_list = agent_list
        self.omniscient = omniscient
        self.tag = tag
        self.env = env
        self.script_like = script_like
        self.json_in_script = json_in_script
        self.push_to_db = push_to_db
        self.agents: Agents
        self.state: SotopiaState

    def num_agents(self) -> int:
        return 2  # Original Sotopia actually only supports 2 agents

    def get_default_agent_configs(self) -> list[dict] | None:
        return None  # TODO

    def done(self) -> bool:
        return self.state.done

    def get_result(self) -> SotopiaResult:
        episode_log = _server._arun_one_episode__loop_post(
            env=self.env,
            agent_list=self.agent_list,
            tag=self.tag,
            state=self.state,
        )

        rewards = []
        for agent_id in range(2):
            raw_reward = episode_log.rewards[agent_id]
            if isinstance(raw_reward, tuple):
                rewards.append(SotopiaReward(**raw_reward[1]))
            elif isinstance(raw_reward, float):
                rewards.append(
                    SotopiaReward(
                        believability=0.0,
                        relationship=0.0,
                        knowledge=0.0,
                        secret=0.0,
                        social_rules=0.0,
                        financial_and_material_benefits=0.0,
                        goal=0.0,
                        overall_score=raw_reward,
                    )
                )
            else:
                raise ValueError(f"Unknown reward type: {type(raw_reward)}")

        return SotopiaResult(rewards=rewards)

    async def reset(self) -> dict[int, base.Observation]:
        agents, state = _server._arun_one_episode__pre_loop(
            env=self.env,
            agent_list=self.agent_list,
            omniscient=self.omniscient,
            script_like=self.script_like,
            json_in_script=self.json_in_script,
            tag=self.tag,
            push_to_db=self.push_to_db,
        )
        self.agents = agents
        self.state = state

        return _messages_from_state(self.state, self.env, self.agents, self.script_like)

    async def step(self, responses: dict[int, str | pydantic.BaseModel | None]) -> dict[int, base.Observation]:
        actions = []
        for agent_id, agent_name in enumerate(self.env.agents):
            response = responses.get(agent_id, None)
            if response is None:
                action = AgentAction(action_type="none", argument="")
            elif isinstance(response, AgentAction):
                action = response
            elif isinstance(response, str):
                action = _generate._agenerate__post(response, PydanticOutputParser(pydantic_object=AgentAction))
            else:
                raise ValueError(f"Invalid response type for agent {agent_name}: {type(response)}")
            actions.append(action)

        self.state = await _server._arun_one_episode__loop_body(
            state=self.state,
            env=self.env,
            agents=self.agents,
            script_like=self.script_like,
            actions=actions,
        )

        return _messages_from_state(self.state, self.env, self.agents, self.script_like)


class SotopiaAggregatedResult(pydantic.BaseModel):
    averaged_rewards: list[SotopiaReward] = pydantic.Field(description="The averaged rewards for each agent")
    all_rewards: list[list[SotopiaReward]] = pydantic.Field(description="The rewards for each agent in each episode")


class SotopiaTask(base.Task[SotopiaResult, SotopiaAggregatedResult]):
    def __init__(
        self,
        num_episodes: int = 200,
        env_model: str = "gpt-4-0613",
    ):
        sampler: sotopia.samplers.UniformSampler = sotopia.samplers.UniformSampler()

        # TODO: make them configurable
        action_order: Literal["simutaneous", "round-robin", "random"] = "round-robin"
        model_dict = {"env": env_model, "agent1": "shachi1", "agent2": "shachi2"}
        env_agent_combo_list: list[sotopia.samplers.EnvAgentCombo[sotopia.messages.Observation, AgentAction]] = []
        omniscient: bool = False
        script_like: bool = False
        json_in_script: bool = False
        tag: str | None = None
        push_to_db: bool = False
        using_async: bool = True

        self.action_order = action_order
        self.omniscient = omniscient
        self.script_like = script_like
        self.json_in_script = json_in_script
        self.tag = tag
        self.push_to_db = push_to_db

        env_agent_combo_list = _server._run_async_server(
            num_episodes=num_episodes,
            sampler=sampler,
            action_order=action_order,
            model_dict=model_dict,
            env_agent_combo_list=env_agent_combo_list,
            omniscient=omniscient,
            script_like=script_like,
            json_in_script=json_in_script,
            tag=tag,
            push_to_db=push_to_db,
            using_async=using_async,
        )
        self.env_agent_combo_list = env_agent_combo_list

    async def iterate_environments(self) -> AsyncIterator[base.Environment[SotopiaResult]]:
        for env_agent_combo in self.env_agent_combo_list:
            env, agent_list = env_agent_combo
            yield SotopiaEnvironment(
                env=env,
                agent_list=list(agent_list),
                omniscient=self.omniscient,
                script_like=self.script_like,
                json_in_script=self.json_in_script,
                tag=self.tag,
                push_to_db=self.push_to_db,
            )

    def aggregate_results(self, results: Sequence[SotopiaResult]) -> SotopiaAggregatedResult:
        num_agents = 2  # Original Sotopia actually only supports 2 agents
        num_episodes = len(results)

        all_rewards = [[result.rewards[agent_id] for result in results] for agent_id in range(num_agents)]
        averaged_rewards = [
            SotopiaReward(
                **{
                    k: sum(getattr(r, k) for r in agent_rewards) / num_episodes
                    for k in SotopiaReward.model_fields.keys()
                }
            )
            for agent_rewards in all_rewards
        ]

        return SotopiaAggregatedResult(
            averaged_rewards=averaged_rewards,
            all_rewards=all_rewards,
        )
