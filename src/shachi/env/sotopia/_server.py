from dataclasses import dataclass
from typing import Literal, Sequence, Type

import rich

from sotopia.agents import Agents, BaseAgent, HumanAgent, LLMAgent, RedisAgent, ScriptWritingAgent
from sotopia.database import EpisodeLog, NonStreamingSimulationStatus
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    EpisodeLLMEvaluator,
    EvaluationForTwoAgents,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
)
from sotopia.messages import AgentAction, Message, Observation
from sotopia.samplers import BaseSampler, EnvAgentCombo


@dataclass
class SotopiaState:
    environment_messages: dict[str, Observation]
    messages: list[list[tuple[str, str, Message]]]
    rewards: list[list[float]]
    reasons: list[str]
    info: dict | None
    done: bool = False


def _run_async_server(
    num_episodes: int,
    sampler: BaseSampler[Observation, AgentAction] = BaseSampler(),
    action_order: Literal["simutaneous", "round-robin", "random"] = "round-robin",
    model_dict: dict[str, str] = {},
    env_agent_combo_list: list[EnvAgentCombo[Observation, AgentAction]] = [],
    omniscient: bool = False,
    script_like: bool = False,
    json_in_script: bool = False,
    tag: str | None = None,
    push_to_db: bool = False,
    using_async: bool = True,
):
    assert not (push_to_db and tag is None), "please provide a tag when push to db"
    assert model_dict or env_agent_combo_list, "please provide model_dict or env_agent_combo_list"

    # Create Environment and agents
    # This step will be moved to outside this function

    def get_agent_class(
        model_name: str,
    ) -> Type[BaseAgent[Observation, AgentAction]]:
        if model_name == "human":
            return HumanAgent
        elif model_name == "redis":
            return RedisAgent
        elif script_like and not json_in_script:
            return ScriptWritingAgent
        else:
            return LLMAgent

    if env_agent_combo_list:
        assert type(sampler) is BaseSampler, (
            "No sampler should be used when `env_agent_combo_list` is not empty"
        )
        env_agent_combo_iter = iter(env_agent_combo_list)
    else:
        env_params = {
            "model_name": model_dict["env"],
            "action_order": action_order,
            "evaluators": [
                RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
            ],
            "terminal_evaluators": [
                EpisodeLLMEvaluator(
                    model_dict["env"],
                    EvaluationForTwoAgents[SotopiaDimensions],
                ),
            ],
        }
        agents_model_dict = {
            "agent1": model_dict["agent1"],
            "agent2": model_dict["agent2"],
        }
        env_agent_combo_iter = sampler.sample(
            size=num_episodes,
            agent_classes=[
                get_agent_class(model_name) for model_name in agents_model_dict.values()
            ],
            n_agent=len(agents_model_dict),
            env_params=env_params,
            agents_params=[
                {"model_name": model_name} if model_name != "human" else {}
                for model_name in agents_model_dict.values()
            ],
        )

    return list(env_agent_combo_iter)


def _arun_one_episode__pre_loop(
    env: ParallelSotopiaEnv,
    agent_list: Sequence[BaseAgent[Observation, AgentAction]],
    omniscient: bool = False,
    script_like: bool = False,
    json_in_script: bool = False,
    tag: str | None = None,
    push_to_db: bool = False,
    episode_pk: str | None = None,
    streaming: bool = False,
    simulation_status: NonStreamingSimulationStatus | None = None,
):
    episode_pk: str | None = None
    streaming: bool = False
    simulation_status: NonStreamingSimulationStatus | None = None

    agents = Agents({agent.agent_name: agent for agent in agent_list})
    print(f"Running episode with tag: {tag}------------------")

    environment_messages = env.reset(agents=agents, omniscient=omniscient)
    agents.reset()
    messages: list[list[tuple[str, str, Message]]] = []

    # Main Event Loop
    done = False
    messages.append(
        [
            ("Environment", agent_name, environment_messages[agent_name])
            for agent_name in env.agents
        ]
    )
    # yield messages

    # set goal for agents
    for index, agent_name in enumerate(env.agents):
        agents[agent_name].goal = env.profile.agent_goals[index]
    rewards: list[list[float]] = []
    reasons: list[str] = []

    return agents, SotopiaState(
        environment_messages=environment_messages,
        messages=messages,
        rewards=rewards,
        reasons=reasons,
        done=False,
        info=None,
    )


async def _arun_one_episode__loop_body(
    state: SotopiaState,
    env: ParallelSotopiaEnv,
    agents: Agents,
    script_like: bool,
    actions: Sequence[AgentAction],
):
    assert state.done is False
    environment_messages = state.environment_messages
    messages = state.messages
    rewards = state.rewards
    reasons = state.reasons

    # gather agent messages
    agent_messages: dict[str, AgentAction] = dict()
    # actions = await asyncio.gather(
    #    *[
    #        agents[agent_name].aact(environment_messages[agent_name])
    #        for agent_name in env.agents
    #    ]
    # )
    if script_like:
        # manually mask one message
        agent_mask = env.action_mask
        for idx in range(len(agent_mask)):
            if agent_mask[idx] == 0:
                actions[idx] = AgentAction(action_type="none", argument="")
            else:
                pass

    # actions = cast(list[AgentAction], actions)
    for idx, agent_name in enumerate(env.agents):
        agent_messages[agent_name] = actions[idx]

        messages[-1].append((agent_name, "Environment", agent_messages[agent_name]))

    # send agent messages to environmentee
    (
        environment_messages,
        rewards_in_turn,
        terminated,
        ___,
        info,
    ) = await env.astep(agent_messages)
    messages.append(
        [
            ("Environment", agent_name, environment_messages[agent_name])
            for agent_name in env.agents
        ]
    )

    # yield messages

    rewards.append([rewards_in_turn[agent_name] for agent_name in env.agents])
    reasons.append(" ".join(info[agent_name]["comments"] for agent_name in env.agents))
    done = all(terminated.values())

    return SotopiaState(
        environment_messages=environment_messages,
        messages=messages,
        rewards=rewards,
        reasons=reasons,
        info=info,
        done=done,
    )


def _arun_one_episode__loop_post(
    env: ParallelSotopiaEnv,
    agent_list: Sequence[BaseAgent[Observation, AgentAction]],
    tag: str | None,
    state: SotopiaState,
) -> EpisodeLog:
    epilog = EpisodeLog(
        environment=env.profile.pk,
        agents=[agent.profile.pk for agent in agent_list],
        tag=tag,
        models=[env.model_name, agent_list[0].model_name, agent_list[1].model_name],
        messages=[
            [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
            for messages_in_turn in state.messages
        ],
        reasoning=state.info[env.agents[0]]["comments"],
        rewards=[state.info[agent_name]["complete_rating"] for agent_name in env.agents],
        rewards_prompt=state.info["rewards_prompt"]["overall_prompt"],
    )
    rich.print(epilog.rewards_prompt)
    agent_profiles, conversation = epilog.render_for_humans()
    for agent_profile in agent_profiles:
        rich.print(agent_profile)
    for message in conversation:
        rich.print(message)

    return epilog
    # TODO

    if streaming:
        # yield the rewards and reasonings
        messages.append([("Evaluation", "Rewards", SimpleMessage(message=str(epilog.rewards)))])
        messages.append([("Evaluation", "Reasoning", SimpleMessage(message=epilog.reasoning))])
        # yield messages

    if push_to_db:
        try:
            if episode_pk:
                epilog.pk = episode_pk
                epilog.save()
            else:
                epilog.save()
            if simulation_status:
                simulation_status.status = "Completed"
                simulation_status.save()
        except Exception as e:
            logging.error(f"Failed to save episode log: {e}")
