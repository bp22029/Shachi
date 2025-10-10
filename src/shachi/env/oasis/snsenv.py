from __future__ import annotations

import asyncio
import datetime
import inspect
import logging
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Sequence
from typing import (
    Any,
    Literal,
    cast,
    get_type_hints,
)

import pandas as pd
import pydantic
from oasis.clock.clock import Clock
from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from pydantic import BaseModel, Field, create_model
from yaml import safe_load

from shachi import Environment, Message, Observation, Task


class Comment(pydantic.BaseModel):
    """
    Represents a single comment on a post.
    """

    comment_id: int = pydantic.Field(..., description="A unique identifier for the comment.")
    post_id: int = pydantic.Field(..., description="Identifies which post this comment belongs to.")
    user_id: int = pydantic.Field(..., description="The ID of the user who made the comment.")
    content: str = pydantic.Field(..., description="The text content of the comment.")
    created_at: int = pydantic.Field(description="Timestamp of when the comment was created.")
    num_likes: int = pydantic.Field(0, description="Number of likes for the comment.")
    num_dislikes: int = pydantic.Field(0, description="Number of dislikes for the comment.")


class SNSMessage(Message):
    """
    Represents a single post (tweet) as a message.

    Each element of a list of posts can be mapped to this class.
    """

    post_id: int = pydantic.Field(..., description="A unique identifier for the post.")
    content: str = pydantic.Field(..., description="The text content of the post.")
    num_likes: int = pydantic.Field(0, description="Number of likes for the post.")
    num_dislikes: int = pydantic.Field(0, description="Number of dislikes for the post.")
    num_shares: int = pydantic.Field(0, description="Number of times the post has been shared.")
    comments: list[Comment] = pydantic.Field(
        default_factory=list, description="A list of comments associated with this post."
    )


class SNSObservation(Observation[SNSMessage]):
    def format_as_prompt_text(self) -> str:
        ## arange for reasoning
        prompt = (
            "# OBJECTIVE\n"
            "You're a Twitter user, and I'll present you with some posts. After you see the posts, "
            "choose some actions from the following functions.\n"
            "When you choose an action, explain your thought process — what made you take that action?\n"
            "Suppose you are a real Twitter user. Please simulate real behavior.\n\n"
            "- do_nothing: If a post doesn't catch your attention or interest you enough to interact, "
            "you can choose to do nothing.\n"
            "- repost: If you think a tweet is important or interesting enough that you'd like your "
            "followers to see it, repost it. Mention the post ID.\n"
            "- quote_post: If you want to share a tweet but also add your thoughts or extra context, "
            "quote the tweet. Include the original post ID and what you're adding to it.\n"
            "- like_post: If something resonates with you, makes you laugh, or you simply agree with it, "
            "like the post. Tell me what specifically caught your interest, and don't forget to mention "
            "the post ID.\n"
            "- follow: If someone's posts consistently grab your attention or you want to see more from "
            "them, follow that user. Explain briefly why you're following them and include the user ID.\n"
            "Here is your social media environment:\n\n"
        )

        for msg in self.messages:
            prompt += (
                "=== Post ===\n"
                f"post_id: {msg.post_id}\n"
                f"user_id: {msg.src_agent_id}\n"
                f"content: {msg.content}\n"
                f"created_at: {msg.time}\n"
                f"num_likes: {msg.num_likes}\n"
                f"num_dislikes: {msg.num_dislikes}\n"
                f"num_shares: {msg.num_shares}\n"
            )

            if msg.comments:
                prompt += "comments:\n"
                for c in msg.comments:
                    prompt += (
                        "  ---\n"
                        f"  comment_id: {c.comment_id}\n"
                        f"  post_id: {c.post_id}\n"
                        f"  user_id: {c.user_id}\n"
                        f"  content: {c.content}\n"
                        f"  created_at: {c.created_at}\n"
                        f"  num_likes: {c.num_likes}\n"
                        f"  num_dislikes: {c.num_dislikes}\n"
                    )
                prompt += "  ---\n"
            else:
                prompt += "comments: (No comments yet)\n"

            prompt += "====================\n\n"
        return prompt


class BaseActionModel(pydantic.BaseModel):
    action: str


def create_action_union_from_functions(funcs: list[Callable]) -> BaseModel:
    action_model_list = []
    for func in funcs:
        name = func.__name__
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        fields_dict = OrderedDict()

        fields_dict["action"] = (Literal[name], Field(..., description="Action name."))

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            anno = type_hints.get(param_name)
            if anno is None:
                raise ValueError(f"Missing type hint for parameter {param_name} in {name}.")

            if param.default is param.empty:
                fields_dict[param_name] = (anno, ...)
            else:
                fields_dict[param_name] = (anno, param.default)

        model = create_model(name, __base__=BaseModel, **fields_dict)  # type: ignore
        action_model_list.append(model)

    if not action_model_list:
        # Choose the fallback that fits your semantics:
        # - Any: permissive default
        # - Never: empty/unsatisfiable type (typing.Never, py3.11+)
        return Any  # type: ignore[return-value]

    union_model = action_model_list[0]
    for t in action_model_list[1:]:
        union_model |= t

    return union_model  # type: ignore


openai_funcs = SocialAction(0, None).get_function_list()
ActionUnion = create_action_union_from_functions(openai_funcs)


class TwitterResponse(pydantic.BaseModel):
    actions: list[ActionUnion]  # type: ignore


class SNSResult(pydantic.BaseModel):
    snslog: list[str] = Field(
        description="SNS log containing information about the simulation.",
    )


class AggregatedSNSResult(pydantic.BaseModel):
    log_list: list[list[str]] = Field(
        description="Aggregated SNS logs from multiple environments.",
    )


class SNSEnv(Environment):
    def __init__(self, config_path: str, max_steps: int, timestamp: str, parallel_id: int):
        super().__init__()
        self.time_step = 0
        self.max_steps = max_steps
        # ---------- Read config ----------
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)) as f:
            cfg = safe_load(f)

        self.data_params = cfg["data"]
        simulation_params = cfg["simulation"]
        self.model_configs = cfg["model"]

        # ---------- DB setup ----------
        # self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.data_params["db_path"])
        self.db_path: str | None = None
        self.csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.data_params["csv_path"])
        self.topics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.data_params["topics_path"])

        # ---------- Simulation parameters ----------
        clock_factor = simulation_params.get("clock_factor", 60)
        self.recsys_type = simulation_params.get("recsys_type", "twhin-bert")

        # ---------- Setup clock start time ----------
        self.clock = Clock(k=clock_factor)

        self.timestamp = timestamp
        self.parallel_id = parallel_id
        self.reset_count = 0

        self.setup()

    def setup(self) -> None:
        self.result_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "log",
            self.timestamp,
            f"parallel-{self.parallel_id}",
            f"{self.reset_count}",
        )
        os.makedirs(self.result_dir, exist_ok=True)
        self.db_path = os.path.join(os.path.join(self.result_dir, "db.db"))
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(self.result_dir, f"social.twitter-{now}.log")

        self.twitter_channel = Channel()
        infra = Platform(
            self.db_path,
            self.twitter_channel,
            self.clock,
            start_time=0,
            recsys_type=self.recsys_type,
            refresh_rec_post_count=2,
            max_rec_post_len=2,
            following_post_count=3,
            log_path=self.log_path,
        )
        self.twitter_task = asyncio.create_task(infra.running())

        # ---------- Additional data / time setup ----------
        try:
            all_topic_df = pd.read_csv(self.topics_path)
            if "False" in self.csv_path or "True" in self.csv_path:
                if "-" not in self.csv_path:
                    topic_name = self.csv_path.split("/")[-1].split(".")[0]
                else:
                    topic_name = self.csv_path.split("/")[-1].split(".")[0].split("-")[0]

                source_post_time = (
                    all_topic_df[all_topic_df["topic_name"] == topic_name]["start_time"].item().split(" ")[1]
                )
                self.start_hour = int(source_post_time.split(":")[0]) + float(int(source_post_time.split(":")[1]) / 60)
            else:
                # Fallback
                self.start_hour = 13
        except Exception:
            print("No real-world data, let start_hour be 1PM.")
            self.start_hour = 13

        # ---------- Generate agents ----------
        self.agent_graph, self.user_info_dict = generate_agents(
            agent_info_path=self.csv_path,
            start_time=0,
            recsys_type=self.recsys_type,
            twitter=infra,
            **self.model_configs,
        )

        self.socialenv_dict = {}
        for agent_id in self.agent_graph.get_agent_ids():
            self.socialenv_dict[agent_id] = SocialEnvironment(SocialAction(agent_id, self.twitter_channel))

    def get_default_agent_configs(self) -> list[dict]:
        configs = []
        for agent_id in range(self.num_agents()):
            user_info = self.user_info_dict[agent_id]
            configs.append({"system_prompt": f"You are using SNS. Your profile: {user_info.profile}"})
        return configs

    def num_agents(self) -> int:
        return len(self.agent_graph.get_agent_ids())

    async def reset(self) -> dict[int, Observation]:
        self.reset_count += 1
        self.time_step = 0
        if self.reset_count > 1:
            self.twitter_task.cancel()
            self.setup()
        return await self._get_observations()

    def done(self) -> bool:
        if self.time_step >= self.max_steps:
            return True
        return False

    async def _get_observations(self) -> dict[int, Observation]:
        os.environ["SANDBOX_TIME"] = str(self.time_step * 3)
        simulation_time_hour = self.start_hour + 0.05 * self.time_step

        observations: dict[int, Observation] = {}
        for agent_id in range(self.num_agents()):
            agent_ac_prob = random.random()
            threshold = self.user_info_dict[agent_id].profile["other_info"]["active_threshold"][
                int(simulation_time_hour % 24)
            ]
            if agent_ac_prob < threshold:
                socialenv = self.socialenv_dict[agent_id]
                posts = await socialenv.get_post_list()
                if posts:
                    messages: list[SNSMessage] = []
                    for post in posts:
                        messages.append(
                            SNSMessage(
                                time=post["created_at"],
                                src_agent_id=post["user_id"],
                                dst_agent_id=agent_id,
                                post_id=post["post_id"],
                                content=post["content"],
                                num_likes=post["num_likes"],
                                num_dislikes=post["num_dislikes"],
                                num_shares=post["num_shares"],
                                comments=[
                                    Comment(
                                        comment_id=comment["comment_id"],
                                        post_id=comment["post_id"],
                                        user_id=comment["user_id"],
                                        content=comment["content"],
                                        created_at=comment["created_at"],
                                        num_likes=comment["num_likes"],
                                        num_dislikes=comment["num_dislikes"],
                                    )
                                    for comment in post["comments"]
                                ],
                            )
                        )
                    observations[agent_id] = SNSObservation(
                        agent_id=agent_id,
                        messages=messages,
                        response_type=TwitterResponse,
                    )
        return observations

    def perform_agent_graph_action(
        self,
        agent_id: int,
        action_name: str,
        arguments: dict[str, Any],
    ) -> None:
        if "unfollow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.remove_edge(agent_id, followee_id)
        elif "follow" in action_name:
            followee_id = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.add_edge(agent_id, followee_id)

    async def step(
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
    ) -> dict[int, Observation]:
        self.time_step += 1

        for agent_id, actions in responses.items():
            if actions is None:
                continue
            if not isinstance(actions, TwitterResponse):
                continue
            for action in actions.actions:
                base_action = cast(BaseActionModel, action)
                action_name = base_action.action
                args = {k: getattr(base_action, k) for k in base_action.model_fields_set if k != "action"}
                print(f"Agent {agent_id} is performing action: {action_name} with args: {args}")
                self.perform_agent_graph_action(agent_id, action_name, args)
                socialenv = self.socialenv_dict[agent_id]
                await getattr(socialenv.action, action_name)(**args)
        return await self._get_observations()

    def read_log_file(self, filepath: str) -> list:
        lines = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                lines.append(line.strip())
        return lines

    def get_result(self) -> SNSResult:
        return SNSResult(
            snslog=self.read_log_file(self.log_path),
        )


class SNSTask(Task):
    def __init__(self, num_parallel: int, config_path: str, max_steps: int):
        self.num_parallel = num_parallel
        self.config_path = config_path
        self.max_steps = max_steps
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    async def iterate_environments(self) -> AsyncIterator[Environment[SNSResult]]:
        for i in range(self.num_parallel):
            logging.info(f"Creating environment {i + 1}/{self.num_parallel}")
            yield SNSEnv(
                config_path=self.config_path,
                max_steps=self.max_steps,
                timestamp=self.timestamp,
                parallel_id=i,
            )

    def aggregate_results(self, results: Sequence[SNSResult]) -> AggregatedSNSResult:
        log_list = [result.snslog for result in results]
        return AggregatedSNSResult(log_list=log_list)
