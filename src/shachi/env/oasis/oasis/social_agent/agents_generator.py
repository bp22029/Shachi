# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import ast
import random
from typing import Any

import numpy as np
import pandas as pd
from camel.types import ModelType
from oasis.social_agent import AgentGraph
from oasis.social_platform import Platform
from oasis.social_platform.config import Neo4jConfig, UserInfo


def generate_agents(
    agent_info_path: str,
    start_time,
    recsys_type: str = "twitter",
    twitter: Platform = None,
    num_agents: int = 26,
    model_random_seed: int = 42,
    cfgs: list[Any] | None = None,
    neo4j_config: Neo4jConfig | None = None,
) -> AgentGraph:
    """Generate and return a dictionary of agents from the agent
    information CSV file. Each agent is added to the database and
    their respective profiles are updated.

    Args:
        agent_info_path (str): The file path to the agent information CSV file.
        channel (Channel): Information channel.
        num_agents (int): Number of agents.
        action_space_prompt (str): determine the action space of agents.
        model_random_seed (int): Random seed to randomly assign model to
            each agent. (default: 42)
        cfgs (list, optional): List of configuration. (default: `None`)
        neo4j_config (Neo4jConfig, optional): Neo4j graph database
            configuration. (default: `None`)

    Returns:
        dict: A dictionary of agent IDs mapped to their respective agent
            class instances.
    """
    random.seed(model_random_seed)
    model_types = []
    model_temperatures = []
    model_config_dict = {}
    for _, cfg in enumerate(cfgs):
        model_type = ModelType(cfg["model_type"])
        model_config_dict[model_type] = cfg
        model_types.extend([model_type] * cfg["num"])
        temperature = cfg.get("temperature", 0.0)
        model_temperatures.extend([temperature] * cfg["num"])
    random.shuffle(model_types)
    assert len(model_types) == num_agents
    agent_info = pd.read_csv(agent_info_path)
    # agent_info = agent_info[:10000]
    assert len(model_types) == len(agent_info), (
        f"Mismatch between the number of agents "
        f"and the number of models, with {len(agent_info)} "
        f"agents and {len(model_types)} models."
    )

    mbti_types = ["INTJ", "ENTP", "INFJ", "ENFP"]

    freq = list(agent_info["activity_level_frequency"])
    all_freq = np.array([ast.literal_eval(fre) for fre in freq])
    normalized_prob = all_freq / np.max(all_freq)
    # Make sure probability is not too small
    normalized_prob[normalized_prob < 0.6] += 0.1
    normalized_prob = np.round(normalized_prob, 2)
    prob_list: list[float] = normalized_prob.tolist()

    agent_graph = (
        AgentGraph()
        if neo4j_config is None
        else AgentGraph(
            backend="neo4j",
            neo4j_config=neo4j_config,
        )
    )

    sign_up_list = []
    follow_list = []
    user_update1 = []
    user_update2 = []
    post_list = []
    user_info_dict = {}

    for agent_id in range(len(agent_info)):
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {},
        }
        profile["other_info"]["user_profile"] = agent_info["user_char"][agent_id]
        profile["other_info"]["mbti"] = random.choice(mbti_types)
        profile["other_info"]["activity_level_frequency"] = ast.literal_eval(
            agent_info["activity_level_frequency"][agent_id]
        )
        profile["other_info"]["active_threshold"] = prob_list[agent_id]

        user_info = UserInfo(
            name=agent_info["username"][agent_id],
            description=agent_info["description"][agent_id],
            profile=profile,
            recsys_type=recsys_type,
        )
        user_info_dict[agent_id] = user_info

        agent_graph.add_agent_id(agent_id)
        num_followings = 0
        num_followers = 0
        # print('agent_info["following_count"]', agent_info["following_count"])
        if not agent_info["following_count"].empty:
            num_followings = int(agent_info["following_count"][agent_id])
        if not agent_info["followers_count"].empty:
            num_followers = int(agent_info["followers_count"][agent_id])

        sign_up_list.append(
            (
                agent_id,
                agent_id,
                agent_info["username"][agent_id],
                agent_info["name"][agent_id],
                agent_info["description"][agent_id],
                start_time,
                num_followings,
                num_followers,
            )
        )

        following_id_list = ast.literal_eval(agent_info["following_agentid_list"][agent_id])
        if not isinstance(following_id_list, int):
            if len(following_id_list) != 0:
                for follow_id in following_id_list:
                    follow_list.append((agent_id, follow_id, start_time))
                    user_update1.append((agent_id,))
                    user_update2.append((follow_id,))
                    agent_graph.add_edge(agent_id, follow_id)

        previous_posts = ast.literal_eval(agent_info["previous_tweets"][agent_id])
        if len(previous_posts) != 0:
            for post in previous_posts:
                post_list.append((agent_id, post, start_time, 0, 0))

    # generate_log.info('agent gegenerate finished.')

    user_insert_query = (
        "INSERT INTO user (user_id, agent_id, user_name, name, bio, "
        "created_at, num_followings, num_followers) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?)"
    )
    twitter.pl_utils._execute_many_db_command(user_insert_query, sign_up_list, commit=True)

    follow_insert_query = (
        "INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)"
    )
    twitter.pl_utils._execute_many_db_command(follow_insert_query, follow_list, commit=True)

    if not (agent_info["following_count"].empty and agent_info["followers_count"].empty):
        user_update_query1 = (
            "UPDATE user SET num_followings = num_followings + 1 WHERE user_id = ?"
        )
        twitter.pl_utils._execute_many_db_command(user_update_query1, user_update1, commit=True)

        user_update_query2 = "UPDATE user SET num_followers = num_followers + 1 WHERE user_id = ?"
        twitter.pl_utils._execute_many_db_command(user_update_query2, user_update2, commit=True)

    # generate_log.info('twitter followee update finished.')

    post_insert_query = (
        "INSERT INTO post (user_id, content, created_at, num_likes, "
        "num_dislikes) VALUES (?, ?, ?, ?, ?)"
    )
    twitter.pl_utils._execute_many_db_command(post_insert_query, post_list, commit=True)

    # generate_log.info('twitter creat post finished.')

    return agent_graph, user_info_dict
