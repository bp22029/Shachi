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

from abc import ABC, abstractmethod
from string import Template

from oasis.social_agent.agent_action import SocialAction


class Environment(ABC):
    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    posts_env_template = Template("After refreshing, you see some posts $posts")
    env_template = Template(
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts"
    )

    def __init__(self, action: SocialAction):
        self.action = action

    async def get_post_list(self) -> list | None:
        posts = await self.action.refresh()
        if posts["success"]:
            post_list = posts["posts"]
        else:
            post_list = None
        return post_list

    def to_text_prompt(self) -> str:
        raise NotImplementedError
