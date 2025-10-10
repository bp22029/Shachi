# tests/test_base.py
import pydantic
import pytest

from shachi import base


class MockMessage(base.Message):
    content: str


class MockObservation(base.Observation[MockMessage]):
    def format_as_prompt_text(self) -> str:
        return "\n".join(m.content for m in self.messages)


class MockResponse(pydantic.BaseModel):
    value: int


class MockResult(pydantic.BaseModel):
    score: float


class MockEnvironment(base.Environment[MockResult]):
    def __init__(self, num_agents: int = 2):
        self._num_agents = num_agents
        self._done = False
        self._step = 0

    def num_agents(self) -> int:
        return self._num_agents

    def done(self) -> bool:
        return self._done

    async def reset(self) -> dict[int, base.Observation]:
        self._step = 0
        self._done = False
        obs: dict[int, base.Observation] = {}
        for i in range(self._num_agents):
            msg = MockMessage(time=0, src_agent_id=None, dst_agent_id=i, content="reset")
            obs[i] = MockObservation(agent_id=i, messages=[msg], response_type=MockResponse)
        return obs

    async def step(self, responses: dict[int, str | pydantic.BaseModel | None]) -> dict[int, base.Observation]:
        self._step += 1
        if self._step >= 2:
            self._done = True
            return {}
        obs: dict[int, base.Observation] = {}
        for i in range(self._num_agents):
            msg = MockMessage(time=self._step, src_agent_id=None, dst_agent_id=i, content=f"step {self._step}")
            obs[i] = MockObservation(agent_id=i, messages=[msg], response_type=MockResponse)
        return obs

    def get_result(self) -> MockResult:
        return MockResult(score=100.0)


class MockAgent(base.Agent):
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.calls = 0

    async def step(self, observation: base.Observation):
        self.calls += 1
        if observation.response_type is not None:
            return observation.response_type(value=self.calls)
        return f"agent-{self.agent_id}-text"

    def update_config(self, kwargs_as_dict: dict) -> None:  # optional
        pass


def test_message_and_observation_payload():
    msg = MockMessage(time=1, src_agent_id=None, dst_agent_id=7, content="hello")
    obs = MockObservation(agent_id=7, messages=[msg])
    payload = obs.format_as_prompt_payload()
    assert payload and payload[0]["type"] == "text"
    assert "hello" in payload[0]["text"]
    assert obs.format_as_prompt_text() == "hello"


@pytest.mark.anyio
async def test_environment_lifecycle():
    env = MockEnvironment(num_agents=3)
    assert env.num_agents() == 3
    assert not env.done()

    obs = await env.reset()
    assert len(obs) == 3

    resp = {i: MockResponse(value=i) for i in range(3)}
    obs = await env.step(resp)
    assert len(obs) == 3

    obs = await env.step(resp)
    assert obs == {}
    assert env.done()

    res = env.get_result()
    assert res.score == 100.0


@pytest.mark.anyio
async def test_agent_step_variants():
    agent = MockAgent(agent_id=5)
    msg = MockMessage(time=0, src_agent_id=None, dst_agent_id=5, content="x")
    obs = MockObservation(agent_id=5, messages=[msg], response_type=MockResponse)
    out = await agent.step(obs)
    assert isinstance(out, MockResponse) and out.value == 1

    obs2 = MockObservation(agent_id=5, messages=[msg])
    out2 = await agent.step(obs2)
    assert isinstance(out2, str) and "agent-5" in out2


def test_tool_and_memory():
    class Params(pydantic.BaseModel):
        x: int

    class Resp(base.ToolResponse):
        result: int

        def format_as_prompt_text(self) -> str:
            return f"Result: {self.result}"

    def fn(p: Params) -> Resp:
        return Resp(result=p.x * 2)

    tool = base.Tool(name="mul", description="times 2", parameters_type=Params, fun=fn)
    out = tool.fun(Params(x=7))
    assert out.result == 14 and out.format_as_prompt_text() == "Result: 14"

    class SimpleMem(base.BaseMemory):
        def __init__(self) -> None:
            self._rec: list[dict[str, str]] = []

        def add_record(self, messages: list[dict[str, str]]) -> None:
            self._rec.extend(messages)

        def retrieve(self, query: str | None = None) -> str:
            return f"n={len(self._rec)}" if query is None else f"q={query},n={len(self._rec)}"

        def clear(self) -> None:
            self._rec.clear()

    m = SimpleMem()
    assert m.retrieve() == "n=0"
    m.add_record([{"role": "user", "content": "a"}])
    assert m.retrieve() == "n=1"
    assert m.retrieve("zzz") == "q=zzz,n=1"
    m.clear()
    assert m.retrieve() == "n=0"
