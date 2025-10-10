from unittest.mock import Mock, patch

import pydantic
import pytest

from shachi import Message, Observation
from shachi.agent.auction_arena import (
    AuctionArenaAgent_using_FunctionCalling,
    AuctionArenaAgent_using_StructuredOutput,
    EmptyMemory,
)
from shachi.agent.cognitive_biases_agent import (
    CognitiveBiasAgent_using_FunctionCalling,
    CognitiveBiasAgent_using_StructuredOutput,
)
from shachi.agent.cognitive_biases_agent import (
    RandomAgent as CognitiveRandomAgent,
)
from shachi.agent.digitmat_agent import (
    DigitMatAgent_using_FunctionCalling,
    DigitMatAgent_using_StructuredOutput,
)
from shachi.agent.digitmat_agent import (
    RandomAgent as DigitMatRandomAgent,
)
from shachi.agent.econagent import (
    EconAgentAgent_using_FunctionCalling,
    EconAgentAgent_using_StructuredOutput,
)
from shachi.agent.emotionagent import (
    EmotionAgent_using_FunctionCalling,
    EmotionAgent_using_StructuredOutput,
)
from shachi.agent.emotionagent import (
    RandomAgent as EmotionRandomAgent,
)
from shachi.agent.lm_caricature import LMCaricatureAgent
from shachi.agent.oasisagent import CamelMemory as OasisMemory
from shachi.agent.oasisagent import (
    SNSAgent_using_FunctionCalling,
    SNSAgent_using_StructuredOutput,
)
from shachi.agent.sotopia import SotopiaAgentST, call_llm
from shachi.agent.stockagent import HistoryMemory as StockMem
from shachi.agent.stockagent import (
    NoToolNoMemoryNoConfigStockAgent_using_FunctionCalling,
    StockAgent_using_FunctionCalling,
    StockAgent_using_StructuredOutput,
)


# ------------------------------
# Helpers
# ------------------------------
def _completion_with_content(text: str) -> Mock:
    m = Mock()
    m.choices = [Mock(message=Mock(content=text))]
    return m


def _completion_with_tool_args(args_json: str) -> Mock:
    tool_call = Mock()
    tool_call.function = Mock()
    tool_call.function.arguments = args_json
    msg = Mock()
    msg.tool_calls = [tool_call]
    msg.content = None
    m = Mock()
    m.choices = [Mock(message=msg)]
    return m


class TMsg(Message):
    text: str | None = None


class TObs(Observation[TMsg]):  # type: ignore[type-arg]
    _txt: str | None = None

    def set_text(self, txt: str) -> None:
        self._txt = txt

    def format_as_prompt_text(self) -> str:
        assert self._txt is not None
        return self._txt

    def format_as_prompt_payload(self) -> list[dict]:
        assert self._txt is not None
        return [{"type": "text", "text": self._txt}]


# ------------------------------
# AuctionArena Agents
# ------------------------------
class _BidResponse(pydantic.BaseModel):
    bid_amount: int


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_auction_structured_output(mock_acompletion):
    mock_acompletion.return_value = _completion_with_content('{"bid_amount": 123}')
    agent = AuctionArenaAgent_using_StructuredOutput(
        memory=EmptyMemory(), id=0, model="local", parser_model=None, temperature=0.0
    )
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_BidResponse)
    obs.set_text("bid please")
    out = await agent.step(obs)
    assert isinstance(out, _BidResponse)


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_auction_function_calling(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("plan"),
        _completion_with_tool_args('{"bid_amount": 456}'),
    ]
    agent = AuctionArenaAgent_using_FunctionCalling(
        memory=EmptyMemory(), id=0, model="local", parser_model=None, temperature=0.0
    )
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_BidResponse)
    obs.set_text("bid please")
    out = await agent.step(obs)
    assert isinstance(out, _BidResponse)


# ------------------------------
# Cognitive Bias Agents
# ------------------------------
class _Choice(pydantic.BaseModel):
    option: int


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_cog_structured_output(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("analysis"),
        _completion_with_content('{"option": 2}'),
    ]
    agent = CognitiveBiasAgent_using_StructuredOutput(model="local", temperature=0.0)
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_Choice)
    obs.set_text("Option 1: A\nOption 2: B")
    out = await agent.step(obs)
    assert isinstance(out, _Choice)


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_cog_function_calling(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("analysis"),
        _completion_with_tool_args('{"option": 3}'),
    ]
    agent = CognitiveBiasAgent_using_FunctionCalling(model="local", temperature=0.0)
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_Choice)
    obs.set_text("Option 1: A\nOption 2: B")
    out = await agent.step(obs)
    assert isinstance(out, _Choice)


@pytest.mark.anyio
async def test_cog_random_contract():
    agent = CognitiveRandomAgent()
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_Choice)
    obs.set_text("Option 1: A\nOption 2: B\nOption 3: C")
    out = await agent.step(obs)
    assert isinstance(out, _Choice)
    assert 0 <= out.option <= 3


# ------------------------------
# DigitMat Agents
# ------------------------------
class _DMResp(pydantic.BaseModel):
    pred_list: list[int]


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_digitmat_structured_output(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("plain"),
        _completion_with_content('{"pred_list":[1,2,3]}'),
    ]
    agent = DigitMatAgent_using_StructuredOutput(model="local", temperature=0.0)
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_DMResp)
    obs.set_text("[9 9 9]")
    out = await agent.step(obs)
    assert isinstance(out, _DMResp)


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_digitmat_function_calling(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("plain"),
        _completion_with_tool_args('{"pred_list":[4,5]}'),
    ]
    agent = DigitMatAgent_using_FunctionCalling(model="local", temperature=0.0, max_tokens=32)
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_DMResp)
    obs.set_text("[7 7]")
    out = await agent.step(obs)
    assert isinstance(out, _DMResp)


@pytest.mark.anyio
async def test_digitmat_random_contract():
    agent = DigitMatRandomAgent()
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_DMResp)
    obs.set_text("[1 1 1 1]")
    out = await agent.step(obs)
    assert isinstance(out, _DMResp)
    assert len(out.pred_list) == 4


# ------------------------------
# Emotion Agents
# ------------------------------
class _PANASLike(pydantic.BaseModel):
    model_config = {"extra": "allow"}


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_emotion_structured_output(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("plan"),
        _completion_with_content(
            "{" + ",".join([f'"item{i}":{{"name":"item{i}","score":3}}' for i in range(1, 21)]) + "}"
        ),
    ]
    agent = EmotionAgent_using_StructuredOutput(model="local", temperature=0.0, max_tokens=32)
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_PANASLike)
    obs.set_text("rate emotions")
    out = await agent.step(obs)
    assert isinstance(out, _PANASLike)


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_emotion_function_calling(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("plan"),
        _completion_with_tool_args(
            "{" + ",".join([f'"item{i}":{{"name":"item{i}","score":2}}' for i in range(1, 21)]) + "}"
        ),
    ]
    agent = EmotionAgent_using_FunctionCalling(model="local", temperature=0.0, max_tokens=32)
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_PANASLike)
    obs.set_text("rate emotions")
    out = await agent.step(obs)
    assert isinstance(out, _PANASLike)


@pytest.mark.anyio
async def test_emotion_random_contract():
    agent = EmotionRandomAgent()
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_PANASLike)
    obs.set_text("any")
    out = await agent.step(obs)
    assert isinstance(out, _PANASLike)


# ------------------------------
# EconAgent
# ------------------------------
class _EconResp(pydantic.BaseModel):
    work: float
    consumption: float
    rationale: str


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_econ_structured_output(mock_acompletion):
    mock_acompletion.return_value = _completion_with_content('{"work":0.5,"consumption":0.4,"rationale":"ok"}')
    agent = EconAgentAgent_using_StructuredOutput(model="local", temperature=0.0)
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_EconResp)
    obs.set_text("econ prompt")
    out = await agent.step(obs)
    assert isinstance(out, _EconResp)


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_econ_function_calling(mock_acompletion):
    mock_acompletion.return_value = _completion_with_tool_args('{"work":0.3,"consumption":0.8,"rationale":"fine"}')
    agent = EconAgentAgent_using_FunctionCalling(model="local", temperature=0.0)
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_EconResp)
    obs.set_text("econ prompt")
    out = await agent.step(obs)
    assert isinstance(out, _EconResp)


# ------------------------------
# LM Caricature
# ------------------------------
@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_lm_caricature_answers_with_text(mock_acompletion):
    mock_acompletion.return_value = _completion_with_content("ok-text")
    agent = LMCaricatureAgent(model="local", temperature=0.0, max_tokens=64)
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=None)
    obs.set_text("context prompt")
    out = await agent.step(obs)
    assert out == "ok-text"


# ------------------------------
# Oasis Agents
# ------------------------------
class _SNSResp(pydantic.BaseModel):
    action: str
    content: str


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_oasis_structured_output(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("plan"),
        _completion_with_content('{"action":"post","content":"hello"}'),
    ]
    agent = SNSAgent_using_StructuredOutput(agent_id=0, model="local", temperature=0.0, memory=OasisMemory())
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_SNSResp)
    obs.set_text("sns prompt")
    out = await agent.step(obs)
    assert isinstance(out, _SNSResp)


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_oasis_function_calling(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("plan"),
        _completion_with_tool_args('{"action":"reply","content":"hey"}'),
    ]
    agent = SNSAgent_using_FunctionCalling(agent_id=0, model="local", temperature=0.0, memory=OasisMemory())
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_SNSResp)
    obs.set_text("sns prompt")
    out = await agent.step(obs)
    assert isinstance(out, _SNSResp)


# ------------------------------
# Sotopia Agents (only ST)
# ------------------------------
class _STResp(pydantic.BaseModel):
    text: str


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_sotopia_call_llm_modes(mock_acompletion):
    mock_acompletion.return_value = _completion_with_content("plain-text")
    out = await call_llm(
        messages=[{"role": "user", "content": "x"}],
        model="m",
        temperature=0.0,
        parsing_mode="none",
        response_type=None,
    )
    assert out == "plain-text"


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_sotopia_agent_st_smoke(mock_acompletion):
    mock_acompletion.return_value = _completion_with_content('{"text":"ok"}')
    ag = SotopiaAgentST(model="m", parsing_mode="structured_output", temperature=0.0, drop_memory=False)
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_STResp)
    obs.set_text("Turn #0: hello")
    out = await ag.step(obs)
    assert isinstance(out, _STResp)


# ------------------------------
# Stock Agents
# ------------------------------
class _StockResp(pydantic.BaseModel):
    decision: str


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_stock_structured_output(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("plan"),
        _completion_with_content('{"decision":"hold"}'),
    ]
    agent = StockAgent_using_StructuredOutput(model="local", temperature=0.0, memory=StockMem())
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_StockResp)
    obs.set_text("stock prompt")
    out = await agent.step(obs)
    assert isinstance(out, _StockResp)


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_stock_function_calling(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("plan"),
        _completion_with_tool_args('{"decision":"buy"}'),
    ]
    agent = StockAgent_using_FunctionCalling(model="local", temperature=0.0, memory=StockMem())
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_StockResp)
    obs.set_text("stock prompt")
    out = await agent.step(obs)
    assert isinstance(out, _StockResp)


@pytest.mark.anyio
@patch("litellm.acompletion")
async def test_stock_notool_nomemory_function_calling(mock_acompletion):
    mock_acompletion.side_effect = [
        _completion_with_content("plan"),
        _completion_with_tool_args('{"decision":"sell"}'),
    ]
    agent = NoToolNoMemoryNoConfigStockAgent_using_FunctionCalling(model="local", temperature=0.0)
    msg = TMsg(time=0, src_agent_id=None, dst_agent_id=0)
    obs = TObs(agent_id=0, messages=[msg], response_type=_StockResp)
    obs.set_text("stock prompt")
    out = await agent.step(obs)
    assert isinstance(out, _StockResp)
