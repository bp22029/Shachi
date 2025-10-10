from typing import List, Literal, Union

import pydantic

from shachi.base import Message, Observation

from .auction_item import AuctionItem
from .bidder import DESIRE_DESC, StatusQuo
from .bidder_conventions import get_bidder_name
from .prompt_base import (
    INSTRUCT_BID_TEMPLATE,
    INSTRUCT_PLAN_TEMPLATE,
    INSTRUCT_REPLAN_TEMPLATE,
    INSTRUCT_SUMMARIZE_TEMPLATE,
)


class PlanResponse(pydantic.BaseModel):
    plan: str = pydantic.Field(
        ..., description="The bidding plan, potentially including item priorities and reasoning."
    )


class BidResponse(pydantic.BaseModel):
    bid_amount: int = pydantic.Field(..., description="The bid amount. Use -1 to withdraw.")


class SummarizeResponse(pydantic.BaseModel):
    status_quo: StatusQuo = pydantic.Field(
        ...,
        description="The agent's updated belief about the auction status (remaining budget, profits, winning bids).",
    )


class ReplanResponse(pydantic.BaseModel):
    plan: str = pydantic.Field(..., description="The updated bidding plan.")


class LearnResponse(pydantic.BaseModel):
    learnings: str = pydantic.Field(..., description="Key learnings extracted from past auctions.")


def _get_items_value_str(items: Union[List[AuctionItem], AuctionItem]) -> str:
    """
    Formats item information including name, price, and estimated value as a string.

    Used for generating prompts and instructions for the bidder LLM.

    Args:
        items: Single item or list of items to format

    Returns:
        Formatted string with numbered item details
    """
    if not isinstance(items, list):
        items = [items]
    items_info = ""
    for i, item in enumerate(items):
        estimated_value = item.estimated_value
        _info = f"{i + 1}. {item}, starting price is ${item.price}. Your estimated value for this item is ${estimated_value}.\n"
        items_info += _info
    return items_info.strip()


class AuctionMessage(Message):
    content: str = pydantic.Field(..., description="The message sent out during the auction.")

    @classmethod
    def get_hammer_msg(cls) -> str:
        return ""


class PlanObservation(Observation[AuctionMessage]):
    budget: int = pydantic.Field(..., description="The remaining budget ot the agent.")
    items: List[AuctionItem] = pydantic.Field(..., description="The auction items.")
    desire_desc: Literal["maximize_profit", "maximize_items"] = pydantic.Field(
        ..., description="The desire description for the agent."
    )

    def format_as_prompt_text(self) -> str:
        prompt: str = INSTRUCT_PLAN_TEMPLATE.format(
            bidder_name=get_bidder_name(self.agent_id),
            budget=self.budget,
            item_num=len(self.items),
            items_info=_get_items_value_str(self.items),
            desire_desc=DESIRE_DESC[self.desire_desc],
            learning_statement="",
        )
        return prompt


class RePlanObservation(Observation):
    status_quo: StatusQuo = pydantic.Field(..., description="The currrent status of the auction.")
    remaining_items: List[AuctionItem] = pydantic.Field(
        ..., description="The remaining auction items."
    )
    desire_desc: Literal["maximize_profit", "maximize_items"] = pydantic.Field(
        ..., description="The desire description for the agent."
    )
    response_type: type[pydantic.BaseModel] | None = ReplanResponse

    def format_as_prompt_text(self) -> str:
        prompt: str = INSTRUCT_REPLAN_TEMPLATE.format(
            status_quo=self.status_quo.to_text(),
            remaining_items_info=_get_items_value_str(self.remaining_items),
            bidder_name=get_bidder_name(self.agent_id),
            desire_desc=DESIRE_DESC[self.desire_desc],
            learning_statement="",
        )
        return prompt


class SummarizeObservation(Observation[AuctionMessage]):
    cur_item: AuctionItem
    bidding_history_text: str
    hammer_msg: str
    win_lose_msg: str
    prev_status_text: str
    response_type: type[pydantic.BaseModel] | None = SummarizeResponse

    def format_as_prompt_text(self) -> str:
        return INSTRUCT_SUMMARIZE_TEMPLATE.format(
            cur_item=self.cur_item,
            bidding_history=self.bidding_history_text,
            hammer_msg=self.hammer_msg,
            win_lose_msg=self.win_lose_msg,
            bidder_name=get_bidder_name(self.agent_id),
            prev_status=self.prev_status_text,
        )


class BiddingObservation(Observation[AuctionMessage]):
    cur_item: AuctionItem = pydantic.Field(description="The current auction item for bidding.")
    auctioneer_msg: str = pydantic.Field(
        description="The message from the auctioneer on the current auction round."
    )
    desire_desc: Literal["maximize_profit", "maximize_items"] = pydantic.Field(
        ..., description="The desire description for the agent."
    )
    response_type: type[pydantic.BaseModel] | None = BidResponse

    def format_as_prompt_text(self) -> str:
        auctioneer_msg_content = self.messages[-1].content if self.messages else ""
        prompt: str = INSTRUCT_BID_TEMPLATE.format(
            auctioneer_msg=auctioneer_msg_content,
            bidder_name=get_bidder_name(self.agent_id),
            cur_item=self.cur_item,
            estimated_value=self.cur_item.estimated_value,
            desire_desc=DESIRE_DESC[self.desire_desc],
            learning_statement="",
        )
        return prompt
