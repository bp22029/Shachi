from collections import defaultdict
from logging import getLogger
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Union

import pydantic

from .auction_item import AuctionItem, item_list_equal
from .bidder_conventions import get_bidder_name

logger = getLogger(__name__)


# Function to colorize text (replacing langchain's get_colored_text)
def get_colored_text(text: str, color: str) -> str:
    color_codes = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "end": "\033[0m",
    }
    return f"{color_codes.get(color, '')}{text}{color_codes.get('end', '')}"


DESIRE_DESC = {
    "maximize_profit": "Your primary objective is to secure the highest profit at the end of this auction, compared to all other bidders",
    "maximize_items": "Your primary objective is to win the highest number of items at the end of this auction, compared to everyone else",
}


class WinningBid(pydantic.BaseModel):
    item_name: str = pydantic.Field(..., description="The Item name.")
    bid: int = pydantic.Field(..., description="The winning bid for this item.")


class BiddersStatus(pydantic.BaseModel):
    bidder_name: str = pydantic.Field(..., description="The name of the Bidder.")
    profit: int = pydantic.Field(..., description="Profit for this bidder.")
    winning_bids: List[WinningBid] = pydantic.Field(
        ..., description="The winning bids for this bidder."
    )


class StatusQuo(pydantic.BaseModel):
    remaining_budget: int = pydantic.Field(..., description="The remaining budget of yourself.")
    bidders_status: List[BiddersStatus] = pydantic.Field(
        ..., description="The status of all the bidders."
    )

    def to_text(self) -> str:
        """
        Converts status-quo to a structured text format.

        Used for generating human-readable summaries and prompts.

        Args:
            data: Status data in dictionary form

        Returns:
            Formatted text representation of the status
        """
        structured_text = f"* Remaining Budget: ${self.remaining_budget}\n\n"

        structured_text += "* Total Profits:\n"
        for bidder_status in self.bidders_status:
            structured_text += f"  * {bidder_status.bidder_name}: ${bidder_status.profit}\n"

        structured_text += "\n* Winning Bids:\n"
        for bidder_status in self.bidders_status:
            structured_text += f"  * {bidder_status.bidder_name}:\n"
            if len(bidder_status.winning_bids) == 0:
                structured_text += "    * No winning bids\n"
                continue

            for winning_bid in bidder_status.winning_bids:
                structured_text += f"    * {winning_bid.item_name}: ${winning_bid.bid}\n"

        return structured_text.strip()


class Bidder(pydantic.BaseModel):
    """
    Represents the state and configuration of a bidder in the auction.
    This class primarily holds data; the agent logic (LLM interaction)
    is handled by a separate Agent class (e.g., AuctionArenaAgent).
    """

    id: int
    budget: int
    desire_desc: Literal["maximize_profit", "maximize_items"]

    original_budget: int = 0
    profit: int = 0
    items_won: List[List[Union[AuctionItem, int]]] = []
    withdraw: bool = False
    status_quo: StatusQuo = {}
    cur_plan: str = ""
    learnings: str = ""

    # Rule-based bidder specific state
    is_rule_based: bool = False
    max_bid_cnt: int = 4
    rule_bid_cnt: int = 0

    # Monitoring/Logging state
    cur_item_id_internal: int = 0
    all_items_internal: List[AuctionItem] = []
    bid_history_internal: List[dict] = []

    failed_bid_cnt: int = 0
    total_bid_cnt: int = 0
    self_belief_error_cnt: int = 0
    total_self_belief_cnt: int = 0
    other_belief_error_cnt: int = 0
    total_other_belief_cnt: int = 0
    engagement_count: int = 0
    budget_history: List[int] = []
    profit_history: List[int] = []
    budget_error_history: List[List[Union[str, int]]] = []
    profit_error_history: List[List[Union[str, int]]] = []
    win_bid_error_history: List[List[Union[str, str]]] = []
    engagement_history: DefaultDict[str, int] = defaultdict(int)
    plan_history: List = []

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        return f"Bidder(id={self.id}, name='{self.name}')"

    def __str__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return get_bidder_name(self.id)

    @classmethod
    def create(cls, **data: Any) -> "Bidder":
        instance = cls(**data)
        instance._post_init()
        return instance

    def _post_init(self) -> None:
        self.original_budget = self.budget
        self.budget_history.append(self.budget)
        self.profit_history.append(self.profit)
        self.status_quo = StatusQuo(remaining_budget=self.budget, bidders_status=[])

    def reset_for_new_item(self) -> None:
        """Resets state variables specific to bidding on a single item."""
        self.withdraw = False
        self.rule_bid_cnt = 0
        self.bid_history_internal = []

    def reset_for_new_auction(self, items: List[AuctionItem]) -> None:
        """Resets state for the beginning of a new auction."""
        self.budget = self.original_budget
        self.profit = 0
        self.items_won = []
        self.withdraw = False
        self.status_quo = StatusQuo(remaining_budget=self.budget, bidders_status=[])
        self.cur_plan = ""
        self.rule_bid_cnt = 0
        self.cur_item_id_internal = 0
        self.all_items_internal = items
        self.bid_history_internal = []

        # Reset monitoring stats
        self.failed_bid_cnt = 0
        self.total_bid_cnt = 0
        self.self_belief_error_cnt = 0
        self.total_self_belief_cnt = 0
        self.other_belief_error_cnt = 0
        self.total_other_belief_cnt = 0
        self.engagement_count = 0
        self.budget_history = [self.budget]
        self.profit_history = [self.profit]
        self.budget_error_history = []
        self.profit_error_history = []
        self.win_bid_error_history = []
        self.engagement_history = defaultdict(int)
        self.plan_history = []

    # --- State Access/Helper Methods ---

    def get_current_item(self) -> AuctionItem:
        """Gets the current item being auctioned based on internal tracking."""
        return self.all_items_internal[self.cur_item_id_internal]

    def get_remaining_items(self) -> List[AuctionItem]:
        """Gets all items that are yet to be auctioned based on internal tracking."""
        return self.all_items_internal[self.cur_item_id_internal + 1 :]

    # --- Rule-Based Bidder Logic ---

    def bid_rule(self, cur_bid: int, min_markup_pct: float = 0.1) -> int:
        """
        Implements a rule-based bidding strategy.
        Calculates next bid amount based on current highest bid and markup.
        Will withdraw (-1) if exceeding budget or max bid count.
        """
        cur_item = self.get_current_item()

        if cur_bid <= 0:
            next_bid = cur_item.price
        else:
            next_bid = cur_bid + min_markup_pct * cur_item._original_price
        next_bid = int(next_bid)

        if self.budget >= next_bid and self.rule_bid_cnt < self.max_bid_cnt:
            self.rule_bid_cnt += 1
            return next_bid
        else:
            return -1

    # --- State Update Methods (called by Environment) ---

    def record_win(self, item: AuctionItem, bid_price: int) -> None:
        """Updates state after winning an item."""
        self.budget -= bid_price
        self.profit += item._true_value - bid_price
        self.items_won.append([item, bid_price])
        self.budget_history.append(self.budget)
        self.profit_history.append(self.profit)

    def record_loss(self) -> None:
        """Updates state after losing an item (if necessary)."""
        self.budget_history.append(self.budget)
        self.profit_history.append(self.profit)

    def set_withdraw_status(self, bid_amount: int) -> None:
        """
        Sets the bidder's withdrawal status based on bid value.
        Updates engagement metrics if the bidder makes a positive bid.
        """
        cur_item = self.get_current_item()
        if bid_amount < 0:  # withdraw
            self.withdraw = True
        elif bid_amount == 0:  # Can happen if item price is discounted and bidder rejoins
            self.withdraw = False
        else:  # normal bid
            self.withdraw = False
            self.engagement_count += 1
            self.engagement_history[cur_item.name] += 1
        self.total_bid_cnt += 1

    def advance_to_next_item(self) -> None:
        """Advances the internal item counter and resets item-specific state."""
        self.cur_item_id_internal += 1
        self.reset_for_new_item()

    def bid_sanity_check(
        self, bid_price: int, current_highest_bid: int, min_markup_pct: float
    ) -> Optional[str]:
        """
        Validates a bid against auction rules and constraints.
        Checks for sufficient budget, minimum bid requirements, and markup percentage.
        """
        cur_item = self.get_current_item()

        # Allow withdrawal
        if bid_price < 0:
            return None

        min_bid_increase = int(
            min_markup_pct * cur_item._original_price
        )

        if bid_price > self.budget:
            return f"you have Insufficient budget (${self.budget} left)"
        if bid_price < cur_item.price:
            return f"your Bid is lower than the starting bid (${cur_item.price})"
        if current_highest_bid > 0 and bid_price < current_highest_bid + min_bid_increase:
            return f"you must advance previous highest bid (${current_highest_bid}) by at least ${min_bid_increase} ({int(100 * min_markup_pct)}%)."

        return None

    def _sanity_check_status_json(self, data: Dict[str, Any]) -> str:
        """
        Validates the structure and content of the status JSON (agent's belief).
        """
        if not isinstance(data, dict) or data == {}:
            return "Error: Status must be a non-empty JSON object."

        expected_keys = ["remaining_budget", "total_profits", "winning_bids"]
        for key in expected_keys:
            if key not in data:
                return f"Error: Missing '{key}' field in the status JSON."

        if not isinstance(data["remaining_budget"], (int, float)):
            return "Error: 'remaining_budget' should be a number."

        if not isinstance(data["total_profits"], dict):
            return "Error: 'total_profits' should be a dictionary."
        for bidder, profit in data["total_profits"].items():
            if not isinstance(profit, (int, float)):
                return f"Error: Profit for bidder '{bidder}' should be a number."

        if not isinstance(data["winning_bids"], dict):
            return "Error: 'winning_bids' should be a dictionary."
        for bidder, bids in data["winning_bids"].items():
            if not isinstance(bids, dict):
                return f"Error: Winning bids for bidder '{bidder}' should be a dictionary."
            for item, amount in bids.items():
                if not isinstance(amount, (int, float)):
                    return f"Error: Bid amount for item '{item}' under bidder '{bidder}' should be a number."

        return ""

    def _status_json_to_text(self, data: dict) -> str:
        """
        Converts status JSON (agent's belief) to a structured text format.
        """
        logger.info(f"STATUS JSON: {data}")
        structured_text = f"* Remaining Budget: ${data.get('remaining_budget', 'N/A')}\n\n"

        profits = dict()
        winning_bids = dict()
        structured_text += f"* Bidders Status:\n"
        for bidder_data in data.get("bidders_status", []):
            bidder_name = bidder_data["bidder_name"]
            profit = bidder_data["profit"]
            winning_bids = bidder_data["winning_bids"]
            structured_text += f"  * Bidder Name: {bidder_name}\n"
            structured_text += f"    * Profit: {profit}\n"
            structured_text += f"    * Winning Bids:\n"
            if len(winning_bids) > 0:
                for bids in winning_bids:
                    structured_text += f"      * Item Name: {bids['item_name']}\n"
                    structured_text += f"      * Bid: {bids['bid']}\n"
            else:
                structured_text += f"      * No winning bids\n"

        return structured_text.strip()

    def check_belief(
        self, agent_status_quo: Dict[str, Any], all_bidder_states: Dict[int, "Bidder"]
    ) -> str:
        """
        Compares the agent's reported status_quo belief against the ground truth.
        Updates internal error tracking state.

        Args:
            agent_status_quo: The status quo JSON reported by the agent.
            all_bidder_states: Dictionary mapping agent ID to the true Bidder state object.

        Returns:
            A string message describing discrepancies, or empty string if consistent.
        """
        belief_json = agent_status_quo
        budget_belief = belief_json.get("remaining_budget")
        profits_belief = belief_json.get("total_profits", {})
        winning_bids_belief = belief_json.get("winning_bids", {})

        msg = ""
        current_item_name = self.get_current_item().name

        self.total_self_belief_cnt += 1
        if budget_belief is None or not isinstance(budget_belief, (int, float)):
            msg += f"- Invalid or missing 'remaining_budget' in belief.\n"
            self.self_belief_error_cnt += 1
        elif budget_belief != self.budget:
            msg += f"- Budget belief mismatch: Believed ${budget_belief}, Actual ${self.budget}.\n"
            self.self_belief_error_cnt += 1
            self.budget_error_history.append([current_item_name, budget_belief, self.budget])

        for bidder_id, true_state in all_bidder_states.items():
            bidder_name = true_state.name
            is_self = bidder_id == self.id

            if is_self:
                self.total_self_belief_cnt += 1
            else:
                self.total_other_belief_cnt += 1

            believed_profit = profits_belief.get(bidder_name)
            if believed_profit is None or not isinstance(believed_profit, (int, float)):
                msg += f"- Invalid or missing profit belief for {bidder_name}.\n"
                if is_self:
                    self.self_belief_error_cnt += 1
                else:
                    self.other_belief_error_cnt += 1
            elif believed_profit != true_state.profit:
                msg += f"- Profit belief mismatch for {bidder_name}: Believed ${believed_profit}, Actual ${true_state.profit}.\n"
                if is_self:
                    self.self_belief_error_cnt += 1
                else:
                    self.other_belief_error_cnt += 1
                self.profit_error_history.append(
                    [f"{bidder_name} ({current_item_name})", believed_profit, true_state.profit]
                )

            if is_self:
                self.total_self_belief_cnt += 1
            else:
                self.total_other_belief_cnt += 1

            believed_wins_dict = winning_bids_belief.get(bidder_name, {})
            if not isinstance(believed_wins_dict, dict):
                msg += f"- Invalid winning bids belief format for {bidder_name}.\n"
                if is_self:
                    self.self_belief_error_cnt += 1
                else:
                    self.other_belief_error_cnt += 1
                continue

            believed_wins_list = list(believed_wins_dict.keys())
            true_wins_list = [str(item) for item, _ in true_state.items_won]

            if not item_list_equal(believed_wins_list, true_wins_list):
                msg += f"- Winning items belief mismatch for {bidder_name}: Believed {believed_wins_list}, Actual {true_wins_list}.\n"
                if is_self:
                    self.self_belief_error_cnt += 1
                else:
                    self.other_belief_error_cnt += 1
                self.win_bid_error_history.append(
                    [
                        f"{bidder_name} ({current_item_name})",
                        ", ".join(believed_wins_list),
                        ", ".join(true_wins_list),
                    ]
                )

        return msg.strip()

    # ****************** Logging / Monitoring ****************** #

    def profit_report(self) -> str:
        """
        Generates a summary report of the bidder's performance.
        """
        msg = f"* {self.name} (ID: {self.id}), starting with ${self.original_budget}, won {len(self.items_won)} items, final profit: ${self.profit}, budget left: ${self.budget}.\n"
        for item, bid in self.items_won:
            item_profit = item._true_value - bid
            msg += f"  - Won '{item.name}' (Value: ${item._true_value}) for ${bid} (Profit: ${item_profit})\n"
        return msg.strip()

    def get_monitoring_data(self) -> Dict[str, Any]:
        """
        Prepares bidder data for monitoring and logging at the end of an auction.
        """
        items_won_log = [[str(item), bid, item._true_value] for item, bid in self.items_won]

        return {
            "bidder_id": self.id,
            "bidder_name": self.name,
            "desire": self.desire_desc,
            "budget_original": self.original_budget,
            "budget_final": self.budget,
            "profit_final": self.profit,
            "items_won_count": len(self.items_won),
            "items_won_details": items_won_log,
            "failed_bid_count": self.failed_bid_cnt,
            "total_bid_count": self.total_bid_cnt,
            "self_belief_error_count": self.self_belief_error_cnt,
            "total_self_belief_count": self.total_self_belief_cnt,
            "other_belief_error_count": self.other_belief_error_cnt,
            "total_other_belief_count": self.total_other_belief_cnt,
            "failed_bid_rate": round(self.failed_bid_cnt / (self.total_bid_cnt + 1e-8), 3),
            "self_belief_error_rate": round(
                self.self_belief_error_cnt / (self.total_self_belief_cnt + 1e-8), 3
            ),
            "other_belief_error_rate": round(
                self.other_belief_error_cnt / (self.total_other_belief_cnt + 1e-8), 3
            ),
            "engagement_bid_count": self.engagement_count,  # Bids > 0
            "engagement_history_by_item": dict(self.engagement_history),
            "budget_error_log": self.budget_error_history,
            "profit_error_log": self.profit_error_history,
            "win_bid_error_log": self.win_bid_error_history,
        }
