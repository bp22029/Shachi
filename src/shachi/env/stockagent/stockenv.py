import datetime
import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import math
import random
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeAlias, TypedDict, cast

import pydantic
import util
from log.custom_logger import CustomLogger
from procoder.functional import format_prompt
from procoder.prompt import Collection, sharp2_indexing
from prompt.agent_prompt import (
    BACKGROUND_PROMPT,
    DECIDE_BUY_STOCK_PROMPT,
    DECIDE_IF_LOAN_PROMPT,
    FIRST_DAY_BACKGROUND_KNOWLEDGE,
    FIRST_DAY_FINANCIAL_REPORT,
    LASTDAY_FORUM_AND_STOCK_PROMPT,
    LOAN_TYPE_PROMPT,
    NEXT_DAY_ESTIMATE_PROMPT,
    POST_MESSAGE_PROMPT,
    SEASONAL_FINANCIAL_REPORT,
)
from pydantic import BaseModel, field_validator
from record import (
    AgentRecordDaily,
    create_agentses_record,
    create_stock_record,
    create_trade_record,
)
from stock import Stock

from shachi import Environment, Message, Observation, Task, Tool, ToolResponse


class LoanMessage(Message):
    prompt: str = pydantic.Field(
        description="The prompt for the laon.",
    )


class LoanResponse(pydantic.BaseModel):
    loan: bool = pydantic.Field(description="Loan decision. 'True' indicates approval, 'False' indicates rejection.")
    loan_type: int | None = pydantic.Field(
        default=None,
        description="Indicates the type of loan. If no loan is taken, this field should be 'None'.",
    )
    amount: int = pydantic.Field(
        default=0,
        description="The loan amount. If no loan is taken, this field should be 'None'.",
    )

    @field_validator("loan_type", mode="before")
    @classmethod
    def cast_loan_type(cls: type["LoanResponse"], v: Any) -> int | None:
        if v is None or v == "":
            return None
        if isinstance(v, int | float):
            return int(v) if int(v) < len(util.LOAN_TYPE) else None
        if isinstance(v, str):
            try:
                return int(float(v.replace(",", "")))
            except ValueError:
                return None
        return None

    @field_validator("amount", mode="before")
    @classmethod
    def cast_amount(cls: type["LoanResponse"], v: Any) -> int | None:
        if v is None or v == "":
            return 0
        if isinstance(v, int | float):
            return int(v)
        if isinstance(v, str):
            try:
                return int(float(v.replace(",", ""))) if int(float(v.replace(",", ""))) < len(util.LOAN_TYPE) else 0
            except ValueError:
                return 0
        return 0


class LoanObservation(Observation[LoanMessage]):
    def format_as_prompt_text(self) -> str:
        return self.messages[0].prompt


class StockMessage(Message):
    prompt: str = pydantic.Field(description="The prompt for the stock operation.")


class StockResponse(pydantic.BaseModel):
    action_type: str | None = pydantic.Field(
        default=None,
        description='Type of action for the stock transaction. Either "buy" or "sell". If neither, returns None.',
    )
    stock: str | None = pydantic.Field(
        default=None,
        description='The stock identifier, which can be "A" or "B". Returns None if no valid stock is specified.',
    )
    amount: int | None = pydantic.Field(
        default=None,
        description="The amount of stock to trade. Returns None if no valid transaction is made.",
    )
    price: float = pydantic.Field(
        default=0.0,
        description="The price per stock unit. Returns None if no valid transaction is made.",
    )

    @field_validator("amount", mode="before")
    @classmethod
    def cast_amount(cls: type["StockResponse"], v: Any) -> int | None:
        if v is None or v == "":
            return None
        if isinstance(v, int | float):
            return int(v)
        if isinstance(v, str):
            try:
                return int(float(v.replace(",", "")))
            except ValueError:
                return None
        return None

    @field_validator("price", mode="before")
    @classmethod
    def cast_price(cls: type["StockResponse"], v: Any) -> float:
        if v in (None, "", 0):
            return 0.0
        if isinstance(v, int | float):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v.replace(",", ""))
            except ValueError:
                return 0.0
        return 0.0


class StockObservation(Observation[StockMessage]):
    def format_as_prompt_text(self) -> str:
        prompt = self.messages[0].prompt

        if self.tools:
            prompt += "\nAvailable tools:\n"
            for tool in self.tools:
                prompt += f"- {tool.name}: {tool.description}\n"
            prompt += "\nPlease use the necessary tools in sequence to complete the task."
        return prompt


class EstimateMessage(Message):
    prompt: str = pydantic.Field(description="The prompt for the estimation for the next day")


class EstimateResponse(pydantic.BaseModel):
    buy_A: bool = pydantic.Field(default=False, description="Indicates whether to buy stock A. Default is False.")
    buy_B: bool = pydantic.Field(default=False, description="Indicates whether to buy stock B. Default is False.")
    sell_A: bool = pydantic.Field(default=False, description="Indicates whether to sell stock A. Default is False.")
    sell_B: bool = pydantic.Field(default=False, description="Indicates whether to sell stock B. Default is False.")
    loan: bool = pydantic.Field(default=False, description="Indicates whether a loan is applied. Default is False.")


class EstimateObservation(Observation[EstimateMessage]):
    def format_as_prompt_text(self) -> str:
        return self.messages[0].prompt


class PostMessage(Message):
    prompt: str = pydantic.Field(description="The prompt for the post message.")


class PostResponse(pydantic.BaseModel):
    post_content: str | None = pydantic.Field(
        description="The content of the post message. Returns None if no message is posted."
    )

    @field_validator("post_content", mode="before")
    @classmethod
    def cast_post_content(cls: type["PostResponse"], v: Any) -> str:
        if not isinstance(v, str):
            return ""
        return v


class PostObservation(Observation[PostMessage]):
    def format_as_prompt_text(self) -> str:
        return self.messages[0].prompt


class ForumPost(pydantic.BaseModel):
    name: int
    message: str


class GetForumPostResponse(ToolResponse):
    posts: list[ForumPost] = pydantic.Field(
        description="List of forum posts.",
    )

    def format_as_prompt_text(self) -> str:
        formatted_posts = []
        for post in self.posts:
            formatted_posts.append(f"{post.name}: {post.message}")
        return "Posts by other traders on the forum are as follows:\n" + "\n".join(formatted_posts)


class GetForumPostParameters(pydantic.BaseModel):
    pass


class GetNewsResponse(ToolResponse):
    day: int = pydantic.Field(description="date")
    news: str = pydantic.Field(description="news content")

    def format_as_prompt_text(self) -> str:
        return f"News for day {self.day}: {self.news}"


class GetNewsParameters(pydantic.BaseModel):
    pass


class StockAgentResult(pydantic.BaseModel):
    agent_day_record: pd.DataFrame = pydantic.Field(
        description="DataFrame containing daily records of the agent's actions and decisions.",
    )
    agent_session_record: pd.DataFrame = pydantic.Field(
        description="DataFrame containing session records of the agent's actions and decisions.",
    )
    stock_record: pd.DataFrame = pydantic.Field(
        description="DataFrame containing records of stock prices and transactions.",
    )
    trade_record: pd.DataFrame = pydantic.Field(
        description="DataFrame containing records of trades made by the agent.",
    )

    model_config = {"arbitrary_types_allowed": True}


class AggregatedStockAgentResult(pydantic.BaseModel):
    agent_day_records: list[pd.DataFrame] = pydantic.Field(
        description="List of DataFrames containing daily records of the agents' actions and decisions.",
    )
    agent_session_records: list[pd.DataFrame] = pydantic.Field(
        description="List of DataFrames containing session records of the agents' actions and decisions.",
    )
    stock_records: list[pd.DataFrame] = pydantic.Field(
        description="List of DataFrames containing records of stock prices and transactions.",
    )
    trade_records: list[pd.DataFrame] = pydantic.Field(
        description="List of DataFrames containing records of trades made by the agents.",
    )

    model_config = {"arbitrary_types_allowed": True}


ResponseT: TypeAlias = LoanResponse | StockResponse | EstimateResponse | PostResponse | str | None


class StockAgentAccount:
    def __init__(
        self,
        i: int,
        stock_a_price: float,
        stock_b_price: float,
        total_date: int,
        log: CustomLogger,
    ) -> None:
        self.agent_id = i
        self.stock_a_amount, self.stock_b_amount, self.cash, init_debt = self.random_init(stock_a_price, stock_b_price)
        self.init_proper = self.get_total_proper(stock_a_price, stock_b_price)
        self.action_history: list[list[Any]] = [[] for _ in range(total_date)]
        self.loans = [init_debt]
        self.is_bankrupt = False
        self.quit = False
        self.log = log

    def random_init(self, stock_a_initial: float, stock_b_initial: float) -> tuple[float, float, float, dict]:
        stock_a, stock_b, cash, debt_amount = 0.0, 0.0, 0.0, 0.0
        while (
            stock_a * stock_a_initial + stock_b * stock_b_initial + cash < util.MIN_INITIAL_PROPERTY
            or stock_a * stock_a_initial + stock_b * stock_b_initial + cash > util.MAX_INITIAL_PROPERTY
            or debt_amount > stock_a * stock_a_initial + stock_b * stock_b_initial + cash
        ):
            stock_a = int(random.uniform(0, util.MAX_INITIAL_PROPERTY / stock_a_initial))
            stock_b = int(random.uniform(0, util.MAX_INITIAL_PROPERTY / stock_b_initial))
            cash = random.uniform(0, util.MAX_INITIAL_PROPERTY)
            debt_amount = random.uniform(0, util.MAX_INITIAL_PROPERTY)
        debt = {
            "loan": "yes",
            "amount": debt_amount,
            "loan_type": random.randint(0, len(util.LOAN_TYPE) - 1),
            "repayment_date": random.choice(util.REPAYMENT_DAYS),
        }
        return stock_a, stock_b, cash, debt

    def get_total_proper(self, stock_a_price: float, stock_b_price: float) -> float:
        return self.stock_a_amount * stock_a_price + self.stock_b_amount * stock_b_price + self.cash

    def loan_repayment(self, date: int) -> None:
        if self.quit:
            return
        for loan in self.loans[:]:
            if loan["repayment_date"] == date:
                self.cash -= loan["amount"] * (1 + util.LOAN_RATE[loan["loan_type"]])
                self.loans.remove(loan)
        if self.cash < 0:
            self.is_bankrupt = True

    def interest_payment(self) -> None:
        if self.quit:
            return
        for loan in self.loans:
            self.cash -= loan["amount"] * util.LOAN_RATE[loan["loan_type"]] / 12
            if self.cash < 0:
                self.is_bankrupt = True

    def get_total_loan(self) -> float:
        debt = 0
        for loan in self.loans:
            debt += loan["amount"]
        return debt

    def get_proper_cash_value(self, stock_a_price: float, stock_b_price: float) -> tuple[float, float, float, float]:
        proper = self.stock_a_amount * stock_a_price + self.stock_b_amount * stock_b_price + self.cash
        a_value = self.stock_a_amount * stock_a_price
        b_value = self.stock_b_amount * stock_b_price
        return proper, self.cash, a_value, b_value

    def buy_stock(self, stock_name: str, price: float, amount: int) -> bool:
        if self.quit:
            return False
        if self.cash < price * amount or stock_name not in ["A", "B"]:
            self.log.logger.warning(f"ILLEGAL STOCK BUY BEHAVIOR: remain cash {self.cash}")
            return False
        self.cash -= price * amount
        if stock_name == "A":
            self.stock_a_amount += amount
        elif stock_name == "B":
            self.stock_b_amount += amount

        return True

    def sell_stock(self, stock_name: str, price: float, amount: int) -> bool:
        if self.quit:
            return False
        if stock_name == "B" and self.stock_b_amount < amount:
            self.log.logger.warning(
                f"ILLEGAL STOCK SELL BEHAVIOR: remain stock_b {self.stock_b_amount}, amount {amount}"
            )
            return False
        elif stock_name == "A" and self.stock_a_amount < amount:
            self.log.logger.warning(
                f"ILLEGAL STOCK SELL BEHAVIOR: remain stock_a {self.stock_a_amount}, amount {amount}"
            )
            return False
        if stock_name == "A":
            self.stock_a_amount -= amount
        elif stock_name == "B":
            self.stock_b_amount -= amount
        self.cash += price * amount
        return True

    def bankrupt_process(self, stock_a_price: float, stock_b_price: float) -> bool:
        if self.quit:
            return False
        total_value_of_stock = self.stock_a_amount * stock_a_price + self.stock_b_amount * stock_b_price
        if total_value_of_stock + self.cash < 0:
            self.log.logger.warning(f"Agent {self.agent_id} bankrupt. ")
            return True
        if stock_a_price * self.stock_a_amount >= -self.cash:
            sell_a = math.ceil(-self.cash / stock_a_price)
            self.stock_a_amount -= sell_a
            self.cash += sell_a * stock_a_price
        else:
            self.cash += stock_a_price * self.stock_a_amount
            self.stock_a_amount = 0
            sell_b = math.ceil(-self.cash / stock_b_price)
            self.stock_b_amount -= sell_b
            self.cash += sell_b * stock_b_price

        if self.stock_a_amount < 0 or self.stock_b_amount < 0 or self.cash < 0:
            raise RuntimeError("ERROR: WRONG BANKRUPT PROCESS")
        self.is_bankrupt = False
        return False


class StockAgentEnv(Environment):
    def __init__(
        self,
        num_agents: int,
        total_date: int,
        total_session: int,
        order_book: bool,
        timestamp: str,
        parallel_id: int,
    ):
        random.seed(42)
        self._num_agents = num_agents
        self.total_date = total_date
        self.total_session = total_session
        self.order_book = order_book
        self.date = 0
        self.session = 1
        self.timestamp = timestamp
        self.parallel_id = parallel_id
        self.reset_count = 0
        self._active_agent_ids: list[int] = []
        self._current_idx: int = 0

    def num_agents(self) -> int:
        return self._num_agents

    def done(self) -> bool:
        return self.date > self.total_date

    def get_default_agent_configs(self) -> list[dict]:
        configs = []
        for _ in range(self.num_agents()):
            character = random.choice(["Conservative", "Aggressive", "Balanced", "Growth-Oriented"])
            configs.append({"system_prompt": f"Your are a {character} investor."})
        return configs

    async def reset(self) -> dict[int, Observation]:
        self.reset_count += 1
        self.result_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "res",
            self.timestamp,
            f"parallel-{self.parallel_id}",
            f"{self.reset_count}",
        )
        os.makedirs(self.result_dir, exist_ok=True)
        self.log = CustomLogger(os.path.join(self.result_dir, "test.txt"))
        self.stock_a = Stock("A", util.STOCK_A_INITIAL_PRICE, 0, is_new=False)
        self.stock_b = Stock("B", util.STOCK_B_INITIAL_PRICE, 0, is_new=False)
        self.agent_accounts = {}
        for i in range(self._num_agents):
            account = StockAgentAccount(
                i, self.stock_a.get_price(), self.stock_b.get_price(), self.total_date, self.log
            )
            self.agent_accounts[i] = account
            self.log.logger.debug(
                f"cash: {account.cash}, stock a: {account.stock_a_amount}, "
                f"stock b:{account.stock_b_amount}, debt: {account.loans}"
            )

        self.last_day_forum_message: list = []
        self.stock_a_deals: dict[str, list] = {"sell": [], "buy": []}
        self.stock_b_deals: dict[str, list] = {"sell": [], "buy": []}
        self.session = 1
        self.date = 0
        self._active_agent_ids.clear()
        self._current_idx = 0
        await self._day_one_step()
        self.phase = "loan"
        return await self._get_loan_observations()

    async def _day_one_step(self) -> None:
        self.date += 1
        self.log.logger.debug(f"--------DAY {self.date}---------")
        self.stock_a_deals["sell"].clear()
        self.stock_a_deals["buy"].clear()
        self.stock_b_deals["buy"].clear()
        self.stock_b_deals["sell"].clear()
        self.daily_agent_records: list = []
        # check if an agent needs to repay loans
        for _, account in self.agent_accounts.items():
            account.loan_repayment(self.date)
        # repayment days
        if self.date in util.REPAYMENT_DAYS:
            for _, account in self.agent_accounts.items():
                account.interest_payment()
        # deal with cash<0 agents
        for agent_id in list(self.agent_accounts.keys()):
            account = self.agent_accounts[agent_id]
            if account.is_bankrupt:
                quit_sig = account.bankrupt_process(self.stock_a.get_price(), self.stock_b.get_price())
                if quit_sig:
                    account.quit = True
                    del self.agent_accounts[agent_id]
        # special events
        if self.date == util.EVENT_1_DAY:
            util.LOAN_RATE = util.EVENT_1_LOAN_RATE
            self.last_day_forum_message.append({"name": -1, "message": util.EVENT_1_MESSAGE})
        if self.date == util.EVENT_2_DAY:
            util.LOAN_RATE = util.EVENT_2_LOAN_RATE
            self.last_day_forum_message.append({"name": -1, "message": util.EVENT_2_MESSAGE})

    async def _get_loan_observations(self) -> dict[int, Observation]:
        observations: dict[int, Observation] = {}
        for agent_id, account in self.agent_accounts.items():
            if account.quit:
                continue
            if self.date == 1:
                prompt = (
                    Collection(BACKGROUND_PROMPT, LOAN_TYPE_PROMPT, DECIDE_IF_LOAN_PROMPT)
                    .set_indexing_method(sharp2_indexing)
                    .set_sep("\n")
                )
                max_loan = account.init_proper - account.get_total_loan()
                inputs = {
                    "date": self.date,
                    "stock_a": account.stock_a_amount,
                    "stock_b": account.stock_b_amount,
                    "cash": account.cash,
                    "debt": account.loans,
                    "max_loan": max_loan,
                    "loan_rate1": util.LOAN_RATE[0],
                    "loan_rate2": util.LOAN_RATE[1],
                    "loan_rate3": util.LOAN_RATE[2],
                }

            else:
                prompt = (
                    Collection(
                        BACKGROUND_PROMPT,
                        LASTDAY_FORUM_AND_STOCK_PROMPT,
                        LOAN_TYPE_PROMPT,
                        DECIDE_IF_LOAN_PROMPT,
                    )
                    .set_indexing_method(sharp2_indexing)
                    .set_sep("\n")
                )
                max_loan = account.init_proper - account.get_total_loan()
                inputs = {
                    "date": self.date,
                    "stock_a": account.stock_a_amount,
                    "stock_b": account.stock_b_amount,
                    "cash": account.cash,
                    "debt": account.loans,
                    "max_loan": max_loan,
                    "stock_a_price": self.stock_a.get_price(),
                    "stock_b_price": self.stock_b.get_price(),
                    "loan_rate1": util.LOAN_RATE[0],
                    "loan_rate2": util.LOAN_RATE[1],
                    "loan_rate3": util.LOAN_RATE[2],
                }
            if max_loan <= 0:
                continue
            prompt = format_prompt(prompt, inputs)

            get_forum_post_tool = Tool(
                name="get_last_day_forum_post",
                description="Get the last day forum post.",
                parameters_type=GetForumPostParameters,
                fun=self.get_last_day_forum_post,
            )

            get_news_tool = Tool(
                name="get_today_news",
                description="Get today's news.",
                parameters_type=GetNewsParameters,
                fun=self.get_today_news,
            )

            messages: list[LoanMessage] = []
            messages.append(
                LoanMessage(
                    time=self.date,
                    src_agent_id=None,
                    dst_agent_id=agent_id,
                    prompt=prompt,
                )
            )
            observations[agent_id] = LoanObservation(
                agent_id=agent_id,
                messages=messages,
                response_type=LoanResponse,
                tools=[get_forum_post_tool, get_news_tool],
            )
        return observations

    async def _loan_step(self, responses: dict[int, LoanResponse]) -> None:
        for agent_id, response in responses.items():
            account = self.agent_accounts[agent_id]
            try:
                loan_requested, loan_type, amount = (
                    response.loan,
                    response.loan_type,
                    response.amount,
                )
                self.log.logger.info(f"INFO: Agent {agent_id} decide to loan: {loan_requested}, {loan_type}, {amount}")
                max_loan = account.init_proper - account.get_total_loan()

                if loan_requested and (loan_type is not None) and (amount is not None) and (amount > 0):
                    if amount > max_loan:
                        loan: dict[str, Any] = {"loan": "no"}
                    else:
                        loan = {
                            "loan": "yes",
                            "loan_type": loan_type,
                            "amount": amount,
                            "repayment_date": self.date + util.LOAN_TYPE_DATE[loan_type],
                        }
                        account.loans.append(loan)
                        account.cash += float(loan["amount"])
                else:
                    loan = {"loan": "no"}
            except Exception:
                loan = {"loan": "no"}
            self.daily_agent_records.append(AgentRecordDaily(self.date, agent_id, loan))

    def get_last_day_forum_post(self, parameters: GetForumPostParameters) -> GetForumPostResponse:
        return GetForumPostResponse(posts=self.last_day_forum_message)

    def get_today_news(self, parameters: GetNewsParameters) -> GetNewsResponse:
        news_dict: dict[int, list[str]] = {
            1: [
                "Trump Unveils Sweeping Levies - Trading partners face a 10% rate or higher.",
            ],
            2: [
                "Tariffs Send Dow to 1600-Point Decline - fears of recession rose.",
            ],
            3: [
                "China Retaliates - Dow -2200pts, Nasdaq falls, oil sinks.",
            ],
        }

        today_news_list = news_dict.get(self.date, ["No major news today."])
        picked_news = random.choice(today_news_list)

        return GetNewsResponse(day=self.date, news=picked_news)

    async def _get_stock_observations(self, target_agent_id: int | None = None) -> dict[int, Observation]:
        self.log.logger.debug(f"SESSION {self.session}")
        observations: dict[int, Observation] = {}
        for agent_id, account in self.agent_accounts.items():
            if target_agent_id is not None and agent_id != target_agent_id:
                continue
            if account.quit:
                continue
            if self.date in util.SEASON_REPORT_DAYS and self.session == 1:
                index = util.SEASON_REPORT_DAYS.index(self.date)
                prompt = (
                    Collection(
                        FIRST_DAY_FINANCIAL_REPORT,
                        FIRST_DAY_BACKGROUND_KNOWLEDGE,
                        SEASONAL_FINANCIAL_REPORT,
                        DECIDE_BUY_STOCK_PROMPT,
                    )
                    .set_indexing_method(sharp2_indexing)
                    .set_sep("\n")
                )
                inputs = {
                    "date": self.date,
                    "time": self.session,
                    "stock_a": account.stock_a_amount,
                    "stock_b": account.stock_b_amount,
                    "stock_a_price": self.stock_a.get_price(),
                    "stock_b_price": self.stock_b.get_price(),
                    "stock_a_deals": self.stock_a_deals,
                    "stock_b_deals": self.stock_b_deals,
                    "cash": account.cash,
                    "stock_a_report": self.stock_a.gen_financial_report(index),
                    "stock_b_report": self.stock_b.gen_financial_report(index),
                }
            elif self.session == 1:
                prompt = (
                    Collection(
                        FIRST_DAY_FINANCIAL_REPORT,
                        FIRST_DAY_BACKGROUND_KNOWLEDGE,
                        DECIDE_BUY_STOCK_PROMPT,
                    )
                    .set_indexing_method(sharp2_indexing)
                    .set_sep("\n")
                )
                inputs = {
                    "date": self.date,
                    "time": self.session,
                    "stock_a": account.stock_a_amount,
                    "stock_b": account.stock_b_amount,
                    "stock_a_price": self.stock_a.get_price(),
                    "stock_b_price": self.stock_b.get_price(),
                    "stock_a_deals": self.stock_a_deals,
                    "stock_b_deals": self.stock_b_deals,
                    "cash": account.cash,
                }
            else:
                prompt = DECIDE_BUY_STOCK_PROMPT
                inputs = {
                    "date": self.date,
                    "time": self.session,
                    "stock_a": account.stock_a_amount,
                    "stock_b": account.stock_b_amount,
                    "stock_a_price": self.stock_a.get_price(),
                    "stock_b_price": self.stock_b.get_price(),
                    "stock_a_deals": self.stock_a_deals,
                    "stock_b_deals": self.stock_b_deals,
                    "cash": account.cash,
                }

            prompt = format_prompt(prompt, inputs)

            messages: list[StockMessage] = []
            messages.append(
                StockMessage(
                    time=self.date,
                    src_agent_id=None,
                    dst_agent_id=agent_id,
                    prompt=prompt,
                )
            )
            observations[agent_id] = StockObservation(agent_id=agent_id, messages=messages, response_type=StockResponse)
        return observations

    def _stock_step(self, responses: dict[int, StockResponse]) -> None:
        items = list(responses.items())
        random.shuffle(items)
        for agent_id, response in items:
            try:
                account = self.agent_accounts[agent_id]
                action_type, stock_name, amount, price = (
                    response.action_type,
                    response.stock,
                    response.amount,
                    response.price,
                )

                if action_type is None or stock_name is None or amount is None or price == 0:
                    self.log.logger.warning(
                        f"INFO: Agent {agent_id} decide to no action: action_type "
                        f"{action_type}, stock {stock_name}, amount {amount}, price {price}"
                    )
                    continue
                self.log.logger.info(
                    f"INFO: Agent {agent_id} decide to action_type: {action_type}, "
                    f"stock: {stock_name}, amount: {amount}, price: {price}"
                )
                proper, cash, valua_a, value_b = account.get_proper_cash_value(
                    self.stock_a.get_price(), self.stock_b.get_price()
                )

                class StockAction(TypedDict):
                    action_type: str
                    stock: str
                    amount: int
                    price: float
                    agent: int
                    date: str

                action: StockAction = {
                    "action_type": action_type,
                    "stock": stock_name,
                    "amount": amount,
                    "price": price,
                    "agent": agent_id,
                    "date": str(self.date),
                }

                create_agentses_record(
                    agent_id,
                    self.date,
                    self.session,
                    proper,
                    cash,
                    valua_a,
                    value_b,
                    action,
                    self.result_dir,
                )

                if not action_type:
                    continue

                if stock_name == "A":
                    stock_deals = self.stock_a_deals
                    stock_obj = self.stock_a
                else:
                    stock_deals = self.stock_b_deals
                    stock_obj = self.stock_b

                if action["action_type"] == "buy":
                    for sell_action in stock_deals["sell"][:]:
                        if action["price"] == sell_action["price"]:
                            close_amount: int = min(action["amount"], sell_action["amount"])
                            self.agent_accounts[action["agent"]].buy_stock(
                                stock_obj.name, action["price"], close_amount
                            )
                            if not sell_action["agent"] == -1:
                                self.agent_accounts[sell_action["agent"]].sell_stock(
                                    stock_obj.name, action["price"], close_amount
                                )
                            stock_obj.add_session_deal({"price": action["price"], "amount": close_amount})
                            create_trade_record(
                                action["date"],
                                self.session,
                                stock_obj.name,
                                action["agent"],
                                sell_action["agent"],
                                close_amount,
                                action["price"],
                                self.result_dir,
                            )

                            if action["amount"] > close_amount:
                                self.log.logger.info(
                                    f"ACTION - BUY:{action['agent']}, SELL:{sell_action['agent']}, "
                                    f"STOCK:{stock_obj.name}, PRICE:{action['price']}, AMOUNT:{close_amount}"
                                )
                                stock_deals["sell"].remove(sell_action)
                                action["amount"] -= close_amount
                            else:
                                self.log.logger.info(
                                    f"ACTION - BUY:{action['agent']}, SELL:{sell_action['agent']}, "
                                    f"STOCK:{stock_obj.name}, PRICE:{action['price']}, AMOUNT:{close_amount}"
                                )
                                sell_action["amount"] -= close_amount
                                break
                    else:
                        stock_deals["buy"].append(action)

                else:
                    for buy_action in stock_deals["buy"][:]:
                        if action["price"] == buy_action["price"]:
                            close_amount = min(action["amount"], buy_action["amount"])
                            self.agent_accounts[action["agent"]].sell_stock(
                                stock_obj.name, action["price"], close_amount
                            )
                            self.agent_accounts[buy_action["agent"]].buy_stock(
                                stock_obj.name, action["price"], close_amount
                            )
                            stock_obj.add_session_deal({"price": action["price"], "amount": close_amount})
                            create_trade_record(
                                action["date"],
                                self.session,
                                stock_obj.name,
                                buy_action["agent"],
                                action["agent"],
                                close_amount,
                                action["price"],
                                self.result_dir,
                            )

                            if action["amount"] > close_amount:
                                self.log.logger.info(
                                    f"ACTION - BUY:{buy_action['agent']}, SELL:{action['agent']}, "
                                    f"STOCK:{stock_obj.name}, PRICE:{action['price']}, AMOUNT:{close_amount}"
                                )
                                stock_deals["buy"].remove(buy_action)
                                action["amount"] -= close_amount
                            else:
                                self.log.logger.info(
                                    f"ACTION - BUY:{buy_action['agent']}, SELL:{action['agent']}, "
                                    f"STOCK:{stock_obj.name}, PRICE:{action['price']}, AMOUNT:{close_amount}"
                                )
                                buy_action["amount"] -= close_amount
                                break
                    else:
                        stock_deals["sell"].append(action)
            except Exception as e:
                self.log.logger.error(f"handle_action error: {e}")

        self.stock_a.update_price(self.date)
        self.stock_b.update_price(self.date)
        create_stock_record(
            self.date,
            self.session,
            self.stock_a.get_price(),
            self.stock_b.get_price(),
            self.result_dir,
        )

    async def _get_next_day_estimate_observations(self) -> dict[int, Observation]:
        observations: dict[int, Observation] = {}
        for agent_id, account in self.agent_accounts.items():
            if account.quit:
                continue
            prompt = format_prompt(NEXT_DAY_ESTIMATE_PROMPT, inputs={})
            messages: list[EstimateMessage] = []
            messages.append(
                EstimateMessage(
                    time=self.date,
                    src_agent_id=None,
                    dst_agent_id=agent_id,
                    prompt=prompt,
                )
            )
            observations[agent_id] = EstimateObservation(
                agent_id=agent_id,
                messages=messages,
                response_type=EstimateResponse,
            )
        return observations

    async def _next_day_estimate_step(self, responses: dict[int, EstimateResponse]) -> None:
        for idx, (agent_id, response) in enumerate(responses.items()):
            try:
                buy_A, buy_B, sell_A, sell_B, loan = (
                    response.buy_A,
                    response.buy_B,
                    response.sell_A,
                    response.sell_B,
                    response.loan,
                )
                self.log.logger.info(
                    f"Tomorrow estimation: Agent {agent_id} decide to buy_A: {buy_A}, "
                    f"buy_B: {buy_B}, sell_A: {sell_A}, sell_B: {sell_B}, loan: {loan}"
                )
                estimation = {
                    "buy_A": buy_A,
                    "buy_B": buy_B,
                    "sell_A": sell_A,
                    "sell_B": sell_B,
                    "loan": loan,
                }
                self.daily_agent_records[idx].add_estimate(estimation)
                self.daily_agent_records[idx].write_to_excel(self.result_dir)
            except Exception:
                self.log.logger.warning(f"Agent {agent_id} estimation failed.")
                continue
        self.daily_agent_records.clear()

    async def _get_post_message_observations(self) -> dict[int, Observation]:
        observations: dict[int, Observation] = {}
        for agent_id, account in self.agent_accounts.items():
            if account.quit:
                continue
            prompt = format_prompt(POST_MESSAGE_PROMPT, inputs={})
            messages: list[PostMessage] = []
            messages.append(
                PostMessage(
                    time=self.date,
                    src_agent_id=None,
                    dst_agent_id=agent_id,
                    prompt=prompt,
                )
            )
            observations[agent_id] = PostObservation(
                agent_id=agent_id,
                messages=messages,
                response_type=PostResponse,
            )
        return observations

    async def _post_message_step(self, responses: dict[int, PostResponse]) -> None:
        self.last_day_forum_message.clear()
        for agent_id, response in responses.items():
            try:
                post_content = response.post_content
                self.log.logger.info(f"INFO: Agent {agent_id} post message: {post_content}")
                self.last_day_forum_message.append({"name": agent_id, "message": post_content})
            except Exception:
                self.log.logger.warning(f"Agent {agent_id} post message failed.")
                continue

    async def step(self, responses: dict[int, str | BaseModel | None]) -> dict[int, Observation]:
        if self.phase == "loan":
            await self._loan_step(cast(dict[int, LoanResponse], responses))
            self.phase = "stock"
            self._active_agent_ids = [aid for aid, acct in self.agent_accounts.items() if not acct.quit]
            random.shuffle(self._active_agent_ids)
        elif self.phase == "stock":
            self._stock_step(cast(dict[int, StockResponse], responses))
            if self.order_book:
                if len(self._active_agent_ids) == 0:
                    self.session += 1
                    self._active_agent_ids = list(self.agent_accounts.keys())
                    random.shuffle(self._active_agent_ids)
            else:
                self.session += 1
            if self.session > self.total_session:
                self.phase = "next_day_estimate"
                self.session = 1
        elif self.phase == "next_day_estimate":
            await self._next_day_estimate_step(cast(dict[int, EstimateResponse], responses))
            self.phase = "post_message"
        elif self.phase == "post_message":
            await self._post_message_step(cast(dict[int, PostResponse], responses))
            self.phase = "new_day"

        if self.phase == "new_day":
            await self._day_one_step()
            self.phase = "loan"
            return await self._get_loan_observations()
        elif self.phase == "stock":
            if self.order_book:
                agent_id = self._active_agent_ids.pop(0)
                return await self._get_stock_observations(agent_id)
            else:
                return await self._get_stock_observations()
        elif self.phase == "next_day_estimate":
            return await self._get_next_day_estimate_observations()
        elif self.phase == "post_message":
            return await self._get_post_message_observations()
        else:
            raise ValueError(f"Unknown phase '{self.phase}' encountered after state updates.")

    def get_result(self) -> StockAgentResult:
        return StockAgentResult(
            agent_day_record=pd.read_excel(os.path.join(self.result_dir, "agent_day_record.xlsx")),
            agent_session_record=pd.read_excel(os.path.join(self.result_dir, "agent_session_record.xlsx")),
            stock_record=pd.read_excel(os.path.join(self.result_dir, "stocks.xlsx")),
            trade_record=pd.read_excel(os.path.join(self.result_dir, "trades.xlsx")),
        )


class StockAgentTask(Task):
    def __init__(
        self,
        num_parallel: int,
        num_agents: int,
        total_date: int,
        total_session: int,
        order_book: bool,
    ):
        self.num_parallel = num_parallel
        self.num_agents = num_agents
        self.total_date = total_date
        self.total_session = total_session
        self.order_book = order_book

    async def iterate_environments(self) -> AsyncIterator[Environment[StockAgentResult]]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        for i in range(self.num_parallel):
            logging.info(f"Creating environment {i + 1}/{self.num_parallel}")
            yield StockAgentEnv(
                num_agents=self.num_agents,
                total_date=self.total_date,
                total_session=self.total_session,
                order_book=self.order_book,
                timestamp=timestamp,
                parallel_id=i,
            )

    def aggregate_results(self, results: Sequence[StockAgentResult]) -> AggregatedStockAgentResult:
        return AggregatedStockAgentResult(
            agent_day_records=[result.agent_day_record for result in results],
            agent_session_records=[result.agent_session_record for result in results],
            stock_records=[result.stock_record for result in results],
            trade_records=[result.trade_record for result in results],
        )
