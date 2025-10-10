import random
from typing import Any, List, Literal, Union

import pydantic


class AuctionItem(pydantic.BaseModel):
    id: int = pydantic.Field(..., description="Item ID.")
    name: str = pydantic.Field(..., description="Item name.")
    price: int = pydantic.Field(..., description="Item price.")
    desc: str = pydantic.Field(..., description="Item description.")
    estimated_value: int = pydantic.Field(..., description="Item's Estimated value.")

    # Private attributes managed by Pydantic
    # Allow passing _true_value during initialization
    _true_value: int = pydantic.PrivateAttr(default=0)
    # _original_price will be set in model_post_init
    _original_price: int = pydantic.PrivateAttr(default=0)

    def model_post_init(self, __context: Any) -> None:
        """Set private attributes after standard initialization."""
        self._original_price = self.price

    def get_desc(self) -> str:
        return f"{self.name}, starting at ${int(self.price)}."

    def __repr__(self) -> str:
        return f"{self.name}"

    def __str__(self) -> str:
        return f"{self.name}"

    def info(self) -> str:
        return f"{self.name}: ${int(self.price)} to ${self._true_value}."

    def lower_price(self, percentage: float = 0.2) -> None:
        # lower starting price by 20%
        self.price = int(self.price * (1 - percentage))

    def reset_price(self) -> None:
        self.price = self._original_price


def create_items(
    items_info: list[dict], item_order: Literal["random", "desc", "asc"]
) -> List[AuctionItem]:
    """
    item_info: a list of dict (name, price, desc, id)
    """
    item_list: List[AuctionItem] = []
    for info in items_info:
        _true_value = info.pop("_true_value")
        item = AuctionItem(**info)
        item._true_value = _true_value

        item_list.append(item)

    if item_order == "random":
        random.shuffle(item_list)
    elif item_order == "asc":
        item_list.sort(key=lambda x: x.price)
    elif item_order == "desc":
        item_list.sort(key=lambda x: x.price, reverse=True)
    else:
        raise ValueError(f"Invalid item_order {item_order}")
    return item_list


def item_list_equal(
    items_1: List[Union[AuctionItem, str]], items_2: List[Union[AuctionItem, str]]
) -> bool:
    # could be a list of strings (names) or a list of Items
    item_1_names = [item.name if isinstance(item, AuctionItem) else item for item in items_1]
    item_2_names = [item.name if isinstance(item, AuctionItem) else item for item in items_2]
    return set(item_1_names) == set(item_2_names)
