"""Define the state structures for the agent."""

from __future__ import annotations

from typing import Sequence

from langgraph.graph import MessagesState
from typing_extensions import Annotated

from agent.common.utils import merge_sequences


# DOC: This is a basic state that will be used by all nodes in the graph. It ha one key: "messages" : list[AnyMessage]


class BaseGraphState(MessagesState):
    """Basic state"""
    user_id: str = None
    node_history: Annotated[Sequence[str], merge_sequences] = []
    node_params: dict = dict()
