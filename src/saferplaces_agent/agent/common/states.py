"""Define the state structures for the agent."""

from __future__ import annotations

from typing import Sequence

from langgraph.graph import MessagesState
from typing_extensions import Annotated

from agent.common.utils import merge_sequences, merge_dictionaries


# DOC: This is a basic state that will be used by all nodes in the graph. It ha one key: "messages" : list[AnyMessage]


class BaseGraphState(MessagesState):
    """Basic state"""
    node_history: Annotated[Sequence[str], merge_sequences] = []
    node_params: Annotated[dict, merge_dictionaries] = dict()
    layer_registry: list[dict] = []
    avaliable_tools: list[str] = []
    
    user_id: str = None
    project_id: str = None
