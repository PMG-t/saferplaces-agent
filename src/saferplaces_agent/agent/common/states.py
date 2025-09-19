"""Define the state structures for the agent."""

from __future__ import annotations

import datetime

from typing import Sequence

from langgraph.graph import MessagesState
from typing_extensions import Annotated

from agent.common.utils import merge_sequences, merge_dictionaries


# DOC: This is a basic state that will be used by all nodes in the graph. It ha one key: "messages" : list[AnyMessage]


class BaseGraphState(MessagesState):
    """Basic state"""
    nowtime: str = datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None).isoformat()
    node_history: Annotated[Sequence[str], merge_sequences] = []
    node_params: Annotated[dict, merge_dictionaries] = dict()
    layer_registry: list[dict] = []
    avaliable_tools: list[str] = []
    
    user_id: str = None
    project_id: str = None


def src_layer_exists(graph_state: BaseGraphState, layer_src: str) -> bool:
    """Check if the layer exists in the graph state."""
    return any(layer.get('src') == layer_src for layer in graph_state.get('layer_registry', []))