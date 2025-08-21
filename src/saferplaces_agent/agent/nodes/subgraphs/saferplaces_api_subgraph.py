from langgraph.graph import StateGraph
from langgraph.graph import StateGraph, START

from agent import utils
from agent import names as N
from agent.common.states import BaseGraphState
from agent.nodes.tools import DemoWeatherTool
from agent.nodes.base import BaseToolHandlerNode, BaseToolInterruptNode



# DOC: SAFERPLACES API subgraph

api_saferbuildings_tool = DemoWeatherTool()
api_tools_dict = {
    api_saferbuildings_tool.name: api_saferbuildings_tool,
}
api_tool_names = list(api_tools_dict.keys())
api_tools = list(api_tools_dict.values())

llm_with_api_tools = utils._base_llm.bind_tools(api_tools)