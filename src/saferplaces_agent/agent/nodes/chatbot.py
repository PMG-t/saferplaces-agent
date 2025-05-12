# DOC: Chatbot node and router

from typing_extensions import Literal

from langgraph.graph import END
from langgraph.types import Command

from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput


from agent import utils
from agent import names as N
from agent.common.states import BaseGraphState
from agent.nodes.tools import (
    DemoWeatherTool,
)



demo_weather_tool = DemoWeatherTool()

tools_map = {
    demo_weather_tool.name : demo_weather_tool,
}

tool_node = ToolNode([tool for tool in tools_map.values()])

llm_with_tools = utils._base_llm.bind_tools([tool for tool in tools_map.values()])


def set_tool_choice(tool_choice: str = None) -> Runnable[LanguageModelInput, BaseMessage]:
    if tool_choice is None:
        llm_with_tools = utils._base_llm.bind_tools([tool for tool in tools_map.values()])
    else:
        llm_with_tools = utils._base_llm.bind_tools([tool for tool in tools_map.values()], tool_choice=tool_choice)
    return llm_with_tools


def chatbot_update_messages(state: BaseGraphState):
    """Update the messages in the state with the new messages."""
    messages = state.get("node_params", dict()).get(N.CHATBOT_UPDATE_MESSAGES, dict()).get("update_messages", [])
    return {'messages': messages, 'node_params': dict()}


def chatbot(state: BaseGraphState) -> Command[Literal[END, N.CHATBOT_UPDATE_MESSAGES, N.DEMO_SUBGRAPH]]:     # type: ignore
    state["messages"] = state.get("messages", [])
    
    if len(state["messages"]) > 0:
        
        if state.get("node_params", dict()).get(N.CHATBOT_UPDATE_MESSAGES, None) is not None:
            return Command(goto=N.CHATBOT_UPDATE_MESSAGES)
        
        llm_with_tools = set_tool_choice(tool_choice = state.get("node_params", dict()).get(N.CHATBOT, dict()).get("tool_choice", None))
            
        ai_message = llm_with_tools.invoke(state["messages"])
        
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            
            # DOC: get the first tool call, discard others (this is ugly asf) edit: this works btw → user "get this and that" and the tool calls are "get-this-tool-call" and when it finishes "get-that-tool-call"
            tool_call = ai_message.tool_calls[0]
            ai_message.tool_calls = [tool_call] 
            
            if tool_call['name'] == demo_weather_tool.name:
                return Command(goto = N.DEMO_SUBGRAPH, update = { "messages": [ ai_message ], "node_history": [N.CHATBOT, N.DEMO_SUBGRAPH] })
    
        return Command(goto = END, update = { "messages": [ ai_message ], "requested_agent": None, "node_params": dict(), "node_history": [N.CHATBOT] })
    
    return Command(goto = END, update = { "messages": [], "requested_agent": None, "node_params": dict(), "node_history": [N.CHATBOT] })