from langgraph.graph import StateGraph
from langgraph.graph import StateGraph, START

from ...common import utils
from ...common import names as N
from ...common.states import BaseGraphState
from ...nodes.tools import DemoWeatherTool
from ...nodes.base import BaseToolHandlerNode, BaseToolInterruptNode



# DOC: Demo subgraph



demo_weather_tool = DemoWeatherTool()
demo_tools_dict = {
    demo_weather_tool.name: demo_weather_tool,
}
demo_tool_names = list(demo_tools_dict.keys())
demo_tools = list(demo_tools_dict.values())

llm_with_demo_tools = utils._base_llm.bind_tools(demo_tools)



# DOC: Base tool handler: runs the tool, if tool interrupt go to interrupt node handler
demo_tool_handler = BaseToolHandlerNode(
    state = BaseGraphState,
    tool_handler_node_name = N.DEMO_TOOL_HANDLER,
    tool_interrupt_node_name = N.DEMO_TOOL_INTERRUPT,
    tools = demo_tools_dict,
    additional_ouput_state = { 'requested_agent': None, 'node_params': dict() }
)


# DOC: Base tool interrupt node: handle tool interrupt by type and go back to tool hndler with updatet state to rerun tool
demo_tool_interrupt = BaseToolInterruptNode(
    state = BaseGraphState,
    tool_handler_node_name = N.DEMO_TOOL_HANDLER,
    tool_interrupt_node_name = N.DEMO_TOOL_INTERRUPT,
    tools = demo_tools_dict,
    custom_tool_interupt_handlers = dict()     # DOC: use default 
)
    
    
    
# DOC: State
demo_graph_builder = StateGraph(BaseGraphState)

# DOC: Nodes
demo_graph_builder.add_node(N.DEMO_TOOL_HANDLER, demo_tool_handler)
demo_graph_builder.add_node(N.DEMO_TOOL_INTERRUPT, demo_tool_interrupt)

# DOC: Edges
demo_graph_builder.add_edge(START, N.DEMO_TOOL_HANDLER)

# DOC: Compile
demo_weather_subgraph = demo_graph_builder.compile()
demo_weather_subgraph.name = N.DEMO_SUBGRAPH