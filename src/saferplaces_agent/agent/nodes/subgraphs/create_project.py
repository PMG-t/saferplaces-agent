from typing_extensions import Literal
from langgraph.types import Command
from langgraph.graph import StateGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import RemoveMessage, AIMessage, ToolMessage, ToolCall

from agent import utils
from agent import names as N
from agent.common.states import BaseGraphState
from agent.nodes.base import BaseToolHandlerNode, BaseToolInterruptNode
from agent.nodes.tools import (
    CreateProjectSelectDTMTool,
    CreateProjectSelectBuildingsTool,
    CreateProjectSelectInfiltrationRateTool,
    CreateProjectSelectLithologyTool,
    CreateProjectSelectOtherLayersTool
)




# DOC: Create Project subgraph



# DOC: Tools to be subsequently used in the subgraph
# TODO: node implementation (all)
select_dtm_tool = CreateProjectSelectDTMTool()
select_buildings_tool = CreateProjectSelectBuildingsTool()
select_infiltration_tool = CreateProjectSelectInfiltrationRateTool()
select_lithology_tool = CreateProjectSelectLithologyTool()
select_other_layers_tool = CreateProjectSelectOtherLayersTool()

create_project_tools_dict = {
    select_dtm_tool.name: select_dtm_tool,
    select_buildings_tool.name: select_buildings_tool,
    select_infiltration_tool.name: select_infiltration_tool,
    select_lithology_tool.name: select_lithology_tool,
    select_other_layers_tool.name: select_other_layers_tool
}



# DOC: Base tool handler: runs the tool, if tool interrupt go to interrupt node handler
create_project_tool_handler = BaseToolHandlerNode(
    state = BaseGraphState,
    tool_handler_node_name = N.CREATE_PROJECT_TOOL_HANDLER,
    tool_interrupt_node_name = N.CREATE_PROJECT_TOOL_INTERRUPT,
    tools = create_project_tools_dict,
    additional_ouput_state = { 'requested_agent': None, 'node_params': dict() },
    exit_nodes = [
        N.CREATE_PROJECT_SELECT_DTM_TOOL_RUNNER, 
        N.CREATE_PROJECT_SELECT_BUILDINGS_TOOL_RUNNER,
        N.CREATE_PROJECT_SELECT_INFILTRATION_TOOL_RUNNER,
        N.CREATE_PROJECT_SELECT_LITHOLOGY_TOOL_RUNNER,
        N.CREATE_PROJECT_SELECT_OTHER_LAYERS_TOOL_RUNNER
    ]
)


# DOC: Base tool interrupt node: handle tool interrupt by type and go back to tool hndler with updatet state to rerun tool
create_project_tool_interrupt = BaseToolInterruptNode(
    state = BaseGraphState,
    tool_handler_node_name = N.CREATE_PROJECT_TOOL_HANDLER,
    tool_interrupt_node_name = N.CREATE_PROJECT_TOOL_INTERRUPT,
    tools = create_project_tools_dict,
    custom_tool_interupt_handlers = dict()     # DOC: use default 
)



# DOC: Node 1 - Select DTM
def create_project_select_dtm_tool_runner(state: BaseGraphState) -> Command[Literal[END, N.CREATE_PROJECT_TOOL_HANDLER, N.CREATE_PROJECT_SELECT_BUILDINGS_TOOL_RUNNER]]:     # type: ignore
    
    """Select DTM tool."""
    state["messages"] = state.get("messages", [])
    
    if len(state["messages"]) > 0:
        # TODO: infer if usere specified some args → use _utils.ask_llm() → maybe we need to save all inference also for fututre steps
        pass
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(tool_name = select_dtm_tool.name)
    
    return Command(
        goto = N.CREATE_PROJECT_TOOL_HANDLER, 
        update = {
            'messages': [tool_call_message],
            'node_params': { N.CREATE_PROJECT_TOOL_HANDLER: { 'next_node': N.CREATE_PROJECT_SELECT_BUILDINGS_TOOL_RUNNER } }
        }
    )
    
    
# DOC: Node 2 - Select Buildings
def create_project_select_buildings_tool_runner(state: BaseGraphState) -> Command[Literal[END, N.CREATE_PROJECT_TOOL_HANDLER, N.CREATE_PROJECT_SELECT_INFILTRATION_TOOL_RUNNER]]:     # type: ignore
    
    """Select buildings tool."""
    
    # TODO: check if some inference was made by the user
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(tool_name = select_buildings_tool.name)
    
    return Command(
        goto = N.CREATE_PROJECT_TOOL_HANDLER, 
        update = {
            'messages': [tool_call_message],
            'node_params': { N.CREATE_PROJECT_TOOL_HANDLER: { 'next_node': N.CREATE_PROJECT_SELECT_INFILTRATION_TOOL_RUNNER } }
        }
    )
    

# DOC: Node 3 - Select Infiltration
def create_project_select_infiltration_tool_runner(state: BaseGraphState) -> Command[Literal[END, N.CREATE_PROJECT_TOOL_HANDLER, N.CREATE_PROJECT_SELECT_LITHOLOGY_TOOL_RUNNER]]:     # type: ignore
    
    """Select infiltration tool."""
    
    # TODO: check if some inference was made by the user
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(tool_name = select_infiltration_tool.name)
    
    return Command(
        goto = N.CREATE_PROJECT_TOOL_HANDLER, 
        update = {
            'messages': [tool_call_message],
            'node_params': { N.CREATE_PROJECT_TOOL_HANDLER: { 'next_node': N.CREATE_PROJECT_SELECT_LITHOLOGY_TOOL_RUNNER } }
        }
    )
    
    
# DOC: Node 4 - Select Lithology
def create_project_select_lithology_tool_runner(state: BaseGraphState) -> Command[Literal[END, N.CREATE_PROJECT_TOOL_HANDLER, N.CREATE_PROJECT_SELECT_OTHER_LAYERS_TOOL_RUNNER]]:     # type: ignore
    
    """Select lithology tool."""
    
    # TODO: check if some inference was made by the user
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(tool_name = select_lithology_tool.name)
    return Command(
        goto = N.CREATE_PROJECT_TOOL_HANDLER, 
        update = {
            'messages': [tool_call_message],
            'node_params': { N.CREATE_PROJECT_TOOL_HANDLER: { 'next_node': N.CREATE_PROJECT_SELECT_OTHER_LAYERS_TOOL_RUNNER } }
        }
    )
    

# DOC: Node 5 - Select Other Layers
def create_project_select_other_layers_tool_runner(state: BaseGraphState) -> Command[Literal[END, N.CREATE_PROJECT_TOOL_HANDLER]]:     # type: ignore
    
    """Select other layers tool."""
    
    # TODO: check if some inference was made by the user
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(tool_name = select_other_layers_tool.name)
    return Command(
        goto = N.CREATE_PROJECT_TOOL_HANDLER, 
        update = {
            'messages': [tool_call_message],
            'node_params': { N.CREATE_PROJECT_TOOL_HANDLER: { 'next_node': END } }
        }
    )
    
    

# DOC: State
create_project_graph_builder = StateGraph(BaseGraphState)

# DOC: Nodes
create_project_graph_builder.add_node(N.CREATE_PROJECT_TOOL_HANDLER, create_project_tool_handler)
create_project_graph_builder.add_node(N.CREATE_PROJECT_TOOL_INTERRUPT, create_project_tool_interrupt)

create_project_graph_builder.add_node(N.CREATE_PROJECT_SELECT_DTM_TOOL_RUNNER, create_project_select_dtm_tool_runner)
create_project_graph_builder.add_node(N.CREATE_PROJECT_SELECT_BUILDINGS_TOOL_RUNNER, create_project_select_buildings_tool_runner)
create_project_graph_builder.add_node(N.CREATE_PROJECT_SELECT_INFILTRATION_TOOL_RUNNER, create_project_select_infiltration_tool_runner)
create_project_graph_builder.add_node(N.CREATE_PROJECT_SELECT_LITHOLOGY_TOOL_RUNNER, create_project_select_lithology_tool_runner)
create_project_graph_builder.add_node(N.CREATE_PROJECT_SELECT_OTHER_LAYERS_TOOL_RUNNER, create_project_select_other_layers_tool_runner)

# DOC: Edges
create_project_graph_builder.add_edge(START, N.CREATE_PROJECT_SELECT_DTM_TOOL_RUNNER)

# DOC: Compile
create_project_subgraph = create_project_graph_builder.compile()
create_project_subgraph.name = N.CREATE_PROJECT_SUBGRAPH
        