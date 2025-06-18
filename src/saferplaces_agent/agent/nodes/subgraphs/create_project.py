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


# def test1(req): # TEST: just to test it from ipynb
#     def tool_decription(tool):
#         def args_description(args_schema):
#             args_description = '\n'.join([
#                 f'- {field} : {args_schema[field].description}'
#                 for field in args_schema.keys()
#             ])   
#             return args_description if args_description else "No arguments required."
#         tool_desc = f"""Tool: {tool.name}
#         Description: {tool.description}
#         Args description:
#         {args_description(tool.args_schema.model_fields)}
#         """
#         return tool_desc
    
#     tools_description = '\n'.join([f"{itool+1}. {tool_decription(tool)}" for itool,tool in enumerate(create_project_tools_dict.values())])
    
#     subtool_args_prompt = f"""User asked to create a new project.
#     The creation of a new project is composed by a sequence of steps, some of thema are mandatory and some of them are optional. In each steps users is asked to provide some parameters.
    
#     The steps are:
#     {tools_description}
    
#     The user request is: {req}
    
#     If the user has provided valid arguments please reply with a dictionary with key as tool name and value a dictionary with the arguments to be passed to the tool (if any).
#     If a value for an argument was not provided, then the value should be None.
#     User can provide only some of the arguments.
#     Reply with only the dictionary and nothing else.
#     """
    
#     provided_subtool_args = utils.ask_llm(role='system', message=subtool_args_prompt, eval_output=True)
#     return subtool_args_prompt, provided_subtool_args


# DOC: Node 0 - Main subtools handler
def create_project_main(state: BaseGraphState) -> Command[Literal[END, N.CREATE_PROJECT_SELECT_DTM_TOOL_RUNNER]]:     # type: ignore
    
    """Main subtools handler."""
    
    human_message = state["messages"][-1]
    
    def tool_decription(tool):
        def args_description(args_schema):
            args_description = '\n'.join([
                f'- {field} : {args_schema[field].description}'
                for field in args_schema.keys()
            ])   
            return args_description if args_description else "No arguments required."
        tool_desc = f"""Tool: {tool.name}
        Description: {tool.description}
        Args description:
        {args_description(tool.args_schema.model_fields)}
        """
        return tool_desc
    
    tools_description = '\n'.join([f"{itool+1}. {tool_decription(tool)}" for itool,tool in enumerate(create_project_tools_dict.values())])
    
    subtool_args_prompt = f"""User asked to create a new project.
    The creation of a new project is composed by a sequence of steps, some of thema are mandatory and some of them are optional. In each steps users is asked to provide some parameters.
    
    The steps are:
    {tools_description}
    
    The user request is: {human_message.content}
    
    If the user has provided valid arguments please reply with a dictionary with key as tool name and value a dictionary with the arguments to be passed to the tool (if any).
    If a value for an argument was not provided, then the value should be None.
    User can provide only some of the arguments.
    Reply with only the dictionary and nothing else.
    """
    
    provided_subtool_args = utils.ask_llm(role='system', message=subtool_args_prompt, eval_output=True)
    update_node_params = state.get('node_params', dict())
    if type(provided_subtool_args) is dict:
        for tool_name, tool_args in provided_subtool_args.items():
            if type(tool_args) is dict:
                update_node_params[tool_name] = tool_args
                
    # TODO: RESTART FROM HERE - after complete sequence of subtool, we return here and we have to colllect and check te provided args then run main tool (that is api call "saferplaces/new-project?name=<provided-name>" ???)
    
    return Command(goto=N.CREATE_PROJECT_SELECT_DTM_TOOL_RUNNER, update={'node_params': update_node_params})
    
    


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
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(
        tool_name = select_dtm_tool.name, 
        tool_args=state.get('node_params', dict()).get(N.CREATE_PROJECT_SELECT_DTM_TOOL, dict())
    )
    
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
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(
        tool_name = select_buildings_tool.name,
        tool_args=state.get('node_params', dict()).get(N.CREATE_PROJECT_SELECT_BUILDINGS_TOOL, dict())
    )
    
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
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(
        tool_name = select_infiltration_tool.name,
        tool_args=state.get('node_params', dict()).get(N.CREATE_PROJECT_SELECT_INFILTRATION_TOOL, dict())
    )
    
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
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(
        tool_name = select_lithology_tool.name,
        tool_args=state.get('node_params', dict()).get(N.CREATE_PROJECT_SELECT_LITHOLOGY_TOOL, dict())
    )
    
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
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(
        tool_name = select_other_layers_tool.name,
        tool_args=state.get('node_params', dict()).get(N.CREATE_PROJECT_SELECT_OTHER_LAYERS_TOOL, dict())
    )
    
    return Command(
        goto = N.CREATE_PROJECT_TOOL_HANDLER, 
        update = {
            'messages': [tool_call_message],
            'node_params': { N.CREATE_PROJECT_TOOL_HANDLER: { 'next_node': N.CREATE_PROJECT_MAIN } }
        }
    )
    
    

# DOC: State
create_project_graph_builder = StateGraph(BaseGraphState)

# DOC: Nodes
create_project_graph_builder.add_node(N.CREATE_PROJECT_MAIN, create_project_main)

create_project_graph_builder.add_node(N.CREATE_PROJECT_TOOL_HANDLER, create_project_tool_handler)
create_project_graph_builder.add_node(N.CREATE_PROJECT_TOOL_INTERRUPT, create_project_tool_interrupt)

create_project_graph_builder.add_node(N.CREATE_PROJECT_SELECT_DTM_TOOL_RUNNER, create_project_select_dtm_tool_runner)
create_project_graph_builder.add_node(N.CREATE_PROJECT_SELECT_BUILDINGS_TOOL_RUNNER, create_project_select_buildings_tool_runner)
create_project_graph_builder.add_node(N.CREATE_PROJECT_SELECT_INFILTRATION_TOOL_RUNNER, create_project_select_infiltration_tool_runner)
create_project_graph_builder.add_node(N.CREATE_PROJECT_SELECT_LITHOLOGY_TOOL_RUNNER, create_project_select_lithology_tool_runner)
create_project_graph_builder.add_node(N.CREATE_PROJECT_SELECT_OTHER_LAYERS_TOOL_RUNNER, create_project_select_other_layers_tool_runner)

# DOC: Edges
create_project_graph_builder.add_edge(START, N.CREATE_PROJECT_MAIN)

# DOC: Compile
create_project_subgraph = create_project_graph_builder.compile()
create_project_subgraph.name = N.CREATE_PROJECT_SUBGRAPH
        