from typing_extensions import Literal
from langgraph.types import Command
from langgraph.graph import StateGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import RemoveMessage, AIMessage, ToolMessage, ToolCall, SystemMessage
from langchain_core.tools import tool

from ...common import utils
from ...common import names as N
from ...common.states import BaseGraphState
from ...nodes.base import BaseToolHandlerNode, BaseToolHandlerNodeCallback, BaseToolInterruptNode
from ...nodes.tools import (
    FloodingRainfallDefineRainTool,
    FloodingRainfallDefineModelTool
)


# DOC: Flooding Rainfall subgraph



define_rain_tool = FloodingRainfallDefineRainTool()
define_model_tool = FloodingRainfallDefineModelTool()
flooding_rainfall_tools_dict = {
    define_rain_tool.name: define_rain_tool,
    define_model_tool.name: define_model_tool
}

@tool
def flooding_rainfall_subgraph_interface_tool(user_request: str) -> str:
    """This subgraph is used to handle the flooding rainfall simulation. It allows users to define the rainfall parameters and the model to be used for the simulation.
    The subgraph is composed by two main tools:
    1. Define Rain Tool: This tool is used to define the rainfall parameters for the simulation. It allows users to specify the type of rainfall, the amount of rainfall, and the non-uniform rainfall parameters.
    2. Define Model Tool: This tool is used to define the model to be used for the simulation. It allows users to specify the model name, simulation time, Manning coefficient, and number of layers.
    Invoke the function regardless of the level of detail of the request provided by the user as the pipeline will handle missing parameters and human-in-the-loop mechanism, as the subgraph will handle the tool calls and the responses from the user, and will return the final result of the simulation.
    
    Args:
        user_request (str): The user message requesting to start a flooding rainfall simulation.
    """
    
    return
    



# DOC: Node 0 - Main subtool handler node: handles the main subtool for the flooding rainfall simulation
def flooding_rainfall_main(state: BaseGraphState) -> Command[Literal[END, N.FLOODING_RAINFALL_TOOL_HANDLER, N.FLOODING_RAINFALL_DEFINE_RAIN_TOOL_RUNNER]]:      # type: ignore
    """Main subtools handler."""
    
    flooding_rainfall_params = state.get('node_params', dict()).get(N.FLOODING_RAINFALL_MAIN, dict()).get('tool_outputs', None)
    
    if flooding_rainfall_params is None:
    
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
        
        tools_description = '\n'.join([f"{itool+1}. {tool_decription(tool)}" for itool,tool in enumerate(flooding_rainfall_tools_dict.values())])
        
        subtool_args_prompt = f"""User asked to start a rainfall flooding simulation.
        The definition of the rainfall simulation is composed by a sequence of steps, some of them are mandatory and some of them are optional. In each steps users is asked to provide some parameters.
        
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
                    
        return Command(goto=N.FLOODING_RAINFALL_DEFINE_RAIN_TOOL_RUNNER, update={'node_params': update_node_params})
    
    else:
        # DOC: If params exists then pipeline is already executed, so we collect tool results and return them
        
        interface_tool_call_message = utils.build_tool_call_message(
            tool_name = N.FLOODING_RAINFALL_SUBGRAPH_INTERFACE_TOOL,
            tool_args = {
                'user_request': "compute a rainfall flood simulation",
            }
        )
        interface_tool_call_response = {
            "role": "tool",
            "name": N.FLOODING_RAINFALL_SUBGRAPH_INTERFACE_TOOL, 
            "content": flooding_rainfall_params,
            "tool_call_id": interface_tool_call_message.tool_calls[0]['id'],
        }
        
        return Command(goto=END, update={'messages': [ interface_tool_call_message, interface_tool_call_response ], 'node_params': dict() })


base_tool_handler_node_callback = BaseToolHandlerNodeCallback()

# DOC: Base tool handler: runs the tool, if tool interrupt go to interrupt node handler
flooding_rainfall_tool_handler = BaseToolHandlerNode(
    state = BaseGraphState,
    tool_handler_node_name = N.FLOODING_RAINFALL_TOOL_HANDLER,
    tool_interrupt_node_name = N.FLOODING_RAINFALL_TOOL_INTERRUPT,
    tools = flooding_rainfall_tools_dict,
    additional_ouput_state = { 'requested_agent': None, 'node_params': dict() },
    exit_nodes = [
        N.FLOODING_RAINFALL_DEFINE_RAIN_TOOL_RUNNER,
        N.FLOODING_RAINFALL_DEFINE_MODEL_TOOL_RUNNER
    ],
    on_handle_end_callback = base_tool_handler_node_callback
)


# DOC: Base tool interrupt node: handle tool interrupt by type and go back to tool hndler with updatet state to rerun tool
flooding_rainfall_tool_interrupt = BaseToolInterruptNode(
    state = BaseGraphState,
    tool_handler_node_name = N.FLOODING_RAINFALL_TOOL_HANDLER,
    tool_interrupt_node_name = N.FLOODING_RAINFALL_TOOL_INTERRUPT,
    tools = flooding_rainfall_tools_dict,
    custom_tool_interupt_handlers = dict()     # DOC: use default 
)
    
    

# DOC: Node 1 - Define Rain
def flooding_rainfall_define_rain_tool_runner(state: BaseGraphState) -> Command[Literal[END, N.FLOODING_RAINFALL_TOOL_HANDLER, N.FLOODING_RAINFALL_DEFINE_MODEL_TOOL_RUNNER]]:     # type: ignore
    
    """Define rain tool."""
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(
        tool_name = define_rain_tool.name, 
        tool_args = state.get('node_params', dict()).get(N.FLOODING_RAINFALL_DEFINE_RAIN_TOOL, dict())
    )
    
    def on_handle_end(tool_output):
        return {
            'update': {
                'node_params': {
                    N.FLOODING_RAINFALL_MAIN: {
                        'tool_outputs': {
                            N.FLOODING_RAINFALL_DEFINE_RAIN_TOOL: tool_output
                        }
                    }
                }
            },
            'next_node': N.FLOODING_RAINFALL_DEFINE_MODEL_TOOL_RUNNER
        }
    
    base_tool_handler_node_callback.callback = on_handle_end
    
    return Command(
        goto = N.FLOODING_RAINFALL_TOOL_HANDLER, 
        update = {
            'messages': [tool_call_message],
            'node_params': { N.FLOODING_RAINFALL_TOOL_HANDLER: { 'next_node': N.FLOODING_RAINFALL_DEFINE_MODEL_TOOL_RUNNER } }
        }
    )
    

# DOC: Node 2 - Define Model
def flooding_rainfall_define_model_tool_runner(state: BaseGraphState) -> Command[Literal[END, N.FLOODING_RAINFALL_TOOL_HANDLER]]:     # type: ignore
    
    """Define model tool."""
    
    # DOC: Build tool call message
    tool_call_message = utils.build_tool_call_message(
        tool_name = define_model_tool.name, 
        tool_args = state.get('node_params', dict()).get(N.FLOODING_RAINFALL_DEFINE_MODEL_TOOL, dict())
    )
    
    def on_handle_end(tool_output):
        return {
            'update': {
                'node_params': {
                    N.FLOODING_RAINFALL_MAIN: {
                        'tool_outputs': {
                            N.FLOODING_RAINFALL_DEFINE_MODEL_TOOL: tool_output
                        }
                    }
                }
            },
            'next_node': N.FLOODING_RAINFALL_MAIN
        }
    
    base_tool_handler_node_callback.callback = on_handle_end
    
    return Command(
        goto = N.FLOODING_RAINFALL_TOOL_HANDLER, 
        update = {
            'messages': [tool_call_message],
            'node_params': { N.FLOODING_RAINFALL_TOOL_HANDLER: { 'next_node': N.FLOODING_RAINFALL_MAIN } }
        }
    )


    
# DOC: State
flooding_rainfall_graph_builder = StateGraph(BaseGraphState)

# DOC: Nodes
flooding_rainfall_graph_builder.add_node(N.FLOODING_RAINFALL_MAIN, flooding_rainfall_main)

flooding_rainfall_graph_builder.add_node(N.FLOODING_RAINFALL_TOOL_HANDLER, flooding_rainfall_tool_handler)
flooding_rainfall_graph_builder.add_node(N.FLOODING_RAINFALL_TOOL_INTERRUPT, flooding_rainfall_tool_interrupt)

flooding_rainfall_graph_builder.add_node(N.FLOODING_RAINFALL_DEFINE_RAIN_TOOL_RUNNER, flooding_rainfall_define_rain_tool_runner)
flooding_rainfall_graph_builder.add_node(N.FLOODING_RAINFALL_DEFINE_MODEL_TOOL_RUNNER, flooding_rainfall_define_model_tool_runner)

# DOC: Edges
flooding_rainfall_graph_builder.add_edge(START, N.FLOODING_RAINFALL_MAIN)


# DOC: Compile
flooding_rainfall_subgraph = flooding_rainfall_graph_builder.compile()
flooding_rainfall_subgraph.name = N.FLOODING_RAINFALL_SUBGRAPH