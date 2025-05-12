
import types
from typing_extensions import Literal

from langgraph.graph import END
from langgraph.types import Command

from . import BaseToolInterrupt


# DOC: base_tool_handler is a function that creates a tool handler function for a specific AgentTool.

class BaseToolHandlerNode:
    
    # DOC: This is a template function that will be used to create the tool handler function.
    
    def __new__(
        cls,
        state,
        tool_handler_node_name: str,
        tool_interrupt_node_name: str,
        tools: dict,
        additional_ouput_state: dict = dict()
    ):
        instance = super().__new__(cls) 
        instance.__init__(
            state,
            tool_handler_node_name,
            tool_interrupt_node_name,
            tools,
            additional_ouput_state
        )
        return instance.setup()
        
    
    def __init__( 
            self,
            state,
            tool_handler_node_name: str,
            tool_interrupt_node_name: str,
            tools: dict,
            additional_ouput_state: dict = dict()
    ):
        self.state = state
        self.state_type = type(state)
        self.tool_handler_node_name = tool_handler_node_name
        self.tool_interrupt_node_name = tool_interrupt_node_name
        self.tools = tools
        self.additional_ouput_state = additional_ouput_state
        
        
    def setup(self):
        
        # DOC: This is a template function that will be used to create the tool handler function node.
        def tool_handler_template(state):
            tool_message = state["messages"][-1]
            tool_call = tool_message.tool_calls[-1]
            
            result = None
            try:
                tool = self.tools[tool_call['name']]
                tool.graph_state = state
                result = tool.invoke(tool_call['args'])
            except BaseToolInterrupt as tool_interrupt:                
                update_state = {}
                update_state['node_params'] = { 
                    self.tool_interrupt_node_name: {
                        'tool_message': tool_message,
                        'tool_interrupt': tool_interrupt.as_dict,
                        'tool_handler_node': self.tool_handler_node_name,    # INFO: Where to return interrupt "response" data
                    }
                }
                return Command(goto=self.tool_interrupt_node_name, update = update_state)     
                    
            tool_response_message = {
                "role": "tool",
                "name": tool_call['name'], 
                "content": result,
                "tool_call_id": tool_call['id'],
            }
            
            return {"messages": tool_response_message, **self.additional_ouput_state}
                
                

        # DOC: Creating the tool handler function using the template function.
        tool_handler = types.FunctionType(
            tool_handler_template.__code__,
            globals(),
            name = self.tool_handler_node_name,
            argdefs = tool_handler_template.__defaults__,
            closure = tool_handler_template.__closure__
        )
        
        tool_handler.__annotations__ = {
            'state': type(self.state),
            'return': Command[Literal[END, self.tool_interrupt_node_name]]
        }
        
        return tool_handler