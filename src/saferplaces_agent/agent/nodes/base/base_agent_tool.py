from typing import Optional
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from . import BaseToolInterrupt


# DOC: This is a base agent tool that exploit ToolInterrupt for human-in-the-loop paradigm
class BaseAgentTool(BaseTool):
    
    # DOC: BaseTool args
    name: str = None
    description: str = None
    args_schema: Optional[ArgsSchema] = None
    return_direct: bool = True
    
    # DOC: Additional args
    graph_state: dict = None
    execution_confirmed: bool = False
    output_confirmed: bool = False
    output: dict = None
    
    # DOC: Setup specific tool with a given name, description and args_schema
    def __init__(self, name: str, description: str, args_schema: Optional[ArgsSchema], **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.description = description
        self.args_schema = args_schema
        
    # DOC: Check missing arguments based on the args_schema (if Deafult is None than it's not required)
    def check_required_args(self, tool_args):
        missing_args = [arg for arg, schema in self.args_schema.model_fields.items() if schema.is_required() and tool_args[arg] is None]
        
        if len(missing_args) > 0:
            self.execution_confirmed = False
            raise BaseToolInterrupt(
                interrupt_tool = self.name,
                interrupt_type = BaseToolInterrupt.BaseToolInterruptType.PROVIDE_ARGS,
                interrupt_reason = f"Missing required arguments: {missing_args}.",
                interrupt_data = {
                    "missing_args": missing_args,
                    "args_schema": self.args_schema.model_fields
                }
            )
            
    # DOC: Check invalid arguments based on a list of function related to each argument { argname: [ test(**tool_args) -> Invalid-Reason else None" , ... ], ... }
    def _set_args_validation_rules(self):
        return { arg: [] for arg in self.args_schema.model_fields.keys() }
            
    def check_validation_rules(self, tool_args):
        args_validation_rules = self._set_args_validation_rules()
        
        invalid_args = dict()
        
        for arg in self.args_schema.model_fields.keys():
            for rule in args_validation_rules.get(arg, []):
                invalid_reason = None
                invalid_reason = rule(**tool_args)
                if invalid_reason is not None:
                    invalid_args[arg] = invalid_reason 
                    continue
                
        if len(invalid_args) > 0:
            self.execution_confirmed = False
            raise BaseToolInterrupt(
                interrupt_tool = self.name,
                interrupt_type = BaseToolInterrupt.BaseToolInterruptType.INVALID_ARGS,
                interrupt_reason = f"Invalid arguments: {list(invalid_args.keys())}.",
                interrupt_data = {
                    "invalid_args": invalid_args,
                    "args_schema": self.args_schema.model_fields
                }
            )
            
    # DOC: Infer argument values based on current provided values and one function reated to argument { argname: test(**tool_args) -> inferred_value , ... } 
    def _set_args_inference_rules(self):
        return { arg: None for arg in self.args_schema.model_fields.keys() }
    
    def infer_args(self, tool_args):
        original_tool_args = tool_args.copy()
        args_inference_rules = self._set_args_inference_rules()
        for arg in self.args_schema.model_fields.keys():
            if arg in args_inference_rules and args_inference_rules[arg] is not None:
                tool_args[arg] = args_inference_rules[arg](**tool_args)
    
    
    # DOC: Confirm args if needed 
    def confirm_args(self, tool_args): 
        if not self.execution_confirmed:
            raise BaseToolInterrupt(
                interrupt_tool = self.name,
                interrupt_type = BaseToolInterrupt.BaseToolInterruptType.CONFIRM_ARGS,
                interrupt_reason = "Please confirm the execution of the tool with the provided arguments.",
                interrupt_data = {
                    "args": tool_args,
                    "args_schema": self.args_schema.model_fields,
                }
            )
            
    # DOC: Confirm output if needed      
    def confirm_ouputs(self, tool_args):
        if not self.output_confirmed:
            raise BaseToolInterrupt(
                interrupt_tool = self.name,
                interrupt_type = BaseToolInterrupt.BaseToolInterruptType.CONFIRM_OUTPUT,
                interrupt_reason = "A user confirmation of the ouput is required.",
                interrupt_data = {
                    "args": tool_args,
                    "output": self.output
                }
            )
            
    # DOC: Tool esecution, what this returns will be settet as tool.ouput so this should be overriden by the user
    def _execute(self, **tool_args):
        return None
                
    
    # DOC: Run tool with the given arguments, this function should be overridden by the user that will call super() to do args validation and confirmation
    def _run(
        self, 
        tool_args: dict = None,
        run_manager: None | Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        """Run the tool with the given arguments."""
        
        def controls_before_execution(tool_args):
            self.check_required_args(tool_args)         # 1. Required arguments
            self.check_validation_rules(tool_args)      # 2. Invalid arguments
            self.infer_args(tool_args)                  # 3. Infer arguments)
            self.confirm_args(tool_args)                # 4. Confirm arguments
            
        controls_before_execution(tool_args)
        
        self.output = self._execute(**tool_args)
        
        def controls_after_execution(tool_args):
            self.confirm_ouputs(tool_args)              # 5. Confirm output
        
        controls_after_execution(tool_args)
        
        return self.output