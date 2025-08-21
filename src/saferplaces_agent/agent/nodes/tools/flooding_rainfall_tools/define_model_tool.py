import os
import datetime
from dateutil import relativedelta
from enum import Enum

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from agent import utils
from agent import names as N
from agent.nodes.base import BaseAgentTool



# DOC: This is a demo tool to retrieve weather data.
class FloodingRainfallDefineModelTool(BaseAgentTool):
    
    
    # DOC: Tool input schema
    class InputSchema(BaseModel):
        
        model_name: None | str = Field (
            title = "Model name",
            description = """The name of the rainfall model to be used in the project. This can be one of the following:
            - saferplaces: Saferplaces rainfall model, which is simple static model.
            - untrim: Untrim model, which is a more complex dynamic model that can be used for flooding projects.
            """,
            examples = [
                "saferplaces",
                "untrim"
            ],
            default = "saferplaces"
        )
        simulation_time: None | Optional[int] = Field (
            title = "Simulation time",
            description = """The time in hours for which the rainfall model will be simulated. This is only used for untrim model.""",
            examples = [
                1, 2, 3, 4, 6, 
                12, 24, 48
            ],
            default = None
        )
        manning_coefficient: None | Optional[float] = Field (
            title = "Manning coefficient",
            description = """The Manning coefficient for the rainfall model. This is only used for untrim model.""",
            examples = [
                0.01, 0.02, 0.05, 
                0.1, 0.2, 0.5,
                1.0, 2.0, 3.0, 4.0, 5.0
            ],  
            default = None
        )
        nl: None | Optional[int] = Field (
            title = "Number of layers",
            description = """The number of pixel for each element side. This is only used for untrim model.""",
            examples = [
                1, 2, 3, 4, 5,
                10, 20, 25, 50, 100
            ],
            default = None
        )
        delta_t: None | Optional[int] = Field (
            title = "Time step",
            description = """The time step in seconds for the rainfall model. Lower make simulation more precise but slower. A good value is between [600,900]s. This is only used for untrim model.""",
            examples = [
                6,
                60, 120, 300,
                600, 900
            ],
            default = None
        )
        time_shot_interval: None | Optional[int] = Field (
            title = "Time shot interval",
            description = """The time interval in seconds for the rainfall model. This is only used for untrim model.""",
            examples = [
                600, 900, 1800,
                3600, 7200, 10800
            ],
            default = None
        )
        apply_damage: None | Optional[bool] = Field (
            title = "Apply damage",
            description = """Whether to apply damage to the model.""",
            examples = [
                True,
                False
            ],
            default = True
        )
            

    
    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.FLOODING_RAINFALL_DEFINE_MODEL_TOOL,
            description = """Useful when user wants to define the rainfall model for the flooding project.""",
            args_schema = FloodingRainfallDefineModelTool.InputSchema,
            **kwargs
        )
        self.execution_confirmed = False
        self.output_confirmed = True
        
    
    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        
        return {
            'model_name': [
                lambda **ka: f"Invalid model name: {ka['model_name']}. It should be one of ['saferplaces', 'untrim']."
                    if ka['model_name'] not in ['saferplaces', 'untrim'] else None,
            ],
            'simulation_time': [
                lambda **ka: f"Invalid simulation time: {ka['simulation_time']}. It should be a positive integer representing hours between 1 and 48."
                    if ka['model_name'] == 'untrim' and (ka['simulation_time'] is not None and (not isinstance(ka['simulation_time'], int) or ka['simulation_time'] < 1 or ka['simulation_time'] > 48)) else None,
            ],
            'manning_coefficient': [
                lambda **ka: f"Invalid Manning coefficient: {ka['manning_coefficient']}. It should be a positive float value between 0 and 5."
                    if ka['model_name'] == 'untrim' and (ka['manning_coefficient'] is not None and (not isinstance(ka['manning_coefficient'], (int, float)) or ka['manning_coefficient'] < 0 or ka['manning_coefficient'] > 5)) else None,
            ],
            'nl': [
                lambda **ka: f"Invalid number of layers: {ka['nl']}. It should be a positive integer between 1 and 100."
                if ka['model_name'] == 'untrim' and (ka['nl'] is not None and (not isinstance(ka['nl'], int) or ka['nl'] < 1 or ka['nl'] > 100)) else None,
            ],
            'delta_t': [
                lambda **ka: f"Invalid time step: {ka['delta_t']}. It should be a positive integer representing seconds between 6 and 900."
                    if ka['model_name'] == 'untrim' and (ka['delta_t'] is not None and (not isinstance(ka['delta_t'], int) or ka['delta_t'] < 6 or ka['delta_t'] > 900)) else None,
            ],
            'time_shot_interval': [
                lambda **ka: f"Invalid time shot interval: {ka['time_shot_interval']}. It should be a positive integer representing seconds between 600 and 10800."
                    if ka['model_name'] == 'untrim' and (ka['time_shot_interval'] is not None and (not isinstance(ka['time_shot_interval'], int) or ka['time_shot_interval'] < 600 or ka['time_shot_interval'] > 10800)) else None,
            ],
            'apply_damage': [
                lambda **ka: f"Invalid apply damage: {ka['apply_damage']}. It should be a boolean value."
                    if ka['apply_damage'] is not None and not isinstance(ka['apply_damage'], bool) else None,
            ]   
        }
        
    
    # DOC: Inference rules ( i.e.: from location name to bbox ... )
    def _set_args_inference_rules(self) -> dict:
        
        return {
            'model_name': lambda **ka: 'saferplaces',
            
            'simulation_time': lambda **ka: 12 if ka['model_name'] == 'untrim' and ka['simulation_time'] is None else ka['simulation_time'],
            'manning_coefficient': lambda **ka: 0.02 if ka['model_name'] == 'untrim' and ka['manning_coefficient'] is None else ka['manning_coefficient'],
            'nl': lambda **ka: 50 if ka['model_name'] == 'untrim' and ka['nl'] is None else ka['nl'],
            'delta_t': lambda **ka: 600 if ka['model_name'] == 'untrim' and ka['delta_t'] is None else ka['delta_t'],
            'time_shot_interval': lambda **ka: 3600 if ka['model_name'] == 'untrim' and ka['time_shot_interval'] is None else ka['time_shot_interval'],
            
            'apply_damage': lambda **ka: True if ka['apply_damage'] is None else ka['apply_damage']
        }
        
    
    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file
    def _execute(
        self,
        model_name: None | str = None,
        simulation_time: None | Optional[int] = None,
        manning_coefficient: None | Optional[float] = None,
        nl: None | Optional[int] = None,
        delta_t: None | Optional[int] = None,
        time_shot_interval: None | Optional[int] = None,
        apply_damage: None | Optional[bool] = None,
    ): 
        # DOC: return dicitionry with only necessry rain definition args
        
        untrim_args = dict()
        if model_name == 'untrim':
            untrim_args = {
                'simulation_time': simulation_time,
                'manning_coefficient': manning_coefficient,
                'nl': nl,
                'delta_t': delta_t,
                'time_shot_interval': time_shot_interval
            }
        
        return {
            'model_name': model_name,
            'apply_damage': apply_damage,
            ** untrim_args
        }
        
    
    # DOC: Back to a consisent state
    def _on_tool_end(self):
        self.execution_confirmed = False
        self.output_confirmed = True
        
    
    # DOC: Try running AgentTool → Will check required, validity and inference over arguments thatn call and return _execute()
    def _run(
        self, 
        model_name: None | str = None,
        simulation_time: None | Optional[int] = None,
        manning_coefficient: None | Optional[float] = None,
        nl: None | Optional[int] = None,
        delta_t: None | Optional[int] = None,
        time_shot_interval: None | Optional[int] = None,
        apply_damage: None | Optional[bool] = None,
        run_manager: None | Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        
        return super()._run(
            tool_args = {
                'model_name': model_name,
                'simulation_time': simulation_time,
                'manning_coefficient': manning_coefficient,
                'nl': nl,
                'delta_t': delta_t,
                'time_shot_interval': time_shot_interval,
                'apply_damage': apply_damage
            },
            run_manager=run_manager,
        )