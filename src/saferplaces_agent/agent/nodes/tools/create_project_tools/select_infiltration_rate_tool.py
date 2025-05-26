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
class CreateProjectSelectInfiltrationRateTool(BaseAgentTool):
    
    
    # DOC: Tool input schema
    class InputSchema(BaseModel):
        
        infiltration_rate_file: None | str = Field(
            title = "Infiltration File",
            description = """The path to a infilitration raster file.""",
            examples=[
                None,
                "s3://my-bucket/infiltration_rate_file.tif"
            ],
            default = None
        ) 
        
    
    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.CREATE_PROJECT_SELECT_INFILTRATION_TOOL,
            description = """Useful when user wants to get infiltartion rate in a new project. Infiltartion rate is not a mandatory information but it can be uploaded from user.""",
            args_schema = CreateProjectSelectInfiltrationRateTool.InputSchema,
            **kwargs
        )
        self.execution_confirmed = True     # DOC: This tool needs to be executed fast forward
        self.output_confirmed = True
        
    
    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        
        return {
            'infiltration_rate_file': [
                lambda **ka: f"Invalid infiltration rate file path: {ka['infiltration_rate_file']}. It should be a valid file path or None."
                    if ka['infiltration_rate_file'] is not None and not (os.path.isfile(ka['infiltration_rate_file']) or ka['infiltration_rate_file'].startswith('s3://')) else None
            ]
        }
        
    
    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file
    def _execute(
        self,
        infiltration_rate_file: None | str = None,
    ): 
        if infiltration_rate_file is not None:
        
            # DOC: Download / Gather Buildings and then store it to bucket
            infiltration_project_file = 's3://saferplaces-projects/fake_infiltration_rate_project_file.shp'
            
            return {
                "infiltration_project_file": infiltration_project_file,
            }
            
        else:
            return {
                "infiltration_project_file": None,
            }
        
    
    # DOC: Back to a consisent state
    def _on_tool_end(self):
        self.execution_confirmed = True
        self.output_confirmed = True
        
    
    # DOC: Try running AgentTool → Will check required, validity and inference over arguments thatn call and return _execute()
    def _run(
        self,
        infiltration_rate_file: None | str = None,
        run_manager: None | Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        
        return super()._run(
            tool_args = {
                'infiltration_rate_file': infiltration_rate_file,
            },
            run_manager=run_manager,
        )