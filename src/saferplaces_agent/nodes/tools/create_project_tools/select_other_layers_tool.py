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

from ....common import utils
from ....common import names as N
from ....nodes.base import BaseAgentTool



# DOC: This is a demo tool to retrieve weather data.
class CreateProjectSelectOtherLayersTool(BaseAgentTool):
    
    
    # DOC: Tool input schema
    class InputSchema(BaseModel):
        
        other_layers_file: None | str = Field(
            title = "Other Layers File",
            description = """The path to other layers raster file(s).""",
            examples=[
                None,
                "s3://my-bucket/other_layer.tif",
                "s3://my-bucket/other_layers.zip"
            ],
            default = None
        ) 
        
    
    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.CREATE_PROJECT_SELECT_OTHER_LAYERS_TOOL,
            description = """Useful when user wants to get other layers in a new project. Other layers are not a mandatory information but they can be uploaded from user.""",
            args_schema = CreateProjectSelectOtherLayersTool.InputSchema,
            **kwargs
        )
        self.execution_confirmed = True     # DOC: This tool needs to be executed fast forward
        self.output_confirmed = True
        
    
    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        
        return {
            'other_layers_file': [
                lambda **ka: f"Invalid lithology file path: {ka['other_layers_file']}. It should be a valid file path or None."
                    if ka['other_layers_file'] is not None and not (os.path.isfile(ka['other_layers_file']) or ka['other_layers_file'].startswith('s3://')) else None
            ]
        }
        
    
    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file
    def _execute(
        self,
        other_layers_file: None | str = None,
    ): 
        if other_layers_file is not None:
        
            # DOC: Download / Gather Buildings and then store it to bucket
            other_layers_project_file = 's3://saferplaces-projects/fake_other_layers_project_file.shp'
            
            return {
                "other_layers_project_file": other_layers_project_file,
            }
            
        else:
            return {
                "other_layers_project_file": None,
            }
        
    
    # DOC: Back to a consisent state
    def _on_tool_end(self):
        self.execution_confirmed = True
        self.output_confirmed = True
        
    
    # DOC: Try running AgentTool → Will check required, validity and inference over arguments thatn call and return _execute()
    def _run(
        self,
        other_layers_file: None | str = None,
        run_manager: None | Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        
        return super()._run(
            tool_args = {
                'other_layers_file': other_layers_file,
            },
            run_manager=run_manager,
        )