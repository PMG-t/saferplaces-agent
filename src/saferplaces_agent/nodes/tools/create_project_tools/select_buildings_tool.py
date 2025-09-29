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
class CreateProjectSelectBuildingsTool(BaseAgentTool):
    
    
    # DOC: Tool input schema
    class InputSchema(BaseModel):
        
        do_download: bool = Field(
            title = "Download Buildings",
            description = """If True, the tool will automatically download the buildings in the specified area.
            If False, the user will be prompted to upload a geo-features file containing the buildings. Default is True.""",
            examples = [
                True,
                False,
            ],
            default = True
        )
        buildings_file: None | str = Field(
            title = "Buildings File",
            description = """The path to a geo-features file containing the buildings.""",
            examples=[
                None,
                "s3://my-bucket/buildings_file.shp"
            ],
            default = None
        ) 
        
    
    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.CREATE_PROJECT_SELECT_BUILDINGS_TOOL,
            description = """Useful when user wants to get buildings in a new project. Buildings can be automatically downloaded otherwise user can upload a geo-features file.""",
            args_schema = CreateProjectSelectBuildingsTool.InputSchema,
            **kwargs
        )
        self.execution_confirmed = True     # DOC: This tool needs to be executed fast forward
        self.output_confirmed = True
        
    
    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        
        return {
            'do_download': [
                lambda **ka: f"Invalid value for do_download: {ka['do_download']}. It should be a boolean value (True or False)."
                    if not isinstance(ka['do_download'], bool) else None
            ],
            'buildings_file': [
                lambda **ka: f"Invalid buildings file path: {ka['buildings_file']}. It should be a valid file path or None."
                    if ka['buildings_file'] is not None and not (os.path.isfile(ka['buildings_file']) or ka['buildings_file'].startswith('s3://')) else None,
                lambda **ka: f"No buildgs file needs to be provided if do_download is True."
                    if ka['do_download'] and ka['buildings_file'] is not None else None,
                lambda **ka: f"Buildings file must be provided if do_download is False."
                    if not ka['do_download'] and ka['buildings_file'] is None else None,
            ]
        }
        
    
    # DOC: Inference rules ( i.e.: from location name to bbox ... )
    def _set_args_inference_rules(self) -> dict:
        
        return {
            'do_download': lambda **ka: ka['buildings_file'] is None,
        }
        
    
    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file
    def _execute(
        self,
        do_download: bool = True,
        buildings_file: None | str = None,
    ): 
        # DOC: Download / Gather Buildings and then store it to bucket
        buildings_project_file = 's3://saferplaces-projects/fake_buildings_project_file.shp'
        
        return {
            "buildings_project_file": buildings_project_file,
        }
        
    
    # DOC: Back to a consisent state
    def _on_tool_end(self):
        self.execution_confirmed = True
        self.output_confirmed = True
        
    
    # DOC: Try running AgentTool → Will check required, validity and inference over arguments thatn call and return _execute()
    def _run(
        self, 
        do_download: bool = True,
        buildings_file: None | str = None,
        run_manager: None | Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        
        return super()._run(
            tool_args = {
                'do_download': do_download,
                'buildings_file': buildings_file,
            },
            run_manager=run_manager,
        )