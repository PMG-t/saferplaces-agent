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
class FloodingRainfallDefineRainTool(BaseAgentTool):
    
    
    # DOC: Tool input schema
    class InputSchema(BaseModel):
        
        rain_type: None | str = Field (
            title = "Rain type",
            description = """The type of rainfall to be used in the project. This can be one of the following:
            - uniform: Uniform rainfall across the area.
            - non-uniform-draw: Non-uniform rainfall defined by a polygon drawn on the map.
            - non-uniform-file: Non-uniform rainfall defined by a file (e.g., shapefile).
            - safer003: A specific type of rainfall used in the Safer003 project (not used yet).""",
            examples = [
                "uniform",
                "non-uniform-draw",
                "non-uniform-file",
                "safer003"  # ???: Don't know what this is
            ],
            default = "uniform"  # Default to uniform rainfall
        )
        rain_mm: None | Optional[float] = Field (
            title = "Rainfall in mm",
            description = """The amount of rainfall in millimeters. This is only used for uniform rainfall type.""",
            examples = [
                10.0,
                20.0,
                30.0
            ],
            default = None
        )
        non_uniform_polygon: None | Optional[str] = Field (
            title = "Non-uniform rainfall polygon",
            description = """Poligon coordinates for non-uniform rainfall type. This is only used for non-uniform rainfall type.""",
            examples = [
                ((12.0, 34.0), (13.0, 35.0), (14.0, 36.0)),
            ],
            default = None
        )
        non_uniform_file: None | Optional[str] = Field (
            title = "Non-uniform rainfall file",
            description = """File path for non-uniform rainfall type. This is only used for non-uniform rainfall type.""",
            examples = [
                "s3://saferplaces-projects/non_uniform_rainfall_file.shp"
            ],
            default = None
        )
        rain_duration: None | float = Field (
            title = "Rainfall duration",
            description = """The duration of the rainfall in hours. This is used for all rainfall types.""",
            examples = [
                0.5,  # 30 minutes
                1, 2, 3, 4,  # 1 to 4 hours
            ],
            default = 1.0  # Default to 1 hour
        )

    
    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.FLOODING_RAINFALL_DEFINE_RAIN_TOOL,
            description = """Useful when user wants to define rainfall parameters for flooding simulation.""",
            args_schema = FloodingRainfallDefineRainTool.InputSchema,
            **kwargs
        )
        self.execution_confirmed = False
        self.output_confirmed = True
        
    
    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        
        return {
            'rain_type': [
                lambda **ka: f"Invalid rain type: {ka['rain_type']}. It should be one of the following: uniform, non-uniform-draw, non-uniform-file, safer003."
                    if ka['rain_type'] not in ['uniform', 'non-uniform-draw', 'non-uniform-file', 'safer003'] else None
            ],
            'rain_mm': [
                lambda **ka: f"Invalid rainfall in mm: {ka['rain_mm']}. It should be a positive float value."
                    if ka['rain_mm'] is not None and (not isinstance(ka['rain_mm'], (int, float)) or ka['rain_mm'] <= 0) else None, 
                lambda **ka: f"Rainfall in mm should not be provided if rain type is uniform."
                    if ka['rain_type'] != 'uniform' and ka['rain_mm'] is not None else None,
            ],
            'non_uniform_polygon': [
                lambda **ka: f"Invalid non-uniform polygon: {ka['non_uniform_polygon']}. It should be a valid polygon coordinates or None."
                    if ka['non_uniform_polygon'] is not None and not isinstance(ka['non_uniform_polygon'], (tuple, list)) else None,
                lambda **ka: f"Non-uniform polygon should not be provided if rain type is not non-uniform-draw."
                    if ka['rain_type'] != 'non-uniform-draw' and ka['non_uniform_polygon'] is not None else None,
            ],
            'non_uniform_file': [
                lambda **ka: f"Invalid non-uniform file: {ka['non_uniform_file']}. It should be a valid file path or None."
                    if ka['non_uniform_file'] is not None and not (os.path.isfile(ka['non_uniform_file']) or ka['non_uniform_file'].startswith('s3://')) else None,
                lambda **ka: f"Non-uniform file should not be provided if rain type is not non-uniform-file."
                    if ka['rain_type'] != 'non-uniform-file' and ka['non_uniform_file'] is not None else None,
            ],
            'rain_duration': [
                lambda **ka: f"Invalid rainfall duration: {ka['rain_duration']}. It should be a positive float value representing hours."
                    if not isinstance(ka['rain_duration'], (int, float)) or ka['rain_duration'] <= 0 else None,
            ]
        }
        
    
    # DOC: Inference rules ( i.e.: from location name to bbox ... )
    def _set_args_inference_rules(self) -> dict:
        
        return {
            'rain_type': lambda **ka: 'uniform',
            'rain_mm': lambda **ka: 100.0 if ka['rain_type'] == 'uniform' else None,
            'rain_duration': lambda **ka: 1.0
        }
        
    
    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file
    def _execute(
        self,
        rain_type: None | str = None,
        rain_mm: None | Optional[float] = None,
        non_uniform_polygon: None | Optional[str] = None,
        non_uniform_file: None | Optional[str] = None,
        rain_duration: None | float = None,
    ): 
        # DOC: return dicitionry with only necessry rain definition args
        
        rain_type_args = dict()
        if rain_type == 'uniform':
            rain_type_args['rain_mm'] = rain_mm
        elif rain_type == 'non-uniform-draw':
            rain_type_args['non_uniform_polygon'] = non_uniform_polygon
        elif rain_type == 'non-uniform-file':
            rain_type_args['non_uniform_file'] = non_uniform_file
        
        return {
            'rain_type': rain_type,
            'rain_duration': rain_duration,
            ** rain_type_args,
        }
        
    
    # DOC: Back to a consisent state
    def _on_tool_end(self):
        self.execution_confirmed = False
        self.output_confirmed = True
        
    
    # DOC: Try running AgentTool → Will check required, validity and inference over arguments thatn call and return _execute()
    def _run(
        self, 
        rain_type: None | str = None,
        rain_mm: None | Optional[float] = None,
        non_uniform_polygon: None | Optional[str] = None,
        non_uniform_file: None | Optional[str] = None,
        rain_duration: None | float = None,
        run_manager: None | Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        
        return super()._run(
            tool_args = {
                'rain_type': rain_type,
                'rain_mm': rain_mm,
                'non_uniform_polygon': non_uniform_polygon,
                'non_uniform_file': non_uniform_file,
                'rain_duration': rain_duration
            },
            run_manager=run_manager,
        )