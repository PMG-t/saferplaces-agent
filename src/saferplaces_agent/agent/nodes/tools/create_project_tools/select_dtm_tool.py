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
class CreateProjectSelectDTMTool(BaseAgentTool):
    
    
    # DOC: Tool input schema
    class InputSchema(BaseModel):
        
        area: None | str | list[float] = Field(
            title = "Area",
            description = """The area of interest for the weather data. If not specified use None as default.
            It could be a bouning-box defined by [min_x, min_y, max_x, max_y] coordinates provided in EPSG:4326 Coordinate Reference System.
            Otherwise it can be the name of a country, continent, or specific geographic area.""",
            examples=[
                None,
                "Italy",
                "Paris",
                "Continental Spain",
                "Alps",
                [12, 52, 14, 53],
                [-5.5, 35.2, 5.58, 45.10],
            ],
            default = None
        )
        crs: None | str = Field(
            title = "CRS",
            description = f"The Coordinate Reference System (CRS) for the area coordinates. If not specified, use None as default. The default is EPSG:4326.",
            examples = [
                None,
                "EPSG:4326",
                "EPSG:3004",
                "EPSG:32632",
                "EPSG:7791",
            ],
            default = None
        )
        dtm_file: None | str = Field(
            title = "DTM File",
            description = """The path to a Digital Terrain Model (DTM) remote file. If not specified, use None as default.""",
            examples=[
                None,
                "s3://my-bucket/dtm_file.geotiff"
            ],
            default = None
        )     
           
    
    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.CREATE_PROJECT_SELECT_DTM_TOOL,
            description = """Useful when user wants to select a DTM for a new project. New DTM can be provided by bbox coordinates or a location name plus a CRS (EPSG:4326).
            Otherwise user can upload a DTM file.""",
            args_schema = CreateProjectSelectDTMTool.InputSchema,
            **kwargs
        )
        self.output_confirmed = True    # INFO: There is already the execution_confirmed:True
        
    
    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        
        return {
            'area': [
                lambda **ka: f"Invalid area coordinates: {ka['area']}. It should be a list of 4 float values representing the bounding box [min_x, min_y, max_x, max_y]." 
                    if isinstance(ka['area'], list) and len(ka['area']) != 4 else None,
                lambda **ka: f"Area cordinates or name (and relative output CRS) must be provided if dtm_file is not specified."
                    if ka['dtm_file'] is None and (ka['area'] is None) else None,
            ],
            'crs' : [
                lambda **ka: f"Invalid CRS: {ka['crs']}. It should be a valid EPSG code or None."
                    if ka['crs'] is not None and 'EPSG' not in ka['crs'] else None,
                lambda **ka: f"CRS must be provided if area is specified as a list of coordinates."
                    if ka['dtm_file'] is None and ka['area'] is not None and ka['crs'] is None else None,
            ],
            'dtm_file': [
                lambda **ka: f"Invalid DTM file path: {ka['dtm_file']}. It should be a valid file path or None."
                    if ka['dtm_file'] is not None and not (os.path.isfile(ka['dtm_file']) or ka['dtm_file'].startswith('s3://')) else None,
                lambda **ka: f"DTM file must be provided if area is not specified."
                    if ka['dtm_file'] is None and (ka['area'] is None) else None,
            ],
        }
        
    
    # DOC: Inference rules ( i.e.: from location name to bbox ... )
    def _set_args_inference_rules(self) -> dict:
        
        def infer_area(**ka):
            def bounding_box_from_location_name(area):
                if type(area) is str:
                    area = utils.ask_llm(
                        role = 'system',
                        message = f"""Please provide the bounding box coordinates for the area: {area} with format [min_x, min_y, max_x, max_y] in EPSG:4326 Coordinate Reference System. 
                        Provide only the coordinates list without any additional text or explanation.""",
                        eval_output = True
                    )
                    self.execution_confirmed = False
                return area
            def round_bounding_box(area):
                if type(area) is list:
                    precision = 1
                    area = [
                        utils.floor_decimals(area[0], precision),
                        utils.floor_decimals(area[1], precision),
                        utils.ceil_decimals(area[2], precision),
                        utils.ceil_decimals(area[3], precision)
                    ]
                return area
            area = bounding_box_from_location_name(ka['area'])
            area = round_bounding_box(area)
            return area
        
        
        return {
            'area': infer_area,
        }
        
    
    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file
    def _execute(
        self,
        area: None | str | list[float] = None,
        crs: None | str = None,
        dtm_file: None | str = None,
    ): 
        # DOC: Download / Gather DTM and then store it to bucket
        dtm_project_file = 's3://saferplaces-projects/fake_dtm_project_file.geotiff'
        
        return {
            "dtm_project_file": dtm_project_file,
        }
        
    
    # DOC: Back to a consisent state
    def _on_tool_end(self):
        self.execution_confirmed = False
        self.output_confirmed = True
        
    
    # DOC: Try running AgentTool → Will check required, validity and inference over arguments thatn call and return _execute()
    def _run(
        self, 
        area: None | str | list[float] = None,
        crs: None | str = None,
        dtm_file: None | str = None,
        run_manager: None | Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        
        return super()._run(
            tool_args = {
                'area': area,
                'crs': crs,
                'dtm_file': dtm_file,
            },
            run_manager=run_manager,
        )