import os
import datetime
from dateutil import relativedelta
from enum import Enum
import requests

from typing import Optional, Literal, Union, List, Dict, Any
from pydantic import BaseModel, Field, AliasChoices, field_validator, model_validator

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from agent import utils, s3_utils
from agent import names as N
from agent.common import states as GraphStates
from agent.nodes.base import BaseAgentTool


_URI_HINT = "HTTP(S) URL, S3 URI (s3://...)"


class SaferRainInputSchema(BaseModel):
    """
    Run a flood simulation using a terrain elevation raster (DEM/DTM) and rainfall input
    (either a single numeric amount applied uniformly or a rainfall raster).
    If the rainfall raster is multiband, bands are interpreted as a time series and
    can be cumulatively summed over a band range.
    """

    # ----------------------------- Required inputs -----------------------------
    dem: str = Field(
        ...,
        title="DEM (GeoTIFF)",
        description=(
            "Digital Elevation Model raster used as ground elevation.\n"
            "- You can pass a direct URL, S3 URI, or local path to a GeoTIFF.\n"
            "- Or you can reference an **existing project raster layer** "
            "from the Layer Registry (e.g., when the user says 'use the DTM of Rome').\n"
            "- When a layer is referenced, use that layer's `src` value."
        ),
        examples=[
            "https://example.com/dem_10m.tif",
            "s3://bucket/project/dtm_rome.tif",
            "Rome DTM"
        ],
        validation_alias=AliasChoices("dem", "dtm", "elevation", "dem_path"),
    )

    rain: Union[str, float] = Field(
        ...,
        title="Rainfall input (raster or constant)",
        description=(
            "Rainfall data for the simulation.\n"
            "- It can be a **numeric value** (uniform rainfall in millimeters applied to the whole DEM extent).\n"
            "- Or a URL, S3 URI, or local path to a rainfall raster (GeoTIFF).\n"
            "- It can be a reference an **existing project raster layer** "
            "from the Layer Registry (e.g., 'use layer rainfall-*').\n"
            "- When a layer is referenced, the tool will internally use that layer's `src` value."
        ),
        examples=[
            25.0,
            "https://example.com/rainfall_2025_05.tif",
            "s3://bucket/project/rainfall_v1.tif",
            "Rainfall V1"
        ],
        validation_alias=AliasChoices("rain", "rainfall", "rain_path", "precip", "precipitation"),
    )

    water: Optional[str] = Field(
        default=None,
        title="Output Water Depth (GeoTIFF, optional)",
        description=(
            f"Destination {_URI_HINT} where the simulated water depth raster (GeoTIFF) will be written. "
            "If omitted, the tool returns the path/URI produced by the execution environment."
        ),
        examples=[
            "https://example.com/outputs/water_depth.tif",
            "s3://my-bucket/floods/wd.tif",
        ],
        validation_alias=AliasChoices("water", "waterdepth", "wd", "water_path"),
    )

    # ------------------------------- Parameters --------------------------------
    band: int = Field(
        default=1,  # ???: Default should be None (ora t leas 1) → FIRST
        title="Rain band start (1-based)",
        description=(
            "For multiband rainfall rasters: index of the first band to use (1-based). "
            "If `rain` is numeric (constant), this is ignored."
        ),
        examples=[1],
        validation_alias=AliasChoices("band", "rain_band", "input_band"),
    )

    to_band: int = Field(
        default=1,  # ???: Default should be None (ora t least -1) → LAST
        title="Rain band end (1-based, inclusive)",
        description=(
            "For multiband rainfall rasters: index of the last band to include (inclusive, 1-based). "
            "If `to_band` > `band`, rainfall is cumulatively summed over bands [band..to_band]. "
            "If `rain` is numeric (constant), this is ignored."
        ),
        examples=[1, 3],
        validation_alias=AliasChoices("to_band", "target_band", "out_band", "end_band"),
    )

    t_srs: Optional[str] = Field(
        default=None,
        title="Target SRS (EPSG)",
        description=(
            "Target spatial reference for outputs (e.g., 'EPSG:32633'). "
            "If None, the DEM CRS is used."
        ),
        examples=["EPSG:32633", "EPSG:4326"],
        validation_alias=AliasChoices("t_srs", "target_srs", "crs", "out_crs"),
    )

    mode: Literal["lambda", "batch"] = Field(
        default="lambda",
        title="Execution mode",
        description='Execution backend: "lambda" for AWS Lambda, "batch" for AWS Batch. Default is "lambda".',
        examples=["batch", "lambda"],
        validation_alias=AliasChoices("mode", "execution_mode", "run_mode"),
    )

    
# DOC: This is a demo tool to retrieve weather data.
class SaferRainTool(BaseAgentTool):
        
    
    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.SAFER_RAIN_TOOL,
            description = (
                "Run a **flood simulation** using a Digital Elevation Model (DEM) and rainfall input.\n\n"
                "Rainfall can be provided as:\n"
                "- A **constant numeric value** in millimeters (applied uniformly across the DEM extent), or\n"
                "- A **rainfall raster** (GeoTIFF). If the raster is **multiband**, each band represents a time step, "
                "and rainfall can be **cumulatively summed** between `band` and `to_band`.\n\n"
                "**Layer Registry Integration:**\n"
                "- The project context may include a set of preloaded geospatial layers (DEM, rainfall, etc.).\n"
                "- For `dem` and `rain`, you can pass either a direct URL/S3 URI or **reference an existing project layer** "
                "by its `src` as shown in the Layer Registry.\n"
                "- When a layer is referenced by title, the tool will internally resolve it and use its `src` as input.\n\n"
                "### Outputs:\n"
                "- A **water-depth raster (GeoTIFF)** representing simulated flood depths over the DEM area. "
                "If the `water` argument is omitted, the tool will return the path/URI of the generated file."
            ),
            args_schema = SaferRainInputSchema,
            **kwargs
        )
        self.execution_confirmed = False
        self.output_confirmed = True


    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        return dict()
        
    
    # DOC: Inference rules ( i.e.: from location name to bbox ... )
    def _set_args_inference_rules(self) -> dict:
        def infer_water(**kwargs):
            """
            Infer the S3 bucket destination based on user ID and project ID.
            """
            water = kwargs.get('water', f"saferrain-out.tif")
            return f"{s3_utils._BASE_BUCKET}/saferrain-out/{water}"
            
        infer_rules = {
            'water': infer_water
        }
        return infer_rules
        
    
    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file
    def _execute(
        self,
        /,
        **kwargs: Any,  # dict[str, Any] = None,
    ): 
        # DOC: Call the SaferBuildings API ...
        # api_root_local = "http://localhost:5000" # TEST: only when running locally
        # api_url = f"{os.getenv('SAFERPLACES_API_ROOT')}/processes/safer-rain-process/execution"
        # payload = { 
        #     "inputs": kwargs  | {
        #         "token": os.getenv("SAFERPLACES_API_TOKEN"),
        #         "user": os.getenv("SAFERPLACES_API_USER"),
        #     } | {
        #         "debug": True,  # TEST: enable debug mode
        #     }
        # }
        # print(f"Executing {self.name} with args: {payload}")
        # response = requests.post(api_url, json=payload)
        # print(f"Response status code: {response.status_code} - {response.content}")
        # response = response.json() 
        # # TODO: Check output_code ...

        # TEST: Simulate a response for testing purposes
        api_response = {
            # DOC: Use this when running with true API
            # 'water_depth_file': kwargs.get('water', f"saferplaces.co/SaferPlaces-Agent/dev/user=={self.graph_state.get('user_id', 'test')}/safer-rain-water.tif"),
            
            # TEST: This is a simulated response for testing purposes
            'water_depth_file': "s3://saferplaces.co/Directed/data-fabric-rwl2/Rimini_coast_cropped_buildings_rain_240mm.tif",
        
        }

        # TODO: Check if the response is valid
        
        tool_response = {
            'tool_response': api_response,
            'updates': {
                'layer_registry': self.graph_state.get('layer_registry', []) + [
                    {
                        'title': f"SaferRain Output",
                        'description': f"SaferRain output file with flooding waterdepth from this inputs: ({', '.join([f'{k}: {v}' for k,v in kwargs.items() if k!='water'])})",
                        'src': api_response['water_depth_file'],
                        'type': 'raster',
                        'metadata': dict()
                    }
                ]
                if not GraphStates.src_layer_exists(self.graph_state, api_response['water_depth_file'])
                else []
            }
        }
        
        # tool_response = {
        #     'saferrain_response': api_response,
            
        #     'map_actions': [
        #         # {
        #         #     'action': 'new_layer',
        #         #     'layer_data': {
        #         #         'name': 'digital twin dem',
        #         #         'type': 'raster',
        #         #         'src': api_response['water_depth_file'],
        #         #         'styles': [
        #         #             { 'name': 'waterdepth', 'type': 'scalar', 'colormap': 'blues' }
        #         #         ]
        #         #     }
        #         # }
        #         utils.map_action_new_layer(
        #             layer_name = 'digital twin dem',
        #             layer_src = api_response['water_depth_file'],
        #             layer_styles = [
        #                 { 'name': 'waterdepth', 'type': 'scalar', 'colormap': 'blues' }
        #             ]
        #         )
        #     ]
        # }
        
        # print('\n', '-'*80, '\n')
        
        return tool_response
        
    
    # DOC: Back to a consisent state
    def _on_tool_end(self):
        self.execution_confirmed = False
        self.output_confirmed = True
        
    
    # DOC: Try running AgentTool → Will check required, validity and inference over arguments thatn call and return _execute()
    def _run(
        self, 
        /,
        **kwargs: Any, # dict[str, Any] = None,
    ) -> dict:
        
        run_manager: Optional[CallbackManagerForToolRun] = kwargs.pop("run_manager", None)
        return super()._run(
            tool_args = kwargs,
            run_manager = run_manager
        )