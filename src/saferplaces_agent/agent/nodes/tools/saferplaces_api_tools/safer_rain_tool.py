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

from agent import utils
from agent import names as N
from agent.nodes.base import BaseAgentTool


class SaferRainInputSchema(BaseModel):
    """
    Inputs for a tool that estimates flood extent and water depth from:
    - a rainfall raster (GeoTIFF) and
    - a DTM/DEM raster (GeoTIFF)
    with optional precomputed water-depth raster.

    Capabilities:
    - read DEM and rainfall rasters from local/HTTP(S)/S3/VSIs,
    - select input band and (optionally) convert to a target band index,
    - reproject output to a target SRS (EPSG),
    - run in 'batch' or 'lambda' execution mode,
    - produce flood extent and per-pixel water depth.
    """
    _URI_HINT = (
        "Path/URI to a GeoTIFF. Supported forms include local paths, HTTP(S), S3, and GDAL VSIs: "
        "e.g., '/data/dem.tif', 'https://host/file.tif', 's3://bucket/key.tif', '/vsis3/bucket/key.tif'."
    )


    # ----------------------------- Required inputs ---------------------------
    dem: str = Field(
        title="DEM (GeoTIFF)",
        description=f"Digital Elevation Model raster. {_URI_HINT}",
        examples=[
            "/data/dem.tif",
            "https://example.com/dem_10m.tif",
            "s3://my-bucket/dems/siteA_dem.tif",
            "/vsis3/my-bucket/dems/dem.tif",
        ],
        validation_alias=AliasChoices("dem", "dtm", "elevation", "dem_path"),
    )

    rain: str | float = Field(
        title="Rainfall data",
        description=f"Rainfall value. It can be a raster at {_URI_HINT}. It also can be a float value for constant rainfall.",
        examples=[
            "/data/rain_24h.tif",
            "https://example.com/rain/event_202505.tif",
            "s3://my-bucket/rain/rain_001.tif",
        ],
        validation_alias=AliasChoices("rain", "rainfall", "rain_path", "precip", "precipitation"),
    )

    # ----------------------------- Optional inputs ---------------------------
    water: Optional[str] = Field(
        default=None,
        title="Water depth (GeoTIFF, optional)",
        description=f"Optional precomputed water-depth raster. {_URI_HINT}",
        examples=[
            "/data/water_depth.tif",
            "https://example.com/water_depth.tif",
            "s3://my-bucket/floods/wd.tif",
        ],
        validation_alias=AliasChoices("water", "waterdepth", "wd", "water_path"),
    )

    # ------------------------------- Parameters ------------------------------
    band: int = Field(
        default=1,
        title="Rain band index",
        description="Band number to use from the rainfall raster (1-based).",
        examples=[1],
        validation_alias=AliasChoices("band", "rain_band", "input_band"),
    )

    to_band: int = Field(
        default=1,
        title="Target band index",
        description="Band number to convert the rain data to (1-based).",
        examples=[1],
        validation_alias=AliasChoices("to_band", "target_band", "out_band"),
    )

    t_srs: Optional[str] = Field(
        default=None,
        title="Target SRS (EPSG)",
        description='Target spatial reference for outputs, e.g., "EPSG:4326" or "EPSG:3857".',
        examples=["EPSG:4326", "EPSG:3857"],
        validation_alias=AliasChoices("t_srs", "target_srs", "crs", "out_crs"),
    )

    mode: Literal["lambda", "batch"] = Field(
        default="batch",
        title="Execution mode",
        description='Execution mode: "lambda" for AWS Lambda, "batch" for AWS Batch.',
        examples=["batch", "lambda"],
        validation_alias=AliasChoices("mode", "execution_mode", "run_mode"),
    )

    class Config:
        # Impedisci campi sconosciuti; abilita uso dei nomi/alias indifferentemente
        extra = "forbid"
        populate_by_name = True

    
# DOC: This is a demo tool to retrieve weather data.
class SaferRainTool(BaseAgentTool):
        
    
    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.SAFER_RAIN_TOOL,
            description =  """This tool computes flood extent and water depth over an AOI using a rainfall raster
            and a DEM/DTM raster (both GeoTIFF). Optionally, it can ingest an existing water-depth
            raster. It supports local paths as well as HTTP(S), S3, and GDAL VSI prefixes.

            Capabilities:
            - Read DEM (elevation) and rainfall rasters (GeoTIFF) from local/HTTP(S)/S3/VSIs.
            - Select the rainfall input band ('band') and optionally convert it to a target band ('to_band').
            - Reproject outputs to a target SRS (EPSG) if 't_srs' is provided.
            - Execute in 'batch' (default) or 'lambda' mode for AWS environments.
            - Produce flood extent and per-pixel water depth layers suitable for mapping/analysis.

            Inputs:
            - dem (string, required): DEM GeoTIFF path/URI. Examples: '/data/dem.tif', 'https://…/dem.tif', 's3://bucket/dem.tif'.
            - rain (string | float, required): Rainfall value. It can be a GeoTIFF path/URI or a constant float value.
            - water (string, optional): Precomputed water-depth GeoTIFF path/URI (if available).
            - band (int, default=1): 1-based band index to read from the rainfall raster.
            - to_band (int, default=1): 1-based band index for target band conversion.
            - t_srs (string, optional): Target EPSG code for outputs, e.g., 'EPSG:4326'.
            - mode ('batch' | 'lambda', default='batch'): Execution mode.

            When to use:
            - The user asks to estimate flood extent or water depth from rainfall and topography.
            - The user provides (or references) a DEM and a rainfall GeoTIFF (optionally a water-depth raster).
            - The user mentions AWS execution contexts (Batch/Lambda), band selection, or EPSG reprojection.

            Behavior & assumptions:
            - 'band' and 'to_band' are 1-based indices (>= 1).
            - Input rasters must be GeoTIFFs; URIs can be local paths, HTTP(S), S3 ('s3://'), or GDAL VSI (e.g., '/vsis3/...').
            - If 't_srs' is given, outputs are reprojected to that EPSG SRS.
            - 'mode' controls runtime: 'batch' (default) for bulk processing, 'lambda' for serverless execution.

            Output:
            - Flood extent and water depth rasters aligned for downstream visualization and analysis.""",
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
            """Infer water file path based on the user ID."""
            water = kwargs.get("water", f"saferplaces.co/SaferPlaces-Agent/dev/user=={self.graph_state.get('user_id', 'test')}/safer-rain-water.tif")
            return water

        infer_rules = {
            "water": infer_water
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

            'id': 'saferplacesapi.SaferBuildingsProcessor'
        }

        # TODO: Check if the response is valid
        
        tool_response = {
            'saferrain_response': api_response,
            
            # TODO: Move in a method createMapActions()
            'map_actions': [
                # {
                #     'action': 'new_layer',
                #     'layer_data': {
                #         'name': 'digital twin dem',  # TODO: add a autoincrement code
                #         'type': 'raster',
                #         'src': api_response['water_depth_file'],
                #         'styles': [
                #             { 'name': 'waterdepth', 'type': 'scalar', 'colormap': 'blues' }
                #         ]
                #     }
                # }
                utils.map_action_new_layer(
                    layer_name = 'digital twin dem',
                    layer_src = api_response['water_depth_file'],
                    layer_styles = [
                        { 'name': 'waterdepth', 'type': 'scalar', 'colormap': 'blues' }
                    ]
                )
                # TODO: Add action for each file (see above)
            ]
        }
        
        print('\n', '-'*80, '\n')
        
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