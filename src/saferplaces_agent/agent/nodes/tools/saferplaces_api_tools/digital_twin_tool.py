import os
import datetime
from dateutil import relativedelta
from enum import Enum
import requests

from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field, AliasChoices, field_validator, model_validator

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from agent import utils
from agent import names as N
from agent.nodes.base import BaseAgentTool


class DigitalTwinInputSchema(BaseModel):
    """
    Inputs for a tool that builds a geospatial Digital Twin for a given AOI with layers:
    - DTM (DEM) raster,
    - building footprints,
    - land-use (land cover),
    - sea mask (land/sea).

    Capabilities:
    - fetch layers from named datasets/providers,
    - accept AOI as EPSG:4326 bbox or a place name (country/continent/region/city),
    - resample DEM to a given pixel size (meters),
    - return aligned, analysis-ready layers.
    """

    # ----------------------------- Data sources ------------------------------
    dataset_dem: str = Field(
        title="DEM/DTM dataset",
        description=(
            "Identifier/name of the elevation dataset from which the DTM will be obtained "
            "(e.g., a catalog key or provider path)."
        ),
        examples=[
            "COPERNICUS/DEM/GLO-30",
            "USGS/3DEP/10m",
            "SRTM/V3",
        ],
        validation_alias=AliasChoices("dataset_dem", "dem", "dtm", "dem_dataset", "dtm_dataset"),
    )

    dataset_building: Optional[str] = Field(
        default="OSM/BUILDINGS",
        title="Buildings provider",
        description=(
            "Provider/dataset name to fetch building footprints. "
            "Defaults to 'OSM/BUILDINGS'."
        ),
        examples=[
            "OSM/BUILDINGS",
            "MS/BUILDINGS",
            "LOCAL/BUILDINGS",
        ],
        validation_alias=AliasChoices(
            "dataset_building",
            "building_dataset",
            "buildings",
            "buildings_provider",
            "provider_buildings",
        ),
    )

    dataset_land_use: Optional[str] = Field(
        default="ESA/WorldCover/v100",
        title="Land-use dataset",
        description=(
            "Name of the land-use/land-cover dataset. "
            "Defaults to 'ESA/WorldCover/v100'."
        ),
        examples=[
            "ESA/WorldCover/v100",
            "Copernicus/CLC",
            "NLCD/2019",
        ],
        validation_alias=AliasChoices(
            "dataset_land_use",
            "land_use",
            "landuse",
            "landcover",
            "land_cover",
        ),
    )

    # ------------------------------ Spatial scope ----------------------------
    bbox: Optional[Union[list[float], str]] = Field(
        default=None,
        title="Area of interest (bbox or place name)",
        description=(
            "AOI as a bounding box [min_lon, min_lat, max_lon, max_lat] in EPSG:4326, "
            "or as a place name (country/continent/region/city). "
            "If omitted, use None."
        ),
        examples=[
            [9.05, 45.42, 9.25, 45.55],
            "Italy",
            "Europe",
            "Lombardia",
        ],
        validation_alias=AliasChoices(
            "usr_bbox",
            "bbox",
            "aoi",
            "area",
            "extent",
            "bounds",
            "bounding_box",
        ),
    )

    # ------------------------------ Resolution -------------------------------
    pixelsize: float = Field(
        title="DEM pixel size (meters)",
        description="Target ground sampling distance (meters) for the DEM/DTM resampling.",
        examples=[30, 10, 5],
        validation_alias=AliasChoices("pixelsize", "pixel_size", "resolution", "res", "gsd"),
    )

    class Config:
        # Impedisci campi sconosciuti; abilita uso dei nomi/alias indifferentemente
        extra = "forbid"
        populate_by_name = True


class DigitalTwinTool(BaseAgentTool):

    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.DIGITAL_TWIN_TOOL,
            description =  """This tool generates a geospatial Digital Twin for a given Area of Interest (AOI).
            It assembles multiple spatial layers (DEM/DTM, buildings, land-use, and sea mask) into
            a coherent, analysis-ready dataset.

            Capabilities:
            - Fetch a DTM (DEM) from the specified dataset and resample it at the requested pixel size (meters).
            - Retrieve building footprints from a chosen provider (default: 'OSM/BUILDINGS').
            - Add land-use/land-cover information (default: 'ESA/WorldCover/v100').
            - Produce a land/sea mask for the AOI.
            - Align and return all layers consistently over the AOI.

            Inputs:
            - `dataset_dem` (string, required): DEM/DTM dataset identifier.
            - `dataset_building` (string, optional, default 'OSM/BUILDINGS'): Provider for buildings.
            - `dataset_land_use` (string, optional, default 'ESA/WorldCover/v100'): Land-use/land-cover dataset.
            - `usr_bbox` (list of [min_lon, min_lat, max_lon, max_lat] in EPSG:4326 OR string with place name): Defines the AOI. Can be a bounding box or a place name (country, region, continent, city).
            - `pixelsize` (float, required): DEM resolution in meters. Must be > 0.

            When to use:
            - When the user requests a “Digital Twin” of an area.
            - When they need DEM/DTM data with specific resolution (pixel size).
            - When they ask for buildings, land-use, or sea mask layers for a defined AOI.
            - When AOI is given as coordinates (bbox) or as a place name.

            Behavior:
            - If `usr_bbox` is a bbox, it must be [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.
            - If `usr_bbox` is a string, interpret it as a geographic name (e.g., 'Italy', 'Lombardia').
            - Default providers are used if not specified.
            - Output is a set of harmonized geospatial layers ready for mapping and analysis.
            """,
            args_schema = DigitalTwinInputSchema,
            **kwargs
        )
        self.execution_confirmed = True
        self.output_confirmed = True

    
    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        
        # return {
        #     'area': [
        #         lambda **ka: f"Invalid area coordinates: {ka['area']}. It should be a list of 4 float values representing the bounding box [min_x, min_y, max_x, max_y]." 
        #             if isinstance(ka['area'], list) and len(ka['area']) != 4 else None  
        #     ]
        # }
        return dict()
        
    
    # DOC: Inference rules ( i.e.: from location name to bbox ... )
    def _set_args_inference_rules(self) -> dict:
        return dict()
        
    
    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file
    def _execute(
        self,
        /,
        **kwargs: Any,  # dict[str, Any] = None,
    ): 
        # DOC: Call the SaferBuildings API ...
        # api_root_local = "http://localhost:5000" # TEST: only when running locally
        # api_url = f"{os.getenv('SAFERPLACES_API_ROOT')}/processes/digital-twin-process/execution"
        # payload = { 
        #     "inputs": kwargs  | {
        #         "token": os.getenv("SAFERPLACES_API_TOKEN"),
        #         "user": os.getenv("SAFERPLACES_API_USER"),
        #     } | {
        #         "workspace": f"saferplaces.co/SaferPlaces-Agent/dev')/user=={self.graph_state.get('user_id', 'test')}",
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
            'files': {
                'file_building': f"{os.getenv('BUCKET_NAME', 's3://saferplaces.co/SaferPlaces-Agent/dev')}/user=={self.graph_state.get('user_id', 'test')}/Rimini_coast_flooded_deb-2407.geojson"
            }, 
            'id': 'saferplacesapi.SaferBuildingsProcessor',
            'message': {
                'body': {
                    'result': { 
                        's3_uri': f"{os.getenv('BUCKET_NAME', 's3://saferplaces.co/SaferPlaces-Agent/dev')}/user=={self.graph_state.get('user_id', 'test')}/saferbuildings-out/Rimini_coast_flooded_deb-2407.geojson"
                    }
                }
            }
        }
        
        # TODO: Check if the response is valid
        
        tool_response = {
            'saferbuildings_response': api_response['message']['body']['result'],
            
            # TODO: Move in a method createMapActions()
            'map_actions': [
                {
                    'action': 'new_layer',
                    'layer_data': {
                        'name': 'saferbuildings-water',  # TODO: add a autoincrement code
                        'type': 'raster',
                        'src': kwargs['water']
                        # TODO: style for leafmap (?)
                    }
                },
                {
                    'action': 'new_layer',
                    'layer_data': {
                        'name': 'saferbuildings-out',   # TODO: add a autoincrement code (same as the water layer)
                        'type': 'vector',
                        'src': api_response['files']['file_building'],
                        # TODO: style for leafmap (?) 
                        'styles': [
                            { 'name': 'flooded buildings', 'type': 'categoric', 'property': 'is_flooded', 'colormap': { 'true': '#FF0000', 'false': '#00FF00'} },
                            # { 'name': 'flood depth', 'type': 'numeric', 'property': 'wd_median', 'colormap': { 0: '#FFFFFF', 0.2: "#00D9FF", 0.4: "#4D7BE5", 0.6:"#9934E7", 0.8:"#FF0073" } }
                        ]
                    }
                }
            ]
        }
        
        print('\n', '-'*80, '\n')
        
        return tool_response
        
    
    # DOC: Back to a consisent state
    def _on_tool_end(self):
        self.execution_confirmed = True
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
