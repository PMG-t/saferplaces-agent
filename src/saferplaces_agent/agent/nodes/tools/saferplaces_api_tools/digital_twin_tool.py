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
from agent.common import states as GraphStates
from agent.nodes.base import base_models, BaseAgentTool


class DigitalTwinInputSchema(BaseModel):
    """
    Create a geospatial **Digital Twin** for a given Area of Interest (AOI) by assembling:
    - a DEM/DTM raster from the specified elevation dataset,
    - building footprints from the chosen dataset/provider,
    - land-use/land-cover from the chosen dataset.

    The DEM is resampled to the requested `pixelsize` (meters) and all outputs are aligned over the AOI.
    """

    # ----------------------------- Data sources ------------------------------
    dataset_dem: Optional[str] = Field(
        default=None,
        title="DEM/DTM dataset",
        description=(
            "Identifier of the elevation dataset to derive the DTM/DEM (catalog key or provider path). "
            "You may set this explicitly (e.g., 'USGS/3DEP/1M') **or leave it as `None` to let the tool "
            "auto-select the most suitable dataset from the AOI (bbox/place) using region-aware rules**.\n\n"
            "Region-aware hints (preferred sources by AOI):\n"
            "- **Italy** → GECOSISTEMA/ITALY\n"
            "- **Netherlands** → AHN/NETHERLANDS/05M | AHN/NETHERLANDS/5M\n"
            "- **Belgium** → NGI/BELGIUM/5M;  Flanders → VLAANDEREN/FLANDERS/BE/1M;  Wallonia → GEOPORTAIL/WALLONIE/BE/1M\n"
            "- **France** → IGN/RGE_ALTI/1M\n"
            "- **Spain** → IGN/ES/2M\n"
            "- **UK** → UK/LIDAR\n"
            "- **Denmark** → DK-DEM\n"
            "- **Norway** → NO/KARTVERKET\n"
            "- **Switzerland** → SWISSALTI3D/SWISS\n"
            "- **Australia** → AU/GA/AUSTRALIA_5M_DEM | AU/GEOSCIENCE | ELVIS/AUSTRALIA | ICSM.GOV/AUSTRALIA\n"
            "- **New Zealand** → NZ/LINZ\n"
            "- **Canada** → NRCAN/CANADA/2M | NRCAN/CDEM\n"
            "- **USA** → USGS/3DEP/1M | USGS/3DEP/10m | US/NED3 | US/NED10\n"
            "- **Mexico** → MX/LIDAR\n"
            "- **Angola** → ANGOLA/HUAMBO | ANGOLA/KUITO | ANGOLA/LOBITO | AIRBUS/ANGOLA\n"
            "- **Pan-EU / Europe** → COPERNICUS/EUDEM\n"
            "- **Global fallback** → NASA/NASADEM_HGT/001 | NASA/SRTM; **coastal** → DeltaDTM\n\n"
            "Selection rules:\n"
            "1) Prefer the **highest native resolution** covering the AOI; "
            "2) for **coastal AOI**, consider **DeltaDTM**; "
            "3) if no national source fits, use **COPERNICUS/EUDEM** (Europe) or global fallback."
        ),
        examples=["COPERNICUS/EUDEM", "USGS/3DEP/1M", None],
        validation_alias=AliasChoices("dataset_dem", "dem", "dtm", "dem_dataset", "dtm_dataset"),
    )

    dataset_building: str = Field(
        default="OSM/BUILDINGS",
        title="Buildings dataset",
        description="Provider/dataset to fetch building footprints. Default: 'OSM/BUILDINGS'.",
        examples=["OSM/BUILDINGS"],
        validation_alias=AliasChoices("dataset_building", "building_dataset", "buildings_provider"),
    )
    
    
    dataset_land_use: str = Field(
        default="ESA/WorldCover/v100",
        title="Land-use dataset",
        description="Dataset for land-use/land-cover. Default: 'ESA/WorldCover/v100'.",
        examples=["ESA/WorldCover/v100"],
        validation_alias=AliasChoices("dataset_land_use", "land_use", "landcover", "land_cover", "landuse"),
    )

    # ------------------------------ Spatial scope ----------------------------
    bbox: base_models.BBox = Field(
        ...,
        title="Area of Interest (bbox)",
        description=(
            "Geographic extent in EPSG:4326 using named keys west,south,east,north. "
            "It is used to define the Area of Interest (AOI) for the Digital Twin. "
        ),
        examples=[
            {"west": 9.05, "south": 45.42, "east": 9.25, "north": 45.55},
        ],
        validation_alias=AliasChoices("bbox", "aoi", "extent", "bounds", "bounding_box"),
    )

    # ------------------------------ Resolution -------------------------------
    pixelsize: Optional[float] = Field(
        default = None,
        title="DEM pixel size (meters). If not explicitly provided, prefer default value `None` as output resolution will be the native resolution of the DEM dataset.",
        description="Target ground sampling distance (meters) for the DEM/DTM resampling. Must be > 0.",
        examples=[None, 1, 2, 5, 10, 30],
        validation_alias=AliasChoices("pixelsize", "pixel_size", "resolution", "res", "gsd"),
    )


class DigitalTwinTool(BaseAgentTool):

    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.DIGITAL_TWIN_TOOL,
            description = (
                "Generate a **geospatial Digital Twin** for a given Area of Interest (AOI). "
                
                "### Purpose\n"
                "This tool is typically the **first step** in a workflow. It provides harmonized base layers "
                "that can later be used by other tools, such as flood simulation, building analysis, or land-use planning.\n\n"

                "### What it creates\n"
                "- **DEM/DTM raster**, resampled to the requested pixel size (`pixelsize`).\n"
                "- **Building footprints** for the AOI from the selected provider (`dataset_building`, default: 'OSM/BUILDINGS').\n"
                "- **Land-use/land-cover** layer for the AOI (`dataset_land_use`, default: 'ESA/WorldCover/v100').\n"
                "- **Sea mask** that separates land and water areas within the AOI.\n"
                "- All outputs are spatially aligned and clipped to the AOI.\n\n"
                
                "### Capabilities\n"
                "- Fetch a DEM/DTM from the specified dataset (if given, otherwise from the auto-detected most suitable dataset).\n"
                "- Retrieve **building footprints** from the given provider or use the default OSM-based source.\n"
                "- Retrieve **land-use/land-cover** information for better classification of terrain and regions.\n"
                "- Generate a **land/sea mask** covering the AOI.\n"
                "- Produce a set of harmonized layers ready for mapping, simulation, or other geospatial analyses.\n\n"

                "### Inputs\n"
                "- `dataset_dem (optional): Identifier of the DEM/DTM dataset **or `None`**. If `None`, the tool **auto-selects** the best dataset. \n"
                "- `dataset_building` (optional, default 'OSM/BUILDINGS'): Provider for building footprints.\n"
                "- `dataset_land_use` (optional, default 'ESA/WorldCover/v100'): Dataset for land-use/land-cover information.\n"
                "- `bbox` (required): AOI as EPSG:4326 bounding box. Use named keys `west,south,east,north`. If user provides a location name, you have to infer the bounding box.\n"
                "- `pixelsize` (optional): Desired DEM resolution in meters (> 0). Prefer None if user does not specify it, so the tool uses the native resolution of the DEM dataset.\n\n"

                "### When to use this tool\n"
                "- When the user explicitly asks for a **Digital Twin** of an area.\n"
                "- When harmonized layers of DEM, buildings, and land-use are needed for further analysis or simulations.\n"
                "- When a sea/land boundary mask is required for coastal or flood-related studies.\n"
                "- When the AOI is provided as geographic coordinates (bbox).\n\n"

                "### Behavior and defaults\n"
                "- The bounding box must be in EPSG:4326 coordinates.\n"
                "- If `dataset_dem` is **not provided** (None), the tool maps the AOI to country/region and selects a suitable DEM.\n"
                "- If `dataset_building` or `dataset_land_use` are not provided, the defaults are used.\n"
                "- Output is a set of raster and vector layers aligned on the same grid, ready for downstream tools and analyses.\n\n"

                "### Output\n"
                "The tool returns paths or URIs for each generated layer: DEM, buildings, land-use, and sea mask. "
                "These outputs form the core components of the Digital Twin for the specified AOI."
            ),
            args_schema = DigitalTwinInputSchema,
            **kwargs
        )
        self.execution_confirmed = False
        self.output_confirmed = True

    
    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        return dict()
        
    
    # DOC: Inference rules ( i.e.: from location name to bbox ... )
    def _set_args_inference_rules(self) -> dict:
        # def infer_bbox(**kwargs):
        #     print(f'\n\ninfer_bbox called with kwargs: {kwargs} \n\n')
        #     def bounding_box_from_location_name(bbox):
        #         if type(bbox) is str:
        #             bbox = utils.ask_llm(
        #                 role = 'system',
        #                 message = f"""Please provide the bounding box coordinates for the area: {bbox} with format [min_x, min_y, max_x, max_y] in EPSG:4326 Coordinate Reference System. 
        #                 Provide only the coordinates list without any additional text or explanation.""",
        #                 eval_output = True
        #             )
        #             self.execution_confirmed = False
        #         return bbox
        #     def round_bounding_box(bbox):
        #         deg_r = 3 # round 3 decimals of degree ~ 111.32 meters
        #         if type(bbox) is list:
        #             bbox = [
        #                 utils.floor_decimals(bbox[0], deg_r),
        #                 utils.floor_decimals(bbox[1], deg_r),
        #                 utils.ceil_decimals(bbox[2], deg_r),
        #                 utils.ceil_decimals(bbox[3], deg_r)
        #             ]
        #         return bbox
        #     bbox = bounding_box_from_location_name(kwargs['bbox'])
        #     bbox = round_bounding_box(bbox)
        #     return bbox
        
        infer_rules = {
            # 'bbox': infer_bbox
        }
        
        return infer_rules
        
    
    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file
    def _execute(
        self,
        /,
        **kwargs: Any,  # dict[str, Any] = None,
    ): 
        # DOC: set params for api, we do dis by hand and not using arg_schema to have better control over the output
        exec_uuid = utils.b64uuid()
        workspace = f"saferplaces.co/SaferPlaces-Agent/dev/user=={self.graph_state.get('user_id', 'test')}/project=={self.graph_state.get('project_id', 'dev')}"
        project = self.graph_state.get('project_id', f'digital-twin-tool-{exec_uuid}')
        file_dem = f'digital_twin_dem-{exec_uuid}.tif'
        file_building = f'digital_twin_building-{exec_uuid}.shp'
        file_landuse = f'digital_twin_landuse-{exec_uuid}.tif'
        file_dem_building = f'digital_twin_dem_building-{exec_uuid}.tif'
        file_seamask = f'digital_twin_seamask-{exec_uuid}.tif'

        # DOC: Call the SaferBuildings API ...
        # api_root_local = "http://localhost:5000" # TEST: only when running locally
        # api_url = f"{os.getenv('SAFERPLACES_API_ROOT')}/processes/digital-twin-process/execution"
        # payload = { 
        #     "inputs": kwargs  | {
        #         "token": os.getenv("SAFERPLACES_API_TOKEN"),
        #         "user": os.getenv("SAFERPLACES_API_USER"),
        #     } | {
        #         "workspace": f"saferplaces.co/SaferPlaces-Agent/dev/user=={self.graph_state.get('user_id', 'test')}",
        #         "project": "saferplaces-agent"    # TODO: variable from state (setted from client session in graph update like user_id)
        #     } | {
        #         "debug": True,  # TEST: enable debug mode
        #     }
        # }
        # print(f"Executing {self.name} with args: {payload}")
        # response = requests.post(api_url, json=payload)
        # print(f"Response status code: {response.status_code} - {response.content}")
        # response = response.json() 
        # TODO: Check output_code ...
        # TODO: Check if the response is valid
        
        # TEST: Simulate a response for testing purposes
        api_response = {
            'files': {
                # DOC: Use this when running with true API
                # 'file_dem': f"{workspace}/api_data/{project}/{file_dem}",
                # 'file_building': f"{workspace}/api_data/{project}/{file_building}",
                # 'file_landuse': f"{workspace}/api_data/{project}/{file_landuse}",
                # 'file_dem_building': f"{workspace}/api_data/{project}/{file_dem_building}",
                # 'file_seamask': f"{workspace}/api_data/{project}/{file_seamask}"

                # TEST: This is a simulated response for testing purposes
                'file_dem': "s3://saferplaces.co/Directed/Rimini/dtm_cropped_32633.tif",

            }, 
            'id': 'saferplacesapi.DigitalTwinProcessor',
            'message': {
                'body': {
                    'result': { 
                        'demo': 'demo :('
                    }
                }
            }
        }
        
        tool_response = {
            'tool_response': api_response,
            
            'updates': {
                'layer_registry': self.graph_state.get('layer_registry', []) + [
                    {
                        'title': 'Digital Twin DEM',
                        'description': 'Digital Twin DEM generated by SaferPlaces API',
                        'type': 'raster',
                        'src': api_response['files']['file_dem'],
                        'metadata': dict(),
                    }, 
                ] if not GraphStates.src_layer_exists(self.graph_state, api_response['files']['file_dem']) else [] + [
                    {
                        'title': 'Digital Twin Buildings',
                        'description': 'Digital Twin Buildings generated by SaferPlaces API',
                        'type': 'vector',
                        'src': api_response['files']['file_building'],
                        'metadata': dict(),
                    }
                ] if not GraphStates.src_layer_exists(self.graph_state, api_response['files']['file_building']) else [] + [
                    {
                        'title': 'Digital Twin Land Use',
                        'description': 'Digital Twin Land Use generated by SaferPlaces API',
                        'type': 'raster',
                        'src': api_response['files']['file_landuse'],
                        'metadata': dict(),
                    }
                ] if not GraphStates.src_layer_exists(self.graph_state, api_response['files']['file_landuse']) else [] + [
                    {
                        'title': 'Digital Twin DEM + Buildings',
                        'description': 'Digital Twin DEM + Buildings generated by SaferPlaces API',
                        'type': 'raster',
                        'src': api_response['files']['file_dem_building'],
                        'metadata': dict(),
                    }
                ] if not GraphStates.src_layer_exists(self.graph_state, api_response['files']['file_dem_building']) else [] + [
                    {
                        'title': 'Digital Twin Sea Mask',
                        'description': 'Digital Twin Sea Mask generated by SaferPlaces API',
                        'type': 'raster',
                        'src': api_response['files']['file_seamask'],
                        'metadata': dict(),
                    }
                ] if not GraphStates.src_layer_exists(self.graph_state, api_response['files']['file_seamask']) else []
            }
        }
        
        # tool_response = {
        #     'digitaltwin_response': api_response,
        #     'map_actions': [
        #         {
        #             'action': 'new_layer',
        #             'layer_data': {
        #                 'name': 'digital twin dem',
        #                 'type': 'raster',
        #                 'src': api_response['files']['file_dem'],
        #                 'styles': [
        #                     { 'name': 'dtm', 'type': 'scalar', 'colormap': 'viridis' }
        #                 ]
        #             }
        #         }
        #     ]
        # }
        
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
