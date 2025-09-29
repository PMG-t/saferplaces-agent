import os
import datetime
from dateutil import relativedelta
from enum import Enum
import requests

from typing import Optional, Union, List, Dict, Any, Literal
from pydantic import BaseModel, Field, AliasChoices, field_validator, model_validator

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from agent import utils, s3_utils
from agent import names as N
from agent.common import states as GraphStates
from agent.nodes.base import base_models, BaseAgentTool


class BBox(BaseModel):
    """
    Bounding box in EPSG:4326 (WGS84).
    west=min lon, south=min lat, east=max lon, north=max lat.
    Prefer this over list formats to avoid ordering mistakes.
    """
    west: float = Field(..., description="Minimum longitude (°), e.g., 9.12")
    south: float = Field(..., description="Minimum latitude (°), e.g., 45.43")
    east: float = Field(..., description="Maximum longitude (°), e.g., 9.24")
    north: float = Field(..., description="Maximum latitude (°), e.g., 45.50")


ProviderCode = Literal[
    "OVERTURE",
    "RER-REST/*",
    "VENEZIA-WFS/*",
    "VENEZIA-WFS-CRITICAL-SITES",
]

FloodMode = Literal["BUFFER", "IN-AREA", "ALL"]


class SaferBuildingsInputSchema(BaseModel):
    """
    Identify flooded buildings from a water depth raster, optionally fetch/buildings from a provider,
    compute per-building water statistics, and return summaries grouped by chosen attributes.
    """

    # --- Inputs (primary) ---
    water: str = Field(
        ...,
        title="Water Depth Raster",
        description=(
            "URL or S3 URI pointing to the water depth raster (GeoTIFF). "
            "Example: 'https://...' or 's3://...'. "
            "It can be a reference to an **existing project raster layer** "
            "from the Layer Registry (e.g., 'use layer buildings-*').\n"
            "- When a layer is referenced, the tool will internally use that layer's `src` value."
        ),
        examples=[
            "https://example.com/data/floods/rimini-wd.tif",
            "s3://bucket/project/rimini-wd.tif",
            "rimini-wd.tif"
        ],
    )

    buildings: Optional[str] = Field(
        default=None,
        title="Buildings Vector (mutually exclusive with `provider`)",
        description=(
            "URL or S3 URI pointing to the buildings dataset (e.g., GeoJSON, GPKG). "
            "It can be a reference to an **existing project raster layer** "
            "from the Layer Registry (e.g., 'use layer buildings-*').\n"
            "- When a layer is referenced, the tool will internally use that layer's `src` value."
            "Example: 'https://...' or 's3://...'. "
            "**Mutual exclusivity rule:**\n"
            "- If you provide `buildings`, **do NOT set `provider`**.\n"
            "- If you do NOT have a local file or existing layer, leave `buildings` empty and use `provider` instead.\n\n"
            "Typical usage:\n"
            "- Provide a buildings file directly when you already have the dataset to analyze.\n"
            "- Leave this blank and use `provider` to fetch building data automatically."
        ),
        examples=[
            "https://example.com/data/buildings/rimini.geojson",
            "s3://bucket/project/buildings.geojson",
            "Rimini ROI"
        ],
    )

    provider: Optional[ProviderCode] = Field(
        default=None,
        title="Buildings Provider (optional)",
        description=(
            "Provider to retrieve buildings when `buildings` is not supplied. "
            "Allowed: OVERTURE | RER-REST/* | VENEZIA-WFS/* | VENEZIA-WFS-CRITICAL-SITES."
            "OVERTURE covers the entire world, so it is the default provider. "
            "RER-REST/* is for Emilia-Romagna region in Italy. "
            "VENEZIA-WFS/* is for Venice, and VENEZIA-WFS-CRITICAL-SITES is for critical sites in Venice."
        ),
        examples=["OVERTURE", "RER-REST/*"],
    )
    provider: Optional[ProviderCode] = Field(
        default=None,
        title="Buildings Provider (mutually exclusive with `buildings`)",
        description=(
            "Name of the provider to fetch building geometries automatically when no `buildings` file is provided.\n\n"
            "**Mutual exclusivity rule:**\n"
            "- If you set `provider`, **do NOT provide a `buildings` file**.\n"
            "- If you already have a local file or a project layer for buildings, set `buildings` and leave `provider` empty.\n\n"
            "Allowed values:\n"
            "- `OVERTURE`, default provider for global coverage.\n"
            "- `RER-REST/*`, for Emilia-Romagna region in Italy.\n"
            "- `VENEZIA-WFS/*`, for Venice area.\n"
            "- `VENEZIA-WFS-CRITICAL-SITES`, for critical sites in Venice.\n\n"
        ),
        examples=["OVERTURE", "RER-REST/*", "VENEZIA-WFS/*"],
    )

    
    # DOC: Disabled: too complex for now, will be added later
    # filters: Optional[Union[str, Dict[str, Any]]] = Field(
    #     default=None,
    #     title="Provider Filters (optional)",
    #     description=(
    #         "Filters for provider features. Accepts a JSON string or object "
    #         '(e.g., {"municipality":"Rimini","use":"residential"}).'
    #     ),
    #     examples=['{"municipality":"Rimini","use":"residential"}'],
    # )

    # --- Spatial scope ---
    bbox: Optional[base_models.BBox] = Field(
        default=None,
        title="Bounding Box (preferred)",
        description=(
            "Geographic extent in EPSG:4326 using named keys west,south,east,north. "
            "If omitted, the water raster total bounds are used."
        ),
        examples=[{"west": 12.52, "south": 44.01, "east": 12.60, "north": 44.08}],
    )

    t_srs: Optional[str] = Field(
        default=None,
        title="Target CRS (EPSG)",
        description=(
            "Target spatial reference (e.g., 'EPSG:4326'). "
            "If None, CRS of buildings is used if provided, otherwise CRS of water raster."
        ),
        examples=["EPSG:4326"],
    )

    # --- Flood logic & thresholds ---
    wd_thresh: float = Field(
        default=0.5,
        title="Water Depth Threshold (m)",
        description="Buildings are considered flooded when water depth ≥ this threshold (meters).",
        examples=[0.5, 0.3, 1.0],
    )

    flood_mode: FloodMode = Field(
        default="BUFFER",
        title="Flood Search Mode",
        description=(
            "Where to search for flood relative to buildings. "
            "`BUFFER`: flood around building geometry. This is the default mode. "
            "`IN-AREA`: flood inside geometry; "
            "`ALL`: both approaches."
            "When having buildings from an Overture or RER-REST source, it is recommended to use `BUFFER` mode, for VENEZIA-WFS-* sources, `IN-AREA` is recommended.\n\n"
        ),
        examples=["BUFFER"],
    )

    # --- Output controls ---
    only_flood: bool = Field(
        default=False,
        title="Return Only Flooded Buildings",
        description="If True, exclude non-flooded buildings from the output. It is often important to keep it False to include all buildings in the output.",
        examples=[True],
    )

    stats: bool = Field(
        default=False,
        title="Compute Per-Building Stats",
        description=(
            "If True, compute water depth statistics for each flooded building. "
            "It is an expensive operation, so use only when needed. "
            "Unless it is requested by the user, it is recommended to keep it False. "
        ),
        examples=[True],
    )

    summary: bool = Field(
        default=False,
        title="Compute Aggregated Summary",
        description=(
            "If True, compute an aggregated summary of flooded buildings grouped by selected attributes (`summary_on`). "
            "If `summary_on` is not provided, the default grouping depends on the `provider`:\n"
            "- **OVERTURE** → uses `'subtype'`\n"
            "- **RER-REST/*` → uses `'service_class'`\n"
            "- **VENEZIA-WFS/*` or `VENEZIA-WFS-CRITICAL-SITES` → uses `'service_id'`"
        ),
        examples=[True],
    )

    summary_on: Optional[List[str]] = Field(
        default=None,
        title="Summary Grouping Columns",
        description=(
            "List of attribute columns to group the summary by. "
            "If omitted and `summary=true`, the grouping will be based on the building category/type and depends on the chosen `provider`:\n"
            "- **OVERTURE** → `'subtype'`\n"
            "- **RER-REST/*` → `'service_class'`\n"
            "- **VENEZIA-WFS/*` or `VENEZIA-WFS-CRITICAL-SITES` → `'service_id'`\n\n"
            "You may also provide a comma-separated string which will be automatically split into a list."
        ),
        examples=[["building_type", "class"], ["subtype"], "building_type,class"],
    )

    out: Optional[str] = Field(
        default=None,
        title="Output Path (optional)",
        description=(
            "Destination URL or S3 URI where the **output vector file** will be saved (e.g., GeoJSON, GPKG).\n\n"
            "**Output contents:**\n"
            "- Each feature represents a building from the input dataset or provider.\n"
            "- A boolean attribute `is_flooded` is added to indicate whether the building is flooded "
            "based on the water depth threshold (`wd_thresh`).\n"
            "- If `stats=true`, additional water depth statistics per flooded building are included:\n"
            "  - `wd_min`: Minimum water depth inside the building footprint.\n"
            "  - `wd_mean`: Mean water depth inside the building footprint.\n"
            "  - `wd_max`: Maximum water depth inside the building footprint.\n\n"
        ),
        examples=[
            "https://example.com/results/flooded_buildings.geojson",
            "s3://bucket/project/output/flooded_buildings.geojson"
        ],
    )

    # --- Diagnostics ---
    debug: bool = Field(
        default=False,
        title="Debug Mode",
        description="Enable detailed logging for troubleshooting.",
        examples=[True],
    )
        




# DOC: This is a demo tool to retrieve weather data.
class SaferBuildingsTool(BaseAgentTool):
        
    
    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.SAFERBUILDINGS_TOOL,
            description = (
                "Use this tool to **detect flooded buildings** from a water depth raster (GeoTIFF). "
                "It supports two modes for obtaining building geometries:\n"
                "1. **Direct file input (`buildings`)** – use this when you already have a buildings dataset.\n"
                "2. **Provider-based fetching (`provider`)** – use this when you don't have a local file and want to fetch building data automatically.\n\n"
                "**Important:** `buildings` and `provider` are **mutually exclusive**.\n"
                "- If you provide a `buildings` file or reference a project layer, **do NOT set `provider`**.\n"
                "- If you want to fetch building data from a provider, **leave `buildings` empty** and set `provider`.\n\n"
                "### What this tool produces:\n"
                "- A **vector output file** (`out`) containing **all buildings** in the analysis area.\n"
                "- Each building feature includes:\n"
                "  - `is_flooded`: Boolean flag indicating if the building is flooded above the threshold (`wd_thresh`).\n"
                "  - If `stats=true`, additional fields:\n"
                "    - `wd_min`: Minimum water depth inside the building footprint.\n"
                "    - `wd_mean`: Mean water depth inside the building footprint.\n"
                "    - `wd_max`: Maximum water depth inside the building footprint.\n"
                "- If `summary=true`, the tool also returns **aggregated statistics** grouped by selected attributes "
                "(`summary_on`). This summary is **not saved into the vector file** but is returned as metadata.\n\n"
                "### Typical use cases:\n"
                "- Identify which buildings are flooded above a given water depth threshold.\n"
                "- Generate a vector layer with flood status and per-building water statistics.\n"
                "- Produce summaries grouped by building type, class, or provider-specific attributes.\n\n"
                "### Key arguments:\n"
                "- `water` (required): URL or S3 URI of the water depth raster.\n"
                "- `buildings` (optional): URL or S3 URI of a buildings dataset. "
                "Mutually exclusive with `provider`.\n"
                "- `provider` (optional): Provider for building geometries. "
                "Mutually exclusive with `buildings`. Supported values: OVERTURE, RER-REST/*, VENEZIA-WFS/*, VENEZIA-WFS-CRITICAL-SITES.\n"
                "- `filters`: Optional JSON object for filtering provider data.\n"
                "- `bbox`: Limit analysis to a geographic extent (EPSG:4326). If omitted, the water raster bounds are used.\n"
                "- `wd_thresh`: Flood threshold in meters (default 0.5).\n"
                "- `flood_mode`: How to detect flood relative to buildings — "
                "`BUFFER` (around buildings), `IN-AREA` (inside buildings), or `ALL` (both).\n"
                "- `stats`: Compute water depth stats per flooded building (`wd_min`, `wd_mean`, `wd_max`).\n"
                "- `summary`: Compute aggregated statistics grouped by `summary_on`. "
                "If `summary_on` is not provided, defaults depend on provider "
                "(e.g., `subtype` for OVERTURE, `service_class` for RER-REST, `service_id` for VENEZIA-WFS).\n"
                "- `out`: Destination URL or S3 URI to save the output vector file.\n"
                "- `out_geojson`: If true, results are returned in-memory as a GeoJSON FeatureCollection instead of being saved.\n"
                "- `only_flood`: If true, only flooded buildings are included in the output.\n\n"
                "### Agent notes:\n"
                "- Check if the user is providing a file (`buildings`) or requesting data from a provider. Never set both.\n"
                "- Always include the `is_flooded` attribute in the output.\n"
                "- Prefer `bbox` with named keys to avoid coordinate order mistakes."
            ),
            args_schema = SaferBuildingsInputSchema,
            **kwargs
        )
        self.execution_confirmed = False
        self.output_confirmed = True
        
    
    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        # DOC: No specific validation rules for this tool
        return dict()
        
    
    # DOC: Inference rules ( i.e.: from location name to bbox ... )
    def _set_args_inference_rules(self) -> dict:

        def infer_out(**kwargs):
            """
            Infer the S3 bucket destination based on user ID and project ID.
            """
            out = kwargs.get('out', f"saferbuildings_out.geojson")
            return f"{s3_utils._BASE_BUCKET}/saferbuildings-out/{out}"
            
        infer_rules = {
            'out': infer_out
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
        # api_url = f"{os.getenv('SAFERPLACES_API_ROOT')}/processes/safer-buildings-process/execution"
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
        # TODO: Check output_code ...
        
        # TEST: Simulate a response for testing purposes
        api_response = {
            'files': {
                'file_building': f"{s3_utils._BASE_BUCKET}/saferbuildings-out/Rimini_coast_flooded_deb-2407.geojson"
            }, 
            'id': 'saferplacesapi.SaferBuildingsProcessor',
            'message': {
                'body': {
                    'result': { 
                        's3_uri': f"{s3_utils._BASE_BUCKET}/saferbuildings-out/Rimini_coast_flooded_deb-2407.geojson"
                    }
                }
            }
        }
        
        tool_response = {
            'tool_response': api_response['message']['body']['result'],
            'updates': {
                'layer_registry': self.graph_state.get('layer_registry', []) + [
                    {
                        'title': f"SaferBuildings Output",
                        'description': f"SaferBuildings output file with flooded buildings from this inputs: ({', '.join([f'{k}: {v}' for k,v in kwargs.items() if k!='out'])})",
                        'src': api_response['message']['body']['result']['s3_uri'],
                        'type': 'vector',
                        'metadata': dict()
                    }
                ]
                if not GraphStates.src_layer_exists(self.graph_state, api_response['message']['body']['result']['s3_uri'])
                else []
            }
        }
        
        # TODO: Check if the response is valid
        
        # tool_response = {
        #     'saferbuildings_response': api_response['message']['body']['result'],
            
        #     # TODO: Move in a method createMapActions()
        #     'map_actions': [
        #         {
        #             'action': 'new_layer',
        #             'layer_data': {
        #                 'name': 'saferbuildings-water',  # TODO: add a autoincrement code
        #                 'type': 'raster',
        #                 'src': kwargs['water']
        #                 # TODO: style for leafmap (?)
        #             }
        #         },
        #         {
        #             'action': 'new_layer',
        #             'layer_data': {
        #                 'name': 'saferbuildings-out',   # TODO: add a autoincrement code (same as the water layer)
        #                 'type': 'vector',
        #                 'src': api_response['files']['file_building'],
        #                 # TODO: style for leafmap (?) 
        #                 'styles': [
        #                     { 'name': 'flooded buildings', 'type': 'categoric', 'property': 'is_flooded', 'colormap': { 'true': '#FF0000', 'false': '#00FF00'} },
        #                     # { 'name': 'flood depth', 'type': 'numeric', 'property': 'wd_median', 'colormap': { 0: '#FFFFFF', 0.2: "#00D9FF", 0.4: "#4D7BE5", 0.6:"#9934E7", 0.8:"#FF0073" } }
        #                 ]
        #             }
        #         }
        #     ]
        # }
        
        # print('\n', '-'*80, '\n')
        # print('tool_response:', tool_response)
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