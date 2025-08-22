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


class SaferBuildingsInputSchema(BaseModel):
    """
    Inputs for a tool that identifies flooded buildings from:
    - a provided water-depth raster (flood map),
    - buildings supplied by file OR fetched on-the-fly from a provider.

    Capabilities:
    - apply a water-depth threshold,
    - restrict analysis to a bounding box,
    - compute per-building stats and aggregated summaries,
    - output as GeoJSON or as JSON/paths on disk.
    """

    # ----------------------------- Data sources ------------------------------
    water: Optional[str] = Field(
        default=None,
        title="Water-depth raster",
        description=(
            "Path to the water-depth raster (e.g., GeoTIFF). "
            "Required to compute flooding."
        ),
        examples=[
            "/data/floods/water_depth.tif",
            "C:\\data\\wd_2024_05.tif",
            "/vsis3/my-bucket/flood/wd.tif",
        ],
        validation_alias=AliasChoices("water", "waterdepth", "wd", "water_filename"),
    )

    buildings: Optional[str] = Field(
        default=None,
        title="Buildings (vector)",
        description=(
            "Path to the buildings vector dataset (GeoPackage/GeoJSON/Shapefile). "
            "If omitted, a `provider` must be given to fetch buildings on-the-fly."
        ),
        examples=[
            "/data/buildings.gpkg",
            "/data/buildings.geojson",
            "C:\\data\\buildings.shp",
        ],
        validation_alias=AliasChoices("building", "buildings", "buildings_filename"),
    )

    provider: Optional[str] = Field(
        default=None,
        title="Buildings provider (on-the-fly fetch)",
        description=(
            "Provider used to fetch buildings when no file is supplied. "
            "Allowed values: `OVERTURE`, `RER-REST/*`, `VENEZIA-WFS/*`, `VENEZIA-WFS-CRITICAL-SITES`. "
            "Patterns with `/*` indicate specific endpoints/collections."
        ),
        examples=[
            "OVERTURE",
            "RER-REST/buildings",
            "VENEZIA-WFS/edifici",
            "VENEZIA-WFS-CRITICAL-SITES",
        ],
        validation_alias=AliasChoices("provider"),
    )

    # ----------------------------- Analysis params ---------------------------
    wd_thresh: float = Field(
        default=0.5,
        ge=0.0,
        title="Water-depth threshold (m)",
        description=(
            "Depth threshold in meters above which flooding is considered significant. "
            "Buildings with depth < threshold are not marked as flooded."
        ),
        examples=[0.0, 0.3, 0.5, 1.0],
        validation_alias=AliasChoices("wd_thresh", "thresh"),
    )

    flood_mode: str = Field(
        default="BUFFER",
        title="Flood detection mode",
        description=(
            "How to search for flood relative to the building geometry. "
            "Valid values: 'BUFFER' (look around buildings using a buffer), "
            "'IN-AREA' (look inside the building footprint), 'ALL' (both). "
            "Default: 'BUFFER'."
        ),
        examples=["BUFFER", "IN-AREA", "ALL"],
        validation_alias=AliasChoices("flood_mode"),
    )

    bbox: Optional[List[float]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        title="Bounding box",
        description=(
            "Bounding box as a list of 4 floats in the exact order "
            "[minx, miny, maxx, maxy] (coordinates in EPSG:4326). "
            "If None, the total bounds of the water-depth raster are used."
        ),
        examples=[
            [12.30, 44.45, 12.65, 44.65],
            [-5.5, 35.2, 5.58, 45.10],
        ],
        validation_alias=AliasChoices("bbox"),
    )

    t_srs: Optional[str] = Field(
        default=None,
        title="Target CRS (EPSG)",
        description=(
            "Output spatial reference (e.g., 'EPSG:4326'). "
            "If None, uses the CRS of buildings if provided, otherwise the raster CRS."
        ),
        examples=["EPSG:4326", "EPSG:3857"],
        validation_alias=AliasChoices("t_srs"),
    )

    filters: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        title="Provider filters (JSON)",
        description=(
            "Filters to apply to provider features as JSON (string or dict). "
            "Example: {\"city\":\"Venezia\",\"subtype\":[\"residential\",\"industrial\"]}."
        ),
        examples=[
            '{"city":"Bologna"}',
            {"city": "Venezia", "subtype": ["residential", "industrial"]},
        ],
        validation_alias=AliasChoices("filters"),
    )

    # ----------------------------- Output & reporting ------------------------
    out: Optional[str] = Field(
        default=None,
        title="Output path",
        description=(
            "Destination folder or file for results. "
            "If `out_geojson=True`, the output will be a GeoJSON FeatureCollection file."
        ),
        examples=[
            "/outputs/flooded_buildings.geojson",
            "/outputs/run_01/",
        ],
        validation_alias=AliasChoices("out"),
    )

    out_geojson: bool = Field(
        default=False,
        title="Output as GeoJSON",
        description=(
            "If True, output a GeoJSON FeatureCollection. "
            "If False, produce a JSON with references/metadata or files on disk."
        ),
        examples=[True, False],
        validation_alias=AliasChoices("out_geojson"),
    )

    only_flood: bool = Field(
        default=False,
        title="Return flooded buildings only",
        description="If True, return only buildings classified as flooded.",
        examples=[True, False],
        validation_alias=AliasChoices("only_flood"),
    )

    stats: bool = Field(
        default=False,
        title="Compute water-depth statistics",
        description=(
            "Compute per-building statistics on water depth (e.g., min/mean/max) "
            "and overall aggregates."
        ),
        examples=[True, False],
        validation_alias=AliasChoices("stats"),
    )

    summary: bool = Field(
        default=False,
        title="Add aggregated summary",
        description=(
            "Add aggregated statistics by building type/class. "
            "If `summary_on` is None, the summary will cover all flooded buildings. "
            "If not provided and `provider=OVERTURE`, `subtype` is typically used; "
            "if `provider` starts with 'RER-REST/', `service_class` is typically used."
        ),
        examples=[True, False],
        validation_alias=AliasChoices("summary"),
    )

    summary_on: Optional[Union[str, List[str]]] = Field(
        default=None,
        title="Columns to summarize on",
        description=(
            "Columns to compute aggregated stats on. "
            "Accepts a list (e.g., ['subtype','class']) or a comma-separated string "
            "without spaces (e.g., 'subtype,class')."
            "Prefer 'subtype' for Overture, 'service_class' for RER-REST, 'service_id' for VENEZIA-WFS."
        ),
        examples=[
            None,
            "subtype",
            "subtype,class",
            ["subtype", "class"],
            ["service_class"],
        ],
        validation_alias=AliasChoices("summary_on"),
    )

    class Config:
        # Impedisci campi sconosciuti; abilita uso dei nomi/alias indifferentemente
        extra = "forbid"
        populate_by_name = True
        




# DOC: This is a demo tool to retrieve weather data.
class SaferBuildingsTool(BaseAgentTool):
        
    
    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name = N.SAFERBUILDINGS_TOOL,
            description =  """This tool identifies buildings affected by flooding by intersecting a water-depth raster (flood map) with building footprints that are either provided as a local/vector file or fetched on-the-fly from a data provider.

            Inputs and behavior:
            • Required inputs: a water-depth raster path; plus either a buildings file path OR a provider (OVERTURE, RER-REST/*, VENEZIA-WFS/*, or VENEZIA-WFS-CRITICAL-SITES).  
            • Water-depth threshold: depths ≥ wd_thresh (meters, default 0.5) are considered flooded.  
            • Flood detection mode:  
            - BUFFER: search for flood around buildings (buffered vicinity),  
            - IN-AREA: search for flood inside the building footprint,  
            - ALL: apply both. Default is BUFFER.  
            • Spatial extent: optional bbox as a list of four floats in the exact order [minx, miny, maxx, maxy], coordinates in EPSG:4326. If omitted, the raster’s total bounds are used.  
            • Coordinate reference system: t_srs sets the output EPSG (e.g., "EPSG:4326"). If not provided, the buildings CRS is used when available, otherwise the raster CRS.  
            • Provider filters: optional JSON (string or dict) to pre-filter provider features.  
            • Output: set out to choose the destination path. If out_geojson=True, results are written as a GeoJSON FeatureCollection; otherwise a JSON/paths with references/metadata are produced.  
            • Post-processing: only_flood returns flooded buildings only. stats computes per-building water-depth statistics; summary adds aggregated statistics by building class/type. If summary is enabled and summary_on is not supplied, sensible defaults are used (e.g., "subtype" for OVERTURE, "service_class" for RER-REST/*).

            Assumptions:
            • Water depth values are in meters.  
            • Inputs may have different CRSs; outputs are produced in t_srs when provided.  
            • bbox must be exactly four numbers: [minx, miny, maxx, maxy] in EPSG:4326.""",
            args_schema = SaferBuildingsInputSchema,
            **kwargs
        )
        self.execution_confirmed = False
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
        print('\n'*2, '-'*80, '\n')

        api_root_local = "http://localhost:5000" # TEST: only when running locally
        api_url = f"{os.getenv('SAFERPLACES_API_ROOT')}/processes/safer-buildings-process/execution"
        payload = { 
            "inputs": kwargs  | {
                "token": os.getenv("SAFERPLACES_API_TOKEN"),
                "user": os.getenv("SAFERPLACES_API_USER"),
            } | {
                "wd_thresh": 1.0, # speed-up
                "out": f"s3://saferplaces.co/SaferPlaces-Agent/dev/saferbuildings_tool/out_{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}.gpkg",
                "summary_on": "subtype",  # TEST: use subtype for summary
            } | {
                "debug": True,  # TEST: enable debug mode
            }
        }
        print(f"Executing {self.name} with args: {payload}")
        response = requests.post(api_url, json=payload)
        print(f"Response status code: {response.status_code} - {response.content}")
        response = response.json() 
        # TODO: Check output_code ...
        
        print('\n', '-'*80, '\n')
        
        return response
        
    
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