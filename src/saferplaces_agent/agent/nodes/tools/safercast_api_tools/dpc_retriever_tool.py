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


from typing import Optional, List, Literal
from pydantic import BaseModel, Field, model_validator
from datetime import datetime, timezone


# ---- Product enum (allowed values) ----
DPCProductCode = Literal[
    "VMI",          # Vertical Maximum Intensity (max reflectivity, dBZ) – ~5min
    "SRI",          # Surface Rainfall Intensity (mm/h from radar + rain gauges) – ~5min
    "SRT1",         # Cumulative precipitation over 1 hour – ~1h
    "SRT3",         # Cumulative precipitation over 3 hours – ~1h
    "SRT6",         # Cumulative precipitation over 6 hours – ~1h
    "SRT12",        # Cumulative precipitation over 12 hours – ~1h
    "SRT24",        # Cumulative precipitation over 24 hours – ~1h
    "IR108",        # Cloud cover derived from MSG IR 10.8 channel – ~5min
    "TEMP",         # Temperature map interpolated from ground stations – ~1h
    "LTG",          # Lightning strike map (LAMPINET network) – ~10min
    "AMV",          # Atmospheric Motion Vectors (wind in upper levels, 50x50 km grid) – ~20min
    "HRD",          # Heavy Rain Detection (multi-sensor severe rainfall index) – ~5min
    "RADAR_STATUS", # Radar site status (ON/OFF)
    "CAPPI1", "CAPPI2", "CAPPI3", "CAPPI4", "CAPPI5", "CAPPI6", "CAPPI7", "CAPPI8"  # CAPPI reflectivity at fixed altitude (1–8 km) – ~10min
]


# ---- Bounding Box class ----
class BBox(BaseModel):
    """
    Bounding box in EPSG:4326 (WGS84).
    - `west` = min longitude
    - `south` = min latitude
    - `east` = max longitude
    - `north` = max latitude
    """
    west: float = Field(..., description="Minimum longitude (degrees), e.g., 10.0")
    south: float = Field(..., description="Minimum latitude (degrees), e.g., 44.0")
    east: float = Field(..., description="Maximum longitude (degrees), e.g., 12.0")
    north: float = Field(..., description="Maximum latitude (degrees), e.g., 46.0")


# ---- Main schema ----
class DPCRetrieverSchema(BaseModel):
    """
    Retrieve meteorological products from the Italian Civil Protection Department (DPC).

    This tool downloads a wide variety of **real-time and historical meteorological datasets**
    from the DPC API for a **specific geographic area** and **time window**.

    **Preferred parameters for the LLM:**
    - `bbox` (west, south, east, north) in EPSG:4326 to define the geographic extent.
    - `time_start` and `time_end` in ISO8601 format to specify the time range.

    **Fallback parameters (for compatibility):**
    - `lat_range` + `long_range` as lists `[min, max]` for latitude and longitude.
    - `time_range` as a list `[start, end]` in ISO8601.

    The `product` field determines **which dataset** to retrieve, such as precipitation,
    reflectivity, lightning strikes, or cloud cover.
    """

    # ---- WHAT to retrieve ----
    product: DPCProductCode = Field(
        ...,
        title="DPC Product",
        description=(
            "The code of the DPC dataset to retrieve. Examples:\n"
            "- `SRI`: Surface Rainfall Intensity (mm/h)\n"
            "- `VMI`: Vertical Maximum Intensity (max reflectivity, dBZ)\n"
            "- `SRT1/3/6/12/24`: Cumulative precipitation over different time intervals\n"
            "- `IR108`: Cloud cover from MSG IR 10.8 satellite imagery\n"
            "- `TEMP`: Interpolated temperature map from ground stations\n"
            "- `LTG`: Lightning strike frequency map\n"
            "- `AMV`: Upper-level wind vectors (Atmospheric Motion Vectors)\n"
            "- `HRD`: Heavy Rain Detection (multi-sensor severe rainfall index)\n"
            "- `RADAR_STATUS`: Status of radar network sites (ON/OFF)\n"
            "- `CAPPI1..8`: Reflectivity at fixed altitudes from 1 to 8 km"
        ),
        examples=["SRI"],
    )

    # ---- WHERE (preferred) ----
    bbox: Optional[BBox] = Field(
        default=None,
        title="Bounding Box",
        description="Geographic extent in EPSG:4326 with named keys: west, south, east, north.",
        examples=[{"west": 10.0, "south": 44.0, "east": 12.0, "north": 46.0}],
    )

    # ---- WHERE (fallback) ----
    lat_range: Optional[List[float]] = Field(
        default=None,
        title="Latitude Range (fallback)",
        description="Latitude range as [lat_min, lat_max] in EPSG:4326. Prefer using `bbox`.",
        examples=[[44.0, 46.0]],
    )
    long_range: Optional[List[float]] = Field(
        default=None,
        title="Longitude Range (fallback)",
        description="Longitude range as [lon_min, lon_max] in EPSG:4326. Prefer using `bbox`.",
        examples=[[10.0, 12.0]],
    )

    # ---- WHEN (preferred) ----
    time_start: Optional[str] = Field(
        default=None,
        title="Start Time (ISO8601)",
        description="Start timestamp for the data query in ISO8601 format, e.g., 2025-09-18T00:00:00Z.",
        examples=["2025-09-18T00:00:00Z"],
    )
    time_end: Optional[str] = Field(
        default=None,
        title="End Time (ISO8601)",
        description="End timestamp for the data query in ISO8601 format. Must be greater than `time_start`.",
        examples=["2025-09-18T06:00:00Z"],
    )

    # ---- WHEN (fallback) ----
    time_range: Optional[List[str]] = Field(
        default=None,
        title="Time Range (fallback)",
        description="Time range as [start, end] in ISO8601 format. Prefer using `time_start` and `time_end`.",
        examples=[["2025-09-18T00:00:00Z", "2025-09-18T06:00:00Z"]],
    )

    # ---- OUTPUT ----
    out: Optional[str] = Field(
        default=None,
        title="Local Output Path",
        description="Local file path where the retrieved data will be saved.",
        examples=["/tmp/dpc/result.geojson"],
    )

    out_format: Optional[Literal["geojson", "dataframe"]] = Field(
        default=None,
        title="Return Format",
        description=(
            'Format of the returned data when not saving to a file. '
            'Allowed values: `"geojson"` or `"dataframe"`. Default is `"geojson"`.'
        ),
        examples=["geojson"],
    )

    bucket_destination: Optional[str] = Field(
        default=None,
        title="Destination S3 Bucket (Optional)",
        description=(
            "AWS S3 bucket where the data will be stored. Format: `s3://bucket/path`.\n\n"
            "If neither `out` nor `bucket_destination` are provided, the output will "
            "be returned in-memory in the format specified by `out_format` "
            "(default: GeoJSON FeatureCollection)."
        ),
        examples=["s3://my-dest/dpc/results"],
    )

    debug: Optional[bool] = Field(
        default=False,
        title="Debug Mode",
        description="Enable verbose logging and diagnostics for troubleshooting.",
        examples=[True],
    )

    # ---- Normalization & Validation ----
    @model_validator(mode="after")
    def _normalize_and_validate(self):
        # Build bbox from lat/long ranges if not explicitly provided
        if self.bbox is None and self.lat_range and self.long_range:
            if len(self.lat_range) == 2 and len(self.long_range) == 2:
                lat_min, lat_max = self.lat_range
                lon_min, lon_max = self.long_range
                self.bbox = BBox(west=lon_min, south=lat_min, east=lon_max, north=lat_max)

        if self.bbox is None:
            raise ValueError("You must provide either `bbox` or both `lat_range` and `long_range`.")

        # Build time_start/time_end from time_range if needed
        if (self.time_start is None or self.time_end is None) and self.time_range:
            if len(self.time_range) == 2:
                self.time_start, self.time_end = self.time_range

        # Validate timestamps
        if self.time_start and self.time_end:
            try:
                ts = self.time_start.replace("Z", "+00:00")
                te = self.time_end.replace("Z", "+00:00")
                dt_start = datetime.fromisoformat(ts)
                dt_end = datetime.fromisoformat(te)
            except Exception as e:
                raise ValueError(f"Invalid ISO8601 timestamp in time_start/time_end: {e}")

            if dt_end <= dt_start:
                raise ValueError("`time_end` must be greater than `time_start`.")

        return self



class DPCRetrieverTool(BaseAgentTool):
    """
    Tool for retrieving meteorological products from the Italian Civil Protection Department (DPC).

    This tool allows you to **download and query real-time and historical meteorological datasets** 
    provided by the DPC. It is designed for use cases such as severe weather monitoring, 
    precipitation tracking, and environmental analysis.

    Supported features:
      - Select a specific **DPC product** such as rainfall intensity (SRI), reflectivity (VMI), 
        cumulative precipitation (SRT1/3/6/12/24), cloud cover (IR108), temperature (TEMP),
        lightning strikes (LTG), upper-level wind vectors (AMV), or radar network status.
      - Define the **geographic area** using a bounding box (`bbox`) in EPSG:4326.
      - Specify the **time window** with `time_start` and `time_end` (ISO8601).
      - Save retrieved data **locally** or **upload directly to AWS S3**.
      - Return data in multiple formats (`geojson` or `dataframe`).

    Example use cases:
      - "Retrieve rainfall intensity for northern Italy in the last 6 hours."
      - "Get cloud cover and lightning strike data for a specific area and save to S3."
      - "Download cumulative precipitation over 24 hours for monitoring flood risks."

    Output behavior:
      - If `bucket_destination` or `out` is provided → data will be saved to that location.
      - If neither is provided → data is returned **in-memory** in the format specified by `out_format` (default: GeoJSON FeatureCollection).
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the DPCRetrieverTool.

        Args:
            **kwargs**: Additional keyword arguments forwarded to BaseAgentTool.
        """
        super().__init__(
            name=N.DPC_RETRIEVER_TOOL,
            description=(
                "Use this tool to **retrieve meteorological datasets** from the Italian Civil Protection Department (DPC).\n\n"
                "Typical usage:\n"
                "- Specify `product` (e.g., SRI for rainfall intensity, VMI for reflectivity, IR108 for cloud cover).\n"
                "- Define the target **area** with a `bbox` (west, south, east, north) in EPSG:4326.\n"
                "- Set the **time window** using `time_start` and `time_end` (ISO8601).\n"
                "- Optionally provide `bucket_destination` or `out` to save results, "
                "or return them directly in memory as GeoJSON or DataFrame.\n\n"
                "Ideal for scenarios involving precipitation analysis, storm tracking, "
                "radar network monitoring, and other real-time weather-related tasks."
            ),
            args_schema=DPCRetrieverSchema,
            **kwargs
        )
        self.execution_confirmed = False
        self.output_confirmed = True


    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        # TODO: | Validate time range according dpc API docs | check product validity
        return dict()
    

    # DOC: Inference rules ( i.e.: from location name to bbox ... )
    def _set_args_inference_rules(self) -> dict:
        
        def infer_time_range(**kwargs):
            time_range = kwargs.get('time_range', None)
            if time_range is None:
                # DOC: default previous hour to now
                now = datetime.datetime.now(tz=datetime.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
                time_range = [
                    (now - relativedelta.relativedelta(hours=1)).isoformat(),
                    now.isoformat(),
                ]
            return time_range
                  
        infer_rules = {
            'time_range': infer_time_range,
        }
        return infer_rules
    

    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file
    def _execute(
        self,
        /,
        **kwargs: Any,  # dict[str, Any] = None,
    ): 
        # DOC: Call the SaferBuildings API ...
        api_url = f"{os.getenv('SAFERCAST_API_ROOT', 'http://localhost:5002')}/processes/dpc-retriever-process/execution"
        payload = { 
            "inputs": kwargs | {
                "token": os.getenv("SAFERCAST_API_TOKEN"),
                "user": os.getenv("SAFERCAST_API_USER"),
            } | {
                "bucket_destination": f"{os.getenv('BUCKET_NAME', 's3://saferplaces.co/SaferPlaces-Agent/dev')}/user=={self.graph_state.get('user_id', 'test')}/dpc-out"
            } | {
                "debug": True,  # TEST: enable debug mode
            }
        }
        print(f"Executing {self.name} with args: {payload}")
        response = requests.post(api_url, json=payload)
        print(f"Response status code: {response.status_code} - {response.content}")
        response = response.json() 
        # TODO: Check output_code ...

        # TEST: Simulate a response for testing purposes
        # api_response = {}
        api_response = response

        # TODO: Check if the response is valid
        
        tool_response = {
            'dpc_retriever_response': api_response,
        }
        
        print('\n', '-'*80, '\n')
        print('tool_response:', tool_response)
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