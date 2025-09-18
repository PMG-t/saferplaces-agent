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

from agent import utils
from agent import names as N
from agent.nodes.base import BaseAgentTool



# opzionale: enum per evitare variabili non supportate
Variable = Literal[
    'temperature',
    'dewpoint_temperature',
    'u_wind_component',
    'v_wind_component',
    'total_cloud_cover',
    'temperature_g',
    'snow_depth_water_equivalent',
    'pressure_reduced_to_msl',
    'total_precipitation'
]

class BBox(BaseModel):
    """
    Bounding box in EPSG:4326 (WGS84).
    west  = min longitude
    south = min latitude
    east  = max longitude
    north = max latitude
    """
    west: float = Field(..., description="Minimum longitude (°), e.g., 10.0")
    south: float = Field(..., description="Minimum latitude (°), e.g., 44.0")
    east: float = Field(..., description="Maximum longitude (°), e.g., 12.0")
    north: float = Field(..., description="Maximum latitude (°), e.g., 46.0")


class ICON2IRetrieverSchema(BaseModel):
    """
    Retrieve forecast data from the ICON-2I model for a given area and time window.

    ✅ Preferiti dall'agente:
      - Usa `bbox` con chiavi nominative (west,south,east,north).
      - Usa `time_start` e `time_end` in ISO8601.

    🔁 Supportati come alternativa (alias/fallback):
      - `lat_range` + `long_range` come liste [min, max].
      - `time_range` come lista [start, end] in ISO8601.

    Limite: l'orizzonte di previsione non può superare **72 ore avanti** dal momento corrente.
    Coordinate: **EPSG:4326 (WGS84)**.
    """

    variable: Variable = Field(
        ...,
        title="Forecast Variable",
        description="Meteorological variable to retrieve from ICON-2I. If not specified, use 'total_precipitation'.",
        examples=["total_precipitation"],
    )

    # ✅ Preferito: bbox nominato
    bbox: Optional[BBox] = Field(
        default=None,
        title="Bounding Box",
        description="Geographic extent in EPSG:4326 as named keys: west,south,east,north.",
        examples=[{"west": 10.0, "south": 44.0, "east": 12.0, "north": 46.0}],
    )

    # 🔁 Fallback/compat: range lat/lon come liste [min,max]
    lat_range: Optional[List[float]] = Field(
        default=None,
        title="Latitude Range (fallback)",
        description="Latitude range as [lat_min, lat_max] in EPSG:4326. Prefer `bbox`.",
        examples=[[44.0, 46.0]],
    )
    long_range: Optional[List[float]] = Field(
        default=None,
        title="Longitude Range (fallback)",
        description="Longitude range as [lon_min, lon_max] in EPSG:4326. Prefer `bbox`.",
        examples=[[10.0, 12.0]],
    )

    # ✅ Preferiti: time_start / time_end
    time_start: Optional[str] = Field(
        default=None,
        title="Start Time (ISO8601)",
        description="Forecast start time in ISO8601, e.g., 2025-09-18T00:00:00Z.",
        examples=["2025-09-18T00:00:00Z"],
    )
    time_end: Optional[str] = Field(
        default=None,
        title="End Time (ISO8601)",
        description="Forecast end time in ISO8601 (≤ now+72h).",
        examples=["2025-09-19T00:00:00Z"],
    )

    # 🔁 Fallback/compat: lista [start, end]
    time_range: Optional[List[str]] = Field(
        default=None,
        title="Time Range (fallback)",
        description="Time range as [start, end] in ISO8601. Prefer `time_start` and `time_end`.",
        examples=[["2025-09-18T00:00:00Z", "2025-09-19T00:00:00Z"]],
    )

    out: Optional[str] = Field(
        default=None,
        title="Local Output Directory",
        description="Local folder where data will be saved. If omitted, data is returned in-memory.",
        examples=["/data/icon2i"],
    )

    bucket_source: Optional[str] = Field(
        default=None,
        title="Source S3 Bucket (Optional)",
        description="AWS S3 bucket to read from instead of querying ICON-2I. Format: s3://bucket/path",
        examples=["s3://my-source/icon2i"],
    )

    bucket_destination: Optional[str] = Field(
        default=None,
        title="Destination S3 Bucket (Optional)",
        description=(
            "AWS S3 bucket where to store results. Format: s3://bucket/path. "
            "If neither `out` nor `bucket_destination` are provided, output is returned as a FeatureCollection."
        ),
        examples=["s3://my-dest/icon2i/results"],
    )

    @model_validator(mode="after")
    def _normalize_and_validate(self):
        # --- bbox fallback from lat/long ranges ---
        if self.bbox is None and self.lat_range and self.long_range:
            if len(self.lat_range) == 2 and len(self.long_range) == 2:
                lat_min, lat_max = self.lat_range
                lon_min, lon_max = self.long_range
                self.bbox = BBox(west=lon_min, south=lat_min, east=lon_max, north=lat_max)

        # require at least some spatial constraint (optional: puoi renderlo obbligatorio)
        if self.bbox is None:
            raise ValueError("Provide `bbox` or both `lat_range` and `long_range`.")

        # --- time fallback from time_range ---
        if (self.time_start is None or self.time_end is None) and self.time_range:
            if len(self.time_range) == 2:
                self.time_start, self.time_end = self.time_range

        # validate time order and horizon
        if self.time_start and self.time_end:
            try:
                # support 'Z' by replacing with +00:00
                ts = self.time_start.replace("Z", "+00:00")
                te = self.time_end.replace("Z", "+00:00")
                dt_start = datetime.datetime.fromisoformat(ts).replace(tzinfo=None)
                dt_end = datetime.datetime.fromisoformat(te).replace(tzinfo=None)
            except Exception as e:
                raise ValueError(f"Invalid ISO8601 in time_start/time_end: {e}")

            if dt_end <= dt_start:
                raise ValueError("`time_end` must be greater than `time_start`.")

            now = datetime.datetime.now(datetime.timezone.utc)
            max_end = now.replace(microsecond=0)  # normalize
            # consentiamo end nel futuro ma ≤ 72h
            horizon = (dt_end - now).total_seconds() / 3600.0
            if horizon > 72:
                raise ValueError("`time_end` cannot exceed 72 hours ahead from now.")

        return self
    

class ICON2IRetrieverTool(BaseAgentTool):
    """
    Tool for retrieving forecast data from the ICON-2I weather model.

    This tool queries the **ICON-2I numerical weather prediction model** to retrieve 
    forecast data for a specific **geographic area** and **time window**.

    It supports:
      - Selecting a **forecast variable** such as precipitation, temperature, wind, etc.
      - Defining the area of interest using a **bounding box (bbox)** in EPSG:4326 (WGS84).
      - Specifying a **time window** with a start and end time in ISO8601 format.
      - Saving results **locally** or to an **AWS S3 bucket**.

    The forecast horizon cannot exceed **72 hours ahead** from the current time.

    Example use cases:
      - "Get the total precipitation forecast for northern Italy for the next 48 hours."
      - "Retrieve wind speed and direction for a specific area and save the results to S3."

    Output format:
      - If `bucket_destination` or `out` is provided → data is saved to that location.
      - If neither is provided → output is returned as a **GeoJSON FeatureCollection**.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the ICON2IRetrieverTool.

        Args:
            **kwargs: Additional keyword arguments forwarded to BaseAgentTool.
        """
        super().__init__(
            name=N.ICON2I_RETRIEVER_TOOL,
            description=(
                "Use this tool to **retrieve weather forecast data** from the ICON-2I model. "
                "It is designed for tasks involving meteorological variables such as "
                "precipitation, temperature, cloud cover, wind components, and more.\n\n"
                "Provide:\n"
                "- `variable`: the forecast variable to retrieve. If not specified, defaults to `total_precipitation`.\n"
                "- `bbox`: bounding box for the area of interest in EPSG:4326.\n"
                "- `time_start` and `time_end`: ISO8601 timestamps (≤ 72 hours ahead).\n"
                "- Optional `bucket_destination` or `out` to save the data.\n\n"
                "If no storage location is provided, the tool returns the forecast data "
                "directly as a GeoJSON FeatureCollection."
            ),
            args_schema=ICON2IRetrieverSchema,
            **kwargs
        )
        self.execution_confirmed = False
        self.output_confirmed = True


    # DOC: Validation rules ( i.e.: valid init and lead time ... ) 
    def _set_args_validation_rules(self) -> dict:
        # TODO: | Validate time range  according icon2i API docs
        return dict()
    

    # DOC: Inference rules ( i.e.: from location name to bbox ... )
    def _set_args_inference_rules(self) -> dict:                  
        infer_rules = dict()
        return infer_rules
    

    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file
    def _execute(
        self,
        /,
        **kwargs: Any,  # dict[str, Any] = None,
    ): 
        # DOC: Call the SaferBuildings API ...
        api_url = f"{os.getenv('SAFERCAST_API_ROOT', 'http://localhost:5002')}/processes/icon2i-retriever-process/execution"
        payload = { 
            "inputs": kwargs | {
                "token": os.getenv("SAFERCAST_API_TOKEN"),
                "user": os.getenv("SAFERCAST_API_USER"),
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
            'icon2i_retriever_response': api_response,
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