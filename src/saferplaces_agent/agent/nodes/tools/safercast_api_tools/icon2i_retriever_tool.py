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



class ICON2IRetrieverSchema(BaseModel):
    """
    Schema for retrieving ICON-2I data.

    This schema defines the parameters required to retrieve specifc subsets data from the ICON-2I API.
    It includes fields for specifying the variable, latitude and longitude ranges, time range,
    output directory, and cloud bucket source and destination.
    """

    variable: str = Field(
        ...,
        title="Variable",
        description='The variable to retrieve. Allowed values are "precipitation", "temperature", "dewpoint_temperature", "pressure_reduced_to_msl", "snow_depth_water_equivalent", "total_cloud_cover", "u_wind_component", "v_wind_component".',
        example="total_precipitation",
        validation_alias="variable"
    )

    lat_range: Optional[List[float]] = Field(
        default=None,
        title="Latitude Range",
        description=(
            "The latitude range in the format [lat_min, lat_max]. "
            "Values must be expressed in EPSG:4326 CRS. "
            "If not provided, all latitude values will be returned."
        ),
        example=[40.0, 45.0],
        validation_alias="lat_range"
    )

    long_range: Optional[List[float]] = Field(
        default=None,
        title="Longitude Range",
        description=(
            "The longitude range in the format [long_min, long_max]. "
            "Values must be expressed in EPSG:4326 CRS. "
            "If not provided, all longitude values will be returned."
        ),
        example=[10.0, 12.0],
        validation_alias="long_range"
    )

    time_range: Optional[List[str]] = Field(
        default=None,
        title="Time Range",
        description=(
            "The time range in the format [time_start, time_end]. "
            "Both must be valid ISO 8601 date strings and refer to at least one week ago. "
            "If not provided, all available time values will be returned."
        ),
        example=["2024-09-01T00:00:00", "2024-09-10T00:00:00"],
        validation_alias="time_range"
    )

    out: Optional[str] = Field(
        default=None,
        title="Output Directory",
        description=(
            "The local directory where the retrieved data will be stored. "
            "If not provided, the data will not be saved locally."
        ),
        example="/path/to/output",
        validation_alias="out"
    )

    bucket_source: Optional[str] = Field(
        default=None,
        title="Bucket Source",
        description=(
            "The cloud bucket where the data will be retrieved from. "
            "If not provided, the data will be retrieved directly from the ICON-2I API."
        ),
        example="s3://source-bucket/folder",
        validation_alias="bucket_source"
    )

    bucket_destination: Optional[str] = Field(
        default=None,
        title="Bucket Destination",
        description=(
            "The cloud bucket where the data will be stored. "
            "If not provided, the data will not be saved to a bucket. "
            "If **neither out nor bucket_destination** are provided, "
            "the output will be returned as a Feature Collection."
        ),
        example="s3://destination-bucket/folder",
        validation_alias="bucket_destination"
    )

class ICON2IRetrieverTool(BaseAgentTool):
    """
    Tool for retrieving ICON-2I data from the API.
    
    This tool retrieves data from the ICON-2I API based on the specified parameters.
    It supports retrieving specific variables, latitude and longitude ranges, time ranges,
    and storing the data in a local directory or cloud bucket.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the ICON2IRetrieverTool with the provided parameters.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the BaseAgentTool.
        """
        super().__init__(
            name=N.ICON2I_RETRIEVER_TOOL,
            description=(
                "Retrieves data from the ICON-2I API based on the specified parameters. "
                "Supports retrieving specific variables, latitude and longitude ranges, "
                "time ranges, and storing the data in a local directory or cloud bucket."
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
        
        def infer_time_range(**kwargs):
            time_range = kwargs.get('time_range', None)
            if time_range is None:
                # DOC: default whole next forecast
                now = datetime.datetime.now(tz=datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
                time_range = [
                    now.isoformat(),
                    None
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