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


class DPCRetrieverSchema(BaseModel):
    
    lat_range: Optional[List[float]] = Field(
        default=None,
        title="Latitude Range",
        description=(
            "Latitude range in the format [lat_min, lat_max]. "
            "Values must be in EPSG:4326 CRS. If omitted, all latitudes are used."
        ),
        example=[40.0, 45.0],
        validation_alias="lat_range",
    )

    long_range: Optional[List[float]] = Field(
        default=None,
        title="Longitude Range",
        description=(
            "Longitude range in the format [long_min, long_max]. "
            "Values must be in EPSG:4326 CRS. If omitted, all longitudes are used."
        ),
        example=[10.0, 12.0],
        validation_alias="long_range",
    )

    time_range: Optional[List[str]] = Field(
        default=None,
        title="Time Range",
        description=(
            "Time range in the format [time_start, time_end]. "
            "Both must be valid ISO 8601 strings and refer to at least one week ago. "
            "If omitted, all available times are used."
        ),
        example=["2024-09-01T00:00:00", "2024-09-10T00:00:00"],
        validation_alias="time_range",
    )

    product: str = Field(
        ...,
        title="Product",
        description="The product code to retrieve. It can be one of the following: 'SRI', 'VMI'.",
        example="SRI",
        validation_alias="product",
    )

    # ???: This is not used by agent
    # out: Optional[str] = Field(
    #     default=None,
    #     title="Output File Path",
    #     description=(
    #         "Local file path for the retrieved data. "
    #         "If neither out nor bucket_destination are provided, "
    #         "the output will be returned as a Feature Collection."
    #     ),
    #     example="/tmp/icon2i/result.geojson",
    #     validation_alias="out",
    # )

    # ???: This is not used by agent
    # out_format: Optional[str] = Field(
    #     default=None,
    #     title="Return Format Type",
    #     description=(
    #         'Return format type. Allowed values: "geojson" or "dataframe". '
    #         'Default behavior prefers "geojson" if unspecified by the tool.'
    #     ),
    #     example="geojson",
    #     validation_alias="out_format",
    # )

    bucket_destination: Optional[str] = Field(
        default=None,
        title="Bucket Destination",
        description=(
            "Cloud bucket where the data will be stored. If not provided, "
            "data will not be saved to a bucket. If neither out nor "
            "bucket_destination are provided, the output will be returned as "
            "a Feature Collection."
        ),
        example="s3://destination-bucket/folder",
        validation_alias="bucket_destination",
    )

    debug: Optional[bool] = Field(
        default=None,
        title="Debug",
        description="Enable Debug mode. Can be true or false.",
        example=True,
        validation_alias="debug",
    )


class DPCRetrieverTool(BaseAgentTool):
    """
    Tool for retrieving data from the DPC API.
    
    This tool retrieves data based on specified latitude and longitude ranges, time ranges, and product codes.
    It supports storing the data in a local directory or cloud bucket.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the DPCRetrieverTool with the provided parameters.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the BaseAgentTool.
        """
        super().__init__(
            name=N.DPC_RETRIEVER_TOOL,
            description=(
                "Retrieves data from the DPC API based on specified parameters. "
                "Supports latitude/longitude ranges, time ranges, and product codes."
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