import os
import base64
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


class GeospatialOpsInputSchema(BaseModel):
    """
    Inputs for the Geospatial Operation Tool.

    This tool interprets a user prompt that requests geospatial operations on layers.

    Capabilities include:
    - Responding with geographic knowledge or descriptive/statistical outputs 
      (e.g., bounding box of a city, centroid of a polygon, maximum value of a raster).
      These results usually do not require creating a new layer, unless the user explicitly requests it.
    - Performing spatial operations that generate new geospatial datasets 
      (e.g., intersections, unions, clipping, dissolve, difference).
      These operations typically produce a new output layer that can be added to the map.
    - If it is unclear whether a new layer should be produced, the tool may ask the user 
      for confirmation before proceeding.
    """

    prompt: str = Field(
        title="User Prompt",
        description=(
            "The user request describing the geospatial operation to perform. "
            "This can include descriptive/statistical queries (e.g., bounding box of a city, "
            "maximum raster value) or operations that generate new datasets "
            "(e.g., intersection, union, clip, dissolve). "
            "The tool will decide whether the request should return a direct value/geometry "
            "or produce a new output layer."
        ),
        examples=[
            "Give me the bounding box of Rome",
            "Find all buildings within the bounding box of Rome",
            "What is the maximum elevation value in this raster?",
        ],
    )

    output_layer: str | None = Field(
        title="Output Layer Name",
        description=(
            "Optional. The name of the new layer to be created with the results of the operation. "
            "This should only be provided if the operation generates new geospatial data suitable "
            "for visualization (e.g., intersection, clip, union). "
            "For descriptive/statistical operations (e.g., bounding box, centroid, raster statistics), "
            "this should normally be None unless the user explicitly requests a layer."
        ),
        examples=[
            "intersected_buildings.geojson",
            "clipped_area.tif",
        ],
        default=None,
    )

    # class Config:
    #     # Impedisci campi sconosciuti; abilita uso dei nomi/alias indifferentemente
    #     extra = "forbid"
    #     populate_by_name = True


# DOC: This is a demo tool to retrieve weather data.
class GeospatialOpsTool(BaseAgentTool):

    # DOC: Initialize the tool with a name, description and args_schema
    def __init__(self, **kwargs):
        super().__init__(
            name=N.GEOSPATIAL_OPS_TOOL,
            description="""The Geospatial Operation Tool interprets natural language prompts requesting geospatial operations.
            The tool can respond in two different ways:
            1. For descriptive or statistical queries (e.g., bounding boxes, centroids, raster statistics), 
            it returns direct values or simple geometries, without creating a new layer, 
            unless the user explicitly requests one.
            2. For productive or transformative operations (e.g., intersections, unions, clipping, differences, dissolves), 
            it generates a new layer suitable for visualization and storage.

            If the intent is ambiguous, the tool may first ask the user to confirm whether a new layer should be created.

            Possible outputs:
            - Direct values or simple geometries (numbers, text, GeoJSON snippets).
            - Renderable layers (GeoJSON, raster) when appropriate.
            """,
            args_schema=GeospatialOpsInputSchema,
            **kwargs
        )
        self.execution_confirmed = True
        self.output_confirmed = False

    # DOC: Validation rules ( i.e.: valid init and lead time ... )

    def _set_args_validation_rules(self) -> dict:
        return dict()

    # DOC: Inference rules ( i.e.: from location name to bbox ... )

    def _set_args_inference_rules(self) -> dict:

        def infer_output_layer(**kwargs):
            """Infer the output layer name based on the prompt."""
            output_layer = kwargs.get('output_layer', None)
            if output_layer is None:
                output_layer = utils.ask_llm(
                    role='system',
                    message=f"""You are an assistant specialized in geospatial operations. 
                    Decide whether the user's request requires creating a new output layer, and if so, propose a valid filename.

                    Decision rules:
                    1. **Do not create a layer** if the request is descriptive or statistical:
                    - Examples: bounding box, centroid, area, perimeter, maximum/minimum/mean value of a raster, summary statistics.
                    - In these cases, return "None".
                    2. **Create a layer** only if the operation produces new or modified geospatial data 
                    that could be visualized on a map:
                    - Examples: intersection, union, difference, clip, dissolve, spatial filtering that returns a dataset.
                    3. If the user explicitly asks to save the result as a layer, always propose a filename.

                    Naming rules (only when a layer is needed):
                    - Lowercase, no spaces, only letters, numbers, and underscores (`_`).
                    - File extension:
                    - `.geojson` for vector data (points, lines, polygons, bounding boxes, filtered sets).
                    - `.tif` for raster data (grids, clipped rasters, raster operations).
                    - Filename only, no path or explanation.

                    Output strictly one value:
                    - Either the filename (e.g. `buildings_within_rome.geojson`, `elevation_difference.tif`).
                    - Or `None` if no layer should be created.

                    User request:
                    "{kwargs['prompt']}"
                    """,
                    eval_output=True
                )
                if output_layer is None:
                    return None
                output_layer = output_layer.strip()
                output_layer = f"s3://saferplaces.co/SaferPlaces-Agent/dev/user=={self.graph_state.get('user_id', 'test')}/{output_layer}"
            # TODO: SHould come fron env variable
            elif not (kwargs['output_layer'].startswith('s3://saferplaces.co/') or kwargs['output_layer'].startswith('https://s3.us-east-1.amazonaws.com/saferplaces.co/')):
                output_layer = utils.justfname(kwargs['output_layer'])
                output_layer = f"s3://saferplaces.co/SaferPlaces-Agent/dev/user=={self.graph_state.get('user_id', 'test')}/{output_layer}"
            return output_layer

        infer_rules = {
            'output_layer': infer_output_layer,
        }
        return infer_rules

    # DOC: Execute the tool → Build notebook, write it to a file and return the path to the notebook and the zarr output file

    def _execute(
        self,
        /,
        **kwargs: Any,  # dict[str, Any] = None,
    ):

        def describe_layer_registry():
            """Describe the layers available in the registry."""
            layers = self.graph_state.get('layer_registry', [])
            if not layers:
                return "No layers available."
            layer_descriptions = []
            for layer in layers:
                layer_description = []
                layer_description.extend([
                    f"Layer: {layer['name']}",
                    f"- type: {layer['type']}",
                    f"- src: {layer['src']}"
                ])
                layer_descriptions.append('\n'.join(layer_description))
            return '\n\n'.join(layer_descriptions)

        additional_info = '\n'.join(
            [f'- {key}: {value}' for key, value in kwargs.items() if key not in ['prompt', 'output_layer']])
        additional_info = f"Additional and useful informations:\n{additional_info}" if additional_info else ""

        print('\n\n' + '-'*80 + '\n')
        print(f'layers: {self.graph_state.get("layer_registry", [])}')
        print(f'kwargs: {kwargs}')
        print(f'additional_info: {additional_info}\n')
        print('\n' + '-'*80 + '\n')

        output = utils.ask_llm(
            role='system',
            message=f"""You are a Python code generator specialized in geospatial operations. 

            Your task: Given a user request describing a geospatial data operation, output only valid Python code that produces the requested data. 

            Constraints:

            1. Always respond with Python code only. Do NOT include explanations, text, or commentary.  
            2. You may use only the following libraries: geopandas, shapely, pandas, fiona, rasterio, pyproj, numpy. No other libraries.  
            3. You can produce results based on:  
            a) Your internal geographic knowledge (e.g., bounding boxes of known cities, coordinates of countries).  
            b) Provided layers, given usually as S3 URL.
            4. If the user requires to produce a new layer, ensure the code stores the results in the layer provided in the `output_layer` argument.

            Instructions:

            - At the end of the code produce print statement tha show some information about the operation performed and its output.
            - The code should be self-contained and executable, assuming the required libraries are imported and any input layers are accessible.  
            - If the user request requires working with a layer, assume it is provided as a variable `layer` (loaded with geopandas.read_file or similar) or as a file path / S3 URL string.  
            - Use proper geospatial operations: intersections, within, filtering by bbox, or attribute-based selections.  
            - Produce only data, e.g., a Raster or Vector layer, a Pandas DataFrame, a GeoPandas GeoDataFrame, a shapely geometry, or a dictionary summarizing geometry stats.  
            - Do NOT print, log, or output anything else.  
            - Do not perform file system operations outside of reading allowed layers.  
            - Do not import os, sys, subprocess, or any unsafe modules.  

            User request:
            "{kwargs['prompt']}"
            Output layer:
            "{kwargs.get('output_layer', 'None')}"
            
            You have access to the following layers:
            {describe_layer_registry()}
            
            {additional_info}
            """,
            eval_output=False
        )

        tool_response = {
            # TODO: Geospatial ops layer info dict
            # TODO: Map actions to be executed by the frontend (new-layer)
            'generated_code': output
        }

        print('\n', '-'*80, '\n')

        return tool_response

    # DOC: Back to a consisent state

    def _on_tool_end(self):
        self.execution_confirmed = True
        self.output_confirmed = False

    # DOC: Try running AgentTool → Will check required, validity and inference over arguments thatn call and return _execute()

    def _run(
        self,
        /,
        **kwargs: Any,  # dict[str, Any] = None,
    ) -> dict:

        run_manager: Optional[CallbackManagerForToolRun] = kwargs.pop(
            "run_manager", None)
        return super()._run(
            tool_args=kwargs,
            run_manager=run_manager
        )
