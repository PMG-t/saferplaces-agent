import os
import json
from textwrap import indent
import datetime

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from .graph import graph
from .common import utils

class GraphInterface:

    def __init__(
        self, 
        thread_id: str,
        user_id: str,
        project_id: str = None,
    ):
        self.G: CompiledStateGraph = graph
        self.thread_id = thread_id
        self.user_id = user_id
        self.project_id = project_id

        self.interrupted = False

        self.config = { "configurable": { "thread_id": self.thread_id } }


    def _event_value_is_interrupt(self, event_value):
        return type(event_value) is tuple and type(event_value[0]) is Interrupt

    def _interrupt2dict(self, event_value):
        interrupt_data = event_value[0].value
        agent_interrupt_message = { 'interrupt': interrupt_data }
        return agent_interrupt_message


    def user_prompt(
        self,
        prompt: str,
        state_updates: dict = dict(),
    ):
        
        def prepare_system_messages():

            def build_layer_registry_system_message(layer_registry):
                """
                Generate a system message dynamically from a list of layer dictionaries.
                
                Args:
                    layer_registry (list[dict]): List of layers where each layer has at least:
                        - title (str)
                        - type (str) -> "raster" or "vector"
                        - src (str)
                        - description (optional)
                        - metadata (optional dict)
                        
                Returns:
                    str: A formatted system message ready to be injected before the user prompt.
                """
                lines = []
                lines.append("[LAYER REGISTRY]")
                lines.append("The following geospatial layers are currently available in the project.")
                lines.append("Each layer has a `title` that should be referenced in conversations or tool calls "
                            "when you need to use it. "
                            "If the user refers to an existing dataset, check this registry to see if the dataset "
                            "already exists before creating new data.\n")
                lines.append("Layers:")
                for idx, layer in enumerate(layer_registry, start=1):
                    lines.append(f"{idx}.")
                    lines.append(f"  - title: \"{layer.get('title', utils.juststem(layer['src']))}\"")
                    lines.append(f"  - type: {layer['type']}")
                    if 'description' in layer and layer['description']:
                        lines.append(f"  - description: {layer['description']}")
                    lines.append(f"  - src: {layer['src']}")

                    # Metadata, if present
                    if 'metadata' in layer and layer['metadata']:
                        lines.append("  - metadata:")
                        # Pretty print nested metadata with indentation
                        meta_json = json.dumps(layer['metadata'], indent=4)
                        lines.append(indent(meta_json, prefix="      "))

                lines.append("\nInstructions:")
                lines.append("- When a user request can be satisfied by using one of these layers, prefer re-using "
                            "the layer instead of creating a new one.")
                lines.append("- Always refer to the `title` when mentioning or selecting a layer in your tool arguments.")
                lines.append("- If the type is 'vector', assume it contains geographic features like polygons, lines, or points.")
                lines.append("- If the type is 'raster', assume it contains gridded geospatial data.")
                lines.append("[/LAYER REGISTRY]")
                return {
                    'role': 'system',
                    'content': "\n".join(lines)
                }
            
            def build_nowtime_system_message():
                """
                Generate a system message with the current time in ISO8601 UTC0 format.
                
                Returns:
                    dict: A system message with the current time and timezone.
                """
                nowtime = datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None).isoformat()
                lines = []
                lines.append("[CONTEXT]")
                lines.append(f"current_time: {nowtime}")
                lines.append("timezone: UTC0")
                lines.append("\nInstructions:")
                lines.append("- Resolve any relative time expressions (e.g., today, yesterday, next N hours) using `current_time`.")
                lines.append("- If a year is missing, assume the year from `current_time`.")
                lines.append("- Always output absolute timestamps in ISO8601 UTC0 format without timezone.")
                lines.append("[/CONTEXT]")
                return {
                    'role': 'system',
                    'content': "\n".join(lines)
                }

            system_messages = []
            system_messages.append(build_nowtime_system_message())
            if 'layer_registry' in state_updates and state_updates['layer_registry']:
                system_messages.append(build_layer_registry_system_message(state_updates['layer_registry']))
            return system_messages
        
        def build_stream():
            stream_obj = dict()
            if self.interrupted:
                self.interrupted = False
                stream_obj = Command(resume={'response': prompt})
            else:
                stream_obj = {
                    'messages': [
                        * prepare_system_messages(),
                        {'role': 'user', 'content': prompt}
                    ],
                    'user_id': self.user_id,
                    'project_id': self.project_id,
                    'node_params': state_updates.get('node_params', dict()),
                    'node_history': state_updates.get('node_history', []),
                    'layer_registry': state_updates.get('layer_registry', []),
                    'avaliable_tools': state_updates.get('avaliable_tools', []),
                    'nowtime': datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None).isoformat(),
                }
            return stream_obj
        
        agent_messages = []
        for event in self.G.stream(
            input = build_stream(),
            config = self.config,
            stream_mode = 'updates'
        ):
            for value in event.values():
                
                if self._event_value_is_interrupt(value):
                    self.interrupted = True
                    agent_messages.append(self._interrupt2dict(value))
                
                else:
                    if 'messages' in value:
                        value['message'] = value['messages'][-1].to_json()
                        del value['messages']

                    if 'node_params' in value:
                        for node, params in value['node_params'].items():
                                if 'tool_message' in params:
                                    value['node_params'][node]['tool_message'] = params['tool_message'].to_json()

                    agent_messages.append(value)
            
        return agent_messages
    

class GraphRegistry:

    """Registry for the agent graph."""
    
    def __init__(self):
        self.graphs = dict()

    def register(self, thread_id: str, user_id: str) -> GraphInterface:
        self.graphs[thread_id] = GraphInterface(thread_id, user_id)
        return self.graphs[thread_id]

    def get(self, thread_id: str) -> GraphInterface:
        return self.graphs.get(thread_id, None)