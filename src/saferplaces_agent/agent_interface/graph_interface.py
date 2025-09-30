import os
import json
from textwrap import indent
import datetime

from typing import Any, Literal

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage
from langchain_core.load import load as lc_load

from ..graph import graph
from ..common import s3_utils, utils
from ..common import states as GraphStates
from .chat_handler import ChatHandler

from IPython.display import display, Markdown

class GraphInterface:

    def __init__(
        self, 
        thread_id: str,
        user_id: str,
        project_id: str,
    ):
        self.G: CompiledStateGraph = graph
        self.thread_id = thread_id
        self.user_id = user_id
        self.project_id = project_id

        self.interrupt = None

        self.config = { "configurable": { "thread_id": self.thread_id } }
        
        self.chat_events = []
        self.chat_handler = ChatHandler(chat_id=self.thread_id, title=f"Chat {user_id}", subtitle=f"Thread {thread_id}")
        
        s3_utils.setup_base_bucket(user_id=self.user_id, project_id=self.project_id)
        self.restore_state()
            
    @property
    def graph_state(self):
        """ graph_state - returns the graph state """
        return self.G.get_state(self.config).values
            
    def restore_state(self):
        
        def restore_layer_registry():
            lr_uri = f'{s3_utils._BASE_BUCKET}/layer_registry.json'
            print(f"Restoring layer registry from {lr_uri}")
            lr_fp = s3_utils.s3_download(uri=lr_uri, fileout=os.path.join(os.getcwd(), f'{self.user_id}__{self.project_id}__layer_registry.json'))   # TODO: TMP DIR! + garbage collect
            if lr_fp is not None and os.path.exists(lr_fp):
                with open(lr_fp, 'r') as f:
                    layer_registry = json.load(f)
                return layer_registry
            return []
        
        restored_layer_registry = restore_layer_registry()
        _ = list( self.G.stream(
            input = { 
                'messages': [ GraphStates.build_layer_registry_system_message(restored_layer_registry) ],
                'layer_registry': restored_layer_registry
            }, 
            config = self.config, stream_mode = 'updates'
        ) )
        
        
    def get_state(self, key: str | list | None = None, fallback: Any = None) -> Any:
        state = self.graph_state
        if key is None:
            return state
        if isinstance(key, str):
            return state.get(key, fallback)
        if isinstance(key, list):
            return {k: state.get(k, fallback) for k in key}
        
        
    def register_layer(self, 
        src: str,
        title: str | None = None, 
        description: str | None = None,
        type: Literal['vector', 'raster'] = None,
        metadata: dict | None = None,
    ):
        """Register a new layer in the Layer Registry."""
        layer_dict = {
            'title': title if title else utils.juststem(src),
            ** ({ 'description': description } if description else dict()),
            'src': src,
            'type': type if type else 'vector' if utils.justext(src) in ['geojson', 'gpkg', 'shp'] else 'raster',
            ** ({ 'metadata': metadata } if metadata else dict()),
        }
        event_value = { 'layer_registry': [layer_dict] }
        
        _ = list( self.G.stream(
            input = event_value,
            config = self.config, stream_mode = 'updates'
        ) )
        self.on_end_event(event_value)


    def _event_value_is_interrupt(self, event_value):
        return type(event_value) is tuple and type(event_value[0]) is Interrupt
    
    def _event_value2interrupt(self, event_value):
        if self._event_value_is_interrupt(event_value):
            return event_value[0]
        return None
        
    def _interrupt2dict(self, interrupt):
        interrupt_data = interrupt.value
        agent_interrupt_message = { 'interrupt': interrupt_data }
        return agent_interrupt_message
    
    
    def update_events(self, new_events: AnyMessage | Interrupt | list[AnyMessage | Interrupt]):
        """Update the chat events with new events."""
        if isinstance(new_events, list):
            self.chat_events.extend(new_events)
            self.chat_handler.add_events(new_events)
        else:
            self.chat_events.append(new_events)
            self.chat_handler.add_events(new_events)
            
    def on_end_event(self, event_value):
            
        def update_layer_registry(event_value):
            if type(event_value) is dict and event_value.get('layer_registry'):
                layer_registry = self.get_state('layer_registry')
                lr_uri = f'{s3_utils._BASE_BUCKET}/layer_registry.json'
                lr_fp = os.path.join(os.getcwd(), f'{self.user_id}__{self.project_id}__layer_registry.json')
                with open(lr_fp, 'w') as f:
                    json.dump(layer_registry, f, indent=4)
                _ = s3_utils.s3_upload(filename=lr_fp, uri=lr_uri, remove_src= True )
                
        update_layer_registry(event_value)
        

    def user_prompt(
        self,
        prompt: str,
        state_updates: dict = dict(),
    ):
        
        def prepare_system_messages():            
            system_messages = []
            system_messages.append(GraphStates.build_nowtime_system_message())
            if 'layer_registry' in state_updates and state_updates['layer_registry']:
                system_messages.append(GraphStates.build_layer_registry_system_message(state_updates.get('layer_registry', [])))
            return system_messages
        
        def build_stream():
            stream_obj = dict()
            if self.interrupt is not None:
                self.update_events(HumanMessage(content=prompt, resume_interrupt={ 'interrupt_type': self.interrupt.value['interrupt_type'], 'resumable': self.interrupt.resumable, 'ns': self.interrupt.ns }))
                self.interrupt = None
                stream_obj = Command(resume={'response': prompt})
            else:
                self.update_events(HumanMessage(content=prompt))
                stream_obj = {
                    'messages': [
                        * prepare_system_messages(),
                        HumanMessage(content=prompt)
                    ],
                    'user_id': self.user_id,
                    'project_id': self.project_id,
                    'node_params': state_updates.get('node_params', dict()),
                    'node_history': state_updates.get('node_history', []),
                    'layer_registry': state_updates.get('layer_registry', []),
                    'avaliable_tools': state_updates.get('avaliable_tools', self.get_state('avaliable_tools', [])),
                    'nowtime': datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None).isoformat(),
                }
            return stream_obj
        
        def process_event_value(event_value):
            if 'messages' in event_value:
                event_value['message'] = event_value['messages'][-1].to_json()
                del event_value['messages']
                self.update_events(lc_load(event_value['message']))     # !!!: json-message to obj-message → lc_load is in BETA mode, so may not work in future versions
                
            elif self._event_value_is_interrupt(event_value):
                self.interrupt = self._event_value2interrupt(event_value)
                self.update_events(self.interrupt)
                
            self.on_end_event(event_value)
            
                     
        stream_prompt = build_stream()
        yield self.chat_handler.get_new_events
        
        for event in self.G.stream(
            input = stream_prompt,
            config = self.config,
            stream_mode = 'updates'
        ):
            for event_value in event.values():
                if event_value is not None:
                    process_event_value(event_value)    
                    yield self.chat_handler.get_new_events
                    
        self.on_end_event(stream_prompt) # ???: non so se serve forse
        
        
    def chat_markdown(self, user_prompt: str, display_output: bool = True):
        gen = (
            Markdown(self.chat_handler.chat_to_markdown(chat=e, include_header=False))
            for e in self.user_prompt(prompt=user_prompt, state_updates={'avaliable_tools': []})
        )
        if display_output:
            for md in gen:
                display(md)
        else:
            yield from gen
    

class GraphRegistry:

    """Registry for the agent graph."""
    
    def __init__(self):
        self.graphs = dict()

    def register(self, thread_id: str, user_id: str, **gi_kwargs) -> GraphInterface:
        self.graphs[thread_id] = GraphInterface(thread_id, user_id, **gi_kwargs)
        return self.graphs[thread_id]

    def get(self, thread_id: str) -> GraphInterface:
        return self.graphs.get(thread_id, None)