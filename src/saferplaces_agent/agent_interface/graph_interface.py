import os
import json
import uuid
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

from .leafmap_interface import LeafmapInterface

from IPython.display import display, Markdown, clear_output

class GraphInterface:

    def __init__(
        self, 
        thread_id: str,
        user_id: str,
        project_id: str,
        map_handler: bool = False
    ):
        self.G: CompiledStateGraph = graph
        self.thread_id = thread_id
        self.user_id = user_id
        self.project_id = project_id

        self.interrupt = None

        self.config = { "configurable": { "thread_id": self.thread_id } }
        
        self.chat_events = []
        self.chat_handler = ChatHandler(chat_id=self.thread_id, title=f"Chat {user_id}", subtitle=f"Thread {thread_id}")
        
        self.map_handler = LeafmapInterface() if map_handler else None
        
        s3_utils.setup_base_bucket(user_id=self.user_id, project_id=self.project_id)
        self.restore_state()
            
        # if self.map_handler:
        #     clear_output(wait=True)
        #     display(self.map_handler.m)  # Display the map in the notebook
             
            
    @property
    def graph_state(self):
        """ graph_state - returns the graph state """
        return self.G.get_state(self.config).values
    
            
    def restore_state(self):
        
        def restore_layer_registry():
            lr_uri = f'{s3_utils._BASE_BUCKET}/layer_registry.json'
            print(f"Restoring layer registry from {lr_uri} ...")
            lr_fp = s3_utils.s3_download(uri=lr_uri, fileout=os.path.join(os.getcwd(), f'{self.user_id}__{self.project_id}__layer_registry.json'))   # TODO: TMP DIR! + garbage collect
            if lr_fp is not None and os.path.exists(lr_fp):
                with open(lr_fp, 'r') as f:
                    layer_registry = json.load(f)
                os.remove(lr_fp)
                return layer_registry
            return []
        
        restored_layer_registry = restore_layer_registry()
        event_value = { 
            'messages': [ GraphStates.build_layer_registry_system_message(restored_layer_registry) ],
            'layer_registry': restored_layer_registry 
        }
        _ = list( self.G.stream(
            input = event_value,
            config = self.config, stream_mode = 'updates'
        ) )
        self.on_end_event(event_value)
        
        
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
        layer_type: Literal['vector', 'raster'] = None,
        metadata: dict | None = None,
    ):
        """Register a new layer in the Layer Registry."""
        layer_dict = {
            'title': title if title else utils.juststem(src),
            ** ({ 'description': description } if description else dict()),
            'src': src,
            'type': layer_type if layer_type else 'vector' if utils.justext(src) in ['geojson', 'gpkg', 'shp'] else 'raster',
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
                
        def update_map(event_value):
            if self.map_handler and type(event_value) is dict and event_value.get('layer_registry'):
                display_map = False
                for layer in event_value['layer_registry']:
                    displayed = self.map_handler.add_layer(
                        src=layer['src'],
                        layer_type=layer['type'],
                        colormap_name=layer.get('metadata', {}).get('colormap_name', 'viridis'),
                        nodata=layer.get('metadata', {}).get('nodata', -9999),
                    )
                    if displayed:
                        display_map = True
                # if display_map:
                #     display(self.map_handler.m)
                
        update_layer_registry(event_value)
        update_map(event_value)
        

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
                self.update_events(lc_load(event_value['message']))     # !!!: json-message to obj-message → LangChainBetaWarning: The function `load` is in beta. It is actively being worked on, so the API may change.
                
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
                    
        self.on_end_event(stream_prompt) # ???: maybe it should be called before G.stream()
        
    
    # TODO: this handling will need to be moved in autonomous class 'ChatMarkdown') using GraphInterface    
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
            
    class ChatMarkdownBreak(StopIteration):
        """Custom exception to stop the chat markdown generator."""
        def __init__(self, message="Chat markdown generation stopped."):
            super().__init__(message)
            
    class ChatMarkdownContinue(Exception):
        """Custom exception to continue the chat markdown generator."""
        def __init__(self, message="Chat markdown generation continued."):
            super().__init__(message)
            
    class _ChatMarkdownCommandsHandler():
        
        def __init__(self, graph_interface_istance: 'GraphInterface' = None):
            self.graph_interface_istance = graph_interface_istance
            # self.NEW_CHAT = {
            #     'name': 'new_chat',
            #     'description': 'Start a new chat session.',
            #     'handler': self.command_new_chat
            # }
            self.LAYERS = {
                'name': 'layers',
                'description': 'List all layers in the Layer Registry.',
                'handler': self.command_layers
            }
            self.CLEAR = {
                'name': 'clear',
                'description': 'List all layers in the Layer Registry.',
                'handler': self.command_clear
            }
            self.HISTORY = {
                'name': 'history',
                'description': 'Display all past messages in the conversation.',
                'handler': self.command_history
            }
            self.MAP = {
                'name': 'map',
                'description': 'List all layers in the Layer Registry.',
                'handler': self.command_map
            }
            self.EXIT = {
                'name': 'exit',
                'description': 'List all layers in the Layer Registry.',
                'handler': self.command_exit
            }
            self.QUIT = {
                'name': 'quit',
                'description': 'List all layers in the Layer Registry.',
                'handler': self.command_exit
            }
            self.HELP = {
                'name': 'help',
                'description': 'List all layers in the Layer Registry.',
                'handler': self.command_help
            }
            
        def command_new_chat(self):
            
            print("New chat session started.")
            
        def command_layers(self):
            layers = self.graph_interface_istance.get_state('layer_registry')
            print(layers)
            
        def command_clear(self):
            clear_output()
            
        def command_history(self):
            """
            Display all past messages in the conversation.
            """
            chat_events = self.graph_interface_istance.chat_events
            if not chat_events:
                print("No past messages in the conversation.")
                return
            display(Markdown(self.graph_interface_istance.chat_handler.chat_to_markdown(chat=chat_events, include_header=False)))
            
        def command_map(self):
            if self.graph_interface_istance.map_handler:
                display(self.graph_interface_istance.map_handler.m)
                raise self.graph_interface_istance.ChatMarkdownBreak("Map displayed.")
            else:
                print("No map handler available.")
                
        def command_exit(self):
            print("Exiting the conversation.")
            raise self.graph_interface_istance.ChatMarkdownBreak("Exiting the conversation.")
        
        def command_help(self):
            commands = [
                self.LAYERS,
                self.CLEAR,
                self.MAP,
                self.EXIT,
                self.QUIT,
                self.HELP
            ]
            help_text = "\n".join([f"/{cmd['name']}: {cmd['description']}" for cmd in commands])
            print(f"Available commands:\n{help_text}")
        
        def handle_command(self, command:str):
            """
            Handle chat markdown commands based on the command string.
            Args:
                command (str): The command string to handle.
                self.graph_interface_istance (GraphInterface): The GraphInterface instance to use for handling commands.
            """
            # if command == self.NEW_CHAT['name']:  # TODO: ENABLE ONLY WHEN class ChatMarkowdnHandler is defined (will use the registry)
            #     self.NEW_CHAT['handler']()
            if command == self.LAYERS['name']:
                self.LAYERS['handler']()
            elif command == self.CLEAR['name']:
                self.CLEAR['handler']()
            elif command == self.HISTORY['name']:
                self.HISTORY['handler']()
            elif command == self.MAP['name']:
                self.MAP['handler']()
            elif command == self.EXIT['name'] or command == self.QUIT['name']:
                self.EXIT['handler']()
                
            elif command == self.HELP['name']:
                self.HELP['handler']()
            else:
                print(f"Unknown command: {command}")
            raise self.graph_interface_istance.ChatMarkdownContinue("Continuing the conversation.")
            
    @property
    def chat_markdown_commands_handler(self):
        """Returns the chat markdown commands handler."""
        if not hasattr(self, '_chat_markdown_commands_handler'):
            self._chat_markdown_commands_handler = self._ChatMarkdownCommandsHandler(graph_interface_istance=self)
        return self._chat_markdown_commands_handler
    

    

class GraphRegistry:

    """Registry for the agent graph."""
    
    def __init__(self):
        self.graphs = dict()

    def register(self, thread_id: str, user_id: str, **gi_kwargs) -> GraphInterface:
        self.graphs[thread_id] = GraphInterface(thread_id, user_id, **gi_kwargs)
        return self.graphs[thread_id]

    def get(self, thread_id: str) -> GraphInterface:
        return self.graphs.get(thread_id, None)