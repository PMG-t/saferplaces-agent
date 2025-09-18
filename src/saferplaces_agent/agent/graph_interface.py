import os
import datetime

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from .graph import graph


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
        
        def build_stream():
            stream_obj = dict()
            if self.interrupted:
                self.interrupted = False
                stream_obj = Command(resume={'response': prompt})
            else:
                stream_obj = {
                    'messages': [{'role': 'user', 'content': prompt}],
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