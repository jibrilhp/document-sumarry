from typing import Annotated, Callable, Dict, Any
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables.base import RunnableLike
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

class ConversationIdentity:
    def __init__(self, tenant_id: str, project_uuid: str, conversation_uuid: str, message: str):
        self.tenant_id = tenant_id
        self.project_uuid = project_uuid
        self.conversation_uuid = conversation_uuid
        self.message = message

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    question: str

class ConversationalChatbot:
    def __init__(self):
        self._graph_builder = StateGraph(State)
        # self.memory = MemorySaver()

    def add_node(self, node_name: str, node: RunnableLike):
        self._graph_builder.add_node(node_name, node)

    def set_start_edge_node(self, node: str):
        self._graph_builder.add_edge(START, node)

    def set_end_edge_node(self, node: str):
        self._graph_builder.add_edge(node, END)

    def compile_graph(self, checkpointer: BaseCheckpointSaver) -> CompiledGraph:
        return self._graph_builder.compile(checkpointer=checkpointer)