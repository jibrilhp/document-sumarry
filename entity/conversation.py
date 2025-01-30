from typing import Annotated, List
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables.base import RunnableLike
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.documents import Document
from IPython.display import Image
from PIL import Image as PILImage
import io

class Conversation:

    document_from_user: List[Document] = None

    def __init__(self, tenant_id: str, project_uuid: str, conversation_uuid: str, message: str):
        self.tenant_id = tenant_id
        self.project_uuid = project_uuid
        self.conversation_uuid = conversation_uuid
        self.message = message

    def set_document_from_user(self, document_from_user: List[Document]):
        self.document_from_user = document_from_user

class State(TypedDict):
    context: str
    question: str
    answer: str
    index: int
    document_from_user: List[Document]

class ConversationalChatbot:
    def __init__(self):
        self._graph_builder = StateGraph(State)

    def add_node(self, node_name: str, node: RunnableLike):
        self._graph_builder.add_node(node_name, node)

    def add_conditional_edges(self, source, path):
        self._graph_builder.add_conditional_edges(source, path)

    def add_edges(self, start_key, end_key):
        self._graph_builder.add_edge(start_key, end_key)

    def compile_graph(self, checkpointer: BaseCheckpointSaver) -> CompiledGraph:
        app = self._graph_builder.compile()
        image_data = app.get_graph().draw_mermaid_png()

        # Convert the image data to a PIL Image object
        pil_image = PILImage.open(io.BytesIO(image_data))

        # Save the image to a file
        pil_image.save('output_image.png')
        return app