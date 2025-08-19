from typing import Annotated, List, Optional, Set
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.runnables.base import RunnableLike
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, BaseMessage

class Conversation(BaseModel):
    project_id: str | None = ""
    conversation_uuid: str | None = ""
    message: str | None = ""
    tenant_id: str | None = ""
    document_from_user: List[Document] | None = list()
    is_stream: bool | None = False

class State(TypedDict):
    context: str
    question: str
    answer: str
    index: int
    document_from_user: List[Document]
    conversation: Annotated[list, add_messages]
    tenant_id: str
    project_uuid: str
    request_token_count: int
    response_token_count: int

class ConversationState(BaseModel):
    question: str | None = ""
    answer: str | None = ""

class ConversationalChatbot:
    def __init__(self):
        self._graph_builder = StateGraph(StateV2)

    def add_node(self, node_name: str, node: RunnableLike):
        self._graph_builder.add_node(node_name, node)

    def add_conditional_edges(self, source, path):
        self._graph_builder.add_conditional_edges(source, path)

    def add_edges(self, start_key, end_key):
        self._graph_builder.add_edge(start_key, end_key)

    def compile_graph(self, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
        app = self._graph_builder.compile(checkpointer=checkpointer)
        return app
    
class RouterOutputV2(BaseModel):
    """Response from LLM Router"""
    output_router: str = Field(description="The output of the LLM Router")
    table_name: str = Field(description="The name of the table to be used choosen by LLM")
    rephrased_question: str = Field(description="The rephrased question of the user's question")


class AgentResponseV2(BaseModel):
    """Response for user's question"""
    answer: str = Field(description="The answer of the user's question")
    references: Set[str] = Field(description="list of resource's url")
    needs_clarification: bool = Field(description="Whether system needed clarification from user or not")
    
class DatabaseConfig(BaseModel):
    dataset_name: str
    db_type: str
    db_host: str
    db_port: int
    db_name: str
    db_username: str
    db_password: str
    table_name: str
    db_uri: Optional[str] = ""

    def set_db_uri(self) -> str:
        if self.db_type == "postgresql":
            return f"postgresql://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        elif self.db_type == "mysql":
            return f"mysql://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        elif self.db_type == "sqlite":
            return f"sqlite:///{self.dataset_name}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

class StateV2(MessagesState):
    # Conversation memory
    summarized_messages: List[AnyMessage]
    # Agent related
    router_output: str
    question: str
    agent_answer: AgentResponseV2
    # Document related
    document_from_user: List[Document]
    document_idx: int
    context: List[Document]
    # Database related
    database_config: List[DatabaseConfig]
    db_name: str

