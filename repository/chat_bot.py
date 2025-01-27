from typing import Dict
from entity.conversation import ConversationalChatbot
from langchain_core.runnables.base import RunnableLike
from langgraph.graph.graph import CompiledGraph
from repository.data_store import PostgresCheckpointer

class ChatBotRepository:
    def __init__(self, postgres_checkpointer: PostgresCheckpointer):
        self.chat_states: Dict[str,ConversationalChatbot] = {}
        self.checkpointer = postgres_checkpointer.get_checkpointer()

    def create_chat_state(self, thread_id: str, runnable: RunnableLike) -> CompiledGraph:
        chat_bot = ConversationalChatbot()
        chat_bot.add_node("chatbot", runnable)
        chat_bot.set_start_edge_node("chatbot")
        chat_bot.set_end_edge_node("chatbot")
        compiled_graph = chat_bot.compile_graph(checkpointer=self.checkpointer)
        self.chat_states[thread_id] = compiled_graph
        return compiled_graph

    def get_chat_state(self, thread_id: str) ->CompiledGraph:
        print("in repo", self.chat_states)
        chat_bot = self.chat_states.get(thread_id)
        return chat_bot