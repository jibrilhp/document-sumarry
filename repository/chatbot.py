from typing import Dict, Literal, List
from entity.conversation import ConversationalChatbot, State, ConversationState
from langgraph.graph.graph import CompiledGraph
from infra.data_store import PostgresAdapter
from infra.generative_provider import GenerativeAdapter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END
import logging

class ChatBotRepository:
    def __init__(self, generative_provider: GenerativeAdapter, postgres_adapter: PostgresAdapter):
        self.chat_states: Dict[str,ConversationalChatbot] = {}
        self.checkpointer = postgres_adapter.get_checkpointer()
        self.generative_adapter = generative_provider
        self.pgvector = postgres_adapter.get_vector_store()
        self.logger = logging.getLogger(__name__)

    def create_chatbot(self, thread_id: str) -> CompiledGraph:
        chatbot = ConversationalChatbot()

        chatbot.add_node("generate_initial_summary", self.__generate_initial_summary)
        chatbot.add_node("refine_summary", self.__generate_summary_refinement)
        chatbot.add_node("generate_chat_response", self.__generate_chat_response)
        chatbot.add_node("fetch_context", self.__fetch_context)
        chatbot.add_node("add_answer_to_conversation", self.__add_answer_to_conversation)

        chatbot.add_conditional_edges(source=START, path=self.__should_summarize)
        chatbot.add_edges(start_key="fetch_context", end_key="generate_chat_response")
        chatbot.add_edges(start_key="generate_chat_response", end_key="add_answer_to_conversation")
        chatbot.add_conditional_edges(source="generate_initial_summary", path=self.__should_refine)
        chatbot.add_conditional_edges(source="refine_summary", path=self.__should_refine)
        chatbot.add_edges(start_key="add_answer_to_conversation", end_key=END)

        compiled_graph = chatbot.compile_graph(checkpointer=self.checkpointer)
        self.chat_states[thread_id] = compiled_graph
        return compiled_graph

    def get_chatbot(self, thread_id: str) ->CompiledGraph:
        chat_bot = self.chat_states.get(thread_id)
        return chat_bot
    
    def __create_initial_summary_chain(self):
        summarize_prompt = ChatPromptTemplate(
            [
                ("human", "Buat ringkasan dari konteks berikut {context}")
            ]
            )
        return summarize_prompt | self.generative_adapter.chat_model | StrOutputParser()
    
    def __create_summary_refinement_chain(self):
        refine_template = """
            Buatkan sebuah ringkasan akhir

            Ringkasan saat ini adalah:
            {existing_answer}

            Konteks baru:
            ------------
            {context}
            ------------

            Dengan konteks baru, perbaiki ringkasan awal
        """
        refine_prompt = ChatPromptTemplate([("human", refine_template)])
        return refine_prompt | self.generative_adapter.chat_model | StrOutputParser()

    def __create_answer_chain(self):
        template = """
            Anda adalah pembantu yang handal dan punya pekerjaan untuk membantu pengguna dan meringkas dokumen.
            Anda bisa mendiskusikan konten dari dokumen dengan konteks berikut {context} atau anda bisa gunakan informasi dari percakapan berikut {conversation}.
            Apabila pengguna mengguna
            Jawab pertanyaan ini {question}  
        """
        answer_prompt = ChatPromptTemplate([("human", template)])
        return answer_prompt | self.generative_adapter.chat_model | StrOutputParser()
    
    def __generate_initial_summary(self, state: State, config: RunnableConfig) -> State:
        self.logger.info("on __generate_initial_summary")
        summary = self.__create_initial_summary_chain().invoke(
              input=state["document_from_user"][0],
              config=config,
          )
        return {"answer": summary, "index": 1}
    
    def __generate_summary_refinement(self, state: State, config: RunnableConfig):
        self.logger.info("on __generate_summary_refinement")
        refined_summary = self.__create_summary_refinement_chain().invoke(
                {
                    "existing_answer": state["answer"],
                    "context": state["document_from_user"][state["index"]]
                },
                config=config
        )
        return  {"answer": refined_summary, "index": state["index"] + 1}
    
    def __generate_chat_response(self, state: State):
        self.logger.info("on __generate_chat_response")
        chain = self.__create_answer_chain()
        answer = chain.invoke(
          {    
            "conversation": state["conversation"],
            "question": state["conversation"][-1],
            "context": state["context"]
          }
        )
        self.logger.info(answer)
        return {"answer": answer}
    
    def __should_refine(self, state: State) -> Literal["refine_summary", "add_answer_to_conversation"]:
        self.logger.info("on __should_refine")
        if state["index"] >= len(state["document_from_user"]):
            self.logger.info("go to add_answer_to_conversation")
            return "add_answer_to_conversation"
        else:
            self.logger.info("go to refine summary")
            return "refine_summary"
    
    def __should_summarize(self, state: State) -> Literal["fetch_context", "generate_initial_summary"]:
        self.logger.info("on __should_summarize")
        if state.get("document_from_user") is None:
            self.logger.info("go to fetch_context")
            return "fetch_context"
        self.logger.info("go to generate_initial_summary")
        return "generate_initial_summary"

    def __fetch_context(self, state: State) -> State:
        relevant_docs_with_score = self.pgvector.similarity_search_with_relevance_scores(
            query=state.get("conversation")[-1].content,
            score_threshold=0.6
        )
        context = "\n".join([similiar_document[0].page_content for similiar_document in relevant_docs_with_score])
        return {"context": context}
    
    def __add_answer_to_conversation(self, state: State)-> State:
        self.logger.info("on __add_response_to_conversation")
        return {"conversation": [state.get("answer")]}
    
    def get_chat_history(self, config: RunnableConfig, compiled_graph: CompiledGraph):
        conversation_list: List[ConversationState] = []
        conversation_state_value: List = compiled_graph.get_state(config).values.get("conversation")

        if conversation_state_value is None:
            return []

        for idx, v in enumerate(conversation_state_value, start=1):
            print(v)
            if idx % 2 == 1:
                conversation = ConversationState()
                conversation.question = v.content
            else:  
                conversation.answer = v.content
                conversation_list.append(conversation)

        return conversation_list


        