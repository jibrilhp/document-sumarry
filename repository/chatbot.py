from typing import Dict, Literal
from entity.conversation import ConversationalChatbot, State
from langgraph.graph.graph import CompiledGraph
from infra.data_store import PostgresCheckpointer, PGVectorAdapter
from infra.generative_provider import GenerativeAdapter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END
from flask import current_app

class ChatBotRepository:
    def __init__(self, generative_provider: GenerativeAdapter, postgres_checkpointer: PostgresCheckpointer, pgvector: PGVectorAdapter):
        self.chat_states: Dict[str,ConversationalChatbot] = {}
        self.checkpointer = postgres_checkpointer.get_checkpointer()
        self.generative_adapter = generative_provider
        self.pgvector = pgvector
        self.app = current_app

    def create_chatbot(self, thread_id: str) -> CompiledGraph:
        chatbot = ConversationalChatbot()

        chatbot.add_node("generate_initial_summary", self.__generate_initial_summary)
        chatbot.add_node("refine_summary", self.__generate_summary_refinement)
        chatbot.add_node("generate_chat_response", self.__generate_chat_response)
        chatbot.add_node("fetch_context", self.__fetch_context)

        chatbot.add_conditional_edges(source=START, path=self.__should_summarize)
        chatbot.add_edges(start_key="fetch_context", end_key="generate_chat_response")
        chatbot.add_edges(start_key="generate_chat_response", end_key=END)
        chatbot.add_conditional_edges(source="generate_initial_summary", path=self.__should_refine)
        chatbot.add_conditional_edges(source="refine_summary", path=self.__should_refine)

        compiled_graph = chatbot.compile_graph(checkpointer=self.checkpointer)
        self.chat_states[thread_id] = compiled_graph
        return compiled_graph

    def get_chatbot(self, thread_id: str) ->CompiledGraph:
        print("in repo", self.chat_states)
        chat_bot = self.chat_states.get(thread_id)
        return chat_bot
    
    def __create_initial_summary_chain(self):
        summarize_prompt = ChatPromptTemplate(
            [
                ("human", "Write a concise summary of the following: {context}")
            ]
            )
        return summarize_prompt | self.generative_adapter.chat_model | StrOutputParser()
    
    def __create_summary_refinement_chain(self):
        refine_template = """
            Produce a final summary.

            Existing summary up to this point:
            {existing_answer}

            New context:
            ------------
            {context}
            ------------

            Given the new context, refine the original summary.
        """
        refine_prompt = ChatPromptTemplate([("human", refine_template)])
        return refine_prompt | self.generative_adapter.chat_model | StrOutputParser()

    def __create_answer_chain(self):
        template = """
            You are a helpful helper whose job to help your user to summarize and document.
            You can discuss the content of the document with this context {context} or you can use information from previous conversation.
            Answer this question {question}  
        """
        answer_prompt = ChatPromptTemplate([("human", template)])
        return answer_prompt | self.generative_adapter.chat_model | StrOutputParser()
    
    def __generate_initial_summary(self, state: State, config: RunnableConfig) -> State:
        self.app.logger.info("on __generate_initial_summary")
        summary = self.__create_initial_summary_chain().invoke(
              input=state["document_from_user"][0],
              config=config,
          )
        return {"answer": summary, "index": 1}
    
    def __generate_summary_refinement(self, state: State, config: RunnableConfig):
        self.app.logger.info("on __generate_summary_refinement")
        print(state.get("answer"))
        print(state.get("document_from_user")[state.get("index")])
        print(state.get("index"))
        refined_summary = self.__create_summary_refinement_chain().invoke(
                {
                    "existing_answer": state["answer"],
                    "context": state["document_from_user"][state["index"]]
                },
                config=config
        )
        return  {"answer": refined_summary, "index": state["index"] + 1}
    
    def __generate_chat_response(self, state: State):
        self.app.logger.info("on __generate_chat_response")
        print(state)
        chain = self.__create_answer_chain()
        print(chain)
        answer = chain.invoke(
          {    
            "question": state["question"],
            "context": state["context"]
          }
        )
        self.app.logger.info(answer)
        return {"answer": answer}
    
    def __should_refine(self, state: State) -> Literal["refine_summary", END]:
        self.app.logger.info("on __should_refine")
        if state["index"] >= len(state["document_from_user"]):
            self.app.logger.info("go to END")
            return END
        else:
            self.app.logger.info("go to refine summary")
            return "refine_summary"
    
    def __should_summarize(self, state: State) -> Literal["fetch_context", "generate_initial_summary"]:
        self.app.logger.info("on __should_summarize")
        if state.get("document_from_user") is None:
            self.app.logger.info("go to fetch_context")
            return "fetch_context"
        self.app.logger.info("go to generate_initial_summary")
        return "generate_initial_summary"

    def __fetch_context(self, state: State) -> State:
        self.app.logger.info("on __fetch_context")
        relevant_docs_with_score = self.pgvector.vector_store.similarity_search_with_relevance_scores(
            query=state["question"],
            score_threshold=0.6
        )
        context = "\n".join([similiar_document[0].page_content for similiar_document in relevant_docs_with_score])
        self.app.logger.info(context)
        return {"context": context}
    

        