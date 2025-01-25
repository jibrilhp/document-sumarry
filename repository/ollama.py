from flask import current_app
from langchain_ollama import OllamaEmbeddings, ChatOllama
from entity.document import Chat
from typing import List
from langchain_core.documents import Document as LangchaincoreDocument
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
class OllamaAdapter:
    def __init__(self):
        self.app = current_app
        self.ollama_host = current_app.config["OLLAMA_HOST"]
        self.ollama_embedding_model = current_app.config["EMBEDDING_MODEL_NAME"]
        self.ollama_chat_model = current_app.config["CHAT_MODEL_NAME"]
        self.embedding_model = OllamaEmbeddings(
            base_url=self.ollama_host,
            model=self.ollama_embedding_model
        ) 
        self.chat_model = ChatOllama(
            base_url=self.ollama_host,
            model=self.ollama_chat_model
        )
        current_app.logger.info("ollama adapter ready")

    def generate_chat_response(self, chat: Chat, similiar_documents: List[LangchaincoreDocument]):
        self.app.logger.info("generate chat response with {} prompt".format(chat.chat))
        context = "\n".join([similiar_document.page_content for similiar_document in similiar_documents])
        template = """
        Given the following context:
        {context}

        Answer the question: {question}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = prompt | self.chat_model | StrOutputParser()
        response = chain.invoke({"context": context, "question": chat.chat})
        return response
    
    def generate_streamable_chat_response(self, chat: Chat, similiar_documents: List[LangchaincoreDocument]):
        self.app.logger.info("generate streamable chat response with {} prompt".format(chat.chat))
        context = "\n".join([similiar_document.page_content for similiar_document in similiar_documents])
        template = """
        Given the following context:
        {context}

        Answer the question: {question}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = prompt | self.chat_model | StrOutputParser()
        response = chain.stream({"context": context, "question": chat.chat}, )
        return response