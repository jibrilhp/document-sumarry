from flask import current_app
from langchain_ollama import OllamaEmbeddings, ChatOllama
from entity.document import Chat
from entity.conversation import State
from typing import List
from langchain_core.documents import Document as LangchaincoreDocument
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from entity.conversation import ConversationIdentity


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
        # self.chat_model = ChatOllama(
        #     base_url=self.ollama_host,
        #     model=self.ollama_chat_model
        # )
        # self.embedding_model = GoogleGenerativeAIEmbeddings(
        #     model="models/embedding-001"
        # )
        self.chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        current_app.logger.info("ollama adapter ready")

    def generate_chat_response(self, chat: Chat, similiar_documents: List[LangchaincoreDocument]):
        self.app.logger.info("generate chat response with {} prompt".format(chat.chat))
        context = "\n".join([similiar_document.page_content for similiar_document in similiar_documents])
        template = """
        Given the following context:
        {context}

        Answer the question: {question}

        Important:
        - If the answer is known, respond accurately using only the information in the provided context.
        - If the answer is unknown or not explicitly mentioned in the context, respond **exactly** with:
          "Mohon maaf, kami tidak dapat menjawab pertanyaan anda" 
          (do not add any additional information, explanation, or encouragement).           """
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
        Important:
        - If the answer is known, respond accurately using only the information in the provided context.
        - If the answer is unknown or not explicitly mentioned in the context, respond **exactly** with:
          "Mohon maaf, kami tidak dapat menjawab pertanyaan anda" 
          (do not add any additional information, explanation, or encouragement).        
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = prompt | self.chat_model | StrOutputParser()
        response = chain.stream({"context": context, "question": chat.chat}, )
        return response
    
    def chat_bot(self, state: State):
        return {"messages": [self.chat_model.invoke(state["messages"])]}
    
    def generate_prompt(self, chat: ConversationIdentity, similiar_documents: List[LangchaincoreDocument]):
        context = "\n".join([similiar_document.page_content for similiar_document in similiar_documents])
        template = """
        Given the following context:
        {context}

        or you can take information from previous chat

        Answer the question: {question}

        Important:
        - If the answer is known, respond accurately using only the information in the provided context or previous chat.
        - If the answer is unknown or not explicitly mentioned in the context or previous chat, respond **exactly** with:
          "Mohon maaf, kami tidak dapat menjawab pertanyaan anda" 
          (do not add any additional information, explanation, or encouragement).           """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"]).format(context=context, question=chat.message)
        print(prompt)
        return prompt