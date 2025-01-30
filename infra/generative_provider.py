from flask import current_app
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


class GenerativeAdapter:
    def __init__(self):
        self.app = current_app
        self.ollama_host = current_app.config["OLLAMA_HOST"]
        self.ollama_embedding_model = current_app.config["EMBEDDING_MODEL_NAME"]
        self.ollama_chat_model = current_app.config["CHAT_MODEL_NAME"]
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        current_app.logger.info("ollama adapter ready")
