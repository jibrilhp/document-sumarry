from flask import current_app
from langchain_ollama import OllamaEmbeddings

class OllamaAdapter:
    def __init__(self):
        self.ollama_host = current_app.config["OLLAMA_HOST"]
        self.ollama_embedding_model = current_app.config["EMBEDDING_MODEL_NAME"]
        self.ollama_chat_model = current_app.config["CHAT_MODEL_NAME"]
        self.embedding_model = OllamaEmbeddings(
            base_url=self.ollama_host,
            model=self.ollama_embedding_model
        ) 
        current_app.logger.info("ollama adapter ready")

    def ollama_embedding(self):
        pass