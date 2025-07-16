from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from infra.settings import Settings
import logging


class GenerativeAdapter:
    def __init__(self, settings: Settings):
        self.__choose_provider(settings)

    def __choose_provider(self, settings: Settings):
        if settings.LLM_PROVIDER == "ollama":
            self.embedding_model = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL
            )
            self.chat_model = OllamaLLM(
                model=settings.CHAT_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL
            )
            logging.info(f"using ollama provider with base URL: {settings.OLLAMA_BASE_URL}.  chat model: {settings.CHAT_MODEL_NAME}, embedding model: {settings.EMBEDDING_MODEL_NAME}")
        elif settings.LLM_PROVIDER == "google":
            self.embedding_model = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
            self.chat_model = ChatGoogleGenerativeAI(model=settings.CHAT_MODEL_NAME)
            logging.info(f"using google generative provider")
        elif settings.LLM_PROVIDER == "ollama_cpu":
            self.embedding_model = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
            )
            self.chat_model = OllamaLLM(
                model=settings.CHAT_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
                num_gpu=0  # Force CPU usage
            )
            logging.info(f"using ollama provider (force CPU usage) with base URL: {settings.OLLAMA_BASE_URL}. chat model: {settings.CHAT_MODEL_NAME}, embedding model: {settings.EMBEDDING_MODEL_NAME}")
        else:
            raise ValueError("Invalid LLM provider")