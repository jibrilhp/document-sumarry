from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama  
from langchain_ollama.embeddings import OllamaEmbeddings
from infra.settings import Settings
import logging
from langchain_core.rate_limiters import InMemoryRateLimiter

class GenerativeAdapter:
    def __init__(self, settings: Settings):
        self.__choose_provider(settings)
    
    def __choose_provider(self, settings: Settings):
        if settings.LLM_PROVIDER == "ollama":
            self.embedding_model = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL
            )
       
            self.chat_model = ChatOllama(
                model=settings.CHAT_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.0,
                format="json" 
            )
            logging.info(f"using ollama provider with base URL: {settings.OLLAMA_BASE_URL}. chat model: {settings.CHAT_MODEL_NAME}, embedding model: {settings.EMBEDDING_MODEL_NAME}")
        
        elif settings.LLM_PROVIDER == "google":
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=0.2,
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )
            self.embedding_model = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
            self.chat_model = ChatGoogleGenerativeAI(
                model=settings.CHAT_MODEL_NAME,
                rate_limiter=rate_limiter,
                max_retries=3,
                temperature=0.0
            )
            logging.info(f"using google generative provider")
        
        elif settings.LLM_PROVIDER == "ollama_cpu":
            self.embedding_model = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
            )
            self.chat_model = ChatOllama(
                model=settings.CHAT_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.0,
                format="json",
                num_gpu=0  
            )
            logging.info(f"using ollama provider (force CPU usage) with base URL: {settings.OLLAMA_BASE_URL}. chat model: {settings.CHAT_MODEL_NAME}, embedding model: {settings.EMBEDDING_MODEL_NAME}")
        
        else:
            raise ValueError("Invalid LLM provider")