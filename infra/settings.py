from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    EMBEDDING_MODEL_NAME: str
    CHAT_MODEL_NAME: str
    COLLECTION_NAME: str
    LLM_PROVIDER: str
    OLLAMA_BASE_URL: str
    MAX_FILE_SIZE_IN_MB: int
    SECRET_KEY: str
    JWT_ALGORITHM: str

    model_config = SettingsConfigDict(env_file=".env")
