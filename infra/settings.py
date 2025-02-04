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

    model_config = SettingsConfigDict(env_file=".env")
