from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # App
    app_name: str = "TaskRAG"
    app_version: str = "1.0.0"

    # Dataset
    DATASET_PATH: str = "mandarjoshi/trivia_qa"
    DATASET_CONFIG: str = "rc"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"  # 'cpu' or 'cuda'

    # Vector Search
    SIMILARITY_METRIC: Literal["cosine", "l2"] = "cosine"
    TOP_K: int = 5

    # Cohere LLM settings
    COHERE_API_KEY: str = "mW5bq0Cvna7A352obfYScujemgBuoqatuCixztcl"
    COHERE_MODEL: str = "command-a-03-2025"  # default Cohere model name
    COHERE_MAX_TOKENS: int = 256
    COHERE_TEMPERATURE: float = 0.3

    class Config:
        env_file = ".env"

def get_settings() -> Settings:
    return Settings()
