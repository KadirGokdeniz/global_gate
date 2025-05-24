from pydantic_settings import BaseSettings
from pydantic import Field, validator
from functools import lru_cache
from typing import Optional
import os

class Settings(BaseSettings):
    # Database
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_database: str = Field(default="global_gate")
    db_user: str = Field(default="postgres")
    db_password: str = Field(...)  # ✅ Zorunlu, default yok
    
    # AI & Embeddings
    openai_api_key: Optional[str] = Field(default=None)
    embedding_model: str = Field(default="text-embedding-ada-002")
    embedding_dimension: int = Field(default=384)  # ✅ Sabit boyut
    
    # Performance & Limits
    max_search_results: int = Field(default=10, le=50)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Environment
    environment: str = Field(default="development")
    debug_mode: bool = Field(default=False)  # ✅ Production'da False
    log_level: str = Field(default="INFO")
    
    @validator('embedding_dimension')
    def validate_dimension(cls, v):
        allowed_dims = [384, 768, 1536]  # Common dimensions
        if v not in allowed_dims:
            raise ValueError(f"Dimension must be one of {allowed_dims}")
        return v
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed_envs = ["development", "staging", "production"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of {allowed_envs}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
    
    def get_asyncpg_params(self) -> dict:
        return {
            "host": self.db_host,
            "port": self.db_port,
            "database": self.db_database,
            "user": self.db_user,
            "password": self.db_password,
        }
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

@lru_cache()
def get_settings() -> Settings:
    return Settings()