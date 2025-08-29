import os
from typing import Optional
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    """Application settings with environment variable support."""
    
    gemini_api_key: str = ""
    max_upload_mb: int = 25
    model_store_path: str = "./artifacts"
    default_model: str = "prophet"
    sandbox_timeout: int = 30
    max_output_size: int = 1024 * 1024  # 1MB
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

# Global settings instance
settings = AppSettings()
