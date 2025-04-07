import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Manim Animation Server"
    MANIM_OUTPUT_DIR: str = "./output_videos"
    MAX_QUEUE_SIZE: int = 10
    NGROK_BASE_URL: str = os.getenv("NGROK_BASE_URL", "")
    CACHE_EXPIRY_HOURS: int = 24
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 5
    
    # Gemini API settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    # Use a standard Gemini model that's widely available
    GEMINI_MODEL: str = "models/gemini-1.5-pro"  # This is a standard available model

    # Ngrok Auth Token
    NGROK_AUTHTOKEN: str = os.getenv("NGROK_AUTHTOKEN", "")

    class Config:
        env_file = ".env"

settings = Settings()