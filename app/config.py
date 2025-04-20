import os
import multiprocessing
import psutil
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Manim Animation Server"
    MANIM_OUTPUT_DIR: str = "./output_videos"
    MAX_QUEUE_SIZE: int = 10
    NGROK_BASE_URL: str = os.getenv("NGROK_BASE_URL", "")
    CACHE_EXPIRY_HOURS: int = 24
    PORT: int = int(os.getenv("PORT", 8000))
    
    # Multi-processing settings
    # Leave one core free for system operations by default
    MAX_PARALLEL_RENDERINGS: int = int(os.getenv("MAX_PARALLEL_RENDERINGS", max(1, multiprocessing.cpu_count() - 1)))
    # Set a minimum amount of available memory required per worker (in MB)
    MIN_MEMORY_PER_WORKER_MB: int = int(os.getenv("MIN_MEMORY_PER_WORKER_MB", 2048))
    # Maximum memory percentage to use (0-100)
    MAX_MEMORY_PERCENT: int = int(os.getenv("MAX_MEMORY_PERCENT", 80))
    # Enable dynamic worker scaling based on system load
    ENABLE_DYNAMIC_SCALING: bool = os.getenv("ENABLE_DYNAMIC_SCALING", "True").lower() in ("true", "1", "t")
    # System load threshold to reduce workers (0.0-1.0)
    HIGH_LOAD_THRESHOLD: float = float(os.getenv("HIGH_LOAD_THRESHOLD", 0.8))
    # System load threshold to add workers (0.0-1.0)
    LOW_LOAD_THRESHOLD: float = float(os.getenv("LOW_LOAD_THRESHOLD", 0.3))
    # Resource monitoring interval in seconds
    MONITORING_INTERVAL_SEC: int = int(os.getenv("MONITORING_INTERVAL_SEC", 30))
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 5
    
    # Gemini API settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    # Update to use the exact model name from list_gemini_models.py output
    GEMINI_MODEL: str = "models/gemini-2.0-flash"

    # Ngrok settings
    NGROK_AUTHTOKEN: str = os.getenv("NGROK_AUTHTOKEN", "")
    ENABLE_NGROK: bool = os.getenv("ENABLE_NGROK", "True").lower() in ("true", "1", "t")
    NGROK_MAX_RETRIES: int = int(os.getenv("NGROK_MAX_RETRIES", 2))
    NGROK_RETRY_DELAY: int = int(os.getenv("NGROK_RETRY_DELAY", 2))
    NGROK_DOMAIN: str = os.getenv("NGROK_DOMAIN", "")

    # Error recovery settings
    ERROR_RECOVERY_ENABLED: bool = True
    ERROR_RECOVERY_MAX_RETRIES: int = 3
    ERROR_RECOVERY_SANDBOX_TIMEOUT: int = 30  # seconds

    # RAG Configuration
    RAG_INDEX_PATH: str = os.getenv("RAG_INDEX_PATH", "./data/manim_index.faiss")
    RAG_BLOCKS_PATH: str = os.getenv("RAG_BLOCKS_PATH", "./data/manim_blocks.npy")
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true").lower() == "true"
    RAG_MIN_SCORE: float = float(os.getenv("RAG_MIN_SCORE", "0.5"))
    RAG_MAX_EXAMPLES: int = int(os.getenv("RAG_MAX_EXAMPLES", "3"))

    class Config:
        env_file = ".env"

settings = Settings()