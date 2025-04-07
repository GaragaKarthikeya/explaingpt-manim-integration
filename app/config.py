import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Basic app settings
    APP_NAME: str = "Manim Animation Server"
    MANIM_OUTPUT_DIR: str = "./output_videos"
    MAX_QUEUE_SIZE: int = 10
    CACHE_EXPIRY_HOURS: int = 24
    PORT: int = int(os.getenv("PORT", 8000))
    
    # Environment detection
    IS_RAILWAY: bool = bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PUBLIC_DOMAIN"))
    
    # Base URL configuration - determined during startup
    BASE_URL: str = ""
    NGROK_BASE_URL: str = os.getenv("NGROK_BASE_URL", "")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 5
    
    # Gemini API settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "models/gemini-1.5-pro"  # Standard available model

    # Ngrok Auth Token (only used in development)
    NGROK_AUTHTOKEN: str = os.getenv("NGROK_AUTHTOKEN", "")

    class Config:
        env_file = ".env"
    
    def get_base_url(self):
        """Return the appropriate base URL based on environment"""
        if self.IS_RAILWAY:
            # Use Railway URL
            railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")
            if railway_domain:
                return f"https://{railway_domain}"
        
        # For development, prefer ngrok URL if available
        if self.NGROK_BASE_URL:
            return self.NGROK_BASE_URL
            
        # Fallback to localhost
        return f"http://localhost:{self.PORT}"

settings = Settings()