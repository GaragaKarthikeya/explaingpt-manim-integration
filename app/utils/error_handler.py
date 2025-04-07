from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AppError(Exception):
    def __init__(self, status_code: int, message: str, details: Any = None):
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(self.message)

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle application-specific errors."""
    error_response = {
        "success": False,
        "message": exc.message,
    }
    
    if exc.details:
        error_response["details"] = exc.details
    
    logger.error(f"AppError: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response,
    )

class ErrorHandler:
    @staticmethod
    def handle_rendering_error(error: Exception) -> AppError:
        """Handle errors that occur during rendering."""
        message = str(error)
        if "MemoryError" in message or "memory" in message.lower():
            return AppError(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Not enough memory to render animation. Try reducing complexity.",
                details=str(error)
            )
        elif "Timeout" in message or "timed out" in message.lower():
            return AppError(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                message="Animation rendering timed out. Try reducing complexity.",
                details=str(error)
            )
        else:
            return AppError(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Failed to render animation",
                details=str(error)
            )