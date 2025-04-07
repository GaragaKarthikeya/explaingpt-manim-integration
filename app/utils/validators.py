from app.models import AnimationRequest
from app.config import settings
import time
from fastapi import Request, HTTPException, status
import logging

logger = logging.getLogger(__name__)

# Store request timestamps for rate limiting
request_timestamps = {}

async def validate_animation_request(request: AnimationRequest) -> None:
    """Validate the animation request."""
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt cannot be empty"
        )
    
    if len(request.prompt) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt is too long, maximum 1000 characters allowed"
        )

async def rate_limit_middleware(request: Request):
    """Check if the client is rate limited."""
    client_ip = request.client.host
    
    # Only rate limit the generate endpoint
    if not request.url.path.endswith("/generate"):
        return
    
    current_time = time.time()
    
    # Get the list of timestamps for this client, or create a new one
    timestamps = request_timestamps.get(client_ip, [])
    
    # Remove timestamps older than 1 minute
    timestamps = [ts for ts in timestamps if current_time - ts < 60]
    
    # Check if client has exceeded rate limit
    if len(timestamps) >= settings.RATE_LIMIT_PER_MINUTE:
        logger.warning(f"Rate limit exceeded for client {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute allowed."
        )
    
    # Add current timestamp to the list
    timestamps.append(current_time)
    request_timestamps[client_ip] = timestamps