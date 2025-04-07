from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    RENDERING = "rendering"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"

class AnimationRequest(BaseModel):
    prompt: str = Field(..., description="User prompt for animation generation")
    animate: bool = Field(True, description="Flag to indicate if animation is needed")
    complexity: int = Field(1, description="Animation complexity (1-3)", ge=1, le=3)
    
class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    position_in_queue: Optional[int] = None
    progress: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None
    
class AnimationResponse(BaseModel):
    job_id: str
    message: str
    status: JobStatus
    
class AnimationResult(BaseModel):
    job_id: str
    video_url: Optional[str] = None
    success: bool
    error: Optional[str] = None