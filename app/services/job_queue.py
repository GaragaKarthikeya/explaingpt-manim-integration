import uuid
import time
from datetime import datetime
from typing import Dict, Optional, List
from app.models import JobInfo, JobStatus, AnimationRequest
from app.config import settings
import threading
import queue

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, JobInfo] = {}
        self.job_queue = queue.Queue(maxsize=settings.MAX_QUEUE_SIZE)
        self.job_data: Dict[str, AnimationRequest] = {}
        self._lock = threading.Lock()
        
    def add_job(self, job_request: AnimationRequest) -> JobInfo:
        """Add a new job to the queue."""
        if self.job_queue.qsize() >= settings.MAX_QUEUE_SIZE:
            raise ValueError("Queue is full, please try again later")
        
        job_id = str(uuid.uuid4())
        now = datetime.now()
        
        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            position_in_queue=self.job_queue.qsize() + 1
        )
        
        with self._lock:
            self.jobs[job_id] = job_info
            self.job_data[job_id] = job_request
            self.job_queue.put(job_id)
            
        return job_info
    
    def get_next_job(self) -> Optional[str]:
        """Get the next job from the queue."""
        try:
            return self.job_queue.get(block=False)
        except queue.Empty:
            return None
    
    def update_job_status(self, job_id: str, status: JobStatus, 
                          progress: Optional[float] = None, 
                          message: Optional[str] = None,
                          error: Optional[str] = None) -> Optional[JobInfo]:
        """Update the status of a job."""
        with self._lock:
            if job_id not in self.jobs:
                return None
                
            job_info = self.jobs[job_id]
            job_info.status = status
            job_info.updated_at = datetime.now()
            
            if progress is not None:
                job_info.progress = progress
                
            if message is not None:
                job_info.message = message
                
            if error is not None:
                job_info.error = error
                
            # Recalculate queue positions
            if status != JobStatus.QUEUED:
                job_info.position_in_queue = None
                self._update_queue_positions()
                
            return job_info
    
    def get_job_status(self, job_id: str) -> Optional[JobInfo]:
        """Get the status of a job."""
        with self._lock:
            return self.jobs.get(job_id)
    
    def get_job_request(self, job_id: str) -> Optional[AnimationRequest]:
        """Get the original request data for a job."""
        with self._lock:
            return self.job_data.get(job_id)
    
    def _update_queue_positions(self):
        """Update the queue positions of all queued jobs."""
        queued_jobs = [job_id for job_id, info in self.jobs.items() 
                       if info.status == JobStatus.QUEUED]
        
        for position, job_id in enumerate(queued_jobs, 1):
            self.jobs[job_id].position_in_queue = position

# Create a singleton instance
job_queue = JobQueue()