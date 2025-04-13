import uuid
import time
import threading
import logging
import multiprocessing
from multiprocessing import Manager
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from queue import Queue

from app.models import JobStatus, AnimationRequest

logger = logging.getLogger(__name__)

@dataclass
class JobInfo:
    job_id: str
    request: AnimationRequest
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None


class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, JobInfo] = {}
        self.queue = Queue()
        self.lock = threading.Lock()
        self._running = False
        self._worker_thread = None
        
        # Multiprocessing variables
        self.mp_manager = None
        self.mp_job_queue = None
        self.mp_result_queue = None
        
        logger.info("Job queue initialized")
    
    def start_mp_manager(self):
        """Start the multiprocessing manager for inter-process communication."""
        if self.mp_manager is not None:
            return
            
        self.mp_manager = Manager()
        self.mp_job_queue = self.mp_manager.Queue()
        self.mp_result_queue = self.mp_manager.Queue()
        logger.info("Multiprocessing manager started")
    
    def stop_mp_manager(self):
        """Stop the multiprocessing manager."""
        if self.mp_manager is not None:
            # Clear any remaining items in the queues
            while not self.mp_job_queue.empty():
                try:
                    self.mp_job_queue.get_nowait()
                except:
                    pass
                    
            while not self.mp_result_queue.empty():
                try:
                    self.mp_result_queue.get_nowait()
                except:
                    pass
            
            self.mp_job_queue = None
            self.mp_result_queue = None
            self.mp_manager.shutdown()
            self.mp_manager = None
            logger.info("Multiprocessing manager stopped")
    
    def process_results(self):
        """Process any results from the multiprocessing queue."""
        if not self.mp_result_queue:
            return
            
        while not self.mp_result_queue.empty():
            try:
                result = self.mp_result_queue.get_nowait()
                if result and len(result) >= 3:
                    job_id, status, progress, message, error = result if len(result) >= 5 else (*result, None, None)
                    self.update_job_status(job_id, status, error)
                    logger.info(f"Updated job {job_id} status to {status} from MP queue")
            except Exception as e:
                logger.exception(f"Error processing MP result: {str(e)}")
                break
    
    def start(self):
        """Start the job queue worker thread."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self._worker_thread.start()
        logger.info("Job queue worker thread started")
    
    def stop(self):
        """Stop the job queue worker thread."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        logger.info("Job queue worker thread stopped")
    
    def add_job(self, request: AnimationRequest) -> JobInfo:
        """Add a new job to the queue."""
        job_id = str(uuid.uuid4())
        
        with self.lock:
            job_info = JobInfo(job_id=job_id, request=request)
            self.jobs[job_id] = job_info
            self.queue.put(job_id)
            
            # Also add to multiprocessing queue if available
            if self.mp_job_queue is not None:
                self.mp_job_queue.put((job_id, request))
                
            logger.info(f"Added job {job_id} to queue")
        
        return job_info
    
    def get_job_status(self, job_id: str) -> Optional[JobInfo]:
        """Get the status of a job by ID."""
        with self.lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[JobInfo]:
        """Get all jobs in the system."""
        with self.lock:
            return list(self.jobs.values())
    
    def update_job_status(self, job_id: str, status: JobStatus, error: Optional[str] = None) -> bool:
        """Update the status of a job."""
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            job_info = self.jobs[job_id]
            job_info.status = status
            
            if status == JobStatus.PROCESSING and not job_info.started_at:
                job_info.started_at = time.time()
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                job_info.completed_at = time.time()
            
            if error:
                job_info.error = error
            
            logger.info(f"Updated job {job_id} status to {status}")
            return True
    
    def _process_jobs(self):
        """Worker thread function to process jobs."""
        from app.services.renderer import renderer
        
        while self._running:
            try:
                # Get the next job from the queue with a timeout
                try:
                    job_id = self.queue.get(timeout=1)
                except:
                    continue
                
                with self.lock:
                    if job_id not in self.jobs:
                        logger.warning(f"Job {job_id} not found in job list")
                        self.queue.task_done()
                        continue
                    
                    job_info = self.jobs[job_id]
                
                # Update job status to processing
                self.update_job_status(job_id, JobStatus.PROCESSING)
                
                try:
                    # Send job to renderer
                    renderer.render_animation(job_id, job_info.request)
                    self.update_job_status(job_id, JobStatus.COMPLETED)
                except Exception as e:
                    logger.exception(f"Error processing job {job_id}: {str(e)}")
                    self.update_job_status(job_id, JobStatus.FAILED, str(e))
                finally:
                    self.queue.task_done()
            
            except Exception as e:
                logger.exception(f"Unexpected error in job queue worker: {str(e)}")


# Create singleton instance
job_queue = JobQueue()