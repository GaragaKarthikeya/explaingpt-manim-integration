import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from app.config import settings
import threading
import time
import logging

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.output_dir = Path(settings.MANIM_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.cleaner_thread = threading.Thread(target=self._cleanup_old_files, daemon=True)
        self.running = False
    
    def start(self):
        """Start the storage service."""
        self.running = True
        self.cleaner_thread.start()
        logger.info("Storage service started")
    
    def stop(self):
        """Stop the storage service."""
        self.running = False
        if self.cleaner_thread.is_alive():
            self.cleaner_thread.join(timeout=5.0)
        logger.info("Storage service stopped")
    
    def get_video_path(self, job_id: str) -> Path:
        """Get the path to a video file."""
        return self.output_dir / f"{job_id}.mp4"
    
    def video_exists(self, job_id: str) -> bool:
        """Check if a video file exists."""
        video_path = self.get_video_path(job_id)
        return video_path.exists()
    
    def get_video_url(self, job_id: str) -> str:
        """Get the URL for a video file."""
        return f"{settings.NGROK_BASE_URL}/videos/{job_id}.mp4"
    
    def _cleanup_old_files(self):
        """Periodically clean up old video files."""
        while self.running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=settings.CACHE_EXPIRY_HOURS)
                
                for file_path in self.output_dir.glob("*.mp4"):
                    file_modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_modified_time < cutoff_time:
                        logger.info(f"Removing old video: {file_path}")
                        file_path.unlink()
            
            except Exception as e:
                logger.exception(f"Error during file cleanup: {str(e)}")
            
            # Sleep for 1 hour before next cleanup
            for _ in range(60):  # Check every minute if we should stop
                if not self.running:
                    break
                time.sleep(60)

# Create a singleton instance
storage_service = StorageService()