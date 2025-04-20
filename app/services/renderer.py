import os
import threading
import subprocess
import tempfile
import time
import json
import logging
import asyncio
import shutil
import multiprocessing
import psutil
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import google.generativeai as genai
from google.generativeai import types
from typing import Dict, Optional, List
import re

from app.models import JobStatus, AnimationRequest
from app.services.job_queue import job_queue
from app.services.llm_service import gemini_service, GeminiService
from app.services.error_recovery import ErrorRecovery, CodeAnalyzer  # Import the new error recovery framework
from app.config import settings

logger = logging.getLogger(__name__)

# Create a global instance of the error recovery framework
error_recovery = ErrorRecovery()

# Helper function to trim blank screens at the end of videos
def _trim_blank_ending(video_path, temp_dir):
    """
    More sophisticated video trimming function that uses FFmpeg to detect and trim any excess content,
    ensuring the animation stops when the actual animation ends, not when the soundtrack ends.
    
    Args:
        video_path: Path to the video file
        temp_dir: Directory for temporary files
        
    Returns:
        Path to the trimmed video or None if trimming failed
    """
    try:
        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=temp_dir) as tmp_video:
            output_path = tmp_video.name
            
        # Get the video duration
        duration_cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", 
            str(video_path)
        ]
        
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        
        if duration_result.returncode != 0:
            logger.error(f"Failed to get video duration: {duration_result.stderr}")
            return None
            
        try:
            video_duration = float(duration_result.stdout.strip())
        except (ValueError, TypeError):
            logger.error("Could not parse video duration")
            return None
        
        # If video is too short (less than 10 seconds), don't trim
        if video_duration < 10:
            logger.info("Video too short, skipping trimming")
            return None
            
        # Get scene changes to determine when the animation activity stops
        # Use FFmpeg's scene detection filter to find the last significant scene change
        scene_cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-filter_complex", "select=gt(scene\\,0.1),showinfo",  # Detect scene changes with 10% threshold
            "-f", "null",
            "-"
        ]
        
        scene_result = subprocess.run(scene_cmd, capture_output=True, text=True)
        
        # Extract timestamps from the scene detection output
        scene_output = scene_result.stderr
        scene_timestamps = []
        for line in scene_output.split('\n'):
            if "pts_time:" in line:
                try:
                    # Extract timestamp
                    ts_match = re.search(r"pts_time:(\d+\.\d+)", line)
                    if ts_match:
                        scene_timestamps.append(float(ts_match.group(1)))
                except:
                    pass
        
        end_time = video_duration
        if scene_timestamps:
            # Find the last meaningful scene change
            # Ignore scene changes in the last 3 seconds
            filtered_scenes = [ts for ts in scene_timestamps if ts < (video_duration - 3)]
            if filtered_scenes:
                # Get the last scene change + 3 seconds (for proper completion)
                last_scene = max(filtered_scenes)
                # Only use it if it's not too close to the end already
                if video_duration - last_scene > 5:
                    end_time = min(last_scene + 5, video_duration - 0.5)
                    logger.info(f"Detected last active scene at {last_scene:.2f}s, trimming to {end_time:.2f}s")
        
        # If we're not changing much, just return the original
        if video_duration - end_time < 2:
            logger.info("No significant excess content detected, using original video")
            return None
            
        # Simply trim the end of the video using segment copy (no re-encoding)
        trim_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-ss", "0",  # Start from the beginning
            "-to", f"{end_time}",  # End at the calculated time
            "-c", "copy",  # Use stream copy (no re-encoding)
            output_path
        ]
        
        logger.info(f"Trimming video from {video_duration:.2f}s to {end_time:.2f}s")
        
        trim_result = subprocess.run(trim_cmd, capture_output=True)
        
        if trim_result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
            logger.info(f"Successfully trimmed {video_duration - end_time:.2f} seconds from end of video")
            return output_path
        else:
            error_msg = trim_result.stderr.decode() if trim_result.stderr else "Unknown error"
            logger.error(f"Error in trimming: {error_msg}")
            # Clean up failed output
            try:
                os.unlink(output_path)
            except:
                pass
            return None
    except Exception as e:
        logger.error(f"Error in _trim_blank_ending: {str(e)}")
        return None


class MusicManager:
    """Manages background music tracks and creates dynamic audio with transitions."""
    
    def __init__(self, music_dir: str = "/app/assets/bgmusic"):
        self.music_dir = Path(music_dir)
        self.supported_formats = [".flac", ".mp3", ".wav", ".ogg"]
        
        # Find all music tracks
        self.tracks = []
        for ext in self.supported_formats:
            self.tracks.extend(list(self.music_dir.glob(f"*{ext}")))
        
        if not self.tracks:
            logger.warning(f"No music tracks found in {music_dir}")
        else:
            logger.info(f"Found {len(self.tracks)} music tracks in {music_dir}")
    
    def get_random_tracks(self, count: int = 2, exclude_tracks: List[str] = None) -> List[str]:
        """
        Get a random selection of tracks, optionally excluding specific tracks.
        
        Args:
            count: Number of tracks to select
            exclude_tracks: Tracks to exclude from selection
            
        Returns:
            List of track paths as strings
        """
        if not self.tracks:
            return []
        
        # Filter out excluded tracks if provided
        available_tracks = self.tracks
        if exclude_tracks:
            available_tracks = [t for t in self.tracks if str(t) not in exclude_tracks]
        
        if not available_tracks:
            available_tracks = self.tracks  # Fallback if all were excluded
            
        # Ensure we don't request more tracks than available
        count = min(count, len(available_tracks))
        
        # Select random tracks without replacement
        selected_tracks = random.sample(available_tracks, count)
        return [str(track) for track in selected_tracks]
    
    def create_soundtrack_for_animation(self, job_id: str, prompt: str, expected_duration: int) -> str:
        """
        Create a soundtrack specifically for an animation with proper timing.
        
        Args:
            job_id: The unique job ID
            prompt: The animation prompt text
            expected_duration: Expected animation duration in seconds
            
        Returns:
            Path to the created soundtrack file
        """
        if not self.tracks:
            return None
            
        try:
            # Create a temp file for the output
            with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as tmp_audio:
                output_path = tmp_audio.name
            
            # Seed randomization with job_id and prompt
            random.seed(hash(job_id + prompt))
            
            # Select 2-3 tracks for variety
            track_count = random.randint(2, 3)
            selected_tracks = self.get_random_tracks(track_count)
            
            if not selected_tracks:
                return None
                
            logger.info(f"Creating soundtrack with {len(selected_tracks)} tracks for job {job_id}")
                
            # Calculate segment lengths - ensure the soundtrack is slightly SHORTER than animation
            # This ensures animation completes before music ends
            target_duration = max(min(expected_duration - 5, expected_duration * 0.9), expected_duration - 10)
            segment_duration = target_duration / len(selected_tracks)
            
            # Use a much simpler filter approach that's less likely to have syntax errors
            if len(selected_tracks) == 1:
                # For a single track, just trim it and adjust volume
                cmd = [
                    "ffmpeg", "-y",
                    "-i", selected_tracks[0],
                    "-t", str(target_duration),
                    "-af", f"volume=0.25,afade=t=out:st={target_duration-3}:d=3",
                    output_path
                ]
            else:
                # For multiple tracks, concatenate them with basic volume adjustment
                # First create intermediate files for each track
                temp_files = []
                try:
                    for i, track in enumerate(selected_tracks):
                        # Create temp file for each processed track
                        with tempfile.NamedTemporaryFile(suffix=f'.{i}.flac', delete=False) as tmp:
                            temp_file = tmp.name
                            temp_files.append(temp_file)
                            
                        # Calculate duration for this segment
                        if i < len(selected_tracks) - 1:
                            # Add some overlap for crossfade
                            this_duration = segment_duration + 3
                        else:
                            this_duration = segment_duration
                            
                        # Process each track: trim and adjust volume
                        trim_cmd = [
                            "ffmpeg", "-y",
                            "-i", track,
                            "-t", str(this_duration),
                            "-af", "volume=0.25",
                            temp_file
                        ]
                        
                        # Run the command to create processed track
                        subprocess.run(trim_cmd, capture_output=True, check=True)
                    
                    # Create a concat file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as concat_file:
                        concat_file_path = concat_file.name
                        for temp_file in temp_files:
                            concat_file.write(f"file '{temp_file}'\n")
                    
                    # Concatenate all processed files
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", concat_file_path,
                        "-t", str(target_duration),  # Ensure we don't exceed target duration
                        "-af", f"afade=t=out:st={target_duration-3}:d=3",  # Add fade out at the end
                        output_path
                    ]
                    
                    # Run the command
                    subprocess.run(cmd, capture_output=True, check=True)
                    
                    # Clean up temp files
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                    try:
                        os.unlink(concat_file_path)
                    except:
                        pass
                        
                    return output_path
                        
                except Exception as e:
                    # Clean up temp files on error
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                    logger.error(f"Error in multi-track processing: {str(e)}")
                    
                    # Fall back to just using the first track
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", selected_tracks[0],
                        "-t", str(target_duration),
                        "-af", f"volume=0.25,afade=t=out:st={target_duration-3}:d=3",
                        output_path
                    ]
            
            # Execute FFmpeg (for single track or fallback case)
            logger.info(f"Running FFmpeg to create soundtrack of {target_duration} seconds")
            process = subprocess.run(cmd, capture_output=True)
            
            if process.returncode == 0 and os.path.exists(output_path):
                logger.info(f"Created soundtrack for job {job_id}: {output_path}")
                return output_path
            else:
                error = process.stderr.decode() if process.stderr else "Unknown error"
                logger.error(f"Failed to create soundtrack: {error}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating soundtrack: {str(e)}")
            return None

# Create a singleton instance of the music manager
music_manager = MusicManager()

class ResourceMonitor:
    """Monitors system resource usage and recommends optimal worker count."""
    
    def __init__(self):
        self.cpu_history = []  # Store recent CPU usage percentages
        self.memory_history = []  # Store recent memory usage percentages
        self.history_size = 5  # Number of samples to keep
    
    def get_system_resources(self):
        """Get current system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory_percent = psutil.virtual_memory().percent
        load_avg = os.getloadavg()[0] / os.cpu_count()  # Normalize load average
        
        # Update history
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        if len(self.cpu_history) > self.history_size:
            self.cpu_history.pop(0)
        if len(self.memory_history) > self.history_size:
            self.memory_history.pop(0)
            
        # Calculate averages
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'load_avg': load_avg,
            'avg_cpu': avg_cpu,
            'avg_memory': avg_memory,
            'total_memory': psutil.virtual_memory().total,
            'available_memory': psutil.virtual_memory().available
        }
    
    def calculate_optimal_workers(self, current_workers):
        """Calculate the optimal number of worker processes based on system resources."""
        resources = self.get_system_resources()
        
        # Calculate workers based on memory
        available_memory_mb = resources['available_memory'] / (1024 * 1024)
        memory_based_workers = int(available_memory_mb / settings.MIN_MEMORY_PER_WORKER_MB)
        
        # Calculate workers based on CPU load
        max_cpu_workers = max(1, multiprocessing.cpu_count() - 1)
        if resources['avg_cpu'] > 80:  # High CPU usage
            cpu_based_workers = max(1, current_workers - 1)
        elif resources['avg_cpu'] < 50:  # Low CPU usage
            cpu_based_workers = min(max_cpu_workers, current_workers + 1)
        else:
            cpu_based_workers = current_workers
            
        # Take the minimum to ensure we don't overload either CPU or memory
        optimal_workers = min(memory_based_workers, cpu_based_workers, settings.MAX_PARALLEL_RENDERINGS)
        
        # Always ensure at least 1 worker
        return max(1, optimal_workers)
    
    def log_resource_usage(self):
        """Log current resource usage."""
        resources = self.get_system_resources()
        logger.info(
            f"System resources: CPU: {resources['cpu_percent']}%, "
            f"Memory: {resources['memory_percent']}%, "
            f"Load: {resources['load_avg']:.2f}"
        )


# Helper function to run async code in a new event loop
def run_async_in_new_loop(coroutine):
    """Run an async coroutine in a new event loop and return the result."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


# Enhanced worker process function with error recovery, overlap detection, and dynamic music transitions
def worker_process(job_id, request_dict, output_dir, temp_dir, result_queue):
    """Worker process to render an animation."""
    try:
        # Setup logging for the worker process
        logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=logging_format)
        worker_logger = logging.getLogger(f"worker-{job_id}")
        
        # Start monitoring process resources
        current_process = psutil.Process()
        initial_memory = current_process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.time()
        
        # Update status via result queue
        result_queue.put((job_id, JobStatus.PROCESSING, 0.1, "Preparing animation", None))
        
        # Parse the request dictionary
        request = AnimationRequest(**request_dict)
        prompt = request.prompt
        complexity = request.complexity
        worker_logger.info(f"Processing prompt: '{prompt}' with complexity level {complexity}")
        
        # Create GeminiService instance for code generation
        # Create a fresh instance for each worker to avoid shared event loop issues
        gemini = GeminiService()
        
        try:
            # Estimate animation duration based on complexity
            estimated_duration = 180 if complexity == "low" else (300 if complexity == "medium" else 480)  # in seconds
            
            # Generate Manim code using synchronous API with explicit duration control
            duration_prompt = f"""Generate a Manim animation explaining: '{prompt}'
            
IMPORTANT: The animation MUST have these properties:
1. Animation duration of EXACTLY {estimated_duration} seconds
2. Must end cleanly with a clear ending (summary or conclusion slide)
3. Include a finale/summary that visually indicates the end of the animation
4. NO BLANK SCREENS at any point - always have visual content
5. Add a final "Animation Complete" text at the end that stays visible

Structure your code like this:
```python
from manim import *

class AnimationScene(Scene):
    def construct(self):
        # All your animation code here
        
        # At the end, add a conclusion/summary with a clear "Animation Complete" message
        conclusion = Text("Animation Complete", color=BLUE).scale(1.5)
        self.play(FadeIn(conclusion))
        self.wait(2)  # End with a reasonable pause on the final screen
```"""

            # Generate Manim code with explicit duration control
            manim_code = gemini.generate_manim_code_sync(duration_prompt, complexity)
            
            if not manim_code:
                raise RuntimeError("Failed to generate valid Manim code")
            
            # Write the code to a file for rendering
            max_rendering_attempts = 1
            if settings.ERROR_RECOVERY_ENABLED:
                max_rendering_attempts = settings.ERROR_RECOVERY_MAX_RETRIES + 1
            
            # Create a dynamic soundtrack with multiple music tracks
            bg_music_path = None
            try:
                # Use our music manager to create a dynamic soundtrack for this animation
                # Make sure the soundtrack is shorter than the animation
                bg_music_path = music_manager.create_soundtrack_for_animation(
                    job_id=job_id,
                    prompt=prompt,
                    expected_duration=estimated_duration - 10  # Make soundtrack MUCH shorter than animation
                )
                
                if bg_music_path:
                    worker_logger.info(f"Created dynamic soundtrack for animation: {os.path.basename(bg_music_path)}")
                else:
                    worker_logger.warning("Failed to create dynamic soundtrack")
            except Exception as e:
                worker_logger.warning(f"Error creating soundtrack: {str(e)}")
            
            for attempt in range(max_rendering_attempts):
                # Create a modified version of the code that adds background music
                modified_manim_code = manim_code
                
                # Inject soundtrack if available
                if bg_music_path and os.path.exists(bg_music_path):
                    worker_logger.info(f"Injecting background music into Manim code: {os.path.basename(bg_music_path)}")
                    
                    # Simple approach to add the music to Scene initialization
                    # Add the music import and handling as a simple setup override
                    music_path = bg_music_path.replace('\\', '/')
                    setup_code = """    def setup(self):
        super().setup()
        # Add background music
        self.add_sound("%s")
""" % (music_path)
                    
                    # Insert after class definition
                    class_pattern = r'(class\s+AnimationScene\s*\(\s*Scene\s*\)\s*:)'
                    modified_manim_code = re.sub(class_pattern, f'\\1\n{setup_code}', modified_manim_code)
                
                # Write the modified code to a file
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False, dir=temp_dir) as tmp:
                    tmp_path = tmp.name
                    worker_logger.info(f"Writing Manim code to temporary file for job {job_id} (attempt {attempt+1}/{max_rendering_attempts})")
                    tmp.write(modified_manim_code.encode('utf-8'))
                
                # Update status
                progress = 0.5 + (0.1 * attempt)
                result_queue.put((job_id, JobStatus.RENDERING, progress, 
                                f"Rendering animation with Manim (attempt {attempt+1})", None))
                
                # Create a temporary directory for Manim output
                with tempfile.TemporaryDirectory(dir=temp_dir) as temp_media_dir:
                    # Run Manim to generate the animation
                    output_filename = f"{job_id}.mp4"
                    
                    # Updated Manim command
                    cmd = [
                        "manim", 
                        tmp_path, 
                        "AnimationScene", 
                        "-q", "h", 
                        "-o", output_filename,
                        "--media_dir", temp_media_dir
                    ]
                    
                    worker_logger.info(f"Running Manim command: {' '.join(cmd)}")
                    
                    manim_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    stdout, stderr = manim_process.communicate()
                    stdout_text = stdout.decode('utf-8')
                    stderr_text = stderr.decode('utf-8')
                    
                    if manim_process.returncode != 0:
                        worker_logger.error(f"Manim rendering failed (attempt {attempt+1}): {stderr_text}")
                        
                        # Only try error recovery if we have attempts left and it's enabled
                        if (attempt < max_rendering_attempts - 1) and settings.ERROR_RECOVERY_ENABLED:
                            worker_logger.info(f"Attempting error recovery for job {job_id}")
                            result_queue.put((job_id, JobStatus.PROCESSING, 0.6 + (0.1 * attempt), 
                                           "Attempting to fix rendering errors", None))
                            
                            try:
                                # Generate a recovery prompt with error feedback
                                recovery_prompt = (
                                    f"You are an expert in Manim animation generation. "
                                    f"The following Manim code has encountered an error during rendering:\n\n"
                                    f"```python\n{manim_code}\n```\n\n"
                                    f"Error output:\n{stderr_text}\n\n"
                                    f"Please fix the code to render properly. The animation should explain: '{prompt}'\n\n"
                                    f"Common fixes to consider:\n"
                                    f"1. Ensure all parentheses are properly matched\n"
                                    f"2. Fix any syntax errors in the code\n"
                                    f"3. Make sure all variables are defined before use\n"
                                    f"4. Ensure proper LaTeX formatting with double backslashes\n"
                                    f"5. Add buffer to prevent objects from overlapping\n"
                                    f"6. Use appropriate sizing for objects to fit within the frame\n\n"
                                    f"Return ONLY the complete, fixed Manim code."
                                )
                                
                                # Generate fixed code
                                manim_code = gemini.generate_manim_code_sync(recovery_prompt, complexity)
                                worker_logger.info("Generated new code based on error feedback")
                                
                                # Clean up the temporary file
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                                    
                                continue  # Try again with the fixed code
                                
                            except Exception as recovery_error:
                                worker_logger.error(f"Error during recovery attempt: {str(recovery_error)}")
                        
                        # If we're out of attempts or recovery is disabled, report failure
                        if attempt == max_rendering_attempts - 1:
                            error_msg = f"Manim rendering failed after {max_rendering_attempts} attempts: {stderr_text}"
                            result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))
                            
                            # Clean up temp files
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                            return
                    else:
                        # Success! We have a rendered animation
                        # Find the generated file
                        expected_path = Path(temp_media_dir) / "videos" / "720p30" / output_filename
                        if not expected_path.exists():
                            # Look for file in different locations
                            found_files = list(Path(temp_media_dir).glob(f"**/{output_filename}"))
                            if not found_files:
                                error_msg = f"Output video file not found in {temp_media_dir}"
                                result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))
                                # Clean up temp file
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                                return
                            expected_path = found_files[0]
                        
                        # Use FFmpeg to trim the video to ensure it doesn't have blank ending frames
                        worker_logger.info(f"Ensuring animation ends properly using FFmpeg...")
                        
                        # Trim based on actual video length
                        processed = _trim_blank_ending(expected_path, temp_dir)
                        if processed:
                            expected_path = processed
                        # Copy to final destination
                        final_path = Path(output_dir) / output_filename
                        shutil.copy2(expected_path, final_path)
                        
                        if not final_path.exists():
                            error_msg = f"Failed to copy output file to {final_path}"
                            result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))
                            # Clean up temp file
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                            return
                        
                        # Update status to completed
                        completion_msg = f"Animation completed successfully after {attempt+1} attempts: {output_filename}"
                        if bg_music_path:
                            completion_msg += f" (with background music: {os.path.basename(bg_music_path)})"
                        result_queue.put((job_id, JobStatus.COMPLETED, 1.0, completion_msg, None))
                        
                        # Clean up temp files
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                            
                        # Log resource usage statistics
                        end_time = time.time()
                        final_memory = current_process.memory_info().rss / (1024 * 1024)  # MB
                        total_memory_used = final_memory - initial_memory
                        execution_time = end_time - start_time
                        worker_logger.info(
                            f"Job {job_id} completed successfully after {attempt+1} attempts. "
                            f"Time: {execution_time:.2f}s, "
                            f"Memory: {total_memory_used:.2f}MB"
                        )
                        return  # Success, exit the function
            
        except Exception as e:
            worker_logger.error(f"Error generating code with Gemini: {str(e)}")
            error_msg = f"Error rendering animation for job {job_id}: {str(e)}"
            result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))
    except Exception as e:
        error_msg = f"Error rendering animation for job {job_id}: {str(e)}"
        result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))


class ManimRenderer:
    def __init__(self):
        self.output_dir = Path(settings.MANIM_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a temporary directory for worker process files
        self.temp_dir = tempfile.mkdtemp(prefix="manim_workers_")
        
        # Set up multiprocessing with optimal worker count
        self.max_workers = min(multiprocessing.cpu_count(), settings.MAX_PARALLEL_RENDERINGS)
        self.current_workers = self.max_workers
        self.process_pool = None
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor()
        
        # Set up a result processing thread
        self.result_thread = threading.Thread(target=self._process_results, daemon=True)
        self.running = False
        
        # Set up a job submission thread
        self.submit_thread = threading.Thread(target=self._submit_jobs, daemon=True)
        
        # Set up a resource monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        
        # Job statistics for performance analysis
        self.job_stats: Dict[str, dict] = {}
    
    def render_animation(self, job_id, request: AnimationRequest):
        """Add a job to the multiprocessing queue for rendering.
        
        This method is called by the JobQueue to submit jobs for rendering.
        The actual rendering happens in worker processes.
        """
        if job_queue.mp_job_queue:
            # Record initial job stats for performance tracking
            self.job_stats[job_id] = {
                'start_time': time.time(),
                'complexity': request.complexity,
                'queue_time': time.time()
            }
            
            # Add job to queue with complexity level for better scheduling
            job_queue.mp_job_queue.put((job_id, request))
            logger.info(f"Job {job_id} added to multiprocessing queue for rendering")
        else:
            raise RuntimeError("Multiprocessing queue not initialized. Call start() first.")
    
    def start(self):
        """Start the renderer worker processes and threads."""
        # Initialize the job queue's multiprocessing manager
        job_queue.start_mp_manager()
        
        # Calculate initial worker count based on available system resources
        resources = self.resource_monitor.get_system_resources()
        available_memory_mb = resources['available_memory'] / (1024 * 1024)
        memory_based_workers = int(available_memory_mb / settings.MIN_MEMORY_PER_WORKER_MB)
        self.current_workers = min(self.max_workers, memory_based_workers)
        self.current_workers = max(1, self.current_workers)  # Ensure at least 1 worker
        
        # Start the process pool with calculated worker count
        self.process_pool = ProcessPoolExecutor(max_workers=self.current_workers)
        
        # Start the threads
        self.running = True
        self.result_thread.start()
        self.submit_thread.start()
        self.monitor_thread.start()
        
        logger.info(f"Manim renderer started with {self.current_workers} worker processes "
                   f"(max: {self.max_workers}, memory per worker: {settings.MIN_MEMORY_PER_WORKER_MB}MB)")
    
    def stop(self):
        """Stop the renderer worker processes and threads."""
        self.running = False
        
        # Shutdown the process pool
        if self.process_pool:
            self.process_pool.shutdown(wait=False)
        
        # Wait for threads to finish
        if self.result_thread.is_alive():
            self.result_thread.join(timeout=5.0)
        
        if self.submit_thread.is_alive():
            self.submit_thread.join(timeout=5.0)
            
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
            
        # Stop the job queue's multiprocessing manager
        job_queue.stop_mp_manager()
        
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
        logger.info("Manim renderer stopped")
    
    def _adjust_worker_count(self, new_count):
        """Adjust the number of worker processes."""
        if new_count == self.current_workers:
            return
            
        logger.info(f"Adjusting worker count from {self.current_workers} to {new_count}")
        
        # Shutdown existing pool
        if self.process_pool:
            self.process_pool.shutdown(wait=False)
            
        # Create new pool with adjusted worker count
        self.current_workers = new_count
        self.process_pool = ProcessPoolExecutor(max_workers=self.current_workers)
        
        logger.info(f"Worker count adjusted to {self.current_workers}")
    
    def _monitor_resources(self):
        """Monitor system resources and adjust worker count if needed."""
        last_check = time.time()
        
        while self.running:
            try:
                # Check system resources periodically
                current_time = time.time()
                if current_time - last_check >= settings.MONITORING_INTERVAL_SEC:
                    last_check = current_time
                    
                    # Log current resource usage
                    self.resource_monitor.log_resource_usage()
                    
                    # Only adjust worker count if dynamic scaling is enabled
                    if settings.ENABLE_DYNAMIC_SCALING:
                        # Calculate optimal worker count
                        optimal_workers = self.resource_monitor.calculate_optimal_workers(self.current_workers)
                        
                        # Adjust worker count if necessary
                        if optimal_workers != self.current_workers:
                            self._adjust_worker_count(optimal_workers)
                
                # Sleep for a short time to reduce CPU usage of monitoring thread
                time.sleep(5)
                
            except Exception as e:
                logger.exception(f"Error in resource monitoring: {str(e)}")
                time.sleep(10)
    
    def _submit_jobs(self):
        """Submit jobs to the process pool from the queue."""
        while self.running:
            try:
                if job_queue.mp_job_queue and not job_queue.mp_job_queue.empty():
                    job_id, request = job_queue.mp_job_queue.get()
                    
                    # Update job stats with dequeue time
                    if job_id in self.job_stats:
                        self.job_stats[job_id]['dequeue_time'] = time.time()
                        queue_duration = self.job_stats[job_id]['dequeue_time'] - self.job_stats[job_id]['queue_time']
                        logger.info(f"Job {job_id} waited in queue for {queue_duration:.2f} seconds")
                    
                    # Check if we have enough memory before submitting
                    resources = self.resource_monitor.get_system_resources()
                    available_memory_mb = resources['available_memory'] / (1024 * 1024)
                    
                    if available_memory_mb < settings.MIN_MEMORY_PER_WORKER_MB:
                        logger.warning(
                            f"Low memory: {available_memory_mb:.2f}MB available, "
                            f"need {settings.MIN_MEMORY_PER_WORKER_MB}MB. "
                            f"Delaying job {job_id} submission."
                        )
                        # Put the job back in the queue and wait
                        job_queue.mp_job_queue.put((job_id, request))
                        time.sleep(10)  # Wait before trying again
                        continue
                    
                    # Convert request to dictionary for serialization
                    request_dict = request.dict()
                    
                    # Submit job to process pool
                    self.process_pool.submit(
                        worker_process,
                        job_id,
                        request_dict,
                        str(self.output_dir),
                        self.temp_dir,
                        job_queue.mp_result_queue
                    )
                    logger.info(f"Submitted job {job_id} to process pool")
                else:
                    time.sleep(1)
            except Exception as e:
                logger.exception(f"Error submitting job to process pool: {str(e)}")
                time.sleep(1)
    
    def _process_results(self):
        """Process results from worker processes."""
        while self.running:
            try:
                # Process any results from the multiprocessing jobs
                job_queue.process_results()
                
                # Check for completed jobs and update statistics
                with job_queue.lock:
                    for job_id, job_info in job_queue.jobs.items():
                        if (job_info.status in [JobStatus.COMPLETED, JobStatus.FAILED] and 
                            job_id in self.job_stats and
                            'end_time' not in self.job_stats[job_id]):
                            
                            # Record job completion statistics
                            self.job_stats[job_id]['end_time'] = time.time()
                            total_duration = self.job_stats[job_id]['end_time'] - self.job_stats[job_id]['start_time']
                            
                            logger.info(
                                f"Job {job_id} completed with status {job_info.status}. "
                                f"Total time: {total_duration:.2f} seconds. "
                                f"Complexity: {self.job_stats[job_id]['complexity']}"
                            )
                            
                            # Clean up old stats to prevent memory leak (keep only the last 100)
                            if len(self.job_stats) > 100:
                                oldest_job = min(self.job_stats, key=lambda k: self.job_stats[k].get('end_time', float('inf')))
                                self.job_stats.pop(oldest_job, None)
                
                time.sleep(0.5)
            except Exception as e:
                logger.exception(f"Error processing results: {str(e)}")
                time.sleep(1)

    # ------------------- Enhanced Asynchronous Code Generation Methods ------------------- #
    async def _assess_topic_complexity(self, topic: str) -> tuple[str, int, float]:
        """Use Gemini to dynamically evaluate topic complexity, duration, and resource needs."""
        PROMPT = (
            "Rate the conceptual complexity of this topic for a computer science animation, "
            f"suggest video duration (in seconds), and estimate resource intensity: '{topic}'\n\n"
            "Respond ONLY in this JSON format:\n"
            '{"complexity": "low/medium/high", "duration_seconds": int, "resource_intensity": float(0.1-1.0)}\n'
            "Guidelines:\n"
            "- 'low' (2-4 min): Simple topics, 2-3 elements on screen\n"
            "- 'medium' (5-8 min): Moderate topics, 3-5 elements on screen\n"
            "- 'high' (9-15 min): Complex topics, handle carefully with staged elements\n"
            "- Consider spreading complex animations across multiple scenes\n"
            "- 'resource_intensity' affects element density and spacing"
        )

        response = await gemini_service.generate_text(PROMPT)
        try:
            assessment = json.loads(response.strip())
            return (
                assessment["complexity"], 
                assessment["duration_seconds"], 
                assessment["resource_intensity"]
            )
        except Exception as e:
            logger.error(f"Error parsing complexity assessment: {e}")
            return "medium", 300, 0.5  # Fallback default

    def _build_manim_prompt(self, user_prompt: str, duration_sec: int, resource_intensity: float) -> str:
        """Generate a prompt that enforces duration-aware and resource-aware animations."""
        spacing_guide = (
            "3.0 units (very sparse)" if resource_intensity < 0.3 else
            "2.5 units (balanced spacing)" if resource_intensity < 0.7 else
            "2.0 units (careful dense layout)"
        )
        
        max_elements = (
            "3-4 elements" if resource_intensity < 0.3 else
            "4-6 elements" if resource_intensity < 0.7 else
            "6-8 elements with careful staging"
        )
        
        return (
            f"Generate a Manim animation explaining: '{user_prompt}'\n"
            f"- *Video length*: Exactly {duration_sec // 60} minutes ({duration_sec} sec).\n"
            "- *CRITICAL SPACING REQUIREMENTS*:\n"
            f"  1. Maintain MINIMUM {spacing_guide} between ALL elements\n"
            f"  2. Maximum of {max_elements} on screen at once\n"
            "  3. Use setup() method with self.object_positions tracking\n"
            "  4. ALWAYS call self.clear_overlaps() after adding elements\n"
            "  5. Use self.safe_add() instead of self.add() for all objects\n"
            "  6. Use self.position_element() instead of next_to()\n"
            "  7. Use self.arrange_elements() instead of arrange()\n"
            "  8. Set explicit z_index for EVERY element\n"
            "  9. Scale all objects to 70% of default size\n"
            "  10. Keep ALL content within 60% of frame width/height\n"
            "- *Advanced Layout Requirements*:\n"
            "  1. For complex scenes, use self.create_grid_layout()\n"
            "  2. Remove old elements before adding new ones\n"
            "  3. Use VGroup for ALL related elements\n"
            "  4. Apply move_to(ORIGIN) for centered layouts\n"
            "- *Element Sizing*:\n"
            "  1. Text: font_size=24 (regular), 32 (titles)\n"
            "  2. Shapes: radius=0.5, side_length=1.0\n"
            "  3. Arrows: buff=0.3, stroke_width=4\n"
            "  4. Apply scale(0.7) to all VGroups\n"
            "- *Timing*:\n"
            "  1. wait(0.3) between animations\n"
            "  2. Create/Write with run_time=1.0\n"
            "  3. Transforms with run_time=1.5\n"
            "- *Output*: Python code only, no explanations."
        )

    async def _generate_manim_code_async(self, request: AnimationRequest) -> str:
        """Dynamic duration-adjusted and resource-aware code generation."""
        # Step 1: Let Gemini judge topic complexity and suggest duration and resource needs
        complexity, duration_sec, resource_intensity = await self._assess_topic_complexity(request.prompt)
        
        # Step 2: Build a prompt with duration and resource constraints
        modified_prompt = self._build_manim_prompt(request.prompt, duration_sec, resource_intensity)
        
        # Step 3: Generate Manim code using Gemini with the updated prompt and complexity value
        raw_code = await gemini_service.generate_manim_code(modified_prompt, complexity)
        
        # Step 4: Apply automatic layout management to prevent overlapping elements
        layout_manager = LayoutManager()
        layout_managed_code = layout_manager.apply_layout_management(raw_code)
        
        # Step 5: Apply blank screen and position fixes
        issues = CodeAnalyzer.analyze(layout_managed_code)
        needs_fixes = any(issue for issue in issues if any(term in issue.lower() 
                                  for term in ["blank screen", "wait time", "long wait", 
                                             "visual interest", "animation density", 
                                             "position", "overlap"]))
        
        if needs_fixes:
            logger.info(f"Detected layout/positioning issues: {[i for i in issues if any(term in i.lower() for term in ['position', 'overlap', 'blank'])]}")
            layout_managed_code = CodeAnalyzer.fix_positioning_issues(layout_managed_code)
            layout_managed_code = CodeAnalyzer.fix_blank_screen_issues(layout_managed_code)
            logger.info("Applied automatic fixes for positioning and blank screens")
            
        return layout_managed_code

# Create a singleton instance of the renderer
renderer = ManimRenderer()
