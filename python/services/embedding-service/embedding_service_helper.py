#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python/embedding_service_helper.py
"""
embedding_service_helper.py – now supports batching & on-disk caching

Usage:
    python embedding_service_helper.py <file_paths> [--num_frames NUM] [--model MODEL] [--debug]

This script computes CLIP embeddings for one or more image or video files.
For video files, it extracts multiple frames selected via an advanced, content-aware sampling
strategy that combines scene-representative sampling with visual entropy-based salience and
temporal smoothing. The embeddings are computed in batch and averaged.
The script uses the Hugging Face Transformers library for CLIP model loading and inference,
and ffmpeg for video processing. It supports caching to avoid redundant computation.

Features:
    • Batch processing of multiple files.
    • On-disk caching to avoid redundant computation.
    • Image and video file support (JPEG, PNG, MP4, etc.).
    • Advanced video frame extraction using scene detection and visual entropy.
    • CLIP model loading and inference using Hugging Face Transformers.
    • Inference on CPU or GPU (if available).
    • Configurable number of frames for video processing.
    • Command-line arguments for file paths, model name, number of frames, and debug metadata.
    • JSON output format for embedding and debug metadata.
    • Error handling and logging.
    • Debug metadata output for advanced sampling methods.
    • Support for multiple CLIP models from Hugging Face.
    • Parallel frame extraction for performance.
    • In-memory frame extraction without temporary file I/O.
    • Modular design with classes for video processing and CLIP inference.
    • Configurable parameters via command-line arguments.
    • Structured logging and centralized error handling.
    • Scene detection via PySceneDetect (if installed) to extract scene midpoints.
    • Visual entropy computation as a salience heuristic.
    • Temporal diversity enforcement (smoothing) via a minimum time-gap filter.
    • Fallback to uniform sampling if advanced scene analysis fails.
    • Optional debug metadata (candidate timestamps, entropy values, selected timestamps).
    • Optional data augmentation for images and video frames.

Performance and production readiness improvements include:
    • Parallel frame extraction using concurrent futures.
    • In-memory frame extraction without temporary file I/O.
    • Modular design with classes for video processing and CLIP inference.
    • Configurable parameters via command-line arguments.
    • Structured logging and centralized error handling.

Advanced improvements in this version:
    • Scene detection via PySceneDetect (if installed) to extract scene midpoints.
    • Visual entropy computation as a salience heuristic.
    • Temporal diversity enforcement (smoothing) via a minimum time-gap filter.
    • Fallback to uniform sampling if advanced scene analysis fails.
    • Optional debug metadata (candidate timestamps, entropy values, selected timestamps).
    • Optional data augmentation for images and video frames.
"""

import logging
import sys
import os
import json
import uuid
from PIL import Image
import torch
import signal
import time
from datetime import datetime
import io
from typing import Union, Optional, Tuple, List, Dict, Any
import tempfile
import subprocess
import concurrent.futures
from contextlib import nullcontext
import re
from fastapi import HTTPException
from PIL import UnidentifiedImageError
import open_clip

# Define logging component and constants
PY_LOG_PREFIX = "[EmbeddingPython]"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_WARNING = "WARNING"


class EmbeddingLogger:
    """
    A specialized logger for the CLIP embedding Python process that provides
    consistent formatting that integrates with the Node.js embeddingsLogger.
    """

    def __init__(self, request_id=None):
        self.request_id = request_id or self._generate_request_id()
        self._setup_logger()

    def _generate_request_id(self):
        return str(uuid.uuid4())

    def _setup_logger(self):
        # Set up handlers
        # Console handler for stderr
        self.console_handler = logging.StreamHandler(sys.stderr)
        self.console_handler.setFormatter(
            logging.Formatter(f"{PY_LOG_PREFIX} %(levelname)s: %(message)s")
        )

        # File handler for detailed logs
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "python_embeddings.log")

        self.file_handler = logging.FileHandler(log_file_path)
        self.file_handler.setFormatter(
            logging.Formatter(
                '{"timestamp":"%(asctime)s", "level":"%(levelname)s", "component":"Python", "requestId":"%(request_id)s", "message":"%(message)s", "file":"%(pathname)s", "line":%(lineno)d}',
                "%Y-%m-%d %H:%M:%S",
            )
        )

        # Create logger
        self.logger = logging.getLogger("embedding_python")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers if any
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Add handlers
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)

        # Make sure logger propagation is off to avoid duplicate logs
        self.logger.propagate = False

    def _log_std(self, level, message, extra=None, exc_info=None):
        """Helper to log using the standard logger instance"""
        log_extra = dict(extra) if extra else {}
        log_extra["request_id"] = self.request_id  # Always include request_id
        if self.logger:
            self.logger.log(level, message, extra=log_extra, exc_info=exc_info)

    def info(self, message, extra=None):
        """Log an info message"""
        self._log_std(logging.INFO, message, extra=extra)

    def debug(self, message, extra=None):
        """Log a debug message"""
        self._log_std(logging.DEBUG, message, extra=extra)

    def error(self, message, error=None, extra=None):
        """Log an error message with optional exception details"""
        log_extra = dict(extra) if extra else {}
        if error:
            log_extra["error_type"] = type(error).__name__
            log_extra["error_message"] = str(error)
        self._log_std(logging.ERROR, message, extra=log_extra, exc_info=error)

    def warning(self, message, extra=None):
        """Log a warning message"""
        self._log_std(logging.WARNING, message, extra=extra)

    def log_progress(self, processed, total, current_file):
        """Log a standardized progress message"""
        progress_data = {
            "processed": processed,
            "total": total,
            "current": current_file,
            "percentage": int((processed / total) * 100) if total > 0 else 0,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        }
        print(f"PROGRESS: {json.dumps(progress_data)}", file=sys.stderr, flush=True)
        self.debug(
            f"Progress: {processed}/{total} files ({progress_data['percentage']}%)",
            extra={"progress": progress_data},
        )

    def set_request_id(self, request_id):
        """Update the request ID"""
        self.request_id = request_id

    def log_operation_time(self, operation, start_time):
        """Log the execution time of an operation"""
        duration_ms = (time.time() - start_time) * 1000
        self.debug(
            f"Operation '{operation}' completed in {duration_ms:.2f}ms",
            extra={"operation": operation, "duration_ms": duration_ms},
        )
        return duration_ms


# Create a default embedding logger
embedding_logger = EmbeddingLogger()

# Suppress verbose logs from libraries
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.ERROR)


def handle_sigint(signum, frame):
    embedding_logger.info("Received SIGINT (Ctrl+C). Exiting gracefully.")
    # Perform any additional cleanup here if needed
    sys.exit(0)


def handle_sigterm(signum, frame):
    embedding_logger.info("Received SIGTERM. Exiting gracefully.")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigterm)

from transformers import CLIPProcessor, CLIPModel
import math
import concurrent.futures
from contextlib import nullcontext
import re

# Supported file extensions for images and videos.
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"]
VIDEO_EXTS = [".mp4", ".mov", ".webm", ".ogg", ".m4v"]


def compute_entropy(image: Image.Image) -> float:
    """
    Compute the visual entropy of an image as a simple salience measure.
    The image is converted to grayscale; its normalized histogram is used to
    calculate entropy (in bits).
    Args:
        image (PIL.Image): The input image.
    Returns:
        float: The computed entropy value.
    """
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    total_pixels = sum(histogram)
    entropy = 0.0
    # Compute entropy as sum(-p * log2(p)) for nonzero probabilities.
    for count in histogram:
        if count > 0:
            p = count / total_pixels
            entropy -= p * math.log(p, 2)
    return entropy


class VideoProcessor:
    """
    A class to manage video processing tasks like duration extraction and advanced
    frame extraction using a combination of scene detection, visual entropy, and
    temporal smoothing.
    The class uses ffmpeg for video processing and PIL for image handling.
    It also provides methods for extracting frames based on advanced sampling
    strategies, including scene detection and visual entropy salience.
    It can accept either a file path or an in-memory buffer (io.BytesIO).
    The class is designed to be modular and reusable, allowing for easy integration
    into larger systems or pipelines.

    Args:
        video_path (Optional[str]): Path to the video file.
        video_buffer (Optional[io.BytesIO]): In-memory video data.
        num_frames (int): Number of frames to sample from the video.
        logger (Optional[EmbeddingLogger]): Optional logger for debug information.
        executor (concurrent.futures.ThreadPoolExecutor): Shared executor for parallel frame extraction.
        request_id (Optional[str]): Request ID for logging correlation.
        duration (Optional[float]): Pre-computed duration (e.g., from headers).

    Attributes:
        video_path (Optional[str]): Path to the video file, if provided.
        video_buffer (Optional[io.BytesIO]): In-memory video data, if provided.
        num_frames (int): Number of frames to sample from the video.
        logger (EmbeddingLogger): Logger for debug information.
        duration (float): Duration of the video in seconds.
        executor (concurrent.futures.ThreadPoolExecutor): Shared executor for parallel frame extraction.
    """

    def __init__(
        self,
        num_frames: int,
        video_path: Optional[str] = None,
        video_buffer: Optional[io.BytesIO] = None,
        logger=None,
        executor=None,
        request_id=None,
        duration: Optional[float] = None,
    ):
        if not (video_path or video_buffer) or (video_path and video_buffer):
            raise ValueError(
                "Exactly one of video_path or video_buffer must be provided."
            )

        if video_buffer and not video_path:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_buffer.getbuffer())
                self._tmp_path = tmp.name
            self.video_path = self._tmp_path
            self.video_buffer = None
        else:
            self.video_path = video_path
            self.video_buffer = video_buffer

        self.num_frames = num_frames
        self.logger = logger or embedding_logger

        if isinstance(logger, logging.Logger):
            # Wrap standard logger to make it compatible with our EmbeddingLogger
            self._log_info = lambda msg, **kwargs: logger.info(msg)
            self._log_debug = lambda msg, **kwargs: logger.debug(msg)
            self._log_error = lambda msg, **kwargs: logger.error(msg)
            self._log_warning = lambda msg, **kwargs: logger.warning(msg)
        else:
            # Use our enhanced logger
            self._log_info = self.logger.info
            self._log_debug = self.logger.debug
            self._log_error = self.logger.error
            self._log_warning = self.logger.warning

            # Set request_id if provided
            if request_id:
                self.logger.set_request_id(request_id)

        self.duration = duration if duration is not None else self.get_duration()
        if self.duration is None or self.duration <= 0:
            raise ValueError(
                "Could not determine video duration or duration is invalid."
            )

        self.executor = executor

    def get_duration(self) -> float:
        """Get the duration of the video in seconds using ffprobe.
        Returns:
            float: Duration of the video in seconds.
        Raises:
            RuntimeError: If ffprobe fails to retrieve the duration.
        """
        start_time = time.time()
        source_display = self.video_path if self.video_path else "in-memory buffer"
        extra_args = {"video_source": source_display}

        self._log_debug(
            f"Getting duration for video: {source_display}", extra=extra_args
        )

        try:
            if self.video_buffer:
                self.video_buffer.seek(0)
                command = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    "-i",
                    "pipe:0",
                ]

                result = subprocess.run(
                    command,
                    input=self.video_buffer.read(),  # send bytes through stdin
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,  # no text=True → no encode()
                )
                duration_str = result.stdout.decode().strip()
                self.video_buffer.seek(0)  # rewind for later use
            else:
                # For file paths, we can use the path directly
                command = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    "-i",
                    self.video_path,
                ]

                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                duration_str = result.stdout.decode().strip()

            if not duration_str or duration_str == "N/A":
                raise RuntimeError("ffprobe did not return a valid duration.")
            duration = float(duration_str)

            self._log_debug(
                f"Video duration: {duration} seconds",
                extra={"duration": duration, **extra_args},
            )

            if hasattr(self.logger, "log_operation_time"):
                self.logger.log_operation_time("get_duration", start_time)

            return duration
        except Exception as e:
            self._log_error(
                f"Failed to get video duration: {e}",
                error=e,
                extra=extra_args,
            )
            raise RuntimeError(
                f"Failed to get video duration for {source_display}: {e}"
            ) from e

    def get_advanced_sample_times(self) -> Tuple[List[float], Dict[str, Any]]:
        """
        Compute candidate sampling timestamps using a combination of scene detection,
        visual entropy salience, and temporal smoothing. The method first attempts
        to extract scene boundaries via PySceneDetect. If successful, it computes a
        candidate timestamp for each scene (using the midpoint). For each candidate, a
        frame is extracted to compute visual entropy. Candidates are then ranked and
        filtered to enforce a minimum time gap (diversity). If scene detection fails or
        does not yield enough candidates, the method falls back to uniform sampling.
        The method returns a tuple containing the selected timestamps and debug metadata
        (if requested). Scene detection is only attempted if a video_path is provided.

        Returns:
            tuple: (selected_timestamps, debug_metadata)
        """
        source_display = self.video_path if self.video_path else "in-memory buffer"
        extra_args = {"video_source": source_display, "num_frames": self.num_frames}

        self._log_debug(
            f"Starting advanced sample times calculation for {source_display}",
            extra=extra_args,
        )

        start_time = time.time()
        candidate_times = None
        debug_metadata = {}
        method_used = ""
        try:
            if self.video_path:
                from scenedetect import VideoManager, SceneManager
                from scenedetect.detectors import ContentDetector

                self._log_debug(f"Using PySceneDetect for video: {self.video_path}")
                video_manager = VideoManager([self.video_path])
                scene_manager = SceneManager()
                # The threshold here is heuristic; adjust based on your domain.
                scene_manager.add_detector(
                    ContentDetector(threshold=25, min_scene_len=10)
                )
                video_manager.start()
                scene_manager.detect_scenes(
                    frame_source=video_manager, show_progress=True
                )
                scene_list = scene_manager.get_scene_list()
                self._log_debug(
                    f"Scene detection complete. Found {len(scene_list)} scenes."
                )
                candidate_times = [
                    (scene[0].get_seconds() + scene[1].get_seconds()) / 2
                    for scene in scene_list
                ]
                debug_metadata["scene_count"] = len(scene_list)
                method_used = "scene_detection"

                # If scene coverage is insufficient or the video is too short, fall back to uniform sampling.
                if (
                    len(candidate_times)
                    < self.num_frames  # fewer scene midpoints than needed frames
                    or len(scene_list)
                    < (self.num_frames // 2)  # too few distinct scenes
                    or float(self.duration)
                    < (self.num_frames * 2)  # video too short for advanced sampling
                ):
                    self._log_warning(
                        "Insufficient scene coverage or short video; falling back to uniform sampling.",
                        extra={
                            "scene_count": len(scene_list),
                            "video_duration": self.duration,
                        },
                    )
                    candidate_times = None

        except Exception as e:
            self._log_warning(
                f"Scene detection failed or PySceneDetect not installed. Using uniform sampling.",
                error=e,
                extra={"video_path": self.video_path},
            )

        # If candidate times are insufficient, fall back to uniform dense extraction.
        if not candidate_times or len(candidate_times) < self.num_frames:
            candidate_times = [
                (i + 1) * self.duration / (self.num_frames + 1)
                for i in range(self.num_frames)
            ]
            method_used = "fallback_uniform"
            debug_metadata = {"method_used": method_used, "timestamps": candidate_times}

            if hasattr(self.logger, "log_operation_time"):
                self.logger.log_operation_time("get_advanced_sample_times", start_time)

            return candidate_times, debug_metadata

        # For each candidate timestamp, extract the frame and compute visual entropy.
        self._log_debug(
            f"Extracting {len(candidate_times)} candidate frames for entropy analysis"
        )
        candidate_frames = []
        executor = self.executor or concurrent.futures.ThreadPoolExecutor()
        with executor if self.executor is None else nullcontext(executor):
            future_to_time = {
                executor.submit(self.extract_frame, t): t for t in candidate_times
            }
            for future in concurrent.futures.as_completed(future_to_time):
                t = future_to_time[future]
                try:
                    frame = future.result()
                    candidate_frames.append((t, frame))
                except Exception as e:
                    self._log_error(
                        f"Failed to extract candidate frame at {t} sec",
                        error=e,
                        extra={"timestamp": t},
                    )
        if not candidate_frames:
            # If extraction completely failed, fallback.
            self._log_warning(
                "All candidate frame extractions failed. Using uniform fallback."
            )
            uniform_times = [
                (i + 1) * self.duration / (self.num_frames + 1)
                for i in range(self.num_frames)
            ]
            debug_metadata = {
                "method_used": "fallback_uniform_extraction",
                "timestamps": uniform_times,
            }

            if hasattr(self.logger, "log_operation_time"):
                self.logger.log_operation_time("get_advanced_sample_times", start_time)

            return uniform_times, debug_metadata

        # Compute entropy values for all candidate frames.
        self._log_debug(f"Computing entropy for {len(candidate_frames)} frames")
        entropy_values = []
        for t, frame in candidate_frames:
            try:
                entropy_val = compute_entropy(frame)
                entropy_values.append((t, entropy_val))
            except Exception as e:
                self._log_error(
                    f"Failed to compute entropy for frame at {t} sec",
                    error=e,
                    extra={"timestamp": t},
                )
                entropy_values.append((t, 0.0))  # Use zero entropy as fallback

        debug_metadata["entropy_values"] = entropy_values

        # Rank candidates by entropy (descending).
        entropy_values.sort(key=lambda x: x[1], reverse=True)

        # Select frames ensuring temporal diversity.
        selected = []
        diversity_threshold = (
            self.duration * 0.05
        )  # At least 5% of video duration apart.

        self._log_debug(
            f"Selecting diverse frames with threshold {diversity_threshold:.2f} sec"
        )

        for t, entropy_val in entropy_values:
            if not selected or all(abs(t - s) > diversity_threshold for s in selected):
                selected.append(t)
            if len(selected) == self.num_frames:
                break

        # In case diversity filtering does not yield enough frames, fill with lower-ranked ones.
        debug_metadata["selected_entropy_values"] = [
            (t, entropy_val) for t, entropy_val in entropy_values if t in selected
        ]

        if len(selected) < self.num_frames:
            self._log_debug(
                f"Diversity filtering resulted in only {len(selected)} frames. "
                + f"Adding {self.num_frames - len(selected)} more."
            )
            remaining = sorted([t for t, _ in entropy_values if t not in selected])
            for t in remaining:
                if len(selected) < self.num_frames:
                    selected.append(t)

        selected.sort()  # Sort timestamps chronologically
        debug_metadata["selected_times"] = selected
        debug_metadata["method_used"] = method_used

        self._log_debug(
            f"Final selection: {len(selected)} frames using method '{method_used}'"
        )

        if hasattr(self.logger, "log_operation_time"):
            self.logger.log_operation_time("get_advanced_sample_times", start_time)

        return selected, debug_metadata

    def extract_frame(self, time_sec: float) -> Image.Image:
        """
        Extract a single frame from the video at a specific time (in seconds)
        using direct pipe communication for in-memory buffers.

        Args:
            time_sec (float): Time in seconds to extract the frame.

        Returns:
            PIL.Image: The extracted frame as a PIL Image object.

        Raises:
            RuntimeError: If frame extraction fails.
        """
        source_display = self.video_path if self.video_path else "in-memory buffer"
        extra_args = {"timestamp": time_sec, "video_source": source_display}

        self._log_debug(
            f"Extracting frame at {time_sec:.2f} sec from {source_display}",
            extra=extra_args,
        )

        start_time = time.time()

        # Base command with optimized settings
        command = [
            "ffmpeg",
            "-y",  # Overwrite output without asking
            "-loglevel",
            "error",  # Only show errors
            "-threads",
            "1",  # Limit to single thread per extraction
        ]

        # Add input source and seeking BEFORE input for faster seeking
        if self.video_buffer:
            command.extend(["-i", "pipe:0"])
        else:
            command.extend(["-i", self.video_path])

        # Add frame extraction settings
        command.extend(
            [
                "-ss",
                str(time_sec),
                "-vframes",
                "1",
                "-f",
                "image2pipe",
                "-pix_fmt",
                "rgb24",
                "-vcodec",
                "rawvideo",
                "-",
            ]
        )

        try:
            # Set up process with appropriate pipes
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if self.video_buffer else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 * 1024 * 1024,  # 10MB buffer
            )

            # If using buffer, send it to stdin
            if self.video_buffer:
                self.video_buffer.seek(0)
                stdout, stderr = process.communicate(input=self.video_buffer.read())
                self.video_buffer.seek(0)
            else:
                stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_output = (
                    stderr.decode("utf-8", errors="ignore")
                    if stderr
                    else "No stderr output"
                )
                raise RuntimeError(
                    f"FFmpeg failed with code {process.returncode}: {error_output}"
                )

            if not stdout:
                raise RuntimeError("No frame data received from FFmpeg")

            # Process raw frame data
            try:
                # Default frame size (can be adjusted based on your needs)
                width = height = 224  # CLIP's default size

                # Calculate frame size from buffer
                pixels = len(stdout) // 3  # 3 bytes per pixel for RGB24
                frame_dim = int(math.sqrt(pixels))
                if frame_dim * frame_dim * 3 == len(stdout):
                    width = height = frame_dim

                # Create PIL image from raw RGB data
                image = Image.frombytes("RGB", (width, height), stdout, "raw", "RGB")
                image = image.convert("RGB")  # Ensure RGB format

                self._log_debug(
                    f"Successfully extracted frame at {time_sec:.2f} sec",
                    extra={
                        "duration_ms": (time.time() - start_time) * 1000,
                        "frame_size": f"{width}x{height}",
                        **extra_args,
                    },
                )

                if hasattr(self.logger, "log_operation_time"):
                    self.logger.log_operation_time("extract_frame", start_time)

                return image

            except Exception as e:
                self._log_error(
                    f"Failed to create image from frame data: {e}",
                    error=e,
                    extra=extra_args,
                )
                return self._extract_frame_fallback(time_sec)

        except Exception as e:
            self._log_error(
                f"Frame extraction failed at {time_sec} sec: {e}",
                error=e,
                extra=extra_args,
            )
            return self._extract_frame_fallback(time_sec)

    def _extract_frame_fallback(self, time_sec: float) -> Image.Image:
        """
        Fallback method to extract a frame using MJPEG when the optimized
        method fails. This is more compatible but potentially slower.

        Args:
            time_sec (float): Time in seconds to extract the frame.

        Returns:
            PIL.Image: The extracted frame as a PIL Image object.

        Raises:
            RuntimeError: If frame extraction completely fails.
        """
        source_display = self.video_path if self.video_path else "in-memory buffer"
        self._log_debug(
            f"Using fallback frame extraction for {source_display} at {time_sec} sec"
        )

        # Fallback to classic MJPEG method which is more compatible
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-ss",
            str(time_sec),
            "-i",
            self.video_path if self.video_path else "-",
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",  # Use MJPEG for compatibility
            "-q:v",
            "2",  # High quality (low compression)
            "-",
        ]

        # If using buffer, read from stdin ('-') and pass buffer content
        input_data = None
        if self.video_buffer:
            self.video_buffer.seek(0)
            input_data = self.video_buffer.read()
            self.video_buffer.seek(0)  # Reset after read

        try:
            result = subprocess.run(
                command,
                input=input_data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=False,
                timeout=30,  # 30 second timeout
            )

            if not result.stdout:
                error_output = (
                    result.stderr.decode("utf-8", errors="ignore")
                    if result.stderr
                    else "No stderr output"
                )
                raise RuntimeError(
                    f"No frame data returned from ffmpeg. Stderr: {error_output}"
                )

            # Create image from JPEG data
            image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            return image

        except Exception as e:
            self._log_error(
                f"Fallback frame extraction also failed at {time_sec} sec: {e}", error=e
            )
            raise RuntimeError(f"Failed to extract frame at {time_sec} sec: {e}")

    def extract_frames(self) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        Efficiently extract multiple frames from the video using a single ffmpeg process.

        Returns:
            Tuple[List[Image.Image], Dict[str, Any]]: List of PIL images and debug metadata.

        Performance:
            - Uses hardware acceleration if available.
            - Reduces frame size to 128x128 for speed.
            - Logs extraction time and frame count.
        """
        start = time.time()
        step = max(self.duration / (self.num_frames + 1), 0.001)
        vf = f"fps=1/{step},scale=128:-1"  # Lower frame size to 128px tall

        # Check if hardware acceleration should be used
        use_hwaccel = os.environ.get("FFMPEG_HWACCEL", "on").lower() not in (
            "off",
            "false",
            "0",
        )

        # Build ffmpeg command with optional hardware acceleration
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
        ]

        # Only add hardware acceleration if enabled
        if use_hwaccel:
            cmd.extend(["-hwaccel", "auto"])  # Use hardware acceleration if available

        cmd.extend(
            [
                "-i",
                self.video_path,
                "-vf",
                vf,
                "-vframes",
                str(self.num_frames),
                "-f",
                "image2pipe",
                "-pix_fmt",
                "rgb24",
                "-vcodec",
                "rawvideo",
                "-",
            ]
        )

        frame_sz = 128 * 128 * 3

        def run_ffmpeg(command):
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            raw = proc.stdout.read()
            stderr = proc.stderr.read()
            proc.wait()
            return proc.returncode, raw, stderr

        # Try with hardware acceleration first (if enabled)
        retcode, raw, stderr = run_ffmpeg(cmd)
        frame_count = len(raw) // frame_sz
        if retcode != 0 or frame_count == 0:
            stderr_text = stderr.decode("utf-8", errors="ignore")
            self._log_warning(
                f"Hardware-accelerated ffmpeg failed or no frames decoded; retrying without -hwaccel. Error: {stderr_text}"
            )
            # Remove -hwaccel and 'auto' from the command
            cmd_no_hwaccel = [c for c in cmd if c not in ("-hwaccel", "auto")]
            retcode, raw, stderr = run_ffmpeg(cmd_no_hwaccel)
            frame_count = len(raw) // frame_sz
            if retcode != 0 or frame_count == 0:
                stderr_text = stderr.decode("utf-8", errors="ignore")
                self._log_error(
                    f"ffmpeg still failed; falling back to per‑frame extraction. Error: {stderr_text}"
                )
                # Fallback: extract frames one by one
                try:
                    timestamps, debug_info = self.get_advanced_sample_times()
                    return [
                        self.extract_frame(t) for t in timestamps[: self.num_frames]
                    ], {"method_used": "frame_by_frame_fallback"}
                except Exception as e:
                    self._log_error(
                        f"Frame-by-frame extraction also failed: {e}", error=e
                    )
                    raise RuntimeError(f"All frame extraction methods failed: {e}")

        # If we got fewer frames than requested, log a warning but proceed
        if frame_count < self.num_frames:
            self._log_warning(
                f"ffmpeg returned only {frame_count} frames (expected {self.num_frames}); using what we have"
            )

        # Process the frames we got
        frames = []
        for i in range(0, min(self.num_frames, frame_count) * frame_sz, frame_sz):
            if i < len(raw):
                try:
                    frame = Image.frombytes("RGB", (128, 128), raw[i : i + frame_sz])
                    frames.append(frame)
                except Exception as e:
                    self._log_error(
                        f"Failed to create frame from bytes at position {i}: {e}",
                        error=e,
                    )

        duration_ms = (time.time() - start) * 1000
        self._log_debug(
            f"One‑shot ffmpeg delivered {len(frames)} frames",
            extra={"duration_ms": duration_ms},
        )

        # If we still got no frames, try the frame-by-frame approach as last resort
        if not frames:
            self._log_warning(
                "No valid frames from one-shot method, falling back to frame-by-frame extraction"
            )
            try:
                timestamps, debug_info = self.get_advanced_sample_times()
                return [self.extract_frame(t) for t in timestamps[: self.num_frames]], {
                    "method_used": "frame_by_frame_fallback"
                }
            except Exception as e:
                self._log_error(f"Complete extraction failure: {e}", error=e)
                raise RuntimeError(f"All frame extraction methods failed: {e}")

        return frames, {
            "method_used": "one_shot_ffmpeg",
            "duration_ms": duration_ms,
            "requested_frames": self.num_frames,
            "actual_frames": len(frames),
        }

    def __del__(self):
        if getattr(self, "_tmp_path", None):
            try:
                os.remove(self._tmp_path)
            except FileNotFoundError:
                pass


class CLIPEmbedder:
    """
    A class to handle CLIP model loading and embedding computation for images and videos.
    The class uses the Hugging Face Transformers library to load the CLIP model and
    processor. It provides methods for computing embeddings for both single images and
    averaged embeddings from multiple video frames.
    Args:
        model_name (str): Name of the CLIP model to load from Hugging Face.
        device (str): Device to run the model on ('mps', 'cuda', or 'cpu').
        logger (logging.Logger): Optional logger for debug information.
        enable_augmentation (bool): If True, apply basic image augmentations.
    Attributes:
        model_name (str): Name of the CLIP model.
        device (str): Device to run the model on.
        logger (logging.Logger): Logger for debug information.
        model (CLIPModel): Loaded CLIP model.
        processor (CLIPProcessor): Processor for pre-processing inputs.
        enable_augmentation (bool): Flag to enable data augmentation.
        augmentation_transforms (torchvision.transforms.Compose): Data augmentation transforms.
    Methods:
        get_image_embedding(image): Compute and return the normalized CLIP embedding for a single image.
        get_video_embedding(frames): Compute and return an averaged CLIP embedding from multiple video frames.
    """

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        device=None,
        logger=None,
        enable_augmentation=False,
        request_id=None,
    ):
        self.logger = logger or embedding_logger
        self.model_name = model_name

        if isinstance(logger, logging.Logger):
            # Wrap standard logger
            self._log_info = lambda msg, **kwargs: logger.info(msg)
            self._log_debug = lambda msg, **kwargs: logger.debug(msg)
            self._log_error = lambda msg, **kwargs: logger.error(msg)
            self._log_warning = lambda msg, **kwargs: logger.warning(msg)
        else:
            # Use enhanced logger
            self._log_info = self.logger.info
            self._log_debug = self.logger.debug
            self._log_error = self.logger.error
            self._log_warning = self.logger.warning

            # Set request_id if provided
            if request_id:
                self.logger.set_request_id(request_id)

        # Device selection: Prefer MPS on Apple, then CUDA, then CPU.
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._log_info(
            f"Initializing CLIP model {model_name} on {self.device}",
            extra={"model": model_name, "device": self.device},
        )

        start_time = time.time()
        # Switch to OpenCLIP pre-trained weights and transforms
        # Using laion/CLIP-ViT-B-32-laion2B-s34B-b79K
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2B-s34B-b79K"
        )
        self.model.to(self.device)

        # If we have GPU or MPS, bump matmul precision for float32 ops
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            torch.set_float32_matmul_precision("high")
            self._log_info(
                "Enabled high float32 matmul precision",
                extra={"device": self.device},
            )

        self._log_info(
            f"Model initialization complete in {(time.time() - start_time):.2f} seconds"
        )

        # Optional data augmentation transforms for images.
        self.enable_augmentation = enable_augmentation
        if enable_augmentation:
            import torchvision.transforms as T

            self._log_info("Data augmentation enabled")
            self.augmentation_transforms = T.Compose(
                [
                    T.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            self.augmentation_transforms = None

    def get_image_embedding(self, image: Image.Image) -> list:
        """
        Compute and return the normalized CLIP embedding for a single image,
        optionally applying data augmentation.
        """
        start_time = time.time()
        self._log_debug(
            "Computing image embedding",
            extra={"image_size": f"{image.width}x{image.height}"},
        )

        if self.enable_augmentation:
            image = self.augmentation_transforms(image)

        # Preprocess via OpenCLIP transforms (returns a tensor)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        # Run model in half-precision on GPU/MPS for speed
        with torch.autocast(self.device, dtype=torch.float16), torch.no_grad():
            image_features = self.model.encode_image(image_input)

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        embedding = image_features[0].cpu().numpy().tolist()

        self._log_debug(
            f"Image embedding computed, shape: {len(embedding)}",
            extra={
                "embedding_size": len(embedding),
                "duration_ms": (time.time() - start_time) * 1000,
            },
        )

        if hasattr(self.logger, "log_operation_time"):
            self.logger.log_operation_time("get_image_embedding", start_time)

        return embedding

    def get_video_embedding(self, frames: list) -> list:
        """
        Compute and return an averaged CLIP embedding from multiple video frames.
        """
        start_time = time.time()
        self._log_debug(
            f"Computing embedding for {len(frames)} video frames",
            extra={"frame_count": len(frames)},
        )

        if self.enable_augmentation:
            frames = [self.augmentation_transforms(f) for f in frames]

        # Preprocess frames via OpenCLIP transforms and batch into a tensor
        video_input = torch.stack([self.preprocess(f) for f in frames]).to(self.device)
        # Run model in half-precision on GPU/MPS for speed
        with torch.autocast(self.device, dtype=torch.float16), torch.no_grad():
            image_features = self.model.encode_image(video_input)

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        averaged_embedding = image_features.mean(dim=0)
        embedding = averaged_embedding.cpu().numpy().tolist()

        self._log_debug(
            f"Video embedding computed, shape: {len(embedding)}",
            extra={
                "embedding_size": len(embedding),
                "duration_ms": (time.time() - start_time) * 1000,
            },
        )

        if hasattr(self.logger, "log_operation_time"):
            self.logger.log_operation_time("get_video_embedding", start_time)

        return embedding


def compute_single_embedding(
    input_data: Union[str, io.BytesIO, Image.Image],
    args: Any,  # Typically argparse.Namespace or similar object
    embedder: CLIPEmbedder,
    executor: Optional[concurrent.futures.Executor] = None,
    media_type: Optional[str] = None,  # Explicit 'image' or 'video' for buffers
    identifier: Optional[str] = None,  # Original filename or identifier for logging
) -> Dict[str, Any]:
    """
    Computes the CLIP embedding for a single input, which can be a file path,
    an in-memory image (PIL), or an in-memory video/image buffer (BytesIO).

    Args:
        input_data: The data to process (file path string, PIL Image, or BytesIO buffer).
        args: Configuration arguments (e.g., num_frames, model name, augmentation flag).
              Should include request_id if available. For buffers, media_type is expected here
              if not passed explicitly.
        embedder: Initialized CLIPEmbedder instance.
        executor: Optional shared thread pool executor for video frame extraction.
        media_type: Explicitly specifies 'image' or 'video' when input_data is BytesIO.
                    Overrides sniffing or args.media_type.
        identifier: A string identifier (like original filename) used for logging/results.
                    If None, a UUID or placeholder will be used.

    Returns:
        A dictionary containing the embedding, debug metadata, and error information.
    """
    start_time = time.time()
    request_id = getattr(args, "request_id", str(uuid.uuid4()))
    logger = getattr(args, "logger", embedding_logger)
    item_id = identifier or (
        input_data if isinstance(input_data, str) else f"buffer_{request_id}"
    )
    input_type_str = type(input_data).__name__

    if hasattr(logger, "set_request_id"):
        logger.set_request_id(request_id)

    try:
        # Determine media type and prepare for processing
        resolved_media_type = None
        video_duration_hint = getattr(args, "duration", None)  # Duration from headers?

        logger.info(
            f"Processing item: {item_id}",
            extra={
                "identifier": item_id,
                "input_type": input_type_str,
                "request_id": request_id,
            },
        )

        debug_metadata = {
            "model": f"{getattr(args, 'model', None)} - {getattr(embedder, 'device', None)}",
            "enable_augmentation": getattr(args, "enable_augmentation", False),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request_id,
        }

        if isinstance(input_data, str):  # File path input
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"Input file not found: {input_data}")
            ext = os.path.splitext(input_data)[1].lower()
            if ext in VIDEO_EXTS:
                resolved_media_type = "video"
            elif ext in IMAGE_EXTS:
                resolved_media_type = "image"
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            debug_metadata["source_type"] = "filepath"
            debug_metadata["extension"] = ext
        elif isinstance(input_data, Image.Image):  # PIL Image input
            resolved_media_type = "image"
            debug_metadata["source_type"] = "pil_image"
        elif isinstance(input_data, io.BytesIO):  # Buffer input
            resolved_media_type = media_type or getattr(args, "media_type", None)
            if not resolved_media_type:
                # Basic content sniffing could be added here if needed
                raise ValueError(
                    "Media type ('image' or 'video') must be specified for buffer input."
                )
            if resolved_media_type not in ["image", "video"]:
                raise ValueError(
                    f"Invalid media type for buffer: {resolved_media_type}"
                )
            input_data.seek(0)  # Ensure buffer is at the start
            debug_metadata["source_type"] = "buffer"
            debug_metadata["specified_media_type"] = resolved_media_type
        else:
            raise TypeError(f"Unsupported input data type: {input_type_str}")

        # --- Process Video ---
        if resolved_media_type == "video":
            logger.debug(
                f"Processing video: {item_id}",
                extra={"identifier": item_id, "source": debug_metadata["source_type"]},
            )
            debug_metadata["num_frames"] = args.num_frames

            video_processor = VideoProcessor(
                video_path=input_data if isinstance(input_data, str) else None,
                video_buffer=input_data if isinstance(input_data, io.BytesIO) else None,
                num_frames=args.num_frames,
                logger=logger,
                executor=executor,
                request_id=request_id,
                duration=video_duration_hint,  # Pass pre-computed duration if available
            )
            frames, adv_debug = video_processor.extract_frames()

            logger.info(
                f"Extracted {len(frames)} frames from video, calculating embedding",
                extra={"frame_count": len(frames), "identifier": item_id},
            )

            embedding = embedder.get_video_embedding(frames)
            debug_metadata.update(adv_debug)

        # --- Process Image ---
        elif resolved_media_type == "image":
            logger.debug(
                f"Processing image: {item_id}",
                extra={"identifier": item_id, "source": debug_metadata["source_type"]},
            )
            if isinstance(input_data, str):
                image = Image.open(input_data).convert("RGB")
            elif isinstance(input_data, io.BytesIO):
                image = Image.open(input_data).convert("RGB")
            else:  # PIL Image
                image = input_data  # Already a PIL image

            embedding = embedder.get_image_embedding(image)
            # No video-specific debug metadata here
        else:
            # This case should theoretically be caught earlier
            raise ValueError(
                f"Internal error: Unhandled resolved_media_type '{resolved_media_type}'"
            )

        # --- Prepare Result ---
        result = {
            "embedding": embedding,
            "debugMetadata": debug_metadata,
            "error": None,
            "detail": None,
        }

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Processed {item_id} in {duration_ms:.2f}ms",
            extra={"identifier": item_id, "duration_ms": duration_ms, "success": True},
        )

        return result
    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Failed to process {item_id}: {e}",
            error=e,
            extra={"identifier": item_id, "duration_ms": duration_ms, "success": False},
        )

        return {
            "embedding": None,
            "debugMetadata": debug_metadata if "debug_metadata" in locals() else {},
            "error": "Processing failed",
            "detail": f"{type(e).__name__}: {e} (item: {item_id})",
        }


def process_batch(
    inputs: List[
        Union[str, Tuple[Union[io.BytesIO, Image.Image], str, Optional[str]]]
    ],  # List of paths or (data, media_type, identifier) tuples
    embedder: CLIPEmbedder,
    args: Any,
    logger=None,
    shared_executor=None,
) -> Dict[str, Any]:
    """
    Processes a batch of inputs (file paths or in-memory data).

    Args:
        inputs: A list where each element is either:
                - A string (file path).
                - A tuple: (input_data, media_type, identifier)
                  where input_data is BytesIO or PIL.Image, media_type is 'image'/'video',
                  and identifier is a string for logging/results.
        embedder: Initialized CLIPEmbedder.
        args: Configuration arguments.
        logger: Logger instance.
        shared_executor: Thread pool executor.

    Returns:
        A dictionary mapping the identifier (file path or provided identifier) to the result dict
        from compute_single_embedding.
    """
    batch_id = str(uuid.uuid4())[:8]  # Generate a short batch ID
    logger = logger or embedding_logger
    request_id = getattr(args, "request_id", str(uuid.uuid4()))

    if hasattr(logger, "set_request_id"):
        logger.set_request_id(request_id)

    start_time = time.time()
    results: Dict[str, Any] = {}
    total = len(inputs)

    logger.info(
        f"Starting batch processing of {total} files in parallel",
        extra={"batch_id": batch_id, "file_count": total, "request_id": request_id},
    )

    import copy

    max_workers = getattr(args, "batch_workers", os.cpu_count() or 4)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as batch_executor:
        future_to_id: Dict[concurrent.futures.Future, str] = {}
        for idx, item_input in enumerate(inputs):
            # Prepare a per-item copy of args with its own request_id
            file_args = copy.copy(args)
            file_args.request_id = f"{request_id}-{idx+1}"
            file_args.logger = logger

            # Resolve input_data, media_type, identifier
            if isinstance(item_input, str):
                input_data = item_input
                identifier = item_input
                ext = os.path.splitext(item_input)[1].lower()
                media_type = "video" if ext in VIDEO_EXTS else "image"
            else:
                input_data, media_type, identifier = item_input

            # Submit compute task
            future = batch_executor.submit(
                compute_single_embedding,
                input_data,
                file_args,
                embedder,
                shared_executor,
                media_type,
                identifier,
            )
            future_to_id[future] = identifier

        processed = 0
        for future in concurrent.futures.as_completed(future_to_id):
            identifier = future_to_id[future]
            result = future.result()
            results[identifier] = result
            processed += 1
            # Report progress
            if hasattr(logger, "log_progress"):
                logger.log_progress(processed, total, identifier)
            else:
                print(
                    json.dumps(
                        {"processed": processed, "total": total, "current": identifier}
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    batch_duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Completed batch processing of {total} items in {batch_duration_ms:.2f}ms",
        extra={
            "batch_id": batch_id,
            "item_count": total,
            "duration_ms": batch_duration_ms,
        },
    )
    return results


def main_service():
    import sys
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="CLIP embedding service (stdin/stdout)"
    )
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("-n", "--num_frames", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--enable_augmentation", action="store_true", default=False)

    args = parser.parse_args()

    # Create service-level logger
    service_logger = EmbeddingLogger()
    service_logger.info(
        "CLIP embedding service starting with model: " + args.model,
        extra={"model": args.model, "num_frames": args.num_frames},
    )

    # Initialize the embedder
    start_time = time.time()
    embedder = CLIPEmbedder(
        model_name=args.model,
        device=None,
        logger=service_logger,
        enable_augmentation=args.enable_augmentation,
    )
    service_logger.info(
        f"CLIPEmbedder initialized on device: {embedder.device}",
        extra={
            "device": embedder.device,
            "init_time_ms": (time.time() - start_time) * 1000,
        },
    )

    shared_executor = concurrent.futures.ThreadPoolExecutor()
    service_logger.info("Thread pool executor initialized for parallel processing")
    service_logger.info("CLIP embedding service ready. Waiting for input...")

    # Main service loop
    request_counter = 0
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                service_logger.info("End of input stream detected, exiting service")
                break  # EOF

            line = line.strip()
            if not line:
                continue

            request_counter += 1
            request_id = f"req-{request_counter}-{int(time.time())}"
            batch_logger = EmbeddingLogger(request_id)

            batch_logger.info(
                f"Received request #{request_counter}",
                extra={"input_length": len(line)},
            )

            try:
                request = json.loads(line)
                image_paths = request.get("imagePaths")

                if not isinstance(image_paths, list):
                    batch_logger.error(
                        "Invalid input: 'imagePaths' must be a list",
                        extra={"input_type": type(image_paths).__name__},
                    )

                    error_response = {
                        "error": "Invalid input",
                        "detail": "'imagePaths' must be a list.",
                    }
                    print(json.dumps(error_response), flush=True)
                    continue

                batch_logger.info(
                    f"Processing batch of {len(image_paths)} files",
                    extra={"file_count": len(image_paths)},
                )
            except Exception as e:
                batch_logger.error("Failed to parse request JSON", error=e)

                error_response = {"error": "Invalid input", "detail": str(e)}
                print(json.dumps(error_response), flush=True)
                continue

            try:
                # Add request_id to args
                args.request_id = request_id
                args.logger = batch_logger

                # Process the batch
                start_time = time.time()
                results = process_batch(
                    image_paths, embedder, args, batch_logger, shared_executor
                )

                duration_ms = (time.time() - start_time) * 1000
                batch_logger.info(
                    f"Batch processing completed in {duration_ms:.2f}ms",
                    extra={
                        "duration_ms": duration_ms,
                        "file_count": len(image_paths),
                        "success_count": sum(
                            1 for r in results.values() if not r.get("error")
                        ),
                    },
                )

                # Send response
                print(json.dumps(results), flush=True)

            except Exception as e:
                batch_logger.error(
                    "Fatal error in batch processing",
                    error=e,
                    extra={"stack_trace": str(e.__traceback__)},
                )

                # Instead of exiting, print a batchError JSON and continue
                batch_error = {
                    "batchError": str(e),
                    "detail": f"{type(e).__name__}: {e}",
                }
                print(json.dumps(batch_error), flush=True)
                continue

        except Exception as e:
            service_logger.error(f"Fatal error in main loop: {e}", error=e)
            break

    service_logger.info("Embedding service shutting down")
    shared_executor.shutdown()


if __name__ == "__main__":
    main_service()
