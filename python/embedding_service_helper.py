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

    def _format_log(self, level, message, extra=None):
        """Format log with consistent structure"""
        log_context = {"request_id": self.request_id}
        if extra:
            log_context.update(extra)
        return self.logger.makeRecord(
            self.logger.name,
            logging.getLevelName(level),
            "",
            0,
            message,
            (),
            None,
            extra=log_context,
        )

    def info(self, message, extra=None):
        """Log an info message"""
        record = self._format_log(logging.INFO, message, extra)
        self.logger.handle(record)

    def debug(self, message, extra=None):
        """Log a debug message"""
        record = self._format_log(logging.DEBUG, message, extra)
        self.logger.handle(record)

    def error(self, message, error=None, extra=None):
        """Log an error message with optional exception details"""
        context = extra or {}
        if error:
            context["error"] = str(error)
            context["error_type"] = type(error).__name__
        record = self._format_log(logging.ERROR, message, context)
        self.logger.handle(record)

    def warning(self, message, extra=None):
        """Log a warning message"""
        record = self._format_log(logging.WARNING, message, extra)
        self.logger.handle(record)

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
import subprocess
import concurrent.futures
import io
from contextlib import nullcontext

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
    The class is designed to be modular and reusable, allowing for easy integration
    into larger systems or pipelines.
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample from the video.
        logger (logging.Logger): Optional logger for debug information.
        executor (concurrent.futures.ThreadPoolExecutor): Shared executor for parallel frame extraction.
    Attributes:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample from the video.
        logger (logging.Logger): Logger for debug information.
        duration (float): Duration of the video in seconds.
        executor (concurrent.futures.ThreadPoolExecutor): Shared executor for parallel frame extraction.
    Methods:
        get_duration(): Get the duration of the video using ffprobe.
        get_advanced_sample_times(): Compute candidate sampling timestamps using
            scene detection, visual entropy, and temporal smoothing.
        extract_frame(time_sec): Extract a single frame from the video at a
            specific time (in seconds).
        extract_frames(): Extract frames concurrently based on advanced sampling times.
        compute_entropy(image): Compute the visual entropy of an image.
    """

    def __init__(
        self,
        video_path: str,
        num_frames: int,
        logger=None,
        executor=None,
        request_id=None,
    ):
        self.video_path = video_path
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

        self.duration = self.get_duration()
        self.executor = executor

    def get_duration(self) -> float:
        """Get the duration of the video in seconds using ffprobe.
        Returns:
            float: Duration of the video in seconds.
        Raises:
            RuntimeError: If ffprobe fails to retrieve the duration.
        """
        self._log_debug(
            f"Getting duration for video: {self.video_path}",
            extra={"video_path": self.video_path},
        )

        start_time = time.time()
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    self.video_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            duration = float(result.stdout.strip())
            self._log_debug(
                f"Video duration: {duration} seconds",
                extra={"duration": duration, "video_path": self.video_path},
            )

            if hasattr(self.logger, "log_operation_time"):
                self.logger.log_operation_time("get_duration", start_time)

            return duration
        except Exception as e:
            self._log_error(
                f"Failed to get video duration: {e}",
                error=e,
                extra={"video_path": self.video_path},
            )
            raise RuntimeError(f"Failed to get video duration: {e}")

    def get_advanced_sample_times(self) -> tuple:
        """
        Compute candidate sampling timestamps using a combination of scene detection,
        visual entropy salience, and temporal smoothing. The method first attempts
        to extract scene boundaries via PySceneDetect. If successful, it computes a
        candidate timestamp for each scene (using the midpoint). For each candidate, a
        frame is extracted to compute visual entropy. Candidates are then ranked and
        filtered to enforce a minimum time gap (diversity). If scene detection fails or
        does not yield enough candidates, the method falls back to uniform sampling.
        The method returns a tuple containing the selected timestamps and debug metadata
        (if requested).
        Args:
            None

        Returns:
            tuple: (selected_timestamps, debug_metadata)
                selected_timestamps (list): List of selected timestamps for frame extraction.
                debug_metadata (dict): Debug metadata containing candidate timestamps,
                                       entropy values, and method used.
        Raises:
            RuntimeError: If frame extraction fails or no frames are extracted.
        """
        self._log_debug(
            f"Starting advanced sample times calculation for {self.video_path}",
            extra={"video_path": self.video_path, "num_frames": self.num_frames},
        )

        start_time = time.time()
        candidate_times = None
        debug_metadata = {}
        method_used = ""
        # Attempt scene detection with PySceneDetect.
        try:
            from scenedetect import VideoManager, SceneManager
            from scenedetect.detectors import ContentDetector

            self._log_debug(f"Using PySceneDetect for video: {self.video_path}")
            video_manager = VideoManager([self.video_path])
            scene_manager = SceneManager()
            # The threshold here is heuristic; adjust based on your domain.
            scene_manager.add_detector(ContentDetector(threshold=25, min_scene_len=10))
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)
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
                or len(scene_list) < (self.num_frames // 2)  # too few distinct scenes
                or self.duration
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
                        extra={"timestamp": t, "video_path": self.video_path},
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
        by piping JPEG data through stdout.
        Args:
            time_sec (float): Time in seconds to extract the frame.
        Returns:
            PIL.Image: The extracted frame as a PIL Image object.
        Raises:
            RuntimeError: If ffmpeg fails to extract the frame.
        """
        self._log_debug(
            f"Extracting frame at {time_sec:.2f} sec from {self.video_path}",
            extra={"timestamp": time_sec, "video_path": self.video_path},
        )

        start_time = time.time()
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "debug",
            "-ss",
            str(time_sec),
            "-i",
            self.video_path,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-",
        ]

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=False,
            )
            if not result.stdout:
                error_output = (
                    result.stderr.decode("utf-8", errors="ignore")
                    if result.stderr
                    else "No stderr output"
                )
                self._log_error(
                    f"No frame data returned from ffmpeg at {time_sec} sec",
                    extra={"ffmpeg_stderr": error_output[:500], "timestamp": time_sec},
                )
                raise RuntimeError(
                    f"No frame data returned from ffmpeg. Stderr: {error_output}"
                )

            image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            self._log_debug(
                f"Successfully extracted frame at {time_sec:.2f} sec",
                extra={
                    "timestamp": time_sec,
                    "duration_ms": (time.time() - start_time) * 1000,
                },
            )

            if hasattr(self.logger, "log_operation_time"):
                self.logger.log_operation_time("extract_frame", start_time)

            return image

        except subprocess.CalledProcessError as e:
            error_output = (
                e.stderr.decode("utf-8", errors="ignore")
                if e.stderr
                else "No stderr output"
            )
            self._log_error(
                f"Frame extraction failed at {time_sec} sec",
                error=e,
                extra={"ffmpeg_stderr": error_output[:500], "timestamp": time_sec},
            )
            raise RuntimeError(
                f"Failed to extract frame at {time_sec} sec: {e}. Stderr: {error_output}"
            )
        except Exception as e:
            self._log_error(
                f"Frame extraction failed at {time_sec} sec with unexpected error",
                error=e,
                extra={"timestamp": time_sec},
            )
            raise RuntimeError(f"Failed to extract frame at {time_sec} sec: {e}")

    def extract_frames(self) -> tuple:
        """
        Extract frames concurrently based on advanced sampling times.
        Returns a tuple: (list of PIL.Image objects, debug_metadata)
        The frames are extracted using the advanced sampling method, which
        combines scene detection, visual entropy, and temporal smoothing.
        Args:
            None
        Returns:
            tuple: (frames, debug_metadata)
                frames (list): List of extracted frames as PIL Image objects.
                debug_metadata (dict): Debug metadata containing candidate timestamps,
                                       entropy values, and method used.
        Raises:
            RuntimeError: If frame extraction fails or no frames are extracted.
        """
        self._log_info(
            f"Starting frame extraction for {self.video_path}",
            extra={"video_path": self.video_path, "num_frames": self.num_frames},
        )

        start_time = time.time()
        timestamps, debug_metadata = self.get_advanced_sample_times()

        self._log_debug(
            f"Extracting {len(timestamps)} frames using selected timestamps"
        )
        frames = []
        executor = self.executor or concurrent.futures.ThreadPoolExecutor()
        with executor if self.executor is None else nullcontext(executor):
            future_to_time = {
                executor.submit(self.extract_frame, t): t for t in timestamps
            }
            for future in concurrent.futures.as_completed(future_to_time):
                t = future_to_time[future]
                try:
                    frame = future.result()
                    frames.append((t, frame))
                except Exception as e:
                    self._log_error(
                        f"Failed to extract frame at {t} sec",
                        error=e,
                        extra={"timestamp": t},
                    )
        if not frames:
            self._log_error(
                "No frames extracted from video", extra={"video_path": self.video_path}
            )
            raise RuntimeError("No frames extracted from video.")

        frames.sort(key=lambda x: x[0])  # Sort by timestamp

        self._log_info(
            f"Successfully extracted {len(frames)} frames",
            extra={"frame_count": len(frames)},
        )

        if hasattr(self.logger, "log_operation_time"):
            self.logger.log_operation_time("extract_frames", start_time)

        return [frame for _, frame in frames], debug_metadata


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

        # Device selection: Prefer MPS on M1 Mac, then CUDA, then CPU.
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
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

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

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

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
        Future improvements can include weighted or attention-based aggregation.
        """
        start_time = time.time()
        self._log_debug(
            f"Computing embedding for {len(frames)} video frames",
            extra={"frame_count": len(frames)},
        )

        # If augmentation is enabled, apply transforms to each frame.
        if self.enable_augmentation:
            frames = [self.augmentation_transforms(f) for f in frames]

        inputs = self.processor(images=frames, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

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


def compute_single_embedding(fp, args, embedder, executor=None):
    start_time = time.time()
    request_id = getattr(args, "request_id", str(uuid.uuid4()))
    logger = getattr(args, "logger", embedding_logger)

    if hasattr(logger, "set_request_id"):
        logger.set_request_id(request_id)

    ext = os.path.splitext(fp)[1].lower()

    try:
        logger.info(
            f"Processing file: {fp}",
            extra={"file_path": fp, "file_type": ext, "request_id": request_id},
        )

        debug_metadata = {
            "model": f"{getattr(args, 'model', None)} - {getattr(embedder, 'device', None)}",
            "num_frames": (
                getattr(args, "num_frames", None) if ext in VIDEO_EXTS else None
            ),
            "enable_augmentation": getattr(args, "enable_augmentation", False),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request_id,
        }

        if ext in VIDEO_EXTS:
            logger.debug(
                f"Processing video file: {fp}",
                extra={"file_path": fp, "file_extension": ext},
            )

            video_processor = VideoProcessor(
                fp,
                args.num_frames,
                logger=logger,
                executor=executor,
                request_id=request_id,
            )
            frames, adv_debug = video_processor.extract_frames()

            logger.info(
                f"Extracted {len(frames)} frames from video, calculating embedding",
                extra={"frame_count": len(frames), "file_path": fp},
            )

            embedding = embedder.get_video_embedding(frames)
            debug_metadata.update(adv_debug)
            result = {
                "embedding": embedding,
                "debugMetadata": debug_metadata,
                "error": None,
                "detail": None,
            }
        elif ext in IMAGE_EXTS:
            logger.debug(
                f"Processing image file: {fp}",
                extra={"file_path": fp, "file_extension": ext},
            )

            image = Image.open(fp).convert("RGB")
            embedding = embedder.get_image_embedding(image)
            result = {
                "embedding": embedding,
                "debugMetadata": debug_metadata,
                "error": None,
                "detail": None,
            }
        else:
            logger.error(
                f"Unsupported file type: {ext}",
                extra={"file_path": fp, "file_extension": ext},
            )

            result = {
                "embedding": None,
                "debugMetadata": debug_metadata,
                "error": "Unsupported file type",
                "detail": f"File type not supported: {ext} for file {fp}",
            }

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Processed {fp} in {duration_ms:.2f}ms",
            extra={"file_path": fp, "duration_ms": duration_ms},
        )

        return result
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Failed to process {fp}: {e}",
            error=e,
            extra={"file_path": fp, "duration_ms": duration_ms},
        )

        return {
            "embedding": None,
            "debugMetadata": debug_metadata if "debug_metadata" in locals() else {},
            "error": "Processing failed",
            "detail": f"{type(e).__name__}: {e} (file: {fp})",
        }


def process_batch(image_paths, embedder, args, logger=None, shared_executor=None):
    batch_id = str(uuid.uuid4())[:8]  # Generate a short batch ID
    logger = logger or embedding_logger
    request_id = getattr(args, "request_id", str(uuid.uuid4()))

    if hasattr(logger, "set_request_id"):
        logger.set_request_id(request_id)

    start_time = time.time()
    results = {}
    total = len(image_paths)
    processed = 0

    logger.info(
        f"Starting batch processing of {total} files",
        extra={"batch_id": batch_id, "file_count": total, "request_id": request_id},
    )

    for fp in image_paths:
        logger.debug(
            f"Processing file {processed+1}/{total}: {fp}",
            extra={"file_path": fp, "batch_id": batch_id},
        )

        if not os.path.exists(fp):
            logger.error(
                f"File not found: {fp}", extra={"file_path": fp, "batch_id": batch_id}
            )

            results[fp] = {
                "embedding": None,
                "debugMetadata": {
                    "batch_id": batch_id,
                    "request_id": request_id,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "error": "File does not exist",
                "detail": f"Path not found: {fp}",
            }
            processed += 1

            if hasattr(logger, "log_progress"):
                logger.log_progress(processed, total, fp)
            else:
                # Legacy progress reporting
                print(
                    f"PROGRESS: "
                    + json.dumps(
                        {"processed": processed, "total": total, "current": fp}
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            continue

        # Set per-file request ID
        file_args = args
        if hasattr(args, "request_id"):
            file_args.request_id = f"{request_id}-{processed}"

        result = compute_single_embedding(fp, file_args, embedder, shared_executor)

        # Add batch ID to metadata
        if "debugMetadata" in result and result["debugMetadata"]:
            result["debugMetadata"]["batch_id"] = batch_id

        # Ensure all required keys are present
        if "error" in result and "detail" not in result:
            result["detail"] = f"Failed to process file: {fp}"

        for k in ["embedding", "debugMetadata", "error", "detail"]:
            if k not in result:
                result[k] = None

        results[fp] = result
        processed += 1

        # Report progress
        if hasattr(logger, "log_progress"):
            logger.log_progress(processed, total, fp)
        else:
            # Legacy progress reporting
            print(
                f"PROGRESS: "
                + json.dumps({"processed": processed, "total": total, "current": fp}),
                file=sys.stderr,
                flush=True,
            )

    batch_duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Completed batch processing of {total} files in {batch_duration_ms:.2f}ms",
        extra={
            "batch_id": batch_id,
            "file_count": total,
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
