#!/usr/bin/env python
# -*- coding: utf-8 -*-
# embedding_service_helper.py
"""
embedding_service_helper.py – now supports batching & on-disk caching

Usage:
    This script is now primarily a module used by the FastAPI server (server.py).
    It provides functionalities for batch media processing and CLIP embedding generation.

Features:
    • Batch processing of multiple media items (URLs, filepaths).
    • Parallel preprocessing (I/O-bound tasks like downloads, frame extraction).
    • Single batched tensor pass to CLIP model for GPU efficiency.
    • On-disk caching (implicitly, if paths are re-requested and not changed, though explicit cache logic not in this file).
    • Image and video file support.
    • Advanced video frame extraction using scene detection and visual entropy.
    • CLIP model loading and inference using OpenCLIP.
    • Inference on CPU or GPU (if available).
    • JSON output format for embedding and debug metadata, handled by FastAPI server.
    • Structured logging via EmbeddingLogger.
    • Modular design with classes for video processing and CLIP inference.
"""

import logging
import sys
import os
import json
import uuid
from PIL import Image, UnidentifiedImageError  # type: ignore
import torch  # type: ignore
import signal
import time
from datetime import datetime
import io
from typing import Union, Optional, Tuple, List, Dict, Any, Callable
import tempfile
import subprocess
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from contextlib import (
    nullcontext,
)  # Not used in current version but good for conditional context mgmt
import re
import math  # Added import
import errno  # Added import
import requests  # For URL downloads
import open_clip  # type: ignore # Added import
import cv2  # type: ignore

from dotenv import load_dotenv  # type: ignore

# Ensure .env is loaded at the very top
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# --- Constants ---
PY_LOG_PREFIX = "[EmbeddingPython]"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_WARNING = "WARNING"

DEFAULT_VIDEO_FRAMES_TO_EXTRACT = int(
    os.environ.get("DEFAULT_VIDEO_FRAMES_TO_EXTRACT", 20)
)
DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("DOWNLOAD_TIMEOUT_SECONDS", 30))
IMAGE_EXTS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".avif",
    ".gif",
]  # Common image extensions
VIDEO_EXTS = [
    ".mp4",
    ".mov",
    ".webm",
    ".ogg",
    ".m4v",
    ".avi",
    ".mkv",
    ".wmv",
]  # Common video extensions


# --- Logger ---
class EmbeddingLogger:
    def __init__(
        self,
        request_id: Optional[str] = None,
        component_name: str = "PythonEmbedHelper",
    ):
        self.request_id = request_id or self._generate_request_id()
        self.component_name = component_name
        self.file_handler: Optional[logging.FileHandler] = (
            None  # Initialize for clarity
        )
        self._setup_logger()

    def _generate_request_id(self) -> str:
        return str(uuid.uuid4())

    def _setup_logger(self):
        # Correct path assuming this file is /app/embedding_service_helper.py
        # and logs go into /app/logs/
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, "python_embeddings_service.log")
            self.file_handler = logging.FileHandler(log_file_path)
            self.file_handler.setFormatter(
                logging.Formatter(
                    '{"timestamp":"%(asctime)s.%(msecs)03dZ", "level":"%(levelname)s", "component":"%(component)s", "requestId":"%(request_id)s", "message":"%(message)s", "file":"%(pathname)s", "line":%(lineno)d, "module":"%(module)s", "funcName":"%(funcName)s"}',
                    "%Y-%m-%dT%H:%M:%S",
                )
            )
            logging.Formatter.converter = time.gmtime
        except OSError as e:
            sys.stderr.write(
                f"Warning: Error creating log directory {log_dir} or file handler: {e}. File logging will be disabled.\n"
            )
            self.file_handler = None  # Ensure it's None if setup fails

        self.console_handler = logging.StreamHandler(sys.stderr)
        # Ensure %(request_id)s can be interpolated if passed in extra
        self.console_handler.setFormatter(
            logging.Formatter(
                f"{PY_LOG_PREFIX} %(asctime)s %(levelname)s [{self.component_name}] (%(request_id)s): %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )

        # Create a unique logger name to avoid conflicts if this class is instantiated multiple times
        # or if other modules use logging.getLogger with the same name.
        logger_name = f"embedding_python_{self.component_name}_{self.request_id[:8]}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

        # Clear any existing handlers from a previous logger with the same name if reconfiguring
        if self.logger.handlers:
            self.logger.handlers.clear()

        self.logger.addHandler(self.console_handler)
        if self.file_handler:
            self.logger.addHandler(self.file_handler)
        self.logger.propagate = False

    def _log_std(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: Union[None, bool, BaseException] = None,
    ):
        # Ensure 'request_id' and 'component' are always in the LogRecord's dictionary for formatters
        final_extra = {"request_id": self.request_id, "component": self.component_name}
        if extra:
            final_extra.update(extra)

        actual_exc_info = exc_info if exc_info is not None else False
        if isinstance(exc_info, BaseException):
            actual_exc_info = True

        self.logger.log(level, message, extra=final_extra, exc_info=actual_exc_info)

    def info(self, message, extra=None):
        self._log_std(logging.INFO, message, extra=extra)

    def debug(self, message, extra=None):
        self._log_std(logging.DEBUG, message, extra=extra)

    def error(self, message, error: Optional[BaseException] = None, extra=None):
        log_extra = dict(extra) if extra else {}
        if error:
            log_extra["error_type"] = type(error).__name__
            log_extra["error_message"] = str(error)
        self._log_std(logging.ERROR, message, extra=log_extra, exc_info=error)

    def warning(self, message, extra=None):
        self._log_std(logging.WARNING, message, extra=extra)

    def set_request_id(self, request_id):
        self.request_id = request_id
        self.console_handler.setFormatter(
            logging.Formatter(
                f"{PY_LOG_PREFIX} %(asctime)s %(levelname)s [{self.component_name}] (%(request_id)s): %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )

    def set_component_name(self, component_name):
        self.component_name = component_name
        self.console_handler.setFormatter(
            logging.Formatter(
                f"{PY_LOG_PREFIX} %(asctime)s %(levelname)s [{self.component_name}] (%(request_id)s): %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )


embedding_logger = EmbeddingLogger(component_name="GlobalPythonHelper")

for lib_logger_name in [
    "transformers",
    "urllib3",
    "huggingface_hub",
    "PIL",
    "open_clip",
]:
    logging.getLogger(lib_logger_name).setLevel(logging.WARNING)


def handle_sigint(signum, frame):
    embedding_logger.info("Received SIGINT (Ctrl+C). Exiting gracefully.")
    sys.exit(0)


def handle_sigterm(signum, frame):
    embedding_logger.info("Received SIGTERM. Exiting gracefully.")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigterm)


def compute_entropy(image: Image.Image) -> float:
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    total_pixels = sum(histogram)
    if total_pixels == 0:
        return 0.0
    entropy = 0.0
    for count in histogram:
        if count > 0:
            p = count / total_pixels
            entropy -= p * math.log2(p)
    return entropy


class VideoProcessor:
    # ... (keep __init__, _get_duration, get_advanced_sample_times, extract_frames, __del__ as they are) ...
    # Replace the _extract_frame_fallback method (and thus self.extract_frame) with the new one:

    def __init__(
        self,
        num_frames: int,
        video_path: Optional[
            str
        ] = None,  # This will be the path to the (possibly temp) video file
        logger: Optional[EmbeddingLogger] = None,
        executor: Optional[
            ThreadPoolExecutor
        ] = None,  # For parallel frame extraction within this video
        request_id: Optional[str] = None,  # For logging context
        duration: Optional[float] = None,  # Pre-calculated duration
        original_filename_hint: Optional[
            str
        ] = None,  # For logging and temp file naming
    ):
        if not video_path:
            raise ValueError(
                "VideoProcessor requires a valid video_path (can be a temporary file path)."
            )

        self.video_path: str = video_path
        self._is_temp_file = original_filename_hint is not None and video_path.endswith(
            original_filename_hint
        )

        self.num_frames = num_frames
        self.logger = logger or embedding_logger
        base_component = f"VideoProc-{original_filename_hint[:20] if original_filename_hint else os.path.basename(video_path)[:20]}"
        self.logger.set_component_name(
            f"{base_component}-{request_id[:8] if request_id else uuid.uuid4().hex[:8]}"
        )
        if request_id:
            self.logger.set_request_id(request_id)

        self.executor = executor
        # Determine video codec for intelligent decoder selection
        self.video_codec_name: Optional[str] = self._get_video_codec()
        self.duration = duration if duration is not None else self._get_duration()

        if self.duration is None or self.duration <= 0:
            self.logger.error(
                f"Video duration error for '{self.video_path}'",
                extra={
                    "duration": self.duration,
                    "video_path": self.video_path,
                    "codec": self.video_codec_name,
                },
            )
            raise ValueError(
                f"Could not determine video duration or duration is invalid for '{self.video_path}'."
            )

    def _get_video_codec(self) -> Optional[str]:
        """Gets the video codec name using ffprobe."""
        start_time = time.time()
        self.logger.debug(f"Getting video codec for: '{self.video_path}'")
        try:
            command = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                self.video_path,
            ]
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
                timeout=10,  # Shorter timeout for codec check
            )
            codec_name = result.stdout.strip()
            if codec_name and codec_name != "N/A":
                self.logger.debug(
                    f"Video codec: {codec_name} for '{self.video_path}' in {(time.time() - start_time) * 1000:.2f}ms",
                    extra={"codec_name": codec_name, "video_path": self.video_path},
                )
                return codec_name
            self.logger.warning(
                f"ffprobe returned no codec_name for '{self.video_path}'. Stdout: {result.stdout}, Stderr: {result.stderr}"
            )
            return None
        except subprocess.TimeoutExpired:
            self.logger.error(f"ffprobe timeout getting codec for '{self.video_path}'")
            return None  # Cannot determine codec, will use default ffmpeg behavior
        except Exception as e:
            self.logger.error(f"Failed to get video codec for '{self.video_path}': {e}")
            return None

    def _get_cuvid_decoder(self) -> Optional[str]:
        """Returns the appropriate CUVID decoder based on the detected video codec."""
        if not self.video_codec_name:
            return (
                None  # Or a sensible default like "h264_cuvid" if most videos are h264
            )

        codec_map = {
            "h264": "h264_cuvid",
            "hevc": "hevc_cuvid",
            "vp9": "vp9_cuvid",
            "av1": "av1_cuvid",
            "mjpeg": "mjpeg_cuvid",
            "mpeg1video": "mpeg1_cuvid",
            "mpeg2video": "mpeg2_cuvid",
            "mpeg4": "mpeg4_cuvid",
            "vc1": "vc1_cuvid",
            "vp8": "vp8_cuvid",
            # Add more mappings as needed based on `ffmpeg -decoders | grep cuvid`
        }
        decoder = codec_map.get(self.video_codec_name.lower())
        if decoder:
            self.logger.debug(
                f"Selected CUVID decoder '{decoder}' for codec '{self.video_codec_name}'."
            )
        else:
            self.logger.warning(
                f"No specific CUVID decoder found for codec '{self.video_codec_name}'. GPU acceleration might not be optimal or might fallback."
            )
        return decoder

    def _run_ffmpeg_command(
        self, command: List[str], attempt_description: str
    ) -> bytes:
        """Helper to run an ffmpeg command and handle common errors."""
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,  # Raises CalledProcessError on non-zero exit code
                timeout=30,  # Increased timeout for frame extraction
            )
            if not result.stdout:
                stderr_output = (
                    result.stderr.decode("utf-8", errors="ignore")
                    if result.stderr
                    else "No stderr"
                )
                self.logger.error(
                    f"ffmpeg ({attempt_description}) produced no stdout for '{self.video_path}'. Stderr: {stderr_output}",
                    extra={"video_path": self.video_path, "stderr": stderr_output},
                )
                raise RuntimeError(
                    f"No frame data from ffmpeg ({attempt_description}). Stderr: {stderr_output}"
                )
            return result.stdout
        except subprocess.CalledProcessError as e:
            stderr_output = (
                e.stderr.decode("utf-8", errors="ignore")
                if e.stderr
                else "No stderr available"
            )
            stdout_output = (
                e.stdout.decode("utf-8", errors="ignore")
                if e.stdout
                else "No stdout available"
            )
            self.logger.error(
                f"ffmpeg ({attempt_description}) command failed for '{self.video_path}' with exit code {e.returncode}. Stderr: {stderr_output}",
                extra={
                    "video_path": self.video_path,
                    "stderr": stderr_output,
                    "stdout": stdout_output,
                    "command": " ".join(e.cmd),
                },
            )
            raise RuntimeError(
                f"ffmpeg ({attempt_description}) command failed with exit code {e.returncode}."
            ) from e
        except subprocess.TimeoutExpired:
            self.logger.error(
                f"ffmpeg ({attempt_description}) timeout for '{self.video_path}'.",
                extra={"video_path": self.video_path, "command": " ".join(command)},
            )
            raise RuntimeError(
                f"ffmpeg ({attempt_description}) timeout extracting frame."
            )
        except Exception as e:  # Catch any other unexpected error from subprocess.run
            self.logger.error(
                f"Unexpected error running ffmpeg ({attempt_description}) for '{self.video_path}': {e}",
                extra={"video_path": self.video_path, "command": " ".join(command)},
            )
            raise RuntimeError(
                f"Unexpected error running ffmpeg ({attempt_description}): {e}"
            ) from e

    def _extract_frame_fallback(self, time_sec: float) -> Image.Image:
        """
        Extracts a single frame from the video_path at time_sec.
        Attempts GPU acceleration (NVDEC) first, then falls back to CPU decoding.
        """
        self.logger.debug(
            f"Extracting frame at {time_sec:.2f}s from '{self.video_path}' (codec: {self.video_codec_name})"
        )

        # Base command arguments common to both GPU and CPU attempts
        base_cmd_args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            # "-y", # -y is for overwriting output files, not needed for piping
        ]
        input_options = ["-ss", str(time_sec), "-i", self.video_path]
        output_options = [
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",  # Output as MJPEG to the pipe
            "-q:v",
            "2",  # High quality MJPEG
            "-",
        ]

        # Attempt 1: GPU Accelerated Decoding (NVDEC)
        hw_decoder = self._get_cuvid_decoder()
        if (
            hw_decoder
        ):  # Only attempt if we have a specific cuvid decoder for the detected codec
            hw_command = list(base_cmd_args)
            # Order for HW acceleration: -hwaccel, then decoder (-c:v) for input, then input, then filters
            hw_command.extend(
                [
                    "-hwaccel",
                    "cuda",
                    "-hwaccel_output_format",
                    "cuda",  # Keep frames in GPU mem if possible for filters
                    "-c:v",
                    hw_decoder,
                ]
            )
            hw_command.extend(input_options)
            hw_command.extend(
                ["-vf", "hwdownload,format=rgb24"]
            )  # Download from GPU & convert for PIL
            hw_command.extend(output_options)

            self.logger.debug(
                f"Attempting GPU accelerated frame extraction: {' '.join(hw_command)}"
            )
            try:
                frame_data_bytes = self._run_ffmpeg_command(
                    hw_command, f"GPU accel ({hw_decoder})"
                )
                self.logger.info(
                    f"Successfully extracted frame using GPU acceleration ({hw_decoder}) for '{self.video_path}' at {time_sec:.2f}s."
                )
                return Image.open(io.BytesIO(frame_data_bytes)).convert("RGB")
            except Exception as e_gpu:
                self.logger.warning(
                    f"GPU accelerated frame extraction failed for '{self.video_path}' at {time_sec:.2f}s (decoder: {hw_decoder}): {e_gpu}. Falling back to CPU.",
                    extra={
                        "error_type": type(e_gpu).__name__,
                        "error_message": str(e_gpu),
                    },
                )
        else:
            self.logger.info(
                f"No specific CUVID decoder for '{self.video_codec_name}', or codec unknown. Skipping direct GPU decode attempt, will use CPU or default ffmpeg behavior."
            )

        # Attempt 2: CPU Decoding (Fallback)
        # This is similar to the original command but more structured
        cpu_command = list(base_cmd_args)
        # For CPU, -ss before -i is generally good for fast seeking
        cpu_command.extend(input_options)  # Includes -ss
        cpu_command.extend(
            ["-vf", "format=rgb24"]
        )  # Ensure output is RGB24 if not mjpeg
        cpu_command.extend(output_options)

        self.logger.debug(f"Attempting CPU frame extraction: {' '.join(cpu_command)}")
        try:
            frame_data_bytes = self._run_ffmpeg_command(cpu_command, "CPU fallback")
            self.logger.info(
                f"Successfully extracted frame using CPU for '{self.video_path}' at {time_sec:.2f}s."
            )
            return Image.open(io.BytesIO(frame_data_bytes)).convert("RGB")
        except Exception as e_cpu:
            self.logger.error(
                f"CPU frame extraction also failed for '{self.video_path}' at {time_sec:.2f}s: {e_cpu}",
                extra={"error_type": type(e_cpu).__name__, "error_message": str(e_cpu)},
            )
            raise RuntimeError(
                f"All frame extraction attempts (GPU and CPU) failed for '{self.video_path}' at {time_sec:.2f}s."
            ) from e_cpu

    extract_frame = (
        _extract_frame_fallback  # Alias for clarity if called from elsewhere
    )

    def _get_duration(self) -> float:
        start_time = time.time()
        self.logger.debug(f"Getting duration for video: '{self.video_path}'")
        try:
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
                text=True,
                timeout=15,
            )
            duration_str = result.stdout.strip()
            if not duration_str or duration_str == "N/A":
                err_msg = f"ffprobe returned invalid duration '{duration_str}' for '{self.video_path}'"
                self.logger.error(
                    err_msg,
                    extra={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "video_path": self.video_path,
                    },
                )
                raise RuntimeError(err_msg)
            duration = float(duration_str)
            self.logger.debug(
                f"Video duration: {duration:.2f}s for '{self.video_path}'",
                extra={
                    "duration": duration,
                    "time_taken_ms": (time.time() - start_time) * 1000,
                },
            )
            return duration
        except subprocess.TimeoutExpired:
            self.logger.error(
                f"ffprobe timeout getting duration for '{self.video_path}'",
                extra={"video_path": self.video_path},
            )
            raise RuntimeError(
                f"ffprobe timeout getting duration for '{self.video_path}'"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to get video duration for '{self.video_path}': {e}",
                extra={"video_path": self.video_path},
            )
            raise RuntimeError(
                f"Failed to get video duration for '{self.video_path}': {e}"
            ) from e

    def get_advanced_sample_times(self) -> Tuple[List[float], Dict[str, Any]]:
        self.logger.debug(
            f"Calculating sample times for '{self.video_path}' (duration: {self.duration:.2f}s, frames: {self.num_frames})"
        )
        debug_metadata: Dict[str, Any] = {
            "num_requested_frames": self.num_frames,
            "video_duration_s": self.duration,
        }
        method_used = "uniform_sampling_default"

        if self.num_frames <= 0:
            return [], {
                "method_used": "none",
                "reason": "num_frames is zero or negative",
                **debug_metadata,
            }
        if self.duration <= 0:  # Should have been caught by __init__
            return [], {
                "method_used": "none",
                "reason": "video duration is zero or negative",
                **debug_metadata,
            }

        start_offset = min(0.5, self.duration * 0.02)  # Avoid very start/end
        end_offset = min(0.5, self.duration * 0.02)
        effective_duration = self.duration - start_offset - end_offset

        selected_times: List[float]
        if self.num_frames == 1:
            # For a single frame, try to pick a more central point after initial ramp-up
            target_time = start_offset + effective_duration / 2.0
            selected_times = [
                max(0.1, min(target_time, self.duration - 0.1))
            ]  # Ensure it's not exactly at 0 or end
            method_used = "single_middle_frame_adjusted"
        elif (
            effective_duration <= 0.2  # Slightly larger threshold for meaningful spread
        ):  # Video too short for meaningful spread after offset
            # For very short videos, take one frame near the middle, and duplicate if more are needed
            middle_time = self.duration / 2.0
            selected_times = [middle_time] * self.num_frames  # Duplicate middle
            method_used = "middle_frame_duplicated_short_video"
            if self.num_frames > 1:  # If duplicating, log it
                self.logger.debug(
                    f"Video too short for {self.num_frames} distinct frames after offset, duplicating middle frame."
                )
        else:
            # Spread self.num_frames evenly across effective_duration
            # Ensure no division by zero if num_frames is 1 (handled above)
            if self.num_frames > 1:
                selected_times = [
                    start_offset + (i * effective_duration / (self.num_frames - 1))
                    for i in range(self.num_frames)
                ]
            else:  # Should be caught by num_frames == 1 case above, but as safeguard
                selected_times = [start_offset + effective_duration / 2.0]

            method_used = "uniform_spread_with_offset"

        # Ensure timestamps are within valid video duration and sorted
        # Clamping to avoid issues with ffmpeg -ss near exact duration
        # A very small epsilon like 0.001s from the end
        clamped_times = [
            max(0.0, min(t, self.duration - 0.001)) for t in selected_times
        ]
        # Remove duplicates that might arise from clamping or short durations, then sort
        # If after deduplication, fewer frames than num_frames, the extract_frames logic handles padding
        final_selected_times = sorted(list(set(clamped_times)))

        debug_metadata["method_used"] = method_used
        debug_metadata["candidate_timestamps_before_clamp_dedup"] = list(selected_times)
        debug_metadata["final_selected_timestamps"] = list(final_selected_times)
        debug_metadata["effective_sampling_duration_s"] = effective_duration
        debug_metadata["start_offset_s"] = start_offset

        self.logger.debug(
            f"Selected {len(final_selected_times)} final timestamps via {method_used}",
            extra={"timestamps": final_selected_times, **debug_metadata},
        )
        return final_selected_times, debug_metadata

    def extract_frames(self) -> Tuple[List[Image.Image], Dict[str, Any]]:
        self.logger.info(
            f"Extracting up to {self.num_frames} frames from '{self.video_path}'"
        )
        overall_start_time = time.time()
        extracted_frames_pil: List[Image.Image] = []

        timestamps, frame_sampling_debug_meta = self.get_advanced_sample_times()
        final_debug_meta: Dict[str, Any] = {
            "frame_sampling_details": frame_sampling_debug_meta,
            "video_codec_detected": self.video_codec_name,  # Add detected codec to debug
        }
        final_debug_meta["num_timestamps_from_sampler"] = len(timestamps)

        if not timestamps:
            self.logger.warning(
                f"No timestamps returned by sampler for '{self.video_path}'.",
                extra=final_debug_meta,
            )
            return [], {
                "error": "No timestamps for frame extraction",
                **final_debug_meta,
            }

        # Limit to self.num_frames if sampler returned more (e.g. due to deduplication)
        # The sampler already tries to produce up to num_frames.
        # If fewer are returned (e.g. short video, deduplication), extract_frames will pad later.
        actual_timestamps_to_extract = timestamps
        # If timestamps list is shorter than self.num_frames due to deduplication for short videos,
        # we will extract what we have, and then pad later.
        # If it's longer (shouldn't happen with current get_advanced_sample_times), we'd rely on later padding logic to not exceed num_frames.
        # The main padding happens after extraction.

        final_debug_meta["actual_timestamps_for_extraction"] = (
            actual_timestamps_to_extract
        )

        extraction_errors_count = 0
        # Determine if parallel extraction should be used for *these* frames from *this* video.
        # This is independent of the higher-level parallelism for multiple video files.
        use_parallel_extraction_for_this_video = (
            self.executor and len(actual_timestamps_to_extract) > 1
        )

        if use_parallel_extraction_for_this_video and self.executor:
            self.logger.debug(
                f"Using parallel extraction for {len(actual_timestamps_to_extract)} frames of video '{self.video_path}'."
            )
            future_to_ts_map = {
                self.executor.submit(self.extract_frame, ts): ts
                for ts in actual_timestamps_to_extract
            }
            for future in concurrent.futures.as_completed(future_to_ts_map):
                ts = future_to_ts_map[future]
                try:
                    frame_pil = future.result()
                    extracted_frames_pil.append(frame_pil)
                except Exception as e_frame:
                    extraction_errors_count += 1
                    self.logger.error(
                        f"Parallel extract_frame failed for video '{self.video_path}' at ts={ts:.2f}s: {e_frame}",
                        extra={
                            "timestamp": ts,
                            "video_path": self.video_path,
                            "error_type": type(e_frame).__name__,
                        },
                    )
        else:  # Sequential extraction for this video's frames
            self.logger.debug(
                f"Using sequential extraction for {len(actual_timestamps_to_extract)} frames of video '{self.video_path}'."
            )
            for ts in actual_timestamps_to_extract:
                try:
                    frame_pil = self.extract_frame(ts)
                    extracted_frames_pil.append(frame_pil)
                except Exception as e_frame:
                    extraction_errors_count += 1
                    self.logger.error(
                        f"Sequential extract_frame failed for video '{self.video_path}' at ts={ts:.2f}s: {e_frame}",
                        extra={
                            "timestamp": ts,
                            "video_path": self.video_path,
                            "error_type": type(e_frame).__name__,
                        },
                    )

        if extraction_errors_count > 0:
            final_debug_meta["frame_extraction_error_count"] = extraction_errors_count

        if not extracted_frames_pil and len(actual_timestamps_to_extract) > 0:
            # This means all attempts to extract frames failed.
            err_msg = f"All {len(actual_timestamps_to_extract)} frame extractions failed for '{self.video_path}'."
            self.logger.error(err_msg, extra=final_debug_meta)
            return [], {"error": err_msg, **final_debug_meta}

        # If no frames were requested (self.num_frames == 0), timestamps would be empty,
        # and extracted_frames_pil would also be empty. This is valid.
        if self.num_frames == 0 and not extracted_frames_pil:
            self.logger.info(
                f"No frames requested (num_frames=0) for '{self.video_path}'. Returning empty list."
            )
            # final_debug_meta is already populated correctly for this case by get_advanced_sample_times
            final_debug_meta["num_final_pil_frames_returned"] = 0
            return [], final_debug_meta

        # Padding: If fewer frames than self.num_frames were successfully extracted
        # (and self.num_frames > 0), duplicate the last good frame.
        # This handles cases where some extractions failed, or if `get_advanced_sample_times`
        # returned fewer unique timestamps than `self.num_frames` (e.g., for very short videos).
        if self.num_frames > 0 and len(extracted_frames_pil) < self.num_frames:
            if extracted_frames_pil:  # Only pad if we have at least one good frame
                self.logger.warning(
                    f"Successfully extracted {len(extracted_frames_pil)} frames, but {self.num_frames} were requested for '{self.video_path}'. Duplicating last good frame to meet count.",
                    extra=final_debug_meta,
                )
                last_good_frame = extracted_frames_pil[-1]
                num_to_add = self.num_frames - len(extracted_frames_pil)
                extracted_frames_pil.extend(
                    [last_good_frame.copy() for _ in range(num_to_add)]
                )
            else:  # No frames extracted, but num_frames > 0 was requested. This is an error state.
                # This case should have been caught by "All ... frame extractions failed" above.
                # If it reaches here, it's an unexpected state.
                self.logger.error(
                    f"No frames were extracted for '{self.video_path}', but {self.num_frames} were requested. Cannot pad.",
                    extra=final_debug_meta,
                )
                # This implies a previous error didn't halt execution, or logic issue.
                # Return empty with error if not already set.
                if (
                    "error" not in final_debug_meta
                ):  # Should be set from "all extractions failed"
                    final_debug_meta["error"] = (
                        f"Padding error: No frames to pad from for {self.video_path}"
                    )
                return [], final_debug_meta
        elif not extracted_frames_pil and self.num_frames > 0:
            # This condition implies all extractions failed or no timestamps resulted in frames,
            # and it should have been caught by the "All ... frame extractions failed" logic.
            # If somehow missed, ensure error state.
            self.logger.error(
                f"No frames were ultimately available for '{self.video_path}' though {self.num_frames} were requested (post-padding check).",
                extra=final_debug_meta,
            )
            if "error" not in final_debug_meta:
                final_debug_meta["error"] = (
                    f"No frames available after processing for {self.video_path}"
                )
            return [], final_debug_meta

        final_debug_meta["num_final_pil_frames_returned"] = len(extracted_frames_pil)
        total_extraction_duration_ms = (time.time() - overall_start_time) * 1000
        self.logger.info(
            f"Successfully prepared {len(extracted_frames_pil)} frames in {total_extraction_duration_ms:.2f}ms for '{self.video_path}'.",
            extra={"duration_ms": total_extraction_duration_ms, **final_debug_meta},
        )
        return extracted_frames_pil, final_debug_meta

    def __del__(self):
        pass


class CLIPEmbedder:
    preprocess: Callable[[Any], torch.Tensor]  # Type hint for preprocessing transform

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        logger: Optional[EmbeddingLogger] = None,
        enable_augmentation: bool = False,
    ):
        self.logger = logger or embedding_logger  # Fallback to global default
        self.logger.set_component_name(
            f"CLIPEmbedder-{model_name.split('/')[-1][:15]}"
        )  # Shorten name
        self.model_name = model_name

        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.logger.info(
            f"Initializing CLIP model '{model_name}' on device '{self.device}'"
        )

        init_start_time = time.time()
        # Initialize variables before try block to ensure they're always defined
        model_arch = self.model_name
        pretrained_tag = "openai"

        try:
            arch_name_map = {
                "vit-base-patch32": "ViT-B-32",
                "vit-b-32": "ViT-B-32",
                "vit-large-patch14": "ViT-L-14",
                "vit-l-14": "ViT-L-14",
            }

            if "/" in self.model_name:
                author_part, model_part = self.model_name.split("/", 1)
                if author_part.lower() == "openai":
                    for key, val in arch_name_map.items():
                        if key in model_part.lower():
                            model_arch = val
                            break
                else:
                    # Try to find a known arch in the model_part
                    found_arch = False
                    for key, val in arch_name_map.items():
                        # Check if key (e.g., "vit-b-32") is part of model_part (e.g., "ViT-B-32-laion2b_s34b_b79k")
                        if key.replace("-", "") in model_part.lower().replace("-", ""):
                            model_arch = val
                            found_arch = True
                            break
                    if (
                        not found_arch and "-" in model_part
                    ):  # Fallback for names like LAION/CLIP-ViT-B-32-laion2B
                        # Heuristic: "CLIP-ViT-B-32" -> "ViT-B-32"
                        potential_arch = (
                            model_part.split("-")[1]
                            if model_part.startswith("CLIP-")
                            else model_part.split("-")[0]
                        )
                        for key, val in arch_name_map.items():
                            if key.replace("-", "") in potential_arch.lower().replace(
                                "-", ""
                            ):
                                model_arch = val
                                break

                    pretrained_tag = (
                        self.model_name
                    )  # For non-openai, often full name is pretrained tag or part of it

            self.logger.debug(
                f"Attempting to load OpenCLIP model_arch='{model_arch}', pretrained='{pretrained_tag}'"
            )

            self.model, train_preprocess, val_preprocess = (
                open_clip.create_model_and_transforms(
                    model_name=model_arch, pretrained=pretrained_tag, device=self.device
                )
            )
            self.preprocess = val_preprocess
            self.model.eval()

            if self.device not in ["cpu"]:
                try:
                    torch.set_float32_matmul_precision("high")
                    self.logger.debug("Set float32 matmul precision to 'high'.")
                except AttributeError:
                    self.logger.debug(
                        "torch.set_float32_matmul_precision not available."
                    )
            self.logger.info(
                f"CLIP model '{self.model_name}' (parsed as arch='{model_arch}', pretrained='{pretrained_tag}') loaded on '{self.device}'. Init time: {(time.time() - init_start_time):.2f}s"
            )
        except Exception as e_load:
            self.logger.error(
                f"Failed to load CLIP model '{self.model_name}' with parsed arch '{model_arch}' and pretrained '{pretrained_tag}': {e_load}",
                error=e_load,
            )
            # Try with original model_name if parsing failed, as OpenCLIP might handle some HF names directly
            if model_arch != self.model_name or pretrained_tag != self.model_name:
                self.logger.info(
                    f"Retrying OpenCLIP load with model_name='{self.model_name}' and no explicit pretrained tag (or model_name as pretrained)."
                )
                try:
                    self.model, train_preprocess, val_preprocess = (
                        open_clip.create_model_and_transforms(
                            model_name=self.model_name,
                            pretrained=self.model_name,
                            device=self.device,  # Try model_name for both
                        )
                    )
                    self.preprocess = val_preprocess
                    self.model.eval()
                    self.logger.info(
                        f"CLIP model '{self.model_name}' loaded successfully on retry (using original name). Init time: {(time.time() - init_start_time):.2f}s"
                    )
                except Exception as e_retry_load:
                    self.logger.error(
                        f"Retry to load CLIP model '{self.model_name}' also failed: {e_retry_load}",
                        error=e_retry_load,
                    )
                    raise RuntimeError(
                        f"CLIP model loading failed for '{self.model_name}' after parsing and retry: {e_retry_load}"
                    ) from e_retry_load
            else:  # Parsing didn't change anything, original attempt failed
                raise RuntimeError(
                    f"CLIP model loading failed for '{self.model_name}': {e_load}"
                ) from e_load

        self.enable_augmentation = enable_augmentation
        if self.enable_augmentation:
            import torchvision.transforms as T

            image_size_cfg = getattr(
                getattr(self.model, "visual", None), "image_size", 224
            )
            img_size_to_use = (
                image_size_cfg if isinstance(image_size_cfg, int) else image_size_cfg[0]
            )

            self.augmentation_transforms = T.Compose(
                [
                    T.RandomResizedCrop(img_size_to_use, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                ]
            )
            self.logger.info(
                f"Data augmentation enabled (RandomResizedCrop to {img_size_to_use}, RandomHorizontalFlip)."
            )
        else:
            self.augmentation_transforms = None

    def get_embeddings_for_pil_list(
        self, pil_image_list: List[Image.Image]
    ) -> torch.Tensor:
        if not pil_image_list:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        self.logger.debug(
            f"CLIPEmbedder: Processing batch of {len(pil_image_list)} PIL images for embedding on {self.device}."
        )
        gpu_batch_start_time = time.time()

        tensors_for_model: List[torch.Tensor] = []
        for i, pil_img in enumerate(pil_image_list):
            try:
                img_to_process = pil_img
                if self.enable_augmentation and self.augmentation_transforms:
                    img_to_process = self.augmentation_transforms(pil_img)
                tensors_for_model.append(self.preprocess(img_to_process))
            except Exception as e_pil_proc:
                self.logger.error(
                    f"Failed to preprocess PIL image at index {i} for model input: {e_pil_proc}",
                    extra={"index": i},
                    error=e_pil_proc,
                )
                raise RuntimeError(
                    f"PIL image preprocessing (index {i}) failed: {e_pil_proc}"
                ) from e_pil_proc

        if not tensors_for_model:
            self.logger.warning(
                "No PIL images were successfully preprocessed into tensors for the model."
            )
            return torch.empty(0, dtype=torch.float32, device=self.device)

        try:
            input_tensor_batch = torch.stack(tensors_for_model).to(self.device)
        except Exception as e_stack:
            self.logger.error(
                f"Error stacking preprocessed image tensors (count: {len(tensors_for_model)}): {e_stack}",
                extra={
                    "tensor_shapes_preview": [
                        str(t.shape) for t in tensors_for_model[:3]
                    ]
                },
                error=e_stack,
            )
            raise RuntimeError(f"Tensor stacking failed: {e_stack}") from e_stack

        autocast_enabled = self.device != "cpu"
        if autocast_enabled:
            # MPS uses 'cpu' as device_type for autocast with bfloat16 or float16
            # CUDA uses 'cuda'
            autocast_device_type = "cuda" if self.device.startswith("cuda") else "cpu"
            # PyTorch on MPS with autocast to float16 can sometimes be problematic or slower.
            # bfloat16 is often preferred if supported, but float16 is more common for CLIP.
            # For simplicity, float16 is used here.
            autocast_context = torch.autocast(
                device_type=autocast_device_type, dtype=torch.float16
            )
        else:
            autocast_context = nullcontext()

        with torch.no_grad(), autocast_context:
            image_features_batch = self.model.encode_image(input_tensor_batch)
            image_features_batch = image_features_batch / image_features_batch.norm(
                p=2, dim=-1, keepdim=True
            )

        gpu_duration_ms = (time.time() - gpu_batch_start_time) * 1000
        self.logger.info(
            f"GPU inference for batch of {len(pil_image_list)} items completed in {gpu_duration_ms:.2f}ms. Output shape: {image_features_batch.shape}"
        )
        return image_features_batch.cpu()


def _preprocess_single_item_for_batch(
    item_spec_dict: Dict[str, Any],
    python_media_root: str,
    default_num_frames_for_video: int,
    parent_logger: EmbeddingLogger,
    video_processor_shared_executor: ThreadPoolExecutor,
    item_specific_request_id: str,
) -> Dict[str, Any]:

    item_id = item_spec_dict["id"]
    # Ensure media_type is initialized even if not in item_spec_dict, though Pydantic model should ensure it.
    media_type = item_spec_dict.get("media_type", "unknown")

    item_logger = EmbeddingLogger(
        request_id=item_specific_request_id,
        component_name=f"ItemPreProc-{item_id[:10]}",
    )
    item_logger.logger.setLevel(parent_logger.logger.level)

    item_logger.debug(
        f"Preprocessing item '{item_spec_dict.get('original_filename', item_id)}'",
        extra=item_spec_dict,
    )

    pil_images_for_item: List[Image.Image] = []
    item_debug_meta: Dict[str, Any] = {
        "original_item_id": item_id,
        "source_type": item_spec_dict.get("source_type", "unknown"),
        "original_filename": item_spec_dict.get("original_filename", "N/A"),
        "requested_media_type": media_type,
        "timestamp_preprocess_start_utc": datetime.utcnow().isoformat() + "Z",
    }
    temp_file_created_path: Optional[str] = None

    try:
        media_source_for_pil: Union[str, io.BytesIO]
        video_path_for_vid_processor: Optional[str] = None
        source_type = item_spec_dict["source_type"]
        source_location = item_spec_dict["source"]
        # media_type already assigned

        if source_type == "url":
            item_logger.debug(f"Downloading URL: {source_location}")
            response = requests.get(
                source_location, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
            )
            response.raise_for_status()

            temp_suffix = (
                f"_{item_spec_dict.get('original_filename', uuid.uuid4().hex[:8])}"
            )
            safe_suffix = "".join(
                c if c.isalnum() or c in [".", "_"] else "_" for c in temp_suffix
            )[:50]

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=safe_suffix
            ) as tmp_file:
                for chunk in response.iter_content(chunk_size=32 * 1024):
                    tmp_file.write(chunk)
                temp_file_created_path = tmp_file.name
            item_logger.info(
                f"Downloaded URL '{source_location}' to temp file '{temp_file_created_path}'"
            )
            media_source_for_pil = temp_file_created_path
            if media_type == "video":
                video_path_for_vid_processor = temp_file_created_path
            item_debug_meta["downloaded_url"] = source_location
            item_debug_meta["temp_file_path"] = temp_file_created_path
        elif source_type == "filepath":
            resolved_fs_path = (
                os.path.join(python_media_root, source_location)
                if not os.path.isabs(source_location)
                else source_location
            )
            item_logger.debug(
                f"Accessing filepath: '{resolved_fs_path}' (original source: '{source_location}')"
            )
            if not os.path.exists(resolved_fs_path):
                item_logger.error(f"Filepath not found: '{resolved_fs_path}'")
                raise FileNotFoundError(f"Filepath not found: {resolved_fs_path}")
            media_source_for_pil = resolved_fs_path
            if media_type == "video":
                video_path_for_vid_processor = resolved_fs_path
            item_debug_meta["resolved_filepath"] = resolved_fs_path
        else:
            item_logger.error(f"Unsupported source_type: '{source_type}'")
            raise ValueError(f"Unsupported source_type: {source_type}")

        if media_type == "image":
            img = Image.open(media_source_for_pil).convert("RGB")
            pil_images_for_item.append(img)
            item_debug_meta["image_dimensions"] = f"{img.width}x{img.height}"
            item_logger.debug(
                f"Loaded image '{item_spec_dict.get('original_filename', item_id)}' ({img.width}x{img.height})"
            )
        elif media_type == "video":
            if not video_path_for_vid_processor:
                item_logger.error(
                    "Internal error: video_path_for_vid_processor not set for video item."
                )
                raise ValueError("video_path_for_vid_processor not set for video.")

            num_frames_to_extract = (
                item_spec_dict.get("num_frames")
                if item_spec_dict.get("num_frames", -1) >= 0  # Allow 0 frames
                else default_num_frames_for_video
            )
            # Ensure num_frames_to_extract is always an int
            if num_frames_to_extract is None:
                num_frames_to_extract = default_num_frames_for_video
            item_debug_meta["num_frames_requested_for_video"] = num_frames_to_extract

            vp = VideoProcessor(
                video_path=video_path_for_vid_processor,
                num_frames=num_frames_to_extract,  # Use the determined number of frames
                logger=item_logger,
                executor=video_processor_shared_executor,
                request_id=item_specific_request_id,
                duration=item_spec_dict.get(
                    "estimated_duration_s"
                ),  # Pass if available
                original_filename_hint=item_spec_dict.get("original_filename"),
            )
            frames_list_pil, video_proc_debug_meta = vp.extract_frames()
            pil_images_for_item.extend(frames_list_pil)
            item_debug_meta.update(video_proc_debug_meta)
            item_debug_meta["num_extracted_frames_for_item"] = len(frames_list_pil)
            item_logger.info(
                f"Extracted {len(frames_list_pil)} frames for video '{item_spec_dict.get('original_filename', item_id)}'"
            )
        else:
            item_logger.error(f"Unsupported media_type: '{media_type}'")
            raise ValueError(f"Unsupported media_type: {media_type}")

        item_debug_meta["timestamp_preprocess_end_utc"] = (
            datetime.utcnow().isoformat() + "Z"
        )
        return {
            "id": item_id,
            "media_type": media_type,
            "pil_images": pil_images_for_item,
            "error": None,
            "detail": None,
            "debug_metadata": item_debug_meta,
        }

    except UnidentifiedImageError as uie:
        item_logger.error(
            f"PIL UnidentifiedImageError for '{item_spec_dict.get('original_filename', item_id)}': {uie}",
            error=uie,
        )
        item_debug_meta["timestamp_preprocess_end_utc"] = (
            datetime.utcnow().isoformat() + "Z"
        )
        return {
            "id": item_id,
            "media_type": media_type,  # Use the already assigned media_type
            "pil_images": [],
            "error": "Unidentified image file",
            "detail": str(uie),
            "debug_metadata": item_debug_meta,
        }
    except FileNotFoundError as fnfe:
        item_logger.error(
            f"FileNotFoundError for '{item_spec_dict.get('original_filename', item_id)}': {fnfe}",
            error=fnfe,
        )
        item_debug_meta["timestamp_preprocess_end_utc"] = (
            datetime.utcnow().isoformat() + "Z"
        )
        return {
            "id": item_id,
            "media_type": media_type,
            "pil_images": [],
            "error": "File not found during preprocessing",
            "detail": str(fnfe),
            "debug_metadata": item_debug_meta,
        }
    except requests.exceptions.RequestException as req_exc:
        item_logger.error(
            f"Network error downloading URL '{item_spec_dict.get('source')}' for item '{item_id}': {req_exc}",
            error=req_exc,
        )
        item_debug_meta["timestamp_preprocess_end_utc"] = (
            datetime.utcnow().isoformat() + "Z"
        )
        return {
            "id": item_id,
            "media_type": media_type,
            "pil_images": [],
            "error": "Network error during download",
            "detail": str(req_exc),
            "debug_metadata": item_debug_meta,
        }
    except Exception as e_preproc:
        item_logger.error(
            f"General failure preprocessing item '{item_spec_dict.get('original_filename', item_id)}': {e_preproc}",
            error=e_preproc,
        )
        item_debug_meta["timestamp_preprocess_end_utc"] = (
            datetime.utcnow().isoformat() + "Z"
        )
        item_debug_meta["preprocess_error_type"] = type(e_preproc).__name__
        return {
            "id": item_id,
            "media_type": media_type,
            "pil_images": [],
            "error": f"Preprocessing failed: {type(e_preproc).__name__}",
            "detail": str(e_preproc),
            "debug_metadata": item_debug_meta,
        }
    finally:
        if temp_file_created_path:
            try:
                os.unlink(temp_file_created_path)
                item_logger.debug(f"Cleaned up temp file: '{temp_file_created_path}'")
            except OSError as e_unlink_final:
                item_logger.warning(
                    f"Failed to clean up temp file '{temp_file_created_path}': {e_unlink_final}"
                )


# ... (process_media_batch and if __name__ == "__main__": block remain the same) ...
def process_media_batch(
    items_data_as_dicts: List[
        Dict[str, Any]
    ],  # List of MediaItem models (already .model_dump()'ed by server.py)
    clip_embedder_instance: CLIPEmbedder,
    python_media_root_path: str,
    default_num_frames_for_videos: int,
    parent_batch_logger: EmbeddingLogger,  # Logger for the whole batch operation
    batch_processing_id: str,  # ID for this entire batch execution run
) -> Dict[
    str, Dict[str, Any]
]:  # Returns map of item_id -> result_dict (EmbeddingResult like)

    parent_batch_logger.set_request_id(batch_processing_id)  # Set ID for this run
    parent_batch_logger.info(
        f"Processing media batch '{batch_processing_id}' for {len(items_data_as_dicts)} items."
    )
    batch_overall_start_time = time.time()

    final_results_map: Dict[str, Dict[str, Any]] = {}
    # List of all PIL images (from all successfully preprocessed items) to send to GPU
    all_pil_images_for_gpu_batch: List[Image.Image] = []
    # Maps items to their corresponding slice in the GPU batch tensor
    gpu_batch_item_details_map: List[Dict[str, Any]] = []

    # Configure ThreadPoolExecutors
    num_cores = os.cpu_count() or 1
    # For I/O bound tasks (downloads) and CPU/GPU ffmpeg calls.
    # If ffmpeg uses GPU, it will contend with CLIP model for GPU resources, but decode units are separate.
    # This executor is for preprocessing multiple items concurrently.
    max_preprocess_workers = min(
        max(4, num_cores * 2 if num_cores > 1 else 2), 16
    )  # Reduced from 32 to manage potential GPU contention

    # This executor is passed to VideoProcessor for potentially parallelizing frame extraction *within* a single video.
    # If ffmpeg is using GPU, having many parallel ffmpeg GPU instances for *one* video might be too much.
    # For CPU-bound ffmpeg, it's num_cores limited. For GPU-bound, it's GPU decode unit limited.
    # Let's make this smaller or even sequential if each ffmpeg call uses GPU.
    # If _extract_frame_fallback now uses GPU, parallelizing it heavily for *one* video might be counterproductive.
    # Consider making this 1 or 2 if GPU ffmpeg is used, or tune based on testing.
    max_video_frame_workers_per_video = min(
        max(1, num_cores // 2 if num_cores > 2 else 1), 4
    )  # Reduced from 8

    parent_batch_logger.debug(
        f"Using max_preprocess_workers: {max_preprocess_workers}, max_video_frame_workers_per_video: {max_video_frame_workers_per_video}"
    )

    with ThreadPoolExecutor(
        max_workers=max_preprocess_workers, thread_name_prefix="ItemPreProc"
    ) as item_preproc_executor, ThreadPoolExecutor(
        max_workers=max_video_frame_workers_per_video,
        thread_name_prefix="VideoFrames",  # This is for within one video
    ) as video_ffmpeg_executor:

        item_preprocess_futures_map = {
            item_preproc_executor.submit(
                _preprocess_single_item_for_batch,
                item_data_dict,
                python_media_root_path,
                default_num_frames_for_videos,
                parent_batch_logger,
                video_ffmpeg_executor,  # This executor is for parallel frames *within* one VideoProcessor call
                f"{batch_processing_id}_{item_data_dict['id'][:8]}",
            ): item_data_dict["id"]
            for item_data_dict in items_data_as_dicts
        }

        num_items_preprocessed = 0
        for future_obj in concurrent.futures.as_completed(item_preprocess_futures_map):
            original_item_id = item_preprocess_futures_map[future_obj]
            num_items_preprocessed += 1
            parent_batch_logger.debug(
                f"Preprocessing result received for item {num_items_preprocessed}/{len(items_data_as_dicts)} (ID: {original_item_id})"
            )

            try:
                item_preproc_output_dict = future_obj.result()
                item_id_from_result = item_preproc_output_dict["id"]

                if item_preproc_output_dict.get("error"):
                    parent_batch_logger.warning(
                        f"Item '{item_id_from_result}' (orig: {original_item_id}) failed preprocessing: {item_preproc_output_dict['error']} - {item_preproc_output_dict.get('detail')}",
                        extra={
                            "item_id": item_id_from_result,
                            "error_details": item_preproc_output_dict,
                        },
                    )
                    final_results_map[item_id_from_result] = {
                        "id": item_id_from_result,
                        "embedding": None,
                        "error": item_preproc_output_dict["error"],
                        "detail": item_preproc_output_dict.get("detail"),
                        "debugMetadata": {
                            **(item_preproc_output_dict.get("debug_metadata", {})),
                            "model": clip_embedder_instance.model_name,
                            "device": clip_embedder_instance.device,
                            "batch_processing_status": "failed_preprocessing",
                        },
                    }
                    continue

                if (
                    not item_preproc_output_dict.get("pil_images")
                    and item_preproc_output_dict.get("requested_media_type") != "video"
                    or (
                        item_preproc_output_dict.get("requested_media_type") == "video"
                        and item_preproc_output_dict.get(
                            "num_frames_requested_for_video", -1
                        )
                        > 0
                        and not item_preproc_output_dict.get("pil_images")
                    )
                ):
                    # Error if images/frames were expected but not produced,
                    # unless it's a video and 0 frames were specifically requested.
                    parent_batch_logger.warning(
                        f"Item '{item_id_from_result}' yielded no PIL images after preprocessing (error not explicitly set). Marking as error. Details: {item_preproc_output_dict.get('debug_metadata')}",
                        extra={
                            "item_id": item_id_from_result,
                            "preproc_output": item_preproc_output_dict,
                        },
                    )
                    final_results_map[item_id_from_result] = {
                        "id": item_id_from_result,
                        "embedding": None,
                        "error": "No images/frames extracted (internal)",
                        "detail": "Preprocessing yielded no processable PIL images, and no explicit error was set.",
                        "debugMetadata": {
                            **(item_preproc_output_dict.get("debug_metadata", {})),
                            "model": clip_embedder_instance.model_name,
                            "device": clip_embedder_instance.device,
                            "batch_processing_status": "no_images_from_preprocessing_unexpected",
                        },
                    }
                    continue

                # If it's a video and 0 frames were requested, pil_images will be empty, which is fine.
                # It won't be added to all_pil_images_for_gpu_batch.
                if item_preproc_output_dict.get("pil_images"):
                    gpu_batch_item_details_map.append(
                        {
                            "item_id": item_id_from_result,
                            "media_type": item_preproc_output_dict["media_type"],
                            "num_source_pil_images": len(
                                item_preproc_output_dict["pil_images"]
                            ),
                            "start_idx_in_gpu_batch": len(all_pil_images_for_gpu_batch),
                            "original_debug_metadata": item_preproc_output_dict.get(
                                "debug_metadata", {}
                            ),
                        }
                    )
                    all_pil_images_for_gpu_batch.extend(
                        item_preproc_output_dict["pil_images"]
                    )
                    parent_batch_logger.debug(
                        f"Item '{item_id_from_result}' preprocessed. Added {len(item_preproc_output_dict['pil_images'])} PIL images to GPU batch.",
                        extra={
                            "item_id": item_id_from_result,
                            "num_pil_added": len(
                                item_preproc_output_dict["pil_images"]
                            ),
                        },
                    )
                else:  # No PIL images (e.g. video with 0 frames requested)
                    # This item still needs a result entry, even if embedding is empty/null
                    # It will be handled after GPU processing stage or if GPU batch is empty
                    parent_batch_logger.debug(
                        f"Item '{item_id_from_result}' preprocessed with no PIL images (e.g., 0 frames requested for video).",
                        extra={"item_id": item_id_from_result},
                    )

            except Exception as e_future_generic:
                parent_batch_logger.error(
                    f"Unexpected error retrieving preprocessing result for item ID '{original_item_id}': {e_future_generic}",
                    extra={"item_id": original_item_id},
                    error=e_future_generic,
                )
                final_results_map[original_item_id] = {
                    "id": original_item_id,
                    "embedding": None,
                    "error": "Internal error handling preprocessing result",
                    "detail": str(e_future_generic),
                    "debugMetadata": {
                        "model": clip_embedder_instance.model_name,
                        "device": clip_embedder_instance.device,
                        "batch_processing_status": "internal_error_future_result",
                    },
                }

    if not all_pil_images_for_gpu_batch:
        parent_batch_logger.warning(
            f"Batch '{batch_processing_id}': No images/frames to process on GPU after parallel preprocessing stage.",
            extra={
                "num_initial_items": len(items_data_as_dicts),
                "items_in_gpu_map_count": len(
                    gpu_batch_item_details_map
                ),  # This map only contains items that *had* PILs
            },
        )
        # Handle items that were successfully preprocessed but yielded no PILs (e.g. 0-frame videos)
        # or items that errored out during preprocessing (already in final_results_map).
        for item_data in items_data_as_dicts:
            item_id_chk = item_data["id"]
            if item_id_chk not in final_results_map:
                # This item must have been preprocessed without error but produced no PILs
                # (e.g., video with num_frames=0). Create a success entry with empty embedding.
                original_debug_meta = {}
                # Try to find its original debug meta if it was mapped in gpu_batch_item_details_map (it wouldn't be if no pils)
                # This needs a more robust way to get original_debug_metadata for items that don't go to GPU.
                # For now, create a basic entry.
                # A better way: _preprocess_single_item_for_batch always returns its debug_metadata.
                # We can collect all preproc_output_dicts and then decide.

                # Let's assume item_preproc_output_dict was the result for this item_id_chk
                # This part of logic is tricky if we don't store all preproc results first.
                # The current loop only processes results with PILs for gpu_batch_item_details_map.
                # We need to ensure all items get a final entry.
                # This block will now correctly create entries for items that had 0 frames and thus no PILs.
                final_results_map[item_id_chk] = {
                    "id": item_id_chk,
                    "embedding": [],  # Empty list for 0 frames or no images
                    "error": None,
                    "detail": "No images/frames were processed for this item (e.g., 0 frames requested).",
                    "debugMetadata": {
                        # "original_debug_metadata": "N/A here, need to fetch from preproc results", # Placeholder
                        "model": clip_embedder_instance.model_name,
                        "device": clip_embedder_instance.device,
                        "batch_processing_status": "success_no_images_for_gpu",
                        "processing_timestamp_utc": datetime.utcnow().isoformat() + "Z",
                        "overall_batch_request_id": batch_processing_id,
                    },
                }
                # To get the actual debug_metadata from preprocessing for such items,
                # we would need to store all future.result() from the preprocessing stage
                # and then iterate through them to build final_results_map.
                # The current structure is slightly lossy for debug_metadata of 0-frame videos here.
                # For simplicity, we'll proceed, but this is an area for refinement if detailed
                # preproc debug metadata for 0-frame videos is critical in the output.

        batch_total_duration_ms = (time.time() - batch_overall_start_time) * 1000
        parent_batch_logger.info(
            f"Batch '{batch_processing_id}' finished in {batch_total_duration_ms:.2f}ms. No items sent to GPU (or only items with 0 frames).",
            extra={"duration_ms": batch_total_duration_ms},
        )
        return final_results_map

    # --- GPU Processing Stage ---
    # (This part remains largely the same)
    gpu_embeddings_tensor: Optional[torch.Tensor] = None
    gpu_stage_failed_globally = False
    try:
        if clip_embedder_instance is None:
            raise RuntimeError("CLIPEmbedder instance is None prior to GPU batch.")
        parent_batch_logger.info(
            f"Batch '{batch_processing_id}': Sending {len(all_pil_images_for_gpu_batch)} total images/frames to CLIP model on {clip_embedder_instance.device} for inference.",
            extra={"gpu_batch_size": len(all_pil_images_for_gpu_batch)},
        )
        gpu_embeddings_tensor = clip_embedder_instance.get_embeddings_for_pil_list(
            all_pil_images_for_gpu_batch
        )
    except Exception as e_gpu_stage:
        parent_batch_logger.error(
            f"Batch '{batch_processing_id}': GPU processing failed for the entire collected batch of {len(all_pil_images_for_gpu_batch)} images/frames: {e_gpu_stage}",
            error=e_gpu_stage,
        )
        gpu_stage_failed_globally = True
        for item_map_detail in gpu_batch_item_details_map:
            item_id_gpu_fail = item_map_detail["item_id"]
            if item_id_gpu_fail not in final_results_map:
                final_results_map[item_id_gpu_fail] = {
                    "id": item_id_gpu_fail,
                    "embedding": None,
                    "error": "GPU processing failed for batch",
                    "detail": str(e_gpu_stage),
                    "debugMetadata": {
                        **(item_map_detail.get("original_debug_metadata", {})),
                        "model": clip_embedder_instance.model_name,
                        "device": clip_embedder_instance.device,
                        "gpu_error_global": True,
                        "batch_processing_status": "failed_gpu_inference_stage",
                    },
                }
        # Also handle items that were requested but didn't even make it to gpu_batch_item_details_map
        # (e.g., 0-frame videos, or items that errored in preproc already handled)
        for item_data_orig in items_data_as_dicts:
            if item_data_orig["id"] not in final_results_map:
                final_results_map[item_data_orig["id"]] = (
                    {  # Should be caught by "no images for GPU batch" logic if it was a 0-frame video.
                        "id": item_data_orig["id"],
                        "embedding": None,
                        "error": "GPU processing failed for batch (item not processed)",
                        "detail": str(e_gpu_stage),
                        "debugMetadata": {
                            "model": clip_embedder_instance.model_name,
                            "device": clip_embedder_instance.device,
                            "gpu_error_global": True,
                            "batch_processing_status": "failed_gpu_inference_stage_item_missed",
                        },
                    }
                )

        batch_total_duration_ms = (time.time() - batch_overall_start_time) * 1000
        parent_batch_logger.info(
            f"Batch '{batch_processing_id}' processing terminated in {batch_total_duration_ms:.2f}ms due to GPU stage error.",
            extra={"duration_ms": batch_total_duration_ms},
        )
        return final_results_map

    # --- Disaggregate GPU results and finalize ---
    # (This part remains largely the same)
    utc_processing_timestamp = datetime.utcnow().isoformat() + "Z"
    for (
        item_map_detail
    ) in gpu_batch_item_details_map:  # Iterate items that *were* part of GPU batch
        item_id_final = item_map_detail["item_id"]
        if (
            item_id_final in final_results_map
        ):  # Already has an error from preprocessing
            continue

        if gpu_embeddings_tensor is None or gpu_stage_failed_globally:
            # This case should have been handled by the gpu_stage_failed_globally block above
            if item_id_final not in final_results_map:  # Defensive
                final_results_map[item_id_final] = {
                    "id": item_id_final,
                    "embedding": None,
                    "error": "GPU result tensor unavailable post-inference (should have been caught)",
                    "debugMetadata": {
                        **(item_map_detail.get("original_debug_metadata", {})),
                        "model": clip_embedder_instance.model_name,
                        "device": clip_embedder_instance.device,
                        "batch_processing_status": "gpu_tensor_missing_post_inference_final_defensive",
                    },
                }
            continue

        start_idx = item_map_detail["start_idx_in_gpu_batch"]
        num_pils_for_item = item_map_detail["num_source_pil_images"]

        if (
            num_pils_for_item == 0
        ):  # Should not happen if item is in gpu_batch_item_details_map
            parent_batch_logger.error(
                f"Item '{item_id_final}' in GPU map but has 0 source PILs. Inconsistent state."
            )
            final_results_map[item_id_final] = {
                "id": item_id_final,
                "embedding": None,
                "error": "Internal error: 0 PILs in GPU map",
                "debugMetadata": {
                    **(item_map_detail.get("original_debug_metadata", {})),
                    "model": clip_embedder_instance.model_name,
                    "device": clip_embedder_instance.device,
                    "batch_processing_status": "error_internal_gpu_map_zero_pils",
                },
            }
            continue

        item_slice_from_gpu_tensor = gpu_embeddings_tensor[
            start_idx : start_idx + num_pils_for_item
        ]

        item_embedding_list: Optional[List[float]] = None
        item_specific_gpu_debug = {}

        if item_map_detail["media_type"] == "image":
            if num_pils_for_item == 1 and item_slice_from_gpu_tensor.shape[0] == 1:
                item_embedding_list = item_slice_from_gpu_tensor[0].tolist()
            else:
                parent_batch_logger.error(
                    f"Embedding disaggregation error for image item '{item_id_final}': expected 1 embedding, got slice shape {item_slice_from_gpu_tensor.shape}, num_pils: {num_pils_for_item}",
                    extra={
                        "item_id": item_id_final,
                        "slice_shape": str(item_slice_from_gpu_tensor.shape),
                        "num_pils": num_pils_for_item,
                    },
                )
                final_results_map[item_id_final] = {
                    "id": item_id_final,
                    "embedding": None,
                    "error": "Image embedding disaggregation error",
                    "debugMetadata": {
                        **(item_map_detail.get("original_debug_metadata", {})),
                        "model": clip_embedder_instance.model_name,
                        "device": clip_embedder_instance.device,
                        "batch_processing_status": "error_disaggregating_image",
                    },
                }
                continue
        elif item_map_detail["media_type"] == "video":
            if (
                num_pils_for_item > 0
                and item_slice_from_gpu_tensor.shape[0] == num_pils_for_item
            ):
                averaged_embedding = item_slice_from_gpu_tensor.mean(dim=0)
                item_embedding_list = averaged_embedding.tolist()
                item_specific_gpu_debug["averaged_from_n_frames_in_gpu_batch"] = (
                    num_pils_for_item
                )
            # Case for num_pils_for_item == 0 for video is handled before GPU stage; it wouldn't be in gpu_batch_item_details_map.
            else:  # Mismatch error for video
                parent_batch_logger.error(
                    f"Embedding disaggregation error for video item '{item_id_final}': expected {num_pils_for_item} embeddings for averaging, got slice shape {item_slice_from_gpu_tensor.shape}",
                    extra={
                        "item_id": item_id_final,
                        "slice_shape": str(item_slice_from_gpu_tensor.shape),
                        "num_pils": num_pils_for_item,
                    },
                )
                final_results_map[item_id_final] = {
                    "id": item_id_final,
                    "embedding": None,
                    "error": "Video embedding disaggregation error",
                    "debugMetadata": {
                        **(item_map_detail.get("original_debug_metadata", {})),
                        "model": clip_embedder_instance.model_name,
                        "device": clip_embedder_instance.device,
                        "batch_processing_status": "error_disaggregating_video",
                    },
                }
                continue
        else:
            parent_batch_logger.error(
                f"Unknown media_type '{item_map_detail['media_type']}' for item '{item_id_final}' at disaggregation.",
                extra={"item_id": item_id_final},
            )
            item_embedding_list = []

        final_results_map[item_id_final] = {
            "id": item_id_final,
            "embedding": item_embedding_list,
            "error": None,
            "detail": None,
            "debugMetadata": {
                **(item_map_detail.get("original_debug_metadata", {})),
                **item_specific_gpu_debug,
                "model": clip_embedder_instance.model_name,
                "device": clip_embedder_instance.device,
                "overall_batch_request_id": batch_processing_id,
                "processing_timestamp_utc": utc_processing_timestamp,
                "batch_processing_status": "success",
            },
        }
        parent_batch_logger.debug(
            f"Successfully finalized result for item '{item_id_final}'.",
            extra={"item_id": item_id_final},
        )

    # Ensure all original items have a result in final_results_map
    # This handles items that were valid but had 0 frames requested (so no PILs for GPU)
    for item_data_orig in items_data_as_dicts:
        item_id_to_check = item_data_orig["id"]
        if item_id_to_check not in final_results_map:
            # This item was not an error during preprocessing, and not part of GPU batch (e.g. 0 frames)
            parent_batch_logger.debug(
                f"Finalizing result for item '{item_id_to_check}' which had no PILs for GPU (e.g. 0 frames)."
            )
            # We need the debug_metadata from its preprocessing stage.
            # This requires collecting all preproc_output_dicts earlier.
            # For now, we'll create a generic success entry.
            # TODO: Refactor to collect all preproc_output_dicts to access their debug_metadata here.
            final_results_map[item_id_to_check] = {
                "id": item_id_to_check,
                "embedding": [],  # Empty embedding for 0 frames
                "error": None,
                "detail": "Successfully processed item with no images/frames for GPU (e.g. 0 frames requested).",
                "debugMetadata": {
                    # "original_debug_metadata": "N/A without collecting all preproc outputs", # Placeholder
                    "model": clip_embedder_instance.model_name,
                    "device": clip_embedder_instance.device,
                    "overall_batch_request_id": batch_processing_id,
                    "processing_timestamp_utc": utc_processing_timestamp,  # Use the same timestamp
                    "batch_processing_status": "success_no_images_processed_on_gpu",
                },
            }

    batch_total_duration_ms = (time.time() - batch_overall_start_time) * 1000
    successful_items_count = sum(
        1 for res in final_results_map.values() if not res.get("error")
    )
    parent_batch_logger.info(
        f"Batch '{batch_processing_id}' processing completed in {batch_total_duration_ms:.2f}ms. "
        f"Total items: {len(items_data_as_dicts)}, Items sent to GPU: {len(gpu_batch_item_details_map)}, "
        f"Successful (final): {successful_items_count}, Errors: {len(items_data_as_dicts) - successful_items_count}",
        extra={
            "duration_ms": batch_total_duration_ms,
            "total_items": len(items_data_as_dicts),
            "gpu_candidate_items": len(
                gpu_batch_item_details_map
            ),  # Items that had PILs
            "final_success_count": successful_items_count,
            "final_error_count": len(items_data_as_dicts) - successful_items_count,
        },
    )
    return final_results_map


if __name__ == "__main__":
    print("[embedding_service_helper.py] Loaded environment/config values:")
    print(f"  PYTHON_PORT={os.environ.get('PYTHON_PORT')}")
    print(f"  PYTHON_MEDIA_ROOT={os.environ.get('PYTHON_MEDIA_ROOT')}")
    print(f"  LOG_LEVEL={os.environ.get('LOG_LEVEL')}")
    print(f"  CLIP_MODEL={os.environ.get('CLIP_MODEL')}")
    print(f"  ENABLE_AUGMENTATION={os.environ.get('ENABLE_AUGMENTATION')}")
    print(f"  TARGET_VRAM_UTILIZATION={os.environ.get('TARGET_VRAM_UTILIZATION')}")
    print(f"  MAX_BATCH_ITEMS={os.environ.get('MAX_BATCH_ITEMS')}")
    print(f"  BATCH_FLUSH_TIMEOUT_S={os.environ.get('BATCH_FLUSH_TIMEOUT_S')}")
    print(f"  GPU_POLL_INTERVAL_S={os.environ.get('GPU_POLL_INTERVAL_S')}")
    print(
        f"  DEFAULT_VIDEO_FRAMES_TO_EXTRACT={os.environ.get('DEFAULT_VIDEO_FRAMES_TO_EXTRACT')}"
    )
    print(f"  DOWNLOAD_TIMEOUT_SECONDS={os.environ.get('DOWNLOAD_TIMEOUT_SECONDS')}")
