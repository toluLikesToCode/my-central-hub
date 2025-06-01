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
    • Advanced video frame extraction using scene detection and visual entropy, with hardware acceleration support.
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
import math
import errno
import requests  # For URL downloads
import open_clip  # type: ignore
import cv2  # type: ignore # Though unused, kept as per original file structure

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
    def __init__(
        self,
        num_frames: int,
        video_path: Optional[str] = None,
        logger: Optional[EmbeddingLogger] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        request_id: Optional[str] = None,
        duration: Optional[float] = None,
        original_filename_hint: Optional[str] = None,
        hwaccel_method: Optional[str] = None,  # Added for HW acceleration
    ):
        if not video_path:
            raise ValueError("VideoProcessor requires a valid video_path.")

        self.video_path: str = video_path
        self.num_frames = num_frames
        self.logger = logger or embedding_logger
        base_component = f"VideoProc-{original_filename_hint[:20] if original_filename_hint else os.path.basename(video_path)[:20]}"
        self.logger.set_component_name(
            f"{base_component}-{request_id[:8] if request_id else uuid.uuid4().hex[:8]}"
        )
        if request_id:
            self.logger.set_request_id(request_id)

        self.executor = executor
        self.hwaccel_method = hwaccel_method
        if self.hwaccel_method:
            self.logger.info(
                f"VideoProcessor for '{self.video_path}' will attempt to use HWAccel: {self.hwaccel_method}"
            )

        self.duration = duration if duration is not None else self._get_duration()
        if self.duration is None or self.duration <= 0:
            self.logger.error(
                f"Video duration error for '{self.video_path}'",
                extra={"duration": self.duration, "video_path": self.video_path},
            )
            raise ValueError(
                f"Could not determine video duration or duration is invalid for '{self.video_path}'."
            )
        # self._is_temp_file attribute was not used, so removed for clarity.

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
                error=None,
                extra={"video_path": self.video_path},
            )
            raise RuntimeError(
                f"ffprobe timeout getting duration for '{self.video_path}'"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to get video duration for '{self.video_path}': {e}",
                error=e,
                extra={"video_path": self.video_path},
            )
            raise RuntimeError(
                f"Failed to get video duration for '{self.video_path}': {e}"
            ) from e

    def _extract_frame_hw_accelerated(self, time_sec: float) -> Image.Image:
        if not self.hwaccel_method:  # Should not be called if None, but as a safeguard
            raise ValueError(
                "HWAccel method not specified for hardware accelerated extraction."
            )

        self.logger.debug(
            f"Attempting HW accelerated frame extraction (method: {self.hwaccel_method}) at {time_sec:.2f}s from '{self.video_path}'"
        )

        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-hwaccel",
            self.hwaccel_method,
        ]

        # For many hardware acceleration methods, explicitly setting hwaccel_output_format
        # to the method name itself ensures frames are kept in hardware memory in their
        # native format, which is often required or optimal for the hwdownload filter.
        if self.hwaccel_method in ["cuda", "vaapi", "qsv", "vulkan", "amf", "opencl"]:
            command.extend(["-hwaccel_output_format", self.hwaccel_method])

        command.extend(
            [
                "-ss",
                str(time_sec),
                "-i",
                self.video_path,
                "-vf",
                "hwdownload,format=rgb24",  # Crucial: download from HW surface & convert to RGB
                "-vframes",
                "1",
                "-f",
                "image2pipe",
                "-c:v",
                "mjpeg",  # Output as MJPEG to pipe
                "-q:v",
                "2",  # High quality MJPEG
                "-",  # Output to stdout
            ]
        )

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=25,  # Slightly longer timeout for HW init
            )
            if not result.stdout:
                err_msg = (
                    result.stderr.decode("utf-8", errors="ignore").strip()
                    if result.stderr
                    else "No stderr"
                )
                self.logger.error(
                    f"ffmpeg (HWAccel: {self.hwaccel_method}) failed for '{self.video_path}' at {time_sec:.2f}s: No stdout. Stderr: {err_msg}",
                    extra={
                        "time_sec": time_sec,
                        "stderr": err_msg,
                        "command": " ".join(command),
                    },
                )
                raise RuntimeError(
                    f"No frame data from HWAccel ({self.hwaccel_method}). Stderr: {err_msg}"
                )
            return Image.open(io.BytesIO(result.stdout)).convert("RGB")
        except subprocess.TimeoutExpired as e_timeout:
            self.logger.error(
                f"ffmpeg (HWAccel: {self.hwaccel_method}) timeout for '{self.video_path}' at {time_sec:.2f}s",
                error=e_timeout,
                extra={"time_sec": time_sec, "command": " ".join(command)},
            )
            raise RuntimeError(
                f"ffmpeg (HWAccel: {self.hwaccel_method}) timeout extracting frame at {time_sec:.2f}s"
            ) from e_timeout
        except subprocess.CalledProcessError as e_call:
            stderr_output = (
                e_call.stderr.decode("utf-8", errors="ignore").strip()
                if e_call.stderr
                else "N/A"
            )
            self.logger.error(
                f"ffmpeg (HWAccel: {self.hwaccel_method}) returned non-zero exit code {e_call.returncode} for '{self.video_path}' at {time_sec:.2f}s. Stderr: {stderr_output}",
                error=e_call,
                extra={
                    "time_sec": time_sec,
                    "return_code": e_call.returncode,
                    "stderr": stderr_output,
                    "command": " ".join(command),
                },
            )
            raise RuntimeError(
                f"HWAccel ({self.hwaccel_method}) frame extraction failed with exit code {e_call.returncode} at {time_sec:.2f}s. Stderr: {stderr_output}"
            ) from e_call
        except Exception as e_generic:
            self.logger.error(
                f"Frame extraction (HWAccel: {self.hwaccel_method}) failed for '{self.video_path}' at {time_sec:.2f}s: {e_generic}",
                error=e_generic,
                extra={"time_sec": time_sec, "command": " ".join(command)},
            )
            raise RuntimeError(
                f"Failed to extract frame at {time_sec:.2f}s using HWAccel ({self.hwaccel_method}): {e_generic}"
            ) from e_generic

    def _extract_frame_software(self, time_sec: float) -> Image.Image:
        self.logger.debug(
            f"Extracting frame (software) at {time_sec:.2f}s from '{self.video_path}'"
        )
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-ss",
            str(time_sec),
            "-i",
            self.video_path,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "mjpeg",  # Use modern -c:v
            "-q:v",
            "2",
            "-",
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=20,
            )
            if not result.stdout:
                err_msg = (
                    result.stderr.decode("utf-8", errors="ignore").strip()
                    if result.stderr
                    else "No stderr"
                )
                self.logger.error(
                    f"ffmpeg (software) failed for '{self.video_path}' at {time_sec:.2f}s: No stdout. Stderr: {err_msg}",
                    extra={
                        "time_sec": time_sec,
                        "stderr": err_msg,
                        "command": " ".join(command),
                    },
                )
                raise RuntimeError(
                    f"No frame data from software mjpeg. Stderr: {err_msg}"
                )
            return Image.open(io.BytesIO(result.stdout)).convert("RGB")
        except subprocess.TimeoutExpired as e_timeout:
            self.logger.error(
                f"ffmpeg (software) timeout for '{self.video_path}' at {time_sec:.2f}s",
                error=e_timeout,
                extra={"time_sec": time_sec, "command": " ".join(command)},
            )
            raise RuntimeError(
                f"ffmpeg (software) timeout extracting frame at {time_sec:.2f}s"
            ) from e_timeout
        except subprocess.CalledProcessError as e_call:
            stderr_output = (
                e_call.stderr.decode("utf-8", errors="ignore").strip()
                if e_call.stderr
                else "N/A"
            )
            self.logger.error(
                f"ffmpeg (software) returned non-zero exit code {e_call.returncode} for '{self.video_path}' at {time_sec:.2f}s. Stderr: {stderr_output}",
                error=e_call,
                extra={
                    "time_sec": time_sec,
                    "return_code": e_call.returncode,
                    "stderr": stderr_output,
                    "command": " ".join(command),
                },
            )
            raise RuntimeError(
                f"Software frame extraction failed with exit code {e_call.returncode} at {time_sec:.2f}s. Stderr: {stderr_output}"
            ) from e_call
        except Exception as e_generic:
            self.logger.error(
                f"Frame extraction (software) failed for '{self.video_path}' at {time_sec:.2f}s: {e_generic}",
                error=e_generic,
                extra={"time_sec": time_sec, "command": " ".join(command)},
            )
            raise RuntimeError(
                f"Failed to extract frame at {time_sec:.2f}s using software mjpeg: {e_generic}"
            ) from e_generic

    def extract_frame(self, time_sec: float) -> Image.Image:
        """
        Extracts a single frame from the video.
        Tries hardware acceleration if configured, otherwise falls back to software.
        """
        if self.hwaccel_method:
            try:
                self.logger.debug(
                    f"Attempting HW accel ({self.hwaccel_method}) for frame at {time_sec:.2f}s from '{self.video_path}'"
                )
                return self._extract_frame_hw_accelerated(time_sec)
            except Exception as e_hw:
                self.logger.warning(
                    f"HW accelerated ({self.hwaccel_method}) frame extraction failed for '{self.video_path}' at {time_sec:.2f}s: {e_hw}. Falling back to software.",
                    extra={
                        "time_sec": time_sec,
                        "error_type": type(e_hw).__name__,
                        "error_message": str(e_hw),
                    },
                )
                # Fall through to software extraction
        else:
            self.logger.debug(
                f"No HWAccel method configured. Using software extraction for frame at {time_sec:.2f}s from '{self.video_path}'"
            )

        return self._extract_frame_software(time_sec)

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
        if self.duration <= 0:
            return [], {
                "method_used": "none",
                "reason": "video duration is zero or negative",
                **debug_metadata,
            }

        start_offset = min(0.5, self.duration * 0.02)
        end_offset = min(0.5, self.duration * 0.02)
        effective_duration = self.duration - start_offset - end_offset

        selected_times: List[float]
        if self.num_frames == 1:
            selected_times = [
                start_offset + effective_duration / 2.0
            ]  # Centered in effective duration
            # Ensure it's not exactly 0 if duration is very short leading to effective_duration being 0
            if effective_duration <= 0:
                selected_times = [self.duration / 2.0]

            method_used = "single_middle_frame"
        elif effective_duration <= 0.1:
            # If effective duration is too small, take the middle frame and duplicate
            middle_time = self.duration / 2.0
            selected_times = [middle_time] * self.num_frames
            method_used = "middle_frame_duplicated_short_video"

        elif (
            self.num_frames > 1
        ):  # Spread self.num_frames evenly across effective_duration
            selected_times = [
                start_offset + (i * effective_duration / (self.num_frames - 1))
                for i in range(self.num_frames)
            ]
            method_used = "uniform_spread_with_offset"
        else:  # Should not happen if num_frames == 1 is handled
            selected_times = [self.duration / 2.0]
            method_used = "fallback_single_middle_frame"

        # Ensure timestamps are within valid range and distinct enough (simplified)
        # Clamp times strictly within [0, duration - a_tiny_bit_for_ffmpeg]
        # ffmpeg -ss can be problematic at the exact end of the video.
        epsilon = 0.001
        selected_times = sorted(
            list(set(max(0.0, min(t, self.duration - epsilon)) for t in selected_times))
        )
        # If clamping resulted in an empty list (e.g., duration was epsilon), provide at least one frame.
        if not selected_times and self.duration > 0:
            selected_times = [max(0.0, self.duration / 2.0 - epsilon)]

        debug_metadata["method_used"] = method_used
        debug_metadata["candidate_timestamps"] = list(
            selected_times
        )  # Log before potential modifications
        debug_metadata["effective_sampling_duration_s"] = effective_duration
        debug_metadata["start_offset_s"] = start_offset

        self.logger.debug(
            f"Selected {len(selected_times)} timestamps via {method_used}",
            extra={"timestamps": selected_times, **debug_metadata},
        )
        return selected_times, debug_metadata

    def extract_frames(self) -> Tuple[List[Image.Image], Dict[str, Any]]:
        self.logger.info(
            f"Extracting up to {self.num_frames} frames from '{self.video_path}'"
        )
        overall_start_time = time.time()
        extracted_frames_pil: List[Image.Image] = []

        timestamps, frame_sampling_debug_meta = self.get_advanced_sample_times()
        final_debug_meta: Dict[str, Any] = {
            "frame_sampling_details": frame_sampling_debug_meta
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

        actual_timestamps_to_extract = timestamps[: self.num_frames]
        final_debug_meta["actual_timestamps_for_extraction"] = (
            actual_timestamps_to_extract
        )

        extraction_errors_count = 0
        use_parallel_extraction = (
            self.executor and len(actual_timestamps_to_extract) > 1
        )

        if use_parallel_extraction and self.executor:
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
                        f"Parallel extract_frame failed for ts={ts:.2f}s: {e_frame}",
                        error=e_frame,
                        extra={"timestamp": ts},
                    )
        else:
            for ts in actual_timestamps_to_extract:
                try:
                    frame_pil = self.extract_frame(ts)
                    extracted_frames_pil.append(frame_pil)
                except Exception as e_frame:
                    extraction_errors_count += 1
                    self.logger.error(
                        f"Sequential extract_frame failed for ts={ts:.2f}s: {e_frame}",
                        error=e_frame,
                        extra={"timestamp": ts},
                    )

        if extraction_errors_count > 0:
            final_debug_meta["frame_extraction_error_count"] = extraction_errors_count

        if not extracted_frames_pil and len(actual_timestamps_to_extract) > 0:
            err_msg = f"All {len(actual_timestamps_to_extract)} frame extractions failed for '{self.video_path}'."
            self.logger.error(err_msg, extra=final_debug_meta)
            return [], {"error": err_msg, **final_debug_meta}

        if 0 < len(extracted_frames_pil) < self.num_frames:
            self.logger.warning(
                f"Extracted {len(extracted_frames_pil)} frames, but {self.num_frames} were requested for '{self.video_path}'. Duplicating last good frame.",
                extra=final_debug_meta,
            )
            last_good_frame = extracted_frames_pil[-1]
            num_to_add = self.num_frames - len(extracted_frames_pil)
            extracted_frames_pil.extend(
                [last_good_frame.copy() for _ in range(num_to_add)]
            )
        elif not extracted_frames_pil and self.num_frames > 0:
            self.logger.error(
                f"No frames were ultimately available for '{self.video_path}' though {self.num_frames} were requested.",
                extra=final_debug_meta,
            )
            return [], {
                "error": f"No frames available after processing for {self.video_path}",
                **final_debug_meta,
            }

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
    preprocess: Callable[[Any], torch.Tensor]

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        logger: Optional[EmbeddingLogger] = None,
        enable_augmentation: bool = False,
    ):
        self.logger = logger or embedding_logger
        self.logger.set_component_name(f"CLIPEmbedder-{model_name.split('/')[-1][:15]}")
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
        try:
            arch_name_map = {
                "vit-base-patch32": "ViT-B-32",
                "vit-b-32": "ViT-B-32",
                "vit-large-patch14": "ViT-L-14",
                "vit-l-14": "ViT-L-14",
            }
            model_arch = self.model_name
            pretrained_tag = "openai"

            if "/" in self.model_name:
                author_part, model_part = self.model_name.split("/", 1)
                if author_part.lower() == "openai":
                    for key, val in arch_name_map.items():
                        if key in model_part.lower():
                            model_arch = val
                            break
                else:
                    model_arch_candidate_parts = model_part.split("-")
                    # Example: "laion/CLIP-ViT-B-32-laion2B-s34B-b79K" -> model_arch = ViT-B-32, pretrained = laion/CLIP-ViT-B-32-laion2B-s34B-b79K
                    # Example: "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
                    # OpenCLIP create_model_and_transforms handles "hf-hub:" prefix for pretrained
                    if (
                        len(model_arch_candidate_parts) > 1
                        and "CLIP" in model_arch_candidate_parts[0].upper()
                    ):  # e.g. CLIP-ViT-B-32
                        arch_try = "-".join(model_arch_candidate_parts[1:3])  # ViT-B-32
                        for key, val in arch_name_map.items():
                            if key == arch_try.lower():
                                model_arch = val
                                break
                    # For non-openai, pretrained is often the full model name, possibly with hf-hub prefix
                    pretrained_tag = (
                        f"hf-hub:{self.model_name}"
                        if not self.model_name.startswith("hf-hub:")
                        and author_part.lower() != "openai"
                        else self.model_name
                    )

            self.logger.debug(
                f"Attempting to load OpenCLIP model_arch='{model_arch}', pretrained='{pretrained_tag}'"
            )

            (
                self.model,
                _,
                self.preprocess,
            ) = open_clip.create_model_and_transforms(  # _ is train_preprocess, we use val_preprocess
                model_name=model_arch,
                pretrained=pretrained_tag,
                device=self.device,
                jit=False,  # jit=False for wider compatibility
            )
            self.model.eval()

            if self.device not in ["cpu"]:
                try:
                    # Only available in PyTorch 1.12+
                    if hasattr(torch, "set_float32_matmul_precision"):
                        torch.set_float32_matmul_precision("high")
                        self.logger.debug("Set float32 matmul precision to 'high'.")
                except AttributeError:
                    self.logger.debug(
                        "torch.set_float32_matmul_precision not available (requires PyTorch 1.12+)."
                    )
            self.logger.info(
                f"CLIP model '{self.model_name}' (parsed as arch='{model_arch}', pretrained='{pretrained_tag}') loaded on '{self.device}'. Init time: {(time.time() - init_start_time):.2f}s"
            )
        except Exception as e_load:
            self.logger.error(
                f"Failed to load CLIP model '{self.model_name}': {e_load}", error=e_load
            )
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
            return torch.empty(
                0, dtype=torch.float32, device="cpu"
            )  # Return on CPU always for empty

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
                    error=e_pil_proc,
                    extra={"index": i},
                )
                raise RuntimeError(
                    f"PIL image preprocessing (index {i}) failed: {e_pil_proc}"
                ) from e_pil_proc

        if not tensors_for_model:
            self.logger.warning(
                "No PIL images were successfully preprocessed into tensors for the model."
            )
            return torch.empty(0, dtype=torch.float32, device="cpu")

        try:
            input_tensor_batch = torch.stack(tensors_for_model).to(self.device)
        except Exception as e_stack:
            self.logger.error(
                f"Error stacking preprocessed image tensors (count: {len(tensors_for_model)}): {e_stack}",
                error=e_stack,
                extra={
                    "tensor_shapes_preview": [
                        str(t.shape) for t in tensors_for_model[:3]
                    ]
                },
            )
            raise RuntimeError(f"Tensor stacking failed: {e_stack}") from e_stack

        autocast_context = nullcontext()
        if self.device != "cpu":
            # For CUDA, float16 is common. For MPS, bfloat16 might be preferred if available, else float16.
            # OpenCLIP/PyTorch autocast handles this reasonably.
            autocast_context = torch.autocast(device_type=self.device.split(":")[0])

        with torch.no_grad(), autocast_context:
            image_features_batch = self.model.encode_image(input_tensor_batch)
            if image_features_batch is not None:  # Should always return a tensor
                image_features_batch = image_features_batch / image_features_batch.norm(
                    p=2, dim=-1, keepdim=True
                )
            else:  # Should not happen with OpenCLIP typically
                self.logger.error(
                    "Model encode_image returned None, which is unexpected."
                )
                raise RuntimeError("Model encoding failed, returned None.")

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
    ffmpeg_hwaccel_method_for_videos: Optional[str],  # Added for HW accel
) -> Dict[str, Any]:

    item_id = item_spec_dict["id"]
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
        "source_type": item_spec_dict["source_type"],
        "original_filename": item_spec_dict.get("original_filename", "N/A"),
        "requested_media_type": item_spec_dict["media_type"],
        "timestamp_preprocess_start_utc": datetime.utcnow().isoformat() + "Z",
    }
    temp_file_created_path: Optional[str] = None

    try:
        media_source_for_pil: Union[str, io.BytesIO]
        video_path_for_vid_processor: Optional[str] = None
        source_type = item_spec_dict["source_type"]
        source_location = item_spec_dict["source"]
        media_type = item_spec_dict["media_type"]  # Re-assign from spec_dict for safety

        if source_type == "url":
            item_logger.debug(f"Downloading URL: {source_location}")
            response = requests.get(
                source_location, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
            )
            response.raise_for_status()

            temp_suffix_raw = item_spec_dict.get(
                "original_filename", uuid.uuid4().hex[:8]
            )
            # Ensure suffix has an extension if original_filename might not
            if "." not in os.path.basename(temp_suffix_raw):  # crude check
                ext_from_url = os.path.splitext(source_location)[1]
                if ext_from_url and ext_from_url.lower() in IMAGE_EXTS + VIDEO_EXTS:
                    temp_suffix_raw += ext_from_url

            safe_suffix = "".join(
                c if c.isalnum() or c in [".", "_", "-"] else "_"
                for c in temp_suffix_raw  # Allow hyphen
            )[
                :64
            ]  # Max length for suffix can be an issue on some systems

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=safe_suffix,
                dir=os.environ.get(
                    "TEMP_DOWNLOAD_DIR"
                ),  # Optional: configurable temp dir
            ) as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):  # 8KB
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

            num_frames = (
                item_spec_dict.get("num_frames") or default_num_frames_for_video
            )
            vp = VideoProcessor(
                video_path=video_path_for_vid_processor,
                num_frames=num_frames,
                logger=item_logger,
                executor=video_processor_shared_executor,
                request_id=item_specific_request_id,
                duration=item_spec_dict.get("estimated_duration_s"),
                original_filename_hint=item_spec_dict.get("original_filename"),
                hwaccel_method=ffmpeg_hwaccel_method_for_videos,  # Pass HW accel method
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
            "media_type": media_type,  # Use media_type from outer scope
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
            except OSError as e_unlink_final:  # Changed variable name to avoid conflict
                item_logger.warning(
                    f"Failed to clean up temp file '{temp_file_created_path}': {e_unlink_final}",
                    extra={"error_type": type(e_unlink_final).__name__},
                )


def process_media_batch(
    items_data_as_dicts: List[Dict[str, Any]],
    clip_embedder_instance: CLIPEmbedder,
    python_media_root_path: str,
    default_num_frames_for_videos: int,
    parent_batch_logger: EmbeddingLogger,
    batch_processing_id: str,
) -> Dict[str, Dict[str, Any]]:

    parent_batch_logger.set_request_id(batch_processing_id)
    parent_batch_logger.info(
        f"Processing media batch '{batch_processing_id}' for {len(items_data_as_dicts)} items."
    )
    batch_overall_start_time = time.time()

    final_results_map: Dict[str, Dict[str, Any]] = {}
    all_pil_images_for_gpu_batch: List[Image.Image] = []
    gpu_batch_item_details_map: List[Dict[str, Any]] = []

    num_cores = os.cpu_count() or 1
    max_preprocess_workers = min(max(4, num_cores * 2), 32)
    # For ffmpeg, which can be CPU intensive even with HW accel (due to pipe, etc.),
    # avoid oversubscribing too much if many videos.
    max_video_frame_workers = min(max(2, num_cores if num_cores > 1 else 1), 16)

    # Read the global HWAccel setting once for this batch run
    ffmpeg_hwaccel_method = os.environ.get("FFMPEG_HWACCEL_METHOD")
    if ffmpeg_hwaccel_method:
        parent_batch_logger.info(
            f"Batch processing will use FFMPEG_HWACCEL_METHOD='{ffmpeg_hwaccel_method}' for videos."
        )
    else:
        parent_batch_logger.info(
            "No FFMPEG_HWACCEL_METHOD configured; videos will use software decoding."
        )

    with ThreadPoolExecutor(
        max_workers=max_preprocess_workers, thread_name_prefix="ItemPreProc"
    ) as item_preproc_executor, ThreadPoolExecutor(
        max_workers=max_video_frame_workers, thread_name_prefix="VideoFrames"
    ) as video_ffmpeg_executor:

        item_preprocess_futures_map = {
            item_preproc_executor.submit(
                _preprocess_single_item_for_batch,
                item_data_dict,
                python_media_root_path,
                default_num_frames_for_videos,
                parent_batch_logger,
                video_ffmpeg_executor,
                f"{batch_processing_id}_{item_data_dict['id'][:8]}",
                ffmpeg_hwaccel_method,  # Pass the determined HW accel method
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

                if not item_preproc_output_dict.get("pil_images"):
                    parent_batch_logger.warning(
                        f"Item '{item_id_from_result}' yielded no PIL images after preprocessing (error not set). Marking as error.",
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
                        "num_pil_added": len(item_preproc_output_dict["pil_images"]),
                    },
                )

            except Exception as e_future_generic:
                parent_batch_logger.error(
                    f"Unexpected error retrieving preprocessing result for item ID '{original_item_id}': {e_future_generic}",
                    error=e_future_generic,
                    extra={"item_id": original_item_id},
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
                        "error_type": type(e_future_generic).__name__,
                    },
                }

    if not all_pil_images_for_gpu_batch:
        parent_batch_logger.warning(
            f"Batch '{batch_processing_id}': No images/frames to process on GPU after parallel preprocessing stage.",
            extra={
                "num_initial_items": len(items_data_as_dicts),
                "items_in_gpu_map_count": len(gpu_batch_item_details_map),
            },
        )
        for item_map_detail in gpu_batch_item_details_map:
            item_id_chk = item_map_detail["item_id"]
            if item_id_chk not in final_results_map:
                final_results_map[item_id_chk] = {
                    "id": item_id_chk,
                    "embedding": None,
                    "error": "Internal: No images for GPU batch despite seemingly successful item preprocessing",
                    "debugMetadata": {
                        **(item_map_detail.get("original_debug_metadata", {})),
                        "model": clip_embedder_instance.model_name,
                        "device": clip_embedder_instance.device,
                        "batch_processing_status": "gpu_batch_empty_after_successful_preproc",
                    },
                }
        batch_total_duration_ms = (time.time() - batch_overall_start_time) * 1000
        parent_batch_logger.info(
            f"Batch '{batch_processing_id}' finished in {batch_total_duration_ms:.2f}ms. No items sent to GPU.",
            extra={"duration_ms": batch_total_duration_ms},
        )
        return final_results_map

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
                        "error_type": type(e_gpu_stage).__name__,
                    },
                }
        batch_total_duration_ms = (time.time() - batch_overall_start_time) * 1000
        parent_batch_logger.info(
            f"Batch '{batch_processing_id}' processing terminated in {batch_total_duration_ms:.2f}ms due to GPU stage error.",
            extra={"duration_ms": batch_total_duration_ms},
        )
        return final_results_map

    utc_processing_timestamp = datetime.utcnow().isoformat() + "Z"
    for item_map_detail in gpu_batch_item_details_map:
        item_id_final = item_map_detail["item_id"]
        if item_id_final in final_results_map:
            continue

        if gpu_embeddings_tensor is None or gpu_stage_failed_globally:
            final_results_map[item_id_final] = {
                "id": item_id_final,
                "embedding": None,
                "error": "GPU result tensor unavailable post-inference",
                "detail": "Tensor was None or GPU processing failed globally for the batch.",
                "debugMetadata": {
                    **(item_map_detail.get("original_debug_metadata", {})),
                    "model": clip_embedder_instance.model_name,
                    "device": clip_embedder_instance.device,
                    "batch_processing_status": "gpu_tensor_missing_post_inference_final",
                },
            }
            continue

        start_idx = item_map_detail["start_idx_in_gpu_batch"]
        num_pils_for_item = item_map_detail["num_source_pil_images"]

        # Ensure the slice is valid
        if start_idx + num_pils_for_item > gpu_embeddings_tensor.shape[0]:
            parent_batch_logger.error(
                f"Embedding disaggregation error for item '{item_id_final}': slice indices [{start_idx}:{start_idx + num_pils_for_item}] out of bounds for tensor shape {gpu_embeddings_tensor.shape}",
                extra={
                    "item_id": item_id_final,
                    "start_idx": start_idx,
                    "num_pils": num_pils_for_item,
                    "tensor_shape": str(gpu_embeddings_tensor.shape),
                },
            )
            final_results_map[item_id_final] = {
                "id": item_id_final,
                "embedding": None,
                "error": "Embedding disaggregation error (slice out of bounds)",
                "debugMetadata": {
                    **(item_map_detail.get("original_debug_metadata", {})),
                    "model": clip_embedder_instance.model_name,
                    "device": clip_embedder_instance.device,
                    "batch_processing_status": "error_disaggregating_slice_oob",
                },
            }
            continue

        item_slice_from_gpu_tensor = gpu_embeddings_tensor[
            start_idx : start_idx + num_pils_for_item
        ]

        item_embedding_list: Optional[List[float]] = (
            None  # Default to Optional for clarity
        )
        item_specific_gpu_debug = {}

        if item_map_detail["media_type"] == "image":
            if num_pils_for_item == 1 and item_slice_from_gpu_tensor.shape[0] == 1:
                item_embedding_list = item_slice_from_gpu_tensor[0].tolist()
            elif (
                num_pils_for_item == 0 and item_slice_from_gpu_tensor.shape[0] == 0
            ):  # Image had no pils, empty tensor slice
                item_embedding_list = []  # No embedding if no image
                parent_batch_logger.warning(
                    f"Image item '{item_id_final}' had 0 PILs, resulting in empty embedding.",
                    extra={"item_id": item_id_final},
                )
            else:
                parent_batch_logger.error(
                    f"Embedding disaggregation error for image item '{item_id_final}': expected {num_pils_for_item} embedding(s), got slice shape {item_slice_from_gpu_tensor.shape}",
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
            elif num_pils_for_item == 0:  # Video yielded no frames
                item_embedding_list = []
                item_specific_gpu_debug["averaged_from_n_frames_in_gpu_batch"] = 0
                parent_batch_logger.warning(
                    f"Video item '{item_id_final}' had 0 associated PILs in GPU batch map for averaging, resulting in empty embedding.",
                    extra={"item_id": item_id_final},
                )
            else:
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
            "embedding": item_embedding_list,  # Can be [] if no frames/image
            "error": None,  # No error at this stage if embedding is produced (even if empty)
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

    batch_total_duration_ms = (time.time() - batch_overall_start_time) * 1000
    successful_items_count = sum(
        1 for res in final_results_map.values() if not res.get("error")
    )
    parent_batch_logger.info(
        f"Batch '{batch_processing_id}' processing completed in {batch_total_duration_ms:.2f}ms. "
        f"Total items: {len(items_data_as_dicts)}, Items mapped for GPU: {len(gpu_batch_item_details_map)}, "
        f"Successful (final): {successful_items_count}, Errors: {len(items_data_as_dicts) - successful_items_count}",
        extra={
            "duration_ms": batch_total_duration_ms,
            "total_items": len(items_data_as_dicts),
            "gpu_candidate_items": len(
                gpu_batch_item_details_map
            ),  # items that passed preprocessing
            "all_pil_images_for_gpu_count": len(
                all_pil_images_for_gpu_batch
            ),  # total frames/images sent to GPU
            "final_success_count": successful_items_count,
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
    print(
        f"  FFMPEG_HWACCEL_METHOD={os.environ.get('FFMPEG_HWACCEL_METHOD')}"
    )  # Added for new feature
    print(
        f"  TEMP_DOWNLOAD_DIR={os.environ.get('TEMP_DOWNLOAD_DIR')}"
    )  # Added for new feature
