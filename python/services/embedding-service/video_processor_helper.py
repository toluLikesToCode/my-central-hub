#!/usr/bin/env python
# -*- coding: utf-8 -*-
# video_processor_helper.py
"""
Helper module for video processing tasks, including frame extraction.

This module contains the VideoProcessor class responsible for extracting
frames from video files using various methods, including hardware acceleration
and software fallbacks, along with scene detection and entropy-based sampling.
It also includes utility functions like compute_entropy.
"""

import os
import subprocess
import time
from datetime import datetime, timezone
import io
import uuid
import math
from PIL import Image  # type: ignore
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    # This import is only for type hinting and is not executed at runtime
    # to prevent circular dependencies. The actual EmbeddingLogger class
    # is expected to be defined in the embedding_service_helper module.
    from embedding_service_helper import EmbeddingLogger


# --- Helper Functions ---
def compute_entropy(image: Image.Image) -> float:
    """
    Compute the visual entropy of an image using histogram-based approach.

    Args:
        image: PIL Image object

    Returns:
        float: Entropy value (higher values indicate more visual information/complexity)
    """
    try:
        # Convert to grayscale for entropy calculation
        grayscale = image.convert("L")

        # Get histogram of pixel intensities (0-255)
        histogram = grayscale.histogram()

        # Calculate total pixels
        total_pixels = sum(histogram)
        if total_pixels == 0:
            return 0.0

        # Calculate Shannon entropy
        entropy = 0.0
        for count in histogram:
            if count > 0:
                probability = count / total_pixels
                entropy -= probability * math.log2(probability)

        return entropy

    except Exception:
        # Return a default low entropy value if computation fails
        return 0.0


class VideoProcessor:
    def __init__(
        self,
        num_frames: int,
        video_path: Optional[str] = None,
        logger: Optional["EmbeddingLogger"] = None,  # String literal for type hint
        executor: Optional[ThreadPoolExecutor] = None,
        request_id: Optional[str] = None,
        duration: Optional[float] = None,
        original_filename_hint: Optional[str] = None,
        hwaccel_method: Optional[str] = None,
    ):
        if not video_path:
            raise ValueError("VideoProcessor requires a valid video_path.")

        self.video_path: str = video_path
        self.num_frames = num_frames

        _effective_logger: "EmbeddingLogger"
        if logger is None:
            # Runtime import to break circular dependency and maintain API
            # This allows VideoProcessor to use a global embedding_logger if no logger is passed.
            from embedding_service_helper import (
                embedding_logger as global_embedding_logger_instance,
            )

            _effective_logger = global_embedding_logger_instance
        else:
            _effective_logger = logger
        self.logger = _effective_logger

        base_component = f"VideoProc-{original_filename_hint[:20] if original_filename_hint else os.path.basename(video_path)[:20]}"
        self.logger.set_component_name(
            f"{base_component}-{request_id[:8] if request_id else uuid.uuid4().hex[:8]}"
        )
        if request_id:  # This request_id is the item_specific_request_id
            self.logger.set_request_id(request_id)
            self.item_processing_request_id = request_id
        else:
            # Assumes the logger instance (either passed or global) has a request_id attribute
            self.item_processing_request_id = self.logger.request_id

        self.executor = executor
        self.hwaccel_method = (
            hwaccel_method  # This is the batch-level configured HW accel method
        )
        self._cv2_capture = (
            None  # OpenCV VideoCapture handle for fast software frame reads
        )
        self.extraction_events: List[Dict[str, Any]] = []

        self._add_event(
            "initialization",
            details={
                "configured_hwaccel_method": self.hwaccel_method,
                "video_path": self.video_path,
                "num_frames_requested": self.num_frames,
            },
        )

        if self.hwaccel_method:
            self.logger.info(
                f"VideoProcessor for '{self.video_path}' will attempt to use HWAccel: {self.hwaccel_method}"
            )

        self.duration = (
            duration if duration is not None else self._get_duration()
        )  # This will add duration events

        if self.duration is None or self.duration <= 0:
            err_msg = f"Could not determine video duration or duration is invalid for '{self.video_path}'."
            self.logger.error(
                err_msg,
                extra={"duration": self.duration, "video_path": self.video_path},
            )
            # Event for this already added by _get_duration if it fails
            raise ValueError(err_msg)

    def _add_event(self, event_type: str, details: Optional[Dict[str, Any]] = None):
        event = {
            "event_type": event_type,
            "timestamp_iso": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),  # MODIFIED TIMESTAMP
            "details": details or {},
        }
        self.extraction_events.append(event)
        # Optionally log high-priority events through the logger as well
        # self.logger.debug(f"VideoEvent: {event_type}", extra=details)

    def _get_duration(self) -> float:
        start_time = time.time()
        self.logger.debug(f"Getting duration for video: '{self.video_path}'")
        self._add_event("duration_check_start", {"video_path": self.video_path})
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
        try:
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
                stderr_output = result.stderr.strip() if result.stderr else "N/A"
                self.logger.error(
                    err_msg,
                    extra={
                        "stdout": result.stdout,
                        "stderr": stderr_output,
                        "video_path": self.video_path,
                    },
                )
                self._add_event(
                    "duration_check_failure",
                    {
                        "error": err_msg,
                        "stdout": result.stdout,
                        "stderr": stderr_output,
                        "command": " ".join(command),
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
            self._add_event(
                "duration_check_success",
                {
                    "duration_seconds": duration,
                    "time_taken_ms": (time.time() - start_time) * 1000,
                },
            )
            return duration

        except subprocess.TimeoutExpired as e_timeout:
            err_msg = f"ffprobe timeout getting duration for '{self.video_path}'"
            self.logger.error(
                err_msg,
                error=e_timeout,
                extra={"video_path": self.video_path, "command": " ".join(command)},
            )
            self._add_event(
                "duration_check_failure",
                {
                    "error": "TimeoutExpired",
                    "message": err_msg,
                    "command": " ".join(command),
                },
            )
            raise RuntimeError(err_msg) from e_timeout
        except Exception as e:
            err_msg = f"Failed to get video duration for '{self.video_path}': {e}"
            self.logger.error(
                err_msg,
                error=e,
                extra={"video_path": self.video_path, "command": " ".join(command)},
            )
            self._add_event(
                "duration_check_failure",
                {
                    "error": type(e).__name__,
                    "message": str(e),
                    "command": " ".join(command),
                },
            )
            raise RuntimeError(err_msg) from e

    def _extract_frame_hw_accelerated(self, time_sec: float) -> Image.Image:
        if not self.hwaccel_method:
            raise ValueError("HWAccel method not specified.")

        self.logger.debug(
            f"Attempting HW accelerated frame extraction (method: {self.hwaccel_method}) at {time_sec:.2f}s"
        )
        cmd_base = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",  # Can be changed to "debug" or "verbose" for more detailed ffmpeg output if needed
            "-hwaccel",
            self.hwaccel_method,
        ]
        # Ensure decoded frames stay on GPU initially for hwdownload to process
        if self.hwaccel_method == "cuda":  # Be specific for cuda
            cmd_base.extend(["-hwaccel_output_format", "cuda"])

        # MODIFIED video_filter: download as nv12 then convert pixel format to yuvj420p
        video_filter = "hwdownload,format=nv12,format=yuvj420p"

        command = cmd_base + [
            "-ss",
            str(time_sec),
            "-i",
            self.video_path,
            "-vf",
            video_filter,  # Use the modified filter chain
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "mjpeg",
            "-q:v",
            "2",  # Standard quality for MJPEG, good balance
            "-",
        ]

        try:
            self.logger.debug(f"Executing FFmpeg command: {' '.join(command)}")
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=25,
            )
            if not result.stdout:
                stderr_output = (
                    result.stderr.decode("utf-8", errors="ignore").strip()
                    if result.stderr
                    else "No stderr"
                )
                err_msg = f"No stdout. Stderr: {stderr_output}"
                self.logger.error(
                    f"ffmpeg (HWAccel: {self.hwaccel_method}) failed for '{self.video_path}' at {time_sec:.2f}s: {err_msg}",
                    extra={
                        "time_sec": time_sec,
                        "stderr": stderr_output,
                        "full_command": " ".join(command),
                    },
                )
                raise RuntimeError(
                    f"No frame data from HWAccel ({self.hwaccel_method}). Stderr: {stderr_output}. Command: {' '.join(command)}"
                )
            return Image.open(io.BytesIO(result.stdout)).convert("RGB")
        except subprocess.TimeoutExpired as e_timeout:
            self.logger.error(
                f"ffmpeg (HWAccel: {self.hwaccel_method}) timeout at {time_sec:.2f}s. Command: {' '.join(command)}",
                error=e_timeout,
                extra={"time_sec": time_sec, "full_command": " ".join(command)},
            )
            raise RuntimeError(
                f"ffmpeg (HWAccel: {self.hwaccel_method}) timeout extracting frame at {time_sec:.2f}s. Command: {' '.join(command)}"
            ) from e_timeout
        except subprocess.CalledProcessError as e_call:
            stderr_output = (
                e_call.stderr.decode("utf-8", errors="ignore").strip()
                if e_call.stderr
                else "N/A"
            )
            runtime_error_message = (
                f"HWAccel ({self.hwaccel_method}) frame extraction failed (code {e_call.returncode}) "
                f"at {time_sec:.2f}s. Stderr: {stderr_output}. Command: {' '.join(command)}"
            )
            self.logger.error(
                runtime_error_message,
                error=e_call,
                extra={
                    "time_sec": time_sec,
                    "return_code": e_call.returncode,
                    "stderr": stderr_output,
                    "full_command": " ".join(command),
                },
            )
            raise RuntimeError(runtime_error_message) from e_call
        except Exception as e_generic:
            self.logger.error(
                f"Frame extraction (HWAccel: {self.hwaccel_method}) failed at {time_sec:.2f}s: {e_generic}. Command: {' '.join(command)}",
                error=e_generic,
                extra={"time_sec": time_sec, "full_command": " ".join(command)},
            )
            raise RuntimeError(
                f"Failed to extract frame at {time_sec:.2f}s using HWAccel ({self.hwaccel_method}): {e_generic}. Command: {' '.join(command)}"
            ) from e_generic

    def _extract_frame_software(self, time_sec: float) -> Image.Image:
        self.logger.debug(
            f"Attempting software frame extraction using FFmpeg (PPM) at {time_sec:.2f}s for '{self.video_path}'."
        )

        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-hide_banner",
            "-threads",
            "0",
            "-ss",
            str(time_sec),
            "-i",
            self.video_path,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "ppm",
            "-",
        ]
        cmd_str_preview = (
            " ".join(command[:5])
            + f" -ss {str(time_sec)} -i ... "
            + " ".join(command[-5:])
        )

        try:
            self.logger.debug(
                f"Executing FFmpeg software extraction command: {' '.join(command)}"
            )

            process_start_time = time.time()
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=20,
            )
            process_duration_ms = (time.time() - process_start_time) * 1000

            if not result.stdout:
                stderr_output = (
                    result.stderr.decode("utf-8", errors="ignore").strip()
                    if result.stderr
                    else "No stderr"
                )
                err_msg = f"No stdout (frame data) from FFmpeg. Stderr: {stderr_output}"
                self.logger.error(
                    f"FFmpeg (software PPM) failed for '{self.video_path}' at {time_sec:.2f}s: {err_msg}",
                    extra={
                        "time_sec": time_sec,
                        "stderr": stderr_output,
                        "full_command": " ".join(command),
                        "video_path": self.video_path,
                        "duration_ms": process_duration_ms,
                    },
                )
                raise RuntimeError(
                    f"No frame data from software FFmpeg (PPM) at {time_sec:.2f}s. Stderr: {stderr_output}. Command preview: {cmd_str_preview}"
                )

            self.logger.debug(
                f"FFmpeg (software PPM) successfully extracted frame at {time_sec:.2f}s in {process_duration_ms:.2f}ms. Output size: {len(result.stdout)} bytes.",
                extra={
                    "time_sec": time_sec,
                    "video_path": self.video_path,
                    "duration_ms": process_duration_ms,
                    "output_bytes": len(result.stdout),
                },
            )
            return Image.open(io.BytesIO(result.stdout)).convert("RGB")

        except subprocess.TimeoutExpired as e_timeout:
            stderr_output = (
                e_timeout.stderr.decode("utf-8", errors="ignore").strip()
                if e_timeout.stderr
                else "N/A (timeout before stderr)"
            )
            self.logger.error(
                f"FFmpeg (software PPM) timeout extracting frame from '{self.video_path}' at {time_sec:.2f}s. Stderr: {stderr_output}",
                error=e_timeout,
                extra={
                    "time_sec": time_sec,
                    "full_command": " ".join(command),
                    "video_path": self.video_path,
                    "stderr_on_timeout": stderr_output,
                },
            )
            raise RuntimeError(
                f"FFmpeg (software PPM) timeout extracting frame at {time_sec:.2f}s. Command preview: {cmd_str_preview}. Stderr: {stderr_output}"
            ) from e_timeout

        except subprocess.CalledProcessError as e_call:
            stderr_output = (
                e_call.stderr.decode("utf-8", errors="ignore").strip()
                if e_call.stderr
                else "N/A"
            )
            self.logger.error(
                f"FFmpeg (software PPM) error (code {e_call.returncode}) for '{self.video_path}' at {time_sec:.2f}s. Stderr: {stderr_output}",
                error=e_call,
                extra={
                    "time_sec": time_sec,
                    "return_code": e_call.returncode,
                    "stderr": stderr_output,
                    "full_command": " ".join(command),
                    "video_path": self.video_path,
                },
            )
            raise RuntimeError(
                f"Software FFmpeg (PPM) frame extraction failed (code {e_call.returncode}) at {time_sec:.2f}s. Stderr: {stderr_output}. Command preview: {cmd_str_preview}"
            ) from e_call

        except FileNotFoundError:
            self.logger.error(
                f"FFmpeg executable not found. Please ensure FFmpeg is installed and in PATH. Video: '{self.video_path}', Time: {time_sec:.2f}s",
                extra={
                    "time_sec": time_sec,
                    "full_command": " ".join(command),
                    "video_path": self.video_path,
                },
            )
            raise RuntimeError(
                f"FFmpeg executable not found for software extraction. Command preview: {cmd_str_preview}"
            )

        except Exception as e_generic:
            stderr_info = "N/A"
            if hasattr(e_generic, "stderr") and e_generic.stderr:  # type: ignore
                try:
                    stderr_info = e_generic.stderr.decode("utf-8", errors="ignore").strip()  # type: ignore
                except Exception:
                    stderr_info = "Error decoding stderr"

            self.logger.error(
                f"Generic error during FFmpeg (software PPM) frame extraction from '{self.video_path}' at {time_sec:.2f}s: {e_generic}. Stderr: {stderr_info}",
                error=e_generic,
                extra={
                    "time_sec": time_sec,
                    "full_command": " ".join(command),
                    "video_path": self.video_path,
                    "stderr_info": stderr_info,
                },
            )
            raise RuntimeError(
                f"Failed to extract frame at {time_sec:.2f}s using software FFmpeg (PPM): {e_generic}. Command preview: {cmd_str_preview}. Stderr: {stderr_info}"
            ) from e_generic

    def extract_frame(
        self, time_sec: float
    ) -> Tuple[Optional[Image.Image], List[Dict[str, Any]]]:
        frame_events: List[Dict[str, Any]] = []
        frame_pil = None

        def add_frame_event(event_type: str, details: Optional[Dict[str, Any]] = None):
            merged_details = {"video_timestamp_sec": time_sec}
            if details:
                merged_details.update(details)
            event = {
                "event_type": event_type,
                "timestamp_iso": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),  # MODIFIED TIMESTAMP
                "details": merged_details,
            }
            frame_events.append(event)

        try:
            if self.hwaccel_method:
                add_frame_event(
                    "hw_extraction_attempt", {"hw_method": self.hwaccel_method}
                )
                try:
                    frame_pil = self._extract_frame_hw_accelerated(time_sec)
                    add_frame_event(
                        "hw_extraction_success", {"hw_method": self.hwaccel_method}
                    )
                except Exception as e_hw:
                    self.logger.warning(
                        f"HW accel ({self.hwaccel_method}) failed for ts {time_sec:.2f}s: {e_hw}. Falling back.",
                        extra={"time_sec": time_sec, "error_type": type(e_hw).__name__},
                    )
                    add_frame_event(
                        "hw_extraction_failed_fallback_to_sw",
                        {
                            "hw_method": self.hwaccel_method,
                            "error_type": type(e_hw).__name__,
                            "error_message": str(e_hw),
                        },
                    )
                    # Fall through to software
            else:
                add_frame_event(
                    "hw_accel_not_configured_using_sw",
                    {
                        # "hw_method": self.hwaccel_method, # Redundant if it's None/empty
                        "time_sec": time_sec,
                        "video_path": self.video_path,
                    },
                )

            if not frame_pil:  # If HW not configured, or HW failed
                reason_for_sw = (
                    "direct_attempt_no_hw_config"
                    if not self.hwaccel_method
                    else "fallback_after_hw_failure"
                )
                add_frame_event("sw_extraction_attempt", {"reason": reason_for_sw})
                try:
                    frame_pil = self._extract_frame_software(time_sec)
                    add_frame_event("sw_extraction_success", {"reason": reason_for_sw})
                except Exception as e_sw:
                    self.logger.error(
                        f"Software extraction failed for ts {time_sec:.2f}s (reason: {reason_for_sw}): {e_sw}",
                        error=e_sw,
                        extra={"time_sec": time_sec},
                    )
                    add_frame_event(
                        "sw_extraction_failed",
                        {
                            "reason": reason_for_sw,
                            "error_type": type(e_sw).__name__,
                            "error_message": str(e_sw),
                        },
                    )
                    # Frame_pil remains None
        except Exception as e:
            # Catch-all for any unexpected error in the extraction logic
            self.logger.error(
                f"Unexpected error in extract_frame at {time_sec:.2f}s: {e}",
                error=e,
                extra={"time_sec": time_sec},
            )
            add_frame_event(
                "frame_extraction_error",
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            frame_pil = None

        return frame_pil, frame_events

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
            selected_times = [start_offset + effective_duration / 2.0]
            if effective_duration <= 0:
                selected_times = [self.duration / 2.0]
            method_used = "single_middle_frame"
        elif effective_duration <= 0.1:
            middle_time = self.duration / 2.0
            selected_times = [
                middle_time
            ] * self.num_frames  # Duplicate for requested count
            method_used = "middle_frame_duplicated_short_video"
        elif self.num_frames > 1:
            # Ensure num_frames-1 is not zero for division
            selected_times = [
                start_offset + (i * effective_duration / (self.num_frames - 1))
                for i in range(self.num_frames)
            ]
            method_used = "uniform_spread_with_offset"
        else:
            selected_times = [self.duration / 2.0]  # Fallback if logic missed
            method_used = "fallback_single_middle_frame"

        epsilon = 0.001
        selected_times = sorted(
            list(set(max(0.0, min(t, self.duration - epsilon)) for t in selected_times))
        )
        if not selected_times and self.duration > 0:
            selected_times = [max(0.0, self.duration / 2.0 - epsilon)]

        debug_metadata["method_used"] = method_used
        debug_metadata["candidate_timestamps"] = list(selected_times)
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
            "frame_sampling_details": frame_sampling_debug_meta,
            "video_processor_instance_hwaccel_method": self.hwaccel_method,
            "video_duration_s": self.duration,
            "item_processing_request_id": self.item_processing_request_id,
        }
        final_debug_meta["num_timestamps_from_sampler"] = len(timestamps)

        if not timestamps:
            self.logger.warning(
                f"No timestamps returned by sampler for '{self.video_path}'.",
                extra=final_debug_meta,
            )
            self._add_event("no_timestamps_from_sampler_error", final_debug_meta)
            final_debug_meta["detailed_extraction_events"] = self.extraction_events
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

        frame_extraction_attempt_details_list: List[Dict] = []

        if use_parallel_extraction and self.executor:
            future_to_ts_map = {
                self.executor.submit(self.extract_frame, ts): ts
                for ts in actual_timestamps_to_extract
            }
            for future in concurrent.futures.as_completed(future_to_ts_map):
                ts = future_to_ts_map[future]
                try:
                    frame_pil, frame_attempt_events = future.result()
                    frame_extraction_attempt_details_list.extend(frame_attempt_events)
                    if frame_pil:
                        extracted_frames_pil.append(frame_pil)
                    else:
                        extraction_errors_count += 1
                except Exception as e_frame:
                    extraction_errors_count += 1
                    self.logger.error(
                        f"Parallel extract_frame wrapper failed for ts={ts:.2f}s: {e_frame}",
                        error=e_frame,
                        extra={"timestamp": ts},
                    )
                    generic_failure_event = {
                        "event_type": "parallel_extraction_future_error",
                        "timestamp_iso": datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "details": {
                            "video_timestamp_sec": ts,
                            "error_type": type(e_frame).__name__,
                            "error_message": str(e_frame),
                        },
                    }
                    frame_extraction_attempt_details_list.append(generic_failure_event)
        else:  # Sequential extraction
            for ts in actual_timestamps_to_extract:
                try:
                    frame_pil, frame_attempt_events = self.extract_frame(ts)
                    frame_extraction_attempt_details_list.extend(frame_attempt_events)
                    if frame_pil:
                        extracted_frames_pil.append(frame_pil)
                    else:
                        extraction_errors_count += 1
                except Exception as e_frame:
                    extraction_errors_count += 1
                    self.logger.error(
                        f"Sequential extract_frame wrapper failed for ts={ts:.2f}s: {e_frame}",
                        error=e_frame,
                        extra={"timestamp": ts},
                    )
                    generic_failure_event = {
                        "event_type": "sequential_extraction_wrapper_error",
                        "timestamp_iso": datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "details": {
                            "video_timestamp_sec": ts,
                            "error_type": type(e_frame).__name__,
                            "error_message": str(e_frame),
                        },
                    }
                    frame_extraction_attempt_details_list.append(generic_failure_event)

        self.extraction_events.extend(frame_extraction_attempt_details_list)

        if extraction_errors_count > 0:
            final_debug_meta["frame_extraction_error_count"] = extraction_errors_count

        if not extracted_frames_pil and len(actual_timestamps_to_extract) > 0:
            err_msg = f"All {len(actual_timestamps_to_extract)} frame extractions failed for '{self.video_path}'."
            self.logger.error(err_msg, extra=final_debug_meta)
            self._add_event(
                "all_frame_extractions_failed_error",
                {"message": err_msg, **final_debug_meta},
            )
            final_debug_meta["detailed_extraction_events"] = self.extraction_events
            return [], {"error": err_msg, **final_debug_meta}

        if 0 < len(extracted_frames_pil) < self.num_frames and extracted_frames_pil:
            warn_msg = f"Extracted {len(extracted_frames_pil)} frames, but {self.num_frames} were requested for '{self.video_path}'. Duplicating last good frame."
            self.logger.warning(warn_msg, extra=final_debug_meta)
            self._add_event(
                "partial_frames_extracted_padding_warning",
                {
                    "message": warn_msg,
                    "extracted_count": len(extracted_frames_pil),
                    "requested_count": self.num_frames,
                },
            )
            last_good_frame = extracted_frames_pil[-1]
            num_to_add = self.num_frames - len(extracted_frames_pil)
            extracted_frames_pil.extend(
                [last_good_frame.copy() for _ in range(num_to_add)]
            )
        elif not extracted_frames_pil and self.num_frames > 0:
            err_msg_alt = f"No frames were ultimately available for '{self.video_path}' though {self.num_frames} were requested."
            self.logger.error(err_msg_alt, extra=final_debug_meta)
            self._add_event(
                "no_frames_available_unexpected_error",
                {"message": err_msg_alt, **final_debug_meta},
            )
            final_debug_meta["detailed_extraction_events"] = self.extraction_events
            return [], {"error": err_msg_alt, **final_debug_meta}

        final_debug_meta["num_final_pil_frames_returned"] = len(extracted_frames_pil)
        total_extraction_duration_ms = (time.time() - overall_start_time) * 1000
        self.logger.info(
            f"Successfully prepared {len(extracted_frames_pil)} frames in {total_extraction_duration_ms:.2f}ms for '{self.video_path}'.",
            extra={"duration_ms": total_extraction_duration_ms, **final_debug_meta},
        )

        self._add_event(
            "frame_extraction_completed",
            {
                "final_pil_frames": len(extracted_frames_pil),
                "duration_ms": total_extraction_duration_ms,
            },
        )
        final_debug_meta["detailed_extraction_events"] = self.extraction_events
        return extracted_frames_pil, final_debug_meta

    def __del__(self):
        # Placeholder for any cleanup, though not strictly necessary with current implementation
        # If _cv2_capture were used, it would be released here:
        # if self._cv2_capture:
        #     self._cv2_capture.release()
        pass
