#!/usr/bin/env python
# -*- coding: utf-8 -*-
# video_processor_helper.py
"""
Helper module for video processing tasks, including frame extraction.

This module contains the VideoProcessor class responsible for extracting
frames from video files using various methods, including hardware acceleration
and software fallbacks. It prioritizes a batch frame extraction method using
a single FFmpeg process per video for efficiency.
"""

import os
import subprocess
import time
from datetime import datetime, timezone
import io
import uuid
import math
from PIL import Image, UnidentifiedImageError  # type: ignore

# Removed concurrent.futures and ThreadPoolExecutor from here as it's not used directly for per-video frame extraction anymore
from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    # This import is only for type hinting and is not executed at runtime
    # to prevent circular dependencies. The actual EmbeddingLogger class
    # is expected to be defined in the embedding_service_helper module.
    from embedding_service_helper import EmbeddingLogger  # type: ignore


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
        logger: Optional["EmbeddingLogger"] = None,
        executor: Optional[
            Any
        ] = None,  # Kept for API compatibility, not used internally for frame extraction logic
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
            from embedding_service_helper import embedding_logger as global_embedding_logger_instance  # type: ignore

            _effective_logger = global_embedding_logger_instance
        else:
            _effective_logger = logger
        self.logger = _effective_logger

        # Ensure request_id is set for the logger instance
        self.item_processing_request_id = request_id or getattr(
            self.logger, "request_id", uuid.uuid4().hex
        )
        if (
            not hasattr(self.logger, "request_id")
            or self.logger.request_id != self.item_processing_request_id
        ):
            self.logger.set_request_id(self.item_processing_request_id)

        base_component_name = f"VideoProc-{original_filename_hint[:20] if original_filename_hint else os.path.basename(video_path)[:20]}"
        log_suffix = (
            self.item_processing_request_id.split("_")[-1]
            if "_" in self.item_processing_request_id
            else self.item_processing_request_id[:8]
        )
        self.logger.set_component_name(f"{base_component_name}-{log_suffix}")

        self.hwaccel_method = hwaccel_method
        self.extraction_events: List[Dict[str, Any]] = []

        self._add_event(
            "initialization_start",
            details={
                "video_path": self.video_path,
                "num_frames_requested": self.num_frames,
                "configured_hwaccel_method": self.hwaccel_method,
            },
        )

        try:
            self.duration = (
                duration if duration is not None else self._get_duration()
            )  # This will add its own events
            if self.duration is None or self.duration <= 0:
                raise ValueError(f"Video duration is invalid: {self.duration}")

            # Get average frame rate (FPS) for frame-based selection
            self.avg_frame_rate = self._get_avg_frame_rate()

            self._add_event(
                "initialization_complete",
                {
                    "resolved_video_duration_s": self.duration,
                    "resolved_avg_frame_rate": self.avg_frame_rate,
                },
            )
        except Exception as e:
            err_msg = f"Initialization failed for VideoProcessor: {e}"
            self.logger.error(
                err_msg, extra={"error": str(e), "video_path": self.video_path}
            )
            self._add_event(
                "initialization_failure",
                {"error": str(e), "video_path": self.video_path},
            )
            raise ValueError(err_msg) from e

    def _add_event(self, event_type: str, details: Optional[Dict[str, Any]] = None):
        event_details = details or {}
        # Ensure sensitive or overly verbose data is not added by default unless specified
        # For example, avoid adding full command strings to every minor event unless it's an error event.
        event = {
            "event_type": event_type,
            "timestamp_iso": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "details": event_details,
        }
        self.extraction_events.append(event)

    def _get_duration(self) -> float:
        operation_name = "get_duration_ffprobe"
        op_start_time = time.monotonic()

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
        event_details = {
            "video_path": self.video_path,
            "command_preview": "ffprobe -show_entries format=duration...",
        }
        self._add_event(f"{operation_name}_start", event_details)

        try:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
                timeout=15,
            )
            duration_str = process.stdout.strip()
            if not duration_str or duration_str == "N/A":
                raise ValueError(
                    f"ffprobe returned invalid duration string: '{duration_str}'"
                )

            duration = float(duration_str)
            time_taken_ms = (time.monotonic() - op_start_time) * 1000
            self._add_event(
                f"{operation_name}_success",
                {"duration_seconds": duration, "time_taken_ms": time_taken_ms},
            )
            self.logger.debug(
                f"Duration {duration:.2f}s in {time_taken_ms:.2f}ms for '{os.path.basename(self.video_path)}'"
            )
            return duration

        except subprocess.TimeoutExpired as e_timeout:
            time_taken_ms = (time.monotonic() - op_start_time) * 1000
            stderr_output = "N/A"
            # Safely handle stderr which may not be present or may be None
            stderr = getattr(e_timeout, "stderr", None)
            if stderr is not None:
                try:
                    stderr_output = stderr.decode("utf-8", errors="ignore").strip()
                except (AttributeError, UnicodeDecodeError):
                    pass
            err_msg = f"Timeout ({time_taken_ms:.0f}ms) getting duration"
            self.logger.warning(
                f"{err_msg} for '{self.video_path}'. Stderr: {stderr_output}",
                extra={"error": str(e_timeout)},
            )
            self._add_event(
                f"{operation_name}_timeout",
                {
                    "error": str(e_timeout),
                    "stderr": stderr_output,
                    "command": " ".join(command),
                    "time_taken_ms": time_taken_ms,
                },
            )
            raise RuntimeError(f"{err_msg}. Stderr: {stderr_output}") from e_timeout
        except Exception as e:  # Includes CalledProcessError
            time_taken_ms = (time.monotonic() - op_start_time) * 1000
            stderr_output = "N/A"
            # Safely handle stderr which may not be present or may be None
            stderr = getattr(e, "stderr", None)
            if stderr is not None:
                try:
                    stderr_output = stderr.decode("utf-8", errors="ignore").strip()
                except (AttributeError, UnicodeDecodeError):
                    pass
            err_msg = (
                f"Failed ({type(e).__name__}, {time_taken_ms:.0f}ms) getting duration"
            )
            self.logger.error(
                f"{err_msg} for '{self.video_path}': {e}. Stderr: {stderr_output}",
                extra={"error": str(e)},
            )
            self._add_event(
                f"{operation_name}_failure",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "stderr": stderr_output,
                    "command": " ".join(command),
                    "time_taken_ms": time_taken_ms,
                },
            )
            raise RuntimeError(f"{err_msg}: {e}. Stderr: {stderr_output}") from e

    def _execute_ffmpeg_for_multiple_frames(
        self, timestamps: List[float], use_hw_accel: bool
    ) -> Tuple[List[Optional[Image.Image]], Dict[str, Any]]:
        """
        Core FFmpeg execution logic for extracting multiple frames in a single process.
        Uses frame indices (n) instead of timestamps (t) for more reliable frame selection.
        Can be used with or without hardware acceleration.
        """
        if not timestamps:
            return [], {
                "extracted_frame_count": 0,
                "requested_frame_count": 0,
                "status": "no_timestamps",
            }

        op_start_time = time.monotonic()

        # Convert timestamps to frame indices based on average frame rate
        # Keep track of original indices for mapping results back
        timestamp_to_frame_indices: Dict[float, int] = {}
        frame_index_to_timestamp_indices: Dict[int, List[int]] = {}

        for i, ts in enumerate(timestamps):
            # Calculate frame index from timestamp and FPS
            frame_index = int(round(ts * self.avg_frame_rate))
            timestamp_to_frame_indices[ts] = frame_index

            # Map frame indices back to original timestamp indices
            # (multiple timestamps could map to same frame)
            if frame_index not in frame_index_to_timestamp_indices:
                frame_index_to_timestamp_indices[frame_index] = []
            frame_index_to_timestamp_indices[frame_index].append(i)

        # Get unique sorted frame indices for the select filter
        unique_sorted_frame_indices = sorted(
            list(set(timestamp_to_frame_indices.values()))
        )

        # Build select filter based on frame indices (n) rather than timestamps (t)
        select_filter_parts = [f"eq(n,{n})" for n in unique_sorted_frame_indices]
        select_filter_str = "select='" + "+".join(select_filter_parts) + "'"

        ffmpeg_cmd = ["ffmpeg", "-y", "-loglevel", "warning", "-hide_banner"]

        current_hwaccel_method = self.hwaccel_method if use_hw_accel else None

        if current_hwaccel_method:
            ffmpeg_cmd.extend(["-hwaccel", current_hwaccel_method])
            if current_hwaccel_method == "cuda":
                ffmpeg_cmd.extend(["-hwaccel_output_format", "cuda"])

        ffmpeg_cmd.extend(["-i", self.video_path])  # Input file

        filter_complex_parts = [select_filter_str]
        if current_hwaccel_method:  # If HW decoding, need to download frames
            filter_complex_parts.append("hwdownload")
            filter_complex_parts.append(
                "format=nv12"
            )  # Common intermediate after hwdownload

        filter_complex_parts.append("format=yuvj420p")  # Final format for MJPEG encoder

        ffmpeg_cmd.extend(["-vf", ",".join(filter_complex_parts)])
        ffmpeg_cmd.extend(
            [
                "-fps_mode",
                "vfr",  # Using modern fps_mode option for frame-accurate selection
                "-f",
                "image2pipe",  # Output to pipe
                "-c:v",
                "mjpeg",  # MJPEG is good for streaming multiple distinct images
                "-q:v",
                "2",  # MJPEG quality
                "-",  # Output to stdout
            ]
        )

        exec_details = {
            "requested_timestamps_count": len(timestamps),
            "unique_sorted_frame_indices_count": len(unique_sorted_frame_indices),
            "hw_method_attempted": current_hwaccel_method or "software",
            "selection_method": "frame_index",  # Indicate we're using frame indices instead of timestamps
            "avg_frame_rate_used": self.avg_frame_rate,
            "ffmpeg_command": " ".join(ffmpeg_cmd),  # Log the full command
        }
        self._add_event("ffmpeg_command_execution_start", exec_details)
        self.logger.debug(f"Executing FFmpeg: {' '.join(ffmpeg_cmd)}")

        frames_pil_ordered: List[Optional[Image.Image]] = [None] * len(timestamps)
        process_return_code = -1
        ffmpeg_stderr_output = ""

        # Initialize variables here to avoid "possibly unbound" errors
        process_start_time = time.monotonic()
        # Increased timeout: base + per frame allowance
        timeout_seconds = 25 + (len(unique_sorted_frame_indices) * 2.5)

        try:
            process = subprocess.Popen(
                ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout_data, stderr_data = process.communicate(timeout=timeout_seconds)

            process_duration_ms = (time.monotonic() - process_start_time) * 1000
            process_return_code = process.returncode
            ffmpeg_stderr_output = stderr_data.decode("utf-8", errors="ignore").strip()

            if process.returncode != 0:
                self.logger.error(
                    f"FFmpeg process failed (code {process.returncode}) for '{self.video_path}'. Stderr: {ffmpeg_stderr_output}"
                )
                exec_details.update(
                    {
                        "status": "ffmpeg_error",
                        "return_code": process.returncode,
                        "stderr": ffmpeg_stderr_output,
                        "duration_ms": process_duration_ms,
                    }
                )
                self._add_event("ffmpeg_command_execution_error", exec_details)
                return frames_pil_ordered, exec_details  # Returns list of Nones

            # Parse MJPEG stream
            image_data_stream = io.BytesIO(stdout_data)
            parsed_image_count = 0
            frames_by_index: Dict[int, Image.Image] = {}

            while parsed_image_count < len(unique_sorted_frame_indices):
                try:
                    img = Image.open(image_data_stream)
                    img = img.convert("RGB")  # Ensure consistent format

                    # Store the image by its corresponding frame index
                    if parsed_image_count < len(unique_sorted_frame_indices):
                        frame_index = unique_sorted_frame_indices[parsed_image_count]
                        frames_by_index[frame_index] = img
                        parsed_image_count += 1

                except (
                    UnidentifiedImageError
                ):  # Expected at end of stream or if data is insufficient
                    if (
                        image_data_stream.tell() < len(stdout_data) - 2
                    ):  # Check if not truly at EOF
                        self.logger.warning(
                            f"UnidentifiedImageError parsing frame {parsed_image_count + 1}/{len(unique_sorted_frame_indices)} but not at EOF. Stream pos: {image_data_stream.tell()}/{len(stdout_data)}"
                        )
                    break
                except Exception as e_parse:
                    self.logger.error(
                        f"Error parsing frame {parsed_image_count + 1} from MJPEG stream: {e_parse}",
                        extra={"video_path": self.video_path},
                    )
                    # Mark this frame as failed by leaving it as None
                    break  # Stop trying to parse further if an unexpected error occurs

            # Map the extracted frames back to their original timestamp indices
            assigned_frames_count = 0
            timestamp_to_frame_mapping = {}

            for frame_index, frame_img in frames_by_index.items():
                timestamp_indices = frame_index_to_timestamp_indices.get(
                    frame_index, []
                )
                for timestamp_index in timestamp_indices:
                    frames_pil_ordered[timestamp_index] = frame_img
                    assigned_frames_count += 1
                    # Record which timestamp got which frame for debugging
                    if timestamp_index < len(timestamps):
                        timestamp_to_frame_mapping[timestamps[timestamp_index]] = (
                            frame_index
                        )

            # Add mapping details to the execution details for debugging
            exec_details["timestamp_to_frame_mapping"] = timestamp_to_frame_mapping

            exec_details.update(
                {
                    "status": "success",
                    "extracted_frame_count": parsed_image_count,
                    "assigned_frames_count": assigned_frames_count,
                    "duration_ms": process_duration_ms,
                    "stdout_bytes": len(stdout_data),
                    "stderr_preview": ffmpeg_stderr_output[
                        :200
                    ],  # Log a preview of stderr even on success
                }
            )
            self._add_event("ffmpeg_command_execution_success", exec_details)

            if parsed_image_count < len(unique_sorted_frame_indices):
                self.logger.warning(
                    f"Expected {len(unique_sorted_frame_indices)} frames, but only parsed {parsed_image_count} for '{self.video_path}'."
                )
                self._add_event(
                    "ffmpeg_frame_count_mismatch",
                    {
                        "expected": len(unique_sorted_frame_indices),
                        "parsed": parsed_image_count,
                    },
                )

        except subprocess.TimeoutExpired as e_timeout:
            process_duration_ms = (
                (time.monotonic() - process_start_time) * 1000
                if "process_start_time" in locals()
                else (time.monotonic() - op_start_time) * 1000
            )
            ffmpeg_stderr_output = "N/A"
            # Safely handle stderr which may not be present or may be None
            stderr = getattr(e_timeout, "stderr", None)
            if stderr is not None:
                try:
                    ffmpeg_stderr_output = stderr.decode(
                        "utf-8", errors="ignore"
                    ).strip()
                except (AttributeError, UnicodeDecodeError):
                    pass
            self.logger.error(
                f"FFmpeg process timeout ({timeout_seconds:.1f}s) for '{self.video_path}'. Stderr: {ffmpeg_stderr_output}"
            )
            exec_details.update(
                {
                    "status": "timeout",
                    "stderr": ffmpeg_stderr_output,
                    "duration_ms": process_duration_ms,
                }
            )
            self._add_event("ffmpeg_command_execution_timeout", exec_details)
        except Exception as e_general:
            process_duration_ms = (
                (time.monotonic() - process_start_time) * 1000
                if "process_start_time" in locals()
                else (time.monotonic() - op_start_time) * 1000
            )
            self.logger.error(
                f"Unexpected error during FFmpeg execution for '{self.video_path}': {e_general}"
            )
            exec_details.update(
                {
                    "status": "unexpected_error",
                    "error": str(e_general),
                    "duration_ms": process_duration_ms,
                }
            )
            self._add_event("ffmpeg_command_execution_unexpected_error", exec_details)

        return frames_pil_ordered, exec_details

    def get_advanced_sample_times(self) -> Tuple[List[float], Dict[str, Any]]:
        op_start_time = time.monotonic()
        self.logger.debug(
            f"Calculating sample times for '{os.path.basename(self.video_path)}' (duration: {self.duration:.2f}s, frames: {self.num_frames})"
        )

        sampling_details: Dict[str, Any] = {
            "num_requested_frames": self.num_frames,
            "video_duration_s": self.duration,
        }

        if self.num_frames <= 0:
            sampling_details.update(
                {
                    "method_used": "none",
                    "reason": "num_frames_zero_or_negative",
                    "candidate_timestamps": [],
                }
            )
            self._add_event("get_sample_times_result", sampling_details)
            return [], sampling_details
        if self.duration <= 0:  # Should be caught by __init__ but defensive
            sampling_details.update(
                {
                    "method_used": "none",
                    "reason": "duration_zero_or_negative",
                    "candidate_timestamps": [],
                }
            )
            self._add_event("get_sample_times_result", sampling_details)
            return [], sampling_details

        start_offset = min(0.5, self.duration * 0.02)  # Avoid very start
        end_offset = min(0.5, self.duration * 0.02)  # Avoid very end
        effective_duration = self.duration - start_offset - end_offset

        method_used: str
        selected_times: List[float]

        if self.num_frames == 1:
            target_time = (
                start_offset + effective_duration / 2.0
                if effective_duration > 0
                else self.duration / 2.0
            )
            selected_times = [target_time]
            method_used = "single_middle_frame_with_offset"
        elif (
            effective_duration <= 0.1
        ):  # Video too short for meaningful spread after offsets
            target_time = self.duration / 2.0
            selected_times = [
                target_time
            ] * self.num_frames  # Duplicate the middle timestamp
            method_used = "middle_frame_duplicated_for_short_video"
        else:  # Spread num_frames across effective_duration
            # If num_frames is 1, (num_frames - 1) would be 0. This case is handled above.
            selected_times = [
                start_offset + (i * effective_duration / (self.num_frames - 1))
                for i in range(self.num_frames)
            ]
            method_used = "uniform_spread_with_offset"

        # Ensure timestamps are within valid range [0, duration] and distinct
        # A small epsilon to prevent requesting frame exactly at duration or slightly beyond due to float precision
        epsilon = 1e-6
        unique_valid_times = sorted(
            list(
                set(max(0.0, min(ts, self.duration - epsilon)) for ts in selected_times)
            )
        )

        # If filtering/clamping resulted in no timestamps for a valid duration video, pick middle.
        if not unique_valid_times and self.duration > 0:
            unique_valid_times = [max(0.0, self.duration / 2.0 - epsilon)]
            method_used += (
                "_fallback_to_middle"  # Append to method_used to indicate adjustment
            )

        selected_times = unique_valid_times

        sampling_details.update(
            {
                "method_used": method_used,
                "candidate_timestamps": selected_times,
                "calculated_effective_duration_s": effective_duration,
                "calculated_start_offset_s": start_offset,
                "time_taken_ms": (time.monotonic() - op_start_time) * 1000,
            }
        )
        self._add_event("get_sample_times_result", sampling_details)
        self.logger.debug(
            f"Selected {len(selected_times)} timestamps ({method_used}) for '{os.path.basename(self.video_path)}'",
            extra=sampling_details,
        )
        return selected_times, sampling_details

    def extract_frames(self) -> Tuple[List[Image.Image], Dict[str, Any]]:
        overall_op_start_time = time.monotonic()
        self.logger.info(
            f"Starting frame extraction for '{os.path.basename(self.video_path)}', {self.num_frames} frames requested."
        )

        timestamps, frame_sampling_debug_meta = self.get_advanced_sample_times()

        # Initialize debug metadata for the overall operation
        final_debug_meta = {
            "frame_sampling_details": frame_sampling_debug_meta,
            "video_duration_s": self.duration,
            "video_avg_frame_rate": self.avg_frame_rate,
            "selection_method": "frame_index",  # Indicate we're using frame number-based selection
            "item_processing_request_id": self.item_processing_request_id,
            "num_timestamps_from_sampler": len(timestamps),
            "initial_hwaccel_method": self.hwaccel_method,
        }

        if not timestamps:
            self._add_event(
                "extract_frames_aborted_no_timestamps",
                {"reason": "Sampler returned no timestamps."},
            )
            final_debug_meta["error"] = "No timestamps from sampler."
            final_debug_meta["detailed_extraction_events"] = self.extraction_events
            return [], final_debug_meta

        actual_timestamps_to_extract = timestamps[: self.num_frames]
        final_debug_meta["actual_timestamps_for_extraction"] = (
            actual_timestamps_to_extract
        )

        extracted_pil_frames: List[Image.Image] = []
        successful_extraction_count = 0

        # Attempt 1: Hardware Acceleration (if configured)
        attempted_hw = False
        if self.hwaccel_method:
            attempted_hw = True
            self._add_event(
                "batch_extraction_attempt_hw_start",
                {
                    "hwaccel_method": self.hwaccel_method,
                    "num_timestamps": len(actual_timestamps_to_extract),
                },
            )
            frames_hw_optional, hw_exec_details = (
                self._execute_ffmpeg_for_multiple_frames(
                    actual_timestamps_to_extract, use_hw_accel=True
                )
            )
            final_debug_meta["hw_extraction_execution_details"] = hw_exec_details

            successful_hw_frames = [f for f in frames_hw_optional if f is not None]
            successful_extraction_count = len(successful_hw_frames)

            if successful_extraction_count == len(actual_timestamps_to_extract):
                extracted_pil_frames = successful_hw_frames
                self._add_event(
                    "batch_extraction_attempt_hw_success_full",
                    {"extracted_count": successful_extraction_count},
                )
            else:
                self.logger.warning(
                    f"HW batch extraction yielded {successful_extraction_count}/{len(actual_timestamps_to_extract)} frames for '{self.video_path}'. Stderr: {hw_exec_details.get('stderr', '')[:200]}"
                )
                self._add_event(
                    "batch_extraction_attempt_hw_partial_or_fail",
                    {
                        "extracted_count": successful_extraction_count,
                        "requested_count": len(actual_timestamps_to_extract),
                        "stderr_preview": hw_exec_details.get("stderr", "")[:200],
                    },
                )
                # Proceed to software fallback if HW didn't get all frames or failed
                extracted_pil_frames = []  # Reset, as we will try to get all via SW
                successful_extraction_count = 0

        # Attempt 2: Software Fallback (if HW was not configured, or failed to get all frames)
        if not extracted_pil_frames or successful_extraction_count < len(
            actual_timestamps_to_extract
        ):
            fallback_reason = (
                "hw_not_configured"
                if not attempted_hw
                else "hw_extraction_incomplete_or_failed"
            )
            self._add_event(
                "batch_extraction_attempt_sw_start",
                {
                    "reason": fallback_reason,
                    "num_timestamps": len(actual_timestamps_to_extract),
                },
            )
            self.logger.info(
                f"Using software batch extraction for '{self.video_path}' (reason: {fallback_reason})."
            )

            frames_sw_optional, sw_exec_details = (
                self._execute_ffmpeg_for_multiple_frames(
                    actual_timestamps_to_extract, use_hw_accel=False
                )
            )
            final_debug_meta["sw_extraction_execution_details"] = sw_exec_details

            successful_sw_frames = [f for f in frames_sw_optional if f is not None]
            successful_extraction_count = len(successful_sw_frames)
            extracted_pil_frames = (
                successful_sw_frames  # This is now the definitive set
            )

            if successful_extraction_count == len(actual_timestamps_to_extract):
                self._add_event(
                    "batch_extraction_attempt_sw_success_full",
                    {"extracted_count": successful_extraction_count},
                )
            else:
                self.logger.error(
                    f"Software batch extraction also failed to get all frames ({successful_extraction_count}/{len(actual_timestamps_to_extract)}) for '{self.video_path}'. Stderr: {sw_exec_details.get('stderr', '')[:200]}"
                )
                self._add_event(
                    "batch_extraction_attempt_sw_partial_or_fail",
                    {
                        "extracted_count": successful_extraction_count,
                        "requested_count": len(actual_timestamps_to_extract),
                        "stderr_preview": sw_exec_details.get("stderr", "")[:200],
                    },
                )

        # Padding if fewer frames were extracted than requested (and num_frames > 0)
        if self.num_frames > 0:
            if not extracted_pil_frames:
                err_msg = f"All frame extraction attempts (HW and SW batch) failed for '{self.video_path}'."
                self.logger.error(err_msg, extra=final_debug_meta)
                self._add_event("all_extraction_methods_failed", {"message": err_msg})
                final_debug_meta["error"] = err_msg
            elif len(extracted_pil_frames) < self.num_frames:
                warn_msg = f"Successfully extracted {len(extracted_pil_frames)} frames, but {self.num_frames} were requested. Padding with last good frame."
                self.logger.warning(warn_msg, extra={"video_path": self.video_path})
                self._add_event(
                    "padding_extracted_frames",
                    {
                        "extracted_count": len(extracted_pil_frames),
                        "requested_count": self.num_frames,
                    },
                )
                last_good_frame = extracted_pil_frames[-1]
                num_to_add = self.num_frames - len(extracted_pil_frames)
                extracted_pil_frames.extend(
                    [last_good_frame.copy() for _ in range(num_to_add)]
                )

        final_debug_meta["num_final_pil_frames_returned"] = len(extracted_pil_frames)
        total_operation_duration_ms = (time.monotonic() - overall_op_start_time) * 1000
        final_debug_meta["total_extract_frames_duration_ms"] = (
            total_operation_duration_ms
        )

        self._add_event(
            "extract_frames_completed",
            {
                "final_pil_frames_count": len(extracted_pil_frames),
                "total_duration_ms": total_operation_duration_ms,
                "hw_attempted": attempted_hw,
                "final_method_success_count": successful_extraction_count,  # from the last successful method
            },
        )

        self.logger.info(
            f"Finished frame extraction for '{os.path.basename(self.video_path)}': "
            f"{len(extracted_pil_frames)} frames in {total_operation_duration_ms:.2f}ms."
        )
        final_debug_meta["detailed_extraction_events"] = self.extraction_events
        return extracted_pil_frames, final_debug_meta

    def _get_avg_frame_rate(self) -> float:
        """
        Get the average frame rate (FPS) of the video using ffprobe.

        Returns:
            float: Average frame rate. Defaults to 30.0 if unable to determine.
        """
        operation_name = "get_avg_frame_rate_ffprobe"
        op_start_time = time.monotonic()

        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            "-i",
            self.video_path,
        ]
        event_details = {
            "video_path": self.video_path,
            "command_preview": "ffprobe -select_streams v:0 -show_entries stream=avg_frame_rate...",
        }
        self._add_event(f"{operation_name}_start", event_details)

        try:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
                timeout=15,
            )
            frame_rate_str = process.stdout.strip()

            # Handle case where frame rate is returned as a fraction like "30000/1001"
            if "/" in frame_rate_str:
                try:
                    numerator, denominator = map(int, frame_rate_str.split("/"))
                    if denominator == 0:  # Avoid division by zero
                        frame_rate = 30.0  # Default value
                    else:
                        frame_rate = numerator / denominator
                except (ValueError, ZeroDivisionError):
                    frame_rate = 30.0  # Default value
            else:
                try:
                    frame_rate = float(frame_rate_str)
                    if frame_rate <= 0 or not math.isfinite(frame_rate):
                        frame_rate = 30.0  # Default to a reasonable FPS
                except (ValueError, TypeError):
                    frame_rate = 30.0  # Default value

            time_taken_ms = (time.monotonic() - op_start_time) * 1000
            self._add_event(
                f"{operation_name}_success",
                {
                    "avg_frame_rate": frame_rate,
                    "raw_value": frame_rate_str,
                    "time_taken_ms": time_taken_ms,
                },
            )
            self.logger.debug(
                f"Average frame rate {frame_rate:.2f}fps in {time_taken_ms:.2f}ms for '{os.path.basename(self.video_path)}'"
            )
            return frame_rate

        except subprocess.TimeoutExpired as e_timeout:
            time_taken_ms = (time.monotonic() - op_start_time) * 1000
            stderr_output = "N/A"
            stderr = getattr(e_timeout, "stderr", None)
            if stderr is not None:
                try:
                    stderr_output = stderr.decode("utf-8", errors="ignore").strip()
                except (AttributeError, UnicodeDecodeError):
                    pass
            err_msg = f"Timeout ({time_taken_ms:.0f}ms) getting average frame rate"
            self.logger.warning(
                f"{err_msg} for '{self.video_path}'. Stderr: {stderr_output}",
                extra={"error": str(e_timeout)},
            )
            self._add_event(
                f"{operation_name}_timeout",
                {
                    "error": str(e_timeout),
                    "stderr": stderr_output,
                    "command": " ".join(command),
                    "time_taken_ms": time_taken_ms,
                    "default_fps_used": 30.0,
                },
            )
            return 30.0  # Default frame rate
        except Exception as e:  # Includes CalledProcessError
            time_taken_ms = (time.monotonic() - op_start_time) * 1000
            stderr_output = "N/A"
            stderr = getattr(e, "stderr", None)
            if stderr is not None:
                try:
                    stderr_output = stderr.decode("utf-8", errors="ignore").strip()
                except (AttributeError, UnicodeDecodeError):
                    pass
            err_msg = f"Failed ({type(e).__name__}, {time_taken_ms:.0f}ms) getting average frame rate"
            self.logger.error(
                f"{err_msg} for '{self.video_path}': {e}. Stderr: {stderr_output}",
                extra={"error": str(e)},
            )
            self._add_event(
                f"{operation_name}_failure",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "stderr": stderr_output,
                    "command": " ".join(command),
                    "time_taken_ms": time_taken_ms,
                    "default_fps_used": 30.0,
                },
            )
            return 30.0  # Default frame rate

    def __del__(self):
        # Cleanup if necessary, e.g., if self._cv2_capture was used.
        pass
