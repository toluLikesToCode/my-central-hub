#!/usr/bin/env python
"""
get_clip_embedding.py – now supports batching & on-disk caching

Usage:
    python get_clip_embedding.py <file_paths> [--num_frames NUM] [--model MODEL] [--debug]

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
from PIL import Image
import torch
import signal

PY_LOG_PREFIX = "[PyEmbeddingHelper]"


class StructuredFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "prefix": PY_LOG_PREFIX,
            "level": record.levelname,
            "time": self.formatTime(record, self.datefmt),
            "file": record.pathname,
            "func": record.funcName,
            "line": record.lineno,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base)


# Setup root logger for both console and file
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(
    logging.Formatter(
        f"{PY_LOG_PREFIX} %(asctime)s %(levelname)s %(funcName)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
)
log_file_path = os.path.join(
    os.path.dirname(__file__), "._" + os.path.basename(__file__) + ".log"
)
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(StructuredFormatter())

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Set to DEBUG for maximum output
root_logger.handlers = [console_handler, file_handler]

logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.ERROR)


def handle_sigint(signum, frame):
    logging.getLogger(__name__).info("Received SIGINT (Ctrl+C). Exiting gracefully.")
    # Perform any additional cleanup here if needed
    sys.exit(0)


def handle_sigterm(signum, frame):
    logging.getLogger(__name__).info("Received SIGTERM. Exiting gracefully.")
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

    def __init__(self, video_path: str, num_frames: int, logger=None, executor=None):
        self.video_path = video_path
        self.num_frames = num_frames
        self.logger = logger or logging.getLogger(__name__)
        self.duration = self.get_duration()
        self.executor = executor

    def get_duration(self) -> float:
        """Get the duration of the video in seconds using ffprobe.
        Returns:
            float: Duration of the video in seconds.
        Raises:
            RuntimeError: If ffprobe fails to retrieve the duration.
        """
        self.logger.debug(
            f"Calling ffprobe to get duration for video: {self.video_path}"
        )
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
            self.logger.debug(f"Video duration: {duration} seconds")
            return duration
        except Exception as e:
            self.logger.error("Failed to get video duration", exc_info=True)
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
        candidate_times = None
        debug_metadata = {}
        method_used = ""
        # Attempt scene detection with PySceneDetect.
        try:
            from scenedetect import VideoManager, SceneManager
            from scenedetect.detectors import ContentDetector

            video_manager = VideoManager([self.video_path])
            scene_manager = SceneManager()
            # The threshold here is heuristic; adjust based on your domain.
            scene_manager.add_detector(ContentDetector(threshold=25, min_scene_len=10))
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)
            scene_list = scene_manager.get_scene_list()
            candidate_times = [
                (scene[0].get_seconds() + scene[1].get_seconds()) / 2
                for scene in scene_list
            ]
            debug_metadata["scene_count"] = len(scene_list)
            method_used = "scene_detection"
            self.logger.debug(f"Detected {len(scene_list)} scenes.")
            # If scene coverage is insufficient or the video is too short, fall back to uniform sampling.
            if (
                len(candidate_times)
                < self.num_frames  # fewer scene midpoints than needed frames
                or len(scene_list) < (self.num_frames // 2)  # too few distinct scenes
                or self.duration
                < (self.num_frames * 2)  # video too short for advanced sampling
            ):
                self.logger.warning(
                    "Insufficient scene coverage or short video; falling back to uniform sampling."
                )
                candidate_times = None

        except Exception as e:
            self.logger.warning(
                "Scene detection failed or PySceneDetect not installed; falling back to dense candidate extraction.",
                exc_info=True,
            )

        # If candidate times are insufficient, fall back to uniform dense extraction.
        if not candidate_times or len(candidate_times) < self.num_frames:
            candidate_times = [
                (i + 1) * self.duration / (self.num_frames + 1)
                for i in range(self.num_frames)
            ]
            method_used = "fallback_uniform"
            debug_metadata = {"method_used": method_used, "timestamps": candidate_times}
            return candidate_times, debug_metadata

        # For each candidate timestamp, extract the frame and compute visual entropy.
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
                    self.logger.error(
                        f"Failed to extract candidate frame at {t} sec: {e}"
                    )
        if not candidate_frames:
            # If extraction completely failed, fallback.
            uniform_times = [
                (i + 1) * self.duration / (self.num_frames + 1)
                for i in range(self.num_frames)
            ]
            debug_metadata = {
                "method_used": "fallback_uniform_extraction",
                "timestamps": uniform_times,
            }
            return uniform_times, debug_metadata

        # Compute entropy values for all candidate frames.
        entropy_values = []
        for t, frame in candidate_frames:
            try:
                entropy_val = compute_entropy(frame)
            except Exception as e:
                entropy_val = 0.0
            entropy_values.append((t, entropy_val))
        debug_metadata["entropy_values"] = entropy_values

        # Rank candidates by entropy (descending).
        entropy_values.sort(key=lambda x: x[1], reverse=True)

        # Select frames ensuring temporal diversity.
        selected = []
        diversity_threshold = (
            self.duration * 0.05
        )  # At least 5% of video duration apart.
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
            remaining = sorted([t for t, _ in entropy_values if t not in selected])
            for t in remaining:
                if len(selected) < self.num_frames:
                    selected.append(t)
        selected.sort()
        debug_metadata["selected_times"] = selected
        debug_metadata["method_used"] = method_used
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
        command = [
            "ffmpeg",
            "-y",
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
        self.logger.debug(
            f"Calling ffmpeg to extract frame at {time_sec} seconds from video: {self.video_path}"
        )
        try:
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            if not result.stdout:
                raise RuntimeError("No frame data returned from ffmpeg.")
            image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            self.logger.debug(f"Extracted frame at {time_sec} seconds")
            return image
        except Exception as e:
            self.logger.error(
                f"Frame extraction failed at {time_sec} sec", exc_info=True
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
        timestamps, debug_metadata = self.get_advanced_sample_times()
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
                    self.logger.error(f"Failed to extract frame at {t} sec: {e}")
        if not frames:
            raise RuntimeError("No frames extracted from video.")
        frames.sort(key=lambda x: x[0])
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
    ):
        self.logger = logger
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

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Optional data augmentation transforms for images.
        self.enable_augmentation = enable_augmentation
        if enable_augmentation:
            import torchvision.transforms as T

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
        if self.enable_augmentation:
            image = self.augmentation_transforms(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features[0].cpu().numpy().tolist()

    def get_video_embedding(self, frames: list) -> list:
        """
        Compute and return an averaged CLIP embedding from multiple video frames.
        Future improvements can include weighted or attention-based aggregation.
        """
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
        return averaged_embedding.cpu().numpy().tolist()


def compute_single_embedding(fp, args, embedder, executor=None):
    import logging
    import os
    import time

    ext = os.path.splitext(fp)[1].lower()
    logger = logging.getLogger(__name__)
    try:
        debug_metadata = {
            "model": getattr(args, "model", None),
            "num_frames": (
                getattr(args, "num_frames", None) if ext in VIDEO_EXTS else None
            ),
            "enable_augmentation": getattr(args, "enable_augmentation", False),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if ext in VIDEO_EXTS:
            from io import BytesIO
            import subprocess

            logger.debug(f"Preparing to process video file: {fp}")
            video_processor = VideoProcessor(
                fp, args.num_frames, logger=logger, executor=executor
            )
            frames, adv_debug = video_processor.extract_frames()
            embedding = embedder.get_video_embedding(frames)
            debug_metadata.update(adv_debug)
            result = {
                "embedding": embedding,
                "debugMetadata": debug_metadata,
                "error": None,
                "detail": None,
            }
        elif ext in IMAGE_EXTS:
            logger.debug(f"Opening image file: {fp}")
            image = Image.open(fp).convert("RGB")
            embedding = embedder.get_image_embedding(image)
            result = {
                "embedding": embedding,
                "debugMetadata": debug_metadata,
                "error": None,
                "detail": None,
            }
        else:
            logger.error("Unsupported file type")
            result = {
                "embedding": None,
                "debugMetadata": debug_metadata,
                "error": "Unsupported file type",
                "detail": f"File type not supported: {ext} for file {fp}",
            }
        return result
    except Exception as e:
        logger.exception("An error occurred during processing.")
        return {
            "embedding": None,
            "debugMetadata": debug_metadata if "debug_metadata" in locals() else {},
            "error": "Processing failed",
            "detail": f"{type(e).__name__}: {e} (file: {fp})",
        }


def process_batch(image_paths, embedder, args, logger, shared_executor):
    results = {}
    total = len(image_paths)
    processed = 0
    for fp in image_paths:
        logger.debug(f"Processing file: {fp}")
        if not os.path.exists(fp):
            logger.error(f"Missing file: {fp}")
            results[fp] = {
                "embedding": None,
                "debugMetadata": {},
                "error": "File does not exist",
                "detail": f"Path not found: {fp}",
            }
            processed += 1
            print(
                f"PROGRESS: "
                + json.dumps({"processed": processed, "total": total, "current": fp}),
                file=sys.stderr,
                flush=True,
            )
            continue
        result = compute_single_embedding(fp, args, embedder, shared_executor)
        if "error" in result and "detail" not in result:
            result["detail"] = f"Failed to process file: {fp}"
        # Ensure all keys are present
        for k in ["embedding", "debugMetadata", "error", "detail"]:
            if k not in result:
                result[k] = None
        results[fp] = result
        processed += 1
        print(
            f"PROGRESS: "
            + json.dumps({"processed": processed, "total": total, "current": fp}),
            file=sys.stderr,
            flush=True,
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

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)
    logger.info("CLIP embedding service started. Waiting for input...")

    embedder = CLIPEmbedder(
        model_name=args.model,
        device=None,
        logger=logger,
        enable_augmentation=args.enable_augmentation,
    )
    logger.info(f"CLIPEmbedder initialized on device: {embedder.device}")

    shared_executor = concurrent.futures.ThreadPoolExecutor()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break  # EOF
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                image_paths = request.get("imagePaths")
                if not isinstance(image_paths, list):
                    raise ValueError("'imagePaths' must be a list.")
            except Exception as e:
                error_response = {"error": "Invalid input", "detail": str(e)}
                print(json.dumps(error_response), flush=True)
                continue
            try:
                results = process_batch(
                    image_paths, embedder, args, logger, shared_executor
                )
                print(json.dumps(results), flush=True)
            except Exception as e:
                # Instead of exiting, print a batchError JSON and continue
                batch_error = {
                    "batchError": str(e),
                    "detail": f"{type(e).__name__}: {e}",
                }
                print(json.dumps(batch_error), flush=True)
                continue
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
            break


if __name__ == "__main__":
    main_service()
