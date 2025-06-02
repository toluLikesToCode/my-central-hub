#!/usr/bin/env python
# -*- coding: utf-8 -*-
# embedding_service_helper.py
"""
embedding_service_helper.py – now supports batching & on-disk caching

Usage:
    This script is now primarily a module used by the FastAPI server (server.py).
    It provides functionalities for batch media processing and CLIP embedding generation.
    Video processing functionalities are handled by the video_processor_helper module.

Features:
    • Batch processing of multiple media items (URLs, filepaths).
    • Parallel preprocessing (I/O-bound tasks like downloads, frame extraction).
    • Single batched tensor pass to CLIP model for GPU efficiency.
    • On-disk caching (implicitly, if paths are re-requested and not changed, though explicit cache logic not in this file).
    • Image and video file support.
    • Advanced video frame extraction (via video_processor_helper) using scene detection and visual entropy, with hardware acceleration support.
    • CLIP model loading and inference using OpenCLIP.
    • Inference on CPU or GPU (if available).
    • JSON output format for embedding and debug metadata, handled by FastAPI server.
    • Structured logging via EmbeddingLogger.
    • Modular design with classes for CLIP inference and video processing (externalized).
    • Reporting of detailed frame extraction failures to a remote endpoint.
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
)
import re
import math
import errno
import requests  # For URL downloads
import open_clip  # type: ignore
import cv2  # type: ignore # Though unused, kept as per original file structure

from dotenv import load_dotenv  # type: ignore

# Import VideoProcessor from the new helper file
from video_processor_helper import VideoProcessor

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

ERROR_LOG_ENDPOINT_URL = "http://192.168.1.145:8080/api/embeddings/error-logs"


# --- Logger ---
class EmbeddingLogger:
    def __init__(
        self,
        request_id: Optional[str] = None,
        component_name: str = "PythonEmbedHelper",
    ):
        self.request_id = request_id or self._generate_request_id()
        self.component_name = component_name
        self.file_handler: Optional[logging.FileHandler] = None
        self._setup_logger()

    def _generate_request_id(self) -> str:
        return str(uuid.uuid4())

    def _setup_logger(self):
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
            self.file_handler = None

        self.console_handler = logging.StreamHandler(sys.stderr)
        self.console_handler.setFormatter(
            logging.Formatter(
                f"{PY_LOG_PREFIX} %(asctime)s %(levelname)s [{self.component_name}] (%(request_id)s): %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        logger_name = f"embedding_python_{self.component_name}_{self.request_id[:8]}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
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
        # Update formatter for console_handler to reflect new request_id for existing logger instance
        self.console_handler.setFormatter(
            logging.Formatter(
                f"{PY_LOG_PREFIX} %(asctime)s %(levelname)s [{self.component_name}] (%(request_id)s): %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        # For file handler, the extra={'request_id': ...} in log calls handles this,
        # but the %(request_id)s in the Formatter string itself is tied to the initial record.
        # Re-setting the formatter for file_handler if needed would be more complex or require passing request_id to every log record.
        # The current JSON formatter for file includes `%(request_id)s` which is populated from `final_extra`.

    def set_component_name(self, component_name):
        self.component_name = component_name
        # Update formatter for console_handler to reflect new component_name
        self.console_handler.setFormatter(
            logging.Formatter(
                f"{PY_LOG_PREFIX} %(asctime)s %(levelname)s [{self.component_name}] (%(request_id)s): %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        # Similar note for file_handler as above for request_id. %(component)s is also part of JSON format.


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


# compute_entropy function has been moved to video_processor_helper.py
# VideoProcessor class has been moved to video_processor_helper.py


class CLIPEmbedder:
    """
    A CLIP model wrapper for generating embeddings from PIL images.
    Supports both CPU and GPU inference with optional data augmentation.
    """

    model: Any
    preprocess: Callable[[Any], torch.Tensor]

    @staticmethod
    def _parse_clip_model_spec(model_spec: str) -> Tuple[str, str]:
        """
        Parse a CLIP_MODEL specification into a (model_arch, pretrained_tag) tuple
        compatible with open_clip.create_model_and_transforms.

        Dynamically matches input strings to available open_clip models and tags.
        Provides robust fallback and clear error reporting if no match is found.
        """
        import difflib

        spec = model_spec.strip()
        # Query all available (arch, tag) pairs from open_clip
        available_models = open_clip.list_pretrained()
        # available_models: List[Tuple[str, str]]
        # Normalize for matching

        def norm(s):
            return s.replace("-", "").replace("_", "").replace("/", "").lower()

        norm_spec = norm(spec)
        # Try exact match on (arch, tag) or HuggingFace/LAION style
        for arch, tag in available_models:
            if norm_spec == norm(f"{arch}-{tag}") or norm_spec == norm(f"{arch}{tag}"):
                return arch, tag
        # Try partial/fuzzy match on arch or tag
        arch_candidates = [
            arch
            for arch, _ in available_models
            if norm(arch) in norm_spec or norm_spec in norm(arch)
        ]
        if arch_candidates:
            # Pick the first arch with the most common pretrained tag
            arch = arch_candidates[0]
            tags = open_clip.list_pretrained_tags(arch)
            # Prefer laion2b or openai tags if present
            preferred = [t for t in tags if "laion" in t or "openai" in t]
            tag = preferred[0] if preferred else tags[0]
            return arch, tag
        # Fuzzy match using difflib
        all_model_strings = [f"{arch}-{tag}" for arch, tag in available_models]
        close = difflib.get_close_matches(
            norm_spec, [norm(s) for s in all_model_strings], n=1, cutoff=0.7
        )
        if close:
            idx = [norm(s) for s in all_model_strings].index(close[0])
            arch, tag = available_models[idx]
            return arch, tag
        # If input is a HuggingFace repo (e.g. laion/CLIP-ViT-B-16-plus-240-laion2B-s34B-b88K), try to extract arch/tag
        if "/" in spec:
            _, name = spec.split("/", 1)
            # Try to find a model arch that matches the start of the name
            for arch, tag in available_models:
                if norm(name).startswith(norm(arch)):
                    return arch, tag
        # If input is a bare arch, try to find a matching arch
        for arch, tag in available_models:
            if norm(arch) == norm_spec:
                return arch, tag
        # No match found: raise error with available options
        available_str = ", ".join([f"{a} ({t})" for a, t in available_models])
        raise ValueError(
            f"Could not match model spec '{model_spec}' to any available open_clip model. Available: {available_str}"
        )

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
            model_arch, pretrained_tag = self._parse_clip_model_spec(self.model_name)
            self.logger.debug(
                f"Parsed CLIP spec → model_arch='{model_arch}', pretrained='{pretrained_tag}'"
            )
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=model_arch,
                pretrained=pretrained_tag,
                device=self.device,
                jit=False,
            )
            self.model.eval()  # Ensure model is in eval mode

            # Set precision for matmul if supported (PyTorch 1.12+)
            if self.device not in ["cpu"]:  # Only relevant for GPU/MPS
                try:
                    if hasattr(torch, "set_float32_matmul_precision"):
                        torch.set_float32_matmul_precision("high")  # or 'medium'
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
            import torchvision.transforms as T  # Import only if needed

            # Determine image size from model config (e.g., self.model.visual.image_size)
            image_size_cfg = getattr(
                getattr(self.model, "visual", None), "image_size", 224
            )
            img_size_to_use = (
                image_size_cfg if isinstance(image_size_cfg, int) else image_size_cfg[0]
            )

            self.augmentation_transforms = T.Compose(
                [
                    T.RandomResizedCrop(
                        img_size_to_use, scale=(0.8, 1.0)
                    ),  # Example augmentation
                    T.RandomHorizontalFlip(p=0.5),
                    # Add other augmentations as needed
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
        """
        Generate CLIP embeddings for a list of PIL images.

        Args:
            pil_image_list: List of PIL Image objects to process

        Returns:
            torch.Tensor: Normalized embeddings tensor on CPU
        """
        if not pil_image_list:
            return torch.empty(0, dtype=torch.float32, device="cpu")

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

        if (
            not tensors_for_model
        ):  # Should not happen if pil_image_list was not empty and no exceptions prior
            self.logger.warning(
                "No PIL images were successfully preprocessed into tensors for the model."
            )
            return torch.empty(0, dtype=torch.float32, device="cpu")

        try:
            input_tensor_batch = torch.stack(tensors_for_model).to(self.device)
        except (
            Exception
        ) as e_stack:  # Handle potential shape mismatches if augmentations are complex
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
        if (
            self.device != "cpu"
        ):  # Enable autocast for CUDA or MPS for potential performance gains
            autocast_context = torch.autocast(
                device_type=self.device.split(":")[0]
            )  # 'cuda' or 'mps'

        with torch.no_grad(), autocast_context:
            image_features_batch = self.model.encode_image(input_tensor_batch)
            if image_features_batch is not None:
                image_features_batch = image_features_batch / image_features_batch.norm(
                    p=2, dim=-1, keepdim=True
                )
            else:  # Should not happen with standard OpenCLIP models
                self.logger.error(
                    "Model encode_image returned None, which is unexpected."
                )
                raise RuntimeError("Model encoding failed, returned None.")

        gpu_duration_ms = (time.time() - gpu_batch_start_time) * 1000
        self.logger.info(
            f"GPU inference for batch of {len(pil_image_list)} items completed in {gpu_duration_ms:.2f}ms. Output shape: {image_features_batch.shape}"
        )
        return image_features_batch.cpu()  # Move to CPU before returning

    def get_single_embedding(self, pil_image: Image.Image) -> List[float]:
        """
        Generate CLIP embedding for a single PIL image.

        Args:
            pil_image: PIL Image object to process

        Returns:
            List[float]: Normalized embedding as a list of floats
        """
        embeddings_tensor = self.get_embeddings_for_pil_list([pil_image])
        if (
            embeddings_tensor.shape[0] == 0
        ):  # Should not happen if input is a valid image
            return []
        return embeddings_tensor[0].tolist()

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Generate CLIP embedding for text.

        Args:
            text: Text string to encode

        Returns:
            List[float]: Normalized text embedding as a list of floats
        """
        try:
            text_tokens = open_clip.tokenize([text]).to(self.device)

            autocast_context = nullcontext()
            if self.device != "cpu":
                autocast_context = torch.autocast(device_type=self.device.split(":")[0])

            with torch.no_grad(), autocast_context:
                text_features = self.model.encode_text(text_tokens)
                if text_features is not None:
                    text_features = text_features / text_features.norm(
                        p=2, dim=-1, keepdim=True
                    )
                else:
                    self.logger.error(
                        "Model encode_text returned None, which is unexpected."
                    )
                    raise RuntimeError("Text encoding failed, returned None.")

            return text_features.cpu()[0].tolist()

        except Exception as e_text:
            self.logger.error(f"Failed to encode text '{text}': {e_text}", error=e_text)
            raise RuntimeError(f"Text encoding failed: {e_text}") from e_text

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding as list of floats
            embedding2: Second embedding as list of floats

        Returns:
            float: Cosine similarity score between -1 and 1
        """
        if not embedding1 or not embedding2:
            return 0.0

        try:
            tensor1 = torch.tensor(embedding1, dtype=torch.float32)
            tensor2 = torch.tensor(embedding2, dtype=torch.float32)

            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                tensor1.unsqueeze(0), tensor2.unsqueeze(0)
            )
            return float(similarity.item())

        except Exception as e_sim:
            self.logger.error(f"Failed to compute similarity: {e_sim}", error=e_sim)
            return 0.0

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the device and model.

        Returns:
            Dict containing device and model information
        """
        info = {
            "device": self.device,
            "model_name": self.model_name,
            "enable_augmentation": self.enable_augmentation,
        }

        if self.device == "cuda" and torch.cuda.is_available():
            info.update(
                {
                    "cuda_device_name": torch.cuda.get_device_name(),
                    "cuda_memory_allocated": torch.cuda.memory_allocated(),
                    "cuda_memory_reserved": torch.cuda.memory_reserved(),
                }
            )
        elif self.device == "mps" and torch.backends.mps.is_available():
            info["mps_available"] = True

        return info

    def __del__(self):
        """Cleanup resources when the embedder is destroyed."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _preprocess_single_item_for_batch(
    item_spec_dict: Dict[str, Any],
    python_media_root: str,
    default_num_frames_for_video: int,
    parent_logger: EmbeddingLogger,  # This is the batch-level logger
    video_processor_shared_executor: ThreadPoolExecutor,
    item_specific_request_id: str,  # This is batch_processing_id + item_id_suffix
    ffmpeg_hwaccel_method_for_videos: Optional[str],
) -> Dict[str, Any]:

    item_id = item_spec_dict["id"]
    media_type = item_spec_dict.get(
        "media_type", "unknown"
    )  # Ensure media_type is fetched early
    item_logger = EmbeddingLogger(  # Create a new logger instance for this item's sub-processing
        request_id=item_specific_request_id,  # Use the specific ID for this item's processing journey
        component_name=f"ItemPreProc-{item_id[:10]}",
    )
    item_logger.logger.setLevel(
        parent_logger.logger.level
    )  # Inherit log level from parent

    item_logger.debug(
        f"Preprocessing item '{item_spec_dict.get('original_filename', item_id)}'",
        extra=item_spec_dict,
    )

    pil_images_for_item: List[Image.Image] = []
    item_debug_meta: Dict[str, Any] = {
        "original_item_id": item_id,
        "source_type": item_spec_dict["source_type"],
        "original_filename": item_spec_dict.get("original_filename", "N/A"),
        "requested_media_type": media_type,  # Use media_type from item_spec_dict
        "timestamp_preprocess_start_utc": datetime.utcnow().isoformat() + "Z",
        "item_processing_request_id": item_specific_request_id,  # For error reporting correlation
    }
    temp_file_created_path: Optional[str] = None

    try:
        media_source_for_pil: Union[str, io.BytesIO]
        video_path_for_vid_processor: Optional[str] = None
        source_type = item_spec_dict["source_type"]
        source_location = item_spec_dict["source"]
        # media_type already assigned from item_spec_dict.get("media_type")

        match source_type:
            case "url":
                item_logger.debug(f"Downloading URL: {source_location}")
                response = requests.get(
                    source_location, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
                )
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                # Create a temporary file to store downloaded content
                # Suffix from original filename or URL to help identify temp files if not cleaned up
                temp_suffix_raw = item_spec_dict.get(
                    "original_filename", uuid.uuid4().hex[:8]
                )
                # Basic sanitization for suffix if it's from a filename that might contain problematic chars for a path
                if "." not in os.path.basename(
                    temp_suffix_raw
                ):  # if no extension, try to get from URL
                    ext_from_url = os.path.splitext(source_location)[1]
                    if ext_from_url and ext_from_url.lower() in IMAGE_EXTS + VIDEO_EXTS:
                        temp_suffix_raw += ext_from_url
                safe_suffix = "".join(
                    c if c.isalnum() or c in [".", "_", "-"] else "_"
                    for c in temp_suffix_raw
                )[:64]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=safe_suffix,
                    dir=os.environ.get("TEMP_DOWNLOAD_DIR"),
                ) as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
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

            case "filepath":
                # Special handling for paths starting with "/public/" - treat as relative to media root
                if source_location.startswith("/public/"):
                    # Strip the "/public/" prefix to get just the filename
                    relative_path = source_location[8:]  # Remove "/public/"
                    resolved_fs_path = os.path.join(python_media_root, relative_path)
                    item_logger.debug(
                        f"Treating '/public/' path as relative: '{resolved_fs_path}' (original source: '{source_location}')"
                    )
                else:
                    # Standard path resolution logic
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

            case "buffer_id":
                # TODO: Implement buffer_id source type handling to support in-memory buffers
                # This would allow processing media that's already loaded in memory without writing to disk
                item_logger.error(f"Buffer ID source type is not implemented yet")
                raise NotImplementedError(
                    f"Source type 'buffer_id' is not implemented yet"
                )

            case _:
                item_logger.error(f"Unsupported source_type: '{source_type}'")
                raise ValueError(f"Unsupported source_type: {source_type}")

        match media_type:
            case "image":
                img = Image.open(media_source_for_pil).convert("RGB")
                pil_images_for_item.append(img)
                item_debug_meta["image_dimensions"] = f"{img.width}x{img.height}"
                item_logger.debug(
                    f"Loaded image '{item_spec_dict.get('original_filename', item_id)}' ({img.width}x{img.height})"
                )

            case "video":
                if (
                    not video_path_for_vid_processor
                ):  # Should be set if URL download or filepath access succeeded
                    item_logger.error(
                        "Internal error: video_path_for_vid_processor not set for video item."
                    )
                    raise ValueError("video_path_for_vid_processor not set for video.")
                num_frames = (
                    item_spec_dict.get("num_frames") or default_num_frames_for_video
                )
                vp = VideoProcessor(  # Using the imported VideoProcessor
                    video_path=video_path_for_vid_processor,
                    num_frames=num_frames,
                    logger=item_logger,  # Pass the item-specific logger
                    executor=video_processor_shared_executor,  # For parallel frame extraction if VP supports it
                    request_id=item_specific_request_id,  # This is the item's unique processing ID
                    duration=item_spec_dict.get(
                        "estimated_duration_s"
                    ),  # Optional pre-fetched duration
                    original_filename_hint=item_spec_dict.get("original_filename"),
                    hwaccel_method=ffmpeg_hwaccel_method_for_videos,  # Pass batch-level HW accel method
                )
                frames_list_pil, video_proc_debug_meta = vp.extract_frames()
                pil_images_for_item.extend(frames_list_pil)
                item_debug_meta.update(
                    video_proc_debug_meta
                )  # This includes detailed_extraction_events
                item_debug_meta["num_extracted_frames_for_item"] = len(frames_list_pil)
                item_logger.info(
                    f"Extracted {len(frames_list_pil)} frames for video '{item_spec_dict.get('original_filename', item_id)}'"
                )

            case _:
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
            "media_type": media_type,
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
    except (
        requests.exceptions.RequestException
    ) as req_exc:  # Covers various network issues
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
    except (
        Exception
    ) as e_preproc:  # This includes ValueError from VideoProcessor if duration is invalid
        item_logger.error(
            f"General failure preprocessing item '{item_spec_dict.get('original_filename', item_id)}': {e_preproc}",
            error=e_preproc,
        )
        item_debug_meta["timestamp_preprocess_end_utc"] = (
            datetime.utcnow().isoformat() + "Z"
        )
        item_debug_meta["preprocess_error_type"] = type(e_preproc).__name__
        # If VideoProcessor failed (e.g. duration error), its extraction_events might be in e_preproc or need specific capture if VP raises early
        # For now, the top-level error is recorded. The error report logic will check for detailed_extraction_events.
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
                    f"Failed to clean up temp file '{temp_file_created_path}': {e_unlink_final}",
                    extra={"error_type": type(e_unlink_final).__name__},
                )


# For native HTTP POST for error logging
import http.client
import urllib.parse
import socket


def _send_error_logs_native(
    payload: List[Dict[str, Any]], logger: EmbeddingLogger, endpoint_url: str
):
    if not payload:
        logger.debug("No frame extraction error details to report.")
        return

    logger.info(
        f"Sending {len(payload)} frame extraction error reports to {endpoint_url}"
    )
    try:
        parsed_url = urllib.parse.urlparse(endpoint_url)
        hostname = parsed_url.hostname
        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
        path = parsed_url.path or "/"  # Ensure path is at least "/"

        if not hostname:
            logger.error(
                f"Invalid endpoint URL for error logs: {endpoint_url}. Missing hostname."
            )
            return

        # Truncate long strings in payload for safety, e.g., command previews or error messages
        def truncate_strings_in_payload(data, max_len=1024):
            if isinstance(data, dict):
                return {
                    k: truncate_strings_in_payload(v, max_len) for k, v in data.items()
                }
            elif isinstance(data, list):
                return [truncate_strings_in_payload(item, max_len) for item in data]
            elif isinstance(data, str) and len(data) > max_len:
                return data[:max_len] + "... (truncated)"
            return data

        truncated_payload = truncate_strings_in_payload(payload)
        json_payload = json.dumps(
            truncated_payload
        )  # Use truncated payload for JSON dump

        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(json_payload.encode("utf-8"))),  # Length of bytes
        }

        connection: Union[http.client.HTTPConnection, http.client.HTTPSConnection]
        if parsed_url.scheme == "https":
            connection = http.client.HTTPSConnection(hostname, port, timeout=10)
        else:
            connection = http.client.HTTPConnection(hostname, port, timeout=10)

        connection.request(
            "POST", path, body=json_payload.encode("utf-8"), headers=headers
        )
        response = connection.getresponse()
        response_data_bytes = response.read()
        response_data_str = response_data_bytes.decode(
            "utf-8", errors="replace"
        )  # Handle potential decoding errors

        if 200 <= response.status < 300:
            logger.info(
                f"Successfully sent error logs. Status: {response.status}. Response: {response_data_str[:200]}"
            )
        else:
            logger.error(
                f"Failed to send error logs. Status: {response.status}. Response: {response_data_str[:500]}",
                extra={
                    "status_code": response.status,
                    "response_body_preview": response_data_str[:500],
                },
            )
        connection.close()

    except ConnectionRefusedError:
        logger.error(
            f"Connection refused when sending error logs to {endpoint_url}.",
            extra={"target_url": endpoint_url},
        )
    except socket.timeout:
        logger.error(
            f"Timeout when sending error logs to {endpoint_url}.",
            extra={"target_url": endpoint_url},
        )
    except (
        http.client.HTTPException,
        OSError,
    ) as e_http:  # More general network/HTTP errors
        logger.error(
            f"HTTP/OS error sending frame extraction logs to {endpoint_url}: {e_http}",
            error=e_http,
            extra={"target_url": endpoint_url},
        )
    except (
        Exception
    ) as e:  # Catch-all for other unexpected errors (e.g., JSON serialization if not caught earlier)
        logger.error(
            f"Unexpected error sending frame extraction logs to {endpoint_url}: {e}",
            error=e,
            extra={"target_url": endpoint_url, "payload_item_count": len(payload)},
        )


def process_media_batch(
    items_data_as_dicts: List[Dict[str, Any]],
    clip_embedder_instance: CLIPEmbedder,
    python_media_root_path: str,
    default_num_frames_for_videos: int,
    parent_batch_logger: EmbeddingLogger,  # This is the main batch logger
    batch_processing_id: str,
) -> Dict[str, Dict[str, Any]]:

    parent_batch_logger.set_request_id(
        batch_processing_id
    )  # Ensure batch logger uses the overall batch ID
    parent_batch_logger.info(
        f"Processing media batch '{batch_processing_id}' for {len(items_data_as_dicts)} items."
    )
    batch_overall_start_time = time.time()

    final_results_map: Dict[str, Dict[str, Any]] = {}
    all_pil_images_for_gpu_batch: List[Image.Image] = []
    gpu_batch_item_details_map: List[Dict[str, Any]] = (
        []
    )  # Info for items successfully preprocessed for GPU

    # Configure thread pool sizes (example values, adjust based on typical workload and resources)
    num_cores = os.cpu_count() or 1
    max_preprocess_workers = min(
        max(4, num_cores * 2), 32
    )  # For I/O bound tasks (download, initial file access)
    max_video_frame_workers = min(
        max(2, num_cores if num_cores > 1 else 1), 16
    )  # For CPU-bound ffmpeg sub-processes if parallelized within VideoProcessor

    ffmpeg_hwaccel_method = os.environ.get(
        "FFMPEG_HWACCEL_METHOD"
    )  # Batch-level config
    if ffmpeg_hwaccel_method:
        parent_batch_logger.info(
            f"Batch processing will use FFMPEG_HWACCEL_METHOD='{ffmpeg_hwaccel_method}' for videos."
        )
    else:
        parent_batch_logger.info(
            "No FFMPEG_HWACCEL_METHOD configured; videos will use software decoding."
        )

    # ThreadPoolExecutor for item preprocessing (downloading, initial file ops)
    with (
        ThreadPoolExecutor(
            max_workers=max_preprocess_workers, thread_name_prefix="ItemPreProc"
        ) as item_preproc_executor,
        ThreadPoolExecutor(
            max_workers=max_video_frame_workers, thread_name_prefix="VideoFrames"
        ) as video_ffmpeg_executor,
    ):  # This executor is passed to VideoProcessor

        item_preprocess_futures_map = {
            item_preproc_executor.submit(
                _preprocess_single_item_for_batch,
                item_data_dict,
                python_media_root_path,
                default_num_frames_for_videos,
                parent_batch_logger,  # Pass the main batch logger for context, though _preprocess creates item-specific ones
                video_ffmpeg_executor,  # Pass the dedicated executor for video frame extraction tasks
                f"{batch_processing_id}_{item_data_dict['id'][:8]}",  # item_specific_request_id
                ffmpeg_hwaccel_method,  # Pass batch-level HW accel method
            ): item_data_dict[
                "id"
            ]  # Map future to original item ID
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
                item_preproc_output_dict = (
                    future_obj.result()
                )  # This is the dict from _preprocess_single_item_for_batch
                item_id_from_result = item_preproc_output_dict["id"]

                # Store all results, including those with preprocessing errors, in final_results_map first.
                # This ensures debug_metadata is preserved.
                final_results_map[item_id_from_result] = {
                    "id": item_id_from_result,
                    "embedding": None,  # Default, may be overwritten
                    "error": item_preproc_output_dict.get("error"),
                    "detail": item_preproc_output_dict.get("detail"),
                    "debugMetadata": {
                        **(item_preproc_output_dict.get("debug_metadata", {})),
                        "model": clip_embedder_instance.model_name,  # Add model/device info early
                        "device": clip_embedder_instance.device,
                        "batch_processing_status": "preprocessing_completed",  # Initial status
                    },
                }

                if item_preproc_output_dict.get("error"):
                    parent_batch_logger.warning(
                        f"Item '{item_id_from_result}' (orig: {original_item_id}) failed preprocessing: {item_preproc_output_dict['error']} - {item_preproc_output_dict.get('detail')}",
                        extra={
                            "item_id": item_id_from_result,
                            "error_details": item_preproc_output_dict,
                        },
                    )
                    final_results_map[item_id_from_result]["debugMetadata"][
                        "batch_processing_status"
                    ] = "failed_preprocessing"
                    continue  # Skip adding to GPU batch

                if not item_preproc_output_dict.get("pil_images"):
                    parent_batch_logger.warning(
                        f"Item '{item_id_from_result}' yielded no PIL images after preprocessing (error not set). Marking as error.",
                        extra={
                            "item_id": item_id_from_result,
                            "preproc_output": item_preproc_output_dict,
                        },
                    )
                    final_results_map[item_id_from_result][
                        "error"
                    ] = "No images/frames extracted (internal)"
                    final_results_map[item_id_from_result][
                        "detail"
                    ] = "Preprocessing yielded no processable PIL images, and no explicit error was set."
                    final_results_map[item_id_from_result]["debugMetadata"][
                        "batch_processing_status"
                    ] = "no_images_from_preprocessing_unexpected"
                    continue

                # If successfully preprocessed and has images, add to GPU batch consideration
                gpu_batch_item_details_map.append(
                    {
                        "item_id": item_id_from_result,
                        "media_type": item_preproc_output_dict["media_type"],
                        "num_source_pil_images": len(
                            item_preproc_output_dict["pil_images"]
                        ),
                        "start_idx_in_gpu_batch": len(all_pil_images_for_gpu_batch),
                        # original_debug_metadata is now part of final_results_map[item_id_from_result]["debugMetadata"]
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
                final_results_map[original_item_id] = (
                    {  # Overwrite if already exists from a partial success
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
                )
    # --- GPU Processing Stage ---
    if not all_pil_images_for_gpu_batch:
        parent_batch_logger.warning(
            f"Batch '{batch_processing_id}': No images/frames to process on GPU after parallel preprocessing stage.",
            extra={
                "num_initial_items": len(items_data_as_dicts),
                "items_in_gpu_map_count": len(gpu_batch_item_details_map),
            },
        )
        # Update status for items that were supposed to go to GPU but didn't because batch was empty
        for (
            item_map_detail
        ) in gpu_batch_item_details_map:  # These are items that passed preprocessing
            item_id_chk = item_map_detail["item_id"]
            if final_results_map[item_id_chk].get("debugMetadata"):
                final_results_map[item_id_chk]["debugMetadata"][
                    "batch_processing_status"
                ] = "gpu_batch_empty_before_inference"
            else:  # Should not happen if preproc was successful
                final_results_map[item_id_chk]["debugMetadata"] = {
                    "batch_processing_status": "gpu_batch_empty_before_inference_no_meta"
                }

    else:
        gpu_embeddings_tensor: Optional[torch.Tensor] = None
        try:
            if clip_embedder_instance is None:  # Should not happen based on code flow
                raise RuntimeError("CLIPEmbedder instance is None prior to GPU batch.")
            parent_batch_logger.info(
                f"Batch '{batch_processing_id}': Sending {len(all_pil_images_for_gpu_batch)} total images/frames to CLIP model on {clip_embedder_instance.device} for inference."
            )
            gpu_embeddings_tensor = clip_embedder_instance.get_embeddings_for_pil_list(
                all_pil_images_for_gpu_batch
            )

            # Update results for successfully processed GPU items
            utc_processing_timestamp = datetime.utcnow().isoformat() + "Z"
            for item_map_detail in gpu_batch_item_details_map:
                item_id_final = item_map_detail["item_id"]
                if (
                    gpu_embeddings_tensor is None
                ):  # Should be caught by exception if model fails
                    final_results_map[item_id_final][
                        "error"
                    ] = "GPU result tensor unavailable post-inference"
                    final_results_map[item_id_final]["debugMetadata"][
                        "batch_processing_status"
                    ] = "gpu_tensor_missing_post_inference_final"
                    continue

                start_idx = item_map_detail["start_idx_in_gpu_batch"]
                num_pils_for_item = item_map_detail["num_source_pil_images"]

                if start_idx + num_pils_for_item > gpu_embeddings_tensor.shape[0]:
                    parent_batch_logger.error(
                        f"Embedding disaggregation error for item '{item_id_final}': slice indices out of bounds",
                        extra={
                            "item_id": item_id_final,
                            "start_idx": start_idx,
                            "num_pils": num_pils_for_item,
                            "tensor_shape": str(gpu_embeddings_tensor.shape),
                        },
                    )
                    final_results_map[item_id_final][
                        "error"
                    ] = "Embedding disaggregation error (slice out of bounds)"
                    final_results_map[item_id_final]["debugMetadata"][
                        "batch_processing_status"
                    ] = "error_disaggregating_slice_oob"
                    continue

                item_slice_from_gpu_tensor = gpu_embeddings_tensor[
                    start_idx : start_idx + num_pils_for_item
                ]
                item_embedding_list: Optional[List[float]] = None
                item_specific_gpu_debug = {}

                if item_map_detail["media_type"] == "image":
                    if (
                        num_pils_for_item == 1
                        and item_slice_from_gpu_tensor.shape[0] == 1
                    ):
                        item_embedding_list = item_slice_from_gpu_tensor[0].tolist()
                    elif (
                        num_pils_for_item == 0
                        and item_slice_from_gpu_tensor.shape[0] == 0
                    ):  # Image was empty, preproc should catch?
                        item_embedding_list = []
                        parent_batch_logger.warning(
                            f"Image item '{item_id_final}' had 0 PILs, resulting in empty embedding.",
                            extra={"item_id": item_id_final},
                        )
                    else:  # Error case
                        parent_batch_logger.error(
                            f"Embedding disaggregation error for image item '{item_id_final}'",
                            extra={
                                "item_id": item_id_final,
                                "slice_shape": str(item_slice_from_gpu_tensor.shape),
                                "num_pils": num_pils_for_item,
                            },
                        )
                        final_results_map[item_id_final][
                            "error"
                        ] = "Image embedding disaggregation error"
                        final_results_map[item_id_final]["debugMetadata"][
                            "batch_processing_status"
                        ] = "error_disaggregating_image"
                        continue
                elif item_map_detail["media_type"] == "video":
                    if (
                        num_pils_for_item > 0
                        and item_slice_from_gpu_tensor.shape[0] == num_pils_for_item
                    ):
                        averaged_embedding = item_slice_from_gpu_tensor.mean(dim=0)
                        item_embedding_list = averaged_embedding.tolist()
                        item_specific_gpu_debug[
                            "averaged_from_n_frames_in_gpu_batch"
                        ] = num_pils_for_item
                    elif num_pils_for_item == 0:  # Video had no extractable frames
                        item_embedding_list = []
                        item_specific_gpu_debug[
                            "averaged_from_n_frames_in_gpu_batch"
                        ] = 0
                        parent_batch_logger.warning(
                            f"Video item '{item_id_final}' had 0 PILs for averaging, resulting in empty embedding.",
                            extra={"item_id": item_id_final},
                        )
                    else:  # Error case
                        parent_batch_logger.error(
                            f"Embedding disaggregation error for video item '{item_id_final}'",
                            extra={
                                "item_id": item_id_final,
                                "slice_shape": str(item_slice_from_gpu_tensor.shape),
                                "num_pils": num_pils_for_item,
                            },
                        )
                        final_results_map[item_id_final][
                            "error"
                        ] = "Video embedding disaggregation error"
                        final_results_map[item_id_final]["debugMetadata"][
                            "batch_processing_status"
                        ] = "error_disaggregating_video"
                        continue
                else:  # Unknown media type
                    item_embedding_list = (
                        []
                    )  # Should not happen if validation is correct

                final_results_map[item_id_final]["embedding"] = item_embedding_list
                final_results_map[item_id_final][
                    "error"
                ] = None  # Clear any prior preproc error if GPU stage was reached and embedding produced
                final_results_map[item_id_final]["detail"] = None
                final_results_map[item_id_final]["debugMetadata"].update(
                    {
                        **item_specific_gpu_debug,
                        "overall_batch_request_id": batch_processing_id,  # Main batch ID
                        "processing_timestamp_utc": utc_processing_timestamp,
                        "batch_processing_status": "success",
                    }
                )
                parent_batch_logger.debug(
                    f"Successfully finalized result for item '{item_id_final}'.",
                    extra={"item_id": item_id_final},
                )

        except Exception as e_gpu_stage:
            parent_batch_logger.error(
                f"Batch '{batch_processing_id}': GPU processing failed for collected batch: {e_gpu_stage}",
                error=e_gpu_stage,
            )
            # Update all items intended for this GPU batch as failed at GPU stage
            for item_map_detail in gpu_batch_item_details_map:
                item_id_gpu_fail = item_map_detail["item_id"]
                final_results_map[item_id_gpu_fail][
                    "error"
                ] = "GPU processing failed for batch"
                final_results_map[item_id_gpu_fail]["detail"] = str(e_gpu_stage)
                final_results_map[item_id_gpu_fail]["debugMetadata"][
                    "gpu_error_global"
                ] = True
                final_results_map[item_id_gpu_fail]["debugMetadata"][
                    "batch_processing_status"
                ] = "failed_gpu_inference_stage"
                final_results_map[item_id_gpu_fail]["debugMetadata"]["error_type"] = (
                    type(e_gpu_stage).__name__
                )

    # --- Frame Extraction Error Reporting ---
    frame_extraction_error_reports_for_batch: List[Dict[str, Any]] = []
    for item_id, result_data in final_results_map.items():
        debug_meta = result_data.get("debugMetadata", {})
        if debug_meta.get("requested_media_type") == "video":
            # Criteria for reporting:
            # 1. There's a top-level error related to video processing for the item.
            # 2. Or, the detailed_extraction_events log contains failure/error events.
            report_this_item = False
            top_level_error = result_data.get("error")
            top_level_detail_lower = (result_data.get("detail", "") or "").lower()

            if top_level_error and any(
                kw in top_level_detail_lower
                for kw in ["video", "frame", "ffmpeg", "duration", "extraction"]
            ):
                report_this_item = True

            extraction_events = debug_meta.get("detailed_extraction_events", [])
            # Report any frame extraction events containing 'fail' or 'error'
            has_failure_events = any(
                (evt.get("event_type", "") or "").lower().find("fail") >= 0
                or (evt.get("event_type", "") or "").lower().find("error") >= 0
                for evt in extraction_events
            )

            if has_failure_events:
                report_this_item = True

            if report_this_item:
                # Summarize hardware acceleration failures
                hw_fail_events = [
                    evt
                    for evt in extraction_events
                    if evt.get("event_type", "").startswith("hw_extraction_failed")
                ]
                hw_timestamps = [
                    evt["details"].get("video_timestamp_sec") for evt in hw_fail_events
                ]
                hw_error_types = {}
                for evt in hw_fail_events:
                    err_type = evt["details"].get("error_type")
                    hw_error_types[err_type] = hw_error_types.get(err_type, 0) + 1
                sample_error_msg = (
                    hw_fail_events[0]["details"].get("error_message")
                    if hw_fail_events
                    else None
                )
                report_entry = {
                    "itemId": item_id,
                    "originalFilename": debug_meta.get("original_filename"),
                    "itemProcessingRequestId": debug_meta.get(
                        "item_processing_request_id"
                    ),
                    "batchConfiguredHwAccelMethod": ffmpeg_hwaccel_method,
                    "videoProcessorEffectiveHwAccelMethod": debug_meta.get(
                        "video_processor_instance_hwaccel_method"
                    ),
                    "videoDurationSeconds": debug_meta.get("video_duration_s"),
                    "topLevelItemError": top_level_error,
                    "topLevelItemErrorDetail": result_data.get("detail"),
                    "hardwareAccelFailures": {
                        "count": len(hw_fail_events),
                        "timestamps": hw_timestamps,
                        "errorTypeCounts": hw_error_types,
                        "sampleErrorMessage": sample_error_msg,
                    },
                    # Sampling details
                    "frameSamplingMethod": debug_meta.get(
                        "frame_sampling_details", {}
                    ).get("method_used"),
                    "requestedFramesForSampling": debug_meta.get(
                        "frame_sampling_details", {}
                    ).get("num_requested_frames"),
                    "finalPilFramesReturnedForItem": debug_meta.get(
                        "num_final_pil_frames_returned"
                    ),
                    "itemFrameExtractionErrorCount": debug_meta.get(
                        "frame_extraction_error_count"
                    ),
                    # Full detailed events can be very large, consider if a summary or specific event types are better
                    # "detailedExtractionEvents": extraction_events # Potentially too verbose for remote log
                }
                frame_extraction_error_reports_for_batch.append(report_entry)

    if frame_extraction_error_reports_for_batch:
        _send_error_logs_native(
            frame_extraction_error_reports_for_batch,
            parent_batch_logger,
            ERROR_LOG_ENDPOINT_URL,
        )

    batch_total_duration_ms = (time.time() - batch_overall_start_time) * 1000
    successful_items_count = sum(
        1 for res in final_results_map.values() if not res.get("error")
    )
    parent_batch_logger.info(
        f"Batch '{batch_processing_id}' processing completed in {batch_total_duration_ms:.2f}ms. "
        f"Total items: {len(items_data_as_dicts)}, Successful (final): {successful_items_count}",
        extra={
            "duration_ms": batch_total_duration_ms,
            "total_items": len(items_data_as_dicts),
            "gpu_candidate_items": len(gpu_batch_item_details_map),
            "all_pil_images_for_gpu_count": len(all_pil_images_for_gpu_batch),
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
    print(f"  FFMPEG_HWACCEL_METHOD={os.environ.get('FFMPEG_HWACCEL_METHOD')}")
    print(f"  TEMP_DOWNLOAD_DIR={os.environ.get('TEMP_DOWNLOAD_DIR')}")
