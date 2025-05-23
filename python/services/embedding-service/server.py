#!/usr/bin/env python3
"""
Embedding Service API

This FastAPI microservice provides CLIP embeddings for images and videos.
It is designed for high-throughput, low-latency inference in a Dockerized environment.

Endpoints:
    - GET /health         : Service health and model/device status
    - GET /gpu-metrics    : (Optional) GPU metrics (if available)
    - POST /embed         : Compute CLIP embedding for an image or video (accepts binary data)

Performance:
    - Video frame extraction is optimized: the video buffer is written to a temp file once,
      and all frames are extracted in a single ffmpeg process using hardware acceleration if available.
    - All major steps are timed and logged for observability.

Usage:
    - Run as a Docker container or locally.
    - POST binary image/video data to /embed with appropriate headers.
    - Returns embedding and debug metadata.
"""

import os
import sys
import time
import uuid
import logging
import io
from typing import List, Dict, Any, Optional, Union

import torch
import uvicorn
from fastapi import (
    FastAPI,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
from datetime import datetime

# Import the embedding helper module
sys.path.append("/app")
from . import embedding_service_helper as emb_helper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("embedding-service")

# Initialize FastAPI app
app = FastAPI(
    title="Embedding Service API",
    description="API for computing CLIP embeddings for images and videos",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
SERVICE_START_TIME = time.time()
PROCESSED_FILES_COUNT = 0
EMBEDDING_MODEL = None
EMBEDDING_DEVICE = None


# Define Pydantic models (updated for Pydantic v2)
class EmbeddingResponse(BaseModel):
    embedding: List[float]
    debug_metadata: Optional[Dict[str, Any]] = Field(None, alias="debugMetadata")
    error: Optional[str] = None
    detail: Optional[str] = None


class BatchEmbeddingRequest(BaseModel):
    image_paths: List[str] = Field(..., alias="imagePaths")


class BatchEmbeddingResponse(BaseModel):
    embeddings: Dict[str, EmbeddingResponse]


class GPUMetrics(BaseModel):
    available: bool
    device_index: Optional[int] = None
    device_name: Optional[str] = None
    driver_version: Optional[str] = None
    memory_allocated_mb: Optional[float] = None
    memory_reserved_mb: Optional[float] = None
    memory_free_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    utilization_percent: Optional[float] = None
    mps_info: Optional[Dict[str, Any]] = None  # For MPS-specific details
    compute_capability: Optional[str] = None
    multiprocessor_count: Optional[int] = None
    gpu_clock_mhz: Optional[int] = None
    pci_bus_id: Optional[str] = None
    cuda_capability: Optional[str] = None


class ServiceHealth(BaseModel):
    status: str
    uptime_seconds: float
    processed_files: int
    gpu_available: bool
    model_loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None


def init_model():
    """Initialize the CLIP model for embeddings with proper error handling and cleanup"""
    global EMBEDDING_MODEL, EMBEDDING_DEVICE

    try:
        logger.info("Initializing CLIP model...")

        # Determine the device with priority:
        # 1. MPS for M1 Mac
        # 2. CUDA for NVIDIA GPUs
        # 3. CPU as fallback
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA on device: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")

        # Get model name from environment or use default
        model_name = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")

        # Clean up any existing model to free memory
        if EMBEDDING_MODEL is not None:
            logger.info("Cleaning up existing model...")
            try:
                del EMBEDDING_MODEL
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error during model cleanup: {e}")

        # Initialize the embedder with proper logging
        logger.info(f"Loading CLIP model: {model_name}")
        EMBEDDING_MODEL = emb_helper.CLIPEmbedder(
            model_name=model_name,
            device=device,
            logger=emb_helper.EmbeddingLogger("model-init"),
            enable_augmentation=os.environ.get("ENABLE_AUGMENTATION", "false").lower()
            == "true",
        )
        EMBEDDING_DEVICE = device

        # Log memory usage if available
        if device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / (1024**2)
            memory_reserved = torch.cuda.memory_reserved() / (1024**2)
            logger.info(
                f"GPU Memory after model load - Allocated: {memory_allocated:.1f}MB, "
                f"Reserved: {memory_reserved:.1f}MB"
            )

        logger.info(f"Model initialized successfully on {device}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)
        EMBEDDING_MODEL = None
        EMBEDDING_DEVICE = None
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts"""
    if not init_model():
        logger.error("Failed to initialize model during startup")
        # Don't exit here, let the health check handle the error state


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the API shuts down"""
    global EMBEDDING_MODEL
    try:
        if EMBEDDING_MODEL is not None:
            logger.info("Cleaning up model resources...")
            del EMBEDDING_MODEL
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")


@app.get("/health", response_model=ServiceHealth)
async def health_check(request: Request):
    """Check the health of the embedding service"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    logger.info(f"Health check requested", extra={"request_id": request_id})
    return {
        "status": "ok" if EMBEDDING_MODEL is not None else "initializing",
        "uptime_seconds": time.time() - SERVICE_START_TIME,
        "processed_files": PROCESSED_FILES_COUNT,
        "gpu_available": torch.cuda.is_available() or torch.backends.mps.is_available(),
        "model_loaded": EMBEDDING_MODEL is not None,
        "model_name": getattr(EMBEDDING_MODEL, "model_name", None),
        "device": EMBEDDING_DEVICE,
    }


@app.get("/gpu-metrics", response_model=GPUMetrics)
async def get_gpu_metrics():
    """Get GPU memory usage and other metrics"""
    metrics = {"available": False}

    if torch.cuda.is_available():
        try:
            device_index = 0  # Assuming single GPU
            device_prop = torch.cuda.get_device_properties(device_index)
            device_name = torch.cuda.get_device_name(device_index)
            driver_version = torch.version.cuda
            memory_allocated = torch.cuda.memory_allocated(device_index) / (
                1024**2
            )  # MB
            memory_reserved = torch.cuda.memory_reserved(device_index) / (1024**2)  # MB
            memory_total = device_prop.total_memory / (1024**2)  # MB
            memory_free = memory_total - memory_allocated

            # Try to get utilization if possible (requires pynvml or torch >= 2.1)
            utilization = None
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu
                pci_bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId.decode()
                pynvml.nvmlShutdown()
            except Exception:
                pci_bus_id = getattr(device_prop, "pci_bus_id", None)
                try:
                    utilization = torch.cuda.utilization(device_index)
                except Exception:
                    utilization = None

            compute_capability = f"{device_prop.major}.{device_prop.minor}"
            multiprocessor_count = getattr(device_prop, "multi_processor_count", None)
            gpu_clock_mhz = getattr(device_prop, "clock_rate", None)
            cuda_capability = f"{device_prop.major}.{device_prop.minor}"

            metrics = {
                "available": True,
                "device_index": device_index,
                "device_name": device_name,
                "driver_version": driver_version,
                "memory_allocated_mb": memory_allocated,
                "memory_reserved_mb": memory_reserved,
                "memory_free_mb": memory_free,
                "memory_total_mb": memory_total,
                "utilization_percent": utilization,
                "mps_info": None,
                "compute_capability": compute_capability,
                "multiprocessor_count": multiprocessor_count,
                "gpu_clock_mhz": gpu_clock_mhz,
                "pci_bus_id": pci_bus_id,
                "cuda_capability": cuda_capability,
            }
        except Exception as e:
            logger.error(f"Error getting CUDA metrics: {e}")

    elif torch.backends.mps.is_available():
        try:
            # MPS (Metal Performance Shaders) for Apple Silicon - has limited metrics
            mps_info = {
                "is_available": torch.backends.mps.is_available(),
                "is_built": torch.backends.mps.is_built(),
                "device_count": 1,  # MPS is always device 0 if available
                "torch_version": torch.__version__,
            }
            metrics = {
                "available": True,
                "device_index": 0,
                "device_name": "Apple MPS (Metal)",
                "driver_version": None,
                "memory_allocated_mb": None,
                "memory_reserved_mb": None,
                "memory_free_mb": None,
                "memory_total_mb": None,
                "utilization_percent": None,
                "mps_info": mps_info,
                "compute_capability": None,
                "multiprocessor_count": None,
                "gpu_clock_mhz": None,
                "pci_bus_id": None,
                "cuda_capability": None,
            }
        except Exception as e:
            logger.error(f"Error getting MPS metrics: {e}")

    return metrics


@app.post("/embed")
async def embed_media(request: Request):
    """
    Endpoint to generate embeddings for images and videos.
    Accepts both file uploads and direct binary data.

    Performance:
        - Measures and logs total request time.
        - Logs per-step timing for video/image processing.
    """
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    media_type = request.headers.get("X-Media-Type", "image").lower()
    filename = request.headers.get("X-Filename", "unknown")

    logger.info(
        f"Processing {media_type} request",
        extra={
            "request_id": request_id,
            "media_type": media_type,
            "file_name": filename,
        },
    )

    try:
        # Read the binary data into a BytesIO buffer
        content = await request.body()
        buffer = io.BytesIO(content)

        # Log the size of the incoming data
        logger.info(
            f"Received {media_type} data, size: {len(content)} bytes",
            extra={"request_id": request_id, "file_name": filename},
        )

        # Process based on media type
        step_start = time.time()
        if media_type == "video":
            embedding = await process_video(buffer, filename, request_id)
        else:
            embedding = await process_image(buffer, filename, request_id)
        step_duration = time.time() - step_start

        logger.info(
            f"{media_type.capitalize()} processing completed in {step_duration:.2f}s",
            extra={
                "request_id": request_id,
                "processing_time_sec": step_duration,
                "media_type": media_type,
            },
        )

        # Track total processing time
        total_time = time.time() - start_time
        logger.info(
            f"Successfully processed {media_type} (total time: {total_time:.2f}s)",
            extra={
                "request_id": request_id,
                "total_processing_time_sec": total_time,
                "media_type": media_type,
            },
        )

        embedding_out = (
            embedding.tolist() if hasattr(embedding, "tolist") else embedding
        )

        return {
            "embedding": embedding_out if embedding is not None else [],
            "debugMetadata": {
                "model": f"{EMBEDDING_MODEL.model_name} - {EMBEDDING_DEVICE}",
                "enable_augmentation": os.environ.get(
                    "ENABLE_AUGMENTATION", "false"
                ).lower()
                == "true",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "request_id": request_id,
                "source_type": "buffer",
                "specified_media_type": media_type,
                "num_frames": 20 if media_type == "video" else None,
            },
            "error": None,
            "detail": None,
        }

    except Exception as e:
        logger.error(
            f"Failed to process {media_type}",
            extra={
                "request_id": request_id,
                "error": str(e),
                "media_type": media_type,
            },
        )
        return {
            "embedding": [],
            "debugMetadata": {
                "model": f"{EMBEDDING_MODEL.model_name} - {EMBEDDING_DEVICE}",
                "enable_augmentation": os.environ.get(
                    "ENABLE_AUGMENTATION", "false"
                ).lower()
                == "true",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "request_id": request_id,
                "source_type": "buffer",
                "specified_media_type": media_type,
                "num_frames": 20 if media_type == "video" else None,
            },
            "error": "Processing failed",
            "detail": str(e),
        }


async def process_video(
    buffer: io.BytesIO, filename: str, request_id: str
) -> np.ndarray:
    """
    Process a video buffer and return its embedding.

    Performance:
        - Uses a single ffmpeg process with hardware acceleration and reduced frame size.
        - Logs timing for frame extraction and embedding computation.
    """
    logger.info(
        f"Processing video: {filename}",
        extra={"request_id": request_id, "file_name": filename},
    )

    try:
        # Create a VideoProcessor instance with the buffer
        processor = emb_helper.VideoProcessor(
            video_buffer=buffer,
            logger=logger,
            num_frames=int(os.environ.get("NUM_FRAMES", 20)),
        )

        # Extract all frames in a single, fast ffmpeg call
        frame_extraction_start = time.time()
        frames, _ = processor.extract_frames()
        frame_extraction_time = time.time() - frame_extraction_start
        logger.info(
            f"Extracted {len(frames)} frames in {frame_extraction_time:.2f}s",
            extra={"request_id": request_id, "frame_count": len(frames)},
        )

        # Compute embedding for all frames (batched)
        embedding_start = time.time()
        embedding = EMBEDDING_MODEL.get_video_embedding(frames)
        embedding_time = time.time() - embedding_start
        logger.info(
            f"Computed video embedding in {embedding_time:.2f}s",
            extra={"request_id": request_id},
        )
        return embedding

    except Exception as e:
        logger.error(
            f"Video processing failed: {e}",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise


async def process_image(
    buffer: io.BytesIO, filename: str, request_id: str
) -> np.ndarray:
    """
    Process an image buffer and return its embedding.

    Performance:
        - Logs timing for image embedding computation.
    """
    logger.info(
        f"Processing image: {filename}",
        extra={"request_id": request_id, "file_name": filename},
    )

    try:
        # Open image from buffer
        image = Image.open(buffer).convert("RGB")

        # Compute embedding
        embedding_start = time.time()
        embedding = EMBEDDING_MODEL.get_image_embedding(image)
        embedding_time = time.time() - embedding_start
        logger.info(
            f"Computed image embedding in {embedding_time:.2f}s",
            extra={"request_id": request_id},
        )
        return embedding

    except Exception as e:
        logger.error(
            f"Image processing failed: {e}",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise


# Starting the API server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3456))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
