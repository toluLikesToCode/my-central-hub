#!/usr/bin/env python3
# server.py - FastAPI server for embedding service with batching support
import os
import sys
import time
import json
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager
from dotenv import load_dotenv  # type: ignore

import torch
import uvicorn  # type: ignore
from fastapi import FastAPI, HTTPException, Request, Body  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from pydantic import BaseModel, Field, ValidationError  # type: ignore

# Load environment variables from .env file, if present
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Assuming embedding_service_helper is in the same directory or PYTHONPATH
import embedding_service_helper as emb_helper

# --- Globals ---
SERVICE_START_TIME = time.time()
PROCESSED_FILES_COUNT_TOTAL = 0  # Atomic counter for total items
EMBEDDING_MODEL: Optional[emb_helper.CLIPEmbedder] = None
EMBEDDING_DEVICE: Optional[str] = None
PYTHON_MEDIA_ROOT = os.environ.get(
    "PYTHON_MEDIA_ROOT", "/public"
)  # Mount point in Docker
TARGET_VRAM_UTILIZATION = float(os.environ.get("TARGET_VRAM_UTILIZATION", 0.85))
MAX_BATCH_ITEMS = int(
    os.environ.get("MAX_BATCH_ITEMS", 128)
)  # Max items in a single GPU batch
BATCH_FLUSH_TIMEOUT_S = float(os.environ.get("BATCH_FLUSH_TIMEOUT_S", 0.5))  # Seconds
GPU_POLL_INTERVAL_S = float(os.environ.get("GPU_POLL_INTERVAL_S", 0.1))  # Seconds
DEFAULT_VIDEO_FRAMES_TO_EXTRACT_CONFIG = int(
    os.environ.get("DEFAULT_VIDEO_FRAMES_TO_EXTRACT", 20)
)


# --- Log all config values on startup ---
logger = logging.getLogger("embedding_service_api")
logger.info("[Startup Config] Loaded environment/config values:")
logger.info(f"  PYTHON_PORT={os.environ.get('PYTHON_PORT')}")
logger.info(f"  PYTHON_MEDIA_ROOT={os.environ.get('PYTHON_MEDIA_ROOT')}")
logger.info(f"  LOG_LEVEL={os.environ.get('LOG_LEVEL')}")
logger.info(f"  CLIP_MODEL={os.environ.get('CLIP_MODEL')}")
logger.info(f"  ENABLE_AUGMENTATION={os.environ.get('ENABLE_AUGMENTATION')}")
logger.info(f"  TARGET_VRAM_UTILIZATION={os.environ.get('TARGET_VRAM_UTILIZATION')}")
logger.info(f"  MAX_BATCH_ITEMS={os.environ.get('MAX_BATCH_ITEMS')}")
logger.info(f"  BATCH_FLUSH_TIMEOUT_S={os.environ.get('BATCH_FLUSH_TIMEOUT_S')}")
logger.info(f"  GPU_POLL_INTERVAL_S={os.environ.get('GPU_POLL_INTERVAL_S')}")
logger.info(
    f"  DEFAULT_VIDEO_FRAMES_TO_EXTRACT={os.environ.get('DEFAULT_VIDEO_FRAMES_TO_EXTRACT')}"
)
logger.info(f"  DOWNLOAD_TIMEOUT_SECONDS={os.environ.get('DOWNLOAD_TIMEOUT_SECONDS')}")


# Configure logging (can be more sophisticated)
# BasicConfig should be called only once.
if not logging.getLogger().handlers:  # Check if root logger is already configured
    logging.basicConfig(
        stream=sys.stdout,
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )

logger = logging.getLogger("embedding_service_api")
# Pass the logger instance to the helper module if it expects one
if hasattr(emb_helper, "embedding_logger") and isinstance(
    emb_helper.embedding_logger, emb_helper.EmbeddingLogger
):
    # If emb_helper has its own logger setup, we might not need to overwrite it,
    # or we make its logger a child of this one.
    # For simplicity, if we want helper to use THIS logger's stream/config:
    emb_helper.embedding_logger.logger = (
        logger  # Make helper use API's configured logger instance
    )
    emb_helper.embedding_logger.console_handler.setLevel(logger.level)
    if emb_helper.embedding_logger.file_handler:
        emb_helper.embedding_logger.file_handler.setLevel(logger.level)


# --- Pydantic Models ---
class MediaItem(BaseModel):
    id: str = Field(
        ..., description="Unique identifier for this media item, provided by client."
    )
    media_type: str = Field(..., description="'image' or 'video'")
    source_type: str = Field(
        ..., description="'url', 'filepath', or 'buffer_id' (if multipart)"
    )
    source: str = Field(
        ..., description="URL, relative filepath, buffer identifier, or filename"
    )
    num_frames: Optional[int] = Field(
        None, description="Number of frames for video (uses default if None)"
    )
    original_filename: Optional[str] = Field(
        None, description="Original filename for logging/debugging"
    )
    # For pre-computation in BatchingManager, not part of external API necessarily unless client sends it.
    estimated_duration_s: Optional[float] = Field(None, exclude=True)


class BatchEmbeddingRequest(BaseModel):
    items: List[MediaItem]
    request_id: Optional[str] = None


class EmbeddingResult(BaseModel):
    id: str
    embedding: Optional[List[float]] = None
    error: Optional[str] = None
    detail: Optional[str] = None
    debug_metadata: Optional[Dict[str, Any]] = Field(None, alias="debugMetadata")


class BatchEmbeddingResponse(BaseModel):
    results: List[EmbeddingResult]
    batch_id: str  # ID of the client's overall request, not individual Python batches
    processed_by_request_id: Optional[str] = None


class ServiceHealth(BaseModel):
    status: str
    uptime_seconds: float
    processed_items_count: int
    gpu_available: bool
    model_loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None
    request_queue_size: int


# --- Batching Manager ---
class BatchingManager:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.batch_id_counter = 0
        self.active_items_in_gpu_processing = (
            0  # Number of items currently in a GPU batch
        )
        self.vram_total_gb = 0.0
        self.vram_reserved_static_gb = 0.0  # Model, etc.
        if torch.cuda.is_available():
            try:
                torch.cuda.init()  # Ensure CUDA is initialized
                self.vram_total_gb = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)
            except Exception as e:
                logger.error(f"Failed to get CUDA device properties: {e}")
                self.vram_total_gb = 0.0  # Indicate VRAM info unavailable
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.vram_total_gb = 8.0  # Placeholder for MPS
        logger.info(
            f"BatchingManager initialized. Detected Total VRAM: {self.vram_total_gb:.2f} GB"
        )

    def estimate_vram_gb(self, item: MediaItem) -> float:
        base_img_vram_gb = (3 * 224 * 224 * 2) / (1024**3)  # Approx for float16 224x224
        if item.media_type == "video":
            num_frames = item.num_frames or DEFAULT_VIDEO_FRAMES_TO_EXTRACT_CONFIG
            return base_img_vram_gb * num_frames
        return base_img_vram_gb

    async def _get_available_dynamic_vram_gb(self) -> float:
        if torch.cuda.is_available():
            try:
                free_mem_bytes, _ = torch.cuda.mem_get_info()
                # Consider static reservation made at startup
                # available_for_dynamic = (free_mem_bytes / (1024**3))
                # A simpler approach: total - static_reserved - current_active_batch_guess
                # This is still tricky. Let's use `free_mem_bytes` as the basis.
                return (free_mem_bytes / (1024**3)) * TARGET_VRAM_UTILIZATION
            except Exception as e:
                logger.warning(
                    f"Could not query CUDA memory: {e}. Falling back to conservative estimate."
                )
                return (
                    self.vram_total_gb - self.vram_reserved_static_gb
                ) * 0.5  # Conservative fallback
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # Very rough heuristic for MPS
            # Assume 2GB for model+system, then use a fraction of the rest
            active_batch_est_gb = (
                self.active_items_in_gpu_processing * 0.01
            )  # Rough guess per item
            return (
                max(0, self.vram_total_gb - 2.0 - active_batch_est_gb)
                * TARGET_VRAM_UTILIZATION
            )
        return 1.0  # Default to allowing some small batches if no GPU info

    async def worker(self):
        logger.info("BatchingManager worker started.")
        while True:
            current_batch_items_with_futures: List[Dict[str, Any]] = []
            current_batch_estimated_vram_gb = 0.0

            # Try to fill a batch
            try:
                while True:  # Inner loop to accumulate batch items
                    if (
                        not current_batch_items_with_futures
                    ):  # First item in a potential batch
                        # Block until at least one item is available
                        item_with_future = await self.queue.get()
                        self.queue.task_done()  # Mark as processed from queue perspective
                    else:
                        # Try to get more items with a very short timeout if batch has started forming
                        try:
                            item_with_future = await asyncio.wait_for(
                                self.queue.get(), timeout=GPU_POLL_INTERVAL_S
                            )
                            self.queue.task_done()
                        except asyncio.TimeoutError:
                            # No more items arrived quickly, process current batch
                            break

                    item_data: MediaItem = item_with_future["item_data"]
                    item_vram_gb = self.estimate_vram_gb(item_data)
                    available_vram_gb = await self._get_available_dynamic_vram_gb()

                    if (
                        current_batch_estimated_vram_gb + item_vram_gb
                        <= available_vram_gb
                        and len(current_batch_items_with_futures) < MAX_BATCH_ITEMS
                    ):
                        current_batch_items_with_futures.append(item_with_future)
                        current_batch_estimated_vram_gb += item_vram_gb
                    else:
                        # Item doesn't fit, or max items reached. Put back and process current.
                        await self.queue.put(
                            item_with_future
                        )  # Re-queue item that didn't fit
                        logger.debug(
                            f"Item {item_data.id} (est: {item_vram_gb:.3f}GB) does not fit or batch full. Available VRAM: {available_vram_gb:.3f}GB. Current batch: {len(current_batch_items_with_futures)} items, {current_batch_estimated_vram_gb:.3f}GB."
                        )
                        break  # Process the batch accumulated so far

                    # If BATCH_FLUSH_TIMEOUT_S is very small, this check might not be effective for timeout-based flushing
                    # The primary flush mechanism is now the short timeout in wait_for above when a batch is forming.

            except Exception as e:  # Should not happen often with queue.get()
                logger.error(f"Exception in batch accumulation loop: {e}")
                await asyncio.sleep(0.1)  # Prevent tight loop on error
                continue  # Restart accumulation logic

            if current_batch_items_with_futures:
                self.batch_id_counter += 1
                internal_batch_id = (
                    f"pygpu-{self.batch_id_counter}-{uuid.uuid4().hex[:6]}"
                )
                actual_items_to_process_dicts = [
                    item_wf["item_data"].model_dump()
                    for item_wf in current_batch_items_with_futures
                ]
                futures_to_resolve_map = {
                    item_wf["item_data"].id: item_wf["future"]
                    for item_wf in current_batch_items_with_futures
                }

                self.active_items_in_gpu_processing += len(
                    actual_items_to_process_dicts
                )
                logger.info(
                    f"Processing batch '{internal_batch_id}' with {len(actual_items_to_process_dicts)} items. Est. VRAM: {current_batch_estimated_vram_gb:.3f}GB. Queue size: {self.queue.qsize()}"
                )

                try:
                    if EMBEDDING_MODEL is None:
                        raise RuntimeError("EMBEDDING_MODEL is None - model not loaded")

                    loop = asyncio.get_event_loop()
                    batch_results_map = await loop.run_in_executor(
                        None,  # Default ThreadPoolExecutor
                        emb_helper.process_media_batch,
                        actual_items_to_process_dicts,  # Pass as list of dicts
                        EMBEDDING_MODEL,
                        PYTHON_MEDIA_ROOT,
                        DEFAULT_VIDEO_FRAMES_TO_EXTRACT_CONFIG,
                        emb_helper.embedding_logger,  # Use EmbeddingLogger instance instead of standard logger
                        internal_batch_id,  # Pass internal batch_id for per-item logging context
                    )

                    global PROCESSED_FILES_COUNT_TOTAL
                    PROCESSED_FILES_COUNT_TOTAL += len(actual_items_to_process_dicts)

                    for item_id_key, result_data_val in batch_results_map.items():
                        future = futures_to_resolve_map.get(item_id_key)
                        if future and not future.done():
                            try:
                                # process_media_batch returns dicts, parse them to EmbeddingResult
                                parsed_result = EmbeddingResult(**result_data_val)
                                future.set_result(parsed_result)
                            except ValidationError as parse_exc:
                                logger.error(
                                    f"Failed to parse result for item {item_id_key} into EmbeddingResult: {parse_exc}",
                                    extra={
                                        "item_id": item_id_key,
                                        "raw_result": result_data_val,
                                    },
                                )
                                error_res = EmbeddingResult(
                                    id=item_id_key,
                                    embedding=None,
                                    error="Result parsing failed",
                                    detail=str(parse_exc),
                                    debugMetadata={
                                        "raw_result_preview": str(result_data_val)[:200]
                                    },
                                )
                                future.set_result(error_res)
                            except (
                                Exception
                            ) as generic_exc:  # Catch any other error during set_result
                                logger.error(
                                    f"Generic error setting future result for item {item_id_key}: {generic_exc}",
                                    extra={"item_id": item_id_key},
                                )
                                error_res = EmbeddingResult(
                                    id=item_id_key,
                                    embedding=None,
                                    error="Internal server error post-processing",
                                    detail=str(generic_exc),
                                    debugMetadata={},
                                )
                                future.set_result(error_res)
                        elif future and future.done():
                            logger.warning(
                                f"Future for item {item_id_key} was already done.",
                                extra={"item_id": item_id_key},
                            )

                except Exception as e_batch_proc:
                    logger.error(
                        f"Core processing for batch '{internal_batch_id}' failed: {e_batch_proc}"
                    )
                    for item_id_key_err in futures_to_resolve_map:
                        future = futures_to_resolve_map.get(item_id_key_err)
                        if future and not future.done():
                            item_error_result = EmbeddingResult(
                                id=item_id_key_err,
                                embedding=None,
                                error="Batch processing pipeline failed",
                                detail=str(e_batch_proc),
                                debugMetadata={"batch_id": internal_batch_id},
                            )
                            future.set_result(item_error_result)
                finally:
                    self.active_items_in_gpu_processing -= len(
                        actual_items_to_process_dicts
                    )
            elif (
                self.queue.empty()
            ):  # No items in batch, and queue is empty, sleep a bit
                await asyncio.sleep(BATCH_FLUSH_TIMEOUT_S)


batch_manager = BatchingManager()


# --- FastAPI Lifecycle and Endpoints ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global EMBEDDING_MODEL, EMBEDDING_DEVICE, batch_manager
    logger.info("Application startup...")
    model_name = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
    if (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ):  # Check MPS is built
        EMBEDDING_DEVICE = "mps"
    elif torch.cuda.is_available():
        EMBEDDING_DEVICE = "cuda"
    else:
        EMBEDDING_DEVICE = "cpu"

    try:
        EMBEDDING_MODEL = emb_helper.CLIPEmbedder(
            model_name=model_name,
            device=EMBEDDING_DEVICE,
            logger=emb_helper.embedding_logger,  # Use the helper's global logger which is now configured
            enable_augmentation=os.environ.get("ENABLE_AUGMENTATION", "false").lower()
            == "true",
        )
        logger.info(f"CLIP Model '{model_name}' loaded on device '{EMBEDDING_DEVICE}'.")
        if EMBEDDING_DEVICE == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.init()
                # Estimate static VRAM usage AFTER model is loaded
                batch_manager.vram_reserved_static_gb = (
                    torch.cuda.memory_allocated() / (1024**3)
                )
                logger.info(
                    f"Static VRAM allocated (model, etc.): {batch_manager.vram_reserved_static_gb:.2f} GB of {batch_manager.vram_total_gb:.2f} GB total."
                )
            except Exception as cuda_err:
                logger.error(
                    f"Error getting CUDA memory info during startup: {cuda_err}"
                )
    except Exception as model_load_err:
        logger.error(
            f"Failed to load CLIP model '{model_name}' during startup: {model_load_err}"
        )
        EMBEDDING_MODEL = None  # Ensure it's None if loading fails

    asyncio.create_task(batch_manager.worker())
    yield
    logger.info("Application shutdown.")


app = FastAPI(lifespan=lifespan, title="Embedding Service API V2 (Batching)")


@app.get("/health", response_model=ServiceHealth)
async def health_check():
    return ServiceHealth(
        status="ok" if EMBEDDING_MODEL is not None else "error_model_not_loaded",
        uptime_seconds=time.time() - SERVICE_START_TIME,
        processed_items_count=PROCESSED_FILES_COUNT_TOTAL,
        gpu_available=(
            torch.cuda.is_available()
            or (torch.backends.mps.is_available() and torch.backends.mps.is_built())
        ),
        model_loaded=EMBEDDING_MODEL is not None,
        model_name=(
            getattr(EMBEDDING_MODEL, "model_name", None) if EMBEDDING_MODEL else None
        ),
        device=EMBEDDING_DEVICE,
        request_queue_size=batch_manager.queue.qsize(),
    )


@app.post("/api/embed_batch", response_model=BatchEmbeddingResponse)
async def embed_batch_endpoint(data: BatchEmbeddingRequest, request: Request):
    client_request_id = (
        data.request_id or request.headers.get("X-Request-ID") or uuid.uuid4().hex
    )
    logger.info(
        f"Received batch request '{client_request_id}' with {len(data.items)} items."
    )

    if EMBEDDING_MODEL is None:
        logger.error(f"Model not ready for request '{client_request_id}'. Raising 503.")
        raise HTTPException(
            status_code=503, detail="Model not ready or failed to load."
        )

    item_futures: Dict[str, asyncio.Future] = {}

    for item_model_instance in data.items:
        future = asyncio.Future()
        item_futures[item_model_instance.id] = future
        try:
            # MediaItem is already validated by FastAPI if type hint is BatchEmbeddingRequest
            await batch_manager.queue.put(
                {"item_data": item_model_instance, "future": future}
            )
        except Exception as q_err:  # Should not happen with asyncio.Queue
            logger.error(
                f"Failed to put item {item_model_instance.id} on queue for request '{client_request_id}': {q_err}"
            )
            error_res = EmbeddingResult(
                id=item_model_instance.id,
                embedding=None,
                error="Failed to queue item",
                detail=str(q_err),
                debugMetadata={},
            )
            if not future.done():
                future.set_result(error_res)

    results_list: List[EmbeddingResult] = []
    try:
        # Only gather futures that were successfully created and queued
        valid_futures_to_gather = [fut for item_id, fut in item_futures.items() if fut]

        all_item_results_from_futures = await asyncio.gather(
            *valid_futures_to_gather, return_exceptions=True
        )

        for i, res_or_exc in enumerate(all_item_results_from_futures):
            # Need to map back to original item ID if gather doesn't preserve order of resolution with IDs
            # Assuming results from gather are in the same order as futures passed to it
            # This is fragile. It's better if futures carry the ID or results are dicts.
            # The `BatchingManager` resolves futures with EmbeddingResult which has an `id`.

            if isinstance(res_or_exc, EmbeddingResult):
                results_list.append(res_or_exc)
            elif isinstance(res_or_exc, Exception):
                # This means an exception bubbled up from a future that wasn't caught and wrapped in EmbeddingResult by BatchManager
                # Attempt to find which item this was for, though it's hard if only exception is returned by gather.
                # For now, log it as a general error for the request.
                # This should be rare if BatchingManager always resolves futures.
                item_id_for_error = "unknown_item_due_to_gather_exception"
                # Try to find the corresponding item if possible (e.g. if only one failed)
                # This part is tricky and ideally BatchingManager handles all errors gracefully by resolving futures
                logger.error(
                    f"Raw exception from asyncio.gather for request '{client_request_id}': {res_or_exc}",
                    exc_info=res_or_exc,
                )
                # If many items, it's hard to attribute.
                # We will rely on the fact that BatchingManager resolves all futures.
            else:  # Should be EmbeddingResult, but if not:
                logger.warning(
                    f"Unknown type {type(res_or_exc)} from asyncio.gather for request '{client_request_id}'. Item: {str(res_or_exc)[:200]}"
                )
                # This case needs an ID to form a proper EmbeddingResult. If it doesn't have one, it's problematic.
                # This indicates a bug in BatchingManager future resolution.

    except Exception as e_gather:
        logger.error(
            f"Critical error during asyncio.gather for request '{client_request_id}': {e_gather}"
        )
        # All items in this client request failed if gather itself fails.
        results_list.clear()
        for item_model_instance in data.items:
            results_list.append(
                EmbeddingResult(
                    id=item_model_instance.id,
                    embedding=None,
                    error="Server error processing batch (gather failed)",
                    detail=str(e_gather),
                    debugMetadata={},
                )
            )

    # Ensure all originally requested items have a result in the list sent back to client
    final_output_results_map = {res.id: res for res in results_list}
    complete_results_list = []
    for requested_item in data.items:
        if requested_item.id in final_output_results_map:
            complete_results_list.append(final_output_results_map[requested_item.id])
        else:
            # This means the item's future was not resolved or result was lost
            # Check if the future was created and if it resolved with an error that wasn't added to results_list
            missing_item_future = item_futures.get(requested_item.id)
            if missing_item_future and missing_item_future.done():
                try:
                    missing_item_result = missing_item_future.result()
                    if isinstance(missing_item_result, EmbeddingResult):
                        complete_results_list.append(missing_item_result)
                    else:  # Fallback
                        complete_results_list.append(
                            EmbeddingResult(
                                id=requested_item.id,
                                embedding=None,
                                error="Missing or malformed result for item",
                                detail=f"Raw future result: {str(missing_item_result)[:100]}",
                                debugMetadata={},
                            )
                        )
                except Exception as fut_final_exc:
                    complete_results_list.append(
                        EmbeddingResult(
                            id=requested_item.id,
                            embedding=None,
                            error="Error retrieving future result for missing item",
                            detail=str(fut_final_exc),
                            debugMetadata={},
                        )
                    )
            else:  # Future not found or not done, means it likely errored before even gather, or logic error
                complete_results_list.append(
                    EmbeddingResult(
                        id=requested_item.id,
                        embedding=None,
                        error="Item processing did not complete or result missing",
                        detail="Future not found or not resolved.",
                        debugMetadata={},
                    )
                )

    # Extract the Python-internal batch ID from the first result's debugMetadata, if present
    python_internal_batch_id = None
    if complete_results_list and hasattr(complete_results_list[0], "debug_metadata"):
        debug_meta = getattr(complete_results_list[0], "debug_metadata", None)
        if not debug_meta:
            debug_meta = getattr(complete_results_list[0], "debugMetadata", None)
        if debug_meta and isinstance(debug_meta, dict):
            python_internal_batch_id = debug_meta.get("batch_id") or debug_meta.get(
                "overallBatchRequestId"
            )

    return BatchEmbeddingResponse(
        results=complete_results_list,
        batch_id=python_internal_batch_id
        or client_request_id,  # Use Python-internal batch_id if available
        processed_by_request_id=client_request_id,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PYTHON_PORT", 3456))
    # Use reload for development if preferred: uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
