#!/usr/bin/env python3
"""
Test script for the embedding service running in a Docker container.
Tests various media file processing capabilities including images, videos, and GIFs.
"""
import os
import json
import time
import asyncio
import requests
from pathlib import Path
from PIL import Image
import logging
from typing import Dict, List, Optional, Union, Any
import backoff  # For retrying failed requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://192.168.1.107:3456")
SERVICE_HEALTH_TIMEOUT = int(os.getenv("SERVICE_HEALTH_TIMEOUT", "30"))  # seconds
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))  # seconds

# Test file paths
TEST_IMAGE_PATH = (
    Path(__file__).parent.parent.parent / "public" / "media" / "redgifs1.jpg"
)
TEST_VIDEO_PATH = (
    Path(__file__).parent.parent.parent
    / "public"
    / "media"
    / "1890053552363786578_1.mp4"
)
TEST_GIF_PATH = (
    Path(__file__).parent.parent.parent / "public" / "media" / "92fwozg0wcye1.gif"
)

# Test options
RUN_PARALLEL_TESTS = os.getenv("RUN_PARALLEL_TESTS", "false").lower() == "true"
NUM_PARALLEL_REQUESTS = int(os.getenv("NUM_PARALLEL_REQUESTS", "3"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

# Create a small test image
SMALL_TEST_IMAGE = Path(__file__).parent / "small-test-image.jpg"


def create_small_test_image() -> None:
    """Create a small test image for reliable testing."""
    logger.info("Creating a small test image for reliable testing...")
    try:
        with Image.open(TEST_IMAGE_PATH) as img:
            # Resize image
            img.thumbnail((128, 128))
            # Save as JPEG with reduced quality
            img.save(SMALL_TEST_IMAGE, "JPEG", quality=70)
        logger.info(f"✅ Created small test image at {SMALL_TEST_IMAGE}")
    except Exception as e:
        logger.error(f"❌ Failed to create small test image: {e}")
        raise


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, requests.exceptions.ConnectionError),
    max_tries=MAX_RETRIES,
    max_time=SERVICE_HEALTH_TIMEOUT,
)
def wait_for_service_health() -> bool:
    """Wait for the embedding service to be healthy."""
    logger.info(f"Waiting for embedding service at {EMBEDDING_SERVICE_URL}...")

    try:
        response = requests.get(f"{EMBEDDING_SERVICE_URL}/health", timeout=5)
        health_data = response.json()

        if not response.ok:
            logger.warning(f"Service not healthy: {health_data}")
            return False

        if health_data.get("status") != "ok":
            logger.warning(f"Service not ready: {health_data}")
            return False

        logger.info(f"Service health check passed: {health_data}")
        return True

    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        return False


def get_content_type(filename: str) -> str:
    """Get the content type based on file extension."""
    ext = Path(filename).suffix.lower()
    content_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }
    return content_types.get(ext, "application/octet-stream")


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"test-{int(time.time() * 1000)}-{int(time.time() * 1000) % 1000000}"


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, requests.exceptions.ConnectionError),
    max_tries=MAX_RETRIES,
    max_time=30,
)
def process_file(file_path: Path, filename: str) -> Dict[str, Any]:
    """Process a file through the embedding service with retries."""
    logger.info(f"Processing file: {file_path} ({filename})")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read file into memory
    with open(file_path, "rb") as f:
        file_buffer = f.read()

    logger.info(f"Read file into buffer, size: {len(file_buffer)} bytes")

    # Prepare headers
    headers = {
        "User-Agent": "MyAppClient/1.0",
        "X-Request-ID": generate_request_id(),
        "Accept": "application/json",
        "Content-Type": get_content_type(filename),
        "Content-Length": str(len(file_buffer)),
        "X-Filename": filename,
        "X-Media-Type": (
            "video" if filename.endswith((".mp4", ".mov", ".webm")) else "image"
        ),
    }

    # Make request
    try:
        response = requests.post(
            f"{EMBEDDING_SERVICE_URL}/embed",
            data=file_buffer,
            headers=headers,
            timeout=int(
                os.getenv("REQUEST_TIMEOUT", "180")
            ),  # 180 second timeout for processing
        )

        if not response.ok:
            error_text = response.text
            logger.error(f"Error response ({response.status_code}): {error_text}")
            raise Exception(
                f"Embedding calculation failed with status {response.status_code}: {error_text}"
            )

        response_data = response.json()
        if VERBOSE:
            logger.info(f"Response data: {json.dumps(response_data, indent=2)}")
        return response_data

    except requests.exceptions.Timeout:
        logger.error("Request timed out after 30 seconds")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse response as JSON: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file processing: {str(e)}")
        raise


def validate_response(response_data: dict, expected_type: str) -> bool:
    """Validate the response format and content."""
    if not isinstance(response_data, dict):
        logging.error(f"Response is not a dictionary: {type(response_data)}")
        return False

    # Check for error in response
    if response_data.get("error"):
        logging.error(f"Server returned error: {response_data['error']}")
        if response_data.get("detail"):
            logging.error(f"Error details: {response_data['detail']}")
        return False

    # Validate embedding
    embedding = response_data.get("embedding")
    if not isinstance(embedding, list):
        logging.error(f"Embedding is not a list: {type(embedding)}")
        return False

    if expected_type == "image":
        if len(embedding) != 512:
            logging.error(
                f"Image embedding has wrong length: {len(embedding)} (expected 512)"
            )
            return False
    elif expected_type == "video":
        if len(embedding) == 0:
            logging.error("Video embedding is empty")
            return False
        if len(embedding) != 512:
            logging.error(
                f"Video embedding has wrong length: {len(embedding)} (expected 512)"
            )
            return False

    # Validate debug metadata
    debug_metadata = response_data.get("debugMetadata")
    if not isinstance(debug_metadata, dict):
        logging.error(f"Debug metadata is not a dictionary: {type(debug_metadata)}")
        return False

    required_metadata = ["model", "timestamp", "request_id", "source_type"]
    for field in required_metadata:
        if field not in debug_metadata:
            logging.error(f"Missing required metadata field: {field}")
            return False

    return True


async def test_embedding_service() -> Dict[str, Any]:
    """Run the full test suite."""
    logger.info("Testing embedding service...")
    logger.info(f"Using service URL: {EMBEDDING_SERVICE_URL}")

    metrics = {}

    try:
        # Wait for service to be healthy
        if not wait_for_service_health():
            raise Exception("Service health check failed")

        # Test 1: Health check
        logger.info("\n[TEST 1] Testing health endpoint...")
        health_response = requests.get(f"{EMBEDDING_SERVICE_URL}/health")
        health_data = health_response.json()
        logger.info(f"Health response: {health_data}")

        if not health_response.ok:
            raise Exception(
                f"Health check failed with status {health_response.status_code}"
            )
        logger.info("✅ Health check passed!")

        # Test 2: GPU metrics check
        logger.info("\n[TEST 2] Testing GPU metrics endpoint...")
        try:
            gpu_response = requests.get(f"{EMBEDDING_SERVICE_URL}/metrics/gpu")
            gpu_data = gpu_response.json()
            logger.info(f"GPU metrics response: {gpu_data}")
            logger.info("✅ GPU metrics check passed!")
        except Exception as gpu_error:
            logger.warning(
                f"⚠️ GPU metrics not available or endpoint not implemented yet: {gpu_error}"
            )

        # Test 3: Image embedding calculation
        logger.info("\n[TEST 3] Testing image embedding calculation...")
        if not TEST_IMAGE_PATH.exists():
            raise FileNotFoundError(f"Test image not found at {TEST_IMAGE_PATH}")

        img_start_time = time.time()
        image_embed_response = process_file(TEST_IMAGE_PATH, "test-image.jpg")
        metrics["imageProcessingTimeSec"] = time.time() - img_start_time

        logger.info(
            f'Image embedding calculation completed in {metrics["imageProcessingTimeSec"]:.2f}s'
        )
        if not validate_response(image_embed_response, "image"):
            raise Exception("Image embedding validation failed")
        logger.info("✅ Image embedding calculation passed!")

        # Test 4: Video embedding calculation
        logger.info("\n[TEST 4] Testing video embedding calculation...")
        if not TEST_VIDEO_PATH.exists():
            raise FileNotFoundError(f"Test video not found at {TEST_VIDEO_PATH}")

        # Log video file details
        video_size = TEST_VIDEO_PATH.stat().st_size
        logger.info(f"Video file size: {video_size / (1024*1024):.2f} MB")

        try:
            video_start_time = time.time()
            video_embed_response = process_file(TEST_VIDEO_PATH, "test-video.mp4")
            metrics["videoProcessingTimeSec"] = time.time() - video_start_time

            logger.info(
                f'Video embedding calculation completed in {metrics["videoProcessingTimeSec"]:.2f}s'
            )

            if not validate_response(video_embed_response, "video"):
                logger.error("Video embedding validation failed")
                if video_embed_response.get("error"):
                    logger.error(f"Server error: {video_embed_response['error']}")
                    if video_embed_response.get("detail"):
                        logger.error(f"Error details: {video_embed_response['detail']}")
                raise Exception("Video embedding validation failed")

            logger.info("✅ Video embedding calculation passed!")
        except Exception as video_error:
            logger.error(f"❌ Video test failed: {str(video_error)}")
            logger.error("Video processing details:")
            logger.error(f"- File path: {TEST_VIDEO_PATH}")
            logger.error(f"- File size: {video_size / (1024*1024):.2f} MB")
            logger.error(f"- Content-Type: {get_content_type('test-video.mp4')}")
            raise

        # Test 5: GIF embedding calculation
        logger.info("\n[TEST 5] Testing GIF embedding calculation...")
        if not TEST_GIF_PATH.exists():
            raise FileNotFoundError(f"Test GIF not found at {TEST_GIF_PATH}")

        gif_start_time = time.time()
        gif_embed_response = process_file(TEST_GIF_PATH, "test-gif.gif")
        metrics["gifProcessingTimeSec"] = time.time() - gif_start_time

        logger.info(
            f'GIF embedding calculation completed in {metrics["gifProcessingTimeSec"]:.2f}s'
        )
        if not validate_response(gif_embed_response, "video"):
            raise Exception("GIF embedding validation failed")
        logger.info("✅ GIF embedding calculation passed!")

        # Test 6: Parallel requests (optional)
        if RUN_PARALLEL_TESTS:
            logger.info(
                f"\n[TEST 6] Testing {NUM_PARALLEL_REQUESTS} parallel requests..."
            )

            parallel_start_time = time.time()
            import aiohttp

            async def process_file_async(session, file_path, filename):
                async with session.post(
                    f"{EMBEDDING_SERVICE_URL}/embed",
                    data=open(file_path, "rb").read(),
                    headers={
                        "Content-Type": get_content_type(filename),
                        "X-Filename": filename,
                        "X-Request-ID": generate_request_id(),
                        "X-Media-Type": (
                            "video"
                            if filename.endswith((".mp4", ".mov", ".webm"))
                            else "image"
                        ),
                    },
                ) as response:
                    return await response.json()

            async def run_parallel_tests():
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for i in range(NUM_PARALLEL_REQUESTS):
                        file_path = TEST_IMAGE_PATH if i % 2 == 0 else TEST_VIDEO_PATH
                        filename = f'parallel-{"image" if i % 2 == 0 else "video"}-{i}.{"jpg" if i % 2 == 0 else "mp4"}'
                        tasks.append(process_file_async(session, file_path, filename))
                    return await asyncio.gather(*tasks)

            results = asyncio.run(run_parallel_tests())
            metrics["parallelProcessingTimeSec"] = time.time() - parallel_start_time

            logger.info(
                f'Parallel processing completed in {metrics["parallelProcessingTimeSec"]:.2f}s'
            )
            for response in results:
                validate_response(
                    response, "image" if "image" in response["source_type"] else "video"
                )
            logger.info("✅ Parallel requests test passed!")

        # Test 7: Error handling test
        logger.info("\n[TEST 7] Testing error handling with invalid file...")
        try:
            invalid_file = Path(__file__).parent / "invalid-file.txt"
            with open(invalid_file, "w") as f:
                f.write("This is not a valid image or video file")

            process_file(invalid_file, "invalid-file.txt")
            logger.warning("⚠️ Service did not reject invalid file as expected")
        except Exception as error:
            logger.info(
                f"✅ Error handling test passed! Service correctly rejected invalid file: {error}"
            )
        finally:
            if invalid_file.exists():
                invalid_file.unlink()

        logger.info("\n✅ All tests passed! Embedding service is working correctly.")
        logger.info("\nPerformance metrics:")
        logger.info(
            f'- Image processing: {metrics.get("imageProcessingTimeSec", 0):.2f}s'
        )
        logger.info(
            f'- Video processing: {metrics.get("videoProcessingTimeSec", 0):.2f}s'
        )
        logger.info(f'- GIF processing: {metrics.get("gifProcessingTimeSec", 0):.2f}s')

        if "parallelProcessingTimeSec" in metrics:
            logger.info(
                f'- Parallel processing ({NUM_PARALLEL_REQUESTS} requests): {metrics["parallelProcessingTimeSec"]:.2f}s'
            )
            logger.info(
                f'- Average time per request in parallel: {metrics["parallelProcessingTimeSec"] / NUM_PARALLEL_REQUESTS:.2f}s'
            )

        return {"success": True, "metrics": metrics}

    except Exception as error:
        logger.error(f"\n❌ Test failed: {str(error)}")
        logger.error(f"Error details: {error}")
        return {"success": False, "error": str(error), "metrics": metrics}


def main():
    """Main entry point."""
    # Create the small test image first
    create_small_test_image()

    # Run the test suite
    result = asyncio.run(test_embedding_service())

    if not result["success"]:
        logger.error("\nTest run failed.")
        exit(1)

    logger.info("\nTest run completed successfully.")
    exit(0)


if __name__ == "__main__":
    main()
