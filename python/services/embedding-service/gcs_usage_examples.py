#!/usr/bin/env python3
"""
Example usage of the GCS-enabled embedding service.

This demonstrates how to make API requests to the embedding service
with GCS blob sources for maximum performance.
"""

import requests
import json
from typing import List, Dict, Any, Optional

# Example API endpoint (adjust as needed)
EMBEDDING_SERVICE_URL = "http://localhost:8080/api/embed_batch"


def create_gcs_batch_request(
    gcs_items: List[Dict[str, Any]], request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a batch embedding request with GCS blob sources.

    Args:
        gcs_items: List of GCS items with blob names and metadata
        request_id: Optional request ID for tracking

    Returns:
        Formatted batch request payload
    """
    return {"items": gcs_items, "request_id": request_id, "isDryRun": False}


def example_image_batch():
    """Example batch request with GCS image blobs."""
    gcs_images = [
        {
            "id": "img-001",
            "media_type": "image",
            "source_type": "gcs_blob",
            "source": "photos/vacation/beach-sunset.jpg",
            "original_filename": "beach-sunset.jpg",
        },
        {
            "id": "img-002",
            "media_type": "image",
            "source_type": "gcs_blob",
            "source": "photos/vacation/mountain-view.jpg",
            "original_filename": "mountain-view.jpg",
        },
        {
            "id": "img-003",
            "media_type": "image",
            "source_type": "gcs_blob",
            "source": "photos/pets/dog-playing.jpg",
            "original_filename": "dog-playing.jpg",
        },
    ]

    return create_gcs_batch_request(gcs_images, "batch-images-001")


def example_video_batch():
    """Example batch request with GCS video blobs."""
    gcs_videos = [
        {
            "id": "vid-001",
            "media_type": "video",
            "source_type": "gcs_blob",
            "source": "videos/events/wedding-ceremony.mp4",
            "num_frames": 20,
            "original_filename": "wedding-ceremony.mp4",
        },
        {
            "id": "vid-002",
            "media_type": "video",
            "source_type": "gcs_blob",
            "source": "videos/tutorials/cooking-demo.mp4",
            "num_frames": 15,
            "original_filename": "cooking-demo.mp4",
        },
    ]

    return create_gcs_batch_request(gcs_videos, "batch-videos-001")


def example_mixed_batch():
    """Example batch request mixing images and videos from GCS."""
    mixed_items = [
        {
            "id": "mixed-001",
            "media_type": "image",
            "source_type": "gcs_blob",
            "source": "portfolio/designs/logo-v1.png",
            "original_filename": "logo-v1.png",
        },
        {
            "id": "mixed-002",
            "media_type": "video",
            "source_type": "gcs_blob",
            "source": "portfolio/demos/app-walkthrough.mp4",
            "num_frames": 10,
            "original_filename": "app-walkthrough.mp4",
        },
        {
            "id": "mixed-003",
            "media_type": "image",
            "source_type": "gcs_blob",
            "source": "portfolio/screenshots/dashboard.jpg",
            "original_filename": "dashboard.jpg",
        },
    ]

    return create_gcs_batch_request(mixed_items, "batch-mixed-001")


def send_batch_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send batch request to embedding service.

    Args:
        payload: Batch request payload

    Returns:
        Response from embedding service
    """
    try:
        response = requests.post(
            EMBEDDING_SERVICE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,  # 5 minutes timeout for large batches
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}


def print_batch_results(response: Dict[str, Any]):
    """Print formatted batch results."""
    if "error" in response:
        print(f"❌ Batch failed: {response['error']}")
        return

    results = response.get("results", [])
    batch_id = response.get("batch_id", "unknown")

    print(f"📊 Batch Results (ID: {batch_id})")
    print(f"   Total Items: {len(results)}")

    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    print(f"   ✅ Successful: {len(successful)}")
    print(f"   ❌ Failed: {len(failed)}")

    if failed:
        print("\n   Failed Items:")
        for item in failed:
            print(f"     - {item.get('id')}: {item.get('error')}")

    if successful:
        print(f"\n   ✅ Successfully processed {len(successful)} items with embeddings")
        # Show embedding dimensions from first successful item
        first_embedding = successful[0].get("embedding")
        if first_embedding:
            print(f"   📏 Embedding dimension: {len(first_embedding)}")


def main():
    """Run example batch requests."""
    print("🚀 GCS Embedding Service Usage Examples")
    print("=" * 50)

    # Example 1: Image batch
    print("\n1️⃣  Processing GCS Image Batch")
    image_batch = example_image_batch()
    print(f"   Request: {len(image_batch['items'])} images")
    print("   Sending request...")

    # Uncomment to actually send requests:
    # image_response = send_batch_request(image_batch)
    # print_batch_results(image_response)

    print("   📄 Example payload:")
    print(json.dumps(image_batch, indent=2))

    print("\n" + "=" * 50)

    # Example 2: Video batch
    print("\n2️⃣  Processing GCS Video Batch")
    video_batch = example_video_batch()
    print(f"   Request: {len(video_batch['items'])} videos")
    print("   📄 Example payload:")
    print(json.dumps(video_batch, indent=2))

    print("\n" + "=" * 50)

    # Example 3: Mixed batch
    print("\n3️⃣  Processing Mixed GCS Batch")
    mixed_batch = example_mixed_batch()
    print(f"   Request: {len(mixed_batch['items'])} mixed items")
    print("   📄 Example payload:")
    print(json.dumps(mixed_batch, indent=2))

    print("\n" + "=" * 50)
    print("💡 Performance Benefits of GCS Integration:")
    print("   • Direct streaming downloads (no temp files for images)")
    print("   • Parallel batch downloads for multiple images")
    print("   • Optimized connection pooling")
    print("   • Automatic retry handling")
    print("   • Reduced memory usage with streaming")
    print("   • Faster than URL downloads due to GCS proximity")


if __name__ == "__main__":
    main()
