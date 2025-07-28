#!/usr/bin/env python3
"""
Simple test script to verify the BatchingManager improvements work correctly.
"""
import asyncio
import os
import sys
import time
from unittest.mock import Mock, patch

# Add the current directory to the path so we can import server
sys.path.insert(0, os.path.dirname(__file__))

# Mock the embedding helper before importing server
sys.modules["embedding_service_helper"] = Mock()
sys.modules["embedding_service_helper"].process_media_batch = Mock()
sys.modules["embedding_service_helper"].CLIPEmbedder = Mock()
sys.modules["embedding_service_helper"].embedding_logger = Mock()

# Import after mocking
from server import BatchingManager, MediaItem


def test_large_item_detection():
    """Test that large items are correctly identified"""
    print("Testing large item detection...")

    # Create a BatchingManager instance
    manager = BatchingManager()

    # Test regular image (should not be large)
    regular_image = MediaItem(
        id="test1", media_type="image", source_type="filepath", source="test.jpg"
    )

    # Test regular video with few frames (should not be large)
    small_video = MediaItem(
        id="test2",
        media_type="video",
        source_type="filepath",
        source="test.mp4",
        num_frames=10,
    )

    # Test large video with many frames (should be large)
    large_video = MediaItem(
        id="test3",
        media_type="video",
        source_type="filepath",
        source="test_large.mp4",
        num_frames=90,
    )

    # Check VRAM estimates
    regular_vram = manager.estimate_vram_gb(regular_image)
    small_video_vram = manager.estimate_vram_gb(small_video)
    large_video_vram = manager.estimate_vram_gb(large_video)

    print(f"Regular image VRAM estimate: {regular_vram:.4f} GB")
    print(f"Small video (10 frames) VRAM estimate: {small_video_vram:.4f} GB")
    print(f"Large video (90 frames) VRAM estimate: {large_video_vram:.4f} GB")

    # Check large item detection
    print(f"Regular image is large: {manager._is_large_item(regular_image)}")
    print(f"Small video is large: {manager._is_large_item(small_video)}")
    print(f"Large video is large: {manager._is_large_item(large_video)}")

    # Verify expectations
    assert not manager._is_large_item(
        regular_image
    ), "Regular image should not be large"
    assert not manager._is_large_item(small_video), "Small video should not be large"
    assert manager._is_large_item(
        large_video
    ), "Large video should be detected as large"

    print("✓ Large item detection test passed!")


async def test_queue_routing():
    """Test that items are routed to the correct queues"""
    print("\nTesting queue routing...")

    manager = BatchingManager()

    # Create test items
    regular_item = MediaItem(
        id="regular", media_type="image", source_type="filepath", source="test.jpg"
    )

    large_item = MediaItem(
        id="large",
        media_type="video",
        source_type="filepath",
        source="large.mp4",
        num_frames=90,
    )

    # Create mock futures
    regular_future = asyncio.Future()
    large_future = asyncio.Future()

    # Add items to manager
    await manager.add_item(regular_item, regular_future)
    await manager.add_item(large_item, large_future)

    # Check queue sizes
    regular_queue_size = manager.queue.qsize()
    large_queue_size = manager.large_item_queue.qsize()

    print(f"Regular queue size: {regular_queue_size}")
    print(f"Large item queue size: {large_queue_size}")

    # Verify routing
    assert regular_queue_size == 1, "Regular item should be in regular queue"
    assert large_queue_size == 1, "Large item should be in large item queue"

    print("✓ Queue routing test passed!")


def test_vram_estimation_accuracy():
    """Test VRAM estimation improvements"""
    print("\nTesting VRAM estimation accuracy...")

    manager = BatchingManager()

    # Test various frame counts
    frame_counts = [1, 10, 30, 60, 90, 120]

    for frames in frame_counts:
        video_item = MediaItem(
            id=f"video_{frames}",
            media_type="video",
            source_type="filepath",
            source="test.mp4",
            num_frames=frames,
        )

        vram_estimate = manager.estimate_vram_gb(video_item)
        print(f"Video with {frames} frames: {vram_estimate:.4f} GB")

        # VRAM should increase with frame count
        if frames > 1:
            prev_video = MediaItem(
                id=f"video_{frames-1}",
                media_type="video",
                source_type="filepath",
                source="test.mp4",
                num_frames=frames - 1,
            )
            prev_vram = manager.estimate_vram_gb(prev_video)
            assert vram_estimate > prev_vram, f"VRAM should increase with frame count"

    print("✓ VRAM estimation test passed!")


async def main():
    """Run all tests"""
    print("Running BatchingManager tests...\n")

    try:
        # Set test environment variables
        os.environ.setdefault("LARGE_ITEM_THRESHOLD_GB", "0.5")
        os.environ.setdefault("MAX_CONCURRENT_LARGE_ITEMS", "2")

        test_large_item_detection()
        await test_queue_routing()
        test_vram_estimation_accuracy()

        print(
            "\n🎉 All tests passed! The BatchingManager improvements are working correctly."
        )

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
