#!/usr/bin/env python3
"""
Test script to verify GCS integration works correctly.
This can be run inside the Docker container to test the GCS functionality.
"""

import sys
import os

sys.path.append("/app")


def test_gcs_import():
    """Test if GCS modules can be imported."""
    try:
        from cloud import get_gcs_client, OptimizedGCSClient

        print("✅ GCS cloud module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import GCS cloud module: {e}")
        return False


def test_gcs_client_creation():
    """Test if GCS client can be created."""
    try:
        from cloud import get_gcs_client

        # Mock logger for testing
        class MockLogger:
            def debug(self, msg, **kwargs):
                pass

            def info(self, msg, **kwargs):
                pass

            def error(self, msg, **kwargs):
                pass

        client = get_gcs_client(logger=MockLogger())
        print("✅ GCS client created successfully")
        print(f"   Bucket: {client.bucket.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to create GCS client: {e}")
        return False


def test_embedding_helper_import():
    """Test if embedding helper with GCS support can be imported."""
    try:
        from embedding_service_helper import _preprocess_single_item_for_batch

        print("✅ Embedding helper with GCS support imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import embedding helper: {e}")
        return False


def test_server_model():
    """Test if server models accept gcs_blob source type."""
    try:
        from server import MediaItem

        # Test creating a GCS blob media item
        gcs_item = MediaItem(
            id="test-gcs-1",
            media_type="image",
            source_type="gcs_blob",
            source="test-images/sample.jpg",
            original_filename="sample.jpg",
        )
        print("✅ Server MediaItem model accepts gcs_blob source type")
        print(f"   Item: {gcs_item.source_type} -> {gcs_item.source}")
        return True
    except Exception as e:
        print(f"❌ Failed to create GCS MediaItem: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing GCS Integration in Embedding Service")
    print("=" * 50)

    tests = [
        test_gcs_import,
        test_gcs_client_creation,
        test_embedding_helper_import,
        test_server_model,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()

    passed = sum(results)
    total = len(results)

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! GCS integration is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
