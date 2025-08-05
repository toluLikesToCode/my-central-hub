import os
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Tuple
from google.cloud import storage  # type: ignore
from dotenv import load_dotenv  # type: ignore
from PIL import Image  # type: ignore

# Ensure .env is loaded at the very top
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "media-content-1")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS", "./personal-project-main-02e1cf2744c0.json"
)

# Performance tuning
GCS_MAX_WORKERS = int(os.environ.get("GCS_MAX_WORKERS", 8))
GCS_TIMEOUT = int(os.environ.get("GCS_TIMEOUT", 60))

storage_client = storage.Client(project="personal-project-main")
storage_client._http.timeout = GCS_TIMEOUT


class OptimizedGCSClient:
    """High-performance GCS client for embedding service."""

    def __init__(self, logger=None):
        self.bucket = storage_client.bucket(GCS_BUCKET_NAME)
        self.logger = logger
        self.executor = ThreadPoolExecutor(max_workers=GCS_MAX_WORKERS)

    def _log(self, level: str, message: str, extra: Optional[Dict] = None):
        if self.logger:
            getattr(self.logger, level.lower())(message, extra=extra)

    def download_to_memory(self, blob_name: str) -> Optional[bytes]:
        """Download blob content to memory."""
        try:
            blob = self.bucket.blob(blob_name)
            buffer = io.BytesIO()
            blob.download_to_file(buffer, checksum="md5")
            buffer.seek(0)
            content = buffer.getvalue()
            self._log(
                "debug", f"Downloaded GCS blob '{blob_name}' ({len(content)} bytes)"
            )
            return content
        except Exception as e:
            self._log("error", f"Error downloading GCS blob '{blob_name}': {e}")
            return None

    def download_to_pil_image(self, blob_name: str) -> Optional[Image.Image]:
        """Download blob and convert directly to PIL Image."""
        content = self.download_to_memory(blob_name)
        if content:
            try:
                image = Image.open(io.BytesIO(content)).convert("RGB")
                self._log(
                    "debug",
                    f"Converted GCS blob '{blob_name}' to PIL Image ({image.width}x{image.height})",
                )
                return image
            except Exception as e:
                self._log(
                    "error",
                    f"Error converting GCS blob '{blob_name}' to PIL Image: {e}",
                )
        return None

    def batch_download_to_pil_images(
        self, blob_names: List[str]
    ) -> Dict[str, Optional[Image.Image]]:
        """Download multiple blobs and convert to PIL Images in parallel."""
        if not blob_names:
            return {}

        def download_and_convert(blob_name: str) -> Tuple[str, Optional[Image.Image]]:
            return blob_name, self.download_to_pil_image(blob_name)

        results = {}
        with ThreadPoolExecutor(
            max_workers=min(len(blob_names), GCS_MAX_WORKERS)
        ) as executor:
            futures = [
                executor.submit(download_and_convert, blob_name)
                for blob_name in blob_names
            ]
            for future in futures:
                blob_name, image = future.result()
                results[blob_name] = image

        successful = sum(1 for img in results.values() if img is not None)
        self._log("info", f"Batch downloaded {successful}/{len(blob_names)} GCS images")
        return results

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)


# Global instance
gcs_client: Optional[OptimizedGCSClient] = None


def get_gcs_client(logger=None) -> OptimizedGCSClient:
    """Get or create GCS client instance."""
    global gcs_client
    if gcs_client is None:
        gcs_client = OptimizedGCSClient(logger=logger)
    return gcs_client
