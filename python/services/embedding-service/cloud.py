import os
import io
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Tuple

from google.cloud import storage  # type: ignore
from google.api_core import exceptions  # type: ignore
from dotenv import load_dotenv  # type: ignore
from PIL import Image  # type: ignore

# Ensure .env is loaded at the very top
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "media-content-1")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS", "./personal-project-main-02e1cf2744c0.json"
)
GCS_MAX_WORKERS = int(os.environ.get("GCS_MAX_WORKERS", 8))
GCS_TIMEOUT = int(os.environ.get("GCS_TIMEOUT", 60))

# --- Global GCS Client Initialization ---
# This client is initialized once and shared.
storage_client = storage.Client(project="personal-project-main")
storage_client._http.timeout = GCS_TIMEOUT


class OptimizedGCSClient:
    """
    High-performance GCS client optimized for streaming and in-memory processing.
    """

    def __init__(self, logger: Optional[Any] = None):
        """
        Initializes the client with a bucket handle and a thread pool.

        Args:
            logger: An optional logger instance for logging operations.
        """
        self.bucket = storage_client.bucket(GCS_BUCKET_NAME)
        self.logger = logger
        self.executor = ThreadPoolExecutor(max_workers=GCS_MAX_WORKERS)

    def _log(self, level: str, message: str, extra: Optional[Dict] = None):
        """Helper method for logging if a logger is provided."""
        if self.logger:
            getattr(self.logger, level.lower())(message, extra=extra or {})

    def _get_blob_stream(self, blob_name: str) -> Optional[io.BufferedReader]:
        """
        Opens a file-like object for a blob that streams its contents.
        This is the foundation for all efficient read operations.

        Args:
            blob_name: The name/path of the object in the GCS bucket.

        Returns:
            A file-like object for reading the blob's content, or None if not found.
        """
        try:
            blob = self.bucket.blob(blob_name)
            # blob.open() returns a file-like object that streams data.
            return blob.open("rb")
        except exceptions.NotFound:
            self._log(
                "error", f"Blob '{blob_name}' not found in bucket '{self.bucket.name}'."
            )
            return None
        except Exception as e:
            self._log("error", f"Error opening stream for blob '{blob_name}': {e}")
            return None

    def stream_to_subprocess(
        self, blob_name: str, command: List[str]
    ) -> Optional[subprocess.CompletedProcess]:
        """
        Streams a blob's content directly to a subprocess's stdin without saving to a temp file.

        Args:
            blob_name: The name/path of the object in the GCS bucket.
            command: A list of strings representing the command and its arguments.

        Returns:
            The subprocess.CompletedProcess object on success, or None on error.
        """
        self._log(
            "info", f"Piping stream from '{blob_name}' to command: {' '.join(command)}"
        )

        blob_stream = self._get_blob_stream(blob_name)
        if not blob_stream:
            return None

        try:
            with blob_stream:
                result = subprocess.run(
                    command,
                    stdin=blob_stream,
                    capture_output=True,
                    check=True,  # Raises CalledProcessError for non-zero exit codes
                    timeout=GCS_TIMEOUT,
                )
            self._log("debug", f"Subprocess for '{blob_name}' completed successfully.")
            return result
        except FileNotFoundError:
            self._log(
                "error",
                f"Command not found: '{command[0]}'. Ensure it's in your system's PATH.",
            )
            return None
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", "ignore")
            self._log(
                "error",
                f"Subprocess for '{blob_name}' failed with exit code {e.returncode}",
                extra={"stderr": stderr},
            )
            return None
        except subprocess.TimeoutExpired:
            self._log(
                "error",
                f"Subprocess for '{blob_name}' timed out after {GCS_TIMEOUT} seconds.",
            )
            return None
        except Exception as e:
            self._log(
                "error",
                f"An unexpected error occurred during subprocess execution for '{blob_name}': {e}",
            )
            return None

    def download_to_memory(self, blob_name: str) -> Optional[bytes]:
        """
        Downloads blob content to memory by reading from a stream.

        Args:
            blob_name: The name of the blob to download.

        Returns:
            The blob content as bytes, or None on error.
        """
        stream = self._get_blob_stream(blob_name)
        if stream:
            with stream:
                try:
                    content = stream.read()
                    self._log(
                        "debug",
                        f"Downloaded GCS blob '{blob_name}' to memory ({len(content)} bytes)",
                    )
                    return content
                except Exception as e:
                    self._log(
                        "error",
                        f"Error reading stream to memory for '{blob_name}': {e}",
                    )
        return None

    def download_to_pil_image(self, blob_name: str) -> Optional[Image.Image]:
        """
        Downloads a blob and converts it directly to a PIL Image by streaming,
        avoiding loading the entire file into a separate memory buffer first.

        Args:
            blob_name: The name of the image blob.

        Returns:
            A PIL Image object, or None on error.
        """
        stream = self._get_blob_stream(blob_name)
        if stream:
            with stream:
                try:
                    # Image.open can read directly from the file-like stream object
                    image = Image.open(stream).convert("RGB")
                    self._log(
                        "debug",
                        f"Converted GCS blob '{blob_name}' to PIL Image ({image.width}x{image.height})",
                    )
                    return image
                except Exception as e:
                    self._log(
                        "error",
                        f"Error converting GCS stream '{blob_name}' to PIL Image: {e}",
                    )
        return None

    def batch_download_to_pil_images(
        self, blob_names: List[str]
    ) -> Dict[str, Optional[Image.Image]]:
        """
        Downloads multiple blobs and converts them to PIL Images in parallel
        using the class's shared thread pool.

        Args:
            blob_names: A list of blob names to download.

        Returns:
            A dictionary mapping blob names to PIL Image objects or None.
        """
        if not blob_names:
            return {}

        results = {}
        # Use the class's persistent executor for efficiency
        futures = {
            self.executor.submit(self.download_to_pil_image, name): name
            for name in blob_names
        }

        for future in futures:
            blob_name = futures[future]
            try:
                results[blob_name] = future.result()
            except Exception as e:
                self._log(
                    "error",
                    f"A task in the batch download for '{blob_name}' failed: {e}",
                )
                results[blob_name] = None

        successful = sum(1 for img in results.values() if img is not None)
        self._log(
            "info",
            f"Batch download complete: {successful}/{len(blob_names)} images processed successfully.",
        )
        return results

    def __del__(self):
        """Shutdown the thread pool executor when the object is destroyed."""
        if hasattr(self, "executor"):
            self._log("debug", "Shutting down ThreadPoolExecutor.")
            self.executor.shutdown(wait=True)


# --- Global Instance Factory ---
gcs_client: Optional[OptimizedGCSClient] = None


def get_gcs_client(logger: Optional[Any] = None) -> OptimizedGCSClient:
    """
    Factory function to get a singleton instance of the GCS client.

    Args:
        logger: A logger instance to be used by the client.

    Returns:
        The singleton OptimizedGCSClient instance.
    """
    global gcs_client
    if gcs_client is None:
        gcs_client = OptimizedGCSClient(logger=logger)
    return gcs_client
