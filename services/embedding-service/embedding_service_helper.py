import subprocess
import time


class VideoProcessor:
    def __init__(self, video_path=None, video_buffer=None, logger=None):
        """Initialize the video processor with either a file path or a buffer."""
        self.video_path = video_path
        self.video_buffer = video_buffer
        self.logger = logger or EmbeddingLogger("video-processor")
        self._duration = None

    def _log_debug(self, message, extra=None):
        """Log debug message with optional extra data."""
        if hasattr(self.logger, "debug"):
            self.logger.debug(message, extra=extra)

    def _log_error(self, message, error=None, extra=None):
        """Log error message with optional error and extra data."""
        if hasattr(self.logger, "error"):
            self.logger.error(message, error=error, extra=extra)

    def get_duration(self) -> float:
        """Get the duration of the video in seconds using ffprobe."""
        if self._duration is not None:
            return self._duration

        start_time = time.time()
        source_display = self.video_path if self.video_path else "in-memory buffer"
        extra_args = {"video_source": source_display}

        self._log_debug(
            f"Getting duration for video: {source_display}", extra=extra_args
        )

        try:
            if self.video_buffer:
                self.video_buffer.seek(0)
                command = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    "-i",
                    "pipe:0",
                ]
                result = subprocess.run(
                    command,
                    input=self.video_buffer.read(),  # send bytes through stdin
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,  # no text=True → no encode()
                )
                duration_str = result.stdout.decode().strip()
                self.video_buffer.seek(0)  # rewind for later use
            else:
                command = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    "-i",
                    self.video_path,
                ]

                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=30,
                )
                duration_str = result.stdout.decode("utf-8").strip()

            if not duration_str or duration_str == "N/A":
                raise RuntimeError("ffprobe did not return a valid duration.")

            self._duration = float(duration_str)

            self._log_debug(
                f"Video duration: {self._duration} seconds",
                extra={"duration": self._duration, **extra_args},
            )

            if hasattr(self.logger, "log_operation_time"):
                self.logger.log_operation_time("get_duration", start_time)

            return self._duration

        except subprocess.TimeoutExpired as e:
            self._log_error(
                f"Timeout while getting video duration: {e}",
                error=e,
                extra=extra_args,
            )
            raise RuntimeError(
                f"Timeout while getting video duration for {source_display}"
            ) from e
        except Exception as e:
            self._log_error(
                f"Failed to get video duration: {e}",
                error=e,
                extra=extra_args,
            )
            raise RuntimeError(
                f"Failed to get video duration for {source_display}: {e}"
            ) from e
