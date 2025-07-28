#!/bin/bash
set -e

# Verify if ffmpeg (from /usr/local/bin, installed by Dockerfile) is available
if ! command -v ffmpeg >/dev/null 2>&1 || ! ffmpeg -version >/dev/null 2>&1; then
  echo "[Entrypoint] CRITICAL: ffmpeg not found or not executable after Docker build!"
  echo "[Entrypoint] This indicates an issue with the FFmpeg static build installation in the Dockerfile."
  # Attempting apt-get install as a last resort, though this will get the OS version.
  echo "[Entrypoint] Attempting to install OS default ffmpeg as a fallback..."
  apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[Entrypoint] Fallback ffmpeg installation failed. Exiting."
    exit 1
  fi
  echo "[Entrypoint] Fallback OS ffmpeg installed. GPU acceleration might not be optimal."
else
  echo "[Entrypoint] Custom ffmpeg build found and verified."
  echo "[Entrypoint] FFmpeg version:"
  ffmpeg -version | head -n 1
  echo "[Entrypoint] Available FFmpeg hardware acceleration methods:"
  ffmpeg -hwaccels
fi

# Start the Python server (passed as CMD to the ENTRYPOINT)
echo "[Entrypoint] Starting Python server: $@"
exec "$@"