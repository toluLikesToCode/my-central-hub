import os
from google.cloud import storage  # type: ignore
from dotenv import load_dotenv  # type: ignore

# Ensure .env is loaded at the very top
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "media-content-1")

GOOGLE_APPLICATION_CREDENTIALS = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS", "./personal-project-main-02e1cf2744c0.json"
)

storage_client = storage.Client(project="personal-project-main")
