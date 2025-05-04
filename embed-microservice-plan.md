Here’s a detailed migration plan to split your current in‑process Python‐spawn CLIP embedding logic into two standalone microservices:

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

1. Audit & clean up existing implementation  
   • Identify all places in embedding.service.ts that spawn or communicate with the Python helper.  
   • Extract any reusable logic in embedding_service_helper.py (e.g. `process_batch`, `CLIPEmbedder`, video sampler) into a “library” that can be imported by our new service.  
   • Write down the exact JSON protocol you’re exchanging over stdin/stdout (keys: `imagePaths`, `PROGRESS:`, response map of file→{embedding,debugMetadata,error,…}).

2. Scaffold the Python embedding microservice  
   A) Create a new service folder  
    • mkdir embedding-service  
    • Move or symlink embedding_service_helper.py (and any other helper modules) into it.  
   B) requirements.txt  
    • fastapi, uvicorn[standard] plus exactly whatever’s in your top‐level requirements.txt (torch, torchvision, torchaudio, scenedetect, ffmpeg‑python if used, etc.).  
   C) server.py (FastAPI)  
    • Expose POST /embed that accepts `{ imagePaths: string[] }`  
    • Internally call your helper’s `process_batch(image_paths, …)`, return 200 JSON map of file→result exactly matching the old stdout JSON.  
    • Implement health‑check endpoint (GET /health) for orchestration.  
   D) Dockerfile.embed  
    • Base on `python:3.10‑slim` or an “‐py” CUDA image if you need GPU.  
    • Install system deps (ffmpeg, git, build‑essential if needed), pip install from requirements.txt, copy `.`.  
    • CMD `uvicorn server:app --host 0.0.0.0 --port 8000`.

3. Build & test the embedding service in isolation  
   • Build the image: `docker build -f Dockerfile.embed -t myhub/clip‐embed:latest .`  
   • Run locally: `docker run -p8000:8000 …` and curl POST /embed with a small set of media paths under `/public/media/...` to verify responses.  
   • Add unit tests in `python/services/embedding-service/tests` against the FastAPI app (pytest + httpx).

4. Refactor Node side to consume the HTTP service  
   A) Remove all `spawn(…, PYTHON_SCRIPT_PATH)` logic in embedding.service.ts—including the requestQueue, processQueue, stdout/stderr handlers, inactivity timers, etc.  
   B) Introduce a new `EmbeddingHttpClient` (or extend the existing `EmbeddingService`):  
    • Read `EMBEDDING_URL` from `config.embedding.serviceUrl` (default `http://localhost:8000/embed`).  
    • Implement `getEmbeddings(textPaths: string[])` that does a single HTTP POST + JSON parse + error handling + optional retry/backoff.  
   C) Wire that client into your controller (`embeddings.handler.ts`), inject via DI or import.  
   D) Remove any python config entries from `server.config.ts` (`pythonExecutable`, `scriptTimeout`, etc.).

5. Update local orchestration & CI  
   • Create a `docker-compose.yml` at project root:

   - service “api” builds your Node image (Dockerfile)
   - service “embed” builds the new Python image
   - both share a network, volume‐mount your project (or use multi‐stage builds)
   - set `EMBEDDING_URL: http://embed:8000/embed` in the “api” env.  
     • Update any Makefile targets or deployment manifests to build / push both images.  
     • Adjust health checks in your Kubernetes/Swarm config to wait on embed’s /health before routing traffic.

6. Migrate tests & mocks  
   • Replace any unit tests that relied on mocking the stdin/stdout Python process with mocks of the HTTP client.  
   • Add integration tests (in tests or stress) that spin up both containers and drive the full end‑to‑end path.

7. Roll‑out & cleanup  
   • Deploy the dual‑service setup to staging, validate performance, logs, monitoring (e.g. Prometheus metrics on both services).  
   • Once stable, remove all legacy Python‐spawn code, config flags, and the old `pythonScriptPath` entries.  
   • Archive or delete the original Dockerfile lines that installed Python into your Node image.

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––  
Outcome:

- The Node API image stays lean and fast‑building.
- The Python embedding service can be scaled, versioned, and GPU‑tuned independently.
- Clear contract over HTTP, easier observability, better CI/CD and fault isolation.
