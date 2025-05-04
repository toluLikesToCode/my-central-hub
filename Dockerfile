# Dockerfile for my-central-hub

# Use an official Node.js Debian-based runtime (slim variant)
# This provides better compatibility for Python packages like PyTorch
FROM node:23-slim

# Set the working directory inside the container
WORKDIR /app

# --- APT Proxy Fix ---
# Copy the custom apt configuration to handle potential proxy/cache issues
# Ensure 'badproxy' file exists in the build context (same directory as Dockerfile)
COPY ./badproxy /etc/apt/apt.conf.d/99fixbadproxy
# --- End APT Proxy Fix ---

# Copy package.json and package-lock.json
COPY package.json ./

RUN npm i
RUN npm install -g typescript



# Install system dependencies needed by your project:
# - python3 and pip for your embedding_service_helper.py
# - ffmpeg for video processing (used by embedding service)
# - git (sometimes needed by pip packages)
# Combine update, install, and clean in one layer for efficiency.
# The proxy fix should help prevent hash mismatches here.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ffmpeg \
        git \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure 'python' points to Python 3 if needed (usually handled by python3 package)
# RUN ln -sf /usr/bin/python3 /usr/bin/python

# Add local node_modules bin to the path (should help find tsc)
# Note: Setting PATH here might not affect subsequent RUN commands in the same way
# depending on the shell execution context within Docker build.
ENV PATH=/app/node_modules/.bin:$PATH

# Set environment for production and unbuffered Python output
# Note: NODE_ENV=production is set *before* npm ci, which is good practice
ENV NODE_ENV=production PYTHONUNBUFFERED=1

# Copy package.json and package-lock.json
COPY package.json ./

RUN npm i
RUN npm install -g typescript

# Install Node.js dependencies including devDependencies needed for the build
# Removed --only=production flag
RUN npm ci --frozen-lockfile

# Build your TypeScript application
# Step 1: Clean previous build output (equivalent to 'npm run clean')
RUN rm -rf dist
# Step 2: Directly execute the local tsc binary
# RUN ./node_modules/.bin/tsc
RUN npm run build

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

# Python and pip are pre-installed in this CUDA+Python base image

# --- DIAGNOSTIC STEP ---
# Check if tsc binary exists after npm ci
RUN ls -la /app/node_modules/.bin/tsc || echo "tsc not found in node_modules/.bin"
# --- END DIAGNOSTIC STEP ---

# Install python3 and pip3 using apt
# Combine update and install in one RUN layer to reduce image size
# Use -y to auto-confirm installation
# Clean up apt cache afterwards to keep image small
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy Python requirements file
# Ensure this path is correct relative to your build context
COPY python/requirements.txt ./requirements.txt

# Install Python dependencies
# PyTorch wheels are generally available for Debian-based images
# Use --break-system-packages to allow pip install in Debian's system Python (PEP 668)
# RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt
RUN pip3 install --break-system-packages -r requirements.txt

# Install PyTorch, torchvision, and torchaudio with CUDA 12.8 support
# RUN pip3 install --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy the rest of your application code into the container
# Ensure your .dockerignore prevents copying unnecessary files
COPY . .


# --- Optional: Prune devDependencies after build ---
# If you want to reduce final image size, you can remove devDependencies now
# RUN npm prune --production

# Expose the port your application listens on (e.g., 8080 from your config)
EXPOSE 8080

# Define the command to run your *backend* application
# This runs the compiled JavaScript output from your build step.
CMD ["node", "dist/main.js"]
