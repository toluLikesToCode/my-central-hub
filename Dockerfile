# Dockerfile for my-central-hub

# Use an official Node.js runtime as a parent image (Choose a version you use, e.g., 18 or 20)
# Using '-slim' variant for smaller image size
FROM node:18-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed by your project:
# - python3 and pip for your embedding_service_helper.py
# - ffmpeg for video processing (used by embedding service)
# - git (sometimes needed by pip packages)
# Clean up apt-get cache afterwards to keep image size down
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package.json and package-lock.json (or yarn.lock)
# This leverages Docker layer caching: if these files don't change,
# npm install won't rerun on subsequent builds, speeding things up.
COPY package*.json ./

# Install Node.js dependencies
RUN npm install
# If you use yarn, replace the above with:
# COPY yarn.lock ./
# RUN yarn install --frozen-lockfile

# Copy Python requirements file
COPY requirements.txt ./

# Install Python dependencies
# --no-cache-dir reduces image size
# Note: Installing PyTorch and Transformers can take time and make the image large.
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Build your TypeScript application (assuming you have a build script in package.json)
# This compiles your .ts files (like main.ts, server.ts) into JavaScript, usually in a 'dist' folder.
RUN npm run build
# If you don't have a build step and run directly with ts-node, skip this line
# and adjust the CMD instruction below accordingly.

# Your server listens on a port defined in config (default 8080)
# Expose this port from the container
EXPOSE 8080

# Define the command to run your application
# This assumes your build output's main entry point is dist/main.js
CMD ["node", "dist/main.js"]
# If you run directly with ts-node (not recommended for production):
# CMD ["npx", "ts-node", "src/main.ts"]