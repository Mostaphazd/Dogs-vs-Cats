# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Set environment variables to prevent Python bytecode and buffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file early to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies with pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final lightweight runtime image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install runtime dependencies (for OpenCV and others)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application files (Ensure the .dockerignore is set up to exclude unnecessary files)
COPY . .

# Expose Flask's default port
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py"]
