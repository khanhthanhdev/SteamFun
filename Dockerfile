# Multi-stage Dockerfile for FastAPI application
FROM python:3.12-slim as base

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    PATH="/root/.TinyTeX/bin/x86_64-linux:$PATH"

# Install system dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    gcc \
    g++ \
    build-essential \
    pkg-config \
    portaudio19-dev \
    libasound2-dev \
    libsdl-pango-dev \
    libcairo2-dev \
    libpango1.0-dev \
    sox \
    ffmpeg \
    texlive-full \
    dvisvgm \
    ghostscript \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e . \
    && python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')" \
    && python -c "import gradio; print(f'Gradio version: {gradio.__version__}')" \
    && find /usr/local -name "*.pyc" -delete \
    && find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Ensure Hugging Face cache directories are writable

# Create models directory and download models efficiently
RUN mkdir -p models && cd models \
    && echo "Downloading models for HF Spaces..." \
    && wget --progress=dot:giga --timeout=30 --tries=3 \
        -O kokoro-v0_19.onnx \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx" \
    && wget --progress=dot:giga --timeout=30 --tries=3 \
        -O voices.bin \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin" \
    && ls -la /app/models/

# Copy application code
COPY app/ ./app/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY .env.example ./
COPY gradio_app.py ./

# Set Python path for the new structure
ENV PYTHONPATH="/app:$PYTHONPATH"
RUN echo "export PYTHONPATH=/app:\$PYTHONPATH" >> ~/.bashrc

# Run embedding creation script at build time


# Ensure all files are writable (fix PermissionError for log file)
RUN chmod -R a+w /app

# Create output directory
RUN mkdir -p output tmp
# Ensure output and tmp directories are writable (fix PermissionError for session_id.txt)
RUN chmod -R a+w /app/output /app/tmp || true

RUN mkdir -p output tmp logs \
    && mkdir -p /app/.cache/huggingface/hub \
    && mkdir -p /app/.cache/transformers \
    && mkdir -p /app/.cache/sentence_transformers \
    && chmod -R 755 /app/.cache \
    && chmod 755 /app/models \
    && ls -la /app/models/ \
    && echo "Cache directories created with proper permissions"
# Add application metadata
LABEL app.name="Video Generation Agents" \
      app.version="0.1.0" \
      app.description="FastAPI-based multi-agent video generation system"

# Expose ports for FastAPI and Gradio
EXPOSE 8000 7860

# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command runs FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Alternative commands for different modes:
# For Gradio UI: CMD ["python", "gradio_app.py"]
# For production: CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]