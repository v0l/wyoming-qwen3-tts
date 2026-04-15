FROM python:3.12-slim

ARG CUDA=130

WORKDIR /app

# Install sox for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA}

# Install TTS dependencies
RUN pip install --no-cache-dir \
    wyoming \
    faster-qwen3-tts

# Hugging Face cache directory
ENV HF_HOME=/data/huggingface

# Copy server code
COPY server.py .

EXPOSE 10200

ENTRYPOINT ["python", "server.py"]