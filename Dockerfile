FROM python:3.12-slim

ARG ARCH=arm64
ARG CUDA=130

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    sox \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA}

RUN pip install --no-cache-dir \
    wyoming \
    qwen-tts

RUN FLASH_ATTN_BASE="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16"; \
    if [ "$ARCH" = "amd64" ]; then \
        WHEEL="flash_attn-2.8.3%2Bcu${CUDA}torch2.10-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"; \
    else \
        WHEEL="flash_attn-2.8.3%2Bcu${CUDA}torch2.10-cp312-cp312-manylinux_2_34_aarch64.whl"; \
    fi; \
    pip install --no-cache-dir "${FLASH_ATTN_BASE}/${WHEEL}"

ENV HF_HOME=/data/huggingface

COPY server.py .

EXPOSE 10200

ENTRYPOINT ["python", "server.py"]