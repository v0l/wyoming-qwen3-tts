FROM python:3.12-slim

ARG ARCH=arm64

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    sox \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu130

RUN pip install --no-cache-dir \
    wyoming \
    qwen-tts

RUN if [ "$ARCH" = "amd64" ]; then \
        pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl; \
    else \
        pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-manylinux_2_34_aarch64.whl; \
    fi

ENV HF_HOME=/data/huggingface

COPY . .

EXPOSE 10200

ENTRYPOINT ["python", "server.py"]