# Qwen3-TTS Wyoming Server

This is a Wyoming protocol wrapper for the Qwen3-TTS model.

## Requirements

- NVIDIA GPU with CUDA support (recommended)
- Docker with NVIDIA Container Toolkit
- Python 3.11+

## Building

```bash
docker build -t qwen3-tts-wyoming .
```

## Running

```bash
docker run -d \
  --name qwen3-tts-wyoming \
  -p 10200:10200 \
  --gpus all \
  -v qwen3-tts-data:/data \
  qwen3-tts-wyoming
```

## Home Assistant Configuration

Add as Wyoming integration:
- **Type**: Text-to-speech (TTS)
- **Host**: `<your-server-ip>`
- **Port**: `10200`

## Features

- Supports English and Chinese
- GPU-accelerated inference
- Streaming audio support
- Compatible with Home Assistant Assist

## Notes

- First run will download the model (~2-3GB)
- Model is cached in `/data` volume
- Requires CUDA-compatible GPU for best performance
