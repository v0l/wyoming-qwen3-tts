# Wyoming Qwen3-TTS

Wyoming protocol TTS server using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) with flash-attn support. Compatible with Home Assistant's Wyoming integration.

## Features

- 9 built-in voices (English, Chinese, Japanese, Korean)
- GPU-accelerated with CUDA + flash-attn
- Streaming audio output via Wyoming protocol
- Multi-arch: ARM64 and AMD64

## Voices

| Speaker | Description | Native Language |
|---------|-------------|----------------|
| Vivian | Bright, slightly edgy young female voice | Chinese |
| Serena | Warm, gentle young female voice | Chinese |
| Uncle_Fu | Seasoned male voice with a low, mellow timbre | Chinese |
| Dylan | Youthful Beijing male voice | Chinese (Beijing Dialect) |
| Eric | Lively Chengdu male voice | Chinese (Sichuan Dialect) |
| Ryan | Dynamic male voice with strong rhythmic drive | English |
| Aiden | Sunny American male voice with a clear midrange | English |
| Ono_Anna | Playful Japanese female voice | Japanese |
| Sohee | Warm Korean female voice with rich emotion | Korean |

All voices support all 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian) via auto-detection.

## Build Args

| Arg | Default | Options | Description |
|-----|---------|---------|-------------|
| `ARCH` | `arm64` | `arm64`, `amd64` | Target architecture |
| `CUDA` | `130` | `126`, `128`, `130` | CUDA toolkit version (12.6, 12.8, 13.0) |

```bash
# ARM64 + CUDA 13.0 (default)
docker compose build

# AMD64 + CUDA 12.8
docker compose build --build-arg ARCH=amd64 --build-arg CUDA=128
```

## Docker Compose

```yaml
services:
  qwen3-tts-wyoming:
    build: .
    container_name: qwen3-tts-wyoming
    ports:
      - "10201:10200"
    environment:
      - TZ=Europe/Dublin
      - HF_HOME=/data/huggingface
    volumes:
      - qwen3-tts-data:/data
      - ${HF_HOME:-${HOME}/.cache/huggingface}:/data/huggingface
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities:
                - gpu
                - utility
                - compute

volumes:
  qwen3-tts-data:
```

## Home Assistant

1. Go to **Settings > Devices & Services > Add Integration**
2. Search for **Wyoming**
3. Configure:
   - **Host**: Your server IP
   - **Port**: `10201` (or whatever you mapped)
4. Select a voice in your Assist pipeline settings

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--uri` | `tcp://0.0.0.0:10200` | Server URI |
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | HuggingFace model name |
| `--speaker` | `Ryan` | Default speaker |
| `--debug` | off | Enable debug logging |

## Voice Cloning

Place reference audio files in `/data/clone-voices/` (inside container). Each voice needs a matching `.wav` + `.txt` pair:

```
/data/clone-voices/
├── my_voice.wav      # Reference audio (3-30 seconds, mono or stereo)
└── my_voice.txt      # Exact transcript of the audio
```

Clone voices appear as additional voices in Home Assistant. Example docker-compose volume:

```yaml
volumes:
  - ./clone-voices:/data/clone-voices
```

## Notes

- First run downloads the model (~4GB) + tokenizer (~1GB). Subsequent runs use the HuggingFace cache.
- Mount `~/.cache/huggingface` to avoid re-downloading across container restarts.
- Flash-attn v2 is installed via prebuilt wheels from [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels).