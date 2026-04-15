#!/usr/bin/env python3
"""Wyoming TTS server for Qwen3-TTS with true streaming using faster-qwen3-tts."""

import argparse
import asyncio
import glob
import logging
import os

import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeStopped

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SAMPLES_PER_CHUNK = 1024
CLONE_VOICES_DIR = "/data/clone-voices"

CUSTOM_VOICE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

SPEAKERS = {
    "Vivian": {
        "description": "Bright, slightly edgy young female voice",
        "language": "zh",
    },
    "Serena": {"description": "Warm, gentle young female voice", "language": "zh"},
    "Uncle_Fu": {
        "description": "Seasoned male voice with a low, mellow timbre",
        "language": "zh",
    },
    "Dylan": {"description": "Youthful Beijing male voice", "language": "zh"},
    "Eric": {"description": "Lively Chengdu male voice", "language": "zh"},
    "Ryan": {
        "description": "Dynamic male voice with strong rhythmic drive",
        "language": "en",
    },
    "Aiden": {
        "description": "Sunny American male voice with a clear midrange",
        "language": "en",
    },
    "Ono_Anna": {"description": "Playful Japanese female voice", "language": "ja"},
    "Sohee": {
        "description": "Warm Korean female voice with rich emotion",
        "language": "ko",
    },
}


def load_clone_voices() -> dict[str, dict]:
    clone_voices = {}
    if not os.path.isdir(CLONE_VOICES_DIR):
        return clone_voices

    for ref_file in sorted(glob.glob(os.path.join(CLONE_VOICES_DIR, "*.wav"))):
        name = os.path.splitext(os.path.basename(ref_file))[0]
        txt_file = os.path.join(CLONE_VOICES_DIR, f"{name}.txt")
        if not os.path.isfile(txt_file):
            _LOGGER.warning("Skipping clone voice '%s': no matching .txt file", name)
            continue
        with open(txt_file, "r") as f:
            ref_text = f.read().strip()
        if not ref_text:
            _LOGGER.warning("Skipping clone voice '%s': empty ref text", name)
            continue
        clone_voices[name] = {"ref_audio": ref_file, "ref_text": ref_text}
        _LOGGER.info("Loaded clone voice: %s", name)

    return clone_voices


class Qwen3TTSEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        custom_voice_model: FasterQwen3TTS,
        base_model: FasterQwen3TTS | None,
        clone_voices: dict[str, dict],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.custom_voice_model = custom_voice_model
        self.base_model = base_model
        self.clone_voices = clone_voices

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            try:
                await self._handle_synthesize(synthesize)
            except Exception:
                _LOGGER.exception("Error synthesizing")
                from wyoming.error import Error

                await self.write_event(Error(text="Synthesis failed").event())
            return True

        return True

    async def _handle_synthesize(self, synthesize: Synthesize) -> None:
        text = " ".join(synthesize.text.strip().splitlines())
        if not text:
            await self.write_event(SynthesizeStopped().event())
            return

        _LOGGER.info("Synthesizing: '%s'", text[:100])

        voice_name = None
        if synthesize.voice is not None:
            voice_name = synthesize.voice.name

        is_clone = voice_name is not None and voice_name in self.clone_voices

        await self.write_event(
            AudioStart(rate=SAMPLE_RATE, width=2, channels=1).event()
        )

        if is_clone:
            if self.base_model is None:
                _LOGGER.error("Voice cloning requires the base model")
                await self.write_event(AudioStop().event())
                await self.write_event(SynthesizeStopped().event())
                return
            voice = self.clone_voices[voice_name]
            # Use true streaming generator
            for audio_chunk, sr, _ in self.base_model.generate_voice_clone_streaming(
                text=text,
                language="Auto",
                ref_audio=voice["ref_audio"],
                ref_text=voice["ref_text"],
            ):
                # Normalize and chunk if needed
                audio_array = self._normalize(audio_chunk)
                for i in range(0, len(audio_array), SAMPLES_PER_CHUNK):
                    chunk = audio_array[i : i + SAMPLES_PER_CHUNK]
                    await self.write_event(
                        AudioChunk(
                            audio=chunk.tobytes(),
                            rate=SAMPLE_RATE,
                            width=2,
                            channels=1,
                        ).event()
                    )
        else:
            speaker = voice_name or "Ryan"
            if speaker not in SPEAKERS:
                _LOGGER.warning("Unknown speaker '%s', falling back to Ryan", speaker)
                speaker = "Ryan"
            # Use true streaming generator
            for (
                audio_chunk,
                sr,
                _,
            ) in self.custom_voice_model.generate_custom_voice_streaming(
                text=text,
                speaker=speaker,
                language="Auto",
            ):
                # Normalize and chunk if needed
                audio_array = self._normalize(audio_chunk)
                for i in range(0, len(audio_array), SAMPLES_PER_CHUNK):
                    chunk = audio_array[i : i + SAMPLES_PER_CHUNK]
                    await self.write_event(
                        AudioChunk(
                            audio=chunk.tobytes(),
                            rate=SAMPLE_RATE,
                            width=2,
                            channels=1,
                        ).event()
                    )

        await self.write_event(AudioStop().event())
        await self.write_event(SynthesizeStopped().event())
        _LOGGER.info("Finished synthesizing: '%s'", text[:50])

    def _normalize(self, audio_float: np.ndarray) -> np.ndarray:
        if audio_float.dtype != np.int16:
            max_val = np.max(np.abs(audio_float))
            if max_val > 1.0:
                audio_float = audio_float / max_val
            return (audio_float * 32767).astype(np.int16)
        return audio_float


def build_wyoming_info(clone_voices: dict[str, dict]) -> Info:
    voices = []
    for name, info in SPEAKERS.items():
        voices.append(
            TtsVoice(
                name=name,
                description=f"{name} ({info['description']})",
                attribution=Attribution(name="Qwen", url=""),
                installed=True,
                version="1.0",
                languages=[info["language"], "en"],
            )
        )

    for name, info in clone_voices.items():
        voices.append(
            TtsVoice(
                name=name,
                description=f"{name} (cloned voice)",
                attribution=Attribution(name="User", url=""),
                installed=True,
                version="1.0",
                languages=["en"],
            )
        )

    return Info(
        tts=[
            TtsProgram(
                name="qwen3-tts",
                description="Qwen3-TTS neural text to speech with true streaming",
                attribution=Attribution(
                    name="Qwen",
                    url="https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                ),
                installed=True,
                version="1.0",
                voices=voices,
                supports_synthesize_streaming=True,
            )
        ],
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200", help="Server URI")
    parser.add_argument(
        "--model", default=CUSTOM_VOICE_MODEL, help="CustomVoice model name"
    )
    parser.add_argument(
        "--base-model", default=BASE_MODEL, help="Base model name for voice cloning"
    )
    parser.add_argument("--speaker", default="Ryan", help="Default speaker")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    _LOGGER.info("Loading CustomVoice model: %s", args.model)
    # Use FasterQwen3TTS which has true streaming support
    # Note: CUDA graphs only work with CUDA, otherwise uses standard inference
    custom_voice_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=device,
        dtype=dtype,
        attn_implementation="sdpa",  # Use sdpa for CPU compatibility
    )
    _LOGGER.info("CustomVoice model loaded on %s", device)

    clone_voices = load_clone_voices()
    base_model = None
    if clone_voices:
        _LOGGER.info("Loading Base model for voice cloning: %s", args.base_model)
        base_model = FasterQwen3TTS.from_pretrained(
            args.base_model,
            device=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        _LOGGER.info("Base model loaded on %s", device)
        _LOGGER.info("Loaded %d clone voice(s)", len(clone_voices))
    else:
        _LOGGER.info("No clone voices found in %s", CLONE_VOICES_DIR)

    wyoming_info = build_wyoming_info(clone_voices)
    server = AsyncServer.from_uri(args.uri)

    _LOGGER.info("Starting TTS server at %s", args.uri)
    await server.run(
        lambda reader, writer: Qwen3TTSEventHandler(
            wyoming_info, custom_voice_model, base_model, clone_voices, reader, writer
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
