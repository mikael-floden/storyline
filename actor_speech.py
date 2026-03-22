"""
Actor speech support powered by Piper TTS.

Speech synthesis runs off the main thread, downloads voices on demand, and
plays synthesized WAV data through pygame's mixer.
"""

from __future__ import annotations

import io
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional


class ActorSpeechController:
    """Asynchronous text-to-speech controller for an actor."""

    def __init__(
        self,
        *,
        default_voice: str = "en_US-lessac-low",
        voices_dir: str | Path = "_piper_voices",
    ):
        self.default_voice = default_voice
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="actor-speech")
        self._future: Optional[Future[tuple[int, bytes]]] = None
        self._request_id = 0
        self._channel = None
        self._sound = None
        self._voice_cache: dict[str, object] = {}
        self._last_error: Optional[str] = None

    @property
    def last_error(self) -> Optional[str]:
        """Return the last speech synthesis/playback error, if any."""
        return self._last_error

    @property
    def is_busy(self) -> bool:
        """Return True while synthesizing or playing audio."""
        if self._future is not None:
            return True

        if self._channel is not None and self._channel.get_busy():
            return True

        return False

    def speak(
        self,
        text: str,
        *,
        voice_name: Optional[str] = None,
        interrupt: bool = True,
    ) -> bool:
        """Start speaking text asynchronously."""
        text = text.strip()
        if not text:
            return False

        if interrupt:
            self.stop()
        elif self.is_busy:
            return False

        self._last_error = None
        self._request_id += 1
        request_id = self._request_id
        voice_name = voice_name or self.default_voice
        self._future = self._executor.submit(
            self._synthesize_wav_bytes,
            request_id,
            text,
            voice_name,
        )
        return True

    def stop(self) -> None:
        """Stop any active or pending speech."""
        self._request_id += 1

        if self._future is not None:
            self._future.cancel()
            self._future = None

        if self._channel is not None:
            try:
                self._channel.stop()
            except Exception:
                pass

        self._channel = None
        self._sound = None

    def update(self) -> bool:
        """Advance synthesis/playback state and return whether speech is active."""
        import pygame

        if self._future is not None and self._future.done():
            future = self._future
            self._future = None
            try:
                request_id, wav_bytes = future.result()
            except Exception as exc:
                self._last_error = str(exc)
                return self.is_busy

            if request_id == self._request_id:
                self._play_wav_bytes(wav_bytes)

        if self._channel is not None and not self._channel.get_busy():
            self._channel = None
            self._sound = None

        return self.is_busy

    def shutdown(self) -> None:
        """Release playback resources and stop background synthesis."""
        self.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        """Play synthesized WAV bytes with pygame."""
        import pygame

        if pygame.mixer.get_init() is None:
            pygame.mixer.init()

        sound_file = io.BytesIO(wav_bytes)
        self._sound = pygame.mixer.Sound(file=sound_file)
        self._channel = pygame.mixer.find_channel(force=True)
        self._channel.play(self._sound)

    def _synthesize_wav_bytes(
        self,
        request_id: int,
        text: str,
        voice_name: str,
    ) -> tuple[int, bytes]:
        """Download/load a Piper voice and synthesize a WAV payload."""
        from piper import PiperVoice
        from piper.download_voices import download_voice

        model_path = self.voices_dir / f"{voice_name}.onnx"
        config_path = self.voices_dir / f"{voice_name}.onnx.json"
        if (not model_path.exists()) or (not config_path.exists()):
            download_voice(voice_name, self.voices_dir)

        voice = self._voice_cache.get(voice_name)
        if voice is None:
            voice = PiperVoice.load(model_path, download_dir=self.voices_dir)
            self._voice_cache[voice_name] = voice

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        return request_id, wav_buffer.getvalue()
